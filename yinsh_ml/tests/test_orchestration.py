"""Tests for the experiment-orchestration thin slice.

Covers the new statistics (SPRT), the failure-mode panel, the Tier-0 funnel, and
the full scheduler thread (schedule -> run -> evaluate -> route -> gate -> ratify)
using fakes so no GPU/checkpoints are required.
"""

import pytest

from yinsh_ml.utils.stats import sprt_decision, SPRTResult
from yinsh_ml.orchestration import (
    EvaluationFunnel,
    ExperimentSpec,
    FailurePanel,
    Journal,
    Launcher,
    OrchestrationStore,
    Scheduler,
)
from yinsh_ml.orchestration.failure_panel import PanelInput
from yinsh_ml.orchestration.launcher import LaunchResult
from yinsh_ml.orchestration.match_runner import MatchOutcome
from yinsh_ml.experiments.experiment_db import ExperimentDB, ExperimentRecord


# --------------------------------------------------------------------------
# SPRT
# --------------------------------------------------------------------------

def test_sprt_accepts_h1_on_strong_wins():
    r = sprt_decision(wins=55, losses=8, p0=0.5, p1=0.55)
    assert r.verdict == "accept_h1"
    assert r.decisive


def test_sprt_accepts_h0_on_losses():
    r = sprt_decision(wins=8, losses=55, p0=0.5, p1=0.55)
    assert r.verdict == "accept_h0"
    assert r.decisive


def test_sprt_continues_when_balanced():
    r = sprt_decision(wins=20, losses=20, p0=0.5, p1=0.55)
    assert r.verdict == "continue"
    assert not r.decisive


def test_sprt_no_games_is_continue():
    assert sprt_decision(0, 0).verdict == "continue"


def test_sprt_rejects_bad_hypotheses():
    with pytest.raises(ValueError):
        sprt_decision(1, 1, p0=0.6, p1=0.55)  # p1 <= p0


# --------------------------------------------------------------------------
# Failure-mode panel
# --------------------------------------------------------------------------

def test_panel_green_on_healthy_signals():
    panel = FailurePanel().evaluate(
        PanelInput(policy_entropy=1.2, value_accuracy=0.6, value_variance=0.05)
    )
    assert panel.green


def test_panel_flags_policy_collapse():
    panel = FailurePanel().evaluate(PanelInput(policy_entropy=0.1))
    assert not panel.green
    assert any(c.name == "policy_entropy" for c in panel.failures)


def test_panel_flags_value_saturation():
    panel = FailurePanel().evaluate(PanelInput(value_accuracy=0.6, value_variance=1e-9))
    assert not panel.green


def test_panel_skips_absent_signals_without_blocking():
    panel = FailurePanel().evaluate(PanelInput())  # nothing supplied
    assert panel.green  # skips don't block
    assert len(panel.skips) == 4


def test_offense_only_detector_flags_runaway():
    # One-sided runaway: completed-runs differential climbs monotonically, no response.
    runaway = [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]] * 4
    panel = FailurePanel().evaluate(PanelInput(run_diff_trajectories=runaway))
    offense = next(c for c in panel.checks if c.name == "offense_only_equilibrium")
    assert not offense.passed


def test_offense_only_detector_passes_balanced_games():
    # Differential swings both ways — both sides respond.
    balanced = [[0, 1, -1, 2, -2, 1, 0, -1, 1, 0]] * 4
    panel = FailurePanel().evaluate(PanelInput(run_diff_trajectories=balanced))
    offense = next(c for c in panel.checks if c.name == "offense_only_equilibrium")
    assert offense.passed


# --------------------------------------------------------------------------
# Funnel (Tier 0) with a fake match runner
# --------------------------------------------------------------------------

class FakeMatchRunner:
    """Plays deterministic batches at a fixed win fraction."""

    def __init__(self, win_frac: float, draw_frac: float = 0.0):
        self.win_frac = win_frac
        self.draw_frac = draw_frac

    def play_batch(self, n: int) -> MatchOutcome:
        draws = round(n * self.draw_frac)
        decisive = n - draws
        wins = round(decisive * self.win_frac)
        losses = decisive - wins
        return MatchOutcome(wins=wins, draws=draws, losses=losses)


def _healthy_input():
    return PanelInput(policy_entropy=1.2, value_accuracy=0.6, value_variance=0.05)


def test_funnel_clear_win():
    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    r = funnel.run_tier0(_healthy_input(), FakeMatchRunner(win_frac=0.8))
    assert r.is_clear_win
    assert not r.is_ambiguous


def test_funnel_clear_loss():
    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    r = funnel.run_tier0(_healthy_input(), FakeMatchRunner(win_frac=0.2))
    assert r.is_clear_loss


def test_funnel_ambiguous_when_balanced():
    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    r = funnel.run_tier0(_healthy_input(), FakeMatchRunner(win_frac=0.5))
    assert r.is_ambiguous
    assert not r.sprt_decisive


def test_funnel_no_baseline_is_ambiguous():
    funnel = EvaluationFunnel()
    r = funnel.run_tier0(_healthy_input(), None)
    assert not r.had_baseline
    assert r.is_ambiguous


def test_funnel_strong_but_flagged_is_ambiguous():
    # Decisive win on the board, but a failure-mode flag -> must get human eyes.
    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    flagged = PanelInput(policy_entropy=0.1)  # collapse
    r = funnel.run_tier0(flagged, FakeMatchRunner(win_frac=0.8))
    assert not r.is_clear_win
    assert r.is_ambiguous


# --------------------------------------------------------------------------
# Scheduler end-to-end thread
# --------------------------------------------------------------------------

class FakeLauncher(Launcher):
    """Creates an experiments row (like ExperimentRunner) and returns a result."""

    def __init__(self, db_path, experiment_id, status="completed", final_metrics=None):
        self.db_path = db_path
        self.experiment_id = experiment_id
        self.status = status
        self.final_metrics = final_metrics or {}

    def launch(self, spec):
        db = ExperimentDB(self.db_path)
        if db.get_experiment(self.experiment_id) is None:
            db.create_experiment(ExperimentRecord(
                experiment_id=self.experiment_id,
                name=spec.name or "fake",
                status="completed",
                total_iterations=1,
            ))
        return LaunchResult(
            experiment_id=self.experiment_id,
            status=self.status,
            save_dir="",
            final_metrics=self.final_metrics,
        )


def _make_scheduler(tmp_path, launcher, match_runner=None, panel_input=None,
                    interpreter=None, triage=None):
    db_path = str(tmp_path / "experiments.db")
    store = OrchestrationStore(db_path)
    journal = Journal(str(tmp_path))
    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    scheduler = Scheduler(
        store=store,
        journal=journal,
        funnel=funnel,
        output_dir=str(tmp_path),
        launcher_factory=lambda target, outdir: launcher,
        match_runner_factory=(lambda spec, launch, outdir: match_runner),
        panel_input_builder=(
            (lambda spec, launch: panel_input) if panel_input is not None else None
        ),
        interpreter=interpreter,
        triage=triage,
    )
    return scheduler, store, db_path


def test_scheduler_clear_win_routes_to_promotion_gate_and_ratifies(tmp_path):
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_win", final_metrics=healthy)
    scheduler, store, _ = _make_scheduler(
        tmp_path, launcher, match_runner=FakeMatchRunner(win_frac=0.8)
    )

    spec = ExperimentSpec(config_path="x.yaml", name="winner", baseline_id="base1")
    result = scheduler.process(spec)

    assert result.decision.route == "gate"
    assert result.decision.gate_kind == "promotion"
    assert store.get_status("exp_win") == "awaiting_ratification"

    # Result lands in the feed (visibility) AND a gate item is queued (blocking).
    assert (tmp_path / "FEED.md").exists()
    assert (tmp_path / "journal" / "exp_win.md").exists()
    gate = store.open_gate_for_experiment("exp_win")
    assert gate is not None and gate.kind == "promotion"

    # An eval row was persisted (queryable tabular truth).
    evals = store.get_evals("exp_win")
    assert len(evals) == 1 and evals[0].sprt_verdict == "accept_h1"

    # Ratify -> promoted, gate resolved.
    assert scheduler.ratify("exp_win", approve=True) == "promoted"
    assert store.get_status("exp_win") == "promoted"
    assert store.open_gate_for_experiment("exp_win") is None


def test_scheduler_clear_loss_auto_rejects_no_gate(tmp_path):
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_loss", final_metrics=healthy)
    scheduler, store, _ = _make_scheduler(
        tmp_path, launcher, match_runner=FakeMatchRunner(win_frac=0.2)
    )

    spec = ExperimentSpec(config_path="x.yaml", name="loser", baseline_id="base1")
    result = scheduler.process(spec)

    assert result.decision.route == "feed"
    assert result.decision.gate_kind is None
    assert store.get_status("exp_loss") == "rejected"
    assert store.open_gate_for_experiment("exp_loss") is None


def test_scheduler_ambiguous_routes_to_review_gate(tmp_path):
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_amb", final_metrics=healthy)
    scheduler, store, _ = _make_scheduler(
        tmp_path, launcher, match_runner=FakeMatchRunner(win_frac=0.5)
    )

    spec = ExperimentSpec(config_path="x.yaml", name="ambig", baseline_id="base1")
    result = scheduler.process(spec)

    assert result.decision.route == "gate"
    assert result.decision.gate_kind == "review"
    assert store.get_status("exp_amb") == "awaiting_ratification"


def test_scheduler_failed_launch_does_not_evaluate(tmp_path):
    launcher = FakeLauncher(
        str(tmp_path / "experiments.db"), "exp_fail", status="failed"
    )
    scheduler, store, _ = _make_scheduler(tmp_path, launcher)

    spec = ExperimentSpec(config_path="x.yaml", name="boom")
    result = scheduler.process(spec)

    assert result.tier0 is None
    assert result.decision is None
    assert store.get_status("exp_fail") == "failed"
    assert store.get_evals("exp_fail") == []


def test_scheduler_queue_runs_all(tmp_path):
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_q", final_metrics=healthy)
    scheduler, _, _ = _make_scheduler(tmp_path, launcher, match_runner=FakeMatchRunner(0.8))

    scheduler.enqueue(ExperimentSpec(config_path="x.yaml", baseline_id="b"))
    assert scheduler.pending == 1
    results = scheduler.run_queue()
    assert len(results) == 1
    assert scheduler.pending == 0


# --------------------------------------------------------------------------
# Rung 1 — PI interpreter (the augmented step)
# --------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock

from yinsh_ml.orchestration import Interpretation, PIInterpreter


def _tier0_and_decision():
    """Build a real (spec, tier0, decision) triple for interpreter tests."""
    from yinsh_ml.orchestration import PIRouter

    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    tier0 = funnel.run_tier0(_healthy_input(), FakeMatchRunner(win_frac=0.8))
    decision = PIRouter().route(tier0)
    spec = ExperimentSpec(config_path="x.yaml", name="t", baseline_id="base1")
    return spec, tier0, decision


def test_interpreter_maps_parsed_output_and_caches_system():
    spec, tier0, decision = _tier0_and_decision()
    client = MagicMock()
    client.messages.parse.return_value = SimpleNamespace(
        parsed_output=SimpleNamespace(
            headline="Strong win vs baseline",
            assessment="The candidate beat the baseline decisively with a clean panel.",
            reasons_to_doubt=["Only one baseline", "Small decisive sample"],
            suggested_next_step="Run the Tier-1 anchor ladder.",
            confidence="medium",
        )
    )

    interp = PIInterpreter(client=client).interpret(spec, tier0, decision)

    assert isinstance(interp, Interpretation)
    assert interp.headline == "Strong win vs baseline"
    assert interp.reasons_to_doubt == ["Only one baseline", "Small decisive sample"]
    assert interp.confidence == "medium"

    # Built the request per the project defaults: Opus 4.8, adaptive thinking,
    # a cached system rubric, and a structured output schema.
    kwargs = client.messages.parse.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "output_format" in kwargs


def test_interpreter_returns_none_on_failure():
    # Any failure (no key, network, SDK) must degrade to the template, not raise.
    spec, tier0, decision = _tier0_and_decision()
    client = MagicMock()
    client.messages.parse.side_effect = RuntimeError("no API key")
    assert PIInterpreter(client=client).interpret(spec, tier0, decision) is None


def test_journal_renders_pi_read_when_interpretation_present(tmp_path):
    spec, tier0, decision = _tier0_and_decision()
    interp = Interpretation(
        headline="h",
        assessment="ASSESSMENT-MARKER",
        reasons_to_doubt=["DOUBT-MARKER"],
        suggested_next_step="NEXT-MARKER",
        confidence="low",
    )
    path = Journal(str(tmp_path)).write_report("e1", spec, tier0, decision, interp)
    text = (tmp_path / "journal" / "e1.md").read_text()
    assert "PI read" in text
    assert "ASSESSMENT-MARKER" in text
    assert "DOUBT-MARKER" in text
    assert "NEXT-MARKER" in text


def test_journal_falls_back_to_template_without_interpretation(tmp_path):
    spec, tier0, decision = _tier0_and_decision()
    Journal(str(tmp_path)).write_report("e2", spec, tier0, decision, None)
    text = (tmp_path / "journal" / "e2.md").read_text()
    assert "PI read" not in text
    assert "Reasons to doubt" in text  # template's honesty section still present


class _StubInterpreter:
    def interpret(self, spec, tier0, decision):
        return Interpretation(
            headline="LLM-HEADLINE",
            assessment="LLM-ASSESSMENT",
            reasons_to_doubt=["d"],
            suggested_next_step="s",
            confidence="high",
        )


def test_scheduler_threads_interpretation_into_feed_and_report(tmp_path):
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_llm", final_metrics=healthy)
    scheduler, _, _ = _make_scheduler(
        tmp_path, launcher, match_runner=FakeMatchRunner(0.8), interpreter=_StubInterpreter()
    )

    scheduler.process(ExperimentSpec(config_path="x.yaml", baseline_id="b"))

    # LLM one-liner enriches the feed detail; assessment lands in the report.
    assert "LLM-HEADLINE" in (tmp_path / "FEED.md").read_text()
    assert "LLM-ASSESSMENT" in (tmp_path / "journal" / "exp_llm.md").read_text()


def test_scheduler_without_interpreter_still_works(tmp_path):
    # The whole pipeline must run with no interpreter wired (LLM is optional).
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_nollm", final_metrics=healthy)
    scheduler, store, _ = _make_scheduler(tmp_path, launcher, match_runner=FakeMatchRunner(0.8))

    scheduler.process(ExperimentSpec(config_path="x.yaml", baseline_id="b"))
    assert store.get_status("exp_nollm") == "awaiting_ratification"
    assert "PI read" not in (tmp_path / "journal" / "exp_nollm.md").read_text()


# --------------------------------------------------------------------------
# Rung 2 — triage workflow (code-controlled tool loop)
# --------------------------------------------------------------------------

from yinsh_ml.orchestration import TriageResult, TriageVerdict, TriageWorkflow
from yinsh_ml.orchestration.triage import _TriageTools


def _ambiguous_tier0(max_games=40):
    """A real inconclusive Tier-0 result that leaves room in the game budget."""
    funnel = EvaluationFunnel(batch_size=10, max_games=max_games)
    return funnel.run_tier0(_healthy_input(), FakeMatchRunner(win_frac=0.5))


def test_triage_tool_orders_games_and_recomputes(tmp_path):
    tier0 = _ambiguous_tier0(max_games=40)
    funnel = EvaluationFunnel(batch_size=10, max_games=200)
    tools = _TriageTools(tier0, FakeMatchRunner(win_frac=0.8), funnel)

    msg = tools.order_more_games(120)
    assert tools.games_added == 120
    assert "SPRT now: accept_h1" in msg  # strong candidate crosses the boundary
    assert tools.build_updated_tier0().is_clear_win


def _tool_use(name, tid, inp):
    return SimpleNamespace(type="tool_use", name=name, id=tid, input=inp)


class _ScriptedMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _ScriptedClient:
    """Fake anthropic client that replays scripted tool-use responses."""

    def __init__(self, responses):
        self.messages = _ScriptedMessages(responses)


def test_triage_workflow_orders_games_then_resolves():
    tier0 = _ambiguous_tier0(max_games=40)
    assert tier0.is_ambiguous  # precondition

    client = _ScriptedClient([
        SimpleNamespace(  # turn 1: order more games
            content=[_tool_use("order_more_games", "t1", {"n": 120})],
            stop_reason="tool_use",
        ),
        SimpleNamespace(  # turn 2: conclude
            content=[_tool_use("submit_triage", "t2", {
                "recommendation": "resolved_promote",
                "rationale": "More games made it decisive.",
                "evidence_summary": "Strong win after the added games.",
            })],
            stop_reason="tool_use",
        ),
    ])
    workflow = TriageWorkflow(
        EvaluationFunnel(batch_size=10, max_games=200), client=client
    )

    result = workflow.run(
        ExperimentSpec(config_path="x.yaml", baseline_id="b"),
        tier0,
        FakeMatchRunner(win_frac=0.8),
    )

    assert isinstance(result, TriageResult)
    assert result.games_added == 120
    assert result.verdict.recommendation == "resolved_promote"
    # Routing flips on the RECOMPUTED statistics, not the agent's word.
    assert result.tier0.is_clear_win


def test_triage_workflow_returns_none_on_failure():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("no API key")
    workflow = TriageWorkflow(EvaluationFunnel(), client=client)
    result = workflow.run(
        ExperimentSpec(config_path="x.yaml", baseline_id="b"),
        _ambiguous_tier0(),
        FakeMatchRunner(0.8),
    )
    assert result is None


class _StubTriage:
    def __init__(self, result):
        self.result = result

    def run(self, spec, tier0, match_runner):
        return self.result


def test_scheduler_triage_reroutes_ambiguous_to_promotion(tmp_path):
    # Ambiguous result (50% win rate) -> triage gathers evidence -> re-route.
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_tri", final_metrics=healthy)

    # The triage stub returns a re-evaluated clear-win Tier-0 (as if more games settled it).
    win_tier0 = EvaluationFunnel(batch_size=10, max_games=200).run_tier0(
        _healthy_input(), FakeMatchRunner(win_frac=0.8)
    )
    stub = _StubTriage(TriageResult(
        win_tier0,
        TriageVerdict("resolved_promote", "settled by more games", "116W/44L"),
        games_added=120,
    ))

    scheduler, store, _ = _make_scheduler(
        tmp_path, launcher, match_runner=FakeMatchRunner(win_frac=0.5), triage=stub
    )
    result = scheduler.process(ExperimentSpec(config_path="x.yaml", baseline_id="b"))

    # Was ambiguous; triage's added evidence re-routed it to the promotion gate.
    assert result.decision.gate_kind == "promotion"
    assert store.get_status("exp_tri") == "awaiting_ratification"
    report = (tmp_path / "journal" / "exp_tri.md").read_text()
    assert "Triage" in report and "120" in report


def test_scheduler_clear_result_skips_triage(tmp_path):
    # A decisive result must NOT invoke triage (only ambiguous review-gate results do).
    healthy = {"policy_target_entropy_mean": 1.2, "value_accuracy": 0.6, "value_variance": 0.05}
    launcher = FakeLauncher(str(tmp_path / "experiments.db"), "exp_clear", final_metrics=healthy)

    called = {"ran": False}

    class _SpyTriage:
        def run(self, spec, tier0, match_runner):
            called["ran"] = True
            return None

    scheduler, store, _ = _make_scheduler(
        tmp_path, launcher, match_runner=FakeMatchRunner(win_frac=0.8), triage=_SpyTriage()
    )
    scheduler.process(ExperimentSpec(config_path="x.yaml", baseline_id="b"))
    assert called["ran"] is False  # clear win never reached triage
    assert store.get_status("exp_clear") == "awaiting_ratification"


# --------------------------------------------------------------------------
# Rung 3 — proposer agent (model-driven trajectory)
# --------------------------------------------------------------------------

from yinsh_ml.orchestration import Proposal, ProposerAgent
from yinsh_ml.orchestration.registry import EvalResult


def _seed_registry(tmp_path):
    """Seed one completed experiment + eval so the proposer has something to read."""
    db_path = str(tmp_path / "experiments.db")
    db = ExperimentDB(db_path)
    db.create_experiment(ExperimentRecord(
        experiment_id="seed1", name="baseline-run", status="promoted",
        config_json='{"mcts": {"early_simulations": 100}}',
    ))
    store = OrchestrationStore(db_path)
    store.record_eval(EvalResult(
        experiment_id="seed1", baseline_id=None, tier=0,
        wins=30, draws=2, losses=10, sprt_verdict="accept_h1", sprt_llr=3.5,
        wilson_lower=0.6, wilson_upper=0.85, panel_green=True,
        panel_json='{"green": true, "checks": []}',
    ))
    return store


def test_registry_digest_and_detail(tmp_path):
    store = _seed_registry(tmp_path)

    digest = store.experiment_digest()
    assert digest[0]["experiment_id"] == "seed1"
    assert digest[0]["last_eval"]["sprt"] == "accept_h1"

    detail = store.experiment_detail("seed1")
    assert detail["evals"][0]["record"] == "30W/2D/10L"
    assert store.experiment_detail("nope") is None


def test_proposer_explores_then_proposes(tmp_path):
    store = _seed_registry(tmp_path)
    client = _ScriptedClient([
        SimpleNamespace(  # explore: overview
            content=[_tool_use("list_experiments", "t1", {"limit": 10})],
            stop_reason="tool_use",
        ),
        SimpleNamespace(  # explore: drill down
            content=[_tool_use("get_experiment", "t2", {"experiment_id": "seed1"})],
            stop_reason="tool_use",
        ),
        SimpleNamespace(  # decide
            content=[_tool_use("propose_experiment", "t3", {
                "base_config": "configs/smoke.yaml",
                "overrides": {"mcts.early_simulations": 200},
                "hypothesis": "More search fixes the offense-only collapse.",
                "rationale": "seed1 won but more search should harden defense.",
            })],
            stop_reason="tool_use",
        ),
    ])

    proposal = ProposerAgent(store, client=client).propose()

    assert isinstance(proposal, Proposal)
    assert proposal.overrides == {"mcts.early_simulations": 200}
    assert "search" in proposal.hypothesis.lower()
    # The agent really drove the trajectory: it explored before proposing.
    assert client.messages._responses == []  # all three turns consumed


def test_proposer_returns_none_on_failure(tmp_path):
    store = _seed_registry(tmp_path)
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("no API key")
    assert ProposerAgent(store, client=client).propose() is None


def test_journal_writes_proposal_artifact(tmp_path):
    proposal = Proposal(
        base_config="configs/smoke.yaml",
        overrides={"mcts.c_puct": 1.5},
        hypothesis="HYP-MARKER",
        rationale="RAT-MARKER",
    )
    path = Journal(str(tmp_path)).write_proposal("prop1", proposal)
    text = (tmp_path / "proposals" / "prop1.md").read_text()
    assert "HYP-MARKER" in text
    assert "c_puct" in text
    assert "yinsh-track schedule" in text  # the human-in-the-loop run instruction
