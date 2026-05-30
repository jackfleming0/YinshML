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


def _make_scheduler(tmp_path, launcher, match_runner=None, panel_input=None):
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
