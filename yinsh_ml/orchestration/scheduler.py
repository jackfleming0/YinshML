"""The control plane: drives specs through the pipeline.

For this slice it runs **one** spec at a time (``max_concurrent`` exists as the
seam where real parallelism hangs, but isn't wired to a worker pool yet). The
lifecycle for each spec:

    queued → (launch) → running → evaluated → route:
        clear loss   → rejected               (auto, feed only)
        clear win    → awaiting_ratification   (gate: promotion)
        ambiguous    → awaiting_ratification   (gate: review)
    ratify → promoted | rejected

Heavy work is reached through three injectable hooks (launcher, panel-input
builder, match-runner factory) so the whole thread can be exercised with fakes —
no GPU, no checkpoints — while production wiring uses the real defaults.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from .failure_panel import FailurePanel, PanelInput
from .funnel import EvaluationFunnel, Tier0Result
from .interpreter import Interpretation, PIInterpreter
from .journal import Journal
from .launcher import Launcher, LaunchResult, get_launcher
from .match_runner import MatchRunner
from .pi import PIRouter, RoutingDecision
from .registry import EvalResult, GateItem, OrchestrationStore
from .spec import ExperimentSpec

logger = logging.getLogger(__name__)

# Hook signatures.
LauncherFactory = Callable[[str, str], Launcher]
PanelInputBuilder = Callable[[ExperimentSpec, LaunchResult], PanelInput]
MatchRunnerFactory = Callable[
    [ExperimentSpec, LaunchResult, str], Optional[MatchRunner]
]


@dataclass
class ProcessResult:
    """What processing one spec produced (returned for inspection/testing)."""

    experiment_id: str
    launch_status: str
    tier0: Optional[Tier0Result]
    decision: Optional[RoutingDecision]
    report_path: str = ""


class Scheduler:
    def __init__(
        self,
        store: OrchestrationStore,
        journal: Journal,
        funnel: Optional[EvaluationFunnel] = None,
        router: Optional[PIRouter] = None,
        output_dir: str = "experiments",
        launcher_factory: LauncherFactory = get_launcher,
        panel_input_builder: Optional[PanelInputBuilder] = None,
        match_runner_factory: Optional[MatchRunnerFactory] = None,
        interpreter: Optional[PIInterpreter] = None,
        max_concurrent: int = 1,
    ):
        self.store = store
        self.journal = journal
        self.funnel = funnel or EvaluationFunnel()
        self.router = router or PIRouter()
        # Rung 1: optional Claude-authored narrative. None -> templated writeups.
        self.interpreter = interpreter
        self.output_dir = output_dir
        self.launcher_factory = launcher_factory
        self.panel_input_builder = panel_input_builder or _default_panel_input
        self.match_runner_factory = match_runner_factory or _default_match_runner
        self.max_concurrent = max_concurrent
        self._queue: List[ExperimentSpec] = []

    # --- queue ------------------------------------------------------------

    def enqueue(self, spec: ExperimentSpec) -> None:
        self._queue.append(spec)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def run_queue(self) -> List[ProcessResult]:
        """Process the whole queue (sequentially in this slice)."""
        results = []
        while self._queue:
            results.append(self.run_next())
        return results

    def run_next(self) -> ProcessResult:
        if not self._queue:
            raise IndexError("scheduler queue is empty")
        return self.process(self._queue.pop(0))

    # --- the pipeline -----------------------------------------------------

    def process(self, spec: ExperimentSpec) -> ProcessResult:
        """Launch one spec, evaluate it (Tier 0), route the result."""
        launcher = self.launcher_factory(spec.target, self.output_dir)
        launch = launcher.launch(spec)

        if launch.status not in ("completed",):
            # Run didn't finish cleanly — surface it, don't evaluate a partial run.
            self.store.set_status(launch.experiment_id, "failed")
            self.journal.append_feed(
                launch.experiment_id,
                f"💥 Run did not complete (status: {launch.status})",
                f"Config `{spec.config_path}`; no evaluation performed.",
            )
            return ProcessResult(launch.experiment_id, launch.status, None, None)

        self.store.set_status(launch.experiment_id, "evaluated")

        panel_input = self.panel_input_builder(spec, launch)
        match_runner = (
            self.match_runner_factory(spec, launch, self.output_dir)
            if spec.baseline_id
            else None
        )
        tier0 = self.funnel.run_tier0(panel_input, match_runner)
        decision = self.router.route(tier0)

        # Rung 1: the LLM advises (narrative), the rules gate (routing already decided).
        interpretation: Optional[Interpretation] = (
            self.interpreter.interpret(spec, tier0, decision)
            if self.interpreter is not None
            else None
        )

        self._persist(spec, launch, tier0, decision)
        report_path = self.journal.write_report(
            launch.experiment_id, spec, tier0, decision, interpretation
        )
        # Feed headline stays deterministic (scannable route + record); the LLM's
        # one-liner enriches the detail when available.
        feed_detail = (
            interpretation.headline if interpretation is not None else decision.feed_detail
        )
        self.journal.append_feed(
            launch.experiment_id, decision.feed_headline, feed_detail
        )

        if decision.gate_kind is not None:
            self.store.enqueue_gate(
                GateItem(
                    experiment_id=launch.experiment_id,
                    kind=decision.gate_kind,
                    summary=decision.gate_summary,
                    report_path=report_path,
                )
            )
        self.store.set_status(launch.experiment_id, decision.next_status)

        return ProcessResult(
            launch.experiment_id, launch.status, tier0, decision, report_path
        )

    def _persist(self, spec, launch, tier0: Tier0Result, decision) -> None:
        self.store.record_eval(
            EvalResult(
                experiment_id=launch.experiment_id,
                baseline_id=spec.baseline_id,
                tier=0,
                wins=tier0.outcome.wins,
                losses=tier0.outcome.losses,
                draws=tier0.outcome.draws,
                sprt_verdict=(tier0.sprt.verdict if tier0.sprt else "no_baseline"),
                sprt_llr=(tier0.sprt.llr if tier0.sprt else 0.0),
                wilson_lower=tier0.wilson_lower,
                wilson_upper=tier0.wilson_upper,
                panel_green=tier0.panel_green,
                panel_json=json.dumps(tier0.panel.to_dict()),
            )
        )

    # --- ratification (the blocking gate) ---------------------------------

    def ratify(self, experiment_id: str, approve: bool = True) -> str:
        """Resolve the open gate item for an experiment.

        Approving a promotion (or accepting an ambiguous result) advances the
        candidate to ``promoted``; declining sends it to ``rejected``. Returns the
        resulting status.
        """
        item = self.store.open_gate_for_experiment(experiment_id)
        if item is None:
            raise ValueError(f"No open gate item for experiment {experiment_id}")

        if approve:
            self.store.resolve_gate(item.id, "ratified")
            self.store.set_status(experiment_id, "promoted")
            new_status = "promoted"
        else:
            self.store.resolve_gate(item.id, "rejected")
            self.store.set_status(experiment_id, "rejected")
            new_status = "rejected"

        self.journal.append_feed(
            experiment_id,
            f"{'👍 Ratified' if approve else '👎 Declined'} ({item.kind})",
            f"{item.summary} → {new_status}",
        )
        return new_status


# --- default hooks (real wiring; bypassed by injected fakes in tests) -----


def _default_panel_input(spec: ExperimentSpec, launch: LaunchResult) -> PanelInput:
    """Build panel signals from the run's final metrics.

    Run-diff trajectory extraction from parquet (for the offense-only detector) is
    a noted gap — left ``None`` here, so that check skips until wired.
    """
    return PanelInput.from_metrics(launch.final_metrics)


def _default_match_runner(
    spec: ExperimentSpec, launch: LaunchResult, output_dir: str
) -> Optional[MatchRunner]:
    """Locate candidate + baseline checkpoints and build a real match runner.

    Degrades gracefully: if either checkpoint can't be found the SPRT is skipped
    (the result then routes to review rather than auto-advancing).
    """
    candidate = _latest_checkpoint(launch.save_dir)
    baseline = _latest_checkpoint(os.path.join(output_dir, spec.baseline_id or ""))
    if candidate is None or baseline is None:
        logger.warning(
            "Could not locate checkpoints (candidate=%s, baseline=%s); skipping SPRT.",
            candidate, baseline,
        )
        return None

    from .match_runner import TournamentMatchRunner

    device = "mps" if spec.target == "local" else "cuda"
    return TournamentMatchRunner(
        candidate_ckpt=candidate,
        baseline_ckpt=baseline,
        training_dir=launch.save_dir,
        device=device,
    )


def _latest_checkpoint(run_dir: str) -> Optional[str]:
    if not run_dir or not os.path.isdir(run_dir):
        return None
    matches = glob.glob(
        os.path.join(run_dir, "**", "checkpoint_iteration_*.pt"), recursive=True
    )
    if not matches:
        return None
    # Highest iteration number wins.
    def iter_num(p: str) -> int:
        stem = os.path.basename(p).replace("checkpoint_iteration_", "").replace(".pt", "")
        try:
            return int(stem)
        except ValueError:
            return -1
    return max(matches, key=iter_num)
