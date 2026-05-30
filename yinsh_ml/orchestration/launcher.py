"""The launch-target seam: local-first, cloud-burst.

The scheduler talks to a ``Launcher`` and never cares *where* a run executes.
``LocalLauncher`` runs the experiment in-process on whatever device torch finds
(MPS by default on the user's Mac) by driving the existing ``ExperimentRunner``.
``CloudLauncher`` is the stubbed seam — same interface, raising until the burst
path is wired — so adding cloud later changes one class, not the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .spec import ExperimentSpec


@dataclass
class LaunchResult:
    """What a launch produces, regardless of target."""

    experiment_id: str
    status: str
    """Final status reported by the runner: completed / failed / cancelled."""
    save_dir: str = ""
    """Directory the run wrote checkpoints to (for the evaluator to find them)."""
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    """Final iteration's metrics dict, fed to the failure panel."""
    raw: Dict[str, Any] = field(default_factory=dict)


class Launcher:
    """Interface every launch target implements."""

    def launch(self, spec: ExperimentSpec) -> LaunchResult:  # pragma: no cover
        raise NotImplementedError


class LocalLauncher(Launcher):
    """Runs the experiment in-process via the existing ExperimentRunner.

    Heavy deps (torch, supervisor) are imported lazily so the orchestration
    package imports cheaply in environments without a GPU/torch.
    """

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = output_dir

    def launch(self, spec: ExperimentSpec) -> LaunchResult:
        from ..experiments.experiment_config import load_config
        from ..experiments.experiment_runner import ExperimentRunner

        config = load_config(spec.config_path)
        runner = ExperimentRunner(config, output_dir=self.output_dir)
        results = runner.run()

        iterations = results.get("iterations", [])
        final_metrics = self._extract_final_metrics(iterations[-1]) if iterations else {}
        save_dir = f"{self.output_dir}/{results['experiment_id']}"

        return LaunchResult(
            experiment_id=results["experiment_id"],
            status=results.get("final_status", "completed"),
            save_dir=save_dir,
            final_metrics=final_metrics,
            raw=results,
        )

    @staticmethod
    def _extract_final_metrics(iteration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten the supervisor's nested iteration result into panel signals."""
        training = iteration_result.get("training", {})
        return {
            "policy_loss": training.get("policy_loss"),
            "value_loss": training.get("value_loss"),
            "value_accuracy": training.get("value_accuracy"),
            "value_variance": training.get("value_variance"),
            "policy_target_entropy_mean": training.get("policy_target_entropy_mean"),
        }


class CloudLauncher(Launcher):
    """Cloud-burst seam — same interface, not yet wired.

    Intentionally a stub: the design decision is that bursting to cloud changes
    only the launch target, never the experiment spec. When implemented this will
    provision an instance, ship the spec + repo state, run the same
    ``ExperimentRunner`` remotely, and pull back checkpoints + metrics.
    """

    def launch(self, spec: ExperimentSpec) -> LaunchResult:
        raise NotImplementedError(
            "Cloud-burst launch is a planned seam. Use target='local' (MPS) for now; "
            "cloud provisioning will reuse the same ExperimentSpec unchanged."
        )


def get_launcher(target: str, output_dir: str = "experiments") -> Launcher:
    """Resolve a launch target string to a Launcher."""
    if target == "local":
        return LocalLauncher(output_dir=output_dir)
    if target == "cloud":
        return CloudLauncher()
    raise ValueError(f"Unknown launch target: {target!r} (expected 'local' or 'cloud')")
