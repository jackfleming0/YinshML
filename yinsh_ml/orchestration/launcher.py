"""The launch-target seam: local-first, cloud-burst.

The scheduler talks to a ``Launcher`` and never cares *where* a run executes.
``LocalLauncher`` runs the **real** training entrypoint (``scripts/run_training.py``)
on whatever device torch finds (MPS by default on the user's Mac) as a subprocess,
and records the run in the shared experiments registry. ``CloudLauncher`` is the
stubbed seam — same interface, raising until the burst path is wired.

Why subprocess to ``run_training.py`` rather than the ``experiments.ExperimentRunner``
it used to call: the repo's configs are all in the campaign format
(``self_play:``/``trainer:``/``arena:``/``num_iterations:``) that ``run_training.py``
consumes. ``ExperimentRunner`` expects a *different* schema and its loader silently
drops unrecognized keys — so pointing it at a real config quietly ran a default
10-iteration job instead of the configured one. Driving the real entrypoint removes
that trap entirely and guarantees orchestrated runs behave exactly like manual ones.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .spec import ExperimentSpec

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUN_TRAINING = _REPO_ROOT / "scripts" / "run_training.py"


@dataclass
class LaunchResult:
    """What a launch produces, regardless of target."""

    experiment_id: str
    status: str
    """Final status: completed / failed."""
    save_dir: str = ""
    """Directory the run wrote checkpoints to (for the evaluator to find them)."""
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    """Final iteration's panel signals, extracted from the run's metrics JSON."""
    raw: Dict[str, Any] = field(default_factory=dict)


class Launcher:
    """Interface every launch target implements."""

    def launch(self, spec: ExperimentSpec) -> LaunchResult:  # pragma: no cover
        raise NotImplementedError


class LocalLauncher(Launcher):
    """Runs the real training entrypoint as a subprocess; records it in the registry.

    Heavy deps (torch, the supervisor) live in the child process, so importing the
    orchestration package stays cheap and a training crash can't take down the
    orchestrator.
    """

    def __init__(self, output_dir: str = "experiments", runner=subprocess.run):
        self.output_dir = output_dir
        self._runner = runner  # injectable for tests

    def launch(self, spec: ExperimentSpec) -> LaunchResult:
        from ..experiments.experiment_db import ExperimentDB, ExperimentRecord
        from ..experiments.experiment_runner import get_git_info

        cfg = self._load_raw_config(spec.config_path)
        git = get_git_info()

        db = ExperimentDB(str(Path(self.output_dir) / "experiments.db"))
        total_iters = spec.iterations or int(cfg.get("num_iterations", 0))
        record = ExperimentRecord(
            name=spec.name or str(cfg.get("name") or Path(spec.config_path).stem),
            git_commit=git["commit"],
            git_branch=git["branch"],
            config_json=json.dumps(cfg),
            status="running",
            total_iterations=total_iters,
        )
        experiment_id = db.create_experiment(record)
        save_dir = str(Path(self.output_dir) / experiment_id)

        cmd = [
            sys.executable, str(_RUN_TRAINING),
            "--config", str(spec.config_path),
            "--save-dir", save_dir,
        ]
        if spec.iterations is not None:
            cmd += ["--iterations", str(spec.iterations)]
        if spec.init_checkpoint:
            # Warm-start: run_training loads weights only, resets optimizer/iteration.
            cmd += ["--init-checkpoint", str(spec.init_checkpoint)]

        logger.info("Launching training: %s", " ".join(cmd))
        proc = self._runner(cmd, cwd=str(_REPO_ROOT))
        status = "completed" if proc.returncode == 0 else "failed"
        db.update_experiment(experiment_id, status=status)

        final_metrics = self._read_final_metrics(save_dir) if status == "completed" else {}
        return LaunchResult(
            experiment_id=experiment_id,
            status=status,
            save_dir=save_dir,
            final_metrics=final_metrics,
        )

    @staticmethod
    def _load_raw_config(config_path: str) -> Dict[str, Any]:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _read_final_metrics(save_dir: str) -> Dict[str, Any]:
        """Pull panel signals from the highest-iteration metrics JSON under ``save_dir``.

        ``run_training.py`` nests a timestamp dir under ``save_dir``, so we glob
        recursively. Only signals the supervisor actually emits are returned; the
        rest stay absent so the panel skips those checks rather than false-flagging.
        ``policy_entropy`` is the network's PREDICTED policy entropy (the collapse
        signal), distinct from the MCTS target entropy. ``value_variance`` isn't
        emitted, so that part of the value check stays absent.
        """
        files = glob.glob(os.path.join(save_dir, "**", "metrics", "iteration_*.json"), recursive=True)
        if not files:
            return {}

        def _iter_num(p: str) -> int:
            stem = os.path.basename(p).replace("iteration_", "").replace(".json", "")
            try:
                return int(stem)
            except ValueError:
                return -1

        try:
            with open(max(files, key=_iter_num)) as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read metrics JSON: %s", exc)
            return {}

        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        training = metrics.get("training")
        last = training[-1] if isinstance(training, list) and training else (
            training if isinstance(training, dict) else {}
        )
        return {
            "policy_loss": last.get("policy_loss"),
            "value_loss": last.get("value_loss"),
            "value_accuracy": last.get("value_accuracy"),
            "policy_entropy": last.get("policy_entropy"),
        }


class CloudLauncher(Launcher):
    """Cloud-burst seam — same interface, not yet wired.

    Intentionally a stub: bursting to cloud changes only the launch target, never
    the experiment spec. When implemented this will provision an instance, ship the
    spec + repo state, run the same training entrypoint remotely, and pull back
    checkpoints + metrics.
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
