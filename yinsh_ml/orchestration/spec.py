"""The unit of work the scheduler moves through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

LaunchTarget = Literal["local", "cloud"]


@dataclass
class ExperimentSpec:
    """A single experiment to schedule, run, and evaluate.

    The spec is deliberately launch-target-agnostic: the *same* spec runs on MPS
    locally or (later) on a cloud instance — only ``target`` changes, never the
    experiment definition. That is the seam that keeps "local-first, cloud-burst"
    from leaking into how experiments are described.
    """

    config_path: str
    """Path to the YAML config consumed by ``experiments.run_experiment``."""

    name: str = ""
    """Human label; defaults to the config's name once loaded."""

    baseline_id: Optional[str] = None
    """Experiment id of the anchor/predecessor this candidate is judged against.

    ``None`` means "first candidate" — Tier-0 still runs the failure panel but
    has no opponent to play, so the SPRT is skipped and the result is gated for
    review rather than auto-advanced.
    """

    target: LaunchTarget = "local"
    """Where to run. ``cloud`` is a stubbed seam in this slice."""

    games_dir: Optional[str] = None
    """Directory of replayable game parquet (GameRecorder format) for the
    offense-only-equilibrium audit. If unset, the panel looks for a conventional
    ``<save_dir>/parquet_data`` and otherwise skips that check."""

    tags: list[str] = field(default_factory=list)
