"""Extract completed-runs-differential trajectories from replayable game parquet.

This closes the one deferred gap in the failure-mode panel: the offense-only
equilibrium detector (`failure_panel.py`) consumes per-game signed
completed-runs-differential trajectories, but nothing produced them, so the check
silently skipped on real runs. This module produces them.

The signal is computed by replaying each game's serialized moves through a fresh
``GameState`` (`viz.game_replay`) and evaluating
``heuristics.features.completed_runs_differential`` at every ply, from a fixed
perspective so the sign is stable across the trajectory. It works off any parquet
written by the ``GameRecorder`` → ``ParquetDataStorage`` audit pipeline (e.g.
`scripts/generate_heuristic_games.py`); training-only state/policy/value parquet is
not replayable, so the detector still skips for those (gracefully) until games are
recorded.

Heavy deps (pandas via ``game_replay``, the heuristics/game engine) are imported
lazily so importing the orchestration package stays light.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def trajectory_from_replay(replay, perspective=None) -> List[float]:
    """Per-ply signed completed-runs differential for one replayed game.

    ``perspective`` is the ``Player`` whose advantage is positive; defaults to
    WHITE. Switching perspective negates the whole trajectory.
    """
    from ..game.types import Player
    from ..heuristics.features import completed_runs_differential

    if perspective is None:
        perspective = Player.WHITE
    return [
        float(completed_runs_differential(state, perspective))
        for _, state in replay.iter_states()
    ]


def run_diff_trajectories_from_parquet(
    parquet_dir,
    sample_k: int = 8,
    seed: int = 0,
    perspective=None,
) -> List[List[float]]:
    """Sample up to ``sample_k`` games from a replayable parquet dir → trajectories.

    Returns ``[]`` if the directory is missing, has no games, or none replay to a
    usable length — never raises, so the panel falls back to skipping the check.
    """
    pdir = Path(parquet_dir)
    if not pdir.is_dir():
        return []

    from ..viz.game_replay import list_games, load_game

    try:
        games = list_games(pdir)
    except Exception as exc:  # noqa: BLE001 - extraction is best-effort
        logger.warning("Could not list games in %s: %s", pdir, exc)
        return []
    if games is None or len(games) == 0:
        return []

    ids = [str(g) for g in games["game_id"].tolist()]
    if len(ids) > sample_k:
        ids = random.Random(seed).sample(ids, sample_k)

    trajectories: List[List[float]] = []
    for game_id in ids:
        try:
            replay = load_game(pdir, game_id)
            traj = trajectory_from_replay(replay, perspective)
        except Exception as exc:  # noqa: BLE001 - skip a bad game, keep the rest
            logger.warning("Skipping game %s: %s", game_id, exc)
            continue
        if len(traj) >= 2:
            trajectories.append(traj)
    return trajectories
