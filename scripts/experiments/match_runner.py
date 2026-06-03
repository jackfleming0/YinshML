"""Game-level parallel match runner for agent ablations (torch-free).

Games are independent, so we fan them across processes. Used by
ablation_phase1.py / ablation_sweep.py to make powered (high-game-count,
multi-depth) sweeps tractable on a many-core box.

On Linux (fork start method) the pool workers inherit the imported agent stack;
the worker function is module-level so it pickles cleanly.
"""

import multiprocessing as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_weights as vw  # noqa: E402
from yinsh_ml.game.constants import Player  # noqa: E402


def _play_one(task):
    """Play one game. task = (a_path, b_path, depth, game_idx, base_seed).
    Returns +1 if A wins, -1 if B wins, 0 for draw. Colors alternate by index
    so neither side keeps the first-move edge; A's and B's agent seeds are
    distinct and game-indexed for independent randomness."""
    a_path, b_path, depth, game_idx, base_seed = task
    a = vw._make_agent(a_path, depth, base_seed + game_idx)
    b = vw._make_agent(b_path, depth, base_seed + 100_000 + game_idx)
    if game_idx % 2 == 0:
        winner, a_is = vw.play_game(a, b), Player.WHITE
    else:
        winner, a_is = vw.play_game(b, a), Player.BLACK
    if winner is None:
        return 0
    return 1 if winner == a_is else -1


def run_ab_parallel(a_path, b_path, games, depth, seed, workers):
    """Parallel A-vs-B match. Returns the same dict shape as
    validate_weights.run_ab (a_wins/b_wins/draws/a_win_rate/elo_delta)."""
    tasks = [(a_path, b_path, depth, g, seed) for g in range(games)]
    if workers and workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_play_one, tasks)
    else:
        results = [_play_one(t) for t in tasks]
    a_wins = sum(1 for r in results if r == 1)
    b_wins = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    decided = a_wins + b_wins
    win_rate = (a_wins + 0.5 * draws) / games if games else 0.0
    return {
        "games": games, "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_win_rate": win_rate,
        "elo_delta_a_over_b": vw.win_rate_to_elo(win_rate) if decided else 0.0,
    }
