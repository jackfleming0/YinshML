#!/usr/bin/env python3
"""Bulk-generate HeuristicAgent-vs-HeuristicAgent games for warm-start data.

Pure-CPU, no GPU needed. Each game is played by independent
``HeuristicAgent`` instances (negamax + alpha-beta + transposition
table) and recorded via the standard ``GameRecorder`` →
``ParquetDataStorage`` pipeline, so output drops straight into
``scripts/dashboard_games.py`` for inspection.

Designed to address the "offense-only equilibrium" concern flagged in
``scripts/replay_heuristic_vs_heuristic.py``: depth 3+ is the cheapest
mitigation (defensive responses fall out of minimax). Diversity knobs
are provided for varying depth, seeds, and small ε-greedy noise so
the generated corpus doesn't collapse to a single playline.

Examples
--------
Quick smoke test — 4 games, depth 2, single-process::

    python scripts/generate_heuristic_games.py \\
        --output-dir self_play_data/ha_smoke --num-games 4 --depth 2 --workers 1

Production — 2000 games at mixed depths, 8 workers, small ε noise::

    python scripts/generate_heuristic_games.py \\
        --output-dir self_play_data/ha_v1 \\
        --num-games 2000 \\
        --depth-mix 2:20,3:60,4:20 \\
        --workers 8 --epsilon 0.05

Live mode — small parquet files appear as each game finishes so the
dashboard's auto-refresh picks them up immediately. ``--batch-size 1``
sets the parquet write cadence to "one file per game"; ``--chunk-size``
controls the multiprocessing pool's task batching and keeps parallelism::

    python scripts/generate_heuristic_games.py \\
        --output-dir self_play_data/live \\
        --num-games 100 --workers 4 --batch-size 1 --chunk-size 8
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.self_play.data_storage import ParquetDataStorage, StorageConfig  # noqa: E402
from yinsh_ml.self_play.game_recorder import GameRecord, GameRecorder  # noqa: E402

logger = logging.getLogger("generate_heuristic_games")


# ---------------------------------------------------------------------------
# Job description — pickle-safe so it can ride a multiprocessing.Pool.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GameJob:
    game_id: str
    depth: int
    seed: int
    time_limit_seconds: float
    epsilon: float
    max_moves: int
    output_dir: str  # only used so workers can write json fallback if needed


@dataclass
class GameOutcome:
    """Lightweight summary returned alongside the heavy ``GameRecord``."""
    game_id: str
    total_turns: int
    winner: Optional[str]
    white_score: int
    black_score: int
    elapsed_seconds: float
    truncated_at_max_moves: bool


def _parse_depth_mix(spec: str) -> List[Tuple[int, float]]:
    """Parse ``"2:30,3:50,4:20"`` → ``[(2, 0.30), (3, 0.50), (4, 0.20)]``."""
    items: List[Tuple[int, float]] = []
    total = 0.0
    for part in spec.split(","):
        depth_s, pct_s = part.split(":")
        depth = int(depth_s.strip())
        pct = float(pct_s.strip())
        items.append((depth, pct))
        total += pct
    if total <= 0:
        raise ValueError(f"Depth mix percentages sum to {total}; must be > 0")
    return [(d, p / total) for d, p in items]


def _sample_depth(depth_mix: Sequence[Tuple[int, float]], rng: random.Random) -> int:
    """Sample a depth from the normalized depth-mix distribution."""
    r = rng.random()
    cum = 0.0
    for depth, p in depth_mix:
        cum += p
        if r <= cum:
            return depth
    return depth_mix[-1][0]


def _chunks(iterable: Iterable, size: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        block = list(islice(it, size))
        if not block:
            return
        yield block


# ---------------------------------------------------------------------------
# Worker entry point.
# ---------------------------------------------------------------------------
def play_one_game(job: GameJob) -> Tuple[GameOutcome, GameRecord]:
    """Play a single game in this process and return ``(outcome, record)``."""
    rng = random.Random(job.seed)
    ha = HeuristicAgent(config=HeuristicAgentConfig(
        max_depth=job.depth,
        time_limit_seconds=job.time_limit_seconds,
        random_seed=job.seed,
    ))
    recorder = GameRecorder(
        output_dir=str(Path(job.output_dir) / "_worker_scratch"),
        save_json=False,
    )

    state = GameState()
    recorder.start_game(job.game_id)
    started = time.time()
    move_count = 0
    truncated = False

    while not state.is_terminal() and move_count < job.max_moves:
        valid = list(state.get_valid_moves())
        if not valid:
            break
        if job.epsilon > 0.0 and rng.random() < job.epsilon:
            # ε-greedy: pick a uniform-random legal move instead of HA's best.
            move = rng.choice(valid)
        else:
            move = ha.select_move(state)

        recorder.record_turn(state, move, state.current_player)
        state.make_move(move)
        move_count += 1

    if not state.is_terminal() and move_count >= job.max_moves:
        truncated = True

    winner = state.get_winner() if state.is_terminal() else None
    record = recorder.end_game(state, winner=winner)
    elapsed = time.time() - started

    outcome = GameOutcome(
        game_id=job.game_id,
        total_turns=move_count,
        winner=record.winner if record else None,
        white_score=record.final_score.get("white", 0) if record else 0,
        black_score=record.final_score.get("black", 0) if record else 0,
        elapsed_seconds=elapsed,
        truncated_at_max_moves=truncated,
    )
    return outcome, record


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------
def _build_jobs(args: argparse.Namespace) -> List[GameJob]:
    seed_rng = random.Random(args.seed_base)
    if args.depth_mix:
        mix = _parse_depth_mix(args.depth_mix)
    else:
        mix = [(args.depth, 1.0)]

    jobs: List[GameJob] = []
    for i in range(args.num_games):
        depth = _sample_depth(mix, seed_rng)
        jobs.append(GameJob(
            game_id=f"{args.game_id_prefix}_{i:06d}",
            depth=depth,
            seed=seed_rng.randint(0, 2**31 - 1),
            time_limit_seconds=args.time_limit_sec,
            epsilon=args.epsilon,
            max_moves=args.max_moves,
            output_dir=args.output_dir,
        ))
    return jobs


def _run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = ParquetDataStorage(config=StorageConfig(
        output_dir=str(output_dir),
        parquet_dir="parquet_data",
        batch_size=args.batch_size,
    ))

    jobs = _build_jobs(args)
    logger.info(
        "Generating %d games | workers=%d | batch_size=%d | depth_mix=%s | ε=%.3f",
        len(jobs), args.workers, args.batch_size,
        args.depth_mix or f"{args.depth}:100", args.epsilon,
    )

    started = time.time()
    n_done = 0
    n_truncated = 0
    n_white_wins = 0
    n_black_wins = 0
    n_draws = 0
    turn_counts: List[int] = []
    per_game_seconds: List[float] = []

    use_pool = args.workers > 1
    pool = mp.Pool(processes=args.workers) if use_pool else None

    # Multiprocessing chunk size is separate from the parquet write
    # cadence — otherwise --batch-size 1 (live mode) kills parallelism.
    chunk_size = args.chunk_size or max(args.batch_size, args.workers)

    try:
        for chunk in _chunks(jobs, chunk_size):
            if use_pool:
                results = pool.imap_unordered(play_one_game, chunk)
            else:
                results = (play_one_game(j) for j in chunk)

            for outcome, record in results:
                storage.store_game_record(record)
                n_done += 1
                turn_counts.append(outcome.total_turns)
                per_game_seconds.append(outcome.elapsed_seconds)
                if outcome.truncated_at_max_moves:
                    n_truncated += 1
                if outcome.winner == "WHITE":
                    n_white_wins += 1
                elif outcome.winner == "BLACK":
                    n_black_wins += 1
                else:
                    n_draws += 1

                if n_done % args.log_every == 0 or n_done == len(jobs):
                    elapsed = time.time() - started
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (len(jobs) - n_done) / rate if rate > 0 else 0
                    mean_turns = sum(turn_counts) / len(turn_counts)
                    logger.info(
                        "[%d/%d] %.1fs elapsed | %.1f games/s | "
                        "ETA %.0fs | mean turns=%.1f | "
                        "W:%d B:%d D:%d | trunc=%d",
                        n_done, len(jobs), elapsed, rate, eta, mean_turns,
                        n_white_wins, n_black_wins, n_draws, n_truncated,
                    )

            # Flush parquet at the end of each chunk so a crash loses <1 batch.
            storage.flush()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        storage.flush()

    total_elapsed = time.time() - started
    logger.info(
        "Done. %d games in %.1fs (%.2f games/s). "
        "Mean turns=%.1f. Outcomes: W=%d B=%d D=%d. "
        "Truncated at %d moves: %d (%.1f%%). "
        "Parquet dir: %s",
        n_done, total_elapsed, n_done / total_elapsed if total_elapsed > 0 else 0,
        sum(turn_counts) / len(turn_counts) if turn_counts else 0,
        n_white_wins, n_black_wins, n_draws,
        args.max_moves, n_truncated,
        100 * n_truncated / n_done if n_done else 0,
        storage.parquet_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, type=str,
                   help="Output directory (parquet written to <dir>/parquet_data/)")
    p.add_argument("--num-games", type=int, default=100,
                   help="Number of games to generate")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (1 = serial)")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Games per parquet file (write cadence). Use 1 for "
                        "live mode where each completed game gets its own file.")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Multiprocessing pool chunk size (independent of "
                        "parquet write cadence). Default: max(batch_size, workers).")

    depth_group = p.add_mutually_exclusive_group()
    depth_group.add_argument("--depth", type=int, default=3,
                             help="Negamax search depth (used if --depth-mix is not set)")
    depth_group.add_argument("--depth-mix", type=str, default=None,
                             help='Mixed depths, e.g. "2:30,3:50,4:20" (percentages)')

    p.add_argument("--time-limit-sec", type=float, default=2.0,
                   help="Per-move time budget (caps iterative deepening)")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="ε-greedy: probability of a uniform-random move at root, "
                        "for trajectory diversity")
    p.add_argument("--max-moves", type=int, default=300,
                   help="Truncate any game past this many moves (matches the "
                        "training self-play cap)")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Base random seed; per-game seeds are derived from this")
    p.add_argument("--game-id-prefix", type=str, default="ha",
                   help="Prefix for generated game_ids")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log progress every N completed games")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _run(args)


if __name__ == "__main__":
    main()
