#!/usr/bin/env python3
"""Head-to-head tournament between specific YinshML checkpoints.

Plays N games between each pair of checkpoints from a list. Reports per-pair
W/L/D and an aggregate Glicko-style ranking. Independent of the supervisor's
sliding-window tournament — useful for cross-iter comparisons that the
sliding window doesn't cover (e.g. iter_0 vs iter_9 from a 12-iter run with
sliding_window=2).

Usage:
    python scripts/eval_head_to_head.py \\
        --run-dir runs_derisk_v2/20260428_184601 \\
        --iterations 0 3 5 7 9 11 \\
        --num-games 40 \\
        --device cuda
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Player
from yinsh_ml.utils.tournament import derive_match_seed

logger = logging.getLogger("eval_h2h")


def play_match(
    white_net: NetworkWrapper,
    black_net: NetworkWrapper,
    num_games: int,
    seed: int,
    pair_label: str,
    max_moves: int = 200,
) -> Tuple[int, int, int]:
    """Play N games between two networks. Returns (white_wins, black_wins, draws).

    Uses raw policy argmax (temperature=0). Fast — no MCTS at this layer; each
    move is one NN forward. Same path used by tournament.run_anchor_eval for
    raw-policy mode.
    """
    white_wins, black_wins, draws = 0, 0, 0
    white_input = white_net._acquire_input_tensor(batch_size=1)
    black_input = black_net._acquire_input_tensor(batch_size=1)

    try:
        for game_num in range(num_games):
            game_seed = derive_match_seed(seed, "white", "black", game_num)
            torch.manual_seed(game_seed)
            np.random.seed(game_seed)

            game = GameState()
            move_count = 0
            while not game.is_terminal() and move_count < max_moves:
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break

                if game.current_player == Player.WHITE:
                    net = white_net
                    inp = white_input
                else:
                    net = black_net
                    inp = black_input

                state_array = net.state_encoder.encode_state(game)
                inp.copy_(torch.from_numpy(np.array(state_array)).unsqueeze(0))
                move_probs, _ = net.predict(inp)
                selected = net.select_move(move_probs, valid_moves, temperature=0.0)
                del move_probs

                if selected is None or not game.make_move(selected):
                    break
                move_count += 1

            winner = game.get_winner()
            if winner == Player.WHITE:
                white_wins += 1
            elif winner == Player.BLACK:
                black_wins += 1
            else:
                draws += 1
    finally:
        white_net._release_tensor(white_input)
        black_net._release_tensor(black_input)

    return white_wins, black_wins, draws


def play_pair(
    a_label: str,
    a_net: NetworkWrapper,
    b_label: str,
    b_net: NetworkWrapper,
    num_games_per_side: int,
    seed: int,
) -> dict:
    """Play `num_games_per_side` with A as white, then `num_games_per_side`
    with B as white. Returns aggregated W/L/D from A's perspective.
    """
    half_seed_a = seed
    half_seed_b = seed + 100_000

    a_white_wins, b_black_wins, draws_aw = play_match(
        a_net, b_net, num_games_per_side, half_seed_a, f"{a_label}_W_vs_{b_label}_B"
    )
    b_white_wins, a_black_wins, draws_bw = play_match(
        b_net, a_net, num_games_per_side, half_seed_b, f"{b_label}_W_vs_{a_label}_B"
    )

    a_wins = a_white_wins + a_black_wins
    b_wins = b_black_wins + b_white_wins
    draws = draws_aw + draws_bw
    total = a_wins + b_wins + draws
    a_score = a_wins / total if total else 0.0
    return {
        "a_label": a_label, "b_label": b_label,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_score": a_score,
        "a_white_wins": a_white_wins, "a_black_wins": a_black_wins,
    }


def wilson_ci_95(p: float, n: int) -> Tuple[float, float]:
    """Wilson score interval at 95% confidence."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to run dir (e.g. runs_derisk_v2/20260428_184601)")
    parser.add_argument("--iterations", type=int, nargs="+", required=True,
                        help="Iteration indices to include (e.g. 0 3 5 7 9 11)")
    parser.add_argument("--num-games", type=int, default=40,
                        help="Total games per pair (split half white / half black)")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / mps / cpu / auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path to write results as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    logger.info(f"Device: {device}")

    if args.num_games % 2 != 0:
        logger.warning(f"num_games={args.num_games} is odd; rounding down to even half-splits.")
    half = args.num_games // 2

    iters = sorted(set(args.iterations))
    nets = {}
    for it in iters:
        ckpt = args.run_dir / f"iteration_{it}" / f"checkpoint_iteration_{it}.pt"
        if not ckpt.exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        net = NetworkWrapper(device=device)
        # Use the wrapper's load_model so size mismatch hard-fails with a clear error.
        net.load_model(str(ckpt))
        nets[it] = net
        logger.info(f"Loaded iter_{it} from {ckpt.name}")

    pairs_results = []
    n_pairs = len(iters) * (len(iters) - 1) // 2
    logger.info(
        f"Running round-robin over {len(iters)} models = {n_pairs} pairs × "
        f"{args.num_games} games = {n_pairs * args.num_games} total games."
    )

    t0 = time.time()
    for i, ai in enumerate(iters):
        for bj in iters[i + 1:]:
            label_a, label_b = f"iter_{ai}", f"iter_{bj}"
            logger.info(f"  Pair: {label_a} vs {label_b} ({half}+{half} games)")
            res = play_pair(
                label_a, nets[ai], label_b, nets[bj],
                num_games_per_side=half, seed=args.seed,
            )
            total = res["a_wins"] + res["b_wins"] + res["draws"]
            ci_lo, ci_hi = wilson_ci_95(res["a_score"], total)
            res["ci95_lo"] = ci_lo
            res["ci95_hi"] = ci_hi
            sig = " (CI excludes 0.5)" if ci_lo > 0.5 or ci_hi < 0.5 else ""
            logger.info(
                f"    → {label_a}: {res['a_wins']}/{total} = {res['a_score']:.3f} "
                f"(CI95=[{ci_lo:.3f}, {ci_hi:.3f}]){sig}"
            )
            pairs_results.append(res)
    elapsed = time.time() - t0

    # Aggregate: per-model win count and average score against the field.
    print("\n" + "=" * 64)
    print(f"Round-robin complete in {elapsed:.0f}s. Pairs:")
    print("=" * 64)
    print(f"{'pair':>20} {'a_wins':>7} {'b_wins':>7} {'draws':>6} {'a_score':>8} {'CI95':>17}  sig")
    for r in pairs_results:
        sig = "***" if r['ci95_lo'] > 0.5 or r['ci95_hi'] < 0.5 else ""
        print(
            f"{r['a_label']+' vs '+r['b_label']:>20} "
            f"{r['a_wins']:>7} {r['b_wins']:>7} {r['draws']:>6} "
            f"{r['a_score']:>8.3f}  "
            f"[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]  {sig}"
        )

    # Per-color split — surfaces "white-wins-everything" pathology where the
    # aggregate looks even but every game just goes to whoever moves first.
    print("\n" + "-" * 64)
    print("Per-color split (a_wins broken down by who was white):")
    print("-" * 64)
    print(f"{'pair':>20} {'a_W_wins':>10} {'a_B_wins':>10}  interpretation")
    for r in pairs_results:
        n_per_side = (r['a_wins'] + r['b_wins'] + r['draws']) // 2
        a_w = r.get('a_white_wins', 0)
        a_b = r.get('a_black_wins', 0)
        # Flag the white-wins-only pattern: a wins almost all when white, almost none when black
        white_dominance = (a_w / n_per_side if n_per_side else 0) - (a_b / n_per_side if n_per_side else 0)
        flag = " ⚠ WHITE-WINS PATTERN" if white_dominance > 0.7 or white_dominance < -0.7 else ""
        print(
            f"{r['a_label']+' vs '+r['b_label']:>20} "
            f"{a_w:>10} {a_b:>10}  "
            f"(of {n_per_side} per side){flag}"
        )

    # Per-model aggregate.
    print("\n" + "-" * 64)
    print("Per-model aggregate (avg score vs field):")
    print("-" * 64)
    per_model = {it: {"score_sum": 0.0, "n_pairs": 0} for it in iters}
    for r in pairs_results:
        a = int(r["a_label"].split("_")[1])
        b = int(r["b_label"].split("_")[1])
        per_model[a]["score_sum"] += r["a_score"]
        per_model[a]["n_pairs"] += 1
        per_model[b]["score_sum"] += 1 - r["a_score"]
        per_model[b]["n_pairs"] += 1
    rankings = []
    for it, agg in per_model.items():
        avg = agg["score_sum"] / agg["n_pairs"] if agg["n_pairs"] else 0.0
        rankings.append((it, avg, agg["n_pairs"]))
    rankings.sort(key=lambda x: x[1], reverse=True)
    for rank, (it, avg, np_) in enumerate(rankings, 1):
        print(f"  #{rank}  iter_{it}: avg score = {avg:.3f} (over {np_} pairs)")

    if args.output_json:
        import json
        out = {
            "iterations": iters, "num_games_per_pair": args.num_games,
            "seed": args.seed, "device": device,
            "elapsed_seconds": elapsed,
            "pairs": pairs_results,
            "rankings": [{"rank": r, "iter": it, "avg_score": s, "n_pairs": np_}
                         for r, (it, s, np_) in enumerate(rankings, 1)],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote results to {args.output_json}")


if __name__ == "__main__":
    main()
