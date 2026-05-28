#!/usr/bin/env python3
"""Head-to-head among arbitrary checkpoint paths.

Less rigid than `eval_head_to_head.py` (no `iteration_N/` dir structure
required) — just pass `--checkpoint LABEL=path` flags. Plays a round-robin
between every pair, with each pair split half-white/half-black.

Optional `--temperature 0.3` adds stochasticity to break the deterministic
40-0 ties we saw in the gating-revert head-to-head (see
MODEL_PLAY_OBSERVATIONS.md). Default 0 still useful for quick directional
reads.

Usage:
    python scripts/eval_compare_checkpoints.py \\
        --checkpoint iter_0_v2=models/supervised_seed_humans_only/best_supervised.pt \\
        --checkpoint iter_0_old=models/supervised_seed/best_supervised.pt \\
        --checkpoint iter_3=runs_derisk_v2/20260428_184601/iteration_3/checkpoint_iteration_3.pt \\
        --num-games 30 --temperature 0.3 --device mps
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Player
from yinsh_ml.utils.tournament import derive_match_seed

logger = logging.getLogger("compare_ckpts")


def play_match(white: NetworkWrapper, black: NetworkWrapper,
               num_games: int, seed: int, temperature: float,
               max_moves: int = 200) -> Tuple[int, int, int]:
    """Play `num_games` games. Returns (white_wins, black_wins, draws)."""
    white_wins = black_wins = draws = 0
    w_inp = white._acquire_input_tensor(batch_size=1)
    b_inp = black._acquire_input_tensor(batch_size=1)
    try:
        for game_num in range(num_games):
            game_seed = derive_match_seed(seed, "W", "B", game_num)
            torch.manual_seed(game_seed)
            np.random.seed(game_seed)

            game = GameState()
            move_count = 0
            while not game.is_terminal() and move_count < max_moves:
                valid = game.get_valid_moves()
                if not valid:
                    break
                if game.current_player == Player.WHITE:
                    net, inp = white, w_inp
                else:
                    net, inp = black, b_inp
                state_array = net.state_encoder.encode_state(game)
                inp.copy_(torch.from_numpy(np.array(state_array)).unsqueeze(0))
                probs, _ = net.predict(inp)
                move = net.select_move(probs, valid, temperature=temperature)
                del probs
                if move is None or not game.make_move(move):
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
        white._release_tensor(w_inp)
        black._release_tensor(b_inp)
    return white_wins, black_wins, draws


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - half, centre + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        metavar="LABEL=PATH",
        help="Repeatable: a labelled checkpoint to include, e.g. "
             "iter_0_v2=models/.../best_supervised.pt",
    )
    parser.add_argument("--num-games", type=int, default=30,
                        help="Total games per pair, split half-W/half-B")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Move-selection temperature (0=argmax/deterministic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    logger.info(f"Device: {device}")

    # Parse --checkpoint LABEL=PATH
    entries: List[Tuple[str, Path]] = []
    for spec in args.checkpoint:
        if "=" not in spec:
            parser.error(f"--checkpoint must be LABEL=PATH (got {spec!r})")
        label, p = spec.split("=", 1)
        path = Path(p)
        if not path.exists():
            parser.error(f"Checkpoint not found: {path}")
        entries.append((label, path))

    if len(entries) < 2:
        parser.error("Need at least 2 checkpoints to compare.")

    # Load all networks. Memory cost: ~130 MB × N on the GPU; fine for 3-4
    # checkpoints on a 24 GB card (and MPS / CPU swap to RAM).
    nets: Dict[str, NetworkWrapper] = {}
    for label, path in entries:
        logger.info(f"Loading {label} from {path}")
        n = NetworkWrapper(model_path=str(path), device=device)
        nets[label] = n

    half = args.num_games // 2
    if args.num_games % 2:
        logger.warning(f"num_games={args.num_games} is odd; effective per-pair = {2*half}")

    pairs_results = []
    n_pairs = len(entries) * (len(entries) - 1) // 2
    logger.info(
        f"Round-robin: {len(entries)} models = {n_pairs} pairs × "
        f"{2*half} games = {n_pairs * 2 * half} total games. "
        f"Temperature={args.temperature}."
    )
    t0 = time.time()
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            la, lb = entries[i][0], entries[j][0]
            seed_a = args.seed
            seed_b = args.seed + 100_000
            logger.info(f"  Pair: {la} vs {lb}")
            # half games with A as white
            aw, bb, da = play_match(nets[la], nets[lb], half,
                                    seed_a, args.temperature, args.max_moves)
            # half games with B as white
            bw, ab, db = play_match(nets[lb], nets[la], half,
                                    seed_b, args.temperature, args.max_moves)
            a_wins = aw + ab     # A wins as white + A wins as black
            b_wins = bb + bw     # B wins as black + B wins as white
            draws = da + db
            total = a_wins + b_wins + draws
            score_a = a_wins / total if total else 0.0
            ci_lo, ci_hi = wilson_ci(score_a, total)
            sig = " ★★★" if (ci_lo > 0.5 or ci_hi < 0.5) else ""
            logger.info(
                f"    → {la}: {a_wins}/{total} = {score_a:.3f} "
                f"(CI95=[{ci_lo:.3f}, {ci_hi:.3f}]){sig}"
            )
            pairs_results.append({
                "a": la, "b": lb,
                "a_wins": int(a_wins), "b_wins": int(b_wins), "draws": int(draws),
                "a_white_wins": int(aw), "a_black_wins": int(ab),
                "a_score": float(score_a),
                "ci95": [float(ci_lo), float(ci_hi)],
                # Force-cast to Python bool — numpy comparisons return np.bool_
                # which json doesn't serialize.
                "significant": bool(ci_lo > 0.5 or ci_hi < 0.5),
            })
    elapsed = time.time() - t0

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Round-robin complete in {elapsed:.0f}s. Temperature={args.temperature}")
    print("=" * 80)
    print(f"{'pair':<40} {'a_wins':>7} {'b_wins':>7} {'draws':>5} {'a_score':>8}  CI95 sig")
    for r in pairs_results:
        sig = "★★★" if r["significant"] else ""
        print(
            f"  {r['a']+' vs '+r['b']:<38} {r['a_wins']:>7} {r['b_wins']:>7} "
            f"{r['draws']:>5} {r['a_score']:>8.3f}  "
            f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] {sig}"
        )

    # Aggregate ranking
    print("\n" + "-" * 80)
    print("Aggregate (average win share across all pairs):")
    print("-" * 80)
    per_label = {label: {"score_sum": 0.0, "n_pairs": 0} for label, _ in entries}
    for r in pairs_results:
        per_label[r["a"]]["score_sum"] += r["a_score"]
        per_label[r["a"]]["n_pairs"] += 1
        per_label[r["b"]]["score_sum"] += 1 - r["a_score"]
        per_label[r["b"]]["n_pairs"] += 1
    rankings = sorted(
        [(lbl, ag["score_sum"] / max(ag["n_pairs"], 1), ag["n_pairs"])
         for lbl, ag in per_label.items()],
        key=lambda x: x[1], reverse=True,
    )
    for rank, (lbl, avg, np_) in enumerate(rankings, 1):
        print(f"  #{rank}  {lbl:<25} avg score = {avg:.3f}  (over {np_} pairs)")
    print()

    if args.output_json:
        out = {
            "device": device,
            "num_games_per_pair": 2 * half,
            "temperature": args.temperature,
            "elapsed_seconds": elapsed,
            "checkpoints": [{"label": l, "path": str(p)} for l, p in entries],
            "pairs": pairs_results,
            "rankings": [{"rank": r, "label": l, "avg_score": s, "n_pairs": np_}
                         for r, (l, s, np_) in enumerate(rankings, 1)],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
