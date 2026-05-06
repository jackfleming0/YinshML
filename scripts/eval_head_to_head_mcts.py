#!/usr/bin/env python3
"""MCTS-based head-to-head tournament between checkpoints.

Production-realistic counterpart to eval_head_to_head.py (which uses raw
policy argmax). Use this to test whether the white-wins-100% pattern
seen under deterministic argmax persists when MCTS search is layered on
top — the production play setup.

Usage:
    python scripts/eval_head_to_head_mcts.py \\
        --run-dir runs/<RUN_DIR> \\
        --iterations 0 3 5 7 9 \\
        --num-games 40 \\
        --num-simulations 50 \\
        --device cuda \\
        --output-json eval_h2h_mcts.json
"""

import argparse
import json
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
from yinsh_ml.search.mcts import MCTS, MCTSConfig, EvaluationMode
from yinsh_ml.utils.encoding import StateEncoder

logger = logging.getLogger("eval_h2h_mcts")


def build_mcts(net: NetworkWrapper, sims: int) -> MCTS:
    """Match the play_step.py construction — pure neural, deterministic-ish."""
    cfg = MCTSConfig(
        num_simulations=sims,
        late_simulations=sims,
        simulation_switch_ply=10_000,
        c_puct=1.0,
        evaluation_mode=EvaluationMode.PURE_NEURAL,
        use_heuristic_evaluation=False,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
        # Disable root noise — we want repeatability, not exploration
        # (set epsilon_mix_* to 0 if those exist on MCTSConfig).
    )
    # MCTSConfig may not expose epsilon_mix_* fields directly; if it does,
    # override here. Otherwise the defaults are fine for diagnostic eval.
    return MCTS(network=net, config=cfg)


def play_match_mcts(
    white_net: NetworkWrapper,
    black_net: NetworkWrapper,
    num_simulations: int,
    num_games: int,
    seed: int,
    pair_label: str,
    max_moves: int = 200,
) -> Tuple[int, int, int]:
    """Play `num_games` between two networks using MCTS for move selection.
    Returns (white_wins, black_wins, draws)."""
    white_wins = black_wins = draws = 0
    encoder = StateEncoder()

    white_mcts = build_mcts(white_net, num_simulations)
    black_mcts = build_mcts(black_net, num_simulations)

    for game_num in range(num_games):
        game_seed = seed + game_num
        torch.manual_seed(game_seed)
        np.random.seed(game_seed)

        game = GameState()
        move_count = 0

        while not game.is_terminal() and move_count < max_moves:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            mcts = white_mcts if game.current_player == Player.WHITE else black_mcts

            # search() returns a flat policy vector over all encoded moves
            policy = mcts.search(game, move_number=move_count)

            # Pick highest-prob move that's legal. Map flat index to Move.
            valid_idxs = [encoder.move_to_index(m) for m in valid_moves]
            valid_probs = np.array([policy[i] for i in valid_idxs], dtype=np.float64)
            if valid_probs.sum() <= 0:
                # Fallback: uniform over valid
                selected = valid_moves[np.random.randint(len(valid_moves))]
            else:
                selected = valid_moves[int(np.argmax(valid_probs))]

            if not game.make_move(selected):
                break
            move_count += 1

        winner = game.get_winner()
        if winner == Player.WHITE:
            white_wins += 1
        elif winner == Player.BLACK:
            black_wins += 1
        else:
            draws += 1

        if (game_num + 1) % 10 == 0:
            logger.info(f"      [{pair_label}] {game_num+1}/{num_games} done")

    return white_wins, black_wins, draws


def play_pair_mcts(
    a_label: str, a_net: NetworkWrapper,
    b_label: str, b_net: NetworkWrapper,
    num_games_per_side: int,
    num_simulations: int,
    seed: int,
) -> dict:
    half_seed_a = seed
    half_seed_b = seed + 100_000

    a_white_wins, b_black_wins, draws_aw = play_match_mcts(
        a_net, b_net, num_simulations, num_games_per_side, half_seed_a,
        f"{a_label}_W_vs_{b_label}_B",
    )
    b_white_wins, a_black_wins, draws_bw = play_match_mcts(
        b_net, a_net, num_simulations, num_games_per_side, half_seed_b,
        f"{b_label}_W_vs_{a_label}_B",
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
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--num-games", type=int, default=40,
                        help="Total games per pair (split half white / half black)")
    parser.add_argument("--num-simulations", type=int, default=50,
                        help="MCTS simulations per move. 50 is enough for diagnostic.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.num_games % 2 != 0:
        logger.warning(f"num_games={args.num_games} is odd; rounding down to even.")
    half = args.num_games // 2

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}, MCTS sims/move: {args.num_simulations}")

    iters = args.iterations
    nets: dict = {}
    for it in iters:
        ckpt = args.run_dir / f"iteration_{it}" / f"checkpoint_iteration_{it}.pt"
        if not ckpt.exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        net = NetworkWrapper(device=device)
        net.load_model(str(ckpt))
        nets[it] = net
        logger.info(f"Loaded iter_{it} from {ckpt.name}")

    pairs_results = []
    t0 = time.time()
    for i, ai in enumerate(iters):
        for bj in iters[i + 1:]:
            label_a, label_b = f"iter_{ai}", f"iter_{bj}"
            logger.info(f"  Pair: {label_a} vs {label_b} ({half}+{half} games)")
            pair_t0 = time.time()
            res = play_pair_mcts(
                label_a, nets[ai], label_b, nets[bj],
                num_games_per_side=half, num_simulations=args.num_simulations,
                seed=args.seed,
            )
            pair_dt = time.time() - pair_t0
            total = res["a_wins"] + res["b_wins"] + res["draws"]
            ci_lo, ci_hi = wilson_ci_95(res["a_score"], total)
            res["ci95_lo"] = ci_lo
            res["ci95_hi"] = ci_hi
            res["seconds"] = pair_dt
            pairs_results.append(res)
            logger.info(
                f"    → {label_a}: {res['a_wins']}/{total} = {res['a_score']:.3f} "
                f"(CI95=[{ci_lo:.3f}, {ci_hi:.3f}]) in {pair_dt:.0f}s"
            )

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print(f"MCTS H2H complete in {elapsed:.0f}s. {len(pairs_results)} pairs:")
    print("=" * 72)
    print(f"{'pair':>22} {'a_wins':>7} {'b_wins':>7} {'draws':>6} {'a_score':>8} {'CI95':>17}  sig")
    for r in pairs_results:
        sig = "***" if r['ci95_lo'] > 0.5 or r['ci95_hi'] < 0.5 else ""
        print(
            f"{r['a_label']+' vs '+r['b_label']:>22} "
            f"{r['a_wins']:>7} {r['b_wins']:>7} {r['draws']:>6} "
            f"{r['a_score']:>8.3f}  "
            f"[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]  {sig}"
        )

    print("\n" + "-" * 72)
    print("Per-color split — flags white-wins pathology under MCTS:")
    print("-" * 72)
    print(f"{'pair':>22} {'a_W_wins':>10} {'a_B_wins':>10}  flag")
    for r in pairs_results:
        n_per_side = (r['a_wins'] + r['b_wins'] + r['draws']) // 2
        a_w = r.get('a_white_wins', 0)
        a_b = r.get('a_black_wins', 0)
        white_dominance = (a_w / n_per_side if n_per_side else 0) - (a_b / n_per_side if n_per_side else 0)
        flag = " ⚠ WHITE-WINS PATTERN" if abs(white_dominance) > 0.7 else ""
        print(
            f"{r['a_label']+' vs '+r['b_label']:>22} "
            f"{a_w:>10} {a_b:>10}  "
            f"(of {n_per_side} per side){flag}"
        )

    if args.output_json:
        out = {
            "config": {
                "run_dir": str(args.run_dir),
                "iterations": iters,
                "num_games": args.num_games,
                "num_simulations": args.num_simulations,
                "seed": args.seed,
                "device": device,
            },
            "pairs": pairs_results,
            "elapsed_seconds": elapsed,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote results to {args.output_json}")


if __name__ == "__main__":
    main()
