#!/usr/bin/env python3
"""Measure a trained YinshML checkpoint's win rate against the sharkdp/yinsh
negamax engine (https://github.com/sharkdp/yinsh).

This is the sharkdp twin of ``scripts/eval_vs_yngine.py`` — same game loop,
same JSON shape, same alternating-colors protocol — but the opponent searches
to a fixed negamax *depth* (``--depth``) instead of an MCTS sim budget, and is
driven through ``yinsh_ml.sharkdp.Sharkdp`` (which reuses the yngine wire codec).

Build the driver once::

    cd third_party/sharkdp_yinsh && cargo build --release -p yinsh_driver

Usage::

    python scripts/eval_vs_sharkdp.py \\
        --model-path models/iter1_ema_2026-05-27/iter1_ema.pt \\
        --num-sims 200 --num-games 20 --depth 6 \\
        --output logs/iter1_ema_vs_sharkdp_d6.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.game.constants import Player  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.training.self_play import MCTS as BatchedMCTS  # noqa: E402
from yinsh_ml.sharkdp import Sharkdp, default_sharkdp_binary_path  # noqa: E402
from yinsh_ml.sharkdp.bridge import SharkdpError  # noqa: E402

logger = logging.getLogger("eval_vs_sharkdp")


def wilson_ci_95(p: float, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def verdict(ci_lo: float, ci_hi: float) -> str:
    if ci_lo > 0.5:
        return "STRONGER"
    if ci_hi < 0.5:
        return "WEAKER"
    return "inconclusive"


def build_batched_mcts(net: NetworkWrapper, sims: int) -> BatchedMCTS:
    """Pure-neural batched MCTS, matching eval_vs_yngine.py."""
    return BatchedMCTS(
        network=net,
        evaluation_mode="pure_neural",
        heuristic_evaluator=None,
        num_simulations=sims,
        late_simulations=sims,
        simulation_switch_ply=10_000,
        enable_subtree_reuse=False,
        epsilon_mix_start=0.0,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=0,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
    )


def play_one_game(
    *,
    net: NetworkWrapper,
    mcts: BatchedMCTS,
    depth: int,
    model_is_white: bool,
    seed: int,
    opening_plies: int,
    opening_temp: float,
    max_moves: int,
    sharkdp_binary: Optional[Path],
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_color = Player.WHITE if model_is_white else Player.BLACK
    shark_color = model_color.opponent

    state = GameState()
    move_count = 0
    t0 = time.time()
    last_err: Optional[str] = None

    eng = Sharkdp.start(binary_path=sharkdp_binary, depth=depth)
    try:
        while not state.is_terminal() and move_count < max_moves:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                break

            if state.current_player == model_color:
                visit_probs = mcts.search_batch(state, move_count, batch_size=32)
                temp = opening_temp if move_count < opening_plies else 0.0
                probs_t = torch.from_numpy(np.asarray(visit_probs)).to(net.device)
                selected = net.select_move(probs_t, valid_moves, temperature=temp)
                del probs_t
                if selected is None or not state.make_move(selected):
                    last_err = "model picked illegal move"
                    break
                try:
                    eng.apply(selected)
                except SharkdpError as e:
                    last_err = f"sharkdp rejected our move: {e}"
                    break
            else:
                try:
                    chosen, wire = eng.get_move(player=shark_color, depth=depth)
                except SharkdpError as e:
                    last_err = f"sharkdp search failed: {e}"
                    break
                if not state.make_move(chosen):
                    last_err = f"sharkdp picked illegal move (per our engine): {chosen}"
                    break
                try:
                    eng.apply_wire(wire)
                except SharkdpError as e:
                    last_err = f"sharkdp apply_wire failed: {e}"
                    break
            move_count += 1

        winner = state.get_winner()
    finally:
        eng.stop()

    del state
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_color": "white" if model_is_white else "black",
        "winner": (
            "white" if winner == Player.WHITE
            else "black" if winner == Player.BLACK
            else "draw"
        ),
        "model_won": winner == model_color,
        "moves": move_count,
        "seconds": time.time() - t0,
        "error": last_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval a checkpoint vs the sharkdp/yinsh negamax engine.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--num-sims", type=int, default=200,
                        help="MCTS sims/move for our model.")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=6,
                        help="sharkdp negamax search depth.")
    parser.add_argument("--sharkdp-binary", type=Path, default=None)
    parser.add_argument("--opening-sample-plies", type=int, default=20,
                        help="Sample our model's moves for the first N plies "
                             "to diversify games.")
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "--output-json", dest="output",
                        type=Path, default=None)
    parser.add_argument("--quiet-mcts", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if args.quiet_mcts:
        logging.getLogger("MCTS").setLevel(logging.WARNING)

    if not args.model_path.exists():
        logger.error(f"Model checkpoint not found: {args.model_path}")
        sys.exit(1)

    binary = args.sharkdp_binary or default_sharkdp_binary_path()
    if not binary.exists():
        logger.error(
            f"yinsh-driver binary not found at {binary}. Build with:\n"
            "  cd third_party/sharkdp_yinsh && cargo build --release -p yinsh_driver"
        )
        sys.exit(1)

    model_label = f"{args.model_path.parent.name}/{args.model_path.stem}"
    logger.info(f"Loading model {model_label} from {args.model_path}")
    net = NetworkWrapper(model_path=str(args.model_path), device=args.device)
    net.load_model(str(args.model_path))
    mcts = build_batched_mcts(net, args.num_sims)
    resolved_device = str(net.device)
    logger.info(
        f"Device: {resolved_device}; model sims/move: {args.num_sims}; "
        f"sharkdp depth: {args.depth}; games: {args.num_games} (alternating colors)"
    )
    logger.info(f"sharkdp binary: {binary}")

    t0 = time.time()
    per_game: List[dict] = []
    for g in range(args.num_games):
        model_is_white = (g % 2 == 0)
        rec = play_one_game(
            net=net, mcts=mcts, depth=args.depth,
            model_is_white=model_is_white,
            seed=args.seed + g,
            opening_plies=args.opening_sample_plies,
            opening_temp=args.opening_temperature,
            max_moves=args.max_moves,
            sharkdp_binary=binary,
        )
        rec["game_idx"] = g
        per_game.append(rec)
        winner = rec["winner"]
        mw = "✓" if rec["model_won"] else ("=" if winner == "draw" else "✗")
        side = "W" if model_is_white else "B"
        running_w = sum(1 for r in per_game if r["model_won"])
        running_d = sum(1 for r in per_game if r["winner"] == "draw")
        running_l = len(per_game) - running_w - running_d
        suffix = f" [err: {rec['error']}]" if rec["error"] else ""
        logger.info(
            f"  game {g+1:>3}/{args.num_games}  model={side}  winner={winner}  "
            f"{mw}  moves={rec['moves']:>3}  {rec['seconds']:.1f}s  "
            f"score={running_w}-{running_l}-{running_d}{suffix}"
        )

    elapsed = time.time() - t0
    total = len(per_game)
    model_wins = sum(1 for r in per_game if r["model_won"])
    draws = sum(1 for r in per_game if r["winner"] == "draw")
    shark_wins = total - model_wins - draws
    wr = model_wins / total if total else 0.0
    ci_lo, ci_hi = wilson_ci_95(wr, total)
    model_white_wins = sum(
        1 for r in per_game if r["model_won"] and r["model_color"] == "white")
    model_black_wins = sum(
        1 for r in per_game if r["model_won"] and r["model_color"] == "black")
    result_verdict = verdict(ci_lo, ci_hi)

    print("\n" + "=" * 92)
    print(f"{model_label} vs sharkdp (depth {args.depth})  @ our sims="
          f"{args.num_sims}  [fixed-n {args.num_games}]  —  {elapsed:.0f}s")
    print("=" * 92)
    print(f"games: {total}   WR (model): {wr:.3f}   "
          f"CI95=[{ci_lo:.3f}, {ci_hi:.3f}]   verdict: {result_verdict}")
    print(f"  model wins: {model_wins}  (W: {model_white_wins}, B: {model_black_wins})")
    print(f"  sharkdp wins: {shark_wins}    draws: {draws}")

    if args.output:
        out = {
            "config": {
                "model_path": str(args.model_path),
                "model_label": model_label,
                "num_sims": args.num_sims,
                "num_games": args.num_games,
                "depth": args.depth,
                "sharkdp_binary": str(binary),
                "opening_sample_plies": args.opening_sample_plies,
                "opening_temperature": args.opening_temperature,
                "max_moves": args.max_moves,
                "device": resolved_device,
                "seed": args.seed,
                "engine": "self_play.MCTS.search_batch",
                "opponent": "sharkdp/yinsh negamax",
            },
            "results": [{
                "model_label": model_label,
                "model_wr": wr,
                "model_wins": model_wins,
                "sharkdp_wins": shark_wins,
                "draws": draws,
                "model_white_wins": model_white_wins,
                "model_black_wins": model_black_wins,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "verdict": result_verdict,
                "per_game": per_game,
            }],
            "elapsed_seconds": elapsed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
