#!/usr/bin/env python3
"""Measure a trained YinshML checkpoint's win rate against the external yngine
engine (https://github.com/temhelk/yngine).

yngine is a pure-MCTS C++ engine with no neural net — the same engine that
generated the 200K-game `yngine_volume` pretraining corpus (May 2026). The
"defer yngine until you can beat the anchor" reasoning in
`VOLUME_PRETRAIN_RESULTS.md §"Session update — 2026-05-21"` has aged; this
harness is the V2b bridge that note flagged as deferred.

The model and yngine each play one full game; colors alternate per game so
neither side is locked into White or Black. Yngine and our MCTS each
maintain their own internal state, and `Yngine.apply()` keeps them in sync.

Usage::

    # Build the yngine driver once
    bash third_party/yngine_driver/build.sh

    # Smoke benchmark: deployed model at two sim budgets, 100 games each
    python scripts/eval_vs_yngine.py \\
        --model-path models/iter1_ema_2026-05-27/iter1_ema.pt \\
        --num-sims 200 --num-games 100 --yngine-sims 1000 \\
        --output logs/iter1_ema_vs_yngine_sims200.json

JSON schema (matches `eval_vs_frozen_anchor.py`'s shape where it overlaps):

    {
      "config": { … all CLI args … },
      "results": [{
        "model_wr": float,                         # WR over decisive + draws
        "model_wins": int, "yngine_wins": int, "draws": int,
        "model_white_wins": int, "model_black_wins": int,
        "ci95_lo": float, "ci95_hi": float,
        "verdict": "STRONGER|WEAKER|inconclusive",
        "per_game": [{ game_idx, model_color, winner, moves, seconds }, …]
      }],
      "elapsed_seconds": float
    }
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

from yinsh_ml.game.constants import Player  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.training.self_play import MCTS as BatchedMCTS  # noqa: E402
from yinsh_ml.yngine import Yngine, YngineError, default_binary_path  # noqa: E402

logger = logging.getLogger("eval_vs_yngine")


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
    """Pure-neural batched MCTS, matching `eval_vs_frozen_anchor.py`."""
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
    yngine_sims: int,
    yngine_threads: int,
    model_is_white: bool,
    seed: int,
    opening_plies: int,
    opening_temp: float,
    max_moves: int,
    yngine_binary: Optional[Path],
) -> dict:
    """Play one model-vs-yngine game and return a per-game record."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_color = Player.WHITE if model_is_white else Player.BLACK
    yngine_color = model_color.opponent

    state = GameState()
    move_count = 0
    t0 = time.time()
    last_err: Optional[str] = None

    eng = Yngine.start(binary_path=yngine_binary, threads=yngine_threads)
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
                except YngineError as e:
                    last_err = f"yngine rejected our move: {e}"
                    break
            else:
                try:
                    chosen, wire = eng.get_move(
                        player=yngine_color, sims=yngine_sims
                    )
                except YngineError as e:
                    last_err = f"yngine search failed: {e}"
                    break
                if not state.make_move(chosen):
                    last_err = f"yngine picked illegal move (per our engine): {chosen}"
                    break
                # Echo yngine's choice back so its internal MCTS tree advances.
                try:
                    eng.apply_wire(wire)
                except YngineError as e:
                    last_err = f"yngine apply_wire failed: {e}"
                    break
            move_count += 1

        winner = state.get_winner()
    finally:
        eng.stop()

    # Memory hygiene — long benchmarks accumulate MPS allocator state.
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


def run_sprt(
    *,
    net: NetworkWrapper,
    mcts: BatchedMCTS,
    yngine_sims: int,
    yngine_threads: int,
    yngine_binary: Optional[Path],
    p0: float, p1: float, alpha: float, beta: float, max_games: int,
    base_seed: int,
    opening_plies: int, opening_temp: float, max_moves: int,
) -> Tuple[List[dict], dict]:
    """Bernoulli SPRT on model WR. H0: WR=p0; H1: WR=p1, p1>p0.

    Same structure as eval_vs_frozen_anchor.py's run_sprt: alternate colors,
    update LLR after each decisive game (draws excluded), stop on boundary
    crossing or at max_games. Returns (per_game_records, sprt_summary).
    """
    import math
    upper = math.log((1.0 - beta) / alpha)      # accept H1 (STRONGER)
    lower = math.log(beta / (1.0 - alpha))       # accept H0 (NOT_STRONGER)
    win_inc = math.log(p1 / p0)
    loss_inc = math.log((1.0 - p1) / (1.0 - p0))

    per_game: List[dict] = []
    model_wins = yngine_wins = draws = 0
    model_white_wins = model_black_wins = 0
    llr = 0.0
    decision = "INCONCLUSIVE"
    for g in range(max_games):
        model_is_white = (g % 2 == 0)
        rec = play_one_game(
            net=net, mcts=mcts,
            yngine_sims=yngine_sims,
            yngine_threads=yngine_threads,
            model_is_white=model_is_white,
            seed=base_seed + g,
            opening_plies=opening_plies,
            opening_temp=opening_temp,
            max_moves=max_moves,
            yngine_binary=yngine_binary,
        )
        rec["game_idx"] = g
        per_game.append(rec)

        if rec["model_won"]:
            model_wins += 1
            model_white_wins += int(model_is_white)
            model_black_wins += int(not model_is_white)
            llr += win_inc
        elif rec["winner"] == "draw":
            draws += 1   # excluded from LLR
        else:
            yngine_wins += 1
            llr += loss_inc

        winner = rec["winner"]
        mw = "✓" if rec["model_won"] else ("=" if winner == "draw" else "✗")
        side = "W" if model_is_white else "B"
        suffix = f" [err: {rec['error']}]" if rec["error"] else ""
        logger.info(
            f"  game {g+1:>3}  model={side}  winner={winner}  {mw}  "
            f"moves={rec['moves']:>3}  {rec['seconds']:.1f}s  "
            f"score={model_wins}-{yngine_wins}-{draws}  "
            f"LLR={llr:+.2f} (L={lower:.2f}, U={upper:+.2f}){suffix}"
        )

        if llr >= upper:
            decision = "STRONGER"
            break
        if llr <= lower:
            decision = "NOT_STRONGER"
            break

    decisive = model_wins + yngine_wins
    wr_decisive = model_wins / decisive if decisive else 0.0
    ci_lo, ci_hi = wilson_ci_95(wr_decisive, decisive)
    summary = {
        "decision": decision,
        "games": len(per_game),
        "model_wins": model_wins,
        "yngine_wins": yngine_wins,
        "draws": draws,
        "model_white_wins": model_white_wins,
        "model_black_wins": model_black_wins,
        "llr": llr,
        "llr_upper": upper,
        "llr_lower": lower,
        "wr_decisive": wr_decisive,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p0": p0, "p1": p1, "alpha": alpha, "beta": beta,
        "max_games": max_games,
    }
    return per_game, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval a checkpoint vs the external yngine MCTS engine.")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Trained checkpoint .pt path.")
    parser.add_argument("--num-sims", type=int, default=800,
                        help="MCTS sims/move for our model.")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Total games (split evenly across colors when "
                             "--alternate-colors).")
    parser.add_argument(
        "--yngine-sims", type=int, default=1000,
        help="MCTS iterations for yngine. Default matches the 200K-game "
             "volume corpus generation level (see VOLUME_PRETRAIN_RESULTS.md).")
    parser.add_argument("--yngine-threads", type=int, default=1,
                        help="yngine worker thread count. 1 = deterministic-ish.")
    parser.add_argument("--yngine-binary", type=Path, default=None,
                        help="Override yngine_driver binary path "
                             "(default: third_party/yngine_driver/build-release/yngine_driver).")
    parser.add_argument("--alternate-colors", action="store_true", default=True,
                        help="Split games half model-as-white, half model-as-black "
                             "(default true).")
    parser.add_argument("--no-alternate-colors", action="store_false",
                        dest="alternate_colors")
    parser.add_argument("--opening-sample-plies", type=int, default=20,
                        help="Sample our model's moves for the first N plies "
                             "to diversify games. Yngine's random rollouts "
                             "give it natural diversity.")
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda|mps|cpu. Default: auto (mps on Apple silicon).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "--output-json", dest="output",
                        type=Path, default=None,
                        help="JSON results path. Created if needed.")
    parser.add_argument("--quiet-mcts", action="store_true", default=True,
                        help="Suppress verbose MCTS construction logging.")
    # --- SPRT (sequential probability ratio test) ---
    # Mirrors eval_vs_frozen_anchor.py. Stops as soon as the model's WR is
    # clearly above p1 or below p0. Big payoff against yngine because if the
    # model is sweeping (which the early MCTS-200 run revealed: 26-0), SPRT
    # terminates ≈10× faster than a fixed-n match — same conclusion, way
    # less compute.
    parser.add_argument("--sprt", action="store_true",
                        help="Use SPRT instead of a fixed-n match. Stops "
                             "as soon as the result is decisive.")
    parser.add_argument("--sprt-p0", type=float, default=0.50,
                        help="H0 win prob (model not stronger than yngine).")
    parser.add_argument("--sprt-p1", type=float, default=0.60,
                        help="H1 win prob (the smallest edge worth declaring "
                             "stronger). Wider (p1-p0) → faster, coarser.")
    parser.add_argument("--sprt-alpha", type=float, default=0.05,
                        help="Type-I error (false STRONGER).")
    parser.add_argument("--sprt-beta", type=float, default=0.05,
                        help="Type-II error (false NOT_STRONGER).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if args.quiet_mcts:
        logging.getLogger("MCTS").setLevel(logging.WARNING)

    if not args.model_path.exists():
        logger.error(f"Model checkpoint not found: {args.model_path}")
        sys.exit(1)

    binary = args.yngine_binary or default_binary_path()
    if not binary.exists():
        logger.error(
            f"yngine_driver binary not found at {binary}. Build with:\n"
            "  bash third_party/yngine_driver/build.sh"
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
        f"yngine sims/move: {args.yngine_sims} threads={args.yngine_threads}; "
        f"games: {args.num_games} ({'alternating' if args.alternate_colors else 'fixed'} colors)"
    )
    logger.info(f"yngine binary: {binary}")

    t0 = time.time()

    if args.sprt:
        per_game, sprt_summary = run_sprt(
            net=net, mcts=mcts,
            yngine_sims=args.yngine_sims,
            yngine_threads=args.yngine_threads,
            yngine_binary=binary,
            p0=args.sprt_p0, p1=args.sprt_p1,
            alpha=args.sprt_alpha, beta=args.sprt_beta,
            max_games=args.num_games,
            base_seed=args.seed,
            opening_plies=args.opening_sample_plies,
            opening_temp=args.opening_temperature,
            max_moves=args.max_moves,
        )
        model_wins = sprt_summary["model_wins"]
        yngine_wins = sprt_summary["yngine_wins"]
        draws = sprt_summary["draws"]
        model_white_wins = sprt_summary["model_white_wins"]
        model_black_wins = sprt_summary["model_black_wins"]
        wr = sprt_summary["wr_decisive"]
        ci_lo = sprt_summary["ci95_lo"]
        ci_hi = sprt_summary["ci95_hi"]
        result_verdict = sprt_summary["decision"]
    else:
        per_game = []
        for g in range(args.num_games):
            if args.alternate_colors:
                model_is_white = (g % 2 == 0)
            else:
                model_is_white = True
            rec = play_one_game(
                net=net, mcts=mcts,
                yngine_sims=args.yngine_sims,
                yngine_threads=args.yngine_threads,
                model_is_white=model_is_white,
                seed=args.seed + g,
                opening_plies=args.opening_sample_plies,
                opening_temp=args.opening_temperature,
                max_moves=args.max_moves,
                yngine_binary=binary,
            )
            rec["game_idx"] = g
            per_game.append(rec)
            winner = rec["winner"]
            mw = "✓" if rec["model_won"] else ("=" if winner == "draw" else "✗")
            side = "W" if model_is_white else "B"
            suffix = f" [err: {rec['error']}]" if rec["error"] else ""
            logger.info(
                f"  game {g+1:>3}/{args.num_games}  model={side}  winner={winner}  "
                f"{mw}  moves={rec['moves']:>3}  {rec['seconds']:.1f}s{suffix}"
            )
        sprt_summary = None
        model_wins = sum(1 for r in per_game if r["model_won"])
        draws = sum(1 for r in per_game if r["winner"] == "draw")
        yngine_wins = len(per_game) - model_wins - draws
        total = len(per_game)
        wr = model_wins / total if total else 0.0
        ci_lo, ci_hi = wilson_ci_95(wr, total)
        model_white_wins = sum(
            1 for r in per_game if r["model_won"] and r["model_color"] == "white"
        )
        model_black_wins = sum(
            1 for r in per_game if r["model_won"] and r["model_color"] == "black"
        )
        result_verdict = verdict(ci_lo, ci_hi)

    elapsed = time.time() - t0
    total = len(per_game)

    result = {
        "model_label": model_label,
        "model_path": str(args.model_path),
        "model_wr": wr,
        "model_wins": model_wins,
        "yngine_wins": yngine_wins,
        "draws": draws,
        "model_white_wins": model_white_wins,
        "model_black_wins": model_black_wins,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "verdict": result_verdict,
        "sprt": sprt_summary,
        "per_game": per_game,
        "elapsed_seconds": elapsed,
    }

    print("\n" + "=" * 92)
    mode = ("SPRT" if args.sprt else f"fixed-n {args.num_games}")
    print(f"{model_label} vs yngine (MCTS-{args.yngine_sims})  "
          f"@ our sims={args.num_sims}  [{mode}]  —  {elapsed:.0f}s")
    print("=" * 92)
    print(f"games: {total}   WR (model): {wr:.3f}   "
          f"CI95=[{ci_lo:.3f}, {ci_hi:.3f}]   verdict: {result_verdict}")
    print(f"  model wins: {model_wins}  (W: {model_white_wins}, B: {model_black_wins})")
    print(f"  yngine wins: {yngine_wins}    draws: {draws}")
    if args.sprt and sprt_summary is not None:
        print(f"  SPRT LLR: {sprt_summary['llr']:+.2f}  "
              f"(L={sprt_summary['llr_lower']:.2f}, U={sprt_summary['llr_upper']:+.2f})")

    if args.output:
        out = {
            "config": {
                "model_path": str(args.model_path),
                "model_label": model_label,
                "num_sims": args.num_sims,
                "num_games": args.num_games,
                "yngine_sims": args.yngine_sims,
                "yngine_threads": args.yngine_threads,
                "yngine_binary": str(binary),
                "alternate_colors": args.alternate_colors,
                "opening_sample_plies": args.opening_sample_plies,
                "opening_temperature": args.opening_temperature,
                "max_moves": args.max_moves,
                "device": resolved_device,
                "seed": args.seed,
                "engine": "self_play.MCTS.search_batch",
                "mode": "sprt" if args.sprt else "fixed_n",
                "sprt_params": (
                    {"p0": args.sprt_p0, "p1": args.sprt_p1,
                     "alpha": args.sprt_alpha, "beta": args.sprt_beta,
                     "max_games": args.num_games}
                    if args.sprt else None
                ),
            },
            "results": [result],
            "elapsed_seconds": elapsed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
