#!/usr/bin/env python3
"""Non-saturated yardstick: candidate checkpoint(s) vs a FROZEN anchor.

The heuristic ladder (HA d1/d2/d3) is saturated — our best checkpoints sweep
it at 100%, so it can no longer discriminate "stronger" from "this strong."
This harness instead measures a candidate against a *fixed* checkpoint
(default `models/branchC_volume_pretrain/best_iter_4.pt`), giving a relative
gradient that stays informative as the model improves:

    candidate WR vs anchor ≈ 0.50  → no measurable improvement over anchor
    candidate WR vs anchor  > 0.55  → genuinely stronger (read CI, not point)
    candidate WR vs anchor saturates → re-freeze the anchor to it and climb again

This is the standard AlphaZero iterate-and-promote ladder (progress measured
against prior selves). An *absolute* external anchor (yngine) is only needed as
a collusion tripwire on long scaled runs, or to quantify how strong "stronger"
actually is — deferred until a run needs it. See VOLUME_PRETRAIN_RESULTS.md.

IMPLEMENTATION NOTE — uses the BATCHED MCTS (`yinsh_ml.training.self_play.MCTS`
.search_batch), the same engine `tournament.run_anchor_eval` uses for the
validated HA-ladder gates. The legacy `yinsh_ml.search.mcts.MCTS.search()` used
by `eval_head_to_head_mcts.py` is BROKEN for this purpose: it never expands the
root and returns a uniform-random policy (verified: effective_child_visits=0.0,
all moves = 1/N), so it is net-blind and every checkpoint plays identically.

Two MCTS players both selecting by argmax replay one deterministic line per
color, so a balanced color split reads ~0.500 for ANY candidate regardless of
strength (the deterministic-side artifact that `run_anchor_eval` also warns
about). We break it by sampling moves with `--opening-temperature` for the
first `--opening-sample-plies` plies, then play greedily.

Usage:
    python scripts/eval_vs_frozen_anchor.py \\
        --candidate models/yngine_volume_pretrain/best_supervised.pt

    python scripts/eval_vs_frozen_anchor.py \\
        --candidate runs_x/iter_3.pt runs_x/iter_5.pt \\
        --anchor models/branchC_volume_pretrain/best_iter_4.pt \\
        --num-games 40 --num-simulations 64 \\
        --output-json logs/frozen_anchor_eval.json
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root

from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.game.types import Player  # noqa: E402
from yinsh_ml.training.self_play import MCTS as BatchedMCTS  # noqa: E402

logger = logging.getLogger("eval_frozen_anchor")

DEFAULT_ANCHOR = "models/branchC_volume_pretrain/best_iter_4.pt"


def build_batched_mcts(net: NetworkWrapper, sims: int) -> BatchedMCTS:
    """Pure-neural batched MCTS, matching `run_anchor_eval`'s `_build_anchor_mcts`
    (subtree reuse off so one instance is safe to reuse across games — every
    search_batch call builds a fresh root from the passed state)."""
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


def wilson_ci_95(p: float, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def verdict(ci_lo: float, ci_hi: float) -> str:
    """One-word read on candidate strength vs anchor (CI95 vs 0.5)."""
    if ci_lo > 0.5:
        return "STRONGER"
    if ci_hi < 0.5:
        return "WEAKER"
    return "inconclusive"   # CI spans 0.5 — ≈ frozen-anchor strength


def play_one_game(
    white_net: NetworkWrapper, white_mcts: BatchedMCTS,
    black_net: NetworkWrapper, black_mcts: BatchedMCTS,
    seed: int, opening_plies: int, opening_temp: float, max_moves: int,
):
    """Play one game; return the winning Player (or None for a draw)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    game = GameState()
    move_count = 0
    while not game.is_terminal() and move_count < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break
        if game.current_player == Player.WHITE:
            net, mcts = white_net, white_mcts
        else:
            net, mcts = black_net, black_mcts

        visit_probs = mcts.search_batch(game, move_count, batch_size=32)
        # Sample during the opening to diversify games; greedy thereafter.
        temp = opening_temp if move_count < opening_plies else 0.0
        probs_t = torch.from_numpy(np.asarray(visit_probs)).to(net.device)
        selected = net.select_move(probs_t, valid_moves, temperature=temp)
        del probs_t
        if selected is None or not game.make_move(selected):
            break
        move_count += 1

    winner = game.get_winner()
    # Memory hygiene: the batched MCTS + MPS driver accumulate memory across a
    # long run (the production tournament reuses tensors for exactly this
    # reason). Free per game so long runs don't OOM in batch_norm.
    del game
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return winner


def play_match(
    white_net: NetworkWrapper, white_mcts: BatchedMCTS,
    black_net: NetworkWrapper, black_mcts: BatchedMCTS,
    num_games: int, seed: int, pair_label: str,
    opening_plies: int, opening_temp: float, max_moves: int,
) -> Tuple[int, int, int]:
    """Play `num_games`; returns (white_wins, black_wins, draws)."""
    white_wins = black_wins = draws = 0
    for game_num in range(num_games):
        winner = play_one_game(
            white_net, white_mcts, black_net, black_mcts,
            seed + game_num, opening_plies, opening_temp, max_moves)
        if winner == Player.WHITE:
            white_wins += 1
        elif winner == Player.BLACK:
            black_wins += 1
        else:
            draws += 1
        if (game_num + 1) % 5 == 0:
            logger.info(f"      [{pair_label}] {game_num + 1}/{num_games} done")
    return white_wins, black_wins, draws


def run_sprt(
    cand_net: NetworkWrapper, cand_mcts: BatchedMCTS,
    anchor_net: NetworkWrapper, anchor_mcts: BatchedMCTS,
    p0: float, p1: float, alpha: float, beta: float, max_games: int,
    base_seed: int, opening_plies: int, opening_temp: float, max_moves: int,
    label: str,
) -> dict:
    """Sequential Probability Ratio Test on the candidate's win probability.

    H0: p = p0 (not meaningfully stronger)   H1: p = p1 (stronger), p0 < p1.
    Bernoulli LLR over *decisive* games (YINSH is ~always decisive; draws are
    logged and excluded from the LLR — if a draw-heavy regime ever appears,
    switch to a GSPRT/pentanomial model). Colors alternate per game so the test
    can't be biased by playing one color first. Stops when the LLR crosses a
    boundary or at `max_games`.
    """
    import math
    upper = math.log((1.0 - beta) / alpha)        # accept H1 (STRONGER)
    lower = math.log(beta / (1.0 - alpha))         # accept H0 (NOT_STRONGER)
    win_inc = math.log(p1 / p0)
    loss_inc = math.log((1.0 - p1) / (1.0 - p0))

    cand_wins = anchor_wins = draws = 0
    cand_white_wins = cand_black_wins = 0
    llr = 0.0
    decision = "INCONCLUSIVE"
    games = 0
    for g in range(max_games):
        cand_is_white = (g % 2 == 0)   # alternate colors
        if cand_is_white:
            winner = play_one_game(cand_net, cand_mcts, anchor_net, anchor_mcts,
                                   base_seed + g, opening_plies, opening_temp, max_moves)
            cand_won = (winner == Player.WHITE)
            anchor_won = (winner == Player.BLACK)
        else:
            winner = play_one_game(anchor_net, anchor_mcts, cand_net, cand_mcts,
                                   base_seed + g, opening_plies, opening_temp, max_moves)
            cand_won = (winner == Player.BLACK)
            anchor_won = (winner == Player.WHITE)
        games += 1

        if cand_won:
            cand_wins += 1
            cand_white_wins += int(cand_is_white)
            cand_black_wins += int(not cand_is_white)
            llr += win_inc
        elif anchor_won:
            anchor_wins += 1
            llr += loss_inc
        else:
            draws += 1   # excluded from LLR

        if (g + 1) % 4 == 0 or llr >= upper or llr <= lower:
            logger.info(f"      [{label}] g={games} cand {cand_wins}-{anchor_wins}-{draws} "
                        f"LLR={llr:+.2f} (L={lower:.2f}, U={upper:+.2f})")
        if llr >= upper:
            decision = "STRONGER"
            break
        if llr <= lower:
            decision = "NOT_STRONGER"
            break

    decisive = cand_wins + anchor_wins
    wr_decisive = cand_wins / decisive if decisive else 0.0
    ci_lo, ci_hi = wilson_ci_95(wr_decisive, decisive)
    return {
        "label": label, "decision": decision, "games": games,
        "cand_wins": cand_wins, "anchor_wins": anchor_wins, "draws": draws,
        "cand_white_wins": cand_white_wins, "cand_black_wins": cand_black_wins,
        "llr": llr, "llr_lower": lower, "llr_upper": upper,
        "wr_decisive": wr_decisive, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "p0": p0, "p1": p1, "alpha": alpha, "beta": beta, "max_games": max_games,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eval candidate checkpoint(s) vs a frozen anchor checkpoint.")
    parser.add_argument("--candidate", type=Path, nargs="+", required=True,
                        help="One or more candidate checkpoint .pt paths.")
    parser.add_argument("--anchor", type=Path, default=Path(DEFAULT_ANCHOR),
                        help=f"Frozen anchor checkpoint (default: {DEFAULT_ANCHOR}).")
    parser.add_argument("--anchor-label", type=str, default=None)
    parser.add_argument("--num-games", type=int, default=40,
                        help="Total games per candidate (split half white / half black).")
    parser.add_argument("--num-simulations", type=int, default=64,
                        help="MCTS sims/move. 64 matches the validation-gate budget.")
    parser.add_argument("--opening-sample-plies", type=int, default=20,
                        help="Sample (not argmax) for the first N plies to diversify "
                             "games. 0 = deterministic (a balanced color split then "
                             "reads ~0.500 for any candidate — the deterministic-side "
                             "artifact).")
    parser.add_argument("--opening-temperature", type=float, default=1.0,
                        help="Sampling temperature applied during the opening plies.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda|mps|cpu. Default: auto (mps on Apple silicon).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quiet-mcts", action="store_true", default=True,
                        help="Suppress the verbose MCTS construction logging.")
    # --- SPRT mode (sequential testing; stops early on clear cases) ---
    parser.add_argument("--sprt", action="store_true",
                        help="Use a sequential probability ratio test instead of a "
                             "fixed-n match. Stops as soon as the result is decisive.")
    parser.add_argument("--sprt-p0", type=float, default=0.50,
                        help="H0 win prob (not meaningfully stronger). Default 0.50.")
    parser.add_argument("--sprt-p1", type=float, default=0.60,
                        help="H1 win prob (the smallest edge worth promoting). "
                             "Wider (p1-p0) → faster decisions, coarser threshold.")
    parser.add_argument("--sprt-alpha", type=float, default=0.05,
                        help="Type-I error (false STRONGER).")
    parser.add_argument("--sprt-beta", type=float, default=0.05,
                        help="Type-II error (false NOT_STRONGER).")
    parser.add_argument("--sprt-max-games", type=int, default=400,
                        help="Cap on games before declaring INCONCLUSIVE.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if args.quiet_mcts:
        logging.getLogger("MCTS").setLevel(logging.WARNING)

    if args.num_games % 2 != 0:
        logger.warning(f"num_games={args.num_games} is odd; rounding down to even.")
    half = args.num_games // 2

    if not args.anchor.exists():
        logger.error(f"Anchor checkpoint not found: {args.anchor}")
        sys.exit(1)

    anchor_label = args.anchor_label or f"anchor:{args.anchor.parent.name}/{args.anchor.stem}"
    logger.info(f"Loading frozen anchor {anchor_label} from {args.anchor}")
    anchor_net = NetworkWrapper(device=args.device)
    anchor_net.load_model(str(args.anchor))
    anchor_mcts = build_batched_mcts(anchor_net, args.num_simulations)
    resolved_device = str(anchor_net.device)
    logger.info(f"Device: {resolved_device}, batched MCTS sims/move: {args.num_simulations}, "
                f"{half}+{half} games/candidate, opening: {args.opening_sample_plies} plies "
                f"@ temp {args.opening_temperature}")

    results: List[dict] = []
    t0 = time.time()
    for cand_path in args.candidate:
        if not cand_path.exists():
            logger.error(f"Candidate checkpoint not found: {cand_path}")
            sys.exit(1)
        # Disambiguating label: parent dir + stem (two seeds both named best_supervised).
        cand_label = f"{cand_path.parent.name}/{cand_path.stem}"
        logger.info(f"  {cand_label} vs {anchor_label} ...")
        cand_net = NetworkWrapper(device=args.device)
        cand_net.load_model(str(cand_path))
        cand_mcts = build_batched_mcts(cand_net, args.num_simulations)

        pair_t0 = time.time()
        if args.sprt:
            s = run_sprt(
                cand_net, cand_mcts, anchor_net, anchor_mcts,
                args.sprt_p0, args.sprt_p1, args.sprt_alpha, args.sprt_beta,
                args.sprt_max_games, args.seed, args.opening_sample_plies,
                args.opening_temperature, args.max_moves, cand_label)
            res = {
                "candidate_path": str(cand_path),
                "candidate_label": cand_label,
                "anchor_path": str(args.anchor),
                "candidate_wr": s["wr_decisive"],
                "candidate_wins": s["cand_wins"], "anchor_wins": s["anchor_wins"],
                "draws": s["draws"],
                "cand_white_wins": s["cand_white_wins"], "cand_black_wins": s["cand_black_wins"],
                "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"],
                "verdict": s["decision"], "sprt": s,
                "seconds": time.time() - pair_t0,
            }
            results.append(res)
            logger.info(
                f"    → {cand_label}: {s['decision']} after {s['games']} games "
                f"(cand {s['cand_wins']}-{s['anchor_wins']}-{s['draws']}, "
                f"LLR={s['llr']:+.2f}) in {res['seconds']:.0f}s"
            )
            continue

        # Fixed-n path: candidate as White, then as Black.
        cand_w_wins, anc_b_wins, draws_w = play_match(
            cand_net, cand_mcts, anchor_net, anchor_mcts,
            half, args.seed, f"{cand_label}_W",
            args.opening_sample_plies, args.opening_temperature, args.max_moves)
        anc_w_wins, cand_b_wins, draws_b = play_match(
            anchor_net, anchor_mcts, cand_net, cand_mcts,
            half, args.seed + 100_000, f"{cand_label}_B",
            args.opening_sample_plies, args.opening_temperature, args.max_moves)

        cand_wins = cand_w_wins + cand_b_wins
        anc_wins = anc_w_wins + anc_b_wins
        draws = draws_w + draws_b
        total = cand_wins + anc_wins + draws
        wr = cand_wins / total if total else 0.0
        ci_lo, ci_hi = wilson_ci_95(wr, total)
        res = {
            "candidate_path": str(cand_path),
            "candidate_label": cand_label,
            "anchor_path": str(args.anchor),
            "candidate_wr": wr,
            "candidate_wins": cand_wins, "anchor_wins": anc_wins, "draws": draws,
            "cand_white_wins": cand_w_wins, "cand_black_wins": cand_b_wins,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "verdict": verdict(ci_lo, ci_hi),
            "seconds": time.time() - pair_t0,
        }
        results.append(res)
        logger.info(
            f"    → {cand_label}: {cand_wins}/{total} = {wr:.3f} "
            f"(CI95=[{ci_lo:.3f}, {ci_hi:.3f}]) {res['verdict']} in {res['seconds']:.0f}s "
            f"[W:{cand_w_wins}/{half} B:{cand_b_wins}/{half}]"
        )

    elapsed = time.time() - t0

    mode = (f"SPRT p0={args.sprt_p0} p1={args.sprt_p1} a={args.sprt_alpha} "
            f"b={args.sprt_beta} max={args.sprt_max_games}") if args.sprt \
           else f"fixed-n {args.num_games} games each"
    print("\n" + "=" * 92)
    print(f"Frozen-anchor eval vs {anchor_label}  ({args.num_simulations} sims, "
          f"{mode})  —  {elapsed:.0f}s")
    print("=" * 92)
    if args.sprt:
        print(f"{'candidate':>40} {'decision':>13} {'games':>6} {'cand':>5} {'anc':>4} "
              f"{'D':>3} {'LLR':>7}  {'W/B':>7}")
        for r in results:
            s = r["sprt"]
            print(
                f"{r['candidate_label']:>40} {r['verdict']:>13} {s['games']:>6} "
                f"{r['candidate_wins']:>5} {r['anchor_wins']:>4} {r['draws']:>3} "
                f"{s['llr']:>+7.2f}  {r['cand_white_wins']:>2}/{r['cand_black_wins']:<2}"
            )
    else:
        print(f"{'candidate':>40} {'WR':>7} {'cand':>5} {'anc':>4} {'D':>3} "
              f"{'CI95':>17}  {'W/B':>7}  verdict")
        for r in results:
            print(
                f"{r['candidate_label']:>40} {r['candidate_wr']:>7.3f} "
                f"{r['candidate_wins']:>5} {r['anchor_wins']:>4} {r['draws']:>3} "
                f"[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]  "
                f"{r['cand_white_wins']:>2}/{r['cand_black_wins']:<2}  {r['verdict']}"
            )

    if args.output_json:
        out = {
            "config": {
                "anchor": str(args.anchor),
                "anchor_label": anchor_label,
                "candidates": [str(c) for c in args.candidate],
                "mode": "sprt" if args.sprt else "fixed_n",
                "num_games": args.num_games,
                "num_simulations": args.num_simulations,
                "opening_sample_plies": args.opening_sample_plies,
                "opening_temperature": args.opening_temperature,
                "sprt": ({"p0": args.sprt_p0, "p1": args.sprt_p1,
                          "alpha": args.sprt_alpha, "beta": args.sprt_beta,
                          "max_games": args.sprt_max_games} if args.sprt else None),
                "seed": args.seed,
                "device": resolved_device,
                "engine": "self_play.MCTS.search_batch",
            },
            "results": results,
            "elapsed_seconds": elapsed,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Wrote results to {args.output_json}")


if __name__ == "__main__":
    main()
