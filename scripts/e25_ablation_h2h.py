#!/usr/bin/env python
"""E25 Part 2 — policy/value ablation H2H: which head bounds played strength?

Takes ONE model (the champion) and plays it against ITSELF under three MCTS
configs, color-balanced:
  full        — real policy prior + real value (reference)
  flatpolicy  — UNIFORM prior + real value   (value head alone guides search)
  blindvalue  — real prior + CONSTANT-0 value (policy prior alone guides search)

Verdict logic: whichever ablation collapses vs `full` is the BINDING head at this
play level. If `blindvalue` ≈ `full` but `flatpolicy` collapses, the POLICY carries
strength and value-target work (E26/E21) is mis-aimed. If `flatpolicy` ≈ `full`
but `blindvalue` collapses, the VALUE head is the lever — proceed to E26.

The ablations are driven by the `ablate_policy` / `ablate_value` flags on
yinsh_ml/training/self_play.py::MCTS (default off; no effect outside pure_neural).

Games are independent, so `--workers N` fans them across N CPU worker processes
(each loads its own net). Default is serial (workers=1, MPS) for back-compat.
NOTE: workers>1 saturates all cores — renice the workers if a co-hosted service
(e.g. the analysis board on the same box) must stay responsive.

Usage (parallel, fast):
  python scripts/e25_ablation_h2h.py \
      --model models/iter1_ema_2026-05-27/iter1_ema.pt \
      --games 60 --sims 200 --workers 8 --device cpu \
      --output docs/experiments/e25_ablation.json
"""
import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import MCTS
from yinsh_ml.utils.encoding import StateEncoder

VARIANTS = {
    "full":       dict(ablate_policy=False, ablate_value=False),
    "flatpolicy": dict(ablate_policy=True,  ablate_value=False),
    "blindvalue": dict(ablate_policy=False, ablate_value=True),
}
# Each ablation is pitted against `full` to localize the binding head.
PAIRINGS = [("full", "flatpolicy"), ("full", "blindvalue")]


def make_mcts(net, sims, variant):
    abl = VARIANTS[variant]
    return MCTS(
        network=net, evaluation_mode="pure_neural", heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=sims, late_simulations=sims,
        simulation_switch_ply=20, c_puct=1.0, dirichlet_alpha=0.0,
        value_weight=1.0, max_depth=300, epsilon_mix_start=0.0,
        epsilon_mix_end=0.0, epsilon_mix_taper_moves=1, initial_temp=0.5,
        final_temp=0.1, annealing_steps=20, temp_clamp_fraction=0.6,
        enable_subtree_reuse=True, fpu_reduction=0.25,
        ablate_policy=abl["ablate_policy"], ablate_value=abl["ablate_value"],
    )


def select_move(policy, valid, encoder, temp, rng):
    probs = np.zeros(len(valid), dtype=np.float64)
    for i, mv in enumerate(valid):
        idx = encoder.move_to_index(mv)
        if 0 <= idx < len(policy):
            probs[i] = float(policy[idx])
    if probs.sum() <= 0:
        return valid[rng.integers(0, len(valid))]
    if temp <= 1e-3:
        return valid[int(np.argmax(probs))]
    probs = probs ** (1.0 / temp)
    probs /= probs.sum()
    return valid[rng.choice(len(valid), p=probs)]


def play_game(white_mcts, black_mcts, encoder, rng, max_moves=300):
    state = GameState()
    move_count = 0
    while not state.is_terminal() and move_count < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break
        mcts = white_mcts if state.current_player == Player.WHITE else black_mcts
        policy = mcts.search_batch(state, move_count, batch_size=32)
        temp = mcts.get_temperature(move_count)
        sel = select_move(policy, valid, encoder, temp, rng)
        state.make_move(sel)
        white_mcts.advance_root(sel)
        black_mcts.advance_root(sel)
        move_count += 1
    if state.white_score > state.black_score:
        return "white", state.white_score, state.black_score, move_count
    if state.black_score > state.white_score:
        return "black", state.white_score, state.black_score, move_count
    return "draw", state.white_score, state.black_score, move_count


def ci95(score, n):
    """Normal-approx 95% CI half-width on a score rate (draws=0.5)."""
    if n == 0:
        return float("nan")
    p = score / n
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def tally(rec, a_is_white, result):
    if result == "draw":
        rec["draws"] += 1
    elif (result == "white") == a_is_white:
        rec["a_wins"] += 1
    else:
        rec["b_wins"] += 1


def finalize(rec):
    g = rec["a_wins"] + rec["b_wins"] + rec["draws"]
    score = rec["a_wins"] + 0.5 * rec["draws"]
    out = dict(rec)
    out["games"] = g
    out["a_score"] = score
    out["a_score_rate"] = score / g if g else float("nan")
    out["ci95"] = ci95(score, g)
    return out


# --- parallel worker: one NetworkWrapper per process ---
_NET = None


def _init_worker(model_path, device):
    global _NET
    _NET = NetworkWrapper(model_path=model_path, device=device)
    _NET.network.eval()


def _play_task(task):
    pi, gi, a, b, a_is_white, sims, seed = task
    encoder = StateEncoder()
    rng = np.random.default_rng(seed)
    wm = make_mcts(_NET, sims, a if a_is_white else b)
    bm = make_mcts(_NET, sims, b if a_is_white else a)
    result, ws, bs, nmv = play_game(wm, bm, encoder, rng)
    return pi, gi, a_is_white, result, ws, bs, nmv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--games", type=int, default=60, help="games per pairing (color-balanced)")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1, help=">1 fans games across CPU worker processes")
    ap.add_argument("--device", default="auto", help="auto|cpu|mps (cpu forced when workers>1)")
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = None if args.device == "auto" else args.device
    if args.workers > 1 and args.device == "auto":
        device = "cpu"  # MPS across forked workers is unsafe; CPU fan-out is the parallel model.
        print("workers>1: forcing device=cpu (pass --device to override)")

    recs = {(a, b): {"a": a, "b": b, "sims": args.sims, "a_wins": 0, "b_wins": 0, "draws": 0}
            for a, b in PAIRINGS}

    def save():
        out = [finalize(recs[p]) for p in PAIRINGS]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "sims": args.sims, "records": out}, f, indent=2)

    # One task per game; color-balanced (a is white on even game indices). Distinct
    # per-game seed keeps games reproducible and decorrelated across workers.
    tasks = []
    for pi, (a, b) in enumerate(PAIRINGS):
        for gi in range(args.games):
            tasks.append((pi, gi, a, b, (gi % 2 == 0), args.sims, args.seed + pi * 100000 + gi))

    print(f'[{time.strftime("%H:%M:%S")}] {len(tasks)} games '
          f'({len(PAIRINGS)} pairings x {args.games}) @ {args.sims} sims, '
          f'workers={args.workers}, device={device or "auto"}', flush=True)
    t0, done = time.time(), 0

    def record(pi, gi, a_is_white, result, ws, bs, nmv):
        nonlocal done
        a, b = PAIRINGS[pi]
        tally(recs[(a, b)], a_is_white, result)
        done += 1
        r = recs[(a, b)]
        rate = done / (time.time() - t0) * 60
        print(f"  [{a} vs {b}] {result} (W{ws}-B{bs}, {nmv}mv, {a}={'W' if a_is_white else 'B'}) | "
              f"{a} {r['a_wins']}-{r['b_wins']}-{r['draws']} | {done}/{len(tasks)} | {rate:.1f}/min",
              flush=True)
        if done % 10 == 0:
            save()

    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.model, device)) as pool:
            for res in pool.imap_unordered(_play_task, tasks):
                record(*res)
    else:
        _init_worker(args.model, device)
        for task in tasks:
            record(*_play_task(task))

    save()
    print(f'\n[{time.strftime("%H:%M:%S")}] ===== SUMMARY (a_score = full\'s score vs the ablation) =====')
    print(f"{'pairing':<26}{'a_score_rate':>14}{'±ci95':>9}{'  W-L-D (a)':>14}")
    print("-" * 64)
    for p in PAIRINGS:
        r = finalize(recs[p])
        print(f"{r['a']+' vs '+r['b']:<26}{r['a_score_rate']:>14.3f}{r['ci95']:>9.3f}"
              f"   {r['a_wins']}-{r['b_wins']}-{r['draws']}")
    print(f"\n[{time.strftime('%H:%M:%S')}] {(time.time()-t0)/60:.1f}m total. "
          "Read: a_score_rate >> 0.5 = ablation HURT (that head matters); "
          "~0.5 = that head was NOT binding.")


if __name__ == "__main__":
    main()
