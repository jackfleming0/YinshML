#!/usr/bin/env python
"""HF-1: generate a self-play corpus, re-fit the 6 production weights, in one go.

No parquet corpus is present in a fresh container, so we generate games here
with the baseline HeuristicAgent (epsilon-greedy for play diversity), label each
main-game position by whether the side-to-move ultimately won, and fit the 6
production weights via the numpy logreg/correlation core.

CAVEAT (logged in EXPERIMENT_BACKLOG HF-1): fitting on baseline-agent games is
mildly circular — it learns "what predicts winning among baseline-vs-baseline
games". For a cheap close-the-loop check that's acceptable; the real fit uses
strong/external games on the cloud. Expected payoff is low (the Phase 1 finding
says the bottleneck is the linear form, not the weights).

Outputs a WeightManager-format JSON; A/B it vs baseline with ablation_phase1.
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_weights as vw  # noqa: E402

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase, MoveType
from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics.features import extract_all_features
from yinsh_ml.heuristics import weight_fitting as wf


def _gen_one(task):
    """Play one epsilon-greedy baseline game; return (phase, features, label)
    records for main-game positions, labeled by side-to-move final result."""
    base_path, depth, eps, seed = task
    rng = random.Random(seed)
    agent = vw._make_agent(base_path, depth, seed)
    state = GameState()
    records = []  # (move_count, features, player)
    for _ in range(400):
        if state.phase == GamePhase.GAME_OVER or state.is_stalemate():
            break
        player = state.current_player
        if state.phase == GamePhase.MAIN_GAME:
            records.append((len(state.move_history),
                            extract_all_features(state, player), player))
        # epsilon-greedy for diversity
        moves = state.get_valid_moves()
        if not moves:
            break
        if rng.random() < eps:
            move = rng.choice(moves)
        else:
            move = agent.select_move(state)
        if move is None or not state.make_move(move):
            break
        if state.white_score >= 3 or state.black_score >= 3:
            break
    winner = state.get_winner()
    if winner is None:
        return []
    return [(wf.phase_of_move_count(mc, 15, 35), feats, int(pl == winner))
            for mc, feats, pl in records]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--gen-depth", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--method", choices=["logreg", "correlation"], default="logreg")
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base = json.loads(Path(args.base).read_text())  # baseline weights (fallback + rescale ref)

    tasks = [(args.base, args.gen_depth, args.epsilon, args.seed + i)
             for i in range(args.games)]
    print(f"generating {args.games} epsilon-greedy games (eps={args.epsilon}, "
          f"depth={args.gen_depth}, workers={args.workers})...")
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            chunks = pool.map(_gen_one, tasks)
    else:
        chunks = [_gen_one(t) for t in tasks]
    samples = [r for chunk in chunks for r in chunk]
    by_phase = {p: sum(1 for s in samples if s[0] == p) for p in wf.PHASES}
    print(f"collected {len(samples)} labeled positions  per-phase={by_phase}")

    weights = wf.fit_weights_from_samples(
        samples, method=args.method, features=wf.PRODUCTION_FEATURES,
        scale=args.scale, min_samples_per_phase=50, fallback=base,
    )
    # Fair-comparison rescale: give each phase the SAME total weight budget
    # (L1) as the baseline, so the re-fit differs only in how it ALLOCATES that
    # budget across features — not in absolute eval magnitude (which was the
    # scaling confound that sank the feature ablation). Under-sampled phases
    # already fell back to baseline; leave those untouched.
    for phase in wf.PHASES:
        base_l1 = sum(abs(base[phase][f]) for f in wf.PRODUCTION_FEATURES)
        fit_l1 = sum(abs(weights[phase][f]) for f in wf.PRODUCTION_FEATURES)
        if fit_l1 > 1e-9 and abs(fit_l1 - base_l1) > 1e-9:
            k = base_l1 / fit_l1
            weights[phase] = {f: weights[phase][f] * k for f in wf.PRODUCTION_FEATURES}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(weights, indent=2))
    print(f"wrote re-fit weights -> {args.out}")
    for phase in wf.PHASES:
        print(f"  {phase}: " + ", ".join(
            f"{f}={weights[phase][f]:.2f}" for f in wf.PRODUCTION_FEATURES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
