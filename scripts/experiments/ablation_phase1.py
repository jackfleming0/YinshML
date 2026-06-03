#!/usr/bin/env python
"""Phase 1: agent-level feature ablation (CPU, no training).

Isolates the *core* hypothesis from the game review — "do the new strategic
features capture real evaluation strength?" — without any of the confounds of a
training run:

  * Each arm = the FIXED baseline 6-feature weights PLUS exactly ONE palette
    feature at a chosen weight. base vs base+X is a one-variable change, so any
    win-rate difference is attributable to feature X alone.
  * No re-fitting => NO circularity (we don't fit on old-weights data at all).
    The "does re-fitting the 6 help" question is a *separate* axis; keep it out
    of this experiment so the feature signal stays isolated.
  * Each arm plays the baseline head-to-head with colors alternated and a FIXED
    seed schedule shared across arms (paired comparison), at several search
    depths (the review predicts defense matters at shallow depth and may wash
    out deeper).
  * Win-rate vs baseline is reported with a Wilson 95% CI so "significant" means
    the CI excludes 0.5 — not eyeballing.

This does NOT test whether a stronger heuristic propagates through training
(that's Phase 2, a separate design with Elo-trajectory metrics and seeds). It
isolates evaluation strength only. A strong negamax agent is also not a perfect
proxy for an MCTS leaf evaluator — see the review notes.

Example:
  python scripts/experiments/ablation_phase1.py \
      --base configs/heuristic_weights/baseline.json \
      --add defensive_disruption:4,8 \
      --add ring_mobility_differential:2,4 \
      --depths 1,2 --games 60 --out ablation_phase1.json
"""

import argparse
import copy
import json
import math
import sys
import tempfile
from pathlib import Path

# Reuse the (torch-free) match runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_weights as vw  # noqa: E402
import match_runner as mr  # noqa: E402

from yinsh_ml.heuristics.feature_registry import EXPERIMENTAL_FEATURES, PRODUCTION_FEATURES


def wilson_ci(wins: int, n_decided: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (decided games only)."""
    if n_decided == 0:
        return (0.0, 1.0)
    p = wins / n_decided
    denom = 1 + z * z / n_decided
    center = (p + z * z / (2 * n_decided)) / denom
    half = z * math.sqrt(p * (1 - p) / n_decided + z * z / (4 * n_decided * n_decided)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def with_feature(base: dict, feature: str, weight: float) -> dict:
    """Return baseline weights with one palette feature added at `weight`
    (same weight in every phase)."""
    out = copy.deepcopy(base)
    for phase in ("early", "mid", "late"):
        out[phase][feature] = float(weight)
    return out


def calibrate_abs_values(features, n_states=None):
    """Mean |value| of each feature over a calibration set (the human game).

    Used to convert a *contribution budget* into a fair per-feature weight:
    weight = budget / mean|value|. Without this, a fixed raw weight makes a
    large-magnitude feature (e.g. ring_mobility ~±8) dominate the tuned 6 while
    a small one (e.g. near_completion ~±0.3) barely registers — a scaling
    artifact that masquerades as "no signal".
    """
    from yinsh_ml.game.constants import Player
    from yinsh_ml.heuristics.experimental_features import extract_experimental_features
    from yinsh_ml.data.human_games import bga_862307561 as g

    sums = {f: 0.0 for f in features}
    n = 0
    for _t, _m, state in g.iter_states():
        ev = extract_experimental_features(state, Player.BLACK)
        for f in features:
            sums[f] += abs(ev.get(f, 0.0))
        n += 1
        if n_states and n >= n_states:
            break
    return {f: (sums[f] / n if n else 1.0) for f in features}


def parse_arms(add_specs):
    """['feat:4,8', ...] -> [(feat, 4.0), (feat, 8.0), ...]."""
    arms = []
    for spec in add_specs:
        feat, _, ws = spec.partition(":")
        feat = feat.strip()
        if feat not in EXPERIMENTAL_FEATURES:
            raise SystemExit(f"unknown palette feature {feat!r}; "
                             f"choose from {list(EXPERIMENTAL_FEATURES)}")
        weights = [float(w) for w in ws.split(",")] if ws else [4.0]
        for w in weights:
            arms.append((feat, w))
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="baseline 6-feature weights JSON")
    ap.add_argument("--add", action="append", default=[], metavar="FEATURE:n1,n2",
                    help="palette feature to ablate. The numbers are raw weights, "
                         "or (with --normalize) target contribution budgets. Repeatable.")
    ap.add_argument("--normalize", action="store_true",
                    help="interpret --add numbers as average CONTRIBUTION budgets "
                         "(weight = budget / mean|value|), so arms are fair across "
                         "features regardless of their value magnitude. Recommended.")
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--games", type=int, default=60, help="games per arm per depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel game workers (games are independent)")
    ap.add_argument("--out", default=None,
                    help="results JSON; written incrementally after each cell so "
                         "long runs never lose progress")
    args = ap.parse_args(argv)

    base = json.loads(Path(args.base).read_text())
    # sanity: base must be the 6 production features
    missing = set(PRODUCTION_FEATURES) - set(base.get("early", {}))
    if missing:
        raise SystemExit(f"base weights missing production features: {sorted(missing)}")

    arms = parse_arms(args.add)
    depths = [int(d) for d in args.depths.split(",")]

    # Optionally convert contribution budgets -> fair per-feature weights.
    abs_vals = {}
    if args.normalize:
        feats = sorted({f for f, _ in arms})
        abs_vals = calibrate_abs_values(feats)
        print("Calibration (mean|value| over the human game):")
        for f in feats:
            print(f"  {f:32} |val|~{abs_vals[f]:.3f}")
        print()

    tmp = Path(tempfile.mkdtemp(prefix="ablation_"))
    base_path = tmp / "base.json"
    base_path.write_text(json.dumps(base))

    rows = []
    print(f"Phase 1 ablation — each arm = baseline + ONE feature, vs baseline")
    print(f"{'arm':>40} {'depth':>5} {'W-L-D':>10} {'winrate':>8} {'95% CI':>15} "
          f"{'elo':>7} {'sig?':>5}")
    print("-" * 100)

    for feat, n in arms:
        # n is a raw weight, or (with --normalize) a contribution budget.
        if args.normalize:
            weight = n / max(abs_vals[feat], 1e-6)
            label = f"base+{feat}(c={n:g},w={weight:.2f})"
        else:
            weight = n
            label = f"base+{feat}@{n:g}"
        arm_weights = with_feature(base, feat, weight)
        arm_path = tmp / f"{feat}_{n}.json"
        arm_path.write_text(json.dumps(arm_weights))
        for depth in depths:
            if args.workers > 1:
                res = mr.run_ab_parallel(str(arm_path), str(base_path),
                                         games=args.games, depth=depth,
                                         seed=args.seed, workers=args.workers)
            else:
                res = vw.run_ab(str(arm_path), str(base_path),
                                games=args.games, depth=depth, seed=args.seed)
            decided = res["a_wins"] + res["b_wins"]
            lo, hi = wilson_ci(res["a_wins"], decided)
            sig = "yes" if (lo > 0.5 or hi < 0.5) else "no"
            print(f"{label:>40} {depth:>5} "
                  f"{res['a_wins']:>3}-{res['b_wins']:<3}-{res['draws']:<2} "
                  f"{res['a_win_rate']:>8.3f} [{lo:.2f},{hi:.2f}] "
                  f"{res['elo_delta_a_over_b']:>+7.0f} {sig:>5}")
            rows.append({
                "feature": feat, "budget_or_weight": n, "weight": weight,
                "normalized": args.normalize, "depth": depth,
                "arm_wins": res["a_wins"], "base_wins": res["b_wins"],
                "draws": res["draws"], "win_rate": res["a_win_rate"],
                "ci95": [lo, hi], "elo": res["elo_delta_a_over_b"],
                "significant": sig == "yes",
            })
            # Incremental write so a multi-hour/day run never loses progress.
            if args.out:
                Path(args.out).write_text(json.dumps(rows, indent=2))

    print("-" * 100)
    print("Reading: win-rate > 0.5 with CI excluding 0.5 => that feature adds "
          "evaluation strength at that weight/depth. CI includes 0.5 => "
          "underpowered or no effect (raise --games).")
    print("Caveat: multiple arms => multiple comparisons; treat marginal "
          "single-arm 'sig' with caution. This isolates evaluation strength, "
          "NOT training propagation (Phase 2).")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
