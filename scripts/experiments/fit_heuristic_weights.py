#!/usr/bin/env python
"""Re-fit production heuristic feature weights from game outcomes (B).

Produces a WeightManager-format JSON that the training loop can consume via the
config key ``self_play.heuristic_weight_config_file`` (see plumbing A).

Data sources (pick one):
  --games-dir DIR   parquet self-play games (uses yinsh_ml.viz.game_replay;
                    needs pandas/pyarrow -> cloud). The real path.
  --demo            tiny smoke source: the transcribed human game fixture.
                    Statistically meaningless (one game) but exercises the full
                    pipeline end-to-end and emits a valid JSON. Runs anywhere.
  --dump-baseline   no fitting: dump the evaluator's current default weights as a
                    JSON baseline (the "old weights" arm for an A/B).

Examples:
  # cloud: refit on 100k parquet games with logistic regression
  python scripts/experiments/fit_heuristic_weights.py \
      --games-dir large_scale_selfplay_data/parquet_data \
      --method logreg --out configs/heuristic_weights/refit_logreg.json

  # baseline arm
  python scripts/experiments/fit_heuristic_weights.py --dump-baseline \
      --out configs/heuristic_weights/baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics import weight_fitting as wf
from yinsh_ml.heuristics.features import extract_all_features
from yinsh_ml.heuristics.experimental_features import extract_experimental_features
from yinsh_ml.heuristics.weight_manager import WeightManager


def _winner_player(winner_str):
    """Map a stored winner field to a Player (or None for draw/unknown)."""
    if winner_str in (1, "1", "WHITE", "white", Player.WHITE):
        return Player.WHITE
    if winner_str in (-1, "-1", "BLACK", "black", Player.BLACK):
        return Player.BLACK
    return None


def _samples_from_states(states_with_turn, winner, early_max, mid_max,
                         include_experimental=False):
    """Yield (phase, feature_dict, label) from one game's per-ply states.

    Features are taken from the side-to-move's perspective; label = 1 if that
    side ultimately won the game, else 0. Draws/unknown winners are skipped.
    """
    win_player = _winner_player(winner)
    if win_player is None:
        return
    for _turn_idx, state in states_with_turn:
        player = state.current_player
        phase = wf.phase_of_move_count(len(state.move_history), early_max, mid_max)
        feats = extract_all_features(state, player)
        if include_experimental:
            feats = {**feats, **extract_experimental_features(state, player)}
        yield phase, feats, int(player == win_player)


def _collect_demo(early_max, mid_max, include_experimental):
    from yinsh_ml.data.human_games import bga_862307561 as game
    states = [(t, s) for t, _m, s in game.iter_states()]
    return list(_samples_from_states(states, "BLACK", early_max, mid_max,
                                     include_experimental))


def _collect_parquet(games_dir, limit, early_max, mid_max, include_experimental):
    # Cloud path: needs pandas/pyarrow.
    from yinsh_ml.viz.game_replay import list_games, load_game
    parquet_dir = Path(games_dir)
    index = list_games(parquet_dir)
    samples = []
    n = 0
    for _, row in index.iterrows():
        gid, winner = row["game_id"], row.get("winner")
        try:
            replay = load_game(parquet_dir, gid)
        except Exception as exc:  # noqa: BLE001 - skip unreadable games
            print(f"  skip {gid}: {exc}", file=sys.stderr)
            continue
        samples.extend(_samples_from_states(
            replay.iter_states(), replay.winner or winner,
            early_max, mid_max, include_experimental))
        n += 1
        if limit and n >= limit:
            break
    print(f"collected {len(samples)} positions from {n} games")
    return samples


def _dump_baseline(out):
    from yinsh_ml.heuristics.evaluator import YinshHeuristics
    weights = YinshHeuristics().weights
    # keep only the 6 production features per phase (drop any extras)
    clean = {p: {f: float(weights[p][f]) for f in wf.PRODUCTION_FEATURES}
             for p in wf.PHASES}
    _write_and_validate(clean, out)
    print(f"wrote baseline (default) weights -> {out}")


def _write_and_validate(weights, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(weights, fh, indent=2)
    # prove it loads (structure + [0,50] constraints)
    WeightManager().load_from_file(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--games-dir")
    src.add_argument("--demo", action="store_true")
    src.add_argument("--dump-baseline", action="store_true")
    ap.add_argument("--method", choices=["logreg", "correlation"], default="logreg")
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--early-max", type=int, default=15)
    ap.add_argument("--mid-max", type=int, default=35)
    ap.add_argument("--min-samples-per-phase", type=int, default=50)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--include-experimental", action="store_true",
                    help="also fit/report experimental palette coefficients "
                         "(informational; not loaded by the production evaluator)")
    args = ap.parse_args(argv)

    if args.dump_baseline:
        _dump_baseline(args.out)
        return 0

    if args.demo:
        samples = _collect_demo(args.early_max, args.mid_max, args.include_experimental)
    else:
        samples = _collect_parquet(args.games_dir, args.limit_games,
                                   args.early_max, args.mid_max, args.include_experimental)

    if not samples:
        print("no samples collected", file=sys.stderr)
        return 1

    # Fit production weights (the loadable artifact).
    weights = wf.fit_weights_from_samples(
        samples, method=args.method, features=wf.PRODUCTION_FEATURES,
        scale=args.scale, l2=args.l2,
        min_samples_per_phase=args.min_samples_per_phase,
    )
    _write_and_validate(weights, args.out)
    print(f"wrote re-fit weights ({args.method}) -> {args.out}")
    for phase in wf.PHASES:
        print(f"  {phase}: " + ", ".join(
            f"{f}={weights[phase][f]:.2f}" for f in wf.PRODUCTION_FEATURES))

    # Informational: experimental coefficients, if requested.
    if args.include_experimental:
        exp_names = sorted(set().union(*[set(fd) for _p, fd, _l in samples])
                           - set(wf.PRODUCTION_FEATURES))
        if exp_names:
            exp_w = wf.fit_weights_from_samples(
                samples, method=args.method, features=exp_names,
                scale=args.scale, l2=args.l2,
                min_samples_per_phase=args.min_samples_per_phase,
            )
            print("\nexperimental-feature coefficients (informational):")
            for phase in wf.PHASES:
                print(f"  {phase}: " + ", ".join(
                    f"{f}={exp_w[phase][f]:.2f}" for f in exp_names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
