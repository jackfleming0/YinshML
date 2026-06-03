#!/usr/bin/env python
"""Replay and analyze a transcribed human YINSH game.

Three analyses, all from a chosen perspective (default: the winner):

1. Replay validation — drive every move through our GameState engine.
2. Feature trajectory — score each position with the production heuristic and
   the experimental palette; flag features that never move (dead signals).
3. Agent divergence — at each of the perspective player's decisions, rank their
   candidate ring moves by a fast greedy (depth-1) heuristic eval and report
   where the human's actual move falls. Big, persistent divergence on the
   winning side marks heuristic blind spots.

This is the durable version of the one-off review that produced
docs/game_reviews/bga_862307561_review.md.

Usage:
    python scripts/review_human_game.py                 # full text report
    python scripts/review_human_game.py --plot out.png  # also write eval curve
    python scripts/review_human_game.py --no-diverge     # skip the slow part
"""

import argparse
import sys

from yinsh_ml.game.types import MoveType, GamePhase
from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.heuristics.features import extract_all_features
from yinsh_ml.heuristics.experimental_features import extract_experimental_features
from yinsh_ml.heuristics.feature_diagnostics import feature_liveness_report
from yinsh_ml.data.human_games import bga_862307561 as game


PERSPECTIVE = Player.BLACK  # the winner of this game


def replay_and_trace(heur):
    """Return list of (turn, mover, eval, prod_feats, exp_feats) per ply."""
    trace = []
    for turn_no, mover, state in game.iter_states():
        ev = heur.evaluate_position(state, PERSPECTIVE)
        trace.append((
            turn_no, mover, ev,
            extract_all_features(state, PERSPECTIVE),
            extract_experimental_features(state, PERSPECTIVE),
        ))
    return trace


def print_trajectory(trace):
    keys = list(trace[0][3].keys())
    print("\n== Feature trajectory (production set, perspective = "
          f"{PERSPECTIVE.name}) ==")
    hdr = f"{'trn':>3} {'by':>2} {'eval':>9} " + " ".join(f"{k[:9]:>9}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for turn_no, mover, ev, feats, _exp in trace:
        ev_show = max(-200.0, min(200.0, ev))  # clamp tactical spikes for reading
        row = f"{turn_no:>3} {mover:>2} {ev_show:>9.2f} " + " ".join(
            f"{feats[k]:>9.2f}" for k in keys)
        print(row)


def print_liveness():
    states = [s for _t, _m, s in game.iter_states()]
    print("\n== Feature liveness (distinct values across the game) ==")
    for label, fn in (("production", extract_all_features),
                      ("experimental", extract_experimental_features)):
        report, dead = feature_liveness_report(states, PERSPECTIVE, feature_fn=fn)
        print(f"  [{label}]")
        for name, n in sorted(report.items(), key=lambda kv: kv[1]):
            mark = "  <== DEAD (constant)" if n <= 1 else ""
            print(f"    {name:<32} {n:>3} distinct{mark}")
        if dead:
            print(f"    dead: {dead}")


def print_divergence():
    """Greedy depth-1 ranking of the perspective player's moves vs human."""
    from yinsh_ml.game.game_state import GameState
    from yinsh_ml.game.types import Move

    # Forced-sequence lookahead is the dominant cost and not needed for a
    # depth-1 ranking; disable it so this stays fast.
    heur = YinshHeuristics(enable_forced_sequence_detection=False)

    state = GameState()
    game.play_placements(state)

    print("\n== Agent divergence (greedy depth-1; perspective = "
          f"{PERSPECTIVE.name}) ==")
    print(f"{'trn':>3} {'human':>9} {'agent_best':>11} {'rank':>7} "
          f"{'humanEval':>10} {'bestEval':>10}")
    print("-" * 60)

    agree = total = 0
    turn_no = len(game.PLACEMENTS)
    for tag, ringmv, removal in game.MAIN:
        turn_no += 1
        pl = Player.WHITE if tag == "W" else Player.BLACK
        src, dst = ringmv.split("-")

        if (pl == PERSPECTIVE and state.phase == GamePhase.MAIN_GAME
                and state.current_player == PERSPECTIVE):
            scored = []
            for m in state.get_valid_moves():
                if m.type != MoveType.MOVE_RING:
                    continue
                nxt = state.copy()
                if nxt.make_move(m):
                    scored.append((heur.evaluate_position(nxt, PERSPECTIVE), m))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_val, best_move = scored[0]
            human_rank = next(
                ((i + 1, v) for i, (v, m) in enumerate(scored)
                 if str(m.source) == src and str(m.destination) == dst),
                None,
            )
            total += 1
            if human_rank:
                rank, hval = human_rank
                if rank == 1:
                    agree += 1
                best_str = f"{best_move.source}-{best_move.destination}"
                print(f"{turn_no:>3} {ringmv:>9} {best_str:>11} "
                      f"{rank:>3}/{len(scored):<3} {hval:>10.1f} {best_val:>10.1f}")

        # advance the real game
        game._apply_turn(state, tag, ringmv, removal)

    print("-" * 60)
    if total:
        print(f"Agent agreed with the human's move on {agree}/{total} "
              f"decisions ({100 * agree / total:.0f}%).")


def write_plot(trace, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    turns = [t for t, *_ in trace]
    evals = [max(-200, min(200, e)) for _t, _m, e, *_ in trace]
    score_turns = [(t, m) for t, m, _e, feats, _x in trace_scoring_turns(trace)]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(turns, evals, color="black", lw=2, label=f"eval ({PERSPECTIVE.name} POV, clamped)")
    ax.fill_between(turns, 0, evals, where=[e >= 0 for e in evals], color="green", alpha=0.15)
    ax.fill_between(turns, 0, evals, where=[e < 0 for e in evals], color="red", alpha=0.15)
    for t, who in score_turns:
        ax.axvline(t, color=("blue" if who == "W" else "darkgreen"), ls="--", alpha=0.6)
        ax.text(t, 185, f"{who} pt", rotation=90, va="top", fontsize=8,
                color=("blue" if who == "W" else "darkgreen"))
    ax.set_xlabel("Turn"); ax.set_ylabel(f"Eval ({PERSPECTIVE.name})")
    ax.set_title(f"{game.BGA_TABLE_ID}: heuristic eval vs. actual result ({game.RESULT})")
    ax.legend(loc="lower left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"\nwrote plot -> {path}")


def trace_scoring_turns(trace):
    # A scoring turn is one where the running score increased; re-derive from
    # the fixture's MAIN removal annotations.
    scoring = {len(game.PLACEMENTS) + i + 1 for i, (_, _, rm) in enumerate(game.MAIN) if rm}
    return [row for row in trace if row[0] in scoring]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", metavar="PATH", help="write the eval-curve PNG")
    ap.add_argument("--no-diverge", action="store_true", help="skip agent divergence")
    ap.add_argument("--no-trajectory", action="store_true", help="skip the per-ply table")
    args = ap.parse_args(argv)

    final = game.replay()
    print(f"Replay OK: {game.BGA_TABLE_ID} -> {game.RESULT} "
          f"(final W{final.white_score}-B{final.black_score}, "
          f"winner {final.get_winner().name})")
    assert final.phase == GamePhase.GAME_OVER

    heur = YinshHeuristics()
    trace = replay_and_trace(heur)

    if not args.no_trajectory:
        print_trajectory(trace)
    print_liveness()
    if not args.no_diverge:
        print_divergence()
    if args.plot:
        write_plot(trace, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
