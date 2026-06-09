#!/usr/bin/env python3
"""Play two *external* engines (sharkdp, yngine) against each other, with our
``GameState`` as referee. No neural net involved, so this is fast.

Two uses:
  1. Calibration — is sharkdp's negamax+defensive-heuristic stronger than
     yngine's pure MCTS? (alternating colors, win/draw tally + Wilson CI)
  2. Corpus skeleton — the same loop is the cross-engine *position generator*
     for diverse opening seeds (A-vs-A, A-vs-B, B-vs-B).

Both drivers speak the same wire protocol, so the referee just shuttles wire
strings between them and validates every move against our rules engine.

Usage::

    python scripts/eval_engine_vs_engine.py \\
        --engine-a sharkdp --a-depth 6 \\
        --engine-b yngine  --b-sims 1000 \\
        --games 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.game.constants import Player  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.sharkdp import Sharkdp  # noqa: E402
from yinsh_ml.yngine import Yngine  # noqa: E402


def wilson_ci_95(p: float, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def _start(kind: str, depth: int, sims: int):
    if kind == "sharkdp":
        return Sharkdp.start(depth=depth)
    if kind == "yngine":
        return Yngine.start(threads=1)
    raise ValueError(f"unknown engine {kind!r}")


def _get_move(eng, kind: str, player: Player, depth: int, sims: int):
    if kind == "sharkdp":
        return eng.get_move(player=player, depth=depth)
    return eng.get_move(player=player, sims=sims)


def play_game(a_kind, a_depth, a_sims, b_kind, b_depth, b_sims,
              a_is_white, max_moves) -> dict:
    """A plays White iff a_is_white. Returns a per-game record."""
    state = GameState()
    a = _start(a_kind, a_depth, a_sims)
    b = _start(b_kind, b_depth, b_sims)
    a.new_game()
    b.new_game()
    a_color = Player.WHITE if a_is_white else Player.BLACK
    moves = 0
    t0 = time.time()
    err = None
    try:
        while not state.is_terminal() and moves < max_moves:
            if not state.get_valid_moves():
                break
            if state.current_player == a_color:
                mover, mk, md, ms = a, a_kind, a_depth, a_sims
            else:
                mover, mk, md, ms = b, b_kind, b_depth, b_sims
            mv, wire = _get_move(mover, mk, state.current_player, md, ms)
            if not state.make_move(mv):
                err = f"{mk} illegal move per referee: {wire}"
                break
            a.apply_wire(wire)
            b.apply_wire(wire)
            moves += 1
        winner = state.get_winner()
    finally:
        a.stop()
        b.stop()

    if winner == Player.WHITE:
        win_kind = a_kind if a_is_white else b_kind
    elif winner == Player.BLACK:
        win_kind = b_kind if a_is_white else a_kind
    else:
        win_kind = "draw"
    return {
        "a_color": "white" if a_is_white else "black",
        "winner_kind": win_kind,
        "a_won": (win_kind == a_kind and win_kind != "draw"),
        "moves": moves,
        "seconds": time.time() - t0,
        "error": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-a", choices=["sharkdp", "yngine"], default="sharkdp")
    ap.add_argument("--engine-b", choices=["sharkdp", "yngine"], default="yngine")
    ap.add_argument("--a-depth", type=int, default=6)
    ap.add_argument("--a-sims", type=int, default=1000)
    ap.add_argument("--b-depth", type=int, default=6)
    ap.add_argument("--b-sims", type=int, default=1000)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-moves", type=int, default=400)
    args = ap.parse_args()

    a_label = (f"{args.engine_a}(d{args.a_depth})" if args.engine_a == "sharkdp"
               else f"{args.engine_a}(s{args.a_sims})")
    b_label = (f"{args.engine_b}(d{args.b_depth})" if args.engine_b == "sharkdp"
               else f"{args.engine_b}(s{args.b_sims})")
    print(f"{a_label}  vs  {b_label}   —   {args.games} games, alternating colors")

    a_wins = b_wins = draws = 0
    a_white_wins = a_black_wins = 0
    t0 = time.time()
    for g in range(args.games):
        a_is_white = (g % 2 == 0)
        rec = play_game(
            args.engine_a, args.a_depth, args.a_sims,
            args.engine_b, args.b_depth, args.b_sims,
            a_is_white, args.max_moves,
        )
        if rec["error"]:
            print(f"  game {g+1}: ERROR {rec['error']}")
            continue
        if rec["winner_kind"] == "draw":
            draws += 1
        elif rec["a_won"]:
            a_wins += 1
            a_white_wins += int(a_is_white)
            a_black_wins += int(not a_is_white)
        else:
            b_wins += 1
        side = "W" if a_is_white else "B"
        print(f"  game {g+1:>3}/{args.games}  A={side}  winner={rec['winner_kind']:>8}"
              f"  moves={rec['moves']:>3}  {rec['seconds']:.1f}s  "
              f"score(A-B-D)={a_wins}-{b_wins}-{draws}")

    decisive = a_wins + b_wins
    wr = a_wins / decisive if decisive else 0.0
    lo, hi = wilson_ci_95(wr, decisive)
    print("\n" + "=" * 70)
    print(f"{a_label} vs {b_label}  —  {time.time()-t0:.0f}s")
    print("=" * 70)
    print(f"  A ({a_label}) wins: {a_wins}  (W:{a_white_wins} B:{a_black_wins})")
    print(f"  B ({b_label}) wins: {b_wins}    draws: {draws}")
    print(f"  A win-rate (decisive): {wr:.3f}   CI95=[{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
