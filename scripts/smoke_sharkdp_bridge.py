#!/usr/bin/env python3
"""End-to-end validation of the sharkdp/yinsh bridge.

Two checks, in order of how much they prove:

1. ``referee`` — two sharkdp engines play each other while *our* ``GameState``
   referees every move. This is the rigorous parity test: every sharkdp move
   must round-trip through the wire codec and be accepted by our rules engine
   (and vice-versa), and the game must reach a legal terminal. If the
   coordinate bijection, direction mapping, or turn bridging were wrong, our
   engine would reject a move and this aborts loudly.

2. ``vs-heuristic`` — sharkdp vs our :class:`HeuristicAgent`, alternating
   colors. The first real "ours vs theirs" number.

Build the driver first::

    cd third_party/sharkdp_yinsh && cargo build --release -p yinsh_driver

Usage::

    python scripts/smoke_sharkdp_bridge.py --mode referee --games 2 --depth 4
    python scripts/smoke_sharkdp_bridge.py --mode vs-heuristic --games 6 \\
        --depth 6 --our-depth 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.game.constants import Player  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.sharkdp import Sharkdp  # noqa: E402
from yinsh_ml.agents.heuristic_agent import (  # noqa: E402
    HeuristicAgent,
    HeuristicAgentConfig,
)


def _winner_str(state: GameState) -> str:
    w = state.get_winner()
    if w == Player.WHITE:
        return "white"
    if w == Player.BLACK:
        return "black"
    return "draw"


def play_referee_game(depth: int, max_moves: int, verbose: bool) -> dict:
    """Two sharkdp engines; our GameState referees and validates every move."""
    state = GameState()
    white = Sharkdp.start(depth=depth)
    black = Sharkdp.start(depth=depth)
    white.new_game()
    black.new_game()
    moves = 0
    t0 = time.time()
    err = None
    try:
        while not state.is_terminal() and moves < max_moves:
            if not state.get_valid_moves():
                err = "no valid moves (our engine) before terminal"
                break
            mover = white if state.current_player == Player.WHITE else black
            other = black if state.current_player == Player.WHITE else white
            mv, wire = mover.get_move(player=state.current_player)
            if not state.make_move(mv):
                err = f"our engine rejected sharkdp move: {wire} ({mv})"
                break
            # Keep both driver boards in sync with the refereed game.
            mover.apply_wire(wire)
            other.apply_wire(wire)
            moves += 1
            if verbose and moves % 20 == 0:
                print(f"    … {moves} moves, score "
                      f"{state.white_score}-{state.black_score}")
    finally:
        white.stop()
        black.stop()
    return {
        "winner": _winner_str(state),
        "moves": moves,
        "seconds": time.time() - t0,
        "terminal": state.is_terminal(),
        "error": err,
    }


def play_vs_heuristic(
    depth: int, our_depth: int, our_time: float, shark_is_white: bool,
    max_moves: int, seed: int,
) -> dict:
    """sharkdp vs our HeuristicAgent. Returns a per-game record."""
    agent = HeuristicAgent(
        HeuristicAgentConfig(
            max_depth=our_depth,
            time_limit_seconds=our_time,
            random_seed=seed,
        )
    )
    shark_color = Player.WHITE if shark_is_white else Player.BLACK

    state = GameState()
    eng = Sharkdp.start(depth=depth)
    eng.new_game()
    moves = 0
    t0 = time.time()
    err = None
    try:
        while not state.is_terminal() and moves < max_moves:
            if not state.get_valid_moves():
                break
            if state.current_player == shark_color:
                mv, wire = eng.get_move(player=state.current_player)
                if not state.make_move(mv):
                    err = f"our engine rejected sharkdp move: {wire}"
                    break
                eng.apply_wire(wire)
            else:
                mv = agent.get_move(state)
                if mv is None or not state.make_move(mv):
                    err = "our HeuristicAgent produced an illegal/empty move"
                    break
                eng.apply(mv)
            moves += 1
    finally:
        eng.stop()

    winner = state.get_winner()
    return {
        "shark_color": "white" if shark_is_white else "black",
        "winner": _winner_str(state),
        "shark_won": winner == shark_color,
        "moves": moves,
        "seconds": time.time() - t0,
        "error": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["referee", "vs-heuristic"],
                    default="referee")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6,
                    help="sharkdp negamax depth.")
    ap.add_argument("--our-depth", type=int, default=3,
                    help="HeuristicAgent max_depth (vs-heuristic mode).")
    ap.add_argument("--our-time", type=float, default=1.0,
                    help="HeuristicAgent per-move time budget (vs-heuristic).")
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mode == "referee":
        print(f"Referee self-play: {args.games} game(s), sharkdp depth "
              f"{args.depth}, our GameState validating every move.")
        ok = True
        for g in range(args.games):
            rec = play_referee_game(args.depth, args.max_moves, args.verbose)
            status = "OK" if (rec["error"] is None and rec["terminal"]) else "FAIL"
            if status != "OK":
                ok = False
            print(f"  game {g+1}: {status}  winner={rec['winner']}  "
                  f"moves={rec['moves']}  {rec['seconds']:.1f}s"
                  + (f"  ERROR: {rec['error']}" if rec["error"] else "")
                  + ("" if rec["terminal"] else "  [hit max_moves, not terminal]"))
        print("\nPARITY:", "PASS — wire codec & rules agree both directions"
              if ok else "FAIL — see errors above")
        sys.exit(0 if ok else 1)

    # vs-heuristic
    print(f"sharkdp (depth {args.depth}) vs HeuristicAgent "
          f"(max_depth {args.our_depth}, {args.our_time}s): {args.games} games")
    shark_wins = our_wins = draws = 0
    for g in range(args.games):
        shark_is_white = (g % 2 == 0)
        rec = play_vs_heuristic(
            args.depth, args.our_depth, args.our_time, shark_is_white,
            args.max_moves, args.seed + g,
        )
        if rec["error"]:
            print(f"  game {g+1}: ERROR {rec['error']}")
            continue
        if rec["shark_won"]:
            shark_wins += 1
            tag = "sharkdp"
        elif rec["winner"] == "draw":
            draws += 1
            tag = "draw"
        else:
            our_wins += 1
            tag = "ours"
        side = "W" if shark_is_white else "B"
        print(f"  game {g+1}: shark={side}  winner={rec['winner']} ({tag})  "
              f"moves={rec['moves']}  {rec['seconds']:.1f}s")
    print(f"\nResult — sharkdp {shark_wins}  ours {our_wins}  draws {draws}")


if __name__ == "__main__":
    main()
