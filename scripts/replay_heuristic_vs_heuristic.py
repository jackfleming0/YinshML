#!/usr/bin/env python3
"""Replay a single game of HeuristicAgent vs HeuristicAgent with verbose ply output.

Diagnostic for the "offense-only equilibrium" hypothesis: if both players use
the heuristic evaluator (which weights completed runs / potential runs / chains
heavily and has no explicit defensive feature), do they ignore opponent threats?

Unlike replay_h2h_game.py, this needs no checkpoint — purely heuristic +
negamax search. Works locally without GPU.

Usage:
    python scripts/replay_heuristic_vs_heuristic.py
    python scripts/replay_heuristic_vs_heuristic.py --depth 2 --seed 7
"""

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Player
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig

logger = logging.getLogger("replay_h_vs_h")


def play_one(white_agent, black_agent, max_moves=200, board_every=0):
    game = GameState()
    move_count = 0

    while not game.is_terminal() and move_count < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            print(f"  [ply {move_count}] {game.current_player.name}: NO VALID MOVES")
            break

        is_white = (game.current_player == Player.WHITE)
        agent = white_agent if is_white else black_agent

        ws_before, bs_before = game.white_score, game.black_score
        move = agent.select_move(game)
        if move is None:
            print(f"  [ply {move_count}] {game.current_player.name}: agent returned None")
            break

        # Heuristic eval of the position BEFORE the move (from the player about to move)
        try:
            v = agent._evaluator.evaluate_position(game, game.current_player)
        except Exception:
            v = float('nan')

        phase_str = game.phase.name[:4]
        print(
            f"  [ply {move_count:3d} {phase_str}] {game.current_player.name:5s} "
            f"h_eval={v:+.2f} | {move}"
        )

        if not game.make_move(move):
            print(f"  [ply {move_count}] make_move REJECTED {move}")
            break

        if game.white_score != ws_before or game.black_score != bs_before:
            print(f"           ↳ SCORE: White={game.white_score} Black={game.black_score}")

        move_count += 1
        if board_every > 0 and move_count % board_every == 0:
            print(f"  --- board after ply {move_count} ---")
            print("  " + str(game.board).replace("\n", "\n  "))

    winner = game.get_winner()
    winner_str = winner.name if winner else "DRAW"
    print(f"  RESULT: winner={winner_str} | plies={move_count} | "
          f"final_score={game.white_score}-{game.black_score}")
    print(f"  --- final board ---")
    print("  " + str(game.board).replace("\n", "\n  "))
    return winner_str, move_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2,
                        help="Negamax search depth. Lower = faster but less defense-aware.")
    parser.add_argument("--time-limit", type=float, default=2.0,
                        help="Per-move time limit in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--board-every", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg_w = HeuristicAgentConfig(
        max_depth=args.depth,
        time_limit_seconds=args.time_limit,
        random_seed=args.seed,
    )
    cfg_b = HeuristicAgentConfig(
        max_depth=args.depth,
        time_limit_seconds=args.time_limit,
        random_seed=args.seed + 1,
    )

    results = []
    for g in range(args.num_games):
        seed = args.seed + g
        random.seed(seed)
        # Fresh agent per game so transposition table doesn't carry state
        white_agent = HeuristicAgent(config=HeuristicAgentConfig(
            max_depth=args.depth, time_limit_seconds=args.time_limit, random_seed=seed,
        ))
        black_agent = HeuristicAgent(config=HeuristicAgentConfig(
            max_depth=args.depth, time_limit_seconds=args.time_limit, random_seed=seed + 1000,
        ))
        print(f"\n{'=' * 70}")
        print(f"GAME {g+1}/{args.num_games}  seed={seed}  depth={args.depth}")
        print(f"{'=' * 70}")
        winner, plies = play_one(
            white_agent, black_agent,
            max_moves=args.max_moves,
            board_every=args.board_every,
        )
        results.append((winner, plies))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    w = sum(1 for r, _ in results if r == "WHITE")
    b = sum(1 for r, _ in results if r == "BLACK")
    d = len(results) - w - b
    avg_plies = sum(p for _, p in results) / len(results) if results else 0
    print(f"  WHITE wins: {w} | BLACK wins: {b} | DRAW/inconclusive: {d}")
    print(f"  Avg plies: {avg_plies:.1f}")


if __name__ == "__main__":
    main()
