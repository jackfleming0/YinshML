#!/usr/bin/env python
"""Offline A/B validation of two heuristic weight sets (B).

Plays HeuristicAgent(weights_A) vs HeuristicAgent(weights_B) head-to-head with
colors alternated, and reports A's win-rate and an approximate Elo delta. This
is the gate before propagating re-fit weights into (expensive) training: if the
new weights don't beat the old ones agent-vs-agent, they aren't worth a run.

Pure heuristic + engine — no torch — so it runs anywhere. Depth/Game count are
small by default; raise them for a real verdict.

Examples:
  python scripts/experiments/validate_weights.py \
      --weights-a configs/heuristic_weights/baseline.json \
      --weights-b configs/heuristic_weights/refit_logreg.json \
      --games 40 --depth 2
"""

import argparse
import math
import sys
from typing import Optional

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase
from yinsh_ml.game.constants import Player
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.heuristics.evaluator import YinshHeuristics


def _make_agent(weights_path: Optional[str], depth: int, seed: int) -> HeuristicAgent:
    evaluator = YinshHeuristics(
        weight_config_file=weights_path,
        enable_forced_sequence_detection=False,
    )
    cfg = HeuristicAgentConfig(
        max_depth=depth, time_limit_seconds=0.0,
        use_iterative_deepening=False, random_seed=seed,
    )
    return HeuristicAgent(config=cfg, evaluator=evaluator)


def play_game(agent_white: HeuristicAgent, agent_black: HeuristicAgent,
              max_moves: int = 400) -> Optional[Player]:
    """Play one full game; return the winner (or None for draw/cap)."""
    state = GameState()
    agents = {Player.WHITE: agent_white, Player.BLACK: agent_black}
    for _ in range(max_moves):
        if state.phase == GamePhase.GAME_OVER:
            break
        if state.is_stalemate():
            return state.current_player.opponent
        move = agents[state.current_player].select_move(state)
        if move is None or not state.make_move(move):
            # No legal/usable move -> current player loses.
            return state.current_player.opponent
        if state.white_score >= 3:
            return Player.WHITE
        if state.black_score >= 3:
            return Player.BLACK
    return state.get_winner()


def win_rate_to_elo(win_rate: float) -> float:
    win_rate = min(max(win_rate, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


def run_ab(weights_a: Optional[str], weights_b: Optional[str],
           games: int, depth: int, seed: int = 0) -> dict:
    a_wins = b_wins = draws = 0
    for g in range(games):
        agent_a = _make_agent(weights_a, depth, seed + g)
        agent_b = _make_agent(weights_b, depth, seed + 1000 + g)
        # alternate colors so neither side keeps the first-move edge
        if g % 2 == 0:
            winner = play_game(agent_a, agent_b)
            a_is = Player.WHITE
        else:
            winner = play_game(agent_b, agent_a)
            a_is = Player.BLACK
        if winner is None:
            draws += 1
        elif winner == a_is:
            a_wins += 1
        else:
            b_wins += 1
    decided = a_wins + b_wins
    win_rate = (a_wins + 0.5 * draws) / games if games else 0.0
    return {
        "games": games, "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_win_rate": win_rate,
        "elo_delta_a_over_b": win_rate_to_elo(win_rate) if decided else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-a", default=None,
                    help="weights JSON for A (omit => default weights)")
    ap.add_argument("--weights-b", default=None,
                    help="weights JSON for B (omit => default weights)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    res = run_ab(args.weights_a, args.weights_b, args.games, args.depth, args.seed)
    print(f"A ({args.weights_a or 'default'}) vs B ({args.weights_b or 'default'})")
    print(f"  games={res['games']} A={res['a_wins']} B={res['b_wins']} draws={res['draws']}")
    print(f"  A win-rate: {res['a_win_rate']:.3f}")
    print(f"  Elo(A) - Elo(B) ~= {res['elo_delta_a_over_b']:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
