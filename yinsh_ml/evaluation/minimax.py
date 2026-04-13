"""Minimax player with alpha-beta pruning for YINSH.

Provides a baseline opponent that uses the heuristic evaluation function
with shallow search (2-3 ply). Good enough to beat random play and serve
as an evaluation opponent for trained models.
"""

import math
import random
from typing import Optional, Tuple, List

from ..game.game_state import GameState
from ..game.constants import Player
from ..game.types import Move, GamePhase
from .heuristic import evaluate_position


class MinimaxPlayer:
    """Minimax-based YINSH player with alpha-beta pruning.

    Args:
        depth: Search depth in plies. 2-3 recommended (deeper is slow).
        player: Which color this player controls.
        randomize: If True, randomize among equally-scored moves.
    """

    def __init__(self, depth: int = 2, player: Player = Player.WHITE,
                 randomize: bool = True):
        self.depth = depth
        self.player = player
        self.randomize = randomize
        self.nodes_searched = 0

    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select the best move for the current position.

        Args:
            game_state: Current game state. Must be this player's turn
                       (or any player's turn — evaluates from current player's POV).

        Returns:
            Best move found, or None if no moves available.
        """
        self.nodes_searched = 0
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]

        # For ring placement, use a simpler heuristic (prefer center)
        if game_state.phase == GamePhase.RING_PLACEMENT:
            return self._select_ring_placement(valid_moves)

        best_score = -math.inf
        best_moves = []
        alpha = -math.inf
        beta = math.inf
        maximizing_player = game_state.current_player

        for move in valid_moves:
            child = game_state.copy()
            if not child.make_move(move):
                continue

            score = self._alphabeta(child, self.depth - 1, alpha, beta,
                                    False, maximizing_player)

            if score > best_score:
                best_score = score
                best_moves = [move]
                alpha = max(alpha, score)
            elif score == best_score:
                best_moves.append(move)

        if not best_moves:
            return valid_moves[0]

        if self.randomize:
            return random.choice(best_moves)
        return best_moves[0]

    def _alphabeta(self, state: GameState, depth: int,
                   alpha: float, beta: float,
                   maximizing: bool, root_player: Player) -> float:
        """Alpha-beta minimax search.

        Args:
            state: Current game state.
            depth: Remaining search depth.
            alpha: Alpha bound.
            beta: Beta bound.
            maximizing: True if maximizing player's turn.
            root_player: The player we're evaluating for.

        Returns:
            Evaluation score from root_player's perspective.
        """
        self.nodes_searched += 1

        # Terminal or depth limit
        if depth == 0 or state.is_terminal():
            return evaluate_position(state, root_player)

        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return evaluate_position(state, root_player)

        # Determine if current player is the maximizing player
        # Note: in YINSH, the same player can have multiple consecutive turns
        # (row completion + ring removal), so we check explicitly
        is_max = (state.current_player == root_player)

        if is_max:
            value = -math.inf
            for move in valid_moves:
                child = state.copy()
                if not child.make_move(move):
                    continue
                child_is_max = (child.current_player == root_player)
                score = self._alphabeta(child, depth - 1, alpha, beta,
                                        child_is_max, root_player)
                value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for move in valid_moves:
                child = state.copy()
                if not child.make_move(move):
                    continue
                child_is_max = (child.current_player == root_player)
                score = self._alphabeta(child, depth - 1, alpha, beta,
                                        child_is_max, root_player)
                value = min(value, score)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _select_ring_placement(self, moves: List[Move]) -> Move:
        """Heuristic ring placement: prefer central positions."""
        from .heuristic import _centrality

        scored = [(m, _centrality(m.source)) for m in moves]
        max_score = max(s for _, s in scored)
        best = [m for m, s in scored if s == max_score]
        return random.choice(best) if self.randomize else best[0]
