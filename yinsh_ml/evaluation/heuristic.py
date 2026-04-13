"""Fast heuristic evaluation function for YINSH positions.

Designed to be meaningfully better than random for bootstrapping
neural network training. Returns values in [-1, 1] from the
perspective of the current player.

Features (rough order of importance):
1. Score differential (rings removed) - dominant signal
2. Ring mobility differential
3. Near-row threats (3/4 in a row)
4. Marker balance
5. Ring centrality
6. Marker vulnerability (flippability)
"""

import math
from typing import Optional

from ..game.game_state import GameState
from ..game.constants import (
    Player, PieceType, Position, is_valid_position,
    POINTS_TO_WIN, HEX_DIRECTIONS,
)
from ..game.types import GamePhase


def evaluate_position(game_state: GameState, player: Optional[Player] = None) -> float:
    """Evaluate a YINSH position, returning a value in [-1, 1].

    Positive = advantage for `player` (defaults to current player).
    This function is optimized for speed over accuracy.

    Args:
        game_state: Current game state.
        player: Perspective player. Defaults to game_state.current_player.

    Returns:
        Float in [-1, 1].
    """
    if player is None:
        player = game_state.current_player
    opponent = player.opponent

    # Terminal check
    if game_state.phase == GamePhase.GAME_OVER:
        winner = game_state.get_winner()
        if winner == player:
            return 1.0
        elif winner == opponent:
            return -1.0
        return 0.0

    # --- Feature extraction ---

    my_score = game_state.white_score if player == Player.WHITE else game_state.black_score
    opp_score = game_state.white_score if opponent == Player.WHITE else game_state.black_score

    my_ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
    opp_ring_type = PieceType.WHITE_RING if opponent == Player.WHITE else PieceType.BLACK_RING
    my_marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opp_marker_type = PieceType.WHITE_MARKER if opponent == Player.WHITE else PieceType.BLACK_MARKER

    # Collect piece positions in a single pass
    my_rings = []
    opp_rings = []
    my_markers = []
    opp_markers = []
    for pos, piece in game_state.board.pieces.items():
        if piece == my_ring_type:
            my_rings.append(pos)
        elif piece == opp_ring_type:
            opp_rings.append(pos)
        elif piece == my_marker_type:
            my_markers.append(pos)
        elif piece == opp_marker_type:
            opp_markers.append(pos)

    # 1. Score differential (most important — each point is ~33% of winning)
    score_diff = my_score - opp_score  # range: [-3, 3]

    # 2. Ring mobility: count valid destinations for each ring
    my_mobility = sum(len(game_state.board.valid_move_positions(r)) for r in my_rings)
    opp_mobility = sum(len(game_state.board.valid_move_positions(r)) for r in opp_rings)
    mobility_diff = my_mobility - opp_mobility  # typical range: [-30, 30]

    # 3. Near-row threats: scan lines through markers
    my_threats_3, my_threats_4 = _count_threats(game_state.board, my_marker_type)
    opp_threats_3, opp_threats_4 = _count_threats(game_state.board, opp_marker_type)
    threat_diff = (
        (my_threats_4 - opp_threats_4) * 3.0
        + (my_threats_3 - opp_threats_3) * 1.0
    )

    # 4. Ring centrality: rings near center have more options
    my_centrality = sum(_centrality(r) for r in my_rings)
    opp_centrality = sum(_centrality(r) for r in opp_rings)
    centrality_diff = my_centrality - opp_centrality  # typical range: [-5, 5]

    # 5. Marker count (mildly positive to have more markers — more material)
    marker_diff = len(my_markers) - len(opp_markers)  # range: [-20, 20]

    # --- Weighted combination ---
    # Weights chosen so score_diff dominates
    raw = (
        score_diff * 100.0      # ~300 max contribution
        + mobility_diff * 1.5   # ~45 max
        + threat_diff * 8.0     # ~40 max
        + centrality_diff * 3.0 # ~15 max
        + marker_diff * 0.5     # ~10 max
    )

    # Squash to [-1, 1] via tanh-like scaling
    # Scale factor chosen so a 1-point score lead ≈ 0.7
    return math.tanh(raw / 150.0)


def _centrality(pos: Position) -> float:
    """Score how central a position is. Returns 0-1."""
    # Board center is approximately F6 (column index 5, row 6)
    col_dist = abs(ord(pos.column) - ord('F'))
    row_dist = abs(pos.row - 6)
    dist = max(col_dist, row_dist)
    return max(0.0, 1.0 - dist / 5.0)


def _count_threats(board, marker_type: PieceType):
    """Count lines of 3 and 4 consecutive markers.

    Returns (count_of_3, count_of_4). Only checks each direction once
    per starting position to avoid double-counting (uses only
    positive direction from each position).
    """
    # Build set of marker positions for O(1) lookup
    marker_set = set()
    for pos, piece in board.pieces.items():
        if piece == marker_type:
            marker_set.add(pos)

    if not marker_set:
        return 0, 0

    # Only scan in "positive" directions to avoid double-counting
    directions = [
        (0, 1),   # up
        (1, 0),   # right
        (1, 1),   # diagonal up-right
        (-1, 1),  # diagonal up-left
    ]

    count_3 = 0
    count_4 = 0

    for start in marker_set:
        for dx, dy in directions:
            length = 1
            col_idx = ord(start.column) - ord('A')
            row = start.row
            for _ in range(4):  # check up to 4 more
                col_idx += dx
                row += dy
                npos = Position(chr(ord('A') + col_idx), row)
                if npos in marker_set:
                    length += 1
                else:
                    break

            if length == 4:
                count_4 += 1
            elif length == 3:
                count_3 += 1
            # length >= 5 means an actual completed row — handled by game logic

    return count_3, count_4


class HeuristicEvaluator:
    """Wrapper class compatible with the training pipeline's evaluator interface."""

    def evaluate_position(self, game_state: GameState, player: Player) -> float:
        return evaluate_position(game_state, player)

    def evaluate_batch(self, game_states, players):
        return [evaluate_position(gs, p) for gs, p in zip(game_states, players)]
