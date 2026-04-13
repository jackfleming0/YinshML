"""Enhanced state encoding for YINSH neural network input.

This module provides EnhancedStateEncoder, which expands the state representation
from 6 to 15 channels to provide the neural network with more strategic information.

New channels include:
- Row threats (positions that complete a row)
- Partial rows (3-4 marker runs)
- Ring mobility (normalized move counts)
- Center distance (proximity heatmap)
- Ring influence (reachable cells)
- Turn number and score differential

Expected impact: +100 ELO from improved feature representation.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Set
import logging
from dataclasses import dataclass

from ..game.constants import (
    Position,
    Player,
    PieceType,
    is_valid_position,
    VALID_POSITIONS,
    MARKERS_FOR_ROW
)
from ..game.game_state import GameState, GamePhase
from .encoding import StateEncoder  # Import base class for inheritance

logger = logging.getLogger(__name__)


@dataclass
class EncodingStats:
    """Statistics for debugging and monitoring encoding behavior."""
    row_threats_found: int = 0
    partial_rows_found: int = 0
    ring_mobility_sum: float = 0.0
    ring_influence_coverage: float = 0.0


class EnhancedStateEncoder(StateEncoder):
    """
    Enhanced state encoder expanding from 6 to 15 channels.

    Inherits from StateEncoder to get move encoding/decoding methods.
    Overrides encode_state to provide enhanced state representation.

    Channel Layout:
        0: Current player's rings (binary)
        1: Opponent's rings (binary)
        2: Current player's markers (binary)
        3: Opponent's markers (binary)
        4: Current player's row threats (cells completing a row)
        5: Opponent's row threats
        6: Current player's partial rows (3-4 marker runs)
        7: Opponent's partial rows
        8: Ring mobility (normalized legal moves per cell)
        9: Center distance (static proximity heatmap)
        10: Ring influence (cells reachable by current player's rings)
        11: Valid move destinations
        12: Game phase (normalized scalar)
        13: Turn number (normalized 0-1)
        14: Score differential (normalized)

    All channels are encoded from the perspective of the current player
    (side-to-move normalization) for consistency.
    """

    NUM_CHANNELS = 15
    BOARD_SIZE = 11

    # Channel indices for clarity
    CH_CURRENT_RINGS = 0
    CH_OPPONENT_RINGS = 1
    CH_CURRENT_MARKERS = 2
    CH_OPPONENT_MARKERS = 3
    CH_CURRENT_ROW_THREATS = 4
    CH_OPPONENT_ROW_THREATS = 5
    CH_CURRENT_PARTIAL_ROWS = 6
    CH_OPPONENT_PARTIAL_ROWS = 7
    CH_RING_MOBILITY = 8
    CH_CENTER_DISTANCE = 9
    CH_RING_INFLUENCE = 10
    CH_VALID_MOVES = 11
    CH_GAME_PHASE = 12
    CH_TURN_NUMBER = 13
    CH_SCORE_DIFF = 14

    def __init__(self, enable_stats: bool = False):
        """Initialize the enhanced encoder.

        Args:
            enable_stats: If True, collect encoding statistics for debugging.
        """
        # Initialize parent class (sets up position_to_index, move encoding, etc.)
        super().__init__()

        self.enable_stats = enable_stats
        self._last_stats: Optional[EncodingStats] = None

        # Precompute center distance heatmap (static, never changes)
        self._center_distance_map = self._precompute_center_distance()

        # Precompute valid positions for efficient iteration
        self._valid_positions = self._get_all_valid_positions()

        logger.info(f"EnhancedStateEncoder initialized: {self.NUM_CHANNELS} channels, "
                   f"{self.num_positions} valid positions")

    def _precompute_center_distance(self) -> np.ndarray:
        """Precompute the center distance heatmap (static, computed once).

        Returns:
            (11, 11) array with values in [0, 1], where 1 = center, 0 = far edge.
        """
        center_col = 5  # 'F' is index 5
        center_row = 5  # Row 6 is index 5 (0-indexed)

        distance_map = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        max_distance = 5.0  # Maximum distance from center on YINSH board

        for col_idx in range(self.BOARD_SIZE):
            for row_idx in range(self.BOARD_SIZE):
                col = chr(ord('A') + col_idx)
                row = row_idx + 1
                pos = Position(col, row)

                if is_valid_position(pos):
                    # Chebyshev distance (max of horizontal and vertical distance)
                    col_dist = abs(col_idx - center_col)
                    row_dist = abs(row_idx - center_row)
                    distance = max(col_dist, row_dist)

                    # Invert and normalize: 1 = center, 0 = far edge
                    distance_map[row_idx, col_idx] = max(0.0, (max_distance - distance) / max_distance)

        return distance_map

    def _get_all_valid_positions(self) -> List[Position]:
        """Get list of all valid board positions."""
        positions = []
        for col, valid_rows in VALID_POSITIONS.items():
            for row in valid_rows:
                positions.append(Position(col, row))
        return positions

    def encode_state(self, game_state: GameState) -> np.ndarray:
        """Encode the game state into a 15-channel tensor.

        Normalizes by side-to-move so channels 0/2 always represent
        the current player's pieces and 1/3 the opponent's.

        Args:
            game_state: The current game state to encode.

        Returns:
            numpy array of shape (15, 11, 11) with dtype float32.
        """
        state = np.zeros((self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE),
                        dtype=np.float32)

        if self.enable_stats:
            self._last_stats = EncodingStats()

        try:
            is_white = (game_state.current_player == Player.WHITE)

            # --- Channels 0-3: Piece positions (side-normalized) ---
            self._encode_pieces(state, game_state, is_white)

            # --- Channels 4-5: Row threats ---
            self._encode_row_threats(state, game_state, is_white)

            # --- Channels 6-7: Partial rows (3-4 marker runs) ---
            self._encode_partial_rows(state, game_state, is_white)

            # --- Channel 8: Ring mobility ---
            self._encode_ring_mobility(state, game_state)

            # --- Channel 9: Center distance (static, precomputed) ---
            state[self.CH_CENTER_DISTANCE] = self._center_distance_map

            # --- Channel 10: Ring influence ---
            self._encode_ring_influence(state, game_state, is_white)

            # --- Channel 11: Valid move destinations ---
            self._encode_valid_moves(state, game_state)

            # --- Channel 12: Game phase (normalized scalar) ---
            phase_value = float(game_state.phase.value) / float(len(GamePhase) - 1)
            state[self.CH_GAME_PHASE] = phase_value

            # --- Channel 13: Turn number (normalized 0-1, capped at 100 moves) ---
            turn_number = game_state.move_count if hasattr(game_state, 'move_count') else 0
            state[self.CH_TURN_NUMBER] = min(turn_number / 100.0, 1.0)

            # --- Channel 14: Score differential (from current player's perspective) ---
            self._encode_score_differential(state, game_state, is_white)

        except Exception as e:
            logger.error(f"Error encoding state: {e}")
            raise

        return state

    def _encode_pieces(self, state: np.ndarray, game_state: GameState, is_white: bool):
        """Encode piece positions (channels 0-3)."""
        # Get piece positions
        white_rings = game_state.board.get_pieces_positions(PieceType.WHITE_RING)
        black_rings = game_state.board.get_pieces_positions(PieceType.BLACK_RING)
        white_markers = game_state.board.get_pieces_positions(PieceType.WHITE_MARKER)
        black_markers = game_state.board.get_pieces_positions(PieceType.BLACK_MARKER)

        # Side-normalized assignment
        current_rings = white_rings if is_white else black_rings
        opponent_rings = black_rings if is_white else white_rings
        current_markers = white_markers if is_white else black_markers
        opponent_markers = black_markers if is_white else white_markers

        # Fill channels
        for pos in current_rings:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_CURRENT_RINGS, row_idx, col_idx] = 1.0

        for pos in opponent_rings:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_OPPONENT_RINGS, row_idx, col_idx] = 1.0

        for pos in current_markers:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_CURRENT_MARKERS, row_idx, col_idx] = 1.0

        for pos in opponent_markers:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_OPPONENT_MARKERS, row_idx, col_idx] = 1.0

    def _encode_row_threats(self, state: np.ndarray, game_state: GameState, is_white: bool):
        """Encode row threat positions (channels 4-5).

        A row threat is a cell that, if filled with a marker, would complete
        a row of 5 (or extend to 5).
        """
        current_marker = PieceType.WHITE_MARKER if is_white else PieceType.BLACK_MARKER
        opponent_marker = PieceType.BLACK_MARKER if is_white else PieceType.WHITE_MARKER

        current_threats = self._find_row_threats(game_state.board, current_marker)
        opponent_threats = self._find_row_threats(game_state.board, opponent_marker)

        if self.enable_stats:
            self._last_stats.row_threats_found = len(current_threats) + len(opponent_threats)

        for pos in current_threats:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_CURRENT_ROW_THREATS, row_idx, col_idx] = 1.0

        for pos in opponent_threats:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_OPPONENT_ROW_THREATS, row_idx, col_idx] = 1.0

    def _find_row_threats(self, board, marker_type: PieceType) -> Set[Position]:
        """Find all positions that would complete a row of 5 if filled.

        A threat is an empty cell where:
        - There are exactly 4 consecutive markers of the same type
        - This empty cell would extend them to 5
        """
        threats = set()

        # Get all marker positions
        marker_positions = set(
            pos for pos, piece in board.pieces.items()
            if piece == marker_type
        )

        if len(marker_positions) < 4:
            return threats

        # Directions for checking lines
        directions = [
            (0, 1),   # Vertical
            (1, 0),   # Horizontal
            (1, 1),   # Diagonal up-right
            (-1, 1),  # Diagonal up-left
        ]

        # Check each marker as potential start of a near-complete row
        for start_pos in marker_positions:
            for dx, dy in directions:
                # Count consecutive markers in this direction
                run_positions = [start_pos]

                # Forward direction
                current = start_pos
                for _ in range(4):  # Check up to 4 more positions
                    col_idx = ord(current.column) - ord('A') + dx
                    row = current.row + dy
                    if not (0 <= col_idx < 11):
                        break
                    next_pos = Position(chr(ord('A') + col_idx), row)
                    if not is_valid_position(next_pos):
                        break
                    if next_pos in marker_positions:
                        run_positions.append(next_pos)
                        current = next_pos
                    else:
                        break

                # If we have exactly 4 consecutive markers
                if len(run_positions) == 4:
                    # Check if either end is empty (would complete the row)
                    # Check forward end
                    last = run_positions[-1]
                    col_idx = ord(last.column) - ord('A') + dx
                    row = last.row + dy
                    if 0 <= col_idx < 11:
                        forward_pos = Position(chr(ord('A') + col_idx), row)
                        if is_valid_position(forward_pos):
                            if board.get_piece(forward_pos) is None:
                                threats.add(forward_pos)

                    # Check backward end
                    first = run_positions[0]
                    col_idx = ord(first.column) - ord('A') - dx
                    row = first.row - dy
                    if 0 <= col_idx < 11:
                        backward_pos = Position(chr(ord('A') + col_idx), row)
                        if is_valid_position(backward_pos):
                            if board.get_piece(backward_pos) is None:
                                threats.add(backward_pos)

        return threats

    def _encode_partial_rows(self, state: np.ndarray, game_state: GameState, is_white: bool):
        """Encode partial row positions (channels 6-7).

        Partial rows are runs of 3-4 consecutive markers.
        """
        current_marker = PieceType.WHITE_MARKER if is_white else PieceType.BLACK_MARKER
        opponent_marker = PieceType.BLACK_MARKER if is_white else PieceType.WHITE_MARKER

        current_partials = self._find_partial_row_cells(game_state.board, current_marker)
        opponent_partials = self._find_partial_row_cells(game_state.board, opponent_marker)

        if self.enable_stats:
            self._last_stats.partial_rows_found = len(current_partials) + len(opponent_partials)

        for pos in current_partials:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_CURRENT_PARTIAL_ROWS, row_idx, col_idx] = 1.0

        for pos in opponent_partials:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_OPPONENT_PARTIAL_ROWS, row_idx, col_idx] = 1.0

    def _find_partial_row_cells(self, board, marker_type: PieceType) -> Set[Position]:
        """Find all cells that are part of a 3-4 marker run."""
        partial_cells = set()

        marker_positions = set(
            pos for pos, piece in board.pieces.items()
            if piece == marker_type
        )

        if len(marker_positions) < 3:
            return partial_cells

        directions = [
            (0, 1),   # Vertical
            (1, 0),   # Horizontal
            (1, 1),   # Diagonal up-right
            (-1, 1),  # Diagonal up-left
        ]

        for start_pos in marker_positions:
            for dx, dy in directions:
                run_positions = [start_pos]
                current = start_pos

                # Forward direction
                for _ in range(4):
                    col_idx = ord(current.column) - ord('A') + dx
                    row = current.row + dy
                    if not (0 <= col_idx < 11):
                        break
                    next_pos = Position(chr(ord('A') + col_idx), row)
                    if not is_valid_position(next_pos):
                        break
                    if next_pos in marker_positions:
                        run_positions.append(next_pos)
                        current = next_pos
                    else:
                        break

                # If run is 3 or 4 markers (but not 5+, which is complete)
                if 3 <= len(run_positions) < MARKERS_FOR_ROW:
                    partial_cells.update(run_positions)

        return partial_cells

    def _encode_ring_mobility(self, state: np.ndarray, game_state: GameState):
        """Encode ring mobility (channel 8).

        For each ring position, encode the number of legal moves available,
        normalized to [0, 1].
        """
        max_mobility = 20.0  # Reasonable max moves for a ring
        mobility_sum = 0.0

        # Get all rings
        white_rings = game_state.board.get_pieces_positions(PieceType.WHITE_RING)
        black_rings = game_state.board.get_pieces_positions(PieceType.BLACK_RING)

        for pos in white_rings + black_rings:
            valid_moves = game_state.board.valid_move_positions(pos)
            num_moves = len(valid_moves)
            mobility_sum += num_moves

            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_RING_MOBILITY, row_idx, col_idx] = min(num_moves / max_mobility, 1.0)

        if self.enable_stats:
            self._last_stats.ring_mobility_sum = mobility_sum

    def _encode_ring_influence(self, state: np.ndarray, game_state: GameState, is_white: bool):
        """Encode ring influence (channel 10).

        Binary mask of all cells reachable by the current player's rings.
        """
        current_rings = game_state.board.get_rings_positions(
            Player.WHITE if is_white else Player.BLACK
        )

        reachable_cells = set()

        for ring_pos in current_rings:
            valid_moves = game_state.board.valid_move_positions(ring_pos)
            for pos in valid_moves:
                reachable_cells.add(pos)

        if self.enable_stats:
            self._last_stats.ring_influence_coverage = len(reachable_cells) / len(self._valid_positions)

        for pos in reachable_cells:
            col_idx = ord(pos.column) - ord('A')
            row_idx = pos.row - 1
            if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                state[self.CH_RING_INFLUENCE, row_idx, col_idx] = 1.0

    def _encode_valid_moves(self, state: np.ndarray, game_state: GameState):
        """Encode valid move destinations (channel 11)."""
        valid_moves = game_state.get_valid_moves()

        for move in valid_moves:
            if move.source:
                col_idx = ord(move.source.column) - ord('A')
                row_idx = move.source.row - 1
                if 0 <= col_idx < self.BOARD_SIZE and 0 <= row_idx < self.BOARD_SIZE:
                    state[self.CH_VALID_MOVES, row_idx, col_idx] = 1.0

    def _encode_score_differential(self, state: np.ndarray, game_state: GameState, is_white: bool):
        """Encode score differential (channel 14).

        Normalized to [-1, 1] from current player's perspective.
        Positive = current player winning.
        """
        white_score = game_state.white_score
        black_score = game_state.black_score

        if is_white:
            diff = white_score - black_score
        else:
            diff = black_score - white_score

        # Normalize: max score is 3, so diff range is [-3, 3]
        normalized_diff = diff / 3.0
        state[self.CH_SCORE_DIFF] = np.clip(normalized_diff, -1.0, 1.0)

    def get_last_stats(self) -> Optional[EncodingStats]:
        """Get statistics from the last encoding call (if enable_stats=True)."""
        return self._last_stats

    def describe_channels(self) -> Dict[int, str]:
        """Return a description of each channel for documentation/debugging."""
        return {
            0: "Current player's rings",
            1: "Opponent's rings",
            2: "Current player's markers",
            3: "Opponent's markers",
            4: "Current player's row threats (cells completing a row)",
            5: "Opponent's row threats",
            6: "Current player's partial rows (3-4 marker runs)",
            7: "Opponent's partial rows",
            8: "Ring mobility (normalized move count per ring position)",
            9: "Center distance (static heatmap, 1=center, 0=edge)",
            10: "Ring influence (cells reachable by current player)",
            11: "Valid move source positions",
            12: "Game phase (normalized)",
            13: "Turn number (normalized, capped at 100)",
            14: "Score differential (current player's perspective)",
        }


# Convenience function for quick testing
def compare_encodings(game_state: GameState,
                      basic_encoder=None,
                      enhanced_encoder=None) -> Dict:
    """Compare basic and enhanced encodings for debugging.

    Args:
        game_state: Game state to encode
        basic_encoder: Optional BasicStateEncoder instance
        enhanced_encoder: Optional EnhancedStateEncoder instance

    Returns:
        Dictionary with comparison metrics
    """
    from .encoding import StateEncoder

    if basic_encoder is None:
        basic_encoder = StateEncoder()
    if enhanced_encoder is None:
        enhanced_encoder = EnhancedStateEncoder(enable_stats=True)

    basic_state = basic_encoder.encode_state(game_state)
    enhanced_state = enhanced_encoder.encode_state(game_state)

    stats = enhanced_encoder.get_last_stats()

    return {
        'basic_shape': basic_state.shape,
        'enhanced_shape': enhanced_state.shape,
        'basic_nonzero': np.count_nonzero(basic_state),
        'enhanced_nonzero': np.count_nonzero(enhanced_state),
        'first_4_channels_match': np.allclose(basic_state[:4], enhanced_state[:4]),
        'encoding_stats': stats,
    }
