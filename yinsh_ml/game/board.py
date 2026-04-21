"""Core game logic for YINSH board."""

from typing import Dict, List, Optional, Set, Tuple, ForwardRef
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import logging

from .constants import (
    Position,
    Player,
    PieceType,
    is_valid_position,
    RINGS_PER_PLAYER,
    MARKERS_FOR_ROW,
    DIRECTIONS,
    HEX_LINE_AXES,
)

# Setup logger
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Row:
    """A row of markers."""
    color: PieceType
    positions: Tuple[Position, ...]  # Changed from List to Tuple

    def __post_init__(self):
        # Ensure positions is always a tuple
        if isinstance(self.positions, list):
            object.__setattr__(self, 'positions', tuple(self.positions))

    @property
    def length(self) -> int:
        return len(self.positions)

class Board:
    """Represents the YINSH game board state."""

    def __init__(self):
        # Initialize an empty board
        self.pieces: Dict[Position, PieceType] = {}

    def copy_from(self, source: 'Board') -> None:
        """Efficiently copy board state from another Board instance.
        
        This is optimized for memory pool usage and avoids creating
        a new dictionary by clearing and updating the existing one.
        
        Args:
            source: Board instance to copy from
        """
        self.pieces.clear()
        self.pieces.update(source.pieces)

    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.pieces = self.pieces.copy()
        return new_board

    def place_piece(self, position: Position, piece: PieceType) -> bool:
        """Place a piece on the board."""
        if not is_valid_position(position) or position in self.pieces:
            return False
        self.pieces[position] = piece
        return True

    def remove_piece(self, position: Position) -> Optional[PieceType]:
        """Remove and return the piece at a position."""
        return self.pieces.pop(position, None)

    def get_piece(self, pos: Position) -> Optional[PieceType]:
        """Get the piece at a position."""
        if not is_valid_position(pos):
            return None
        return self.pieces.get(pos)

    def move_ring(self, source: Position, destination: Position) -> bool:
        """Move a ring from source to destination, flipping markers in between."""
        pass

        # Get the ring at the source position
        ring = self.get_piece(source)
        pass
        if not ring or not ring.is_ring():
            pass
            return False

        # Verify destination is empty
        if self.get_piece(destination):
            pass
            return False

        # CRITICAL FIX: Validate that destination is a legal move
        valid_destinations = self.valid_move_positions(source)
        if destination not in valid_destinations:
            pass
            pass
            return False

        # Calculate directional vector
        col_diff = ord(destination.column) - ord(source.column)
        row_diff = destination.row - source.row
        if col_diff == 0 and row_diff == 0:
            return False

        # Normalize direction
        steps = max(abs(col_diff), abs(row_diff))
        dx = col_diff // steps if col_diff != 0 else 0
        dy = row_diff // steps if row_diff != 0 else 0

        # Get positions between source and destination
        positions_between = []
        current = source
        for _ in range(steps - 1):
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)

            if not is_valid_position(next_pos):
                pass
                return False

            positions_between.append(next_pos)
            current = next_pos

        pass

        # Place marker at source position
        marker_type = (PieceType.WHITE_MARKER if ring == PieceType.WHITE_RING
                       else PieceType.BLACK_MARKER)
        self.remove_piece(source)
        self.place_piece(source, marker_type)

        # Flip markers along the path
        for pos in positions_between:
            piece = self.get_piece(pos)
            if piece and piece.is_marker():
                self.remove_piece(pos)
                new_marker = (PieceType.WHITE_MARKER if piece == PieceType.BLACK_MARKER
                              else PieceType.BLACK_MARKER)
                self.place_piece(pos, new_marker)

        # Place ring at destination
        self.place_piece(destination, ring)

        return True

    def valid_move_positions(self, position: Position) -> List[Position]:
        """Get valid move destinations for a ring at the given position."""
        pass

        if not is_valid_position(position):
            pass
            return []

        piece = self.get_piece(position)
        pass

        if not piece or not piece.is_ring():
            pass
            return []

        valid_positions = []

        # Check in all 6 directions
        directions = [
            (0, 1),  # Up
            (1, 1),  # Up-Right
            (1, 0),  # Right
            (0, -1),  # Down
            (-1, -1),  # Down-Left
            (-1, 0),  # Left
        ]

        for dx, dy in directions:
            pass
            current = position
            jumped_over_marker = False

            while True:
                # Get next position
                col_idx = ord(current.column) - ord('A')
                new_col = chr(ord('A') + col_idx + dx)
                new_row = current.row + dy
                next_pos = Position(new_col, new_row)

                # Check if position is valid
                if not is_valid_position(next_pos):
                    pass
                    break

                # Check what's at the next position
                next_piece = self.get_piece(next_pos)
                pass

                if next_piece:
                    if next_piece.is_ring():
                        # Can't jump over rings
                        pass
                        break
                    else:
                        # Can jump over markers
                        jumped_over_marker = True
                else:
                    # Empty space - valid move
                    valid_positions.append(next_pos)
                    if jumped_over_marker:
                        # Can only continue until first empty space after a marker
                        break

                current = next_pos

        pass
        return valid_positions

    def get_rings_positions(self, player: Player) -> List[Position]:
        """Get all positions of a player's rings."""
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
        return [pos for pos, piece in self.pieces.items() if piece == ring_type]

    # Use a forward reference (string) for the Move type hint
    # def get_ring_valid_moves(self, position: Position) -> List['Move']:
    #     """Get valid moves for a ring at the given position."""
    #     valid_moves = self.get_valid_moves()  # Assuming this method exists in GameState
    #     ring_moves = [
    #         move for move in valid_moves
    #         if move.type == MoveType.MOVE_RING and move.source == position
    #     ]
    #     return ring_moves

    def get_markers_between(self, start: Position, end: Position) -> List[Position]:
        """Get all marker positions between start and end."""
        markers = []

        # Calculate direction vector
        col_diff = ord(end.column) - ord(start.column)
        row_diff = end.row - start.row

        if col_diff == 0 and row_diff == 0:
            return markers

        # Normalize direction
        steps = max(abs(col_diff), abs(row_diff))
        dx = col_diff // steps if col_diff != 0 else 0
        dy = row_diff // steps if row_diff != 0 else 0

        # Get all positions along the path
        current = start
        for _ in range(steps - 1):
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)

            if not is_valid_position(next_pos):
                break

            if self.get_piece(next_pos) and self.get_piece(next_pos).is_marker():
                markers.append(next_pos)

            current = next_pos

        return markers

    def find_marker_rows(self, marker_type: PieceType) -> List[Row]:
        """Find all maximal runs of contiguous same-color markers along a
        hex axis whose length is >= MARKERS_FOR_ROW (5).

        Real YINSH allows rows of 6 or 7 markers: a player may extend a
        5-in-a-line with additional markers before completing the row,
        and when the row is completed they may choose ANY 5 consecutive
        markers from the run to remove. We therefore return the FULL run
        here (5, 6, or 7 positions), and let the move generator
        enumerate every length-5 window across it.

        Duplicate runs (same set of positions detected from multiple
        start points along the same axis) are deduped via a set keyed on
        the sorted full-run position tuple.
        """
        pass
        if not marker_type.is_marker():
            pass
            return []

        unique_rows = set()  # Dedupe by sorted full-run position tuple

        # Get all positions with the marker type
        positions = [
            pos for pos, piece in self.pieces.items()
            if piece == marker_type
        ]
        positions.sort(key=lambda p: (p.column, p.row))  # Sort for consistent processing
        pass

        # Check each position for potential rows
        for start_pos in positions:
            pass

            # Check in each of the 3 hex-axis forward directions. NOTE:
            # the previous inline list included the pseudo-diagonal
            # (-1, 1), which is NOT a real hex axis in this (col, row)
            # coordinate system and produced spurious "rows". See
            # yinsh_ml/game/constants.py::DIRECTIONS for the single
            # source of truth.
            for dx, dy in DIRECTIONS:
                # Only walk forward from start_pos — to avoid counting a
                # longer run multiple times, skip this (start_pos, axis)
                # if the previous cell in the axis is also a matching
                # marker (i.e. start_pos is not the run's lower end).
                prev_col_idx = ord(start_pos.column) - ord('A') - dx
                prev_row = start_pos.row - dy
                if 0 <= prev_col_idx < 11:
                    prev_col = chr(ord('A') + prev_col_idx)
                    prev_pos = Position(prev_col, prev_row)
                    if (is_valid_position(prev_pos)
                            and self.get_piece(prev_pos) == marker_type):
                        continue

                # Walk the full run forward from start_pos.
                row = [start_pos]
                current = start_pos
                pass

                while True:
                    col_idx = ord(current.column) - ord('A')
                    new_col = chr(ord('A') + col_idx + dx)
                    new_row = current.row + dy
                    next_pos = Position(new_col, new_row)

                    pass

                    if not is_valid_position(next_pos):
                        pass
                        break

                    piece = self.get_piece(next_pos)
                    pass

                    if piece != marker_type:
                        pass
                        break

                    row.append(next_pos)
                    current = next_pos
                    pass

                # If we found enough consecutive markers, record the
                # full run (5, 6, or 7 positions).
                if len(row) >= MARKERS_FOR_ROW:
                    sorted_positions = sorted(row, key=lambda p: (p.column, p.row))
                    pos_strings = tuple(str(p) for p in sorted_positions)
                    if pos_strings not in unique_rows:
                        unique_rows.add(pos_strings)
                        pass

        # Convert back to Row objects
        rows = [
            Row(color=marker_type, positions=[Position.from_string(p) for p in pos_tuple])
            for pos_tuple in unique_rows
        ]

        pass
        return rows

    def _get_marker_row(self, start: Position, marker_type: PieceType, direction: Tuple[int, int]) -> List[Position]:
        """Get a row of markers in a given direction."""
        row = [start]
        dx, dy = direction

        # Check forward
        current = start
        while True:
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)

            if not is_valid_position(next_pos):
                break

            piece = self.get_piece(next_pos)
            if piece != marker_type:
                break

            row.append(next_pos)
            current = next_pos

        # Check backward
        current = start
        while True:
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx - dx)
            new_row = current.row - dy
            next_pos = Position(new_col, new_row)

            if not is_valid_position(next_pos):
                break

            piece = self.get_piece(next_pos)
            if piece != marker_type:
                break

            row.insert(0, next_pos)
            current = next_pos

        return row

    def is_valid_marker_sequence(self, positions: List[Position], player: Player) -> bool:
        """Check if a sequence of markers forms a valid row for removal."""
        pass
        pass

        # 1. Check length
        if len(positions) != 5:
            pass
            return False

        # 2. Sort positions first
        sorted_positions = sorted(positions, key=lambda p: (p.column, p.row))
        pass

        # 3. Check all positions have correct markers
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        pass

        for pos in sorted_positions:
            piece = self.get_piece(pos)
            pass
            if piece != marker_type:
                pass
                return False

        # 4. Check if positions form a valid line (horizontal, vertical, or diagonal)
        # Get the direction vector from first two positions
        if len(sorted_positions) < 2:
            return False

        p1, p2 = sorted_positions[0], sorted_positions[1]
        dx = ord(p2.column) - ord(p1.column)
        dy = p2.row - p1.row

        # Require the step (dx, dy) to be a unit step on a real hex
        # axis. Previously this check only enforced collinearity, so
        # 5 markers along the pseudo-diagonal (-1, 1) / (1, -1) —
        # which is NOT a line on the YINSH board in this coordinate
        # system — would be accepted as a valid row.
        if (dx, dy) not in HEX_LINE_AXES:
            pass
            return False

        # Check if all subsequent positions follow the same direction
        for i in range(1, len(sorted_positions) - 1):
            curr = sorted_positions[i]
            next_pos = sorted_positions[i + 1]
            curr_dx = ord(next_pos.column) - ord(curr.column)
            curr_dy = next_pos.row - curr.row

            if curr_dx != dx or curr_dy != dy:
                pass
                return False

            # Check if positions are consecutive
            if abs(curr_dx) > 1 or abs(curr_dy) > 1:
                pass
                return False

        pass
        return True

    def _is_continuous_line(self, positions: List[Position]) -> bool:
        """Check if positions form a continuous line."""
        if len(positions) < 2:
            pass
            return True

        # Get direction from first two positions
        pos1, pos2 = positions[:2]
        dx = ord(pos2.column) - ord(pos1.column)
        dy = pos2.row - pos1.row
        pass

        # Check each subsequent pair follows the same direction
        for i in range(1, len(positions) - 1):
            curr = positions[i]
            next_pos = positions[i + 1]

            curr_dx = ord(next_pos.column) - ord(curr.column)
            curr_dy = next_pos.row - curr.row
            pass

            if curr_dx != dx or curr_dy != dy:
                pass
                return False

        return True

    def to_numpy_array(self) -> np.ndarray:
        """Convert board state to numpy array."""
        state = np.zeros((4, 11, 11), dtype=np.float32)

        for pos, piece in self.pieces.items():
            row = pos.row - 1
            col = ord(pos.column) - ord('A')

            if piece == PieceType.WHITE_RING:
                state[0, row, col] = 1
            elif piece == PieceType.BLACK_RING:
                state[1, row, col] = 1
            elif piece == PieceType.WHITE_MARKER:
                state[2, row, col] = 1
            elif piece == PieceType.BLACK_MARKER:
                state[3, row, col] = 1

        return state

    def __str__(self) -> str:
        """Return string representation of the board."""
        result = []
        result.append("   " + " ".join("ABCDEFGHIJK"))

        for row in range(1, 12):
            row_str = f"{row:2d} "
            for col in "ABCDEFGHIJK":
                pos = Position(col, row)
                if not is_valid_position(pos):
                    row_str += "  "
                else:
                    piece = self.get_piece(pos)
                    if piece:
                        row_str += piece.value + " "
                    else:
                        row_str += ". "
            result.append(row_str)

        return "\n".join(result)

    def get_pieces_positions(self, piece_type: PieceType) -> List[Position]:
        """Get all positions containing pieces of the specified type."""
        return [pos for pos, piece in self.pieces.items() if piece == piece_type]

    def is_empty(self, pos: Position) -> bool:
        """Check if a position is empty and valid."""
        pass

        # First verify position is valid
        if not is_valid_position(pos):
            pass
            return False

        # Then check if position has a piece
        has_piece = pos in self.pieces
        pass

        return not has_piece

    def get_pieces_of_player(self, player: Player) -> List[Position]:
        """Get all pieces belonging to a player."""
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        return [pos for pos, piece in self.pieces.items()
                if piece == ring_type or piece == marker_type]