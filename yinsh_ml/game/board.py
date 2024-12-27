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
    MARKERS_FOR_ROW
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
        logger.debug(f"Moving ring from {source} to {destination}")  # Debug

        # Get the ring at the source position
        ring = self.get_piece(source)
        logger.debug(f"Found ring: {ring}")  # Debug
        if not ring or not ring.is_ring():
            logger.debug("No ring at source position")
            return False

        # Verify destination is empty
        if self.get_piece(destination):
            logger.debug("Destination is occupied")
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
                logger.debug(f"Invalid position in path: {next_pos}")
                return False

            positions_between.append(next_pos)
            current = next_pos

        logger.debug(f"Positions between: {positions_between}")  # Debug

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
        logger.debug(f"\nChecking valid moves for position {position}")  # Debug

        if not is_valid_position(position):
            logger.debug(f"Position {position} is not valid")  # Debug
            return []

        piece = self.get_piece(position)
        logger.debug(f"Found piece: {piece}")  # Debug

        if not piece or not piece.is_ring():
            logger.debug(f"No ring at position {position}")  # Debug
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
            logger.debug(f"Checking direction dx={dx}, dy={dy}")  # Debug
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
                    logger.debug(f"Position {next_pos} is not valid")  # Debug
                    break

                # Check what's at the next position
                next_piece = self.get_piece(next_pos)
                logger.debug(f"Checking position {next_pos}, found piece: {next_piece}")  # Debug

                if next_piece:
                    if next_piece.is_ring():
                        # Can't jump over rings
                        logger.debug(f"Found ring at {next_pos}, stopping")  # Debug
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

        logger.debug(f"Found valid moves: {valid_positions}")  # Debug
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
        """Find all rows of markers of the given type."""
        logger.debug(f"\nLooking for rows of {marker_type}")  # Debug
        if not marker_type.is_marker():
            logger.debug(f"{marker_type} is not a marker type")  # Debug
            return []

        unique_rows = set()  # Use a set to store unique rows

        # Get all positions with the marker type
        positions = [
            pos for pos, piece in self.pieces.items()
            if piece == marker_type
        ]
        positions.sort(key=lambda p: (p.column, p.row))  # Sort for consistent processing
        logger.debug(f"Found markers at positions: {positions}")  # Debug

        # Check each position for potential rows
        for start_pos in positions:
            logger.debug(f"\nChecking for rows starting at {start_pos}")  # Debug

            # Check in each direction
            directions = [
                (0, 1),  # Vertical (up)
                (1, 0),  # Horizontal
                (1, 1),  # Diagonal up-right
                (-1, 1),  # Diagonal up-left
            ]

            for dx, dy in directions:
                # Try to build row in this direction
                row = [start_pos]
                current = start_pos
                logger.debug(f"\nChecking direction dx={dx}, dy={dy} from {start_pos}")  # Debug

                # Check up to 4 more positions in this direction
                for i in range(MARKERS_FOR_ROW - 1):
                    col_idx = ord(current.column) - ord('A')
                    new_col = chr(ord('A') + col_idx + dx)
                    new_row = current.row + dy
                    next_pos = Position(new_col, new_row)

                    logger.debug(f"  Checking position {next_pos}")  # Debug

                    if not is_valid_position(next_pos):
                        logger.debug(f"  Position {next_pos} is not valid")  # Debug
                        break

                    piece = self.get_piece(next_pos)
                    logger.debug(f"  Found piece: {piece}")  # Debug

                    if piece != marker_type:
                        logger.debug(f"  Not a matching marker")  # Debug
                        break

                    row.append(next_pos)
                    current = next_pos
                    logger.debug(f"  Current row: {row}")  # Debug

                # If we found enough consecutive markers, we have a row
                if len(row) >= MARKERS_FOR_ROW:
                    # Sort positions to ensure consistent order for comparison
                    sorted_positions = sorted(row[:MARKERS_FOR_ROW], key=lambda p: (p.column, p.row))
                    # Convert positions to strings for hashability
                    pos_strings = tuple(str(p) for p in sorted_positions)
                    if pos_strings not in unique_rows:
                        unique_rows.add(pos_strings)
                        logger.debug(f"Found new unique row: {sorted_positions}")  # Debug

        # Convert back to Row objects
        rows = [
            Row(color=marker_type, positions=[Position.from_string(p) for p in pos_tuple])
            for pos_tuple in unique_rows
        ]

        logger.debug(f"Found {len(rows)} unique rows: {rows}")  # Debug
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
        logger.debug(f"\nValidating marker sequence for {player}:")
        logger.debug(f"Positions to check: {positions}")

        # 1. Check length
        if len(positions) != 5:
            logger.debug(f"Wrong number of positions: {len(positions)}")
            return False

        # 2. Sort positions first
        sorted_positions = sorted(positions, key=lambda p: (p.column, p.row))
        logger.debug(f"Sorted positions: {[str(pos) for pos in sorted_positions]}")

        # 3. Check all positions have correct markers
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        logger.debug(f"Expected marker type: {marker_type}")

        for pos in sorted_positions:
            piece = self.get_piece(pos)
            logger.debug(f"Position {pos}: found piece {piece}")
            if piece != marker_type:
                logger.debug(f"Wrong piece type at {pos}: expected {marker_type}, found {piece}")
                return False

        # 4. Check if positions form a valid line (horizontal, vertical, or diagonal)
        # Get the direction vector from first two positions
        if len(sorted_positions) < 2:
            return False

        p1, p2 = sorted_positions[0], sorted_positions[1]
        dx = ord(p2.column) - ord(p1.column)
        dy = p2.row - p1.row

        # Check if all subsequent positions follow the same direction
        for i in range(1, len(sorted_positions) - 1):
            curr = sorted_positions[i]
            next_pos = sorted_positions[i + 1]
            curr_dx = ord(next_pos.column) - ord(curr.column)
            curr_dy = next_pos.row - curr.row

            if curr_dx != dx or curr_dy != dy:
                logger.debug(f"Positions do not form a straight line: direction changes at position {i}")
                return False

            # Check if positions are consecutive
            if abs(curr_dx) > 1 or abs(curr_dy) > 1:
                logger.debug(f"Positions are not consecutive: gap between {curr} and {next_pos}")
                return False

        logger.debug("Sequence is valid!")
        return True

    def _is_continuous_line(self, positions: List[Position]) -> bool:
        """Check if positions form a continuous line."""
        if len(positions) < 2:
            logger.debug("Less than 2 positions")  # Debug
            return True

        # Get direction from first two positions
        pos1, pos2 = positions[:2]
        dx = ord(pos2.column) - ord(pos1.column)
        dy = pos2.row - pos1.row
        logger.debug(f"Direction: dx={dx}, dy={dy}")  # Debug

        # Check each subsequent pair follows the same direction
        for i in range(1, len(positions) - 1):
            curr = positions[i]
            next_pos = positions[i + 1]

            curr_dx = ord(next_pos.column) - ord(curr.column)
            curr_dy = next_pos.row - curr.row
            logger.debug(f"Checking pair {i}: {curr}->{next_pos}, dx={curr_dx}, dy={curr_dy}")  # Debug

            if curr_dx != dx or curr_dy != dy:
                logger.debug(f"Direction mismatch at position {i}")  # Debug
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
        logger.debug(f"Checking if position {pos} is empty")

        # First verify position is valid
        if not is_valid_position(pos):
            logger.debug(f"Position {pos} is not valid")
            return False

        # Then check if position has a piece
        has_piece = pos in self.pieces
        logger.debug(f"Position has piece? {has_piece}")

        return not has_piece

    def get_pieces_of_player(self, player: Player) -> List[Position]:
        """Get all pieces belonging to a player."""
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        return [pos for pos, piece in self.pieces.items()
                if piece == ring_type or piece == marker_type]