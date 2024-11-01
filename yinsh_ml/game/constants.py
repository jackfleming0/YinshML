"""Constants for YINSH game logic."""

from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass
import logging

# Setup logger
logger = logging.getLogger(__name__)

class Player(Enum):
    WHITE = 1
    BLACK = -1

    @property
    def opponent(self):
        """Get the opposing player."""
        return Player.BLACK if self == Player.WHITE else Player.WHITE


class PieceType(Enum):
    EMPTY = "."     # Changed from 0
    WHITE_RING = "R"  # Changed from 1
    BLACK_RING = "r"  # Changed from 2
    WHITE_MARKER = "M"  # Changed from 3
    BLACK_MARKER = "m"  # Changed from 4

    def is_ring(self) -> bool:
        return self in {PieceType.WHITE_RING, PieceType.BLACK_RING}

    def is_marker(self) -> bool:
        return self in {PieceType.WHITE_MARKER, PieceType.BLACK_MARKER}

    def get_player(self) -> Player:
        if self in {PieceType.WHITE_RING, PieceType.WHITE_MARKER}:
            return Player.WHITE
        elif self in {PieceType.BLACK_RING, PieceType.BLACK_MARKER}:
            return Player.BLACK
        return None


@dataclass(frozen=True)
class Position:
    """Represents a position on the YINSH board."""
    column: str  # A-K
    row: int  # 1-11

    def __str__(self) -> str:
        """String representation of position (e.g. 'E5')."""
        return f"{self.column}{self.row}"

    def __repr__(self) -> str:
        """Detailed string representation of position."""
        return f"Position(column='{self.column}', row={self.row})"

    @staticmethod
    def from_string(pos_str: str) -> 'Position':
        """Create Position from string like 'E5'."""
        return Position(pos_str[0], int(pos_str[1:]))


# Valid positions for each column - define this explicitly to avoid any confusion
VALID_POSITIONS: Dict[str, Set[int]] = {
    'A': set([2, 3, 4, 5]),  # A2-A5
    'B': set(range(1, 8)),   # B1-B7
    'C': set(range(1, 9)),   # C1-C8
    'D': set(range(1, 10)),  # D1-D9
    'E': set(range(1, 11)),  # E1-E10
    'F': set(range(2, 11)),  # F2-F10
    'G': set(range(2, 12)),  # G2-G11
    'H': set(range(3, 12)),  # H3-H11
    'I': set(range(4, 12)),  # I4-I11
    'J': set(range(5, 12)),  # J5-J11
    'K': set(range(7, 11)),  # K7-K10
}

# Directions for moves and row detection
DIRECTIONS = [
    (0, 1),  # Vertical
    (1, 0),  # Horizontal
    (1, 1),  # Diagonal up-right
    (-1, 1),  # Diagonal up-left
]

# Game configuration
RINGS_PER_PLAYER = 5
MARKERS_FOR_ROW = 5
POINTS_TO_WIN = 3
POINTS_TO_WIN_BLITZ = 1

# Movement constraints
MAX_RING_MOVE_DISTANCE = 5  # Maximum spaces a ring can move
MIN_MARKER_SEQUENCE = 5  # Minimum markers needed for removal
MAX_MARKER_SEQUENCE = 7  # Maximum possible markers in a row

# Hexagonal geometry constants
HEX_DIRECTIONS = [
    (1, 0),  # East
    (1, -1),  # Northeast
    (0, -1),  # Northwest
    (-1, 0),  # West
    (-1, 1),  # Southwest
    (0, 1),  # Southeast
]


def is_valid_position(pos: Position) -> bool:
    """Check if a position is valid on the YINSH board."""
    try:
        logger.debug(f"Checking validity of position: {pos}")
        logger.debug(f"  Column: {pos.column}, Row: {pos.row}")
        logger.debug(f"  Valid columns: {list(VALID_POSITIONS.keys())}")

        if pos.column not in VALID_POSITIONS:
            logger.debug(f"  Invalid column: {pos.column}")
            return False

        valid_rows = VALID_POSITIONS[pos.column]
        is_valid = pos.row in valid_rows
        logger.debug(f"  Valid rows for column {pos.column}: {sorted(list(valid_rows))}")
        logger.debug(f"  Row {pos.row} is valid: {is_valid}")

        return is_valid

    except Exception as e:
        logger.debug(f"Error checking position validity: {e}")
        return False


def get_next_position(pos: Position, direction: tuple[int, int], steps: int = 1) -> Position:
    """Get the next position in a given direction."""
    col_idx = ord(pos.column) - ord('A')
    new_col_idx = col_idx + direction[0] * steps
    new_row = pos.row + direction[1] * steps

    if not (0 <= new_col_idx < 11):  # A-K
        return None

    new_col = chr(ord('A') + new_col_idx)
    new_pos = Position(new_col, new_row)

    return new_pos if is_valid_position(new_pos) else None