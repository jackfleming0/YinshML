"""Shared utilities for notation parsers."""

import logging
from typing import List

from ...game.constants import Position, is_valid_position

logger = logging.getLogger(__name__)

# The 6 valid hex directions on the YINSH board (col_delta, row_delta).
# These correspond to vertical, horizontal, and diagonal line directions.
HEX_LINE_DIRECTIONS = [
    (0, 1), (0, -1),    # vertical
    (1, 0), (-1, 0),    # horizontal
    (1, 1), (-1, -1),   # diagonal
]


def positions_on_line(start: Position, end: Position) -> List[Position]:
    """Generate all valid positions on the hex line from start to end (inclusive).

    The start and end must lie on the same hex line (vertical, horizontal,
    or diagonal). Returns empty list if they don't.

    Args:
        start: Starting position.
        end: Ending position.

    Returns:
        List of positions from start to end along the line, inclusive.
    """
    start_col = ord(start.column) - ord('A')
    start_row = start.row
    end_col = ord(end.column) - ord('A')
    end_row = end.row

    dcol = end_col - start_col
    drow = end_row - start_row

    if dcol == 0 and drow == 0:
        return [start]

    steps = max(abs(dcol), abs(drow))
    if steps == 0:
        return [start]

    ucol = dcol // steps if dcol != 0 else 0
    urow = drow // steps if drow != 0 else 0

    if (ucol, urow) not in HEX_LINE_DIRECTIONS:
        logger.warning(f"Positions {start}-{end} not on a valid hex line")
        return []

    positions = []
    for i in range(steps + 1):
        col = chr(ord('A') + start_col + i * ucol)
        row = start_row + i * urow
        pos = Position(col, row)
        if is_valid_position(pos):
            positions.append(pos)

    return positions
