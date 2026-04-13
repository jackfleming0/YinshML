"""Parser for the official GIPF-project YINSH notation.

The official notation (from gipf.com) uses algebraic coordinates (a-k, 1-11)
which map directly to our internal Position(column, row) system.

Move notation format:
  - Ring placement: "e3" (just the position)
  - Ring movement: "a2-h9" (source-destination)
  - Row removal + ring removal (short): "i8-f5;xa5"
  - Row removal + ring removal (long): "i8-f5;xg5-g9xa5"
  - Opponent row formation: "xa5;d3-b3" (remove first, then move)
  - Multiple removals: "i8-f5;xa5;xg2"

Move numbering: odd = White (player 1), even = Black (player 2).
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

from ...game.constants import Player, Position, is_valid_position
from .utils import positions_on_line

logger = logging.getLogger(__name__)

# Regex for parsing individual components
_POS_RE = re.compile(r'([a-kA-K])(\d{1,2})')
_MOVE_RE = re.compile(r'([a-kA-K]\d{1,2})-([a-kA-K]\d{1,2})')
_REMOVAL_RE = re.compile(r'x([a-kA-K]\d{1,2})(?:-([a-kA-K]\d{1,2}))?x([a-kA-K]\d{1,2})')
_RING_REMOVAL_SHORT_RE = re.compile(r'x([a-kA-K]\d{1,2})')


def parse_position(pos_str: str) -> Position:
    """Convert a GIPF notation position to internal Position.

    The notation uses lowercase letters; our system uses uppercase.

    Args:
        pos_str: Position string like "e5" or "E5".

    Returns:
        Position object.

    Raises:
        ValueError: If position is invalid.
    """
    pos_str = pos_str.strip()
    m = _POS_RE.fullmatch(pos_str)
    if not m:
        raise ValueError(f"Invalid position: {pos_str!r}")

    col = m.group(1).upper()
    row = int(m.group(2))
    pos = Position(col, row)

    if not is_valid_position(pos):
        raise ValueError(f"Position {pos} is not a valid YINSH board position")

    return pos


def parse_game_record(move_text: str) -> List[Dict]:
    """Parse a full game record in official GIPF notation.

    The game record contains numbered moves separated by whitespace:
        "1.e5 2.g7 3.d4 ..."

    Args:
        move_text: Full game record text.

    Returns:
        List of standardized move dicts.
    """
    moves = []
    # Split on whitespace and handle numbered moves
    tokens = move_text.strip().split()

    for token in tokens:
        # Strip move number prefix: "1.e5" → "e5", "24.a2-h9;xa5" → "a2-h9;xa5"
        if '.' in token and token.split('.')[0].isdigit():
            num_str, move_str = token.split('.', 1)
            move_num = int(num_str)
        else:
            move_str = token
            move_num = None

        # Determine player from move number (odd = White, even = Black)
        if move_num is not None:
            player = 'white' if move_num % 2 == 1 else 'black'
        else:
            # Infer from position in sequence
            player = 'white' if len(moves) % 2 == 0 else 'black'

        parsed = _parse_move_token(move_str, player)
        moves.extend(parsed)

    return moves


def _parse_move_token(token: str, player: str) -> List[Dict]:
    """Parse a single move token which may contain multiple actions.

    A token like "a2-h9;xg5-g9xa5" contains:
    1. Ring movement a2 → h9
    2. Marker removal g5-g9
    3. Ring removal a5

    Opponent removals: "xa5;d3-b3" means opponent removes, then player moves.
    """
    result = []
    parts = token.split(';')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith('x'):
            # This is a removal sequence
            removal_moves = _parse_removal(part, player)
            result.extend(removal_moves)
        elif '-' in part:
            # Ring movement: "a2-h9"
            src_str, dst_str = part.split('-', 1)
            src = parse_position(src_str)
            dst = parse_position(dst_str)
            # Determine if this is ring placement phase by context
            # For ring movement, we always emit MOVE_RING
            # The converter will handle the PLACE_MARKER that precedes it
            result.append({
                'move_type': 'MOVE_RING',
                'player': player,
                'source': str(src),
                'destination': str(dst),
            })
        else:
            # Simple position: ring placement
            pos = parse_position(part)
            result.append({
                'move_type': 'PLACE_RING',
                'player': player,
                'position': str(pos),
            })

    return result


def _parse_removal(removal_str: str, player: str) -> List[Dict]:
    """Parse a removal string like "xg5-g9xa5" or just "xa5".

    Long form: xSTART-ENDxRING — remove markers from START to END, remove ring at RING.
    Short form: xRING — just the ring removal (markers inferred).

    Returns list of move dicts (REMOVE_MARKERS + REMOVE_RING).
    """
    result = []

    # Try long form first: "xg5-g9xa5"
    long_match = _REMOVAL_RE.match(removal_str)
    if long_match:
        marker_start = parse_position(long_match.group(1))
        marker_end = parse_position(long_match.group(2))
        ring_pos = parse_position(long_match.group(3))

        # Generate marker positions between start and end
        markers = positions_on_line(marker_start, marker_end)
        if markers:
            result.append({
                'move_type': 'REMOVE_MARKERS',
                'player': player,
                'markers': [str(m) for m in markers],
            })

        result.append({
            'move_type': 'REMOVE_RING',
            'player': player,
            'position': str(ring_pos),
        })
        return result

    # Short form: just "xa5" — ring removal only
    # (markers are inferred by the game engine)
    short_match = _RING_REMOVAL_SHORT_RE.match(removal_str)
    if short_match:
        ring_pos = parse_position(short_match.group(1))
        result.append({
            'move_type': 'REMOVE_RING',
            'player': player,
            'position': str(ring_pos),
        })
        return result

    logger.warning(f"Could not parse removal: {removal_str!r}")
    return result


# Keep as alias for backward compatibility with tests
_positions_between = positions_on_line
