"""Parser for CodinGame YINSH move notation.

CodinGame uses a concentric hexagonal ring coordinate system:
  - Ring 0: center of the board (1 position)
  - Ring h (h >= 1): 6*h positions arranged around the center
  - Position 0 in each ring is the "topmost" point
  - Position numbering increases clockwise

Moves are space-separated sequences of TYPE RING POSITION triples:
  P 1 2         — Place ring at (ring=1, pos=2)
  S 1 2 M 2 4   — Select ring at (1,2), move to (2,4)
  S 1 2 M 2 4 RS 1 2 RE 4 16 X 3 4
                 — Move + row removal + ring removal
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from ...game.constants import Player, Position, is_valid_position
from .utils import positions_on_line as _positions_on_line_shared

logger = logging.getLogger(__name__)

# Board center in algebraic coordinates
_CENTER_COL = 5  # F (0-indexed: A=0, ..., F=5, ..., K=10)
_CENTER_ROW = 6

# The 6 hex directions for the YINSH board in (col_delta, row_delta).
# These correspond to the 3 line directions (and their negatives) on the
# YINSH triangulated hex grid. Verified by BFS: these produce the correct
# concentric ring structure of 1, 6, 12, 18, 24, 24 = 85 positions.
_HEX_DIRS = [
    (0, -1),   # 0: row decreases (used as "top" direction for ring start)
    (1, 0),    # 1: col increases
    (1, 1),    # 2: col + row increase (diagonal)
    (0, 1),    # 3: row increases
    (-1, 0),   # 4: col decreases
    (-1, -1),  # 5: col + row decrease (diagonal)
]


def _build_ring_map() -> Dict[Tuple[int, int], Position]:
    """Build mapping from (ring, position) → algebraic Position.

    Uses concentric hex rings outward from center F6.
    Ring 0 = center. For ring h, position 0 is at the "top" (moving
    in direction 0 from center), numbering goes clockwise.

    Returns:
        Dict mapping (ring, pos) tuples to Position objects.
    """
    mapping = {}

    # Ring 0: center
    center = Position(chr(ord('A') + _CENTER_COL), _CENTER_ROW)
    mapping[(0, 0)] = center

    for h in range(1, 6):  # rings 1-5
        # Start at vertex 0: h steps from center in direction 0
        col = _CENTER_COL + h * _HEX_DIRS[0][0]
        row = _CENTER_ROW + h * _HEX_DIRS[0][1]

        pos_idx = 0
        # Traverse 6 edges, each with h steps
        for edge in range(6):
            # Direction for this edge: 2 positions clockwise from the
            # radial direction that points to the starting vertex of this edge.
            # For standard hex ring traversal, edge i goes in direction (i+2) % 6.
            edge_dir = _HEX_DIRS[(edge + 2) % 6]

            for step in range(h):
                pos_col_chr = chr(ord('A') + col)
                pos = Position(pos_col_chr, row)

                if is_valid_position(pos):
                    mapping[(h, pos_idx)] = pos

                pos_idx += 1
                col += edge_dir[0]
                row += edge_dir[1]

    return mapping


def _build_reverse_map(ring_map: Dict[Tuple[int, int], Position]
                       ) -> Dict[str, Tuple[int, int]]:
    """Build reverse mapping from position string → (ring, pos)."""
    return {str(pos): key for key, pos in ring_map.items()}


# Module-level lookup tables (built once)
_RING_TO_POS = _build_ring_map()
_POS_TO_RING = _build_reverse_map(_RING_TO_POS)


def cg_to_position(ring: int, pos: int) -> Position:
    """Convert CodinGame (ring, position) to internal Position.

    Args:
        ring: Hex ring number (0 = center, 1-5 = outer rings).
        pos: Position within the ring (0 = top, increases clockwise).

    Returns:
        Position object.

    Raises:
        ValueError: If the coordinates don't map to a valid position.
    """
    key = (ring, pos)
    if key not in _RING_TO_POS:
        raise ValueError(
            f"CodinGame position ({ring}, {pos}) does not map to a valid "
            f"YINSH board position"
        )
    return _RING_TO_POS[key]


def position_to_cg(pos: Position) -> Tuple[int, int]:
    """Convert internal Position to CodinGame (ring, position).

    Args:
        pos: Internal Position object.

    Returns:
        (ring, position) tuple.

    Raises:
        ValueError: If position is not in the mapping.
    """
    key = str(pos)
    if key not in _POS_TO_RING:
        raise ValueError(f"Position {pos} not found in CodinGame mapping")
    return _POS_TO_RING[key]


def get_mapping_stats() -> Dict:
    """Return statistics about the coordinate mapping for verification."""
    ring_counts = {}
    for (ring, _pos) in _RING_TO_POS:
        ring_counts[ring] = ring_counts.get(ring, 0) + 1

    return {
        'total_positions': len(_RING_TO_POS),
        'positions_per_ring': ring_counts,
        'expected_total': 85,
        'valid': len(_RING_TO_POS) == 85,
    }


def parse_cg_moves(move_line: str, player: str) -> List[Dict]:
    """Parse a CodinGame move line into standardized move dicts.

    A move line is space-separated TYPE RING POS triples:
      "P 1 2" → ring placement
      "S 1 2 M 2 4" → select ring + move
      "S 1 2 M 2 4 RS 0 0 RE 3 10 X 2 5" → move + removal

    Args:
        move_line: Space-separated move string from CG.
        player: "white" or "black".

    Returns:
        List of standardized move dicts.
    """
    tokens = move_line.strip().split()
    result = []
    i = 0

    while i < len(tokens):
        cmd = tokens[i]

        if cmd == 'P':
            # Place ring: P RING POS
            ring, pos = int(tokens[i + 1]), int(tokens[i + 2])
            position = cg_to_position(ring, pos)
            result.append({
                'move_type': 'PLACE_RING',
                'player': player,
                'position': str(position),
            })
            i += 3

        elif cmd == 'S':
            # Select ring: S RING POS — this is the source for the next M command
            ring, pos = int(tokens[i + 1]), int(tokens[i + 2])
            source = cg_to_position(ring, pos)
            i += 3

            # Expect M next
            if i < len(tokens) and tokens[i] == 'M':
                ring2, pos2 = int(tokens[i + 1]), int(tokens[i + 2])
                dest = cg_to_position(ring2, pos2)
                result.append({
                    'move_type': 'MOVE_RING',
                    'player': player,
                    'source': str(source),
                    'destination': str(dest),
                })
                i += 3
            else:
                logger.warning(f"Expected M after S, got: {tokens[i] if i < len(tokens) else 'EOF'}")

        elif cmd == 'RS':
            # Row removal start: RS RING POS
            ring, pos = int(tokens[i + 1]), int(tokens[i + 2])
            row_start = cg_to_position(ring, pos)
            i += 3

            # Expect RE next
            if i < len(tokens) and tokens[i] == 'RE':
                ring2, pos2 = int(tokens[i + 1]), int(tokens[i + 2])
                row_end = cg_to_position(ring2, pos2)
                i += 3

                # Generate all marker positions between start and end
                markers = _positions_on_line(row_start, row_end)
                if markers:
                    result.append({
                        'move_type': 'REMOVE_MARKERS',
                        'player': player,
                        'markers': [str(m) for m in markers],
                    })
            else:
                logger.warning(f"Expected RE after RS, got: {tokens[i] if i < len(tokens) else 'EOF'}")

        elif cmd == 'X':
            # Remove ring: X RING POS
            ring, pos = int(tokens[i + 1]), int(tokens[i + 2])
            position = cg_to_position(ring, pos)
            result.append({
                'move_type': 'REMOVE_RING',
                'player': player,
                'position': str(position),
            })
            i += 3

        elif cmd == 'M':
            # Standalone M (shouldn't happen in valid CG output, but handle gracefully)
            logger.warning(f"Unexpected standalone M command at position {i}")
            i += 3

        else:
            logger.warning(f"Unknown CG command: {cmd!r}")
            i += 1

    return result


def _positions_on_line(start: Position, end: Position) -> List[Position]:
    """Generate all valid positions on the hex line from start to end (inclusive)."""
    return _positions_on_line_shared(start, end)
