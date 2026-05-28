"""Translate between our `Move` dataclass and the yngine wire format.

yngine native coordinates are (x, y) with x ∈ [0, 10] and y ∈ [0, 10]. The
mapping to our (column, row) is:

    yngine (x, y)  →  our Position(chr('A' + x), 11 - y)
    our (col, row) →  yngine (ord(col) - ord('A'), 11 - row)

yngine's six hex directions are integers 0..5; (dx_yngine, dy_yngine) maps
to our delta (dcol, drow) = (dx_yngine, -dy_yngine) because we mirror the
y-axis. The six directions are:

    0 SE = (+1, 0)   → our (+1, 0)    horizontal +
    1 NE = ( 0,+1)   → our ( 0,-1)    vertical -
    2 N  = (-1,+1)   → our (-1,-1)    diagonal - (matching-sign)
    3 NW = (-1, 0)   → our (-1, 0)    horizontal -
    4 SW = ( 0,-1)   → our ( 0,+1)    vertical +
    5 S  = (+1,-1)   → our (+1,+1)    diagonal + (matching-sign)

The reverse map (`OUR_DELTA_TO_YNGINE_DIR`) is what makes
`move_to_wire(REMOVE_MARKERS)` work — we infer the row's direction from
`markers[1] - markers[0]`.

Wire format mirrors what the corpus generator emits (see
`scripts/yngine_corpus_to_npz.py`):

    P x y                    place ring
    M fx fy tx ty dir        ring move
    R fx fy dir              remove 5-marker row
    X x y                    remove ring
    S                        pass
"""

from __future__ import annotations

from typing import Tuple

from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.types import Move, MoveType


YNGINE_DIR_VEC: dict[int, Tuple[int, int]] = {
    0: (+1,  0),   # SE
    1: ( 0, +1),   # NE
    2: (-1, +1),   # N
    3: (-1,  0),   # NW
    4: ( 0, -1),   # SW
    5: (+1, -1),   # S
}

# Our (dcol, drow) → yngine direction index. Mirrors y: (dx, dy) → (dx, -dy).
OUR_DELTA_TO_YNGINE_DIR: dict[Tuple[int, int], int] = {
    (+1,  0): 0,   # SE
    ( 0, -1): 1,   # NE   (our drow = -1 ⇔ yngine dy = +1)
    (-1, -1): 2,   # N
    (-1,  0): 3,   # NW
    ( 0, +1): 4,   # SW
    (+1, +1): 5,   # S
}


def pos_to_xy(pos: Position) -> Tuple[int, int]:
    """Our Position → yngine (x, y)."""
    return (ord(pos.column) - ord('A'), 11 - pos.row)


def xy_to_pos(x: int, y: int) -> Position:
    """yngine (x, y) → our Position. Raises ValueError on out-of-range."""
    if not (0 <= x <= 10 and 0 <= y <= 10):
        raise ValueError(f"yngine (x, y) out of range: ({x}, {y})")
    return Position(chr(ord('A') + x), 11 - y)


def _row_direction_index(markers) -> int:
    """Infer the yngine direction index from the first two markers.

    YINSH rows are 5 consecutive markers on a hex line. `markers[1] -
    markers[0]` gives the step along that line in our coords; map it to a
    yngine direction. Raises if the delta isn't a hex direction.
    """
    if markers is None or len(markers) < 2:
        raise ValueError(f"REMOVE_MARKERS needs ≥2 markers, got {markers!r}")
    m0, m1 = markers[0], markers[1]
    dcol = (ord(m1.column) - ord('A')) - (ord(m0.column) - ord('A'))
    drow = m1.row - m0.row
    if (dcol, drow) not in OUR_DELTA_TO_YNGINE_DIR:
        raise ValueError(
            f"REMOVE_MARKERS delta ({dcol}, {drow}) is not a hex direction "
            f"— markers {markers[0]} → {markers[1]}"
        )
    return OUR_DELTA_TO_YNGINE_DIR[(dcol, drow)]


def move_to_wire(mv: Move) -> str:
    """Our Move → yngine driver wire format (no trailing newline)."""
    if mv.type == MoveType.PLACE_RING:
        x, y = pos_to_xy(mv.source)
        return f"P {x} {y}"
    if mv.type == MoveType.MOVE_RING:
        fx, fy = pos_to_xy(mv.source)
        tx, ty = pos_to_xy(mv.destination)
        # Infer the ring's slide direction from source → destination delta
        # so yngine's RingMove carries the same direction tag as upstream.
        dcol = (ord(mv.destination.column) - ord('A')) - (ord(mv.source.column) - ord('A'))
        drow = mv.destination.row - mv.source.row
        steps = max(abs(dcol), abs(drow))
        if steps == 0:
            raise ValueError(f"MOVE_RING with zero displacement: {mv}")
        step = (dcol // steps, drow // steps)
        if step not in OUR_DELTA_TO_YNGINE_DIR:
            raise ValueError(
                f"MOVE_RING step {step} from {mv.source}→{mv.destination} "
                f"is not a hex direction"
            )
        d = OUR_DELTA_TO_YNGINE_DIR[step]
        return f"M {fx} {fy} {tx} {ty} {d}"
    if mv.type == MoveType.REMOVE_MARKERS:
        fx, fy = pos_to_xy(mv.markers[0])
        d = _row_direction_index(mv.markers)
        return f"R {fx} {fy} {d}"
    if mv.type == MoveType.REMOVE_RING:
        x, y = pos_to_xy(mv.source)
        return f"X {x} {y}"
    raise ValueError(f"unhandled move type: {mv.type}")


def wire_to_move(line: str, player: Player) -> Move:
    """yngine driver wire format → our Move.

    `player` is supplied by the caller because the wire format is
    color-agnostic. For REMOVE_MARKERS the marker tuple is filled by
    walking yngine's direction vector for 5 steps from (fx, fy).
    """
    parts = line.strip().split()
    if not parts:
        raise ValueError(f"empty wire move: {line!r}")
    kind = parts[0]
    if kind == "P":
        _, x, y = parts
        return Move(type=MoveType.PLACE_RING, player=player,
                    source=xy_to_pos(int(x), int(y)))
    if kind == "M":
        _, fx, fy, tx, ty, _d = parts
        return Move(type=MoveType.MOVE_RING, player=player,
                    source=xy_to_pos(int(fx), int(fy)),
                    destination=xy_to_pos(int(tx), int(ty)))
    if kind == "R":
        _, fx, fy, d = parts
        dx, dy = YNGINE_DIR_VEC[int(d)]
        markers = tuple(
            xy_to_pos(int(fx) + i * dx, int(fy) + i * dy) for i in range(5)
        )
        return Move(type=MoveType.REMOVE_MARKERS, player=player, markers=markers)
    if kind == "X":
        _, x, y = parts
        return Move(type=MoveType.REMOVE_RING, player=player,
                    source=xy_to_pos(int(x), int(y)))
    if kind == "S":
        # PassMove — our codebase has no pass move type; surface it as None.
        # Callers should not normally hit this in eval; treat as an
        # error-ish signal that the position is degenerate.
        raise ValueError(
            "yngine returned PassMove (S) — our engine has no equivalent."
        )
    raise ValueError(f"unknown wire move kind: {kind!r}")
