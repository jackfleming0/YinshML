"""Pure-Python tests for the yngine ↔ ours move/coordinate translation.

These run without the yngine_driver binary — they only exercise the codec.
The subprocess-level tests are in test_bridge.py and require the build.
"""

from __future__ import annotations

import pytest

from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.yngine.move_codec import (
    OUR_DELTA_TO_YNGINE_DIR,
    YNGINE_DIR_VEC,
    move_to_wire,
    pos_to_xy,
    wire_to_move,
    xy_to_pos,
)


def test_pos_xy_roundtrip_covers_all_valid_cells():
    # All 85 valid cells: column → row range, mirror via 11 - row.
    valid_columns = {
        'A': range(2, 6), 'B': range(1, 8), 'C': range(1, 9),
        'D': range(1, 10), 'E': range(1, 11), 'F': range(2, 11),
        'G': range(2, 12), 'H': range(3, 12), 'I': range(4, 12),
        'J': range(5, 12), 'K': range(7, 11),
    }
    for col, rows in valid_columns.items():
        for row in rows:
            pos = Position(col, row)
            x, y = pos_to_xy(pos)
            assert xy_to_pos(x, y) == pos


def test_yngine_dir_vec_matches_inverse_table():
    # Every entry in YNGINE_DIR_VEC must round-trip through our reverse map.
    for d, (dx, dy) in YNGINE_DIR_VEC.items():
        our_delta = (dx, -dy)
        assert OUR_DELTA_TO_YNGINE_DIR[our_delta] == d


def test_place_ring_wire_format():
    mv = Move(type=MoveType.PLACE_RING, player=Player.WHITE,
              source=Position('E', 5))
    # E=4, row 5 → y = 11 - 5 = 6
    assert move_to_wire(mv) == "P 4 6"
    parsed = wire_to_move("P 4 6", Player.WHITE)
    assert parsed == mv


def test_move_ring_wire_format_uses_inferred_direction():
    # Slide from E5 → E8 (vertical +3): drow = +3, our (0, +1) per step,
    # which maps to yngine direction 4 (SW).
    mv = Move(type=MoveType.MOVE_RING, player=Player.BLACK,
              source=Position('E', 5), destination=Position('E', 8))
    wire = move_to_wire(mv)
    parts = wire.split()
    assert parts[0] == "M"
    assert parts[1:5] == ["4", "6", "4", "3"]   # (E5)=(4,6) → (E8)=(4,3)
    assert parts[5] == "4"   # SW
    parsed = wire_to_move(wire, Player.BLACK)
    assert parsed == mv


def test_remove_ring_wire_format():
    mv = Move(type=MoveType.REMOVE_RING, player=Player.WHITE,
              source=Position('A', 5))
    # A=0, row 5 → y = 6
    assert move_to_wire(mv) == "X 0 6"
    parsed = wire_to_move("X 0 6", Player.WHITE)
    assert parsed == mv


def test_remove_markers_horizontal_row():
    # Horizontal row B5..F5 (5 markers along columns +1, row constant).
    # Our delta per step = (+1, 0) → yngine direction 0 (SE).
    markers = tuple(Position(c, 5) for c in 'BCDEF')
    mv = Move(type=MoveType.REMOVE_MARKERS, player=Player.WHITE,
              markers=markers)
    wire = move_to_wire(mv)
    # markers[0] = B5 → (1, 6), direction 0
    assert wire == "R 1 6 0"
    parsed = wire_to_move(wire, Player.WHITE)
    assert parsed == mv


def test_remove_markers_diagonal_matching_sign():
    # Matching-sign diagonal: B2, C3, D4, E5, F6 — delta (+1, +1) per step.
    # Maps to yngine direction 5 (S).
    markers = (Position('B', 2), Position('C', 3), Position('D', 4),
               Position('E', 5), Position('F', 6))
    mv = Move(type=MoveType.REMOVE_MARKERS, player=Player.BLACK,
              markers=markers)
    wire = move_to_wire(mv)
    # B2 → x=1, y=9. Direction 5.
    assert wire == "R 1 9 5"
    parsed = wire_to_move(wire, Player.BLACK)
    assert parsed == mv


def test_remove_markers_reversed_row_yields_opposite_direction():
    # If markers are listed in reverse order, we get the opposite direction.
    # yngine's RemoveRowMove::operator== accepts both forms, so this is fine
    # at the bridge layer.
    forward = (Position('B', 5), Position('C', 5), Position('D', 5),
               Position('E', 5), Position('F', 5))
    reversed_markers = tuple(reversed(forward))
    mv_fwd = Move(type=MoveType.REMOVE_MARKERS, player=Player.WHITE,
                  markers=forward)
    mv_rev = Move(type=MoveType.REMOVE_MARKERS, player=Player.WHITE,
                  markers=reversed_markers)
    assert move_to_wire(mv_fwd) == "R 1 6 0"   # SE
    assert move_to_wire(mv_rev) == "R 5 6 3"   # NW (opposite of SE)


def test_invalid_direction_raises():
    # A non-hex delta should raise.
    markers = (Position('A', 2), Position('B', 4))   # delta (+1, +2)
    bad = Move(type=MoveType.REMOVE_MARKERS, player=Player.WHITE,
               markers=(*markers, Position('C', 6), Position('D', 8), Position('E', 10)))
    with pytest.raises(ValueError, match="hex direction"):
        move_to_wire(bad)


def test_xy_to_pos_out_of_range_raises():
    with pytest.raises(ValueError):
        xy_to_pos(11, 0)
    with pytest.raises(ValueError):
        xy_to_pos(0, -1)


def test_pass_wire_raises_on_decode():
    # We never expect yngine to return PassMove in real games (post-placement
    # YINSH has plenty of legal moves), but make sure we hard-fail rather
    # than silently produce a bad Move.
    import pytest as _pytest
    with _pytest.raises(ValueError, match="PassMove"):
        wire_to_move("S", Player.WHITE)
