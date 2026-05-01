"""Parity test for cell_to_position lookup-table optimization.

BITBOARD_FOLLOWUP_PLAN.md Candidate C-1 replaces the divmod-based
cell_to_position with a 121-slot precomputed Position table. The
correctness gate: across all 121 cell indices the new lookup must
return a Position equal to what the legacy divmod implementation
produced.

Skips when the C++ engine isn't built, so local Macs don't fail
collection. The cloud (where _engine is compiled) is the gating
environment.
"""
from __future__ import annotations

import pytest

from yinsh_ml.game.constants import Position

try:
    from yinsh_ml.game_cpp._convert import (
        cell_to_position,
        _POSITION_BY_CELL,
        COLS,
    )
except ImportError:  # _engine not built
    pytest.skip("yinsh_ml.game_cpp._engine not built", allow_module_level=True)


def _legacy_cell_to_position(cell: int) -> Position:
    """Verbatim re-implementation of the pre-Candidate-C-1 logic."""
    col_idx, row_minus = divmod(cell, 11)
    return Position(COLS[col_idx], row_minus + 1)


class TestCellToPositionParity:
    def test_lookup_matches_legacy_across_all_121_cells(self):
        for cell in range(121):
            self_assert_eq = (cell_to_position(cell), _legacy_cell_to_position(cell))
            assert self_assert_eq[0] == self_assert_eq[1], (
                f"cell {cell}: lookup={self_assert_eq[0]!r} legacy={self_assert_eq[1]!r}"
            )

    def test_lookup_returns_shared_instance(self):
        """Frozen-dataclass invariant: the table returns the same Position
        object on every call so callers don't allocate new instances."""
        p1 = cell_to_position(42)
        p2 = cell_to_position(42)
        assert p1 is p2

    def test_table_has_exactly_121_entries(self):
        assert len(_POSITION_BY_CELL) == 121

    def test_position_to_cell_round_trip(self):
        """For valid YINSH cells, position_to_cell(cell_to_position(c))
        should equal c. Catches off-by-one bugs in the lookup index math."""
        from yinsh_ml.game.constants import VALID_POSITIONS
        from yinsh_ml.game_cpp._convert import position_to_cell

        for col_idx, col in enumerate(COLS):
            for row in VALID_POSITIONS[col]:
                cell = position_to_cell(Position(col, row))
                pos_back = cell_to_position(cell)
                assert pos_back == Position(col, row), (
                    f"cell {cell} ({col}{row}): round-trip gave {pos_back}"
                )
