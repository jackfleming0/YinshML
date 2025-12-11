import numpy as np
import pytest

from yinsh_ml.game.constants import PieceType, VALID_POSITIONS
from yinsh_ml.game.zobrist import DEFAULT_PIECE_ORDER, ZobristInitializer


def test_position_enumeration_matches_valid_board():
    initializer = ZobristInitializer(seed="test-seed")
    expected_count = sum(len(rows) for rows in VALID_POSITIONS.values())
    assert len(initializer.positions) == expected_count
    assert len(set(initializer.positions)) == expected_count


def test_seed_reproducibility():
    initializer_a = ZobristInitializer(seed="sync")
    initializer_b = ZobristInitializer(seed="sync")

    np.testing.assert_array_equal(initializer_a.table.values, initializer_b.table.values)


def test_different_seeds_produce_distinct_tables():
    initializer_a = ZobristInitializer(seed="alpha")
    initializer_b = ZobristInitializer(seed="beta")

    assert not np.array_equal(initializer_a.table.values, initializer_b.table.values)


def test_all_bitstrings_are_unique():
    initializer = ZobristInitializer(seed="collision-check")
    seen = set()
    for value in initializer.table.values.flatten():
        assert value not in seen
        seen.add(int(value))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_bitstrings_are_reasonably_uniform():
    initializer = ZobristInitializer(seed="uniformity")
    values = initializer.table.values.flatten()

    msb = 1 << 63
    ones_ratio = np.count_nonzero(values & msb) / values.size

    assert 0.35 < ones_ratio < 0.65, "MSB should appear roughly half the time"

