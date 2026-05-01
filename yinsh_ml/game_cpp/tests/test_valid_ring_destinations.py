"""Parity + perf test: C++ valid_ring_destinations vs Python Board.valid_move_positions.

This is the load-bearing first slice of the bitboard port. If parity
holds across thousands of random YINSH-shaped boards, the layout +
ray-table + ring-walk semantics in tables.hpp are correct, and the
rest of the engine port can build on this foundation.

The benchmark print-out exists so we can paste a perf number into the
PR description; the assertions are what actually gate this test.
"""
from __future__ import annotations

import random

import pytest

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import (
    Position,
    PieceType,
    VALID_POSITIONS,
)
from yinsh_ml.game_cpp import _engine


COLS = "ABCDEFGHIJK"


def _cell_index(pos: Position) -> int:
    col_idx = ord(pos.column) - ord("A")
    return _engine.cell_index(col_idx, pos.row)


def _board_to_bitboards(board: Board) -> tuple[int, int]:
    """Return (rings_mask, markers_mask) as 128-bit ints, regardless of color."""
    rings = 0
    markers = 0
    for pos, piece in board.pieces.items():
        bit = 1 << _cell_index(pos)
        if piece.is_ring():
            rings |= bit
        elif piece.is_marker():
            markers |= bit
    return rings, markers


def _cpp_valid_dests(source: Position, board: Board) -> set[Position]:
    rings, markers = _board_to_bitboards(board)
    src = _cell_index(source)
    lo, hi = _engine.valid_ring_destinations(
        src,
        rings & ((1 << 64) - 1), rings >> 64,
        markers & ((1 << 64) - 1), markers >> 64,
    )
    dests_mask = lo | (hi << 64)
    out: set[Position] = set()
    for col_idx, col in enumerate(COLS):
        for row in VALID_POSITIONS[col]:
            cell = _engine.cell_index(col_idx, row)
            if dests_mask & (1 << cell):
                out.add(Position(col, row))
    return out


def _random_board(rng: random.Random, n_pieces: int) -> Board:
    """Drop n_pieces randomly typed pieces onto distinct legal cells."""
    board = Board()
    cells = [
        Position(col, row)
        for col in COLS
        for row in VALID_POSITIONS[col]
    ]
    rng.shuffle(cells)
    types = [
        PieceType.WHITE_RING, PieceType.BLACK_RING,
        PieceType.WHITE_MARKER, PieceType.BLACK_MARKER,
    ]
    for pos in cells[:n_pieces]:
        board.place_piece(pos, rng.choice(types))
    return board


@pytest.mark.parametrize("n_pieces", [5, 15, 30, 50])
def test_valid_ring_destinations_parity(n_pieces: int) -> None:
    rng = random.Random(0xBEEF + n_pieces)
    mismatches = 0
    boards_checked = 0

    for _ in range(200):
        board = _random_board(rng, n_pieces)
        # Iterate every ring on the board and compare destination sets.
        for source, piece in list(board.pieces.items()):
            if not piece.is_ring():
                continue
            py_dests = set(board.valid_move_positions(source))
            cpp_dests = _cpp_valid_dests(source, board)
            boards_checked += 1
            if py_dests != cpp_dests:
                mismatches += 1
                # Surface the first failure with detail; pytest prints
                # the diff which is what we want for debugging.
                only_py = py_dests - cpp_dests
                only_cpp = cpp_dests - py_dests
                pytest.fail(
                    f"valid_ring_destinations parity mismatch from {source} "
                    f"(n_pieces={n_pieces}): "
                    f"only_in_py={sorted(map(str, only_py))}, "
                    f"only_in_cpp={sorted(map(str, only_cpp))}\n"
                    f"board:\n{board}"
                )

    assert mismatches == 0, f"{mismatches}/{boards_checked} boards mismatched"


def test_valid_ring_destinations_bench(capsys) -> None:
    """Lock in a perf number against the Python reference.

    The C++ bench rotates through 100 distinct (board, source) pairs
    inside its inner loop so the optimizer can't fold the call body to
    a constant the way it would with a single fixed input. The number
    that comes out is a credible early datapoint for the speedup
    ratio, not a microbench artifact.
    """
    rng = random.Random(0xC0FFEE)
    samples: list[tuple[Position, Board]] = []
    while len(samples) < 100:
        board = _random_board(rng, 30)
        rings = [p for p, piece in board.pieces.items() if piece.is_ring()]
        if rings:
            samples.append((rng.choice(rings), board))

    sources_cells: list[int] = []
    rings_lo: list[int] = []
    rings_hi: list[int] = []
    markers_lo: list[int] = []
    markers_hi: list[int] = []
    for source, board in samples:
        sources_cells.append(_cell_index(source))
        r, m = _board_to_bitboards(board)
        rings_lo.append(r & ((1 << 64) - 1))
        rings_hi.append(r >> 64)
        markers_lo.append(m & ((1 << 64) - 1))
        markers_hi.append(m >> 64)

    iters = 1_000_000
    cpp_secs = _engine.bench_valid_ring_destinations_varied(
        sources_cells, rings_lo, rings_hi, markers_lo, markers_hi, iters,
    )

    # Python side: same workload pattern (rotating through samples) so
    # the comparison is apples-to-apples.
    import time
    py_iters = 100_000
    t0 = time.perf_counter()
    for i in range(py_iters):
        source, board = samples[i % len(samples)]
        board.valid_move_positions(source)
    py_secs = time.perf_counter() - t0

    cpp_per_call_us = 1e6 * cpp_secs / iters
    py_per_call_us = 1e6 * py_secs / py_iters
    speedup = py_per_call_us / cpp_per_call_us

    with capsys.disabled():
        print(
            f"\nvalid_ring_destinations bench (varied inputs, {len(samples)} boards):"
            f"\n  C++ : {cpp_per_call_us:8.3f} us/call ({iters} iters in {cpp_secs:.3f}s)"
            f"\n  Py  : {py_per_call_us:8.3f} us/call ({py_iters} iters in {py_secs:.3f}s)"
            f"\n  C++ vs Py: {speedup:.1f}x"
        )

    assert speedup > 1.0, f"C++ slower than Python? {speedup}x"
