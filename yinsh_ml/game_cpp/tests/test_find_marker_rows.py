"""Parity + perf test: C++ find_marker_rows vs Python Board.find_marker_rows.

Mirrors the parity scaffold in test_valid_ring_destinations.py — same
random-board generator, same comparison strategy. Both Python and C++
report sets of (sorted) cell-tuples; equality on those sets is the
correctness contract.
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


def _index_to_position(cell: int) -> Position:
    col_idx, row_minus = divmod(cell, 11)
    return Position(COLS[col_idx], row_minus + 1)


def _markers_bitboard(board: Board, color: PieceType) -> tuple[int, int]:
    assert color.is_marker()
    mask = 0
    for pos, piece in board.pieces.items():
        if piece == color:
            mask |= 1 << _cell_index(pos)
    return mask & ((1 << 64) - 1), mask >> 64


def _cpp_runs(board: Board, color: PieceType) -> set[tuple[Position, ...]]:
    lo, hi = _markers_bitboard(board, color)
    runs = _engine.find_marker_rows(lo, hi)
    return {
        tuple(sorted((_index_to_position(c) for c in run),
                     key=lambda p: (p.column, p.row)))
        for run in runs
    }


def _py_runs(board: Board, color: PieceType) -> set[tuple[Position, ...]]:
    rows = board.find_marker_rows(color)
    return {
        tuple(sorted(row.positions, key=lambda p: (p.column, p.row)))
        for row in rows
    }


def _planted_run_board(rng: random.Random, color: PieceType) -> Board:
    """Build a board guaranteed to contain at least one same-color run.

    Random sparse boards rarely hit length-5 runs by chance, so we plant
    a 5/6/7-run on a random forward axis and noise the rest.
    """
    board = Board()
    # Forward axes: vertical(0,+1), horizontal(+1,0), diagonal(+1,+1).
    axes = [(0, 1), (1, 0), (1, 1)]
    dx, dy = rng.choice(axes)
    length = rng.choice([5, 6, 7])

    # Find a starting cell that lets us walk `length` steps without
    # falling off the board.
    valid_cells = [
        (col_idx, row)
        for col_idx, col in enumerate(COLS)
        for row in VALID_POSITIONS[col]
    ]
    rng.shuffle(valid_cells)
    for c0, r0 in valid_cells:
        cells = [(c0 + i * dx, r0 + i * dy) for i in range(length)]
        if all(
            0 <= c < 11 and 1 <= r <= 11 and r in VALID_POSITIONS[COLS[c]]
            for c, r in cells
        ):
            for c, r in cells:
                board.place_piece(Position(COLS[c], r), color)
            break
    else:
        # Couldn't fit a run; fall back to a sparse board.
        return board

    # Sprinkle noise that doesn't extend the run on the same axis (stay
    # off the cells immediately before/after the planted run).
    other = (PieceType.BLACK_MARKER if color == PieceType.WHITE_MARKER
             else PieceType.WHITE_MARKER)
    for col_idx, row in valid_cells[:25]:
        pos = Position(COLS[col_idx], row)
        if board.is_empty(pos):
            board.place_piece(pos, rng.choice([color, other,
                                               PieceType.WHITE_RING,
                                               PieceType.BLACK_RING]))
    return board


def _random_sparse_board(rng: random.Random, n_pieces: int) -> Board:
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


@pytest.mark.parametrize("color", [PieceType.WHITE_MARKER, PieceType.BLACK_MARKER])
def test_find_marker_rows_planted_parity(color: PieceType) -> None:
    """Boards with a guaranteed run hit the length≥5 fast path."""
    rng = random.Random(0xCAFE + (1 if color == PieceType.WHITE_MARKER else 2))
    for _ in range(200):
        board = _planted_run_board(rng, color)
        py = _py_runs(board, color)
        cpp = _cpp_runs(board, color)
        if py != cpp:
            only_py = py - cpp
            only_cpp = cpp - py
            pytest.fail(
                f"find_marker_rows parity mismatch ({color.name}):\n"
                f"  only in Py:  {[tuple(map(str, r)) for r in only_py]}\n"
                f"  only in C++: {[tuple(map(str, r)) for r in only_cpp]}\n"
                f"board:\n{board}"
            )


@pytest.mark.parametrize("n_pieces", [10, 30, 60])
def test_find_marker_rows_sparse_parity(n_pieces: int) -> None:
    """Sparse boards exercise the no-run path and edge cases."""
    rng = random.Random(0xDEAD + n_pieces)
    for _ in range(200):
        board = _random_sparse_board(rng, n_pieces)
        for color in (PieceType.WHITE_MARKER, PieceType.BLACK_MARKER):
            py = _py_runs(board, color)
            cpp = _cpp_runs(board, color)
            assert py == cpp, (
                f"sparse parity mismatch ({color.name}, n={n_pieces}): "
                f"only_py={py - cpp}, only_cpp={cpp - py}"
            )


def test_find_marker_rows_bench(capsys) -> None:
    """C++ vs Python on a varied workload of planted-run boards."""
    rng = random.Random(0xC0DE)
    samples: list[tuple[Board, PieceType]] = []
    while len(samples) < 100:
        color = rng.choice([PieceType.WHITE_MARKER, PieceType.BLACK_MARKER])
        board = _planted_run_board(rng, color)
        samples.append((board, color))

    markers_lo: list[int] = []
    markers_hi: list[int] = []
    for board, color in samples:
        lo, hi = _markers_bitboard(board, color)
        markers_lo.append(lo)
        markers_hi.append(hi)

    iters = 500_000
    cpp_secs = _engine.bench_find_marker_rows(markers_lo, markers_hi, iters)

    import time
    py_iters = 50_000
    t0 = time.perf_counter()
    for i in range(py_iters):
        board, color = samples[i % len(samples)]
        board.find_marker_rows(color)
    py_secs = time.perf_counter() - t0

    cpp_per_call_us = 1e6 * cpp_secs / iters
    py_per_call_us = 1e6 * py_secs / py_iters
    speedup = py_per_call_us / cpp_per_call_us

    with capsys.disabled():
        print(
            f"\nfind_marker_rows bench (varied inputs, {len(samples)} boards):"
            f"\n  C++ : {cpp_per_call_us:8.3f} us/call ({iters} iters in {cpp_secs:.3f}s)"
            f"\n  Py  : {py_per_call_us:8.3f} us/call ({py_iters} iters in {py_secs:.3f}s)"
            f"\n  C++ vs Py: {speedup:.1f}x"
        )

    assert speedup > 1.0, f"C++ slower than Python? {speedup}x"
