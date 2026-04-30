"""Parity test: C++ Zobrist vs Python ZobristHasher.

The C++ side is fed the Python-generated table at construction time, so
correctness reduces to: does the C++ XOR sequence produce the same
running hash as Python's? We sample random states and compare, plus
spot-check the incremental update primitives (Toggle, FlipSide).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import Position, Player, PieceType, VALID_POSITIONS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase
from yinsh_ml.game.zobrist import (
    DEFAULT_PIECE_ORDER,
    ZobristInitializer,
    ZobristHasher,
)
from yinsh_ml.game_cpp import _engine


COLS = "ABCDEFGHIJK"


def _cell_index(pos: Position) -> int:
    col_idx = ord(pos.column) - ord("A")
    return _engine.cell_index(col_idx, pos.row)


def _build_cpp_zobrist(initializer: ZobristInitializer) -> _engine.Zobrist:
    """Reorder Python's per-position table into cell-index × piece order
    and stand up a C++ Zobrist over it."""
    cell_count = _engine.CELL_COUNT
    piece_count = len(DEFAULT_PIECE_ORDER)
    flat = np.zeros(cell_count * piece_count, dtype=np.uint64)
    for pos in initializer.positions:
        cell = _cell_index(pos)
        for piece_idx, piece in enumerate(DEFAULT_PIECE_ORDER):
            flat[cell * piece_count + piece_idx] = initializer.value(pos, piece)
    phase_keys = np.array(initializer.phase_keys, dtype=np.uint64)
    return _engine.Zobrist(flat, initializer.side_to_move_key, phase_keys)


def _bitboards_from_board(board: Board) -> tuple[int, int, int, int]:
    """Return (white_rings, black_rings, white_markers, black_markers) as 128-bit ints."""
    wr = br = wm = bm = 0
    for pos, piece in board.pieces.items():
        b = 1 << _cell_index(pos)
        if piece == PieceType.WHITE_RING:    wr |= b
        elif piece == PieceType.BLACK_RING:  br |= b
        elif piece == PieceType.WHITE_MARKER: wm |= b
        elif piece == PieceType.BLACK_MARKER: bm |= b
    return wr, br, wm, bm


def _split(b: int) -> tuple[int, int]:
    return b & ((1 << 64) - 1), b >> 64


def _random_board(rng: random.Random, n_pieces: int) -> Board:
    board = Board()
    cells = [Position(c, r) for c in COLS for r in VALID_POSITIONS[c]]
    rng.shuffle(cells)
    types = [PieceType.WHITE_RING, PieceType.BLACK_RING,
             PieceType.WHITE_MARKER, PieceType.BLACK_MARKER]
    for pos in cells[:n_pieces]:
        board.place_piece(pos, rng.choice(types))
    return board


def test_empty_hash_matches() -> None:
    init = ZobristInitializer(seed="parity-empty")
    py_hasher = ZobristHasher(init.table)
    cpp_zob = _build_cpp_zobrist(init)
    assert cpp_zob.empty_hash == py_hasher.empty_board_hash


@pytest.mark.parametrize("n_pieces", [1, 5, 20, 50])
def test_hash_board_parity(n_pieces: int) -> None:
    init = ZobristInitializer(seed=f"parity-board-{n_pieces}")
    py_hasher = ZobristHasher(init.table)
    cpp_zob = _build_cpp_zobrist(init)

    rng = random.Random(0xB0A4D + n_pieces)
    for _ in range(100):
        board = _random_board(rng, n_pieces)
        py_hash = py_hasher.hash_board(board)

        wr, br, wm, bm = _bitboards_from_board(board)
        wr_lo, wr_hi = _split(wr)
        br_lo, br_hi = _split(br)
        wm_lo, wm_hi = _split(wm)
        bm_lo, bm_hi = _split(bm)
        cpp_hash = cpp_zob.hash_board(wr_lo, wr_hi, br_lo, br_hi,
                                      wm_lo, wm_hi, bm_lo, bm_hi)
        assert cpp_hash == py_hash, (
            f"hash_board mismatch (n={n_pieces}): "
            f"py={py_hash:016x} cpp={cpp_hash:016x}"
        )


def test_hash_state_parity_all_phases_and_players() -> None:
    """Side-to-move and phase keys are the load-bearing fix from the
    April 2026 audit. Hammer them: every phase × every player × random
    boards must match."""
    init = ZobristInitializer(seed="parity-state")
    py_hasher = ZobristHasher(init.table)
    cpp_zob = _build_cpp_zobrist(init)

    rng = random.Random(0x57A7E)
    for _ in range(50):
        gs = GameState()
        gs.board = _random_board(rng, rng.randint(5, 30))
        for player in (Player.WHITE, Player.BLACK):
            for phase in GamePhase:
                gs.current_player = player
                gs.phase = phase

                py_hash = py_hasher.hash_state(gs)

                wr, br, wm, bm = _bitboards_from_board(gs.board)
                wr_lo, wr_hi = _split(wr)
                br_lo, br_hi = _split(br)
                wm_lo, wm_hi = _split(wm)
                bm_lo, bm_hi = _split(bm)
                cpp_hash = cpp_zob.hash_state(
                    wr_lo, wr_hi, br_lo, br_hi,
                    wm_lo, wm_hi, bm_lo, bm_hi,
                    current_player_is_black=(player == Player.BLACK),
                    phase_idx=phase.value,
                )
                assert cpp_hash == py_hash, (
                    f"hash_state mismatch (player={player}, phase={phase}): "
                    f"py={py_hash:016x} cpp={cpp_hash:016x}"
                )


def test_incremental_toggle_matches() -> None:
    """C++ Toggle / UpdatePosition should compose into the same hash a
    full re-hash would produce. This is what HeuristicAgent's
    transposition table relies on for incremental updates."""
    init = ZobristInitializer(seed="parity-toggle")
    py_hasher = ZobristHasher(init.table)
    cpp_zob = _build_cpp_zobrist(init)

    rng = random.Random(0x10661)
    for _ in range(50):
        board = _random_board(rng, rng.randint(5, 25))
        wr, br, wm, bm = _bitboards_from_board(board)
        wr_lo, wr_hi = _split(wr)
        br_lo, br_hi = _split(br)
        wm_lo, wm_hi = _split(wm)
        bm_lo, bm_hi = _split(bm)
        h0 = cpp_zob.hash_board(wr_lo, wr_hi, br_lo, br_hi,
                                wm_lo, wm_hi, bm_lo, bm_hi)

        # Pick an empty cell; toggle EMPTY → WHITE_MARKER and assert
        # this matches placing the marker on the board and re-hashing.
        empties = [
            Position(c, r)
            for c in COLS for r in VALID_POSITIONS[c]
            if board.is_empty(Position(c, r))
        ]
        if not empties:
            continue
        pos = rng.choice(empties)
        cell = _cell_index(pos)
        h_inc = cpp_zob.update_position(h0, cell, 0, 3)  # EMPTY=0, W_MARKER=3

        board.place_piece(pos, PieceType.WHITE_MARKER)
        h_full = py_hasher.hash_board(board)
        assert h_inc == h_full, (
            f"incremental toggle != full rehash: "
            f"inc={h_inc:016x} full={h_full:016x} (placed {pos})"
        )

        # FlipSide is its own inverse.
        flipped = cpp_zob.flip_side(h0)
        assert cpp_zob.flip_side(flipped) == h0
