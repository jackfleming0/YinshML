"""Validate the State struct + clone primitive — and bench it against
copy.deepcopy(GameState), which is 95% of profiled MCTS self-play time
in the Python engine. This is the load-bearing motivator for the
bitboard port from the MCTS side.
"""
from __future__ import annotations

import copy
import random
import time

import pytest

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import Position, Player, PieceType, VALID_POSITIONS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase
from yinsh_ml.game_cpp import _engine


COLS = "ABCDEFGHIJK"


def _cell_index(pos: Position) -> int:
    col_idx = ord(pos.column) - ord("A")
    return _engine.cell_index(col_idx, pos.row)


def _split(b: int) -> tuple[int, int]:
    return b & ((1 << 64) - 1), b >> 64


def _populated_cpp_state() -> _engine.State:
    """Build a State with non-default values across every field so the
    clone test can detect any field that fails to copy."""
    s = _engine.State()
    # ~30 pieces scattered across the board.
    rng = random.Random(0x517A7E)
    cells = [Position(c, r) for c in COLS for r in VALID_POSITIONS[c]]
    rng.shuffle(cells)
    wr = br = wm = bm = 0
    types = [PieceType.WHITE_RING, PieceType.BLACK_RING,
             PieceType.WHITE_MARKER, PieceType.BLACK_MARKER]
    for pos in cells[:30]:
        b = 1 << _cell_index(pos)
        t = rng.choice(types)
        if t == PieceType.WHITE_RING:    wr |= b
        elif t == PieceType.BLACK_RING:  br |= b
        elif t == PieceType.WHITE_MARKER: wm |= b
        else:                            bm |= b
    s.white_rings = _split(wr)
    s.black_rings = _split(br)
    s.white_markers = _split(wm)
    s.black_markers = _split(bm)
    s.phase = GamePhase.ROW_COMPLETION.value
    s.current_player_is_black = True
    s.white_score = 1
    s.black_score = 2
    s.white_rings_placed = 5
    s.black_rings_placed = 5
    s.move_maker_is_black = 0
    return s


def test_clone_preserves_all_fields() -> None:
    src = _populated_cpp_state()
    dup = src.clone()

    # Equality predicate (compiled-side memcmp).
    assert dup.equals(src)

    # And every readable field independently, in case `equals` regresses.
    for attr in ("white_rings", "black_rings", "white_markers", "black_markers"):
        assert getattr(dup, attr) == getattr(src, attr), attr
    for attr in ("phase", "current_player_is_black",
                 "white_score", "black_score",
                 "white_rings_placed", "black_rings_placed",
                 "move_maker_is_black"):
        assert getattr(dup, attr) == getattr(src, attr), attr


def test_clone_is_independent() -> None:
    """Mutating the clone must not affect the source."""
    src = _populated_cpp_state()
    dup = src.clone()
    dup.phase = GamePhase.GAME_OVER.value
    dup.white_score = 99
    dup.white_rings = (0, 0)
    assert src.phase == GamePhase.ROW_COMPLETION.value
    assert src.white_score == 1
    assert src.white_rings != (0, 0)


def _populated_python_state() -> GameState:
    """A GameState equivalent in spirit to _populated_cpp_state — same
    rough piece density, mid-game phase. Fairness over isomorphism;
    the bench is about the cost of *cloning the representation*, not
    semantic equivalence."""
    rng = random.Random(0x517A7E)
    gs = GameState()
    cells = [Position(c, r) for c in COLS for r in VALID_POSITIONS[c]]
    rng.shuffle(cells)
    types = [PieceType.WHITE_RING, PieceType.BLACK_RING,
             PieceType.WHITE_MARKER, PieceType.BLACK_MARKER]
    for pos in cells[:30]:
        gs.board.place_piece(pos, rng.choice(types))
    gs.phase = GamePhase.ROW_COMPLETION
    gs.current_player = Player.BLACK
    gs.white_score = 1
    gs.black_score = 2
    gs.rings_placed = {Player.WHITE: 5, Player.BLACK: 5}
    return gs


def test_clone_bench(capsys) -> None:
    """C++ State.clone() vs copy.deepcopy(GameState).

    This is the second motivator from the brief: deepcopy is 95% of
    MCTS self-play profile, and a 64-byte struct copy is what
    replaces it. The number here predicts the MCTS-side speedup
    ceiling.
    """
    cpp_state = _populated_cpp_state()
    py_state = _populated_python_state()

    iters_cpp = 5_000_000
    cpp_secs = _engine.bench_clone_state(cpp_state, iters_cpp)

    iters_py = 50_000  # deepcopy is slow; 5M would take minutes
    t0 = time.perf_counter()
    for _ in range(iters_py):
        copy.deepcopy(py_state)
    py_secs = time.perf_counter() - t0

    cpp_per_call_us = 1e6 * cpp_secs / iters_cpp
    py_per_call_us = 1e6 * py_secs / iters_py
    speedup = py_per_call_us / cpp_per_call_us

    with capsys.disabled():
        print(
            f"\nState clone bench:"
            f"\n  C++ State.clone()       : {cpp_per_call_us:9.4f} us/call ({iters_cpp} iters in {cpp_secs:.3f}s)"
            f"\n  Py copy.deepcopy(GameState): {py_per_call_us:9.4f} us/call ({iters_py} iters in {py_secs:.3f}s)"
            f"\n  C++ vs Py: {speedup:.1f}x"
        )

    assert speedup > 1.0
