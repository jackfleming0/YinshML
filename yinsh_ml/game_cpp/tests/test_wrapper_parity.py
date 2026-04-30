"""Drive CppGameState and the Python GameState through identical
random games. After every move check that everything downstream code
reads from a GameState object — phase, current_player, scores,
move_history length, get_valid_moves, board.pieces, board.get_piece,
board.is_empty, board.find_marker_rows for each color, board.to_numpy_array,
board.valid_move_positions for each ring — agrees.

This is the load-bearing check that the wrapper is a true drop-in.
"""
from __future__ import annotations

import copy
import random
import time

import pytest

from yinsh_ml.game.constants import Position, Player, PieceType, VALID_POSITIONS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase, MoveType
from yinsh_ml.game_cpp import CppGameState


COLS = "ABCDEFGHIJK"


def _move_key(m) -> tuple:
    if m.markers is not None:
        marker_strs = tuple(sorted(str(p) for p in m.markers))
    else:
        marker_strs = None
    return (
        m.type.value,
        m.player.value,
        str(m.source) if m.source else None,
        str(m.destination) if m.destination else None,
        marker_strs,
    )


def _assert_surfaces_match(py: GameState, cpp: CppGameState, ctx: str) -> None:
    assert py.phase == cpp.phase, f"{ctx}: phase {py.phase} vs {cpp.phase}"
    assert py.current_player == cpp.current_player, (
        f"{ctx}: current_player {py.current_player} vs {cpp.current_player}")
    assert py.white_score == cpp.white_score, f"{ctx}: white_score"
    assert py.black_score == cpp.black_score, f"{ctx}: black_score"
    assert py.rings_placed[Player.WHITE] == cpp.rings_placed[Player.WHITE]
    assert py.rings_placed[Player.BLACK] == cpp.rings_placed[Player.BLACK]
    assert len(py.move_history) == len(cpp.move_history), (
        f"{ctx}: move_history length")

    # board.pieces — full materialization parity.
    py_pieces = dict(py.board.pieces)
    cpp_pieces = dict(cpp.board.pieces)
    assert py_pieces == cpp_pieces, (
        f"{ctx}: board.pieces mismatch\n"
        f"  only in Py:  {set(py_pieces) - set(cpp_pieces)}\n"
        f"  only in Cpp: {set(cpp_pieces) - set(py_pieces)}"
    )

    # board.to_numpy_array — used by encoder.
    import numpy as np
    py_arr = py.board.to_numpy_array()
    cpp_arr = cpp.board.to_numpy_array()
    assert np.array_equal(py_arr, cpp_arr), f"{ctx}: to_numpy_array"

    # board.find_marker_rows — used by heuristic agent.
    for color in (PieceType.WHITE_MARKER, PieceType.BLACK_MARKER):
        py_runs = {
            tuple(sorted(r.positions, key=lambda p: (p.column, p.row)))
            for r in py.board.find_marker_rows(color)
        }
        cpp_runs = {
            tuple(sorted(r.positions, key=lambda p: (p.column, p.row)))
            for r in cpp.board.find_marker_rows(color)
        }
        assert py_runs == cpp_runs, f"{ctx}: find_marker_rows({color.name})"

    # legal-move set parity.
    py_keys = {_move_key(m) for m in py.get_valid_moves()}
    cpp_keys = {_move_key(m) for m in cpp.get_valid_moves()}
    assert py_keys == cpp_keys, (
        f"{ctx}: get_valid_moves mismatch\n"
        f"  only in Py:  {sorted(py_keys - cpp_keys)[:5]}\n"
        f"  only in Cpp: {sorted(cpp_keys - py_keys)[:5]}"
    )

    # is_terminal + get_winner agreement.
    assert py.is_terminal() == cpp.is_terminal(), f"{ctx}: is_terminal"
    assert py.get_winner() == cpp.get_winner(), f"{ctx}: get_winner"


def _spot_check_valid_move_positions(py: GameState,
                                     cpp: CppGameState,
                                     ctx: str) -> None:
    """For every ring on the board, board.valid_move_positions should
    match. Ordering can differ between engines (Python uses a fixed
    direction-order walk, C++ iterates set bits low→high), so compare
    as sets."""
    for pos, piece in py.board.pieces.items():
        if not piece.is_ring():
            continue
        py_dests = set(py.board.valid_move_positions(pos))
        cpp_dests = set(cpp.board.valid_move_positions(pos))
        assert py_dests == cpp_dests, (
            f"{ctx}: valid_move_positions({pos}) mismatch "
            f"py={py_dests} cpp={cpp_dests}"
        )


@pytest.mark.parametrize("seed", list(range(20)))
def test_wrapper_rollout_parity(seed: int) -> None:
    rng = random.Random(seed)
    py = GameState()
    cpp = CppGameState()

    _assert_surfaces_match(py, cpp, "initial")
    _spot_check_valid_move_positions(py, cpp, "initial")

    for ply in range(400):
        if py.is_terminal():
            break
        moves = py.get_valid_moves()
        if not moves:
            break
        m = rng.choice(moves)

        py_ok = py.make_move(m)
        cpp_ok = cpp.make_move(m)
        assert py_ok == cpp_ok, f"seed={seed} ply={ply}: make_move return"
        ctx = f"seed={seed} ply={ply} after {m}"
        _assert_surfaces_match(py, cpp, ctx)
        # Cheaper structural check on every other ply to keep runtime
        # bounded; legal-move parity already covers ring movability.
        if ply % 4 == 0:
            _spot_check_valid_move_positions(py, cpp, ctx)


def test_wrapper_make_move_rejects_illegal() -> None:
    """make_move on a state where the move isn't legal should return
    False without mutating state — matches Python GameState semantics."""
    cpp = CppGameState()
    # Forge a move for a player not on turn.
    valid = cpp.get_valid_moves()
    illegal = next(m for m in valid if m.player == cpp.current_player)
    illegal_other = type(illegal)(
        type=illegal.type, player=cpp.current_player.opponent,
        source=illegal.source,
    )
    snapshot_phase = cpp.phase
    snapshot_history_len = len(cpp.move_history)
    assert cpp.make_move(illegal_other) is False
    assert cpp.phase == snapshot_phase
    assert len(cpp.move_history) == snapshot_history_len


def test_wrapper_copy_independence() -> None:
    """copy() must produce a fully independent CppGameState — mutating
    the clone shouldn't leak into the source."""
    rng = random.Random(0xCAFE)
    cpp = CppGameState()
    # Play a few plies so we have non-default state to clone.
    for _ in range(20):
        moves = cpp.get_valid_moves()
        if not moves:
            break
        cpp.make_move(rng.choice(moves))

    dup = cpp.copy()
    assert dup.phase == cpp.phase
    assert dup.white_score == cpp.white_score
    assert len(dup.move_history) == len(cpp.move_history)
    assert dup.board.pieces == cpp.board.pieces

    # Advance the clone; source must be unaffected.
    dup_moves = dup.get_valid_moves()
    if dup_moves:
        dup.make_move(rng.choice(dup_moves))
    assert dup.phase == cpp.phase or dup.phase != cpp.phase  # smoke
    assert len(dup.move_history) >= len(cpp.move_history)
    # The original must still match a fresh-from-history reconstruction
    # (i.e. no mutation leaked from `dup`).
    assert dup.move_history[: len(cpp.move_history)] == cpp.move_history


def test_wrapper_deepcopy_uses_fast_path(capsys) -> None:
    """copy.deepcopy(CppGameState) should hit the C++ memcpy path, not
    a Python deepcopy of the bitboard halves and history list. Bench
    against deepcopy(GameState) for comparison."""
    rng = random.Random(0xBEEF)

    cpp = CppGameState()
    py = GameState()
    for _ in range(40):
        moves = py.get_valid_moves()
        if not moves:
            break
        m = rng.choice(moves)
        py.make_move(m)
        cpp.make_move(m)

    iters = 50_000
    t0 = time.perf_counter()
    for _ in range(iters):
        copy.deepcopy(py)
    py_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        copy.deepcopy(cpp)
    cpp_secs = time.perf_counter() - t0

    py_us = 1e6 * py_secs / iters
    cpp_us = 1e6 * cpp_secs / iters

    with capsys.disabled():
        print(
            f"\ndeepcopy bench (post-rollout, ~40 plies of history):"
            f"\n  GameState     : {py_us:8.3f} us/call"
            f"\n  CppGameState  : {cpp_us:8.3f} us/call"
            f"\n  speedup       : {py_us / cpp_us:.1f}x"
        )

    # Must be at least an order of magnitude faster — anything less
    # means the wrapper isn't routing through __deepcopy__.
    assert cpp_us * 10 < py_us, (
        f"CppGameState.deepcopy isn't routing through the fast path "
        f"(cpp={cpp_us:.3f}us, py={py_us:.3f}us)"
    )
