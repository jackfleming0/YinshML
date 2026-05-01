"""End-to-end parity: Python GameState and C++ State play random
games in lockstep. After every move we compare:
  - bitboard occupancy (4 colors × ring/marker)
  - phase, current player, scores, rings_placed
  - the legal-move set for the next ply

If they match across thousands of moves, apply_move + get_valid_moves +
phase machinery + player switching are all bit-for-bit faithful to the
Python reference. This is the load-bearing parity test for the C++
engine.
"""
from __future__ import annotations

import random

import pytest

from yinsh_ml.game.constants import Position, Player, PieceType, VALID_POSITIONS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase, Move, MoveType
from yinsh_ml.game_cpp import _engine


COLS = "ABCDEFGHIJK"


def _cell_index(pos: Position) -> int:
    col_idx = ord(pos.column) - ord("A")
    return _engine.cell_index(col_idx, pos.row)


def _index_to_position(cell: int) -> Position:
    col_idx, row_minus = divmod(cell, 11)
    return Position(COLS[col_idx], row_minus + 1)


def _split(b: int) -> tuple[int, int]:
    return b & ((1 << 64) - 1), b >> 64


def _pieces_to_bitboards(gs: GameState) -> tuple[int, int, int, int]:
    wr = br = wm = bm = 0
    for pos, piece in gs.board.pieces.items():
        b = 1 << _cell_index(pos)
        if piece == PieceType.WHITE_RING:    wr |= b
        elif piece == PieceType.BLACK_RING:  br |= b
        elif piece == PieceType.WHITE_MARKER: wm |= b
        elif piece == PieceType.BLACK_MARKER: bm |= b
    return wr, br, wm, bm


def _py_to_cpp_state(gs: GameState) -> _engine.State:
    s = _engine.State()
    wr, br, wm, bm = _pieces_to_bitboards(gs)
    s.white_rings = _split(wr)
    s.black_rings = _split(br)
    s.white_markers = _split(wm)
    s.black_markers = _split(bm)
    s.phase = gs.phase.value
    s.current_player_is_black = (gs.current_player == Player.BLACK)
    s.white_score = gs.white_score
    s.black_score = gs.black_score
    s.white_rings_placed = gs.rings_placed[Player.WHITE]
    s.black_rings_placed = gs.rings_placed[Player.BLACK]
    mm = getattr(gs, "_move_maker", None)
    if mm is None:
        s.move_maker_is_black = -1
    else:
        s.move_maker_is_black = 1 if mm == Player.BLACK else 0
    return s


def _py_move_to_cpp(m: Move) -> _engine.Move:
    pib = (m.player == Player.BLACK)
    if m.type == MoveType.PLACE_RING:
        return _engine.Move.place_ring(pib, _cell_index(m.source))
    if m.type == MoveType.MOVE_RING:
        return _engine.Move.move_ring(
            pib, _cell_index(m.source), _cell_index(m.destination))
    if m.type == MoveType.REMOVE_MARKERS:
        return _engine.Move.remove_markers(
            pib, [_cell_index(p) for p in m.markers])
    if m.type == MoveType.REMOVE_RING:
        return _engine.Move.remove_ring(pib, _cell_index(m.source))
    raise AssertionError(f"unknown move type {m.type}")


def _cpp_move_to_py(cm: _engine.Move) -> Move:
    player = Player.BLACK if cm.player_is_black else Player.WHITE
    if cm.type == _engine.MOVE_PLACE_RING:
        return Move(MoveType.PLACE_RING, player, source=_index_to_position(cm.source))
    if cm.type == _engine.MOVE_MOVE_RING:
        return Move(MoveType.MOVE_RING, player,
                    source=_index_to_position(cm.source),
                    destination=_index_to_position(cm.destination))
    if cm.type == _engine.MOVE_REMOVE_MARKERS:
        return Move(MoveType.REMOVE_MARKERS, player,
                    markers=tuple(_index_to_position(c) for c in cm.markers))
    if cm.type == _engine.MOVE_REMOVE_RING:
        return Move(MoveType.REMOVE_RING, player, source=_index_to_position(cm.source))
    raise AssertionError(f"unknown C++ move type {cm.type}")


def _move_key(m: Move) -> tuple:
    """Hashable key for set-equality comparison of Python moves.
    Markers are sorted since C++ canonicalizes that order."""
    if m.markers is not None:
        marker_strs = tuple(sorted(str(p) for p in m.markers))
    else:
        marker_strs = None
    return (m.type.value,
            m.player.value,
            str(m.source) if m.source else None,
            str(m.destination) if m.destination else None,
            marker_strs)


def _assert_states_equal(gs: GameState, cpp_s: _engine.State, ctx: str) -> None:
    wr, br, wm, bm = _pieces_to_bitboards(gs)
    assert cpp_s.white_rings == _split(wr), f"{ctx}: white_rings mismatch"
    assert cpp_s.black_rings == _split(br), f"{ctx}: black_rings mismatch"
    assert cpp_s.white_markers == _split(wm), f"{ctx}: white_markers mismatch"
    assert cpp_s.black_markers == _split(bm), f"{ctx}: black_markers mismatch"
    assert cpp_s.phase == gs.phase.value, (
        f"{ctx}: phase {cpp_s.phase} vs {gs.phase.value}")
    assert cpp_s.current_player_is_black == (gs.current_player == Player.BLACK), (
        f"{ctx}: current_player mismatch")
    assert cpp_s.white_score == gs.white_score, f"{ctx}: white_score mismatch"
    assert cpp_s.black_score == gs.black_score, f"{ctx}: black_score mismatch"
    assert cpp_s.white_rings_placed == gs.rings_placed[Player.WHITE]
    assert cpp_s.black_rings_placed == gs.rings_placed[Player.BLACK]


def _assert_legal_moves_equal(gs: GameState,
                              cpp_s: _engine.State,
                              ctx: str) -> None:
    py_moves = gs.get_valid_moves()
    cpp_moves = _engine.get_valid_moves(cpp_s)

    py_keys = {_move_key(m) for m in py_moves}
    cpp_keys = {_move_key(_cpp_move_to_py(cm)) for cm in cpp_moves}

    assert py_keys == cpp_keys, (
        f"{ctx}: legal-move set mismatch\n"
        f"  only in Py:  {sorted(py_keys - cpp_keys)[:5]}\n"
        f"  only in C++: {sorted(cpp_keys - py_keys)[:5]}\n"
        f"  total py={len(py_keys)} cpp={len(cpp_keys)}\n"
        f"  phase={gs.phase.name} player={gs.current_player.name}"
    )


@pytest.mark.parametrize("seed", list(range(50)))
def test_random_rollout_parity(seed: int) -> None:
    """Play a single random game in lockstep. The Python engine drives;
    after each move we check that the C++ engine's apply_move produced
    the same state and that its legal-move set for the next ply
    matches Python's."""
    rng = random.Random(seed)
    gs = GameState()
    cpp_s = _py_to_cpp_state(gs)

    # Sanity check the initial states match.
    _assert_states_equal(gs, cpp_s, "initial state")
    _assert_legal_moves_equal(gs, cpp_s, "initial legal moves")

    max_plies = 400
    for ply in range(max_plies):
        if gs.is_terminal():
            break
        py_moves = gs.get_valid_moves()
        if not py_moves:
            break  # stalemate; the parity check above already verified
                   # the C++ engine agrees the move set is empty
        m = rng.choice(py_moves)

        # Apply both sides.
        cpp_m = _py_move_to_cpp(m)
        cpp_next = _engine.apply_move(cpp_s, cpp_m)
        ok = gs.make_move(m)
        assert ok, f"seed={seed} ply={ply}: Python rejected its own legal move {m}"

        ctx = f"seed={seed} ply={ply} after applying {m}"
        _assert_states_equal(gs, cpp_next, ctx)
        # Move the C++ side forward in lockstep.
        cpp_s = cpp_next
        # And check the next-ply legal-move sets agree.
        if not gs.is_terminal():
            _assert_legal_moves_equal(gs, cpp_s, ctx)
        else:
            assert _engine.is_terminal(cpp_s), f"{ctx}: terminal disagreement"

    # Terminal states should also agree on winner if any.
    if gs.is_terminal():
        py_winner = gs.get_winner()
        cpp_winner = _engine.winner(cpp_s)
        py_winner_idx = (-1 if py_winner is None
                         else (0 if py_winner == Player.WHITE else 1))
        # Note: get_winner can also fire on stalemate via is_stalemate,
        # which uses move enumeration. C++ winner() is score-only.
        # Stalemate parity is exercised by the legal-move-set check
        # earlier in the rollout, so we only require score agreement.
        if py_winner_idx in (0, 1):
            assert cpp_winner == py_winner_idx, (
                f"seed={seed}: winner disagreement "
                f"py={py_winner_idx} cpp={cpp_winner}")
