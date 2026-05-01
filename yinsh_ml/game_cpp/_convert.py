"""Move/Position conversion helpers between Python objects and the C++
engine's cell-index encoding.

Kept as a small, pure module so both the CppGameState wrapper and the
parity tests can share the conversions without duplication.
"""
from __future__ import annotations

from typing import Optional

from yinsh_ml.game.constants import Position, Player, PieceType, VALID_POSITIONS
from yinsh_ml.game.types import GamePhase, Move, MoveType
from yinsh_ml.game_cpp import _engine


COLS = "ABCDEFGHIJK"


def position_to_cell(pos: Position) -> int:
    return _engine.cell_index(ord(pos.column) - ord("A"), pos.row)


# Lookup table: cell index → Position. Built once at module load.
# 121 slots cover the full 11×11 lattice (only 99 are valid YINSH
# cells; the remainder are never queried in practice but we materialize
# them anyway so a stray cell_to_position(invalid_cell) gives the same
# Position the legacy divmod path returned). Position is
# @dataclass(frozen=True) so sharing instances across callers is safe.
#
# Profile (BITBOARD_FOLLOWUP_PLAN.md Candidate C-1) showed
# cell_to_position at 4.4s self / 9.9s cum across 7.17M calls per
# game with only 99 unique outputs — perfect cache target.
_POSITION_BY_CELL: tuple[Position, ...] = tuple(
    Position(COLS[i // 11], (i % 11) + 1) for i in range(121)
)


def cell_to_position(cell: int) -> Position:
    return _POSITION_BY_CELL[cell]


def player_to_is_black(player: Player) -> bool:
    return player == Player.BLACK


def is_black_to_player(is_black: bool) -> Player:
    return Player.BLACK if is_black else Player.WHITE


# Cell ordering for iterating bitboards back into the (col, row) order
# that the Python Board uses. Precomputed once at module load.
_CELL_ORDER: list[tuple[int, Position]] = [
    (position_to_cell(Position(c, r)), Position(c, r))
    for c in COLS for r in VALID_POSITIONS[c]
]


def py_move_to_cpp(m: Move) -> _engine.Move:
    pib = player_to_is_black(m.player)
    if m.type == MoveType.PLACE_RING:
        return _engine.Move.place_ring(pib, position_to_cell(m.source))
    if m.type == MoveType.MOVE_RING:
        return _engine.Move.move_ring(
            pib,
            position_to_cell(m.source),
            position_to_cell(m.destination),
        )
    if m.type == MoveType.REMOVE_MARKERS:
        return _engine.Move.remove_markers(
            pib, [position_to_cell(p) for p in m.markers]
        )
    if m.type == MoveType.REMOVE_RING:
        return _engine.Move.remove_ring(pib, position_to_cell(m.source))
    raise ValueError(f"unknown move type {m.type}")


def cpp_move_to_py(cm: _engine.Move) -> Move:
    player = is_black_to_player(cm.player_is_black)
    if cm.type == _engine.MOVE_PLACE_RING:
        return Move(MoveType.PLACE_RING, player, source=cell_to_position(cm.source))
    if cm.type == _engine.MOVE_MOVE_RING:
        return Move(
            MoveType.MOVE_RING,
            player,
            source=cell_to_position(cm.source),
            destination=cell_to_position(cm.destination),
        )
    if cm.type == _engine.MOVE_REMOVE_MARKERS:
        return Move(
            MoveType.REMOVE_MARKERS,
            player,
            markers=tuple(cell_to_position(c) for c in cm.markers),
        )
    if cm.type == _engine.MOVE_REMOVE_RING:
        return Move(MoveType.REMOVE_RING, player, source=cell_to_position(cm.source))
    raise ValueError(f"unknown C++ move type {cm.type}")


def materialize_pieces(state: _engine.State) -> dict[Position, PieceType]:
    """Build the position→piece dict that the Python Board exposes."""
    out: dict[Position, PieceType] = {}
    wr_lo, wr_hi = state.white_rings
    br_lo, br_hi = state.black_rings
    wm_lo, wm_hi = state.white_markers
    bm_lo, bm_hi = state.black_markers
    wr = wr_lo | (wr_hi << 64)
    br = br_lo | (br_hi << 64)
    wm = wm_lo | (wm_hi << 64)
    bm = bm_lo | (bm_hi << 64)
    for cell, pos in _CELL_ORDER:
        bit = 1 << cell
        if wr & bit:    out[pos] = PieceType.WHITE_RING
        elif br & bit:  out[pos] = PieceType.BLACK_RING
        elif wm & bit:  out[pos] = PieceType.WHITE_MARKER
        elif bm & bit:  out[pos] = PieceType.BLACK_MARKER
    return out


__all__ = [
    "COLS",
    "position_to_cell",
    "cell_to_position",
    "player_to_is_black",
    "is_black_to_player",
    "py_move_to_cpp",
    "cpp_move_to_py",
    "materialize_pieces",
]
