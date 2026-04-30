"""GameState-shaped facade over the C++ bitboard engine.

This module is the bridge that turns the engine-side speedups
(documented on the bitboard-port branch's first commit) into wins for
the existing MCTS / self-play / agent code. ``CppGameState`` is meant
to be a duck-typed drop-in for ``yinsh_ml.game.GameState`` — the
methods and attributes downstream code reads from a GameState are
mirrored here.

What's deliberately NOT mirrored:
  * ``_handle_*`` private helpers: callers shouldn't reach into them.
  * ``_update_game_phase``: phase transitions live entirely on the C++
    side (``apply.hpp::UpdatePhaseAndSwitchPlayer``).
  * Mutation hooks on ``board``: ``CppBoard`` is read-only. Use
    ``CppGameState.make_move`` to advance state.

The clone path (``copy``, ``copy_from``, ``__deepcopy__``) bottoms out
in ``_engine.State.clone`` — a 64-byte struct memcpy that benched at
~134000x faster than ``copy.deepcopy(GameState)``. This is the MCTS
self-play motivator from the brief.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from yinsh_ml.game.board import Row
from yinsh_ml.game.constants import (
    Player,
    PieceType,
    Position,
    RINGS_PER_PLAYER,
    VALID_POSITIONS,
)
from yinsh_ml.game.types import GamePhase, Move, MoveType
from yinsh_ml.game_cpp import _engine
from yinsh_ml.game_cpp._convert import (
    cell_to_position,
    cpp_move_to_py,
    is_black_to_player,
    materialize_pieces,
    player_to_is_black,
    position_to_cell,
    py_move_to_cpp,
)


class CppBoard:
    """Read-only Board facade backed by a ``_engine.State``.

    Hot-path methods (``find_marker_rows``, ``valid_move_positions``,
    ``get_piece``, ``is_empty``, ``to_numpy_array``) read straight off
    the bitboards. The ``pieces`` dict and a few lower-traffic helpers
    materialize lazily and cache; the cache is invalidated whenever
    the parent CppGameState advances.
    """

    __slots__ = ("_state", "_pieces_cache")

    def __init__(self, state: _engine.State):
        self._state = state
        self._pieces_cache: Optional[dict[Position, PieceType]] = None

    # --- read-mostly helpers -------------------------------------

    def _bitboard_int(self, halves: tuple[int, int]) -> int:
        lo, hi = halves
        return lo | (hi << 64)

    def _all_rings(self) -> int:
        return (self._bitboard_int(self._state.white_rings)
                | self._bitboard_int(self._state.black_rings))

    def _all_markers(self) -> int:
        return (self._bitboard_int(self._state.white_markers)
                | self._bitboard_int(self._state.black_markers))

    @property
    def pieces(self) -> dict[Position, PieceType]:
        """Position -> PieceType, materialized lazily.

        Only callers that need the full mapping (encoder, debug printer,
        a few legacy paths) should reach for this. ``get_piece`` and
        ``is_empty`` answer point queries without paying the
        materialization cost.
        """
        if self._pieces_cache is None:
            self._pieces_cache = materialize_pieces(self._state)
        return self._pieces_cache

    def get_piece(self, pos: Position) -> Optional[PieceType]:
        if pos.column not in VALID_POSITIONS:
            return None
        if pos.row not in VALID_POSITIONS[pos.column]:
            return None
        cell = position_to_cell(pos)
        bit = 1 << cell
        if self._bitboard_int(self._state.white_rings) & bit:
            return PieceType.WHITE_RING
        if self._bitboard_int(self._state.black_rings) & bit:
            return PieceType.BLACK_RING
        if self._bitboard_int(self._state.white_markers) & bit:
            return PieceType.WHITE_MARKER
        if self._bitboard_int(self._state.black_markers) & bit:
            return PieceType.BLACK_MARKER
        return None

    def is_empty(self, pos: Position) -> bool:
        if pos.column not in VALID_POSITIONS:
            return False
        if pos.row not in VALID_POSITIONS[pos.column]:
            return False
        cell = position_to_cell(pos)
        bit = 1 << cell
        return not ((self._all_rings() | self._all_markers()) & bit)

    def valid_move_positions(self, position: Position) -> list[Position]:
        """Bitboard-driven equivalent of Board.valid_move_positions.

        Returns destinations in cell-index ascending order; callers
        that depended on Python's left-to-right walk order will see a
        different ordering and should sort if they care.
        """
        piece = self.get_piece(position)
        if piece is None or not piece.is_ring():
            return []
        cell = position_to_cell(position)
        rings = self._all_rings()
        markers = self._all_markers()
        lo, hi = _engine.valid_ring_destinations(
            cell,
            rings & ((1 << 64) - 1), rings >> 64,
            markers & ((1 << 64) - 1), markers >> 64,
        )
        dests_mask = lo | (hi << 64)

        out: list[Position] = []
        # Iterate set bits low → high for deterministic order.
        m = dests_mask
        while m:
            lsb = m & (-m)
            cell_idx = lsb.bit_length() - 1
            out.append(cell_to_position(cell_idx))
            m ^= lsb
        return out

    def find_marker_rows(self, marker_type: PieceType) -> list[Row]:
        """C++ find_marker_rows wrapped to return the Row dataclass.

        Single-color only, like the Python original. The C++ side
        already handles 8-/9-/10-cell maximal runs (longer than the
        Python docstring's "6/7" claim — board geometry actually
        permits up to 10).
        """
        if not marker_type.is_marker():
            return []
        if marker_type == PieceType.WHITE_MARKER:
            mask = self._bitboard_int(self._state.white_markers)
        else:
            mask = self._bitboard_int(self._state.black_markers)
        runs = _engine.find_marker_rows(mask & ((1 << 64) - 1), mask >> 64)
        return [
            Row(
                color=marker_type,
                positions=tuple(
                    sorted(
                        (cell_to_position(c) for c in run),
                        key=lambda p: (p.column, p.row),
                    )
                ),
            )
            for run in runs
        ]

    def to_numpy_array(self) -> np.ndarray:
        """4-channel piece layout. Same shape as Board.to_numpy_array.

        Could later be optimized by writing channels directly from
        bitboards via popcount-iter; for now we materialize through
        ``pieces`` so behaviour matches Board exactly.
        """
        state = np.zeros((4, 11, 11), dtype=np.float32)
        for pos, piece in self.pieces.items():
            row = pos.row - 1
            col = ord(pos.column) - ord("A")
            if piece == PieceType.WHITE_RING:
                state[0, row, col] = 1
            elif piece == PieceType.BLACK_RING:
                state[1, row, col] = 1
            elif piece == PieceType.WHITE_MARKER:
                state[2, row, col] = 1
            elif piece == PieceType.BLACK_MARKER:
                state[3, row, col] = 1
        return state

    def get_rings_positions(self, player: Player) -> list[Position]:
        ring_type = (PieceType.WHITE_RING if player == Player.WHITE
                     else PieceType.BLACK_RING)
        return [pos for pos, piece in self.pieces.items() if piece == ring_type]

    def get_pieces_of_player(self, player: Player) -> list[Position]:
        ring = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
        marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        return [
            pos for pos, piece in self.pieces.items()
            if piece == ring or piece == marker
        ]

    def get_pieces_positions(self, piece_type: PieceType) -> list[Position]:
        return [pos for pos, piece in self.pieces.items() if piece == piece_type]


class CppGameState:
    """Drop-in replacement for ``yinsh_ml.game.GameState`` backed by a
    C++ bitboard engine.

    Surface chosen to match the duck-typed interface MCTS, self-play,
    and the encoder all read from. Two semantic differences worth
    knowing:

      * ``copy()`` / ``__deepcopy__`` — both bottom out in the C++
        struct memcpy, not a Python deepcopy. This is the point of
        the wrapper for MCTS hot paths.
      * ``move_history`` is maintained at the wrapper level. The
        underlying C++ State doesn't track it; if you only ever
        manipulate state through ``apply_move`` on the engine you
        won't get history bookkeeping. Callers that need history
        should drive through ``CppGameState`` (most do).
    """

    __slots__ = ("_state", "_move_history", "_board_view", "_board_dirty")

    def __init__(self):
        self._state = _engine.State()
        self._move_history: list[Move] = []
        self._board_view: Optional[CppBoard] = None
        self._board_dirty = True

    # --- properties mirroring GameState ----------------------------

    @property
    def board(self) -> CppBoard:
        if self._board_view is None or self._board_dirty:
            self._board_view = CppBoard(self._state)
            self._board_dirty = False
        return self._board_view

    @property
    def current_player(self) -> Player:
        return is_black_to_player(self._state.current_player_is_black)

    @current_player.setter
    def current_player(self, value: Player) -> None:
        self._state.current_player_is_black = (value == Player.BLACK)

    @property
    def phase(self) -> GamePhase:
        return GamePhase(self._state.phase)

    @phase.setter
    def phase(self, value: GamePhase) -> None:
        self._state.phase = value.value

    @property
    def white_score(self) -> int:
        return self._state.white_score

    @white_score.setter
    def white_score(self, value: int) -> None:
        self._state.white_score = value

    @property
    def black_score(self) -> int:
        return self._state.black_score

    @black_score.setter
    def black_score(self, value: int) -> None:
        self._state.black_score = value

    @property
    def rings_placed(self) -> dict[Player, int]:
        # Build on access — small dict, infrequent reads. Don't expose
        # mutable view; a setter handles writes.
        return {
            Player.WHITE: self._state.white_rings_placed,
            Player.BLACK: self._state.black_rings_placed,
        }

    @rings_placed.setter
    def rings_placed(self, value: dict[Player, int]) -> None:
        self._state.white_rings_placed = value.get(Player.WHITE, 0)
        self._state.black_rings_placed = value.get(Player.BLACK, 0)

    @property
    def move_history(self) -> list[Move]:
        return self._move_history

    @property
    def _move_maker(self) -> Optional[Player]:
        v = self._state.move_maker_is_black
        if v < 0:
            return None
        return Player.BLACK if v == 1 else Player.WHITE

    @_move_maker.setter
    def _move_maker(self, value: Optional[Player]) -> None:
        if value is None:
            self._state.move_maker_is_black = -1
        else:
            self._state.move_maker_is_black = 1 if value == Player.BLACK else 0

    # --- core engine surface --------------------------------------

    def make_move(self, move: Move) -> bool:
        """Apply ``move`` if legal; return whether it was applied.

        Validation goes through ``get_valid_moves`` and a C++-side
        equality check on the move object — fast in absolute terms
        (microseconds) but a non-zero overhead vs the
        trust-the-caller path. MCTS / self-play already enumerate
        valid moves before calling ``make_move``, so the redundant
        check is cheap insurance.
        """
        cpp_move = py_move_to_cpp(move)
        legal = _engine.get_valid_moves(self._state)
        if not any(cpp_move.equals(lm) for lm in legal):
            return False
        self._state = _engine.apply_move(self._state, cpp_move)
        self._move_history.append(move)
        self._board_dirty = True
        return True

    def get_valid_moves(self) -> list[Move]:
        return [cpp_move_to_py(cm) for cm in _engine.get_valid_moves(self._state)]

    def get_ring_valid_moves(self, position: Position) -> list[Move]:
        return [
            m for m in self.get_valid_moves()
            if m.type == MoveType.MOVE_RING and m.source == position
        ]

    def is_valid_move(self, move: Move) -> bool:
        if move.player != self.current_player:
            return False
        cpp_move = py_move_to_cpp(move)
        legal = _engine.get_valid_moves(self._state)
        return any(cpp_move.equals(lm) for lm in legal)

    def is_terminal(self) -> bool:
        return _engine.is_terminal(self._state)

    def get_winner(self) -> Optional[Player]:
        w = _engine.winner(self._state)
        if w == 0:
            return Player.WHITE
        if w == 1:
            return Player.BLACK
        if self.is_stalemate():
            return self.current_player.opponent
        return None

    def is_stalemate(self) -> bool:
        if self.phase == GamePhase.GAME_OVER:
            return False
        return not _engine.get_valid_moves(self._state)

    # --- copy / clone ---------------------------------------------

    def copy(self) -> "CppGameState":
        new = CppGameState.__new__(CppGameState)
        new._state = self._state.clone()
        new._move_history = list(self._move_history)
        new._board_view = None
        new._board_dirty = True
        return new

    def copy_from(self, source: "CppGameState") -> None:
        # Pool-style in-place copy. Replacing the underlying handle is
        # cheap because clone() is a 64-byte memcpy. The wrapper-level
        # bookkeeping (move_history, board cache) gets refreshed below.
        self._state = source._state.clone()
        self._move_history.clear()
        self._move_history.extend(source._move_history)
        self._board_view = None
        self._board_dirty = True

    def __deepcopy__(self, memo) -> "CppGameState":
        # The whole point of this class: copy.deepcopy on a CppGameState
        # bottoms out in the C++ struct memcpy, not a recursive Python
        # copy of the bitboard halves and history list.
        return self.copy()

    # --- representation ------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CppGameState(phase={self.phase.name} "
            f"player={self.current_player.name} "
            f"score={self.white_score}-{self.black_score} "
            f"rings={self._state.white_rings_placed},{self._state.black_rings_placed})"
        )


__all__ = ["CppGameState", "CppBoard"]
