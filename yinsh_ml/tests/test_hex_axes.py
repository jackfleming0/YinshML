"""Regression tests for hex-axis correctness in row detection.

Historically, `Board.find_marker_rows` and `Board.is_valid_marker_sequence`
treated `(-1, 1)` / `(1, -1)` as a diagonal axis. In this (column, row)
coordinate system that direction is a pseudo-diagonal — it does not
correspond to any line on the YINSH board. The three real hex axes are:

    vertical:   (0, 1) / (0, -1)
    horizontal: (1, 0) / (-1, 0)
    diagonal:   (1, 1) / (-1, -1)   # matching signs

These tests lock in that only the three real axes are accepted.
"""

import unittest

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import (
    PieceType,
    Player,
    Position,
    DIRECTIONS,
    HEX_DIRECTIONS,
    HEX_LINE_AXES,
)


def _place_markers(board: Board, positions, marker_type: PieceType) -> None:
    for pos_str in positions:
        board.place_piece(Position.from_string(pos_str), marker_type)


class TestHexAxisConstants(unittest.TestCase):
    def test_directions_is_three_forward_axes(self):
        self.assertEqual(
            set(DIRECTIONS),
            {(0, 1), (1, 0), (1, 1)},
            "DIRECTIONS must be the 3 forward-only hex axes (no pseudo-diagonal).",
        )

    def test_hex_directions_has_six_entries_with_matching_sign_diagonal(self):
        self.assertEqual(len(HEX_DIRECTIONS), 6)
        self.assertEqual(
            set(HEX_DIRECTIONS),
            {(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)},
        )
        # The pseudo-diagonal must NEVER appear.
        self.assertNotIn((1, -1), HEX_DIRECTIONS)
        self.assertNotIn((-1, 1), HEX_DIRECTIONS)

    def test_hex_line_axes_matches_hex_directions(self):
        self.assertEqual(HEX_LINE_AXES, frozenset(HEX_DIRECTIONS))


class TestFindMarkerRowsPseudoDiagonal(unittest.TestCase):
    """Pseudo-diagonal positions must not be detected as rows."""

    def test_pseudo_diagonal_h4_to_d8_not_a_row(self):
        board = Board()
        _place_markers(
            board, ['H4', 'G5', 'F6', 'E7', 'D8'], PieceType.WHITE_MARKER
        )
        rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertEqual(
            rows, [], "H4-G5-F6-E7-D8 is a pseudo-diagonal, not a real hex line."
        )

    def test_pseudo_diagonal_a5_to_e1_not_a_row(self):
        board = Board()
        _place_markers(
            board, ['A5', 'B4', 'C3', 'D2', 'E1'], PieceType.BLACK_MARKER
        )
        rows = board.find_marker_rows(PieceType.BLACK_MARKER)
        self.assertEqual(
            rows, [], "A5-B4-C3-D2-E1 is a pseudo-diagonal, not a real hex line."
        )

    def test_pseudo_diagonal_is_invalid_marker_sequence_white(self):
        board = Board()
        _place_markers(
            board, ['H4', 'G5', 'F6', 'E7', 'D8'], PieceType.WHITE_MARKER
        )
        positions = [Position.from_string(p) for p in ['H4', 'G5', 'F6', 'E7', 'D8']]
        self.assertFalse(
            board.is_valid_marker_sequence(positions, Player.WHITE),
            "Pseudo-diagonal sequence must not validate.",
        )

    def test_pseudo_diagonal_is_invalid_marker_sequence_black(self):
        board = Board()
        _place_markers(
            board, ['A5', 'B4', 'C3', 'D2', 'E1'], PieceType.BLACK_MARKER
        )
        positions = [Position.from_string(p) for p in ['A5', 'B4', 'C3', 'D2', 'E1']]
        self.assertFalse(
            board.is_valid_marker_sequence(positions, Player.BLACK),
            "Pseudo-diagonal sequence must not validate.",
        )


class TestFindMarkerRowsRealAxes(unittest.TestCase):
    """The three real hex axes must continue to work."""

    def test_vertical_row_detected(self):
        board = Board()
        _place_markers(
            board, ['E1', 'E2', 'E3', 'E4', 'E5'], PieceType.WHITE_MARKER
        )
        rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertEqual(len(rows), 1, f"Expected one vertical row, got {rows}")

    def test_horizontal_row_detected(self):
        board = Board()
        _place_markers(
            board, ['B6', 'C6', 'D6', 'E6', 'F6'], PieceType.WHITE_MARKER
        )
        rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertEqual(len(rows), 1, f"Expected one horizontal row, got {rows}")

    def test_matching_sign_diagonal_row_detected(self):
        board = Board()
        # (1, 1) diagonal: column advances A->E while row advances 2->6.
        _place_markers(
            board, ['A2', 'B3', 'C4', 'D5', 'E6'], PieceType.BLACK_MARKER
        )
        rows = board.find_marker_rows(PieceType.BLACK_MARKER)
        self.assertEqual(
            len(rows), 1, f"Expected one matching-sign diagonal row, got {rows}"
        )

    def test_vertical_sequence_is_valid(self):
        board = Board()
        _place_markers(
            board, ['E1', 'E2', 'E3', 'E4', 'E5'], PieceType.WHITE_MARKER
        )
        positions = [Position.from_string(p) for p in ['E1', 'E2', 'E3', 'E4', 'E5']]
        self.assertTrue(board.is_valid_marker_sequence(positions, Player.WHITE))

    def test_horizontal_sequence_is_valid(self):
        board = Board()
        _place_markers(
            board, ['B6', 'C6', 'D6', 'E6', 'F6'], PieceType.WHITE_MARKER
        )
        positions = [Position.from_string(p) for p in ['B6', 'C6', 'D6', 'E6', 'F6']]
        self.assertTrue(board.is_valid_marker_sequence(positions, Player.WHITE))

    def test_matching_sign_diagonal_sequence_is_valid(self):
        board = Board()
        _place_markers(
            board, ['A2', 'B3', 'C4', 'D5', 'E6'], PieceType.BLACK_MARKER
        )
        positions = [Position.from_string(p) for p in ['A2', 'B3', 'C4', 'D5', 'E6']]
        self.assertTrue(board.is_valid_marker_sequence(positions, Player.BLACK))


class TestExtendedMarkerRows(unittest.TestCase):
    """Real-YINSH 6/7-marker-run support.

    Per the rules, when a player completes a row they may choose ANY 5
    consecutive markers from the run to remove. The engine therefore
    returns the FULL run from ``find_marker_rows`` and enumerates every
    5-window in ``_get_marker_removal_moves``.
    """

    def _remove_moves(self, board: Board, player: Player):
        from yinsh_ml.game.moves import MoveGenerator, MoveType
        from yinsh_ml.game.game_state import GameState, GamePhase

        gs = GameState()
        gs.board = board
        gs.phase = GamePhase.ROW_COMPLETION
        gs.current_player = player
        moves = MoveGenerator.get_valid_moves(board, gs)
        return [m for m in moves if m.type == MoveType.REMOVE_MARKERS]

    def test_five_markers_yields_one_row_one_window(self):
        board = Board()
        _place_markers(
            board, ['B1', 'B2', 'B3', 'B4', 'B5'], PieceType.WHITE_MARKER
        )
        rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].positions), 5)

        moves = self._remove_moves(board, Player.WHITE)
        self.assertEqual(len(moves), 1)

    def test_six_markers_yields_one_row_two_windows(self):
        board = Board()
        _place_markers(
            board, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'], PieceType.WHITE_MARKER
        )
        rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].positions), 6)

        moves = self._remove_moves(board, Player.WHITE)
        self.assertEqual(len(moves), 2)
        # Windows must be distinct sets.
        marker_sets = {frozenset(m.markers) for m in moves}
        self.assertEqual(len(marker_sets), 2)
        expected_first = {Position.from_string(p) for p in ['B1', 'B2', 'B3', 'B4', 'B5']}
        expected_second = {Position.from_string(p) for p in ['B2', 'B3', 'B4', 'B5', 'B6']}
        self.assertIn(frozenset(expected_first), marker_sets)
        self.assertIn(frozenset(expected_second), marker_sets)

    def test_seven_markers_yields_one_row_three_windows(self):
        board = Board()
        _place_markers(
            board, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'], PieceType.BLACK_MARKER
        )
        rows = board.find_marker_rows(PieceType.BLACK_MARKER)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].positions), 7)

        moves = self._remove_moves(board, Player.BLACK)
        self.assertEqual(len(moves), 3)
        marker_sets = {frozenset(m.markers) for m in moves}
        self.assertEqual(len(marker_sets), 3)

    def test_five_markers_with_gap_no_row(self):
        board = Board()
        # Gap at B3.
        _place_markers(
            board, ['B1', 'B2', 'B4', 'B5', 'B6'], PieceType.WHITE_MARKER
        )
        rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertEqual(rows, [])

        moves = self._remove_moves(board, Player.WHITE)
        self.assertEqual(moves, [])

    def test_six_run_each_window_validates_as_marker_sequence(self):
        board = Board()
        _place_markers(
            board, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'], PieceType.WHITE_MARKER
        )
        moves = self._remove_moves(board, Player.WHITE)
        self.assertEqual(len(moves), 2)
        for m in moves:
            self.assertTrue(
                board.is_valid_marker_sequence(list(m.markers), Player.WHITE)
            )

    def test_six_run_capture_leaves_sixth_marker(self):
        """End-to-end: choose the B1..B5 window through make_move; B6 must remain."""
        from yinsh_ml.game.game_state import GameState, GamePhase
        from yinsh_ml.game.moves import Move, MoveType

        gs = GameState()
        # Seed rings so black has at least one ring to later remove.
        for pos_str, player in [
            ('A2', Player.WHITE), ('A3', Player.WHITE), ('A4', Player.WHITE),
            ('A5', Player.WHITE), ('C5', Player.WHITE),
            ('I5', Player.BLACK), ('K7', Player.BLACK), ('K8', Player.BLACK),
            ('K9', Player.BLACK), ('K10', Player.BLACK),
        ]:
            gs.current_player = player
            gs.make_move(Move(
                type=MoveType.PLACE_RING,
                player=player,
                source=Position.from_string(pos_str),
            ))

        # Force into ROW_COMPLETION with a 6-run of white markers.
        for m in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']:
            gs.board.place_piece(Position.from_string(m), PieceType.WHITE_MARKER)
        gs.phase = GamePhase.ROW_COMPLETION
        gs.current_player = Player.WHITE

        chosen = tuple(Position.from_string(p) for p in ['B1', 'B2', 'B3', 'B4', 'B5'])
        ok = gs.make_move(Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.WHITE,
            markers=chosen,
        ))
        self.assertTrue(ok, "Choosing B1..B5 from a 6-run must succeed")
        for removed in ['B1', 'B2', 'B3', 'B4', 'B5']:
            self.assertIsNone(gs.board.get_piece(Position.from_string(removed)))
        # B6 must still be on the board.
        self.assertEqual(
            gs.board.get_piece(Position.from_string('B6')),
            PieceType.WHITE_MARKER,
        )


if __name__ == '__main__':
    unittest.main()
