import unittest
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.game.moves import Move, MoveType


class TestYinshGameLogic(unittest.TestCase):
    def setUp(self):
        self.game = GameState()
        # Set up initial rings for both players to get past placement phase
        self._setup_initial_rings()

    def _setup_initial_rings(self):
        """Helper to place initial rings for both players."""
        ring_positions = [
            ('A2', Player.WHITE),
            ('A3', Player.WHITE),
            ('A4', Player.WHITE),
            ('A5', Player.WHITE),
            ('C5', Player.WHITE),
            ('I5', Player.BLACK),
            ('K7', Player.BLACK),
            ('K8', Player.BLACK),
            ('K9', Player.BLACK),
            ('K10', Player.BLACK),
        ]

        for pos, player in ring_positions:
            self.game.current_player = player
            move = Move(
                type=MoveType.PLACE_RING,
                player=player,
                source=Position.from_string(pos)
            )
            if not self.game.make_move(move):
                print(f"Failed to place ring at {pos}")


    def _place_markers(self, positions, marker_type):
        """Helper to place markers directly on the board."""
        for pos in positions:
            pos = Position.from_string(pos) if isinstance(pos, str) else pos
            self.game.board.place_piece(pos, marker_type)

    def test_white_flip_creates_black_row(self):
        """Test white's ring flipping a marker creates a black row."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.WHITE

        # Place markers for our test - six consecutive black markers
        self._place_markers(['E4', 'E6', 'E7', 'E8', 'E9'], PieceType.BLACK_MARKER)
        # Place the white marker that will be flipped
        self._place_markers(['E5'], PieceType.WHITE_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # First remove the black ring placed during setup
        self.game.board.remove_piece(Position.from_string('I5'))
        # Then place our white ring
        self.game.board.place_piece(Position.from_string('I5'), PieceType.WHITE_RING)

        print("\nBoard state after placing white ring:")
        print(self.game.board)

        black_markers_before = [pos for pos, piece in self.game.board.pieces.items()
                                if piece == PieceType.BLACK_MARKER]
        print(f"\nBlack markers before move: {sorted(black_markers_before, key=lambda p: (p.column, p.row))}")

        # Move white ring from I5 to D5 to flip black marker
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('I5'),
            destination=Position.from_string('D5')
        )

        success = self.game.make_move(move)
        print(f"\nMove success: {success}")

        print("\nFinal board state:")
        print(self.game.board)

        # Print some debug info about black markers
        black_markers_after = [pos for pos, piece in self.game.board.pieces.items()
                               if piece == PieceType.BLACK_MARKER]
        print(f"\nBlack markers after move: {sorted(black_markers_after, key=lambda p: (p.column, p.row))}")

        # Print all rows found
        rows_found = self.game.board.find_marker_rows(PieceType.BLACK_MARKER)
        print(f"\nRows found: {rows_found}")

        # Should now be in ROW_COMPLETION phase for Black
        self.assertEqual(self.game.phase, GamePhase.ROW_COMPLETION)
        self.assertEqual(self.game.current_player, Player.BLACK)

    def test_black_flip_creates_white_row(self):
        """Test black's ring flipping a marker creates a white row."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.BLACK

        # Place markers for our test
        self._place_markers(['E4', 'E6', 'E7', 'E8', 'E9'], PieceType.WHITE_MARKER)
        self._place_markers(['E5'], PieceType.BLACK_MARKER)

        print("\nInitial board state:")
        print(self.game.board)

        # Place a black ring at I5 (we'll be moving this)
        self.game.board.place_piece(Position.from_string('I5'), PieceType.BLACK_RING)

        print("\nBoard state after placing ring:")
        print(self.game.board)

        white_markers_before = [pos for pos, piece in self.game.board.pieces.items()
                                if piece == PieceType.WHITE_MARKER]
        print(f"\nWhite markers before move: {sorted(white_markers_before, key=lambda p: (p.column, p.row))}")

        # Move black ring from I5 to D5 to flip markers
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('I5'),
            destination=Position.from_string('D5')
        )

        success = self.game.make_move(move)
        print(f"\nMove success: {success}")

        print("\nFinal board state:")
        print(self.game.board)

        # Print some debug info about white markers
        white_markers_after = [pos for pos, piece in self.game.board.pieces.items()
                               if piece == PieceType.WHITE_MARKER]
        print(f"\nWhite markers after move: {sorted(white_markers_after, key=lambda p: (p.column, p.row))}")

        # Print all rows found
        rows_found = self.game.board.find_marker_rows(PieceType.WHITE_MARKER)
        print(f"\nRows found: {rows_found}")

        # Should now be in ROW_COMPLETION phase for White
        self.assertEqual(self.game.phase, GamePhase.ROW_COMPLETION)
        self.assertEqual(self.game.current_player, Player.WHITE)

    def test_white_creates_multiple_rows(self):
        """Test white creating multiple rows in one move."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.WHITE

        # Set up two potential rows with a gap at E5
        # First vertical row: E1-E4, gap at E5, E6
        self._place_markers(['E1', 'E2', 'E3', 'E4'], PieceType.WHITE_MARKER)
        self._place_markers(['E6'], PieceType.WHITE_MARKER)

        # Second vertical row: D1-D4, gap at D5, D6
        self._place_markers(['D1', 'D2', 'D3', 'D4'], PieceType.WHITE_MARKER)
        self._place_markers(['D6'], PieceType.WHITE_MARKER)

        # Place black markers at the gaps
        self._place_markers(['E5', 'D5'], PieceType.BLACK_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # Remove the black ring placed during setup and place our white ring
        self.game.board.remove_piece(Position.from_string('I5'))
        self.game.board.place_piece(Position.from_string('I5'), PieceType.WHITE_RING)

        print("\nBoard state after placing white ring:")
        print(self.game.board)

        # Move white ring to flip both black markers
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('C5'),
            destination=Position.from_string('F5')
        )

        success = self.game.make_move(move)
        print(f"\nMove success: {success}")
        self.assertTrue(success)

        print("\nBoard state after move:")
        print(self.game.board)

        # Should now be in ROW_COMPLETION phase for first row
        self.assertEqual(self.game.phase, GamePhase.ROW_COMPLETION)
        self.assertEqual(self.game.current_player, Player.WHITE)

        # Remove first row
        move = Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.WHITE,
            markers=[Position.from_string(p) for p in ['E1', 'E2', 'E3', 'E4', 'E5']]
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Remove a ring after first row
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('A2')  # One of white's rings from setup
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Should still be in ROW_COMPLETION phase for second row
        self.assertEqual(self.game.phase, GamePhase.ROW_COMPLETION)
        self.assertEqual(self.game.current_player, Player.WHITE)

        # Remove second row
        move = Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.WHITE,
            markers=[Position.from_string(p) for p in ['D1', 'D2', 'D3', 'D4', 'D5']]
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Remove another ring
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('A3')  # Another white ring from setup
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Should now be back in MAIN_GAME phase
        self.assertEqual(self.game.phase, GamePhase.MAIN_GAME)

        # Should have scored 2 points
        self.assertEqual(self.game.white_score, 2)

        # Should be Black's turn now
        self.assertEqual(self.game.current_player, Player.BLACK)

    def test_black_creates_multiple_rows(self):
        """Test black creating multiple rows in one move."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.BLACK

        # Set up two potential rows with a gap at E5
        # First vertical row: E1-E4, gap at E5, E6
        self._place_markers(['E1', 'E2', 'E3', 'E4'], PieceType.BLACK_MARKER)
        self._place_markers(['E6'], PieceType.BLACK_MARKER)

        # Second vertical row: D1-D4, gap at D5, D6
        self._place_markers(['D1', 'D2', 'D3', 'D4'], PieceType.BLACK_MARKER)
        self._place_markers(['D6'], PieceType.BLACK_MARKER)

        # Place white markers at the gaps
        self._place_markers(['E5', 'D5'], PieceType.WHITE_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # Remove the black ring placed during setup and place our black ring
        self.game.board.remove_piece(Position.from_string('C5'))
        self.game.board.place_piece(Position.from_string('C5'), PieceType.BLACK_RING)

        print("\nBoard state after placing black ring:")
        print(self.game.board)

        # Move black ring to flip both white markers
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('C5'),
            destination=Position.from_string('F5')
        )

        success = self.game.make_move(move)
        print(f"\nMove success: {success}")
        self.assertTrue(success)

        print("\nBoard state after move:")
        print(self.game.board)

        # Should now be in ROW_COMPLETION phase for first row
        self.assertEqual(self.game.phase, GamePhase.ROW_COMPLETION)
        self.assertEqual(self.game.current_player, Player.BLACK)

        # Remove first row
        move = Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.BLACK,
            markers=[Position.from_string(p) for p in ['E1', 'E2', 'E3', 'E4', 'E5']]
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Remove a ring after first row
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('K7')  # One of black's rings from setup
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Should still be in ROW_COMPLETION phase for second row
        self.assertEqual(self.game.phase, GamePhase.ROW_COMPLETION)
        self.assertEqual(self.game.current_player, Player.BLACK)

        # Remove second row
        move = Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.BLACK,
            markers=[Position.from_string(p) for p in ['D1', 'D2', 'D3', 'D4', 'D5']]
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Remove another ring
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('K8')  # Another black ring from setup
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Should now be back in MAIN_GAME phase
        self.assertEqual(self.game.phase, GamePhase.MAIN_GAME)

        # Should have scored 2 points
        self.assertEqual(self.game.black_score, 2)

        # Should be White's turn now
        self.assertEqual(self.game.current_player, Player.WHITE)

    def test_white_creates_both_color_rows(self):
        """Test white creating rows of both colors in one move."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.WHITE

        # Set up white row that will be completed by ring placement
        self._place_markers(['E4', 'E6', 'E7', 'E8'], PieceType.WHITE_MARKER)
        self._place_markers(['F5'], PieceType.WHITE_MARKER)

        # Set up black row that will be completed by flipped markers
        self._place_markers(['F4', 'F6', 'F7', 'F8'], PieceType.BLACK_MARKER)
        self._place_markers(['E5'], PieceType.BLACK_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # Remove default black ring and place our test pieces
        self.game.board.remove_piece(Position.from_string('I5'))
        self.game.board.place_piece(Position.from_string('C5'), PieceType.WHITE_RING)

        print("\nBoard state after placing white ring:")
        print(self.game.board)

        # Move the White ring to create both rows
        success = self.game.make_move(Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('C5'),
            destination=Position.from_string('G5')
        ))
        self.assertTrue(success)

        print("\nBoard state after move:")
        print(self.game.board)

        # White should get to remove their markers first
        self.assertEqual(self.game.current_player, Player.WHITE)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.WHITE,
            markers=[
                Position.from_string('E4'),
                Position.from_string('E5'),
                Position.from_string('E6'),
                Position.from_string('E7'),
                Position.from_string('E8')
            ]
        ))
        self.assertTrue(success)

        # White removes ring
        self.assertEqual(self.game.current_player, Player.WHITE)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('A2')
        ))
        self.assertTrue(success)

        # Black gets to remove their markers
        self.assertEqual(self.game.current_player, Player.BLACK)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.BLACK,
            markers=[
                Position.from_string('F4'),
                Position.from_string('F5'),
                Position.from_string('F6'),
                Position.from_string('F7'),
                Position.from_string('F8')
            ]
        ))
        self.assertTrue(success)

        # Black removes ring
        self.assertEqual(self.game.current_player, Player.BLACK)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('K7')
        ))
        self.assertTrue(success)

        # Should return to MAIN_GAME with Black as current player
        self.assertEqual(self.game.phase, GamePhase.MAIN_GAME)
        self.assertEqual(self.game.current_player, Player.BLACK)



    def test_black_creates_both_color_rows(self):
        """Test black creating rows of both colors in one move."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.BLACK

        # Set up black row that will be completed by ring placement
        self._place_markers(['E4', 'E5', 'E6', 'E8'], PieceType.BLACK_MARKER)
        self._place_markers(['F7'], PieceType.BLACK_MARKER)

        # Set up white row that will be completed by flipped markers
        self._place_markers(['F4', 'F5', 'F6', 'F8'], PieceType.WHITE_MARKER)
        self._place_markers(['E7'], PieceType.WHITE_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # Remove default rings and place our test pieces
        # Place black ring that will make the move
        self.game.board.remove_piece(Position.from_string('I5'))
        self.game.board.place_piece(Position.from_string('C5'), PieceType.BLACK_RING)

        print("\nBoard state after placing black ring:")
        print(self.game.board)

        # Move the Black ring to create both rows
        success = self.game.make_move(Move(
            type=MoveType.MOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('K7'),
            destination=Position.from_string('D7')
        ))
        self.assertTrue(success)

        print("\nBoard state after move:")
        print(self.game.board)

        # Black should get to remove their markers first
        self.assertEqual(self.game.current_player, Player.BLACK)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.BLACK,
            markers=[
                Position.from_string('E4'),
                Position.from_string('E5'),
                Position.from_string('E6'),
                Position.from_string('E7'),
                Position.from_string('E8')
            ]
        ))
        self.assertTrue(success)

        # Black removes ring
        self.assertEqual(self.game.current_player, Player.BLACK)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('K10')
        ))
        self.assertTrue(success)

        # White gets to remove their markers
        self.assertEqual(self.game.current_player, Player.WHITE)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.WHITE,
            markers=[
                Position.from_string('F4'),
                Position.from_string('F5'),
                Position.from_string('F6'),
                Position.from_string('F7'),
                Position.from_string('F8')
            ]
        ))
        self.assertTrue(success)

        # White removes ring
        self.assertEqual(self.game.current_player, Player.WHITE)
        success = self.game.make_move(Move(
            type=MoveType.REMOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('A2')
        ))
        self.assertTrue(success)

        # Should return to MAIN_GAME with White as current player (opponent of Black who made original move)
        self.assertEqual(self.game.phase, GamePhase.MAIN_GAME)
        self.assertEqual(self.game.current_player, Player.WHITE)

    def test_game_end_white(self):
        """Test game ending when a player reaches 3 points."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.WHITE

        # Set up scores - White needs one more point to win
        self.game.white_score = 2
        self.game.black_score = 1

        # Set up a row for white to complete
        self._place_markers(['E1', 'E2', 'E3', 'E4'], PieceType.WHITE_MARKER)
        self._place_markers(['E6'], PieceType.WHITE_MARKER)
        self._place_markers(['E5'], PieceType.BLACK_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # Remove default black ring and place rings for our test
        self.game.board.remove_piece(Position.from_string('I5'))
        self.game.board.place_piece(Position.from_string('C5'), PieceType.WHITE_RING)

        print("\nBoard state after placing rings:")
        print(self.game.board)

        # Move white ring to complete row
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('C5'),
            destination=Position.from_string('F5')
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        print("\nBoard state after completing row:")
        print(self.game.board)

        # Remove markers
        move = Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.WHITE,
            markers=[
                Position.from_string('E1'),
                Position.from_string('E2'),
                Position.from_string('E3'),
                Position.from_string('E4'),
                Position.from_string('E5')
            ]
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Remove ring to complete scoring
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('A2')  # One of White's rings from setup
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Game should be over with White as winner
        self.assertEqual(self.game.phase, GamePhase.GAME_OVER)
        self.assertEqual(self.game.white_score, 3)
        self.assertEqual(self.game.get_winner(), Player.WHITE)

    def test_game_end_black(self):
        """Test game ending when black reaches 3 points."""
        # First make sure we're in MAIN_GAME phase
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.BLACK

        # Set up scores - Black needs one more point to win
        self.game.white_score = 1
        self.game.black_score = 2

        # Set up a row for black to complete
        self._place_markers(['E1', 'E2', 'E3', 'E4'], PieceType.BLACK_MARKER)
        self._place_markers(['E6'], PieceType.BLACK_MARKER)
        self._place_markers(['E5'], PieceType.WHITE_MARKER)

        print("\nInitial board state before ring placement:")
        print(self.game.board)

        # Remove default black ring and place rings for our test
        self.game.board.remove_piece(Position.from_string('C5'))
        self.game.board.place_piece(Position.from_string('C5'), PieceType.BLACK_RING)

        print("\nBoard state after placing rings:")
        print(self.game.board)

        # Move black ring to complete row
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('C5'),
            destination=Position.from_string('F5')
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        print("\nBoard state after completing row:")
        print(self.game.board)

        # Remove markers
        move = Move(
            type=MoveType.REMOVE_MARKERS,
            player=Player.BLACK,
            markers=[
                Position.from_string('E1'),
                Position.from_string('E2'),
                Position.from_string('E3'),
                Position.from_string('E4'),
                Position.from_string('E5')
            ]
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Remove ring to complete scoring
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('K7')  # One of Black's rings from setup
        )
        success = self.game.make_move(move)
        self.assertTrue(success)

        # Game should be over with Black as winner
        self.assertEqual(self.game.phase, GamePhase.GAME_OVER)
        self.assertEqual(self.game.black_score, 3)
        self.assertEqual(self.game.get_winner(), Player.BLACK)

if __name__ == '__main__':
    unittest.main()