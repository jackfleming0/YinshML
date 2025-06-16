"""Test ring movement validation according to YINSH rules."""

import unittest
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.game.types import Move, MoveType


class TestRingMovementValidation(unittest.TestCase):
    """Test that ring movements follow YINSH rules correctly."""
    
    def setUp(self):
        """Set up a game in MAIN_GAME phase with some rings placed."""
        self.game = GameState()
        self.game.phase = GamePhase.MAIN_GAME
        self.game.current_player = Player.WHITE
        
        # Place rings directly on the board for testing
        self.game.board.place_piece(Position.from_string('E5'), PieceType.WHITE_RING)
        self.game.board.place_piece(Position.from_string('F6'), PieceType.BLACK_RING)
        
    def test_ring_can_move_to_empty_adjacent_space(self):
        """Test that a ring can move to an adjacent empty space."""
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E6')
        )
        
        # Check move is valid
        self.assertTrue(self.game.is_valid_move(move))
        
        # Execute move
        success = self.game.make_move(move)
        self.assertTrue(success)
        
        # Verify ring moved and marker was left
        self.assertEqual(self.game.board.get_piece(Position.from_string('E6')), PieceType.WHITE_RING)
        self.assertEqual(self.game.board.get_piece(Position.from_string('E5')), PieceType.WHITE_MARKER)
        
    def test_ring_can_move_over_multiple_empty_spaces(self):
        """Test that a ring can move over multiple empty spaces."""
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E8')  # 3 spaces away
        )
        
        self.assertTrue(self.game.is_valid_move(move))
        success = self.game.make_move(move)
        self.assertTrue(success)
        
    def test_ring_cannot_jump_over_another_ring(self):
        """Test that a ring cannot jump over another ring."""
        # Place another ring in the path
        self.game.board.place_piece(Position.from_string('E6'), PieceType.BLACK_RING)
        
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E7')
        )
        
        # Move should be invalid
        self.assertFalse(self.game.is_valid_move(move))
        
        # Move should fail
        success = self.game.make_move(move)
        self.assertFalse(success)
        
    def test_ring_must_stop_at_first_empty_after_markers(self):
        """Test that after jumping markers, ring must stop at first empty space."""
        # Place markers
        self.game.board.place_piece(Position.from_string('E6'), PieceType.BLACK_MARKER)
        self.game.board.place_piece(Position.from_string('E7'), PieceType.WHITE_MARKER)
        # E8 is empty, E9 is also empty
        
        # This should be valid - stopping at first empty after markers
        valid_move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E8')
        )
        
        self.assertTrue(self.game.is_valid_move(valid_move))
        
        # This should be invalid - trying to go past first empty
        invalid_move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E9')
        )
        
        self.assertFalse(self.game.is_valid_move(invalid_move))
        
        # Execute the valid move
        success = self.game.make_move(valid_move)
        self.assertTrue(success)
        
        # Verify markers were flipped
        self.assertEqual(self.game.board.get_piece(Position.from_string('E6')), PieceType.WHITE_MARKER)
        self.assertEqual(self.game.board.get_piece(Position.from_string('E7')), PieceType.BLACK_MARKER)
        
    def test_ring_can_move_over_empty_then_jump_markers(self):
        """Test complex movement: empty spaces then jump markers."""
        # E6 is empty
        # Place markers at E7, E8
        self.game.board.place_piece(Position.from_string('E7'), PieceType.BLACK_MARKER)
        self.game.board.place_piece(Position.from_string('E8'), PieceType.WHITE_MARKER)
        # E9 is empty
        
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E9')
        )
        
        # This should be valid
        self.assertTrue(self.game.is_valid_move(move))
        success = self.game.make_move(move)
        self.assertTrue(success)
        
        # Verify only jumped markers were flipped
        self.assertEqual(self.game.board.get_piece(Position.from_string('E7')), PieceType.WHITE_MARKER)
        self.assertEqual(self.game.board.get_piece(Position.from_string('E8')), PieceType.BLACK_MARKER)
        
    def test_ring_cannot_continue_after_marker_with_gap(self):
        """Test that ring cannot continue if there's a gap after markers."""
        # Place marker at E6
        self.game.board.place_piece(Position.from_string('E6'), PieceType.BLACK_MARKER)
        # E7 is empty (first empty after marker)
        # Place another marker at E8
        self.game.board.place_piece(Position.from_string('E8'), PieceType.WHITE_MARKER)
        
        # Cannot jump to E9 because there's a gap at E7
        invalid_move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E9')
        )
        
        self.assertFalse(self.game.is_valid_move(invalid_move))
        
    def test_all_six_directions(self):
        """Test ring movement in all six hexagonal directions."""
        # Clear the board and place a single ring at F5
        self.game.board.pieces.clear()
        self.game.board.place_piece(Position.from_string('F5'), PieceType.WHITE_RING)
        
        # Test all six directions
        directions = [
            ('F6', True),   # Up
            ('G6', True),   # Up-Right
            ('G5', True),   # Right
            ('F4', True),   # Down
            ('E4', True),   # Down-Left
            ('E5', True),   # Left - now valid since we cleared the board
        ]
        
        for dest_str, should_be_valid in directions:
            move = Move(
                type=MoveType.MOVE_RING,
                player=Player.WHITE,
                source=Position.from_string('F5'),
                destination=Position.from_string(dest_str)
            )
            
            if should_be_valid:
                self.assertTrue(self.game.is_valid_move(move), 
                              f"Move to {dest_str} should be valid")
            else:
                self.assertFalse(self.game.is_valid_move(move),
                               f"Move to {dest_str} should be invalid")
                
    def test_marker_left_at_source(self):
        """Test that a marker is always left at the source position."""
        # Test white ring
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('E7')
        )
        
        self.game.make_move(move)
        self.assertEqual(self.game.board.get_piece(Position.from_string('E5')), 
                        PieceType.WHITE_MARKER)
        
        # Set up for black's turn
        self.game.current_player = Player.BLACK
        
        # Test black ring
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('F6'),
            destination=Position.from_string('F8')
        )
        
        self.game.make_move(move)
        self.assertEqual(self.game.board.get_piece(Position.from_string('F6')), 
                        PieceType.BLACK_MARKER)
                        
    def test_complex_jump_scenario(self):
        """Test a complex scenario with multiple markers to jump."""
        # Clear the board and set up: Ring at C5, markers at D5, E5, F5, G5
        self.game.board.pieces.clear()
        self.game.board.place_piece(Position.from_string('C5'), PieceType.WHITE_RING)
        self.game.board.place_piece(Position.from_string('D5'), PieceType.BLACK_MARKER)
        self.game.board.place_piece(Position.from_string('E5'), PieceType.WHITE_MARKER)
        self.game.board.place_piece(Position.from_string('F5'), PieceType.BLACK_MARKER) 
        self.game.board.place_piece(Position.from_string('G5'), PieceType.WHITE_MARKER)
        # H5 is empty
        
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('C5'),
            destination=Position.from_string('H5')
        )
        
        # Should be valid
        self.assertTrue(self.game.is_valid_move(move))
        success = self.game.make_move(move)
        self.assertTrue(success)
        
        # All markers should be flipped
        self.assertEqual(self.game.board.get_piece(Position.from_string('D5')), PieceType.WHITE_MARKER)
        self.assertEqual(self.game.board.get_piece(Position.from_string('E5')), PieceType.BLACK_MARKER)
        self.assertEqual(self.game.board.get_piece(Position.from_string('F5')), PieceType.WHITE_MARKER)
        self.assertEqual(self.game.board.get_piece(Position.from_string('G5')), PieceType.BLACK_MARKER)


if __name__ == '__main__':
    unittest.main() 