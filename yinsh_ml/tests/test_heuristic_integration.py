"""Integration tests for heuristic evaluator compatibility with game engine.

This test suite verifies that the heuristic evaluator correctly interfaces
with the existing game engine components, including:
- GameState compatibility
- Board state representation
- Move generation interfaces
- Data structure compatibility
"""

import unittest
from typing import List

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, PieceType, Position
from yinsh_ml.game.types import Move, MoveType, GamePhase
from yinsh_ml.game.board import Board
from yinsh_ml.game.moves import MoveGenerator
from yinsh_ml.heuristics import YinshHeuristics


class TestHeuristicGameEngineCompatibility(unittest.TestCase):
    """Test compatibility between heuristic evaluator and game engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristics = YinshHeuristics()
        self.game_state = GameState()
    
    def test_game_state_interface_compatibility(self):
        """Test that GameState has all required attributes for heuristics."""
        # Verify GameState has required attributes
        self.assertIsInstance(self.game_state.board, Board)
        self.assertIsInstance(self.game_state.current_player, Player)
        self.assertIsInstance(self.game_state.phase, GamePhase)
        self.assertIsInstance(self.game_state.white_score, int)
        self.assertIsInstance(self.game_state.black_score, int)
        self.assertIsInstance(self.game_state.rings_placed, dict)
        self.assertIsInstance(self.game_state.move_history, list)
    
    def test_board_interface_compatibility(self):
        """Test that Board has all methods required by heuristics."""
        board = self.game_state.board
        
        # Test required methods exist
        self.assertTrue(hasattr(board, 'pieces'))
        self.assertTrue(hasattr(board, 'find_marker_rows'))
        self.assertTrue(hasattr(board, 'get_pieces_positions'))
        self.assertTrue(hasattr(board, 'is_empty'))
        self.assertTrue(hasattr(board, 'get_piece'))
        
        # Test method signatures work correctly
        marker_rows = board.find_marker_rows(PieceType.WHITE_MARKER)
        self.assertIsInstance(marker_rows, list)
        
        positions = board.get_pieces_positions(PieceType.WHITE_RING)
        self.assertIsInstance(positions, list)
        
        # Test with empty board
        empty_pos = Position('E', 5)
        self.assertTrue(board.is_empty(empty_pos))
        self.assertIsNone(board.get_piece(empty_pos))
    
    def test_heuristic_evaluation_with_empty_board(self):
        """Test heuristic evaluation with an empty game state."""
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score, float)
        # Empty board should give neutral or slightly positive score
        self.assertGreater(score, -100.0)
        self.assertLess(score, 100.0)
    
    def test_heuristic_evaluation_with_ring_placement(self):
        """Test heuristic evaluation during ring placement phase."""
        # Place some rings
        self.game_state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        self.game_state.board.place_piece(Position('E', 6), PieceType.BLACK_RING)
        
        score_white = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        score_black = self.heuristics.evaluate_position(self.game_state, Player.BLACK)
        
        self.assertIsInstance(score_white, float)
        self.assertIsInstance(score_black, float)
        # Scores should be opposite (approximately)
        self.assertAlmostEqual(score_white, -score_black, delta=0.1)
    
    def test_heuristic_evaluation_with_markers(self):
        """Test heuristic evaluation with markers on board."""
        # Create a simple marker pattern
        positions = [
            Position('E', 5),
            Position('E', 6),
            Position('E', 7),
            Position('E', 8),
            Position('E', 9),
        ]
        
        for pos in positions:
            self.game_state.board.place_piece(pos, PieceType.WHITE_MARKER)
        
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score, float)
        # Completed row should give positive score
        self.assertGreater(score, 0.0)
    
    def test_heuristic_evaluation_different_phases(self):
        """Test heuristic evaluation works across different game phases."""
        # Test ring placement phase
        self.game_state.phase = GamePhase.RING_PLACEMENT
        score1 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score1, float)
        
        # Test main game phase
        self.game_state.phase = GamePhase.MAIN_GAME
        score2 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score2, float)
        
        # Test row completion phase
        self.game_state.phase = GamePhase.ROW_COMPLETION
        score3 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score3, float)
    
    def test_batch_evaluation_compatibility(self):
        """Test batch evaluation with multiple game states."""
        # Create multiple game states
        states = [GameState() for _ in range(5)]
        players = [Player.WHITE, Player.BLACK, Player.WHITE, Player.BLACK, Player.WHITE]
        
        scores = self.heuristics.evaluate_batch(states, players)
        
        self.assertEqual(len(scores), 5)
        self.assertIsInstance(scores, list)
        for score in scores:
            self.assertIsInstance(score, float)
    
    def test_move_generation_compatibility(self):
        """Test that move generation works with game states used by heuristics."""
        moves = self.game_state.get_valid_moves()
        self.assertIsInstance(moves, list)
        
        # Verify moves are valid Move objects
        for move in moves:
            self.assertIsInstance(move, Move)
            self.assertIsInstance(move.type, MoveType)
            self.assertIsInstance(move.player, Player)
    
    def test_game_state_copy_compatibility(self):
        """Test that GameState copying works correctly for heuristic evaluation."""
        # Modify original state
        self.game_state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        
        # Copy state
        copied_state = self.game_state.copy()
        
        # Evaluate both
        score_original = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        score_copied = self.heuristics.evaluate_position(copied_state, Player.WHITE)
        
        # Scores should be identical
        self.assertEqual(score_original, score_copied)
        
        # Modify copy - original should be unaffected
        copied_state.board.place_piece(Position('E', 6), PieceType.BLACK_RING)
        score_original_after = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertEqual(score_original, score_original_after)
    
    def test_position_representation_compatibility(self):
        """Test that Position objects work correctly with heuristics."""
        # Test Position creation and usage
        pos = Position('E', 5)
        self.assertEqual(str(pos), 'E5')
        
        # Place piece at position
        self.game_state.board.place_piece(pos, PieceType.WHITE_RING)
        
        # Verify piece is at position
        piece = self.game_state.board.get_piece(pos)
        self.assertEqual(piece, PieceType.WHITE_RING)
        
        # Evaluate - should work correctly
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score, float)
    
    def test_player_enum_compatibility(self):
        """Test that Player enum works correctly with heuristics."""
        # Test both players
        score_white = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        score_black = self.heuristics.evaluate_position(self.game_state, Player.BLACK)
        
        self.assertIsInstance(score_white, float)
        self.assertIsInstance(score_black, float)
        
        # Test opponent property
        self.assertEqual(Player.WHITE.opponent, Player.BLACK)
        self.assertEqual(Player.BLACK.opponent, Player.WHITE)
    
    def test_piece_type_compatibility(self):
        """Test that PieceType enum works correctly with heuristics."""
        # Test all piece types
        self.assertTrue(PieceType.WHITE_RING.is_ring())
        self.assertTrue(PieceType.BLACK_RING.is_ring())
        self.assertTrue(PieceType.WHITE_MARKER.is_marker())
        self.assertTrue(PieceType.BLACK_MARKER.is_marker())
        
        # Test player association
        self.assertEqual(PieceType.WHITE_RING.get_player(), Player.WHITE)
        self.assertEqual(PieceType.BLACK_RING.get_player(), Player.BLACK)
        self.assertEqual(PieceType.WHITE_MARKER.get_player(), Player.WHITE)
        self.assertEqual(PieceType.BLACK_MARKER.get_player(), Player.BLACK)
    
    def test_game_state_move_history_compatibility(self):
        """Test that move_history works correctly with phase detection."""
        # Empty move history (early game)
        self.assertEqual(len(self.game_state.move_history), 0)
        score1 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score1, float)
        
        # Add some moves (simulate mid game)
        from yinsh_ml.game.types import Move
        for i in range(10):
            move = Move(
                type=MoveType.PLACE_RING,
                player=Player.WHITE if i % 2 == 0 else Player.BLACK,
                source=Position('E', 5 + i)
            )
            self.game_state.move_history.append(move)
        
        score2 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score2, float)
    
    def test_integration_with_move_generator(self):
        """Test integration between heuristics and move generator."""
        # Get valid moves
        moves = MoveGenerator.get_valid_moves(self.game_state.board, self.game_state)
        self.assertIsInstance(moves, list)
        
        # Evaluate position
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score, float)
        
        # Both should work together without conflicts
        self.assertGreater(len(moves), 0)
    
    def test_edge_case_empty_board(self):
        """Test edge case: completely empty board."""
        empty_state = GameState()
        score = self.heuristics.evaluate_position(empty_state, Player.WHITE)
        self.assertIsInstance(score, float)
        # Should not crash and return reasonable value
        self.assertGreater(score, -1000.0)
        self.assertLess(score, 1000.0)
    
    def test_edge_case_full_board(self):
        """Test edge case: board with many pieces."""
        # Place rings and markers
        for col in 'ABCDEFGHIJK':
            for row in range(1, 12):
                pos = Position(col, row)
                if self.game_state.board.is_empty(pos):
                    if len(self.game_state.board.pieces) % 3 == 0:
                        self.game_state.board.place_piece(pos, PieceType.WHITE_MARKER)
                    elif len(self.game_state.board.pieces) % 3 == 1:
                        self.game_state.board.place_piece(pos, PieceType.BLACK_MARKER)
        
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score, float)
        # Should not crash with many pieces


class TestInterfaceDocumentation(unittest.TestCase):
    """Test that interface documentation matches actual implementation."""
    
    def test_game_state_attributes_documented(self):
        """Verify GameState has documented attributes."""
        gs = GameState()
        required_attrs = [
            'board', 'current_player', 'phase', 'white_score',
            'black_score', 'rings_placed', 'move_history'
        ]
        
        for attr in required_attrs:
            self.assertTrue(
                hasattr(gs, attr),
                f"GameState missing required attribute: {attr}"
            )
    
    def test_board_methods_documented(self):
        """Verify Board has documented methods."""
        board = Board()
        required_methods = [
            'find_marker_rows',
            'get_pieces_positions',
            'is_empty',
            'get_piece',
            'place_piece',
            'remove_piece',
        ]
        
        for method in required_methods:
            self.assertTrue(
                hasattr(board, method),
                f"Board missing required method: {method}"
            )
            self.assertTrue(
                callable(getattr(board, method)),
                f"Board.{method} is not callable"
            )


if __name__ == '__main__':
    unittest.main()

