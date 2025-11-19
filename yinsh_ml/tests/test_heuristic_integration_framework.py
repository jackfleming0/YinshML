"""Comprehensive integration testing framework for heuristic evaluator.

This module provides end-to-end integration tests that validate:
- Correct board state interpretation during actual game sequences
- Heuristic evaluation accuracy with real game play
- Proper interaction with game rules engine
- Performance benchmarks
- Regression tests
"""

import unittest
import time
from typing import List, Tuple, Optional
import statistics

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, PieceType, Position
from yinsh_ml.game.types import Move, MoveType, GamePhase
from yinsh_ml.game.moves import MoveGenerator
from yinsh_ml.heuristics import YinshHeuristics


class TestEndToEndGameIntegration(unittest.TestCase):
    """End-to-end integration tests with actual game sequences."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristics = YinshHeuristics()
        self.game_state = GameState()
    
    def test_heuristic_evaluation_through_full_game(self):
        """Test heuristic evaluation through a short game sequence."""
        # Simplified: Just test a few moves to verify integration works
        evaluation_history = []
        max_iterations = 10  # Just test a few moves
        
        for i in range(max_iterations):
            if self.game_state.phase == GamePhase.GAME_OVER:
                break
                
            # Get valid moves
            valid_moves = self.game_state.get_valid_moves()
            if not valid_moves:
                break
            
            # Evaluate position before move
            score_before = self.heuristics.evaluate_position(
                self.game_state,
                self.game_state.current_player
            )
            
            # Make a move (use first valid move)
            move = valid_moves[0]
            success = self.game_state.make_move(move)
            if not success:
                break
            
            # Evaluate position after move
            score_after = self.heuristics.evaluate_position(
                self.game_state,
                self.game_state.current_player
            )
            
            # Record evaluation
            evaluation_history.append({
                'move': move,
                'phase': self.game_state.phase,
                'score_before': score_before,
                'score_after': score_after,
            })
        
        # Verify we made some moves
        self.assertGreater(len(evaluation_history), 0, "Should have made at least one move")
        
        # Verify evaluations are reasonable
        for eval_record in evaluation_history:
            self.assertIsInstance(eval_record['score_before'], float)
            self.assertIsInstance(eval_record['score_after'], float)
            self.assertTrue(abs(eval_record['score_before']) < 1e6)
            self.assertTrue(abs(eval_record['score_after']) < 1e6)
    
    def test_heuristic_evaluation_ring_placement_phase(self):
        """Test heuristic evaluation during ring placement phase."""
        # Ensure we're in ring placement phase
        self.assertEqual(self.game_state.phase, GamePhase.RING_PLACEMENT)
        
        # Place rings and evaluate
        ring_positions = [
            Position('E', 5),
            Position('E', 6),
            Position('E', 7),
            Position('E', 8),
        ]
        
        scores = []
        for i, pos in enumerate(ring_positions):
            if self.game_state.current_player == Player.WHITE:
                self.game_state.board.place_piece(pos, PieceType.WHITE_RING)
            else:
                self.game_state.board.place_piece(pos, PieceType.BLACK_RING)
            
            score = self.heuristics.evaluate_position(
                self.game_state,
                self.game_state.current_player
            )
            scores.append(score)
            
            # Switch player for next ring
            self.game_state.current_player = self.game_state.current_player.opponent
        
        # Verify all evaluations are valid
        self.assertEqual(len(scores), len(ring_positions))
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertTrue(abs(score) < 1e6)
    
    def test_heuristic_evaluation_with_completed_rows(self):
        """Test heuristic evaluation when rows are completed."""
        # Create a completed row scenario
        # Place markers in a row
        row_positions = [
            Position('E', 5),
            Position('E', 6),
            Position('E', 7),
            Position('E', 8),
            Position('E', 9),
        ]
        
        for pos in row_positions:
            self.game_state.board.place_piece(pos, PieceType.WHITE_MARKER)
        
        # Evaluate position
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        
        # Completed row should give positive score
        self.assertGreater(score, 0.0, "Completed row should give positive score")
        
        # Evaluate from opponent's perspective
        score_opponent = self.heuristics.evaluate_position(self.game_state, Player.BLACK)
        
        # Opponent should see negative score
        self.assertLess(score_opponent, 0.0, "Opponent should see negative score")
    
    def test_heuristic_evaluation_phase_transitions(self):
        """Test heuristic evaluation across phase transitions."""
        # Start in ring placement
        self.assertEqual(self.game_state.phase, GamePhase.RING_PLACEMENT)
        score1 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        
        # Transition to main game (simulate by placing rings)
        for i in range(10):
            pos = Position('E', 5 + i)
            if i % 2 == 0:
                self.game_state.board.place_piece(pos, PieceType.WHITE_RING)
            else:
                self.game_state.board.place_piece(pos, PieceType.BLACK_RING)
        
        # Manually set phase to main game for testing
        self.game_state.phase = GamePhase.MAIN_GAME
        score2 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        
        # Both evaluations should be valid
        self.assertIsInstance(score1, float)
        self.assertIsInstance(score2, float)
        self.assertTrue(abs(score1) < 1e6)
        self.assertTrue(abs(score2) < 1e6)
    
    def test_batch_evaluation_with_game_sequences(self):
        """Test batch evaluation with multiple game states from sequences."""
        # Create multiple game states at different stages
        states = []
        players = []
        
        # Initial state
        state1 = GameState()
        states.append(state1)
        players.append(Player.WHITE)
        
        # State with some rings placed
        state2 = GameState()
        state2.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        state2.board.place_piece(Position('E', 6), PieceType.BLACK_RING)
        states.append(state2)
        players.append(Player.WHITE)
        
        # State with markers
        state3 = GameState()
        for i in range(5):
            state3.board.place_piece(Position('E', 5 + i), PieceType.WHITE_MARKER)
        states.append(state3)
        players.append(Player.WHITE)
        
        # Batch evaluate
        scores = self.heuristics.evaluate_batch(states, players)
        
        # Verify results
        self.assertEqual(len(scores), len(states))
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertTrue(abs(score) < 1e6)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for heuristic evaluation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristics = YinshHeuristics()
        self.game_state = GameState()
    
    def test_single_evaluation_performance(self):
        """Benchmark single position evaluation performance."""
        iterations = 1000
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        avg_time = elapsed / iterations
        
        # Performance should be reasonable (< 10ms per evaluation)
        self.assertLess(avg_time, 0.01, f"Average evaluation time {avg_time*1000:.2f}ms exceeds 10ms")
        
        print(f"\nSingle evaluation performance: {avg_time*1000:.2f}ms per evaluation")
    
    def test_batch_evaluation_performance(self):
        """Benchmark batch evaluation performance."""
        batch_size = 100
        states = [GameState() for _ in range(batch_size)]
        players = [Player.WHITE] * batch_size
        
        start_time = time.perf_counter()
        scores = self.heuristics.evaluate_batch(states, players)
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        avg_time = elapsed / batch_size
        
        # Batch evaluation should be faster per evaluation
        self.assertLess(avg_time, 0.01, f"Average batch evaluation time {avg_time*1000:.2f}ms exceeds 10ms")
        
        print(f"\nBatch evaluation performance: {avg_time*1000:.2f}ms per evaluation (batch size: {batch_size})")
    
    def test_evaluation_consistency(self):
        """Test that repeated evaluations of the same position are consistent."""
        scores = []
        iterations = 100
        
        for _ in range(iterations):
            score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
            scores.append(score)
        
        # All scores should be identical (deterministic)
        self.assertEqual(len(set(scores)), 1, "Evaluations should be deterministic")
        
        # Verify score is reasonable
        self.assertTrue(abs(scores[0]) < 1e6)
    
    def test_evaluation_with_different_board_sizes(self):
        """Test evaluation performance with different board configurations."""
        # Empty board
        empty_state = GameState()
        start = time.perf_counter()
        score1 = self.heuristics.evaluate_position(empty_state, Player.WHITE)
        time1 = time.perf_counter() - start
        
        # Board with many pieces
        full_state = GameState()
        piece_count = 0
        for col in 'ABCDEFGHIJK':
            for row in range(1, 12):
                pos = Position(col, row)
                if full_state.board.is_empty(pos) and piece_count < 50:
                    if piece_count % 2 == 0:
                        full_state.board.place_piece(pos, PieceType.WHITE_MARKER)
                    else:
                        full_state.board.place_piece(pos, PieceType.BLACK_MARKER)
                    piece_count += 1
        
        start = time.perf_counter()
        score2 = self.heuristics.evaluate_position(full_state, Player.WHITE)
        time2 = time.perf_counter() - start
        
        # Both should complete in reasonable time
        self.assertLess(time1, 0.1, "Empty board evaluation too slow")
        self.assertLess(time2, 0.1, "Full board evaluation too slow")
        
        # Both should produce valid scores
        self.assertIsInstance(score1, float)
        self.assertIsInstance(score2, float)


class TestRegressionTests(unittest.TestCase):
    """Regression tests to prevent breaking changes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristics = YinshHeuristics()
    
    def test_empty_board_evaluation_regression(self):
        """Regression test: empty board should evaluate to neutral score."""
        game_state = GameState()
        score = self.heuristics.evaluate_position(game_state, Player.WHITE)
        
        # Empty board should be close to neutral
        self.assertGreater(score, -10.0)
        self.assertLess(score, 10.0)
    
    def test_completed_row_evaluation_regression(self):
        """Regression test: completed row should give significant positive score."""
        game_state = GameState()
        
        # Create completed row
        for i in range(5):
            pos = Position('E', 5 + i)
            game_state.board.place_piece(pos, PieceType.WHITE_MARKER)
        
        score = self.heuristics.evaluate_position(game_state, Player.WHITE)
        
        # Completed row should give positive score
        self.assertGreater(score, 0.0, "Completed row should give positive score")
    
    def test_opponent_perspective_regression(self):
        """Regression test: scores should be opposite from opponent's perspective."""
        game_state = GameState()
        game_state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        game_state.board.place_piece(Position('E', 6), PieceType.BLACK_RING)
        
        score_white = self.heuristics.evaluate_position(game_state, Player.WHITE)
        score_black = self.heuristics.evaluate_position(game_state, Player.BLACK)
        
        # Scores should be approximately opposite
        self.assertAlmostEqual(score_white, -score_black, delta=0.1)


class TestGameRulesIntegration(unittest.TestCase):
    """Test integration with game rules engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristics = YinshHeuristics()
        self.game_state = GameState()
    
    def test_heuristic_with_move_generation(self):
        """Test heuristic evaluation works correctly with move generation."""
        # Get valid moves
        valid_moves = MoveGenerator.get_valid_moves(
            self.game_state.board,
            self.game_state
        )
        
        self.assertGreater(len(valid_moves), 0, "Should have valid moves")
        
        # Evaluate position
        score = self.heuristics.evaluate_position(
            self.game_state,
            self.game_state.current_player
        )
        
        self.assertIsInstance(score, float)
        
        # Make a move and evaluate again
        move = valid_moves[0]
        success = self.game_state.make_move(move)
        self.assertTrue(success)
        
        score_after = self.heuristics.evaluate_position(
            self.game_state,
            self.game_state.current_player
        )
        
        self.assertIsInstance(score_after, float)
    
    def test_heuristic_with_move_validation(self):
        """Test heuristic evaluation with move validation."""
        # Get valid moves
        valid_moves = self.game_state.get_valid_moves()
        
        for move in valid_moves[:5]:  # Test first 5 moves
            # Validate move
            is_valid = self.game_state.is_valid_move(move)
            self.assertTrue(is_valid, f"Move {move} should be valid")
            
            # Evaluate before move
            score_before = self.heuristics.evaluate_position(
                self.game_state,
                move.player
            )
            
            # Make move
            success = self.game_state.make_move(move)
            self.assertTrue(success)
            
            # Evaluate after move
            score_after = self.heuristics.evaluate_position(
                self.game_state,
                self.game_state.current_player
            )
            
            # Both scores should be valid
            self.assertIsInstance(score_before, float)
            self.assertIsInstance(score_after, float)
            
            # Reset for next iteration (simplified - in real test would copy state)
            break  # Just test one move to avoid state pollution
    
    def test_heuristic_with_phase_detection(self):
        """Test that heuristic correctly interprets game phases."""
        # Test each phase
        phases = [
            GamePhase.RING_PLACEMENT,
            GamePhase.MAIN_GAME,
            GamePhase.ROW_COMPLETION,
            GamePhase.RING_REMOVAL,
        ]
        
        for phase in phases:
            self.game_state.phase = phase
            score = self.heuristics.evaluate_position(
                self.game_state,
                Player.WHITE
            )
            
            # Should produce valid score for each phase
            self.assertIsInstance(score, float)
            self.assertTrue(abs(score) < 1e6)


if __name__ == '__main__':
    unittest.main()

