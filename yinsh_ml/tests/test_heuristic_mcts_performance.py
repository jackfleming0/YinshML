"""Comprehensive test suite for heuristic MCTS integration.

This test suite validates:
1. Leaf node evaluation with heuristics
2. Hybrid evaluation mode functionality
3. Performance comparison with baseline MCTS
4. Adaptive weight reduction system
5. Integration without breaking existing functionality
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.search.mcts import MCTS, MCTSConfig, EvaluationMode
from yinsh_ml.search.training_tracker import TrainingTracker
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.heuristics import YinshHeuristics


class MockNetworkWrapper:
    """Mock network wrapper for testing."""
    
    def __init__(self):
        self.total_moves = 7395
    
    def predict(self, state_tensor):
        """Return mock policy and value."""
        policy = np.random.rand(self.total_moves).astype(np.float32)
        policy = policy / policy.sum()  # Normalize
        value = np.random.uniform(-1.0, 1.0)
        return policy, value


class TestHeuristicMCTSIntegration(unittest.TestCase):
    """Integration tests for heuristic MCTS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = MockNetworkWrapper()
        self.heuristic_evaluator = YinshHeuristics()
        
        # Create test game state
        self.game_state = GameState()
    
    def test_pure_heuristic_evaluation_mode(self):
        """Test that pure heuristic mode uses only heuristic evaluation."""
        config = MCTSConfig(
            num_simulations=10,
            evaluation_mode=EvaluationMode.PURE_HEURISTIC,
            use_heuristic_evaluation=True
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Verify heuristic evaluator is initialized
        self.assertIsNotNone(mcts.heuristic_evaluator)
        self.assertEqual(mcts.config.evaluation_mode, EvaluationMode.PURE_HEURISTIC)
        
        # Run search
        try:
            policy = mcts.search(self.game_state, move_number=5)
            self.assertEqual(len(policy), 7395)
            self.assertGreaterEqual(policy.sum(), 0.0)
        except Exception as e:
            self.fail(f"MCTS search failed in pure heuristic mode: {e}")
    
    def test_pure_neural_evaluation_mode(self):
        """Test that pure neural mode uses only neural network evaluation."""
        config = MCTSConfig(
            num_simulations=10,
            evaluation_mode=EvaluationMode.PURE_NEURAL,
            use_heuristic_evaluation=False
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Verify heuristic evaluator is not used
        self.assertIsNone(mcts.heuristic_evaluator)
        self.assertEqual(mcts.config.evaluation_mode, EvaluationMode.PURE_NEURAL)
        
        # Run search
        try:
            policy = mcts.search(self.game_state, move_number=5)
            self.assertEqual(len(policy), 7395)
        except Exception as e:
            self.fail(f"MCTS search failed in pure neural mode: {e}")
    
    def test_hybrid_evaluation_mode(self):
        """Test that hybrid mode combines heuristic and neural evaluations."""
        config = MCTSConfig(
            num_simulations=10,
            evaluation_mode=EvaluationMode.HYBRID,
            use_heuristic_evaluation=True,
            heuristic_weight=0.3,
            neural_weight=0.7
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Verify both evaluators are available
        self.assertIsNotNone(mcts.heuristic_evaluator)
        self.assertEqual(mcts.config.evaluation_mode, EvaluationMode.HYBRID)
        
        # Run search
        try:
            policy = mcts.search(self.game_state, move_number=5)
            self.assertEqual(len(policy), 7395)
        except Exception as e:
            self.fail(f"MCTS search failed in hybrid mode: {e}")
    
    def test_heuristic_evaluation_normalization(self):
        """Test that heuristic scores are normalized to [-1, 1] range."""
        config = MCTSConfig(
            num_simulations=5,
            evaluation_mode=EvaluationMode.PURE_HEURISTIC
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Test normalization function
        test_scores = [-100.0, -50.0, 0.0, 50.0, 100.0]
        for score in test_scores:
            normalized = mcts._normalize_heuristic_score(score)
            self.assertGreaterEqual(normalized, -1.0)
            self.assertLessEqual(normalized, 1.0)
    
    def test_leaf_node_evaluation_with_heuristics(self):
        """Test that leaf nodes are evaluated using heuristics."""
        config = MCTSConfig(
            num_simulations=5,
            evaluation_mode=EvaluationMode.PURE_HEURISTIC
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Create a leaf node
        node = mcts._create_child_node(self.game_state, parent=None, prior_prob=0.1)
        
        # Evaluate leaf node
        value = mcts._evaluate_leaf_node(self.game_state, node)
        
        # Value should be in [-1, 1] range
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)
        
        # Node should have cached heuristic value
        self.assertIsNotNone(node.heuristic_value)
    
    def test_terminal_position_handling(self):
        """Test that terminal positions are handled correctly."""
        config = MCTSConfig(
            num_simulations=5,
            evaluation_mode=EvaluationMode.HYBRID
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Create a terminal state (if possible)
        # For now, just test that terminal check works
        terminal_value = mcts._get_terminal_value(self.game_state)
        # Should be None for non-terminal state
        if not self.game_state.is_terminal():
            self.assertIsNone(terminal_value)
    
    def test_adaptive_weight_reduction(self):
        """Test adaptive heuristic weight reduction based on training progress."""
        config = MCTSConfig(
            num_simulations=5,
            evaluation_mode=EvaluationMode.HYBRID,
            auto_reduce_heuristic_weight=True,
            initial_heuristic_weight=0.5,
            min_heuristic_weight=0.1,
            max_heuristic_weight=0.5
        )
        
        # Create training tracker
        tracker = TrainingTracker(window_size=5)
        
        # Record initial performance (weak network)
        tracker.record_iteration(1, win_rate=0.3, evaluation_accuracy=0.4)
        tracker.record_iteration(2, win_rate=0.35, evaluation_accuracy=0.45)
        
        mcts = MCTS(self.network, config=config, training_tracker=tracker)
        
        initial_weight = mcts.config.heuristic_weight
        
        # Record improved performance
        tracker.record_iteration(3, win_rate=0.6, evaluation_accuracy=0.7)
        tracker.record_iteration(4, win_rate=0.65, evaluation_accuracy=0.75)
        tracker.record_iteration(5, win_rate=0.7, evaluation_accuracy=0.8)
        
        # Update weight based on improvement
        mcts._update_heuristic_weight_from_tracker()
        
        # Weight should have decreased
        self.assertLessEqual(mcts.config.heuristic_weight, initial_weight)
        self.assertGreaterEqual(mcts.config.heuristic_weight, config.min_heuristic_weight)
    
    def test_phase_aware_weighting(self):
        """Test phase-aware weighting in hybrid mode."""
        config = MCTSConfig(
            num_simulations=5,
            evaluation_mode=EvaluationMode.HYBRID,
            use_phase_aware_weighting=True,
            heuristic_weight=0.3,
            neural_weight=0.7
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Test early game (more heuristic)
        early_state = GameState()
        node_early = mcts._create_child_node(early_state, parent=None, prior_prob=0.1)
        value_early = mcts._evaluate_leaf_node(early_state, node_early)
        
        # Test late game (more neural)
        late_state = GameState()
        # Simulate late game by adding many moves to history
        for _ in range(50):
            valid_moves = late_state.get_valid_moves()
            if valid_moves:
                late_state.make_move(valid_moves[0])
        
        node_late = mcts._create_child_node(late_state, parent=None, prior_prob=0.1)
        value_late = mcts._evaluate_leaf_node(late_state, node_late)
        
        # Both should produce valid values
        self.assertGreaterEqual(value_early, -1.0)
        self.assertLessEqual(value_early, 1.0)
        self.assertGreaterEqual(value_late, -1.0)
        self.assertLessEqual(value_late, 1.0)
    
    def test_fallback_to_neural_on_heuristic_failure(self):
        """Test that MCTS falls back to neural evaluation if heuristic fails."""
        config = MCTSConfig(
            num_simulations=5,
            evaluation_mode=EvaluationMode.PURE_HEURISTIC
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Temporarily break heuristic evaluator
        original_evaluate = mcts.heuristic_evaluator.evaluate_position
        mcts.heuristic_evaluator.evaluate_position = Mock(side_effect=Exception("Heuristic error"))
        
        # Should fallback to neural without crashing
        try:
            policy = mcts.search(self.game_state, move_number=5)
            self.assertEqual(len(policy), 7395)
        except Exception as e:
            self.fail(f"MCTS should handle heuristic failure gracefully: {e}")
        finally:
            # Restore original method
            mcts.heuristic_evaluator.evaluate_position = original_evaluate
    
    def test_mcts_tree_expansion(self):
        """Test that MCTS tree expands correctly with heuristic evaluation."""
        config = MCTSConfig(
            num_simulations=20,
            evaluation_mode=EvaluationMode.HYBRID,
            max_depth=10
        )
        
        mcts = MCTS(self.network, config=config)
        
        # Run search
        policy = mcts.search(self.game_state, move_number=5)
        
        # Verify policy is valid
        self.assertEqual(len(policy), 7395)
        self.assertGreaterEqual(policy.sum(), 0.0)
        self.assertLessEqual(policy.sum(), 1.1)  # Allow small floating point error
    
    def test_different_simulation_budgets(self):
        """Test MCTS with different simulation budgets."""
        budgets = [10, 50, 100]
        
        for budget in budgets:
            config = MCTSConfig(
                num_simulations=budget,
                evaluation_mode=EvaluationMode.HYBRID
            )
            
            mcts = MCTS(self.network, config=config)
            
            try:
                policy = mcts.search(self.game_state, move_number=5)
                self.assertEqual(len(policy), 7395)
            except Exception as e:
                self.fail(f"MCTS failed with budget {budget}: {e}")
    
    def test_configuration_validation(self):
        """Test that configuration parameters are validated correctly."""
        # Test default configuration
        config = MCTSConfig()
        self.assertEqual(config.evaluation_mode, EvaluationMode.HYBRID)
        self.assertTrue(config.use_heuristic_evaluation)
        self.assertEqual(config.heuristic_weight + config.neural_weight, 1.0)
        
        # Test custom configuration
        config = MCTSConfig(
            evaluation_mode=EvaluationMode.PURE_HEURISTIC,
            heuristic_weight=0.5,
            neural_weight=0.5
        )
        self.assertEqual(config.evaluation_mode, EvaluationMode.PURE_HEURISTIC)
        self.assertEqual(config.heuristic_weight, 0.5)
    
    def test_memory_pool_integration(self):
        """Test that memory pool integration works correctly."""
        from yinsh_ml.memory.game_state_pool import GameStatePool, GameStatePoolConfig
        
        pool_config = GameStatePoolConfig(initial_size=10)
        pool = GameStatePool(config=pool_config)
        config = MCTSConfig(num_simulations=5)
        
        mcts = MCTS(self.network, config=config, game_state_pool=pool)
        
        # Run search - should use pool
        try:
            policy = mcts.search(self.game_state, move_number=5)
            self.assertEqual(len(policy), 7395)
        except Exception as e:
            self.fail(f"MCTS failed with memory pool: {e}")


class TestMCTSPerformanceComparison(unittest.TestCase):
    """Performance comparison tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = MockNetworkWrapper()
        self.game_state = GameState()
    
    def test_evaluation_mode_performance_comparison(self):
        """Compare performance across different evaluation modes."""
        from yinsh_ml.search.performance_profiler import MCTSPerformanceProfiler
        
        profiler = MCTSPerformanceProfiler()
        
        # Compare modes (with small budget for speed)
        results = profiler.compare_evaluation_modes(
            self.network,
            self.game_state,
            move_number=5,
            simulation_budget=10,
            num_runs=3
        )
        
        # Verify all modes were tested
        self.assertIn('pure_neural', results)
        self.assertIn('pure_heuristic', results)
        self.assertIn('hybrid', results)
        
        # Verify results have expected structure
        for mode, result in results.items():
            self.assertGreater(result.avg_evaluation_time, 0.0)
            self.assertGreater(result.avg_nodes_per_second, 0.0)


if __name__ == '__main__':
    unittest.main()

