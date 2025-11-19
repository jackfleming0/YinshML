"""Unit and integration tests for heuristic-enhanced UCB1 selection.

This test suite verifies that EnhancedNode.get_enhanced_ucb_score() correctly
combines UCB1 scores with heuristic evaluations using the alpha parameter.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.training.enhanced_mcts import EnhancedNode, EnhancedMCTSConfig
from yinsh_ml.analysis.heuristic_evaluator import HeuristicEvaluator, EvaluationConfig


class TestHeuristicEnhancedUCB1(unittest.TestCase):
    """Test suite for heuristic-enhanced UCB1 selection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game_state = GameState()
        self.config = EnhancedMCTSConfig(
            use_heuristic_guidance=True,
            heuristic_alpha=0.3,
            epsilon_greedy=0.4
        )
        self.heuristic_evaluator = HeuristicEvaluator(config=EvaluationConfig(speed_target=2000))
    
    def test_pure_ucb1_when_alpha_zero(self):
        """Test that alpha=0.0 results in pure UCB1 (no heuristic influence)."""
        config = EnhancedMCTSConfig(
            use_heuristic_guidance=True,
            heuristic_alpha=0.0
        )
        
        node = EnhancedNode(self.game_state, c_puct=1.0)
        node.visit_count = 10
        node.value_sum = 5.0  # Mean value = 0.5
        
        parent_visit_count = 20
        
        # Calculate UCB1 score
        q_value = node.value()
        exploration_term = (node.c_puct * node.prior_prob * 
                           np.sqrt(parent_visit_count) / (1 + node.visit_count))
        expected_ucb = q_value + exploration_term
        
        # Get enhanced UCB score with alpha=0
        enhanced_score = node.get_enhanced_ucb_score(
            parent_visit_count, config, self.heuristic_evaluator
        )
        
        # Should equal pure UCB1 (within floating point precision)
        self.assertAlmostEqual(enhanced_score, expected_ucb, places=5,
                              msg="Alpha=0 should result in pure UCB1")
    
    def test_pure_heuristic_when_alpha_one(self):
        """Test that alpha=1.0 results in pure heuristic evaluation."""
        config = EnhancedMCTSConfig(
            use_heuristic_guidance=True,
            heuristic_alpha=1.0
        )
        
        node = EnhancedNode(self.game_state, c_puct=1.0)
        node.visit_count = 10
        node.value_sum = 5.0
        
        parent_visit_count = 20
        
        # Get heuristic value
        heuristic_value = node._get_heuristic_value(self.heuristic_evaluator)
        
        # Get enhanced UCB score with alpha=1
        enhanced_score = node.get_enhanced_ucb_score(
            parent_visit_count, config, self.heuristic_evaluator
        )
        
        # Should equal heuristic value (within floating point precision)
        self.assertAlmostEqual(enhanced_score, heuristic_value, places=5,
                              msg="Alpha=1 should result in pure heuristic")
    
    def test_default_alpha_blend(self):
        """Test that default alpha=0.3 correctly blends UCB1 and heuristic."""
        config = EnhancedMCTSConfig(
            use_heuristic_guidance=True,
            heuristic_alpha=0.3  # Default value
        )
        
        node = EnhancedNode(self.game_state, c_puct=1.0)
        node.visit_count = 10
        node.value_sum = 5.0
        
        parent_visit_count = 20
        
        # Calculate components
        q_value = node.value()
        exploration_term = (node.c_puct * node.prior_prob * 
                           np.sqrt(parent_visit_count) / (1 + node.visit_count))
        ucb_score = q_value + exploration_term
        heuristic_value = node._get_heuristic_value(self.heuristic_evaluator)
        
        # Expected combined score
        expected_score = (1 - 0.3) * ucb_score + 0.3 * heuristic_value
        
        # Get enhanced UCB score
        enhanced_score = node.get_enhanced_ucb_score(
            parent_visit_count, config, self.heuristic_evaluator
        )
        
        # Should match expected blend
        self.assertAlmostEqual(enhanced_score, expected_score, places=5,
                              msg="Default alpha should correctly blend UCB1 and heuristic")
    
    def test_heuristic_value_caching(self):
        """Test that heuristic values are cached properly."""
        node = EnhancedNode(self.game_state, c_puct=1.0)
        
        # First call should compute heuristic value
        heuristic_value_1 = node._get_heuristic_value(self.heuristic_evaluator)
        self.assertIsNotNone(node.heuristic_value, "Heuristic value should be cached")
        
        # Second call should use cached value
        with patch.object(self.heuristic_evaluator, 'evaluate_position_fast') as mock_eval:
            heuristic_value_2 = node._get_heuristic_value(self.heuristic_evaluator)
            mock_eval.assert_not_called()  # Should not call evaluator again
            self.assertEqual(heuristic_value_1, heuristic_value_2,
                           "Cached value should match original")
    
    def test_no_heuristic_when_disabled(self):
        """Test that heuristic guidance is skipped when use_heuristic_guidance=False."""
        config = EnhancedMCTSConfig(
            use_heuristic_guidance=False,
            heuristic_alpha=0.3
        )
        
        node = EnhancedNode(self.game_state, c_puct=1.0)
        node.visit_count = 10
        node.value_sum = 5.0
        
        parent_visit_count = 20
        
        # Calculate pure UCB1
        q_value = node.value()
        exploration_term = (node.c_puct * node.prior_prob * 
                           np.sqrt(parent_visit_count) / (1 + node.visit_count))
        expected_ucb = q_value + exploration_term
        
        # Get enhanced UCB score (should ignore heuristic)
        enhanced_score = node.get_enhanced_ucb_score(
            parent_visit_count, config, self.heuristic_evaluator
        )
        
        # Should equal pure UCB1
        self.assertAlmostEqual(enhanced_score, expected_ucb, places=5,
                              msg="Should use pure UCB1 when heuristic guidance disabled")
    
    def test_no_heuristic_when_evaluator_none(self):
        """Test that system falls back to UCB1 when heuristic evaluator is None."""
        config = EnhancedMCTSConfig(
            use_heuristic_guidance=True,
            heuristic_alpha=0.3
        )
        
        node = EnhancedNode(self.game_state, c_puct=1.0)
        node.visit_count = 10
        node.value_sum = 5.0
        
        parent_visit_count = 20
        
        # Calculate pure UCB1
        q_value = node.value()
        exploration_term = (node.c_puct * node.prior_prob * 
                           np.sqrt(parent_visit_count) / (1 + node.visit_count))
        expected_ucb = q_value + exploration_term
        
        # Get enhanced UCB score with None evaluator
        enhanced_score = node.get_enhanced_ucb_score(
            parent_visit_count, config, None
        )
        
        # Should equal pure UCB1
        self.assertAlmostEqual(enhanced_score, expected_ucb, places=5,
                              msg="Should fall back to UCB1 when evaluator is None")
    
    def test_different_alpha_values(self):
        """Test that different alpha values produce different blended scores."""
        node = EnhancedNode(self.game_state, c_puct=1.0)
        node.visit_count = 10
        node.value_sum = 5.0
        
        parent_visit_count = 20
        
        # Calculate components
        q_value = node.value()
        exploration_term = (node.c_puct * node.prior_prob * 
                           np.sqrt(parent_visit_count) / (1 + node.visit_count))
        ucb_score = q_value + exploration_term
        heuristic_value = node._get_heuristic_value(self.heuristic_evaluator)
        
        # Test different alpha values
        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
        scores = []
        
        for alpha in alphas:
            config = EnhancedMCTSConfig(
                use_heuristic_guidance=True,
                heuristic_alpha=alpha
            )
            score = node.get_enhanced_ucb_score(
                parent_visit_count, config, self.heuristic_evaluator
            )
            scores.append(score)
        
        # Scores should vary with alpha
        # If heuristic and UCB differ, scores should be different
        if abs(heuristic_value - ucb_score) > 0.01:
            self.assertNotEqual(len(set(scores)), 1,
                              "Different alpha values should produce different scores")
        
        # Scores should interpolate between UCB and heuristic
        self.assertAlmostEqual(scores[0], ucb_score, places=5,
                              msg="Alpha=0 should equal UCB1")
        self.assertAlmostEqual(scores[-1], heuristic_value, places=5,
                              msg="Alpha=1 should equal heuristic")
    
    def test_node_selection_behavior(self):
        """Integration test: Verify nodes with good heuristic values are explored more."""
        # Create two nodes with different heuristic values
        state1 = GameState()
        state2 = GameState()
        
        # Modify state2 to have better heuristic evaluation
        # (e.g., add markers creating a threat)
        from yinsh_ml.game.constants import Position, PieceType
        state2.board.place_piece(Position.from_string('B1'), PieceType.WHITE_MARKER)
        state2.board.place_piece(Position.from_string('B2'), PieceType.WHITE_MARKER)
        state2.board.place_piece(Position.from_string('B3'), PieceType.WHITE_MARKER)
        state2.board.place_piece(Position.from_string('B4'), PieceType.WHITE_MARKER)
        
        node1 = EnhancedNode(state1, c_puct=1.0, prior_prob=0.5)
        node2 = EnhancedNode(state2, c_puct=1.0, prior_prob=0.5)
        
        node1.visit_count = 10
        node1.value_sum = 5.0
        node2.visit_count = 10
        node2.value_sum = 5.0
        
        parent_visit_count = 20
        
        # Get heuristic values
        heuristic1 = node1._get_heuristic_value(self.heuristic_evaluator)
        heuristic2 = node2._get_heuristic_value(self.heuristic_evaluator)
        
        # Node2 should have better heuristic (threatening position)
        self.assertGreater(heuristic2, heuristic1,
                          "Node with threat should have better heuristic")
        
        # With heuristic guidance, node2 should score higher
        config = EnhancedMCTSConfig(
            use_heuristic_guidance=True,
            heuristic_alpha=0.3
        )
        
        score1 = node1.get_enhanced_ucb_score(
            parent_visit_count, config, self.heuristic_evaluator
        )
        score2 = node2.get_enhanced_ucb_score(
            parent_visit_count, config, self.heuristic_evaluator
        )
        
        self.assertGreater(score2, score1,
                          "Node with better heuristic should score higher")


if __name__ == '__main__':
    unittest.main()

