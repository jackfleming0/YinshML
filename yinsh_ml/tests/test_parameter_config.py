"""Tests for EnhancedMCTSConfig parameter management.

This test suite verifies that EnhancedMCTSConfig correctly manages alpha and epsilon
parameters with validation and runtime configuration support.
"""

import unittest
from yinsh_ml.training.enhanced_mcts import EnhancedMCTSConfig
from yinsh_ml.analysis.phase_analyzer import GamePhase


class TestParameterConfiguration(unittest.TestCase):
    """Test suite for parameter configuration management."""
    
    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        config = EnhancedMCTSConfig()
        
        # Verify defaults
        self.assertEqual(config.heuristic_alpha, 0.3,
                        "Default heuristic_alpha should be 0.3")
        self.assertEqual(config.epsilon_greedy, 0.4,
                        "Default epsilon_greedy should be 0.4")
        self.assertTrue(config.use_heuristic_guidance,
                       "Default use_heuristic_guidance should be True")
        self.assertTrue(config.use_heuristic_rollouts,
                       "Default use_heuristic_rollouts should be True")
    
    def test_custom_parameters(self):
        """Test that custom parameters can be set."""
        config = EnhancedMCTSConfig(
            heuristic_alpha=0.5,
            epsilon_greedy=0.6,
            use_heuristic_guidance=False,
            use_heuristic_rollouts=False
        )
        
        self.assertEqual(config.heuristic_alpha, 0.5)
        self.assertEqual(config.epsilon_greedy, 0.6)
        self.assertFalse(config.use_heuristic_guidance)
        self.assertFalse(config.use_heuristic_rollouts)
    
    def test_parameter_bounds_validation(self):
        """Test that parameters accept valid ranges."""
        # Alpha should be in [0, 1]
        config1 = EnhancedMCTSConfig(heuristic_alpha=0.0)
        self.assertEqual(config1.heuristic_alpha, 0.0)
        
        config2 = EnhancedMCTSConfig(heuristic_alpha=1.0)
        self.assertEqual(config2.heuristic_alpha, 1.0)
        
        config3 = EnhancedMCTSConfig(heuristic_alpha=0.5)
        self.assertEqual(config3.heuristic_alpha, 0.5)
        
        # Epsilon should be in [0, 1]
        config4 = EnhancedMCTSConfig(epsilon_greedy=0.0)
        self.assertEqual(config4.epsilon_greedy, 0.0)
        
        config5 = EnhancedMCTSConfig(epsilon_greedy=1.0)
        self.assertEqual(config5.epsilon_greedy, 1.0)
        
        config6 = EnhancedMCTSConfig(epsilon_greedy=0.5)
        self.assertEqual(config6.epsilon_greedy, 0.5)
    
    def test_runtime_parameter_modification(self):
        """Test that parameters can be modified at runtime."""
        config = EnhancedMCTSConfig()
        
        # Modify alpha
        original_alpha = config.heuristic_alpha
        config.heuristic_alpha = 0.7
        self.assertEqual(config.heuristic_alpha, 0.7)
        self.assertNotEqual(config.heuristic_alpha, original_alpha)
        
        # Modify epsilon
        original_epsilon = config.epsilon_greedy
        config.epsilon_greedy = 0.8
        self.assertEqual(config.epsilon_greedy, 0.8)
        self.assertNotEqual(config.epsilon_greedy, original_epsilon)
        
        # Modify flags
        config.use_heuristic_guidance = False
        self.assertFalse(config.use_heuristic_guidance)
        
        config.use_heuristic_rollouts = False
        self.assertFalse(config.use_heuristic_rollouts)
    
    def test_phase_budget_multipliers_default(self):
        """Test that phase budget multipliers are initialized correctly."""
        config = EnhancedMCTSConfig()
        
        self.assertIsNotNone(config.phase_budget_multipliers)
        self.assertIn(GamePhase.EARLY, config.phase_budget_multipliers)
        self.assertIn(GamePhase.MID, config.phase_budget_multipliers)
        self.assertIn(GamePhase.LATE, config.phase_budget_multipliers)
        
        # Verify default values
        self.assertEqual(config.phase_budget_multipliers[GamePhase.EARLY], 1.0)
        self.assertEqual(config.phase_budget_multipliers[GamePhase.MID], 1.2)
        self.assertEqual(config.phase_budget_multipliers[GamePhase.LATE], 0.8)
    
    def test_custom_phase_budget_multipliers(self):
        """Test that custom phase budget multipliers can be set."""
        custom_multipliers = {
            GamePhase.EARLY: 1.5,
            GamePhase.MID: 1.0,
            GamePhase.LATE: 0.5
        }
        
        config = EnhancedMCTSConfig(phase_budget_multipliers=custom_multipliers)
        
        self.assertEqual(config.phase_budget_multipliers[GamePhase.EARLY], 1.5)
        self.assertEqual(config.phase_budget_multipliers[GamePhase.MID], 1.0)
        self.assertEqual(config.phase_budget_multipliers[GamePhase.LATE], 0.5)
    
    def test_all_standard_mcts_parameters(self):
        """Test that all standard MCTS parameters are configurable."""
        config = EnhancedMCTSConfig(
            num_simulations=200,
            late_simulations=150,
            simulation_switch_ply=25,
            c_puct=1.5,
            dirichlet_alpha=0.5,
            value_weight=1.5,
            max_depth=60
        )
        
        self.assertEqual(config.num_simulations, 200)
        self.assertEqual(config.late_simulations, 150)
        self.assertEqual(config.simulation_switch_ply, 25)
        self.assertEqual(config.c_puct, 1.5)
        self.assertEqual(config.dirichlet_alpha, 0.5)
        self.assertEqual(config.value_weight, 1.5)
        self.assertEqual(config.max_depth, 60)
    
    def test_temperature_parameters(self):
        """Test that temperature parameters are configurable."""
        config = EnhancedMCTSConfig(
            initial_temp=2.0,
            final_temp=0.05,
            annealing_steps=40,
            temp_clamp_fraction=0.9
        )
        
        self.assertEqual(config.initial_temp, 2.0)
        self.assertEqual(config.final_temp, 0.05)
        self.assertEqual(config.annealing_steps, 40)
        self.assertEqual(config.temp_clamp_fraction, 0.9)
    
    def test_heuristic_weight_parameter(self):
        """Test that heuristic_weight parameter is configurable."""
        config = EnhancedMCTSConfig(heuristic_weight=0.5)
        
        self.assertEqual(config.heuristic_weight, 0.5)
        
        # Default should be 0.3
        default_config = EnhancedMCTSConfig()
        self.assertEqual(default_config.heuristic_weight, 0.3)
    
    def test_parameter_combinations(self):
        """Test that different parameter combinations work together."""
        # Pure heuristic (alpha=1, epsilon=0)
        config1 = EnhancedMCTSConfig(
            heuristic_alpha=1.0,
            epsilon_greedy=0.0,
            use_heuristic_guidance=True,
            use_heuristic_rollouts=True
        )
        self.assertEqual(config1.heuristic_alpha, 1.0)
        self.assertEqual(config1.epsilon_greedy, 0.0)
        
        # Pure UCB1 (alpha=0, epsilon=1)
        config2 = EnhancedMCTSConfig(
            heuristic_alpha=0.0,
            epsilon_greedy=1.0,
            use_heuristic_guidance=True,
            use_heuristic_rollouts=True
        )
        self.assertEqual(config2.heuristic_alpha, 0.0)
        self.assertEqual(config2.epsilon_greedy, 1.0)
        
        # Balanced blend (default)
        config3 = EnhancedMCTSConfig()
        self.assertEqual(config3.heuristic_alpha, 0.3)
        self.assertEqual(config3.epsilon_greedy, 0.4)
    
    def test_configuration_immutability_after_creation(self):
        """Test that configuration can be modified after creation (not immutable)."""
        config = EnhancedMCTSConfig()
        
        # Should be able to modify
        config.heuristic_alpha = 0.8
        self.assertEqual(config.heuristic_alpha, 0.8)
        
        config.epsilon_greedy = 0.9
        self.assertEqual(config.epsilon_greedy, 0.9)
    
    def test_configuration_copy(self):
        """Test that configuration can be copied."""
        config1 = EnhancedMCTSConfig(
            heuristic_alpha=0.5,
            epsilon_greedy=0.6
        )
        
        # Create a copy by creating new config with same values
        config2 = EnhancedMCTSConfig(
            heuristic_alpha=config1.heuristic_alpha,
            epsilon_greedy=config1.epsilon_greedy
        )
        
        self.assertEqual(config2.heuristic_alpha, config1.heuristic_alpha)
        self.assertEqual(config2.epsilon_greedy, config1.epsilon_greedy)
        
        # Modify original - copy should be unaffected
        config1.heuristic_alpha = 0.9
        self.assertNotEqual(config2.heuristic_alpha, config1.heuristic_alpha)


if __name__ == '__main__':
    unittest.main()


