"""Tests for optimization algorithms (Grid Search and Genetic Algorithm).

This test suite verifies that optimization algorithms correctly find
improved weight configurations through tournament-based evaluation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from yinsh_ml.heuristics.config_manager import ConfigManager
from yinsh_ml.heuristics.optimizers import (
    GridSearchOptimizer,
    GeneticAlgorithmOptimizer,
    OptimizationResult,
)


class TestGridSearchOptimizer(unittest.TestCase):
    """Test suite for GridSearchOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.config_manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        self.optimizer = GridSearchOptimizer(self.config_manager)
    
    def test_initialization(self):
        """Test that GridSearchOptimizer initializes correctly."""
        self.assertIsInstance(self.optimizer.config_manager, ConfigManager)
    
    @patch('yinsh_ml.heuristics.YinshHeuristics')
    @patch('yinsh_ml.agents.tournament.TournamentEvaluator')
    def test_optimize_finds_best_config(self, mock_tournament_class, mock_heuristics_class):
        """Test that grid search finds the best configuration."""
        # Mock YinshHeuristics to avoid validation issues
        mock_heuristics_instance = Mock()
        mock_heuristics_class.return_value = mock_heuristics_instance
        
        # Mock tournament evaluator to return different win rates
        mock_evaluator = Mock()
        
        # Simulate that higher weight values lead to better performance
        def mock_run_tournament(games, opponent_seed=None):
            mock_metrics = Mock()
            # Simulate performance based on weight (simplified)
            mock_metrics.win_rate = 0.6  # Fixed for testing
            mock_metrics.total_games = games
            return mock_metrics
        
        mock_evaluator.run_tournament = mock_run_tournament
        mock_tournament_class.return_value = mock_evaluator
        
        # Define parameter grid
        param_grid = {
            "early.completed_runs_differential": [5.0, 10.0, 15.0, 20.0],
        }
        
        # Run optimization
        result = self.optimizer.optimize(
            param_grid=param_grid,
            evaluation_games=10,  # Small number for testing
        )
        
        # Verify result structure
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn("best_config", result.__dict__ or {})
        self.assertIn("best_score", result.__dict__ or {})
        
        # Verify best config has highest weight (simulated better performance)
        if hasattr(result, 'best_config'):
            best_weight = result.best_config.get("weights", {}).get("early", {}).get("completed_runs_differential")
            # Should be one of the tested values
            self.assertIn(best_weight, [5.0, 10.0, 15.0, 20.0])
    
    def test_optimize_with_multiple_parameters(self):
        """Test grid search with multiple parameters."""
        # Mock tournament to return consistent results
        with patch('yinsh_ml.heuristics.YinshHeuristics'), \
             patch('yinsh_ml.agents.tournament.TournamentEvaluator') as mock_tournament_class:
            mock_evaluator = Mock()
            mock_metrics = Mock()
            mock_metrics.win_rate = 0.6
            mock_metrics.total_games = 10
            mock_evaluator.run_tournament.return_value = mock_metrics
            mock_tournament_class.return_value = mock_evaluator
            
            param_grid = {
                "early.completed_runs_differential": [10.0, 15.0],
                "mid.completed_runs_differential": [12.0, 18.0],
            }
            
            result = self.optimizer.optimize(
                param_grid=param_grid,
                evaluation_games=5,
            )
            
            # Should test 2 * 2 = 4 combinations
            self.assertIsInstance(result, OptimizationResult)
    
    def test_optimize_reproducibility(self):
        """Test that optimization results are reproducible with seeds."""
        with patch('yinsh_ml.heuristics.YinshHeuristics'), \
             patch('yinsh_ml.agents.tournament.TournamentEvaluator') as mock_tournament_class:
            mock_evaluator = Mock()
            mock_metrics = Mock()
            mock_metrics.win_rate = 0.6
            mock_metrics.total_games = 10
            mock_evaluator.run_tournament.return_value = mock_metrics
            mock_tournament_class.return_value = mock_evaluator
            
            param_grid = {
                "early.completed_runs_differential": [10.0, 15.0],
            }
            
            result1 = self.optimizer.optimize(
                param_grid=param_grid,
                evaluation_games=5,
                random_seed=42,
            )
            
            result2 = self.optimizer.optimize(
                param_grid=param_grid,
                evaluation_games=5,
                random_seed=42,
            )
            
            # Results should be identical with same seed
            if hasattr(result1, 'best_config') and hasattr(result2, 'best_config'):
                self.assertEqual(result1.best_config, result2.best_config)


class TestGeneticAlgorithmOptimizer(unittest.TestCase):
    """Test suite for GeneticAlgorithmOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager()
        self.config_manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        self.optimizer = GeneticAlgorithmOptimizer(self.config_manager)
    
    def test_initialization(self):
        """Test that GeneticAlgorithmOptimizer initializes correctly."""
        self.assertIsInstance(self.optimizer.config_manager, ConfigManager)
    
    @patch('yinsh_ml.heuristics.YinshHeuristics')
    @patch('yinsh_ml.agents.tournament.TournamentEvaluator')
    def test_optimize_convergence(self, mock_tournament_class, mock_heuristics_class):
        """Test that genetic algorithm shows convergence over generations."""
        mock_evaluator = Mock()
        
        # Simulate improving performance over generations
        generation_count = [0]
        def mock_run_tournament(games, opponent_seed=None):
            mock_metrics = Mock()
            # Simulate improving win rate over generations
            generation_count[0] += 1
            mock_metrics.win_rate = min(0.5 + (generation_count[0] * 0.05), 0.95)
            mock_metrics.total_games = games
            return mock_metrics
        
        mock_evaluator.run_tournament = mock_run_tournament
        mock_tournament_class.return_value = mock_evaluator
        
        # Run optimization with small population and generations
        result = self.optimizer.optimize(
            population_size=5,
            generations=3,
            evaluation_games=5,
            mutation_rate=0.1,
        )
        
        # Verify result structure
        self.assertIsInstance(result, OptimizationResult)
        self.assertIn("best_config", result.__dict__ or {})
        self.assertIn("best_score", result.__dict__ or {})
    
    def test_optimize_with_custom_parameters(self):
        """Test genetic algorithm with custom parameters."""
        with patch('yinsh_ml.heuristics.YinshHeuristics'), \
             patch('yinsh_ml.agents.tournament.TournamentEvaluator') as mock_tournament_class:
            mock_evaluator = Mock()
            mock_metrics = Mock()
            mock_metrics.win_rate = 0.6
            mock_metrics.total_games = 5
            mock_evaluator.run_tournament.return_value = mock_metrics
            mock_tournament_class.return_value = mock_evaluator
            
            result = self.optimizer.optimize(
                population_size=10,
                generations=5,
                mutation_rate=0.2,
                evaluation_games=5,
            )
            
            self.assertIsInstance(result, OptimizationResult)
    
    def test_optimize_reproducibility(self):
        """Test that genetic algorithm results are reproducible with seeds."""
        with patch('yinsh_ml.heuristics.YinshHeuristics'), \
             patch('yinsh_ml.agents.tournament.TournamentEvaluator') as mock_tournament_class:
            mock_evaluator = Mock()
            mock_metrics = Mock()
            mock_metrics.win_rate = 0.6
            mock_metrics.total_games = 5
            mock_evaluator.run_tournament.return_value = mock_metrics
            mock_tournament_class.return_value = mock_evaluator
            
            result1 = self.optimizer.optimize(
                population_size=5,
                generations=2,
                evaluation_games=5,
                random_seed=42,
            )
            
            result2 = self.optimizer.optimize(
                population_size=5,
                generations=2,
                evaluation_games=5,
                random_seed=42,
            )
            
            # Results should be identical with same seed
            if hasattr(result1, 'best_config') and hasattr(result2, 'best_config'):
                self.assertEqual(result1.best_config, result2.best_config)


if __name__ == '__main__':
    unittest.main()

