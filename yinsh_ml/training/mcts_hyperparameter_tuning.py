"""MCTS hyperparameter tuning and optimization framework.

This module implements automated hyperparameter tuning for the enhanced MCTS
using grid search and Bayesian optimization methods.
"""

import copy
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from enum import Enum

from ..game.game_state import GameState
from ..game.constants import Player
from ..training.enhanced_mcts import EnhancedMCTS, EnhancedMCTSConfig
from ..network.wrapper import NetworkWrapper
from ..analysis.phase_analyzer import GamePhase


def _to_json_safe(obj):
    """Recursively convert objects to JSON-serializable forms.

    - Enum keys/values -> their value or string
    - numpy scalars -> Python scalars
    - dict keys ensured to be strings
    """
    # Enum values
    if isinstance(obj, Enum):
        # Prefer the enum value if present, fallback to name
        return getattr(obj, "value", str(obj))

    # numpy scalar types
    if isinstance(obj, np.generic):
        return obj.item()

    # dicts: ensure keys are strings and values are converted
    if isinstance(obj, dict):
        converted = {}
        for k, v in obj.items():
            if isinstance(k, Enum):
                key_str = str(getattr(k, "value", str(k)))
            else:
                key_str = str(k)
            converted[key_str] = _to_json_safe(v)
        return converted

    # sequences
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]

    # dataclass dicts or plain python values pass through
    return obj

@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""
    
    # Exploration constant C (UCB1)
    c_puct_min: float = 0.5
    c_puct_max: float = 2.0
    c_puct_step: float = 0.25
    
    # Heuristic weight alpha
    heuristic_alpha_min: float = 0.1
    heuristic_alpha_max: float = 0.7
    heuristic_alpha_step: float = 0.1
    
    # Epsilon for greedy rollouts
    epsilon_greedy_min: float = 0.2
    epsilon_greedy_max: float = 0.6
    epsilon_greedy_step: float = 0.1
    
    # Simulation budget
    num_simulations_min: int = 1000
    num_simulations_max: int = 5000
    num_simulations_step: int = 1000
    
    # Simulation depth limits
    max_depth_min: int = 30
    max_depth_max: int = 80
    max_depth_step: int = 10
    
    def get_grid_search_space(self) -> List[Dict[str, Any]]:
        """Generate grid search parameter combinations."""
        c_puct_values = np.arange(self.c_puct_min, self.c_puct_max + self.c_puct_step, self.c_puct_step)
        heuristic_alpha_values = np.arange(self.heuristic_alpha_min, self.heuristic_alpha_max + self.heuristic_alpha_step, self.heuristic_alpha_step)
        epsilon_greedy_values = np.arange(self.epsilon_greedy_min, self.epsilon_greedy_max + self.epsilon_greedy_step, self.epsilon_greedy_step)
        num_simulations_values = range(self.num_simulations_min, self.num_simulations_max + self.num_simulations_step, self.num_simulations_step)
        max_depth_values = range(self.max_depth_min, self.max_depth_max + self.max_depth_step, self.max_depth_step)
        
        combinations = []
        for combo in itertools.product(c_puct_values, heuristic_alpha_values, epsilon_greedy_values, 
                                     num_simulations_values, max_depth_values):
            combinations.append({
                'c_puct': combo[0],
                'heuristic_alpha': combo[1],
                'epsilon_greedy': combo[2],
                'num_simulations': combo[3],
                'max_depth': combo[4]
            })
        
        return combinations


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    
    parameters: Dict[str, Any]
    win_rate: float
    move_time_avg: float
    move_time_std: float
    games_played: int
    evaluation_time: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Evaluation parameters
    evaluation_games: int = 20  # Number of games per parameter set
    max_move_time: float = 3.0  # Maximum allowed move time in seconds
    min_win_rate_improvement: float = 0.05  # Minimum win rate improvement for significance
    enforce_move_time_cap: bool = True  # If False, do not cap per-move time
    max_moves_per_game: int = 300  # Hard cap on moves to prevent infinite games
    
    # Optimization parameters
    use_bayesian_optimization: bool = False  # Use Bayesian optimization instead of grid search
    bayesian_iterations: int = 50  # Number of Bayesian optimization iterations
    bayesian_acquisition: str = 'EI'  # Acquisition function: 'EI', 'PI', 'UCB'
    
    # Statistical testing
    confidence_level: float = 0.95  # Confidence level for statistical tests
    min_games_for_significance: int = 10  # Minimum games for statistical significance
    
    # Parallel execution
    max_workers: int = 4  # Maximum number of parallel workers
    timeout_per_game: int = 300  # Timeout per game in seconds
    enforce_game_timeout: bool = False  # If True, enforce per-game timeout; default off


class MCTSHyperparameterTuner:
    """Main class for MCTS hyperparameter tuning."""
    
    def __init__(self, 
                 network: NetworkWrapper,
                 baseline_config: EnhancedMCTSConfig,
                 tuning_config: TuningConfig = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            network: Neural network wrapper
            baseline_config: Baseline MCTS configuration
            tuning_config: Tuning configuration
        """
        self.network = network
        self.baseline_config = baseline_config
        self.tuning_config = tuning_config or TuningConfig()
        self.logger = logging.getLogger("MCTSHyperparameterTuner")
        
        # Results storage
        self.results: List[TuningResult] = []
        self.best_result: Optional[TuningResult] = None
        
        # Hyperparameter space
        self.hyperparameter_space = HyperparameterSpace()
        
        self.logger.info("MCTS Hyperparameter Tuner Initialized")
        self.logger.info(f"Evaluation games per parameter set: {self.tuning_config.evaluation_games}")
        self.logger.info(f"Maximum move time: {self.tuning_config.max_move_time}s")
        self.logger.info(f"Max workers: {self.tuning_config.max_workers}")
    
    def _create_mcts_with_params(self, params: Dict[str, Any]) -> EnhancedMCTS:
        """Create MCTS instance with given parameters."""
        config = copy.deepcopy(self.baseline_config)
        
        # Update configuration with tuned parameters
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return EnhancedMCTS(self.network, config)
    
    def _evaluate_parameters(self, params: Dict[str, Any]) -> TuningResult:
        """Evaluate a single parameter set."""
        self.logger.info(f"Evaluating parameters: {params}")
        
        start_time = time.time()
        
        # Create MCTS with tuned parameters
        mcts = self._create_mcts_with_params(params)
        
        # Track performance metrics
        move_times = []
        wins = 0
        games_played = 0
        
        # Play evaluation games
        for game_id in range(self.tuning_config.evaluation_games):
            try:
                game_start = time.time()
                
                # Play a game using the tuned MCTS
                game_result = self._play_evaluation_game(mcts, game_id)
                
                game_time = time.time() - game_start
                move_times.extend(game_result['move_times'])
                
                if game_result['outcome'] > 0:  # Win
                    wins += 1
                
                games_played += 1
                
                # Optional per-game timeout (disabled by default)
                if self.tuning_config.enforce_game_timeout and game_time > self.tuning_config.timeout_per_game:
                    self.logger.warning(f"Game {game_id} timed out after {game_time:.2f}s")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in game {game_id}: {e}")
                continue
        
        evaluation_time = time.time() - start_time
        
        # Calculate metrics
        win_rate = wins / games_played if games_played > 0 else 0.0
        move_time_avg = np.mean(move_times) if move_times else float('inf')
        move_time_std = np.std(move_times) if move_times else 0.0
        
        result = TuningResult(
            parameters=params,
            win_rate=win_rate,
            move_time_avg=move_time_avg,
            move_time_std=move_time_std,
            games_played=games_played,
            evaluation_time=evaluation_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.logger.info(f"Results: Win rate={win_rate:.3f}, Avg move time={move_time_avg:.3f}s")
        
        return result
    
    def _play_evaluation_game(self, mcts: EnhancedMCTS, game_id: int) -> Dict[str, Any]:
        """Play a single evaluation game."""
        state = GameState()
        move_times = []
        move_count = 0
        
        while not state.is_terminal() and move_count < self.tuning_config.max_moves_per_game:
            move_start = time.time()
            
            # Get move from MCTS
            policy = mcts.search(state, move_count + 1)
            
            # Select move (greedy selection for evaluation)
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                break
            
            # Select move with highest probability
            move_probs = []
            for move in valid_moves:
                move_idx = mcts.state_encoder.move_to_index(move)
                if 0 <= move_idx < len(policy):
                    move_probs.append(policy[move_idx])
                else:
                    move_probs.append(0.0)
            
            if move_probs and max(move_probs) > 0:
                selected_idx = np.argmax(move_probs)
                selected_move = valid_moves[selected_idx]
            else:
                selected_move = valid_moves[0]  # Fallback
            
            # Make move
            state.make_move(selected_move)
            move_count += 1
            
            move_time = time.time() - move_start
            move_times.append(move_time)
            
            # Check move time limit
            if self.tuning_config.enforce_move_time_cap and move_time > self.tuning_config.max_move_time:
                self.logger.warning(f"Move {move_count} exceeded time limit: {move_time:.3f}s")
                break
        
        # Determine outcome
        if state.is_terminal():
            winner = state.get_winner()
            if winner == Player.WHITE:
                outcome = 1.0
            elif winner == Player.BLACK:
                outcome = -1.0
            else:
                # Terminal draw
                outcome = 0.5
        else:
            # Non-terminal (move cap or time cap) -> score as draw
            outcome = 0.5
        
        return {
            'outcome': outcome,
            'move_times': move_times,
            'total_moves': move_count
        }
    
    def grid_search(self) -> List[TuningResult]:
        """Perform grid search over hyperparameter space."""
        self.logger.info("Starting grid search hyperparameter tuning")
        
        # Get parameter combinations
        param_combinations = self.hyperparameter_space.get_grid_search_space()
        total_combinations = len(param_combinations)
        
        self.logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Evaluate parameters in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.tuning_config.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_params = {
                executor.submit(self._evaluate_parameters, params): params 
                for params in param_combinations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                    self.results.append(result)
                    
                    self.logger.info(f"Completed {len(results)}/{total_combinations} evaluations")
                    # Periodic autosave to avoid losing long-run progress
                    if len(self.results) % 25 == 0:
                        try:
                            self.save_results('mcts_tuning_autosave.json')
                        except Exception as e:
                            self.logger.warning(f"Autosave failed: {e}")
                    
                except Exception as e:
                    params = future_to_params[future]
                    self.logger.error(f"Error evaluating parameters {params}: {e}")
        
        # Sort results by win rate
        results.sort(key=lambda x: x.win_rate, reverse=True)
        
        # Find best result
        if results:
            self.best_result = results[0]
            self.logger.info(f"Best parameters: {self.best_result.parameters}")
            self.logger.info(f"Best win rate: {self.best_result.win_rate:.3f}")
            self.logger.info(f"Best move time: {self.best_result.move_time_avg:.3f}s")
        
        return results
    
    def bayesian_optimization(self) -> List[TuningResult]:
        """Perform Bayesian optimization over hyperparameter space."""
        self.logger.info("Starting Bayesian optimization hyperparameter tuning")
        
        # This is a simplified implementation
        # In practice, you'd use a library like scikit-optimize or optuna
        
        results = []
        
        # Initialize with random samples
        n_init = 10
        for i in range(n_init):
            params = self._sample_random_parameters()
            result = self._evaluate_parameters(params)
            results.append(result)
        
        # Bayesian optimization loop
        for iteration in range(self.tuning_config.bayesian_iterations):
            # Fit Gaussian Process
            X = np.array([[r.parameters['c_puct'], r.parameters['heuristic_alpha'], 
                          r.parameters['epsilon_greedy']] for r in results])
            y = np.array([r.win_rate for r in results])
            
            # Simple RBF kernel
            kernel = C(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(X, y)
            
            # Acquisition function (simplified)
            # In practice, use Expected Improvement or other acquisition functions
            next_params = self._sample_random_parameters()
            
            # Evaluate next parameters
            result = self._evaluate_parameters(next_params)
            results.append(result)
            
            self.logger.info(f"Bayesian iteration {iteration + 1}/{self.tuning_config.bayesian_iterations}")
        
        # Sort results by win rate
        results.sort(key=lambda x: x.win_rate, reverse=True)
        
        # Find best result
        if results:
            self.best_result = results[0]
            self.logger.info(f"Best parameters: {self.best_result.parameters}")
            self.logger.info(f"Best win rate: {self.best_result.win_rate:.3f}")
        
        return results
    
    def _sample_random_parameters(self) -> Dict[str, Any]:
        """Sample random parameters from hyperparameter space."""
        return {
            'c_puct': np.random.uniform(self.hyperparameter_space.c_puct_min, 
                                       self.hyperparameter_space.c_puct_max),
            'heuristic_alpha': np.random.uniform(self.hyperparameter_space.heuristic_alpha_min,
                                               self.hyperparameter_space.heuristic_alpha_max),
            'epsilon_greedy': np.random.uniform(self.hyperparameter_space.epsilon_greedy_min,
                                              self.hyperparameter_space.epsilon_greedy_max),
            'num_simulations': np.random.randint(self.hyperparameter_space.num_simulations_min,
                                               self.hyperparameter_space.num_simulations_max + 1),
            'max_depth': np.random.randint(self.hyperparameter_space.max_depth_min,
                                         self.hyperparameter_space.max_depth_max + 1)
        }
    
    def statistical_significance_test(self, result1: TuningResult, result2: TuningResult) -> Dict[str, Any]:
        """Perform statistical significance test between two results."""
        # This is a simplified implementation
        # In practice, you'd use proper statistical tests
        
        win_rate_diff = result1.win_rate - result2.win_rate
        
        # Simple significance test based on sample size
        n1, n2 = result1.games_played, result2.games_played
        
        if n1 < self.tuning_config.min_games_for_significance or n2 < self.tuning_config.min_games_for_significance:
            return {
                'significant': False,
                'reason': 'Insufficient sample size',
                'win_rate_diff': win_rate_diff,
                'n1': n1,
                'n2': n2
            }
        
        # Simplified significance test
        # In practice, use proper statistical tests like chi-square or t-test
        significant = abs(win_rate_diff) > self.tuning_config.min_win_rate_improvement
        
        return {
            'significant': significant,
            'win_rate_diff': win_rate_diff,
            'n1': n1,
            'n2': n2,
            'confidence_level': self.tuning_config.confidence_level
        }
    
    def save_results(self, filepath: str):
        """Save tuning results to file."""
        results_data = {
            'tuning_config': _to_json_safe(asdict(self.tuning_config)),
            'hyperparameter_space': _to_json_safe(asdict(self.hyperparameter_space)),
            # baseline_config may contain Enum keys (e.g., GamePhase) -> make JSON-safe
            'baseline_config': _to_json_safe(asdict(self.baseline_config)),
            'results': [_to_json_safe(result.to_dict()) for result in self.results],
            'best_result': _to_json_safe(self.best_result.to_dict()) if self.best_result else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load tuning results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = [TuningResult(**result) for result in data['results']]
        if data['best_result']:
            self.best_result = TuningResult(**data['best_result'])
        
        self.logger.info(f"Results loaded from {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive tuning report."""
        if not self.results:
            return "No results available for report generation."
        
        report = []
        report.append("MCTS Hyperparameter Tuning Report")
        report.append("=" * 50)
        report.append(f"Total parameter combinations tested: {len(self.results)}")
        report.append(f"Evaluation games per combination: {self.tuning_config.evaluation_games}")
        report.append(f"Maximum move time limit: {self.tuning_config.max_move_time}s")
        report.append("")
        
        # Best result
        if self.best_result:
            report.append("Best Parameters:")
            report.append("-" * 20)
            for param, value in self.best_result.parameters.items():
                report.append(f"  {param}: {value}")
            report.append(f"  Win Rate: {self.best_result.win_rate:.3f}")
            report.append(f"  Avg Move Time: {self.best_result.move_time_avg:.3f}s")
            report.append(f"  Games Played: {self.best_result.games_played}")
            report.append("")
        
        # Top 5 results
        report.append("Top 5 Parameter Combinations:")
        report.append("-" * 30)
        for i, result in enumerate(self.results[:5]):
            report.append(f"{i+1}. Win Rate: {result.win_rate:.3f}, Move Time: {result.move_time_avg:.3f}s")
            report.append(f"   Parameters: {result.parameters}")
            report.append("")
        
        # Statistical summary
        win_rates = [r.win_rate for r in self.results]
        move_times = [r.move_time_avg for r in self.results]
        
        report.append("Statistical Summary:")
        report.append("-" * 20)
        report.append(f"Win Rate - Mean: {np.mean(win_rates):.3f}, Std: {np.std(win_rates):.3f}")
        report.append(f"Move Time - Mean: {np.mean(move_times):.3f}s, Std: {np.std(move_times):.3f}s")
        report.append(f"Best Win Rate: {np.max(win_rates):.3f}")
        report.append(f"Best Move Time: {np.min(move_times):.3f}s")
        
        return "\n".join(report)
