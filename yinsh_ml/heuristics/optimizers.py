"""Optimization algorithms for heuristic weight tuning.

This module provides grid search and genetic algorithm optimizers for
automatically tuning heuristic weights through tournament-based evaluation.
"""

import itertools
import random
import copy
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from an optimization run."""
    best_config: Dict[str, Any]
    """Best configuration found (weights + phase_config)."""
    
    best_score: float
    """Best performance score (win rate)."""
    
    all_results: List[Tuple[Dict[str, Any], float]]
    """All configurations tested with their scores."""
    
    num_evaluations: int
    """Total number of configurations evaluated."""
    
    optimization_method: str
    """Method used: 'grid_search' or 'genetic_algorithm'."""


class GridSearchOptimizer:
    """Grid search optimizer for heuristic weights.
    
    Systematically tests all combinations of parameter values to find
    the optimal configuration.
    
    Example:
        >>> optimizer = GridSearchOptimizer(config_manager)
        >>> param_grid = {
        ...     "early.completed_runs_differential": [10.0, 15.0, 20.0],
        ...     "mid.completed_runs_differential": [12.0, 18.0],
        ... }
        >>> result = optimizer.optimize(param_grid, evaluation_games=50)
        >>> print(f"Best win rate: {result.best_score}")
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize grid search optimizer.
        
        Args:
            config_manager: ConfigManager instance for managing configurations
        """
        self.config_manager = config_manager
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        num_games: int,
    ) -> float:
        """Evaluate a configuration using tournament games.
        
        Args:
            config: Configuration dictionary with weights
            num_games: Number of games to play
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        from ..agents.tournament import TournamentEvaluator
        from ..agents.heuristic_agent import HeuristicAgentConfig, HeuristicAgent
        from ..heuristics import YinshHeuristics
        from ..heuristics.phase_config import PhaseConfig
        
        # Create evaluator with custom weights
        phase_config = PhaseConfig.from_dict(config.get("phase_config", {}))
        evaluator = YinshHeuristics(
            weights=config["weights"],
            phase_config=phase_config,
        )
        
        # Create agent config
        agent_config = HeuristicAgentConfig()
        
        # Create agent factory that uses the custom evaluator
        def agent_factory(cfg):
            return HeuristicAgent(config=cfg, evaluator=evaluator)
        
        tournament = TournamentEvaluator(
            heuristic_agent_factory=agent_factory,
            heuristic_config=agent_config,
        )
        
        # Run tournament
        metrics = tournament.run_tournament(games=num_games)
        return metrics.win_rate
    
    def optimize(
        self,
        param_grid: Dict[str, List[float]],
        evaluation_games: int = 50,
        random_seed: Optional[int] = None,
    ) -> OptimizationResult:
        """Run grid search optimization.
        
        Args:
            param_grid: Dictionary mapping parameter paths to lists of values.
                       Parameter paths use dot notation, e.g.,
                       "early.completed_runs_differential"
            evaluation_games: Number of games to play for each configuration
            random_seed: Optional random seed for reproducibility
            
        Returns:
            OptimizationResult with best configuration and all results
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        logger.info(f"Starting grid search with {len(param_grid)} parameters")
        
        # Parse parameter grid into structured format
        parsed_params = self._parse_param_grid(param_grid)
        
        # Generate all combinations
        param_names = list(parsed_params.keys())
        param_values = list(parsed_params.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        all_results = []
        best_config = None
        best_score = -1.0
        
        # Evaluate each combination
        for i, combination in enumerate(combinations):
            # Create config from combination
            test_config = self._create_config_from_params(
                param_names,
                combination,
            )
            
            # Evaluate configuration
            score = self._evaluate_configuration(
                test_config,
                evaluation_games,
            )
            
            all_results.append((test_config, score))
            
            if score > best_score:
                best_score = score
                best_config = test_config
            
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(combinations)} "
                    f"(best score: {best_score:.3f})"
                )
        
        return OptimizationResult(
            best_config=best_config or {},
            best_score=best_score,
            all_results=all_results,
            num_evaluations=len(combinations),
            optimization_method="grid_search",
        )
    
    def _parse_param_grid(
        self,
        param_grid: Dict[str, List[float]]
    ) -> Dict[Tuple[str, str], List[float]]:
        """Parse parameter grid into structured format.
        
        Args:
            param_grid: Dictionary with dot-notation keys like "early.feature"
            
        Returns:
            Dictionary mapping (phase, feature) tuples to value lists
        """
        parsed = {}
        for param_path, values in param_grid.items():
            parts = param_path.split(".")
            if len(parts) != 2:
                raise ValueError(
                    f"Parameter path must be 'phase.feature', got '{param_path}'"
                )
            phase, feature = parts
            parsed[(phase, feature)] = values
        return parsed
    
    def _create_config_from_params(
        self,
        param_names: List[Tuple[str, str]],
        combination: Tuple[float, ...],
    ) -> Dict[str, Any]:
        """Create a configuration dictionary from parameter combination.
        
        Args:
            param_names: List of (phase, feature) tuples
            combination: Tuple of values for each parameter
            
        Returns:
            Configuration dictionary with weights
        """
        # Start with current config
        current_config = self.config_manager.get_current_config()
        config = copy.deepcopy(current_config)
        
        # Update weights based on combination
        for (phase, feature), value in zip(param_names, combination):
            if phase not in config["weights"]:
                config["weights"][phase] = {}
            config["weights"][phase][feature] = value
        
        return config


class GeneticAlgorithmOptimizer:
    """Genetic algorithm optimizer for heuristic weights.
    
    Uses evolutionary algorithms to evolve weight configurations,
    selecting and breeding high-performing configurations.
    
    Example:
        >>> optimizer = GeneticAlgorithmOptimizer(config_manager)
        >>> result = optimizer.optimize(
        ...     population_size=20,
        ...     generations=10,
        ...     evaluation_games=50,
        ... )
        >>> print(f"Best win rate: {result.best_score}")
    """
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize genetic algorithm optimizer.
        
        Args:
            config_manager: ConfigManager instance for managing configurations
        """
        self.config_manager = config_manager
    
    def optimize(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        evaluation_games: int = 50,
        random_seed: Optional[int] = None,
        elite_size: int = 2,
    ) -> OptimizationResult:
        """Run genetic algorithm optimization.
        
        Args:
            population_size: Number of configurations in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutating each weight (0.0 to 1.0)
            evaluation_games: Number of games to play for each configuration
            random_seed: Optional random seed for reproducibility
            elite_size: Number of top configurations to preserve unchanged
            
        Returns:
            OptimizationResult with best configuration and evolution history
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        logger.info(
            f"Starting genetic algorithm: "
            f"population={population_size}, generations={generations}"
        )
        
        # Initialize population
        population = self._initialize_population(population_size)
        
        all_results = []
        best_config = None
        best_score = -1.0
        
        # Evolve for N generations
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate population
            evaluated = []
            for config in population:
                score = self._evaluate_configuration(config, evaluation_games)
                evaluated.append((config, score))
                all_results.append((copy.deepcopy(config), score))
                
                if score > best_score:
                    best_score = score
                    best_config = copy.deepcopy(config)
            
            # Sort by fitness (win rate)
            evaluated.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(
                f"Generation {generation + 1}: "
                f"best={evaluated[0][1]:.3f}, "
                f"avg={sum(s for _, s in evaluated) / len(evaluated):.3f}"
            )
            
            # Select elite and create next generation
            elite = [config for config, _ in evaluated[:elite_size]]
            next_population = elite.copy()
            
            # Breed to fill remaining slots
            while len(next_population) < population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_select(evaluated, tournament_size=3)
                parent2 = self._tournament_select(evaluated, tournament_size=3)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutate
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                
                next_population.append(child)
            
            population = next_population
        
        return OptimizationResult(
            best_config=best_config or {},
            best_score=best_score,
            all_results=all_results,
            num_evaluations=len(all_results),
            optimization_method="genetic_algorithm",
        )
    
    def _initialize_population(self, size: int) -> List[Dict[str, Any]]:
        """Initialize random population of configurations.
        
        Args:
            size: Population size
            
        Returns:
            List of random configuration dictionaries
        """
        current_config = self.config_manager.get_current_config()
        population = []
        
        for _ in range(size):
            config = copy.deepcopy(current_config)
            
            # Randomize weights within valid ranges
            for phase in ["early", "mid", "late"]:
                if phase not in config["weights"]:
                    config["weights"][phase] = {}
                
                for feature in [
                    "completed_runs_differential",
                    "potential_runs_count",
                    "connected_marker_chains",
                    "ring_positioning",
                    "ring_spread",
                    "board_control",
                ]:
                    # Random value between 0 and 50 (valid range)
                    config["weights"][phase][feature] = random.uniform(0.0, 50.0)
            
            population.append(config)
        
        return population
    
    def _tournament_select(
        self,
        evaluated: List[Tuple[Dict[str, Any], float]],
        tournament_size: int = 3,
    ) -> Dict[str, Any]:
        """Select a configuration using tournament selection.
        
        Args:
            evaluated: List of (config, score) tuples
            tournament_size: Number of random candidates to compare
            
        Returns:
            Selected configuration
        """
        tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
        return max(tournament, key=lambda x: x[1])[0]
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create child configuration by blending parents.
        
        Args:
            parent1: First parent configuration
            parent2: Second parent configuration
            
        Returns:
            Child configuration
        """
        child = copy.deepcopy(parent1)
        
        # Blend weights: weighted average
        alpha = random.random()  # Blend factor
        
        for phase in ["early", "mid", "late"]:
            if phase not in child["weights"]:
                child["weights"][phase] = {}
            
            for feature in child["weights"].get(phase, {}).keys():
                if phase in parent2["weights"] and feature in parent2["weights"][phase]:
                    val1 = child["weights"][phase].get(feature, 0.0)
                    val2 = parent2["weights"][phase].get(feature, 0.0)
                    child["weights"][phase][feature] = alpha * val1 + (1 - alpha) * val2
        
        return child
    
    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate configuration by randomly perturbing weights.
        
        Args:
            config: Configuration to mutate
            
        Returns:
            Mutated configuration
        """
        mutated = copy.deepcopy(config)
        
        # Randomly mutate some weights
        for phase in ["early", "mid", "late"]:
            if phase not in mutated["weights"]:
                continue
            
            for feature in mutated["weights"][phase].keys():
                if random.random() < 0.3:  # 30% chance per weight
                    # Perturb by ±20%
                    current = mutated["weights"][phase][feature]
                    perturbation = random.uniform(-0.2, 0.2)
                    new_value = max(0.0, min(50.0, current * (1 + perturbation)))
                    mutated["weights"][phase][feature] = new_value
        
        return mutated
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        num_games: int,
    ) -> float:
        """Evaluate a configuration using tournament games.
        
        Args:
            config: Configuration dictionary with weights
            num_games: Number of games to play
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        from ..agents.tournament import TournamentEvaluator
        from ..agents.heuristic_agent import HeuristicAgentConfig, HeuristicAgent
        from ..heuristics import YinshHeuristics
        from ..heuristics.phase_config import PhaseConfig
        
        # Create evaluator with custom weights
        phase_config = PhaseConfig.from_dict(config.get("phase_config", {}))
        evaluator = YinshHeuristics(
            weights=config["weights"],
            phase_config=phase_config,
        )
        
        # Create agent config
        agent_config = HeuristicAgentConfig()
        
        # Create agent factory that uses the custom evaluator
        def agent_factory(cfg):
            return HeuristicAgent(config=cfg, evaluator=evaluator)
        
        tournament = TournamentEvaluator(
            heuristic_agent_factory=agent_factory,
            heuristic_config=agent_config,
        )
        
        # Run tournament
        metrics = tournament.run_tournament(games=num_games)
        return metrics.win_rate

