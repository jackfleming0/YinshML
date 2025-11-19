"""Performance profiling and benchmarking for heuristic MCTS integration.

This module provides performance analysis tools to measure the impact of
heuristic integration on MCTS search speed and quality.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from ..game.game_state import GameState
from ..game.constants import Player
from .mcts import MCTS, MCTSConfig, EvaluationMode
from ..network.wrapper import NetworkWrapper


@dataclass
class SearchPerformanceMetrics:
    """Performance metrics for a single evaluation."""
    evaluation_time: float
    nodes_per_second: float
    evaluation_mode: str
    simulation_budget: int
    depth_reached: int


@dataclass
class BenchmarkResults:
    """Results from performance benchmarking."""
    mode: str
    total_evaluations: int
    total_time: float
    avg_evaluation_time: float
    avg_nodes_per_second: float
    min_evaluation_time: float
    max_evaluation_time: float
    std_evaluation_time: float


class MCTSPerformanceProfiler:
    """Profiler for MCTS performance analysis."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.logger = logging.getLogger("MCTSPerformanceProfiler")
        self.metrics_history: List[PerformanceMetrics] = []
    
    def profile_search(self,
                      mcts: MCTS,
                      state: GameState,
                      move_number: int,
                      num_runs: int = 10) -> Dict[str, Any]:
        """
        Profile MCTS search performance.
        
        Args:
            mcts: MCTS instance to profile
            state: Game state to search from
            move_number: Current move number
            num_runs: Number of search runs to average
            
        Returns:
            Dictionary with performance metrics
        """
        evaluation_times = []
        nodes_per_second_list = []
        
        for run in range(num_runs):
            start_time = time.time()
            policy = mcts.search(state, move_number)
            elapsed_time = time.time() - start_time
            
            # Calculate nodes per second (approximate)
            budget = mcts._get_simulation_budget(move_number)
            nodes_per_second = budget / elapsed_time if elapsed_time > 0 else 0
            
            evaluation_times.append(elapsed_time)
            nodes_per_second_list.append(nodes_per_second)
        
        return {
            'avg_time': np.mean(evaluation_times),
            'std_time': np.std(evaluation_times),
            'min_time': np.min(evaluation_times),
            'max_time': np.max(evaluation_times),
            'avg_nodes_per_second': np.mean(nodes_per_second_list),
            'evaluation_mode': mcts.config.evaluation_mode.value,
            'simulation_budget': mcts._get_simulation_budget(move_number)
        }
    
    def compare_evaluation_modes(self,
                                 network: NetworkWrapper,
                                 state: GameState,
                                 move_number: int,
                                 simulation_budget: int = 100,
                                 num_runs: int = 10) -> Dict[str, BenchmarkResults]:
        """
        Compare performance across different evaluation modes.
        
        Args:
            network: Neural network wrapper
            state: Game state to evaluate
            move_number: Current move number
            simulation_budget: Number of simulations per search
            num_runs: Number of runs per mode
            
        Returns:
            Dictionary mapping evaluation mode to benchmark results
        """
        results = {}
        
        modes = [
            EvaluationMode.PURE_NEURAL,
            EvaluationMode.PURE_HEURISTIC,
            EvaluationMode.HYBRID
        ]
        
        for mode in modes:
            config = MCTSConfig(
                num_simulations=simulation_budget,
                evaluation_mode=mode,
                use_heuristic_evaluation=(mode != EvaluationMode.PURE_NEURAL)
            )
            
            mcts = MCTS(network, config=config)
            
            # Profile this mode
            evaluation_times = []
            for _ in range(num_runs):
                start_time = time.time()
                mcts.search(state, move_number)
                elapsed_time = time.time() - start_time
                evaluation_times.append(elapsed_time)
            
            # Calculate nodes per second
            nodes_per_second_list = [
                simulation_budget / t if t > 0 else 0
                for t in evaluation_times
            ]
            
            results[mode.value] = BenchmarkResults(
                mode=mode.value,
                total_evaluations=num_runs,
                total_time=sum(evaluation_times),
                avg_evaluation_time=np.mean(evaluation_times),
                avg_nodes_per_second=np.mean(nodes_per_second_list),
                min_evaluation_time=np.min(evaluation_times),
                max_evaluation_time=np.max(evaluation_times),
                std_evaluation_time=np.std(evaluation_times)
            )
        
        return results
    
    def benchmark_heuristic_weight_impact(self,
                                         network: NetworkWrapper,
                                         state: GameState,
                                         move_number: int,
                                         weights: List[float],
                                         simulation_budget: int = 100,
                                         num_runs: int = 5) -> Dict[float, BenchmarkResults]:
        """
        Benchmark performance impact of different heuristic weights.
        
        Args:
            network: Neural network wrapper
            state: Game state to evaluate
            move_number: Current move number
            weights: List of heuristic weights to test
            simulation_budget: Number of simulations per search
            num_runs: Number of runs per weight
            
        Returns:
            Dictionary mapping heuristic weight to benchmark results
        """
        results = {}
        
        for weight in weights:
            config = MCTSConfig(
                num_simulations=simulation_budget,
                evaluation_mode=EvaluationMode.HYBRID,
                heuristic_weight=weight,
                neural_weight=1.0 - weight,
                use_heuristic_evaluation=True
            )
            
            mcts = MCTS(network, config=config)
            
            # Profile this weight
            evaluation_times = []
            for _ in range(num_runs):
                start_time = time.time()
                mcts.search(state, move_number)
                elapsed_time = time.time() - start_time
                evaluation_times.append(elapsed_time)
            
            nodes_per_second_list = [
                simulation_budget / t if t > 0 else 0
                for t in evaluation_times
            ]
            
            results[weight] = BenchmarkResults(
                mode=f"hybrid_weight_{weight}",
                total_evaluations=num_runs,
                total_time=sum(evaluation_times),
                avg_evaluation_time=np.mean(evaluation_times),
                avg_nodes_per_second=np.mean(nodes_per_second_list),
                min_evaluation_time=np.min(evaluation_times),
                max_evaluation_time=np.max(evaluation_times),
                std_evaluation_time=np.std(evaluation_times)
            )
        
        return results
    
    def print_benchmark_results(self, results: Dict[str, BenchmarkResults]):
        """Print benchmark results in a readable format."""
        print("\n" + "="*80)
        print("MCTS Performance Benchmark Results")
        print("="*80)
        
        for mode, result in results.items():
            print(f"\n{mode.upper()}:")
            print(f"  Average Evaluation Time: {result.avg_evaluation_time*1000:.2f} ms")
            print(f"  Std Dev: {result.std_evaluation_time*1000:.2f} ms")
            print(f"  Min/Max: {result.min_evaluation_time*1000:.2f} / {result.max_evaluation_time*1000:.2f} ms")
            print(f"  Average Nodes/Second: {result.avg_nodes_per_second:.1f}")
            print(f"  Total Evaluations: {result.total_evaluations}")
        
        print("\n" + "="*80)
    
    def print_weight_impact_results(self, results: Dict[float, BenchmarkResults]):
        """Print heuristic weight impact results."""
        print("\n" + "="*80)
        print("Heuristic Weight Impact Analysis")
        print("="*80)
        
        for weight, result in sorted(results.items()):
            print(f"\nHeuristic Weight: {weight:.2f}")
            print(f"  Average Evaluation Time: {result.avg_evaluation_time*1000:.2f} ms")
            print(f"  Average Nodes/Second: {result.avg_nodes_per_second:.1f}")
        
        print("\n" + "="*80)


def run_performance_benchmark(network: NetworkWrapper,
                             states: List[GameState],
                             move_numbers: List[int],
                             simulation_budget: int = 100,
                             num_runs: int = 5) -> Dict[str, Any]:
    """
    Run comprehensive performance benchmark across multiple game states.
    
    Args:
        network: Neural network wrapper
        states: List of game states to benchmark
        move_numbers: List of move numbers corresponding to states
        simulation_budget: Number of simulations per search
        num_runs: Number of runs per state/mode combination
        
    Returns:
        Dictionary with comprehensive benchmark results
    """
    profiler = MCTSPerformanceProfiler()
    
    all_results = {
        'pure_neural': [],
        'pure_heuristic': [],
        'hybrid': []
    }
    
    for state, move_number in zip(states, move_numbers):
        # Compare all modes for this state
        mode_results = profiler.compare_evaluation_modes(
            network, state, move_number, simulation_budget, num_runs
        )
        
        for mode, result in mode_results.items():
            all_results[mode].append(result)
    
    # Aggregate results
    aggregated = {}
    for mode, results in all_results.items():
        if not results:
            continue
        
        aggregated[mode] = {
            'avg_evaluation_time': np.mean([r.avg_evaluation_time for r in results]),
            'avg_nodes_per_second': np.mean([r.avg_nodes_per_second for r in results]),
            'std_evaluation_time': np.std([r.avg_evaluation_time for r in results]),
            'num_states_tested': len(results)
        }
    
    return aggregated

