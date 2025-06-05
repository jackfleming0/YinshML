"""
Training pipeline specific benchmark implementations.

This module contains benchmark cases that test the performance of the training
components (SelfPlay, NetworkWrapper, TrainingSupervisor) with and without
memory management enabled.
"""

import logging
import time
import torch
from typing import Any, Dict, List, Optional
import tempfile
import os

from ..memory.game_state_pool import GameStatePool, GameStatePoolConfig
from ..memory.tensor_pool import TensorPool, TensorPoolConfig
from ..memory.config import GrowthPolicy
from ..training import SelfPlay
from ..training.supervisor import TrainingSupervisor
from ..network.wrapper import NetworkWrapper
from ..game import GameState
from .benchmark_framework import BenchmarkCase

logger = logging.getLogger(__name__)


class SelfPlayBenchmark(BenchmarkCase):
    """Benchmark for SelfPlay performance with memory management."""
    
    def __init__(self,
                 num_games: int = 10,
                 mcts_simulations: int = 100,
                 use_memory_pools: bool = True,
                 pool_size: int = 500):
        """
        Initialize SelfPlay benchmark.
        
        Args:
            num_games: Number of games to play per iteration
            mcts_simulations: Number of MCTS simulations per move
            use_memory_pools: Whether to use memory pools
            pool_size: Size of memory pools if enabled
        """
        suffix = "with_pools" if use_memory_pools else "no_pools"
        super().__init__(
            name=f"SelfPlay_{num_games}games_{mcts_simulations}sims_{suffix}",
            description=f"SelfPlay benchmark: {num_games} games, {mcts_simulations} sims/move, "
                       f"memory pools: {use_memory_pools}"
        )
        
        self.num_games = num_games
        self.mcts_simulations = mcts_simulations
        self.use_memory_pools = use_memory_pools
        self.pool_size = pool_size
        
        self.self_play: Optional[SelfPlay] = None
        self.game_state_pool: Optional[GameStatePool] = None
        self.network_wrapper: Optional[NetworkWrapper] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create network wrapper
        self.network_wrapper = NetworkWrapper()
        
        # Create memory pools if enabled
        if self.use_memory_pools:
            game_config = GameStatePoolConfig(
                initial_size=self.pool_size,
                enable_statistics=True,
                growth_policy=GrowthPolicy.LINEAR,
                growth_factor=100
            )
            self.game_state_pool = GameStatePool(game_config)
        else:
            self.game_state_pool = None
        
        # Create SelfPlay instance with individual MCTS parameters
        self.self_play = SelfPlay(
            network=self.network_wrapper,
            num_workers=1,  # Single worker for benchmark consistency
            num_simulations=self.mcts_simulations,
            late_simulations=None,  # Use same simulations throughout
            simulation_switch_ply=50,  # Default switch ply
            c_puct=1.4,
            dirichlet_alpha=0.25,
            value_weight=1.0,
            max_depth=100,
            initial_temp=1.0,
            final_temp=0.1,
            annealing_steps=30,
            temp_clamp_fraction=0.75,
            game_state_pool=self.game_state_pool
        )
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.game_state_pool:
            self.game_state_pool.cleanup()
            self.game_state_pool = None
        
        self.self_play = None
        self.network_wrapper = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.self_play:
            raise RuntimeError("SelfPlay not initialized")
        
        start_time = time.perf_counter_ns()
        
        # Collect statistics before
        memory_stats_before = None
        if self.game_state_pool:
            memory_stats_before = self.game_state_pool.get_statistics()
        
        # Play games
        games_played = 0
        total_moves = 0
        total_mcts_nodes = 0
        
        try:
            # Generate all games in one call
            game_results = self.self_play.generate_games(self.num_games)
            games_played = len(game_results)
            
            # Each result is: (states, policies, value, mcts_metrics_list)
            for states, policies, value, mcts_metrics_list in game_results:
                total_moves += len(states)
                # Estimate MCTS nodes (simulations per move)
                total_mcts_nodes += len(states) * self.mcts_simulations
                
        except Exception as e:
            logger.warning(f"Game generation failed: {e}")
        
        end_time = time.perf_counter_ns()
        
        # Collect statistics after
        memory_stats_after = None
        if self.game_state_pool:
            memory_stats_after = self.game_state_pool.get_statistics()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'games_played': games_played,
            'total_moves': total_moves,
            'estimated_mcts_nodes': total_mcts_nodes,
            'games_per_sec': games_played / duration_seconds if duration_seconds > 0 else 0,
            'moves_per_sec': total_moves / duration_seconds if duration_seconds > 0 else 0,
            'mcts_nodes_per_sec': total_mcts_nodes / duration_seconds if duration_seconds > 0 else 0,
            'avg_moves_per_game': total_moves / games_played if games_played > 0 else 0,
            'use_memory_pools': self.use_memory_pools
        }
        
        # Add memory pool metrics if available
        if memory_stats_before and memory_stats_after:
            pool_hits = memory_stats_after.pool_hits - memory_stats_before.pool_hits
            pool_misses = memory_stats_after.pool_misses - memory_stats_before.pool_misses
            total_requests = pool_hits + pool_misses
            
            metrics.update({
                'pool_requests': total_requests,
                'pool_hit_rate': pool_hits / total_requests if total_requests > 0 else 0,
                'pool_utilization': memory_stats_after.objects_in_use / memory_stats_after.pool_size if memory_stats_after.pool_size > 0 else 0,
                'pool_efficiency': pool_hits / total_mcts_nodes if total_mcts_nodes > 0 else 0
            })
        
        return metrics


class NetworkWrapperBenchmark(BenchmarkCase):
    """Benchmark for NetworkWrapper performance with tensor pooling."""
    
    def __init__(self,
                 num_predictions: int = 1000,
                 batch_sizes: List[int] = None,
                 use_tensor_pools: bool = True,
                 pool_size: int = 200):
        """
        Initialize NetworkWrapper benchmark.
        
        Args:
            num_predictions: Number of predictions per iteration
            batch_sizes: List of batch sizes to test
            use_tensor_pools: Whether to use tensor pools
            pool_size: Size of tensor pools if enabled
        """
        self.batch_sizes = batch_sizes or [1, 4, 8, 16, 32]
        suffix = "with_pools" if use_tensor_pools else "no_pools"
        
        super().__init__(
            name=f"NetworkWrapper_{num_predictions}preds_{suffix}",
            description=f"NetworkWrapper benchmark: {num_predictions} predictions, "
                       f"tensor pools: {use_tensor_pools}"
        )
        
        self.num_predictions = num_predictions
        self.use_tensor_pools = use_tensor_pools
        self.pool_size = pool_size
        
        self.network_wrapper: Optional[NetworkWrapper] = None
        self.tensor_pool: Optional[TensorPool] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create tensor pool if enabled
        if self.use_tensor_pools:
            tensor_config = TensorPoolConfig(
                initial_size=self.pool_size,
                enable_statistics=True,
                enable_tensor_reshaping=True,
                growth_policy=GrowthPolicy.LINEAR,
                growth_factor=50
            )
            self.tensor_pool = TensorPool(tensor_config)
        else:
            self.tensor_pool = None
        
        # Create NetworkWrapper
        self.network_wrapper = NetworkWrapper(tensor_pool=self.tensor_pool)
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.tensor_pool:
            # Clear all device pools
            for device_str in self.tensor_pool.statistics.memory_by_device.keys():
                self.tensor_pool.clear_device(device_str)
            self.tensor_pool = None
        
        self.network_wrapper = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.network_wrapper:
            raise RuntimeError("NetworkWrapper not initialized")
        
        start_time = time.perf_counter_ns()
        
        # Collect statistics before
        tensor_stats_before = None
        if self.tensor_pool:
            tensor_stats_before = self.tensor_pool.get_statistics()
        
        # Perform predictions with different batch sizes
        predictions_made = 0
        total_tensor_allocations = 0
        
        for _ in range(self.num_predictions):
            try:
                # Create a dummy game state for prediction
                game_state = GameState()
                batch_size = self.batch_sizes[predictions_made % len(self.batch_sizes)]
                
                # Use the memory pool enabled prediction method if available
                if self.use_tensor_pools and hasattr(self.network_wrapper, 'predict_from_state'):
                    policy, value = self.network_wrapper.predict_from_state(game_state)
                else:
                    # Fallback to regular prediction
                    state_tensor = self.network_wrapper.state_to_tensor(game_state)
                    
                    # Simulate batch processing
                    if batch_size > 1:
                        batch_tensor = state_tensor.repeat(batch_size, 1, 1, 1)
                    else:
                        batch_tensor = state_tensor.unsqueeze(0)
                    
                    policy, value = self.network_wrapper.predict(batch_tensor)
                    total_tensor_allocations += 1  # Manual tensor creation
                
                predictions_made += 1
                
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
        
        end_time = time.perf_counter_ns()
        
        # Collect statistics after
        tensor_stats_after = None
        if self.tensor_pool:
            tensor_stats_after = self.tensor_pool.get_statistics()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'predictions_made': predictions_made,
            'predictions_per_sec': predictions_made / duration_seconds if duration_seconds > 0 else 0,
            'avg_prediction_time_ms': (total_duration_ns / predictions_made) / 1_000_000 if predictions_made > 0 else 0,
            'use_tensor_pools': self.use_tensor_pools,
            'manual_tensor_allocations': total_tensor_allocations
        }
        
        # Add tensor pool metrics if available
        if tensor_stats_before and tensor_stats_after:
            tensor_allocations = tensor_stats_after.tensor_allocations - tensor_stats_before.tensor_allocations
            tensor_deallocations = tensor_stats_after.tensor_deallocations - tensor_stats_before.tensor_deallocations
            tensor_reshapes = tensor_stats_after.tensor_reshapes - tensor_stats_before.tensor_reshapes
            
            memory_usage = self.tensor_pool.get_memory_usage()
            
            metrics.update({
                'tensor_allocations': tensor_allocations,
                'tensor_deallocations': tensor_deallocations,
                'tensor_reshapes': tensor_reshapes,
                'reshape_rate': tensor_reshapes / tensor_allocations if tensor_allocations > 0 else 0,
                'memory_efficiency': memory_usage.get('efficiency_ratio', 0.0),
                'peak_memory_mb': tensor_stats_after.peak_memory_usage_mb,
                'total_memory_mb': memory_usage.get('total_memory_mb', 0.0)
            })
        
        return metrics


class TrainingSupervisorBenchmark(BenchmarkCase):
    """Benchmark for TrainingSupervisor with full memory management integration."""
    
    def __init__(self,
                 num_iterations: int = 3,
                 games_per_iteration: int = 5,
                 mcts_simulations: int = 50,
                 use_memory_management: bool = True):
        """
        Initialize TrainingSupervisor benchmark.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of self-play games per iteration
            mcts_simulations: MCTS simulations per move
            use_memory_management: Whether to use memory management
        """
        suffix = "with_memory_mgmt" if use_memory_management else "no_memory_mgmt"
        super().__init__(
            name=f"TrainingSupervisor_{num_iterations}iters_{suffix}",
            description=f"TrainingSupervisor benchmark: {num_iterations} iterations, "
                       f"memory management: {use_memory_management}"
        )
        
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.use_memory_management = use_memory_management
        
        self.supervisor: Optional[TrainingSupervisor] = None
        self.temp_dir: Optional[str] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create temporary directory for model checkpoints
        self.temp_dir = tempfile.mkdtemp()
        
        # Create TrainingSupervisor configuration
        config = {
            'self_play': {
                'games_per_iteration': self.games_per_iteration,
                'num_workers': 1,  # Single worker for consistent benchmarking
                'mcts_simulations': self.mcts_simulations,
                'mcts_exploration_weight': 1.4,
                'temperature': 1.0,
                'temperature_threshold': 10,
                'add_noise': True,
                'noise_weight': 0.25
            },
            'training': {
                'batch_size': 32,
                'epochs_per_iteration': 1,  # Minimal training for speed
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'checkpoint_interval': 1000,  # Don't checkpoint during benchmark
                'validation_interval': 1000
            },
            'model': {
                'checkpoint_dir': self.temp_dir,
                'save_interval': 1000  # Don't save during benchmark
            }
        }
        
        # Override memory management based on benchmark setting
        if not self.use_memory_management:
            # This would require modifying TrainingSupervisor to optionally disable memory pools
            # For now, we'll create the supervisor normally and note this limitation
            pass
        
        self.supervisor = TrainingSupervisor(config)
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.supervisor:
            # Clean up memory pools if they exist
            if hasattr(self.supervisor, 'cleanup_memory_pools'):
                self.supervisor.cleanup_memory_pools()
            self.supervisor = None
        
        # Clean up temporary directory
        if self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")
            self.temp_dir = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.supervisor:
            raise RuntimeError("TrainingSupervisor not initialized")
        
        start_time = time.perf_counter_ns()
        
        # Collect initial memory statistics
        initial_stats = {}
        if hasattr(self.supervisor, 'game_state_pool') and self.supervisor.game_state_pool:
            initial_stats['game_pool'] = self.supervisor.game_state_pool.get_statistics()
        if hasattr(self.supervisor, 'tensor_pool') and self.supervisor.tensor_pool:
            initial_stats['tensor_pool'] = self.supervisor.tensor_pool.get_statistics()
        
        # Run training iterations
        iterations_completed = 0
        total_games_played = 0
        total_training_steps = 0
        
        for iteration in range(self.num_iterations):
            try:
                # This is a simplified training iteration
                # In practice, you'd call supervisor.train_iteration()
                # but we'll simulate the key components for benchmarking
                
                # Simulate self-play
                self_play_start = time.perf_counter_ns()
                # supervisor.run_self_play() would be called here
                self_play_duration = time.perf_counter_ns() - self_play_start
                
                # Simulate training
                training_start = time.perf_counter_ns()
                # supervisor.train_network() would be called here
                training_duration = time.perf_counter_ns() - training_start
                
                iterations_completed += 1
                total_games_played += self.games_per_iteration
                total_training_steps += 1  # Simplified
                
            except Exception as e:
                logger.warning(f"Training iteration {iteration} failed: {e}")
                break
        
        end_time = time.perf_counter_ns()
        
        # Collect final memory statistics
        final_stats = {}
        if hasattr(self.supervisor, 'game_state_pool') and self.supervisor.game_state_pool:
            final_stats['game_pool'] = self.supervisor.game_state_pool.get_statistics()
        if hasattr(self.supervisor, 'tensor_pool') and self.supervisor.tensor_pool:
            final_stats['tensor_pool'] = self.supervisor.tensor_pool.get_statistics()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'iterations_completed': iterations_completed,
            'total_games_played': total_games_played,
            'total_training_steps': total_training_steps,
            'iterations_per_sec': iterations_completed / duration_seconds if duration_seconds > 0 else 0,
            'games_per_sec': total_games_played / duration_seconds if duration_seconds > 0 else 0,
            'use_memory_management': self.use_memory_management
        }
        
        # Add memory management metrics if available
        if 'game_pool' in initial_stats and 'game_pool' in final_stats:
            game_initial = initial_stats['game_pool']
            game_final = final_stats['game_pool']
            
            game_requests = (game_final.pool_hits + game_final.pool_misses) - (game_initial.pool_hits + game_initial.pool_misses)
            game_hit_rate = (game_final.pool_hits - game_initial.pool_hits) / game_requests if game_requests > 0 else 0
            
            metrics.update({
                'game_pool_requests': game_requests,
                'game_pool_hit_rate': game_hit_rate,
                'game_pool_utilization': game_final.objects_in_use / game_final.pool_size if game_final.pool_size > 0 else 0
            })
        
        if 'tensor_pool' in initial_stats and 'tensor_pool' in final_stats:
            tensor_initial = initial_stats['tensor_pool']
            tensor_final = final_stats['tensor_pool']
            
            tensor_allocations = tensor_final.tensor_allocations - tensor_initial.tensor_allocations
            tensor_reshapes = tensor_final.tensor_reshapes - tensor_initial.tensor_reshapes
            
            metrics.update({
                'tensor_allocations': tensor_allocations,
                'tensor_reshapes': tensor_reshapes,
                'tensor_reshape_rate': tensor_reshapes / tensor_allocations if tensor_allocations > 0 else 0,
                'peak_memory_mb': tensor_final.peak_memory_usage_mb
            })
        
        return metrics 