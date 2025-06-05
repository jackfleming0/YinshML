"""
Pre-configured benchmark scenarios for common testing patterns.

This module provides standardized benchmark suites for different types of
performance testing and validation scenarios.
"""

import logging
from typing import List, Dict, Any

from .benchmark_framework import BenchmarkCase, BenchmarkSuite
from .memory_benchmarks import (
    GameStatePoolBenchmark, 
    TensorPoolBenchmark, 
    MemoryFragmentationBenchmark
)
from .training_benchmarks import (
    SelfPlayBenchmark,
    NetworkWrapperBenchmark,
    TrainingSupervisorBenchmark
)

logger = logging.getLogger(__name__)


class StandardScenarios:
    """Standard benchmark scenarios for memory management validation."""
    
    @staticmethod
    def create_memory_pool_suite(iterations: int = 50) -> BenchmarkSuite:
        """
        Create a standard memory pool benchmark suite.
        
        Args:
            iterations: Number of iterations per benchmark
            
        Returns:
            Configured benchmark suite
        """
        cases = [
            # GameStatePool benchmarks with different patterns
            GameStatePoolBenchmark(
                pool_size=500,
                pattern="sequential",
                allocation_count=1000,
                enable_statistics=True
            ),
            GameStatePoolBenchmark(
                pool_size=500,
                pattern="interleaved",
                allocation_count=1000,
                enable_statistics=True
            ),
            GameStatePoolBenchmark(
                pool_size=500,
                pattern="random",
                allocation_count=1000,
                enable_statistics=True
            ),
            GameStatePoolBenchmark(
                pool_size=500,
                pattern="burst",
                allocation_count=1000,
                enable_statistics=True
            ),
            
            # TensorPool benchmarks
            TensorPoolBenchmark(
                pool_size=300,
                allocation_count=500,
                enable_reshaping=True
            ),
            TensorPoolBenchmark(
                pool_size=300,
                allocation_count=500,
                enable_reshaping=False
            ),
            
            # Memory fragmentation test
            MemoryFragmentationBenchmark(
                pool_size=1000,
                fragmentation_cycles=10
            )
        ]
        
        return BenchmarkSuite(
            cases=cases,
            iterations=iterations,
            warmup_iterations=10,
            cooldown_seconds=1.0
        )
    
    @staticmethod
    def create_training_pipeline_suite(iterations: int = 10) -> BenchmarkSuite:
        """
        Create a training pipeline benchmark suite.
        
        Args:
            iterations: Number of iterations per benchmark
            
        Returns:
            Configured benchmark suite
        """
        cases = [
            # SelfPlay benchmarks with and without memory pools
            SelfPlayBenchmark(
                num_games=5,
                mcts_simulations=50,
                use_memory_pools=True,
                pool_size=300
            ),
            SelfPlayBenchmark(
                num_games=5,
                mcts_simulations=50,
                use_memory_pools=False,
                pool_size=300
            ),
            
            # NetworkWrapper benchmarks
            NetworkWrapperBenchmark(
                num_predictions=500,
                batch_sizes=[1, 4, 8, 16],
                use_tensor_pools=True,
                pool_size=200
            ),
            NetworkWrapperBenchmark(
                num_predictions=500,
                batch_sizes=[1, 4, 8, 16],
                use_tensor_pools=False,
                pool_size=200
            ),
            
            # TrainingSupervisor benchmark (simplified)
            TrainingSupervisorBenchmark(
                num_iterations=2,
                games_per_iteration=3,
                mcts_simulations=25,
                use_memory_management=True
            )
        ]
        
        return BenchmarkSuite(
            cases=cases,
            iterations=iterations,
            warmup_iterations=2,
            cooldown_seconds=2.0
        )
    
    @staticmethod
    def create_quick_validation_suite() -> BenchmarkSuite:
        """
        Create a quick validation suite for CI/CD or rapid testing.
        
        Returns:
            Configured benchmark suite with minimal iterations
        """
        cases = [
            # Quick memory pool validation
            GameStatePoolBenchmark(
                pool_size=100,
                pattern="sequential",
                allocation_count=200,
                enable_statistics=True
            ),
            TensorPoolBenchmark(
                pool_size=50,
                allocation_count=100,
                enable_reshaping=True
            ),
            
            # Quick training component validation
            SelfPlayBenchmark(
                num_games=2,
                mcts_simulations=20,
                use_memory_pools=True,
                pool_size=100
            ),
            NetworkWrapperBenchmark(
                num_predictions=100,
                batch_sizes=[1, 4],
                use_tensor_pools=True,
                pool_size=50
            )
        ]
        
        return BenchmarkSuite(
            cases=cases,
            iterations=5,
            warmup_iterations=1,
            cooldown_seconds=0.5
        )
    
    @staticmethod
    def create_comprehensive_suite(iterations: int = 100) -> BenchmarkSuite:
        """
        Create a comprehensive benchmark suite for thorough performance analysis.
        
        Args:
            iterations: Number of iterations per benchmark
            
        Returns:
            Configured benchmark suite
        """
        memory_suite = StandardScenarios.create_memory_pool_suite(iterations)
        training_suite = StandardScenarios.create_training_pipeline_suite(iterations // 5)
        
        # Combine all cases
        all_cases = memory_suite.cases + training_suite.cases
        
        return BenchmarkSuite(
            cases=all_cases,
            iterations=iterations,
            warmup_iterations=15,
            cooldown_seconds=2.0
        )


class ScalingScenarios:
    """Benchmark scenarios focused on scaling behavior analysis."""
    
    @staticmethod
    def create_pool_size_scaling_suite() -> BenchmarkSuite:
        """
        Create benchmarks to test pool size scaling behavior.
        
        Returns:
            Configured benchmark suite
        """
        cases = []
        
        # Test different pool sizes for GameStatePool
        pool_sizes = [100, 500, 1000, 2000]
        for size in pool_sizes:
            cases.append(
                GameStatePoolBenchmark(
                    pool_size=size,
                    pattern="random",
                    allocation_count=1000,
                    enable_statistics=True
                )
            )
        
        # Test different pool sizes for TensorPool
        tensor_pool_sizes = [50, 200, 500, 1000]
        for size in tensor_pool_sizes:
            cases.append(
                TensorPoolBenchmark(
                    pool_size=size,
                    allocation_count=500,
                    enable_reshaping=True
                )
            )
        
        return BenchmarkSuite(
            cases=cases,
            iterations=30,
            warmup_iterations=5,
            cooldown_seconds=1.0
        )
    
    @staticmethod
    def create_allocation_scaling_suite() -> BenchmarkSuite:
        """
        Create benchmarks to test allocation count scaling behavior.
        
        Returns:
            Configured benchmark suite
        """
        cases = []
        
        # Test different allocation counts
        allocation_counts = [500, 1000, 2000, 5000]
        for count in allocation_counts:
            cases.append(
                GameStatePoolBenchmark(
                    pool_size=1000,
                    pattern="sequential",
                    allocation_count=count,
                    enable_statistics=True
                )
            )
            cases.append(
                TensorPoolBenchmark(
                    pool_size=500,
                    allocation_count=count,
                    enable_reshaping=True
                )
            )
        
        return BenchmarkSuite(
            cases=cases,
            iterations=20,
            warmup_iterations=3,
            cooldown_seconds=1.5
        )
    
    @staticmethod
    def create_mcts_scaling_suite() -> BenchmarkSuite:
        """
        Create benchmarks to test MCTS simulation scaling.
        
        Returns:
            Configured benchmark suite
        """
        cases = []
        
        # Test different MCTS simulation counts
        simulation_counts = [25, 50, 100, 200, 500]
        for sims in simulation_counts:
            cases.extend([
                SelfPlayBenchmark(
                    num_games=3,
                    mcts_simulations=sims,
                    use_memory_pools=True,
                    pool_size=max(200, sims * 2)
                ),
                SelfPlayBenchmark(
                    num_games=3,
                    mcts_simulations=sims,
                    use_memory_pools=False,
                    pool_size=max(200, sims * 2)
                )
            ])
        
        return BenchmarkSuite(
            cases=cases,
            iterations=10,
            warmup_iterations=2,
            cooldown_seconds=2.0
        )


class StressTestScenarios:
    """Benchmark scenarios designed to stress test the memory management system."""
    
    @staticmethod
    def create_memory_pressure_suite() -> BenchmarkSuite:
        """
        Create benchmarks that apply memory pressure to test system limits.
        
        Returns:
            Configured benchmark suite
        """
        cases = [
            # High allocation count with limited pool size
            GameStatePoolBenchmark(
                pool_size=500,
                pattern="random",
                allocation_count=5000,
                enable_statistics=True
            ),
            
            # Large tensor allocations
            TensorPoolBenchmark(
                pool_size=100,
                tensor_shapes=[
                    (512, 512),        # Large 2D
                    (128, 128, 32),    # Large 3D
                    (64, 64, 64, 8),   # Large 4D
                    (2048,),           # Large 1D
                ],
                allocation_count=200,
                enable_reshaping=True
            ),
            
            # Intensive fragmentation test
            MemoryFragmentationBenchmark(
                pool_size=2000,
                fragmentation_cycles=50
            ),
            
            # High-intensity SelfPlay
            SelfPlayBenchmark(
                num_games=10,
                mcts_simulations=200,
                use_memory_pools=True,
                pool_size=1000
            )
        ]
        
        return BenchmarkSuite(
            cases=cases,
            iterations=15,
            warmup_iterations=3,
            cooldown_seconds=3.0
        )
    
    @staticmethod
    def create_endurance_suite() -> BenchmarkSuite:
        """
        Create benchmarks for endurance testing over extended periods.
        
        Returns:
            Configured benchmark suite
        """
        cases = [
            # Long-running allocation patterns
            GameStatePoolBenchmark(
                pool_size=1000,
                pattern="interleaved",
                allocation_count=10000,
                enable_statistics=True
            ),
            
            # Extended tensor operations
            TensorPoolBenchmark(
                pool_size=300,
                allocation_count=2000,
                enable_reshaping=True
            ),
            
            # Extended SelfPlay session
            SelfPlayBenchmark(
                num_games=20,
                mcts_simulations=100,
                use_memory_pools=True,
                pool_size=800
            )
        ]
        
        return BenchmarkSuite(
            cases=cases,
            iterations=50,  # Higher iteration count for endurance
            warmup_iterations=5,
            cooldown_seconds=2.0
        )
    
    @staticmethod
    def create_concurrency_stress_suite() -> BenchmarkSuite:
        """
        Create benchmarks to test concurrent access patterns.
        
        Returns:
            Configured benchmark suite
        """
        cases = [
            # Burst patterns that simulate concurrent access
            GameStatePoolBenchmark(
                pool_size=800,
                pattern="burst",
                allocation_count=2000,
                enable_statistics=True
            ),
            
            # Random patterns with high churn
            GameStatePoolBenchmark(
                pool_size=600,
                pattern="random",
                allocation_count=3000,
                enable_statistics=True
            ),
            
            # Multiple batch sizes to simulate concurrent predictions
            NetworkWrapperBenchmark(
                num_predictions=1000,
                batch_sizes=[1, 2, 4, 8, 16, 32],
                use_tensor_pools=True,
                pool_size=400
            )
        ]
        
        return BenchmarkSuite(
            cases=cases,
            iterations=25,
            warmup_iterations=5,
            cooldown_seconds=1.5
        )


class ComparisonScenarios:
    """Benchmark scenarios designed for comparing different configurations."""
    
    @staticmethod
    def create_pooled_vs_unpooled_suite() -> BenchmarkSuite:
        """
        Create benchmarks comparing pooled vs unpooled memory management.
        
        Returns:
            Configured benchmark suite
        """
        cases = []
        
        # SelfPlay comparison
        base_config = {
            'num_games': 5,
            'mcts_simulations': 100,
            'pool_size': 500
        }
        
        cases.extend([
            SelfPlayBenchmark(use_memory_pools=True, **base_config),
            SelfPlayBenchmark(use_memory_pools=False, **base_config)
        ])
        
        # NetworkWrapper comparison
        network_config = {
            'num_predictions': 500,
            'batch_sizes': [1, 4, 8, 16],
            'pool_size': 200
        }
        
        cases.extend([
            NetworkWrapperBenchmark(use_tensor_pools=True, **network_config),
            NetworkWrapperBenchmark(use_tensor_pools=False, **network_config)
        ])
        
        # TrainingSupervisor comparison
        supervisor_config = {
            'num_iterations': 3,
            'games_per_iteration': 4,
            'mcts_simulations': 50
        }
        
        cases.extend([
            TrainingSupervisorBenchmark(use_memory_management=True, **supervisor_config),
            TrainingSupervisorBenchmark(use_memory_management=False, **supervisor_config)
        ])
        
        return BenchmarkSuite(
            cases=cases,
            iterations=20,
            warmup_iterations=5,
            cooldown_seconds=2.0
        )
    
    @staticmethod
    def create_configuration_comparison_suite() -> BenchmarkSuite:
        """
        Create benchmarks comparing different pool configurations.
        
        Returns:
            Configured benchmark suite
        """
        cases = []
        
        # Different tensor pool configurations
        tensor_configs = [
            {'enable_reshaping': True, 'pool_size': 200},
            {'enable_reshaping': False, 'pool_size': 200},
            {'enable_reshaping': True, 'pool_size': 400},
            {'enable_reshaping': False, 'pool_size': 400}
        ]
        
        for config in tensor_configs:
            cases.append(
                TensorPoolBenchmark(
                    allocation_count=500,
                    **config
                )
            )
        
        # Different game state pool patterns
        patterns = ['sequential', 'interleaved', 'random', 'burst']
        for pattern in patterns:
            cases.append(
                GameStatePoolBenchmark(
                    pool_size=500,
                    pattern=pattern,
                    allocation_count=1000,
                    enable_statistics=True
                )
            )
        
        return BenchmarkSuite(
            cases=cases,
            iterations=25,
            warmup_iterations=5,
            cooldown_seconds=1.0
        ) 