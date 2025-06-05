"""
Memory pool specific benchmark implementations.

This module contains benchmark cases that specifically test the performance
of memory pools (GameStatePool and TensorPool) under various scenarios.
"""

import logging
import random
import time
import torch
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..memory.game_state_pool import GameStatePool, GameStatePoolConfig
from ..memory.tensor_pool import TensorPool, TensorPoolConfig
from ..memory.config import GrowthPolicy
from ..game import GameState
from .benchmark_framework import BenchmarkCase

logger = logging.getLogger(__name__)


class GameStatePoolBenchmark(BenchmarkCase):
    """Benchmark for GameStatePool allocation and deallocation patterns."""
    
    def __init__(self, 
                 pool_size: int = 1000,
                 pattern: str = "sequential",
                 allocation_count: int = 1000,
                 enable_statistics: bool = True):
        """
        Initialize GameStatePool benchmark.
        
        Args:
            pool_size: Initial size of the pool
            pattern: Allocation pattern ('sequential', 'interleaved', 'random', 'burst')
            allocation_count: Number of allocations per iteration
            enable_statistics: Enable pool statistics collection
        """
        super().__init__(
            name=f"GameStatePool_{pattern}_{pool_size}_{allocation_count}",
            description=f"GameStatePool benchmark with {pattern} pattern, "
                       f"pool size {pool_size}, {allocation_count} allocations"
        )
        
        self.pool_size = pool_size
        self.pattern = pattern
        self.allocation_count = allocation_count
        self.enable_statistics = enable_statistics
        
        self.pool: Optional[GameStatePool] = None
        self.allocated_states: List[GameState] = []
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        config = GameStatePoolConfig(
            initial_size=self.pool_size,
            enable_statistics=self.enable_statistics,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=100,
            factory_func=GameState
        )
        self.pool = GameStatePool(config)
        self.allocated_states = []
        
        # Pre-warm the pool
        for _ in range(min(50, self.pool_size // 2)):
            state = self.pool.get()
            self.pool.return_game_state(state)
    
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        # Release any remaining allocated states
        while self.allocated_states:
            state = self.allocated_states.pop()
            if self.pool:
                self.pool.return_game_state(state)
        
        if self.pool:
            self.pool.cleanup()
            self.pool = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.pool:
            raise RuntimeError("Pool not initialized")
        
        start_stats = self.pool.get_statistics()
        start_time = time.perf_counter_ns()
        
        # Execute allocation pattern
        if self.pattern == "sequential":
            self._run_sequential_pattern()
        elif self.pattern == "interleaved":
            self._run_interleaved_pattern()
        elif self.pattern == "random":
            self._run_random_pattern()
        elif self.pattern == "burst":
            self._run_burst_pattern()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        end_time = time.perf_counter_ns()
        end_stats = self.pool.get_statistics()
        
        # Calculate metrics
        pattern_duration_ns = end_time - start_time
        allocations_performed = end_stats.hits + end_stats.misses - start_stats.hits - start_stats.misses
        releases_performed = end_stats.deallocations - start_stats.deallocations
        
        return {
            'pattern_duration_ns': pattern_duration_ns,
            'allocations_per_sec': allocations_performed * 1_000_000_000 / pattern_duration_ns if pattern_duration_ns > 0 else 0,
            'releases_per_sec': releases_performed * 1_000_000_000 / pattern_duration_ns if pattern_duration_ns > 0 else 0,
            'pool_hit_rate': end_stats.hits / (end_stats.hits + end_stats.misses) if (end_stats.hits + end_stats.misses) > 0 else 0,
            'pool_utilization': len(self.allocated_states) / self.pool_size if self.pool_size > 0 else 0,
            'pool_size': self.pool.size(),
            'objects_in_use': len(self.allocated_states),
            'pool_hits': end_stats.hits - start_stats.hits,
            'pool_misses': end_stats.misses - start_stats.misses,
            'object_releases': releases_performed
        }
    
    def _run_sequential_pattern(self) -> None:
        """Sequential allocation then deallocation."""
        # Allocate all states
        for _ in range(self.allocation_count):
            state = self.pool.get()
            self.allocated_states.append(state)
        
        # Release all states
        while self.allocated_states:
            state = self.allocated_states.pop()
            self.pool.return_game_state(state)
    
    def _run_interleaved_pattern(self) -> None:
        """Interleaved allocation and deallocation."""
        for i in range(self.allocation_count):
            # Allocate a state
            state = self.pool.get()
            self.allocated_states.append(state)
            
            # Every few allocations, release some states
            if i % 5 == 4 and len(self.allocated_states) > 2:
                for _ in range(min(2, len(self.allocated_states))):
                    released_state = self.allocated_states.pop(0)
                    self.pool.return_game_state(released_state)
        
        # Release remaining states
        while self.allocated_states:
            state = self.allocated_states.pop()
            self.pool.return_game_state(state)
    
    def _run_random_pattern(self) -> None:
        """Random allocation and deallocation."""
        for _ in range(self.allocation_count):
            if not self.allocated_states or (len(self.allocated_states) < 100 and random.random() < 0.7):
                # Allocate more often when we have few states
                state = self.pool.get()
                self.allocated_states.append(state)
            else:
                # Release with some probability
                if random.random() < 0.3 and self.allocated_states:
                    idx = random.randint(0, len(self.allocated_states) - 1)
                    state = self.allocated_states.pop(idx)
                    self.pool.return_game_state(state)
                else:
                    state = self.pool.get()
                    self.allocated_states.append(state)
        
        # Release remaining states
        while self.allocated_states:
            state = self.allocated_states.pop()
            self.pool.return_game_state(state)
    
    def _run_burst_pattern(self) -> None:
        """Burst allocation and deallocation."""
        burst_size = self.allocation_count // 10
        
        for burst in range(10):
            # Allocate burst
            for _ in range(burst_size):
                state = self.pool.get()
                self.allocated_states.append(state)
            
            # Release half of allocated states
            release_count = len(self.allocated_states) // 2
            for _ in range(release_count):
                state = self.allocated_states.pop(0)
                self.pool.return_game_state(state)
        
        # Release remaining states
        while self.allocated_states:
            state = self.allocated_states.pop()
            self.pool.return_game_state(state)


class TensorPoolBenchmark(BenchmarkCase):
    """Benchmark for TensorPool allocation and deallocation patterns."""
    
    def __init__(self,
                 pool_size: int = 500,
                 tensor_shapes: List[Tuple[int, ...]] = None,
                 dtype: torch.dtype = torch.float32,
                 allocation_count: int = 1000,
                 enable_reshaping: bool = True):
        """
        Initialize TensorPool benchmark.
        
        Args:
            pool_size: Initial size of the pool
            tensor_shapes: List of tensor shapes to test
            dtype: Tensor data type
            allocation_count: Number of allocations per iteration
            enable_reshaping: Enable tensor reshaping for reuse
        """
        self.tensor_shapes = tensor_shapes or [
            (32, 128),      # Small 2D
            (64, 256),      # Medium 2D
            (16, 16, 16),   # 3D cube
            (8, 32, 32, 3), # 4D (batch, height, width, channels)
            (256,),         # 1D vector
            (1024, 512),    # Large 2D
        ]
        
        super().__init__(
            name=f"TensorPool_{pool_size}_{len(self.tensor_shapes)}shapes_{allocation_count}",
            description=f"TensorPool benchmark with {len(self.tensor_shapes)} shapes, "
                       f"pool size {pool_size}, {allocation_count} allocations"
        )
        
        self.pool_size = pool_size
        self.dtype = dtype
        self.allocation_count = allocation_count
        self.enable_reshaping = enable_reshaping
        
        self.pool: Optional[TensorPool] = None
        self.allocated_tensors: List[torch.Tensor] = []
        
        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
    
    def setup(self) -> None:
        """Set up the benchmark environment."""
        config = TensorPoolConfig(
            initial_size=self.pool_size,
            enable_statistics=True,
            enable_tensor_reshaping=self.enable_reshaping,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=50
        )
        self.pool = TensorPool(config)
        self.allocated_tensors = []
        
        # Pre-warm the pool with common shapes
        warmup_tensors = []
        for shape in self.tensor_shapes[:3]:  # Just a few shapes for warmup
            tensor = self.pool.get(shape, dtype=self.dtype, device=self.device)
            warmup_tensors.append(tensor)
        
        # Release warmup tensors
        for tensor in warmup_tensors:
            self.pool.release(tensor)
    
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        # Release any remaining allocated tensors
        while self.allocated_tensors:
            tensor = self.allocated_tensors.pop()
            if self.pool:
                self.pool.release(tensor)
        
        if self.pool:
            # Force cleanup of all pools
            for device_str in self.pool.statistics.memory_by_device.keys():
                self.pool.clear_device(device_str)
            self.pool = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.pool:
            raise RuntimeError("Pool not initialized")
        
        start_stats = self.pool.get_statistics()
        start_time = time.perf_counter_ns()
        
        # Perform allocations with random shapes
        for _ in range(self.allocation_count):
            shape = random.choice(self.tensor_shapes)
            tensor = self.pool.get(shape, dtype=self.dtype, device=self.device)
            self.allocated_tensors.append(tensor)
            
            # Occasionally release tensors to create churn
            if len(self.allocated_tensors) > 50 and random.random() < 0.3:
                released_tensor = self.allocated_tensors.pop(0)
                self.pool.release(released_tensor)
        
        end_time = time.perf_counter_ns()
        
        # Release all remaining tensors
        while self.allocated_tensors:
            tensor = self.allocated_tensors.pop()
            self.pool.release(tensor)
        
        end_stats = self.pool.get_statistics()
        
        # Calculate metrics
        pattern_duration_ns = end_time - start_time
        allocations_performed = end_stats.tensor_allocations - start_stats.tensor_allocations
        deallocations_performed = end_stats.tensor_deallocations - start_stats.tensor_deallocations
        reshapes_performed = end_stats.tensor_reshapes - start_stats.tensor_reshapes
        
        memory_usage = self.pool.get_memory_usage()
        
        return {
            'pattern_duration_ns': pattern_duration_ns,
            'allocations_per_sec': allocations_performed * 1_000_000_000 / pattern_duration_ns if pattern_duration_ns > 0 else 0,
            'deallocations_per_sec': deallocations_performed * 1_000_000_000 / pattern_duration_ns if pattern_duration_ns > 0 else 0,
            'reshape_rate': reshapes_performed / allocations_performed if allocations_performed > 0 else 0,
            'memory_efficiency': memory_usage.get('efficiency_ratio', 0.0),
            'total_memory_mb': memory_usage.get('total_memory_mb', 0.0),
            'active_memory_mb': memory_usage.get('active_memory_mb', 0.0),
            'pooled_memory_mb': memory_usage.get('pooled_memory_mb', 0.0),
            'tensor_allocations': allocations_performed,
            'tensor_deallocations': deallocations_performed,
            'tensor_reshapes': reshapes_performed,
            'peak_memory_mb': end_stats.peak_memory_usage_mb
        }


class MemoryFragmentationBenchmark(BenchmarkCase):
    """Benchmark to test memory fragmentation patterns."""
    
    def __init__(self, 
                 pool_size: int = 1000,
                 fragmentation_cycles: int = 10):
        """
        Initialize memory fragmentation benchmark.
        
        Args:
            pool_size: Initial pool size
            fragmentation_cycles: Number of fragmentation cycles to run
        """
        super().__init__(
            name=f"MemoryFragmentation_{pool_size}_{fragmentation_cycles}",
            description=f"Memory fragmentation test with {fragmentation_cycles} cycles"
        )
        
        self.pool_size = pool_size
        self.fragmentation_cycles = fragmentation_cycles
        
        self.game_pool: Optional[GameStatePool] = None
        self.tensor_pool: Optional[TensorPool] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create pools with statistics enabled
        game_config = GameStatePoolConfig(
            initial_size=self.pool_size,
            enable_statistics=True,
            growth_policy=GrowthPolicy.LINEAR
        )
        self.game_pool = GameStatePool(game_config)
        
        tensor_config = TensorPoolConfig(
            initial_size=self.pool_size // 2,
            enable_statistics=True,
            enable_tensor_reshaping=True,
            growth_policy=GrowthPolicy.LINEAR
        )
        self.tensor_pool = TensorPool(tensor_config)
    
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.game_pool:
            self.game_pool.cleanup()
            self.game_pool = None
        
        if self.tensor_pool:
            self.tensor_pool.clear_device('cpu')
            if torch.cuda.is_available():
                self.tensor_pool.clear_device('cuda')
            self.tensor_pool = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single fragmentation test iteration."""
        if not self.game_pool or not self.tensor_pool:
            raise RuntimeError("Pools not initialized")
        
        start_time = time.perf_counter_ns()
        
        # Track allocated objects
        game_states = []
        tensors = []
        
        fragmentation_score = 0.0
        
        for cycle in range(self.fragmentation_cycles):
            cycle_start = time.perf_counter_ns()
            
            # Phase 1: Allocate many objects
            allocation_count = 100 + random.randint(0, 50)
            
            for _ in range(allocation_count):
                # Allocate game states
                state = self.game_pool.get()
                game_states.append(state)
                
                # Allocate tensors of various sizes
                shapes = [(32, 32), (64, 64), (16, 16, 16), (128,)]
                shape = random.choice(shapes)
                tensor = self.tensor_pool.get(shape)
                tensors.append(tensor)
            
            # Phase 2: Randomly release objects to create fragmentation
            release_count = len(game_states) // 3
            for _ in range(release_count):
                if game_states:
                    # Remove from random position to create fragmentation
                    idx = random.randint(0, len(game_states) - 1)
                    state = game_states.pop(idx)
                    self.game_pool.return_game_state(state)
                
                if tensors:
                    idx = random.randint(0, len(tensors) - 1)
                    tensor = tensors.pop(idx)
                    self.tensor_pool.release(tensor)
            
            # Phase 3: Try to allocate again (test fragmentation impact)
            realloc_start = time.perf_counter_ns()
            for _ in range(50):
                state = self.game_pool.get()
                self.game_pool.return_game_state(state)
                
                tensor = self.tensor_pool.get((64, 64))
                self.tensor_pool.release(tensor)
            realloc_end = time.perf_counter_ns()
            
            cycle_end = time.perf_counter_ns()
            
            # Calculate fragmentation score for this cycle
            cycle_duration = cycle_end - cycle_start
            realloc_duration = realloc_end - realloc_start
            fragmentation_score += realloc_duration / cycle_duration
        
        # Clean up remaining objects
        for state in game_states:
            self.game_pool.return_game_state(state)
        for tensor in tensors:
            self.tensor_pool.release(tensor)
        
        end_time = time.perf_counter_ns()
        
        # Get final statistics
        game_stats = self.game_pool.get_statistics()
        tensor_stats = self.tensor_pool.get_statistics()
        tensor_memory = self.tensor_pool.get_memory_usage()
        
        total_duration_ns = end_time - start_time
        avg_fragmentation_score = fragmentation_score / self.fragmentation_cycles
        
        return {
            'total_duration_ns': total_duration_ns,
            'fragmentation_cycles': self.fragmentation_cycles,
            'avg_fragmentation_score': avg_fragmentation_score,
            'game_pool_utilization': game_stats.objects_in_use / game_stats.pool_size if game_stats.pool_size > 0 else 0,
            'game_pool_hit_rate': game_stats.pool_hits / (game_stats.pool_hits + game_stats.pool_misses) if (game_stats.pool_hits + game_stats.pool_misses) > 0 else 0,
            'tensor_memory_efficiency': tensor_memory.get('efficiency_ratio', 0.0),
            'tensor_reshape_rate': tensor_stats.tensor_reshapes / tensor_stats.tensor_allocations if tensor_stats.tensor_allocations > 0 else 0,
            'total_memory_mb': tensor_memory.get('total_memory_mb', 0.0),
            'cycles_per_sec': self.fragmentation_cycles * 1_000_000_000 / total_duration_ns if total_duration_ns > 0 else 0
        } 