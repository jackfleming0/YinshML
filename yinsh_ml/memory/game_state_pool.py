"""Specialized memory pool for GameState objects with game-specific recycling logic."""

import logging
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import threading

from .pool import MemoryPool, PoolStatistics
from .config import PoolConfig, GrowthPolicy
from ..game.game_state import GameState
from ..game.constants import Player
from ..game.types import GamePhase
from .adaptive import (
    AdaptiveMetrics,
    AdaptivePoolSizer,
    create_adaptive_metrics,
    create_adaptive_pool_sizer
)

logger = logging.getLogger(__name__)


@dataclass
class GameStatePoolConfig(PoolConfig):
    """Configuration specific to GameState pooling.
    
    Extends base PoolConfig with game-specific settings optimized for
    YINSH training workloads.
    """
    
    # Game-specific optimization settings
    mcts_batch_size: int = 100  # Expected MCTS batch allocation size
    training_mode: bool = False  # Enable training-specific optimizations
    validate_reset: bool = False  # Validate objects are properly reset (debug mode)
    
    # Adaptive features  
    enable_adaptive_sizing: bool = True
    """Enable automatic pool size adjustment based on usage patterns"""
    
    adaptive_window_size: int = 100
    """Number of operations per adaptive metrics window"""
    
    adaptive_min_pool_size: int = 50
    """Minimum pool size for adaptive sizing"""
    
    adaptive_max_pool_size: int = 5000
    """Maximum pool size for adaptive sizing"""
    
    def __post_init__(self):
        """Validate game-specific configuration."""
        super().__post_init__()
        
        if self.mcts_batch_size <= 0:
            raise ValueError("MCTS batch size must be positive")


class GameStatePoolStatistics(PoolStatistics):
    """Extended statistics for GameState pool tracking."""
    
    def __init__(self):
        super().__init__()
        
        # Game-specific metrics
        self.reset_operations = 0
        self.reset_failures = 0
        self.validation_failures = 0
        self.mcts_batch_requests = 0
        self.mcts_batch_efficiency = 0.0  # Percentage of batch fulfilled from pool
        
        # Performance tracking
        self.avg_reset_time = 0.0
        self.total_reset_time = 0.0
        
    def record_reset_time(self, duration: float):
        """Record time taken for a reset operation."""
        self.total_reset_time += duration
        self.reset_operations += 1
        self.avg_reset_time = self.total_reset_time / self.reset_operations
        
    def record_mcts_batch(self, requested: int, from_pool: int):
        """Record MCTS batch allocation statistics."""
        self.mcts_batch_requests += 1
        efficiency = (from_pool / requested * 100) if requested > 0 else 0
        
        # Running average of batch efficiency
        if self.mcts_batch_requests == 1:
            self.mcts_batch_efficiency = efficiency
        else:
            # Exponential moving average with alpha=0.1
            alpha = 0.1
            self.mcts_batch_efficiency = (alpha * efficiency + 
                                        (1 - alpha) * self.mcts_batch_efficiency)


def create_game_state() -> GameState:
    """Factory function to create a new GameState object."""
    return GameState()


def reset_game_state(game_state: GameState) -> GameState:
    """Reset a GameState object to initial state for reuse.
    
    This function efficiently resets all mutable state in a GameState
    object without creating new objects where possible, to maximize
    memory reuse efficiency.
    
    Args:
        game_state: The GameState object to reset
        
    Returns:
        The same GameState object after reset
    """
    start_time = time.time()
    
    try:
        # Reset board state by clearing pieces dict (reuse Board object)
        game_state.board.pieces.clear()
        
        # Reset core game state properties
        game_state.current_player = Player.WHITE
        game_state.phase = GamePhase.RING_PLACEMENT
        game_state.white_score = 0
        game_state.black_score = 0
        
        # Reset ring placement tracking
        game_state.rings_placed = {Player.WHITE: 0, Player.BLACK: 0}
        
        # Clear move history (reuse list object)
        game_state.move_history.clear()
        
        # Clear any temporary attributes (check __dict__ to avoid methods)
        # Only remove instance attributes that start with underscore
        attrs_to_remove = [attr for attr in game_state.__dict__.keys() 
                          if attr.startswith('_') and not attr.startswith('__')]
        for attr in attrs_to_remove:
            try:
                delattr(game_state, attr)
            except (AttributeError, TypeError):
                # Skip attributes that can't be deleted
                pass
        
        reset_time = time.time() - start_time
        logger.debug(f"GameState reset completed in {reset_time:.6f}s")
        
        return game_state
        
    except Exception as e:
        logger.error(f"Failed to reset GameState: {e}")
        raise


def validate_reset_game_state(game_state: GameState) -> bool:
    """Validate that a GameState object is properly reset.
    
    This function verifies that a reset GameState is equivalent to a
    freshly created one. Used for debugging and testing.
    
    Args:
        game_state: The GameState to validate
        
    Returns:
        True if the GameState is properly reset, False otherwise
    """
    try:
        # Check board is empty
        if game_state.board.pieces:
            logger.warning("Reset validation failed: Board not empty")
            return False
            
        # Check core state
        if game_state.current_player != Player.WHITE:
            logger.warning("Reset validation failed: Wrong current player")
            return False
            
        if game_state.phase != GamePhase.RING_PLACEMENT:
            logger.warning("Reset validation failed: Wrong phase")
            return False
            
        if game_state.white_score != 0 or game_state.black_score != 0:
            logger.warning("Reset validation failed: Non-zero scores")
            return False
            
        # Check rings placed
        expected_rings = {Player.WHITE: 0, Player.BLACK: 0}
        if game_state.rings_placed != expected_rings:
            logger.warning("Reset validation failed: Wrong rings_placed")
            return False
            
        # Check move history is empty
        if game_state.move_history:
            logger.warning("Reset validation failed: Move history not empty")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Reset validation error: {e}")
        return False


class GameStatePool(MemoryPool[GameState]):
    """Specialized memory pool for GameState objects.
    
    This pool provides optimized allocation and recycling of GameState objects
    with game-specific reset logic. It's designed for high-frequency allocation
    patterns typical in MCTS simulations and neural network training.
    
    Features:
    - Efficient GameState reset without object recreation
    - Batch allocation optimized for MCTS workloads
    - Game-specific performance metrics
    - Validation for debugging reset logic
    - Training-mode optimizations
    """
    
    def __init__(self, config: Optional[GameStatePoolConfig] = None):
        """Initialize the GameState pool.
        
        Args:
            config: Optional configuration. If None, uses default configuration
                   optimized for training workloads.
        """
        if config is None:
            # Default configuration optimized for training
            config = GameStatePoolConfig(
                initial_size=100,  # Pre-allocate for immediate use
                max_capacity=1000,  # Reasonable limit for training
                growth_policy=GrowthPolicy.LINEAR,
                growth_factor=1.5,
                factory_func=create_game_state,
                reset_func=reset_game_state,
                enable_statistics=True,
                auto_cleanup=True,
                cleanup_interval=300,  # Cleanup every 5 minutes
                cleanup_threshold=0.8,  # Cleanup when 80% of pool is unused
                object_timeout=600,  # Remove objects unused for 10 minutes
            )
            
        super().__init__(config)
        self._game_config = config
        self._game_statistics = GameStatePoolStatistics() if config.enable_statistics else None
        
        # Thread safety for game operations
        self._game_lock = threading.RLock()
        
        # Adaptive sizing
        if self.config.enable_adaptive_sizing:
            self.adaptive_metrics = create_adaptive_metrics(
                window_size=self.config.adaptive_window_size
            )
            self.adaptive_sizer = create_adaptive_pool_sizer(
                initial_size=self.config.initial_size,
                min_size=self.config.adaptive_min_pool_size,
                max_size=self.config.adaptive_max_pool_size
            )
        else:
            self.adaptive_metrics = None
            self.adaptive_sizer = None
        
        # Performance tracking
        self._concurrent_usage = 0
        self._reset_times: List[float] = []
        
        logger.info(f"GameStatePool initialized with adaptive_sizing: {self.config.enable_adaptive_sizing}, "
                   f"training_mode: {self.config.training_mode}")
        
    def get_game_state(self) -> GameState:
        """Get a GameState from the pool.
        
        This is an alias for get() with a more specific name for clarity.
        
        Returns:
            A GameState object ready for use
        """
        return self.get()
        
    def return_game_state(self, game_state: GameState):
        """Return a GameState to the pool.
        
        This is an alias for return_obj() with a more specific name.
        
        Args:
            game_state: The GameState to return to the pool
        """
        self.return_obj(game_state)
        
    def get_batch(self, count: int) -> List[GameState]:
        """Get a batch of GameState objects efficiently.
        
        This method is optimized for MCTS workloads that need multiple
        GameState objects simultaneously.
        
        Args:
            count: Number of GameState objects needed
            
        Returns:
            List of GameState objects ready for use
        """
        start_time = time.time()
        batch = []
        from_pool = 0
        
        try:
            for _ in range(count):
                # Try to get from pool first
                try:
                    game_state = self.get()
                    batch.append(game_state)
                    from_pool += 1
                except Exception:
                    # If pool is empty, create new object
                    game_state = create_game_state()
                    batch.append(game_state)
                    
            # Record batch statistics
            if self._game_statistics:
                self._game_statistics.record_mcts_batch(count, from_pool)
                
            allocation_time = time.time() - start_time
            logger.debug(f"Allocated batch of {count} GameStates in {allocation_time:.6f}s "
                        f"({from_pool} from pool, {count - from_pool} new)")
                        
            return batch
            
        except Exception as e:
            logger.error(f"Failed to allocate GameState batch: {e}")
            # Return partial batch if some allocation succeeded
            return batch
            
    def return_batch(self, game_states: List[GameState]):
        """Return a batch of GameState objects to the pool.
        
        Args:
            game_states: List of GameState objects to return
        """
        start_time = time.time()
        returned = 0
        
        for game_state in game_states:
            try:
                self.return_obj(game_state)
                returned += 1
            except Exception as e:
                logger.warning(f"Failed to return GameState to pool: {e}")
                
        return_time = time.time() - start_time
        logger.debug(f"Returned {returned}/{len(game_states)} GameStates in {return_time:.6f}s")
        
    def return_obj(self, obj: GameState):
        """Return a GameState object to the pool with reset and validation.
        
        Overrides the base implementation to add game-specific reset
        and optional validation.
        
        Args:
            obj: The GameState object to return
        """
        if not isinstance(obj, GameState):
            raise TypeError("GameStatePool can only manage GameState objects")
            
        start_time = time.time()
        
        try:
            # Reset the GameState for reuse
            if self.config.reset_func:
                reset_start = time.time()
                self.config.reset_func(obj)
                reset_time = time.time() - reset_start
                
                if self._game_statistics:
                    self._game_statistics.record_reset_time(reset_time)
                    
            # Optional validation in debug mode
            if self._game_config.validate_reset:
                if not validate_reset_game_state(obj):
                    if self._game_statistics:
                        self._game_statistics.validation_failures += 1
                    logger.warning("GameState reset validation failed")
                    # Don't return invalid object to pool
                    return
                    
            # Call parent implementation to actually return to pool
            super().return_obj(obj)
            
            total_time = time.time() - start_time
            logger.debug(f"GameState returned to pool in {total_time:.6f}s")
            
        except Exception as e:
            if self._game_statistics:
                self._game_statistics.reset_failures += 1
            logger.error(f"Failed to return GameState to pool: {e}")
            raise
            
    def get_game_statistics(self) -> Optional[GameStatePoolStatistics]:
        """Get game-specific pool statistics.
        
        Returns:
            GameStatePoolStatistics object or None if statistics disabled
        """
        return self._game_statistics
        
    def print_statistics(self):
        """Print formatted statistics for the GameState pool."""
        base_stats = self.get_statistics()
        game_stats = self.get_game_statistics()
        
        if not base_stats or not game_stats:
            print("Statistics not available (disabled in configuration)")
            return
            
        print("\n=== GameState Pool Statistics ===")
        print(f"Pool Size: {self.size()}")
        print(f"Capacity: {self.capacity()}")
        print(f"Hit Rate: {base_stats.hit_rate:.1f}%")
        print(f"Miss Rate: {base_stats.miss_rate:.1f}%")
        print(f"Total Allocations: {base_stats.allocations}")
        print(f"Total Returns: {base_stats.deallocations}")
        print(f"Peak Size: {base_stats.peak_size}")
        
        print("\n=== Game-Specific Metrics ===")
        print(f"Reset Operations: {game_stats.reset_operations}")
        print(f"Reset Failures: {game_stats.reset_failures}")
        print(f"Average Reset Time: {game_stats.avg_reset_time:.6f}s")
        print(f"MCTS Batch Requests: {game_stats.mcts_batch_requests}")
        print(f"MCTS Batch Efficiency: {game_stats.mcts_batch_efficiency:.1f}%")
        
        if game_stats.validation_failures > 0:
            print(f"Validation Failures: {game_stats.validation_failures}")
            
        if base_stats.cleanup_events > 0:
            print(f"\n=== Cleanup Metrics ===")
            print(f"Cleanup Events: {base_stats.cleanup_events}")
            print(f"Objects Cleaned: {base_stats.objects_cleaned}")
            print(f"Cleanup Efficiency: {base_stats.cleanup_efficiency:.1f}%")


# Convenience factory function
def create_game_state_pool(
    initial_size: int = 100,
    max_capacity: int = 1000,
    training_mode: bool = False,
    validate_reset: bool = False
) -> GameStatePool:
    """Create a GameStatePool with common configuration.
    
    Args:
        initial_size: Number of GameStates to pre-allocate
        max_capacity: Maximum pool size
        training_mode: Enable training-specific optimizations
        validate_reset: Enable reset validation (debug mode)
        
    Returns:
        Configured GameStatePool ready for use
    """
    config = GameStatePoolConfig(
        initial_size=initial_size,
        max_capacity=max_capacity,
        growth_policy=GrowthPolicy.LINEAR,
        growth_factor=1.5,
        factory_func=create_game_state,
        reset_func=reset_game_state,
        enable_statistics=True,
        auto_cleanup=True,
        cleanup_interval=300,
        cleanup_threshold=0.8,
        object_timeout=600,
        training_mode=training_mode,
        validate_reset=validate_reset,
    )
    
    return GameStatePool(config) 