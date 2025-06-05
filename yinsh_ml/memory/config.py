"""Configuration classes for memory pool management."""

from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum
import warnings


class GrowthPolicy(Enum):
    """Growth policies for memory pools when capacity is exceeded."""
    FIXED = "fixed"  # No growth, fail when full
    LINEAR = "linear"  # Grow by fixed amount
    EXPONENTIAL = "exponential"  # Double the size


@dataclass
class PoolConfig:
    """Configuration for memory pool behavior.
    
    Args:
        initial_size: Initial number of objects to pre-allocate
        max_capacity: Maximum number of objects the pool can hold (None = unlimited)
        growth_policy: How the pool should grow when capacity is exceeded
        growth_factor: Factor for growth (amount for LINEAR, multiplier for EXPONENTIAL)
        enable_statistics: Whether to collect performance statistics
        auto_cleanup: Whether to automatically clean up unused objects
        cleanup_threshold: Fraction of pool that must be unused to trigger cleanup
        factory_func: Function to create new objects when pool is empty
        reset_func: Function to reset objects when returned to pool
        object_timeout: Maximum idle time (seconds) before objects are removed (0 = no timeout)
        cleanup_interval: Interval (seconds) for background cleanup (0 = no background cleanup)
    """
    initial_size: int = 10
    max_capacity: Optional[int] = None
    growth_policy: GrowthPolicy = GrowthPolicy.LINEAR
    growth_factor: int = 10
    enable_statistics: bool = True
    auto_cleanup: bool = False
    cleanup_threshold: float = 0.8
    factory_func: Optional[Callable[[], Any]] = None
    reset_func: Optional[Callable[[Any], None]] = None
    object_timeout: float = 0.0  # 0 = no timeout
    cleanup_interval: float = 0.0  # 0 = no background cleanup
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_size < 0:
            raise ValueError("initial_size must be non-negative")
        
        if self.max_capacity is not None and self.max_capacity < self.initial_size:
            raise ValueError("max_capacity must be >= initial_size")
            
        if self.growth_factor <= 0:
            raise ValueError("growth_factor must be positive")
            
        if not 0.0 <= self.cleanup_threshold <= 1.0:
            raise ValueError("cleanup_threshold must be between 0.0 and 1.0")
            
        if self.object_timeout < 0:
            raise ValueError("object_timeout must be non-negative")
            
        if self.cleanup_interval < 0:
            raise ValueError("cleanup_interval must be non-negative")
            
        # Enhanced validation for growth policies
        if self.growth_policy == GrowthPolicy.FIXED:
            # Fixed policy doesn't use growth_factor, but should have reasonable max_capacity
            if self.max_capacity is None and self.initial_size > 0:
                warnings.warn(
                    "Fixed growth policy with unlimited capacity may not be optimal. "
                    "Consider setting max_capacity for better resource management."
                )
        else:
            # Non-fixed policies need factory function for growth
            if self.factory_func is None:
                warnings.warn(
                    f"{self.growth_policy.value} growth policy requires factory_func "
                    "for automatic pool growth. Pool will not be able to grow automatically."
                )
            
            # Validate growth_factor for specific policies
            if self.growth_policy == GrowthPolicy.LINEAR:
                if self.growth_factor < 1:
                    raise ValueError("LINEAR growth policy requires growth_factor >= 1")
                if self.growth_factor > 100:
                    warnings.warn(
                        f"LINEAR growth_factor of {self.growth_factor} is very large. "
                        "This may cause rapid memory consumption."
                    )
                    
            elif self.growth_policy == GrowthPolicy.EXPONENTIAL:
                if self.growth_factor <= 1:
                    raise ValueError("EXPONENTIAL growth policy requires growth_factor > 1")
                if self.growth_factor > 3:
                    warnings.warn(
                        f"EXPONENTIAL growth_factor of {self.growth_factor} is very aggressive. "
                        "This may cause extremely rapid memory consumption."
                    )
                    
        # Validate capacity relationships for growth policies
        if (self.max_capacity is not None and 
            self.growth_policy != GrowthPolicy.FIXED and 
            self.max_capacity <= self.initial_size):
            warnings.warn(
                "max_capacity equals initial_size with growth policy enabled. "
                "Pool will not be able to grow beyond initial size."
            )
            
        # Validate cleanup settings
        if self.auto_cleanup and not self.enable_statistics:
            warnings.warn(
                "auto_cleanup enabled but statistics disabled. "
                "Cleanup may not work optimally without usage statistics."
            )
            
        # Validate cleanup configuration
        if self.auto_cleanup and self.cleanup_interval <= 0:
            warnings.warn(
                "auto_cleanup enabled but cleanup_interval is 0. "
                "Consider setting cleanup_interval > 0 for background cleanup."
            ) 