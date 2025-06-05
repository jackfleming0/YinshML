"""Adaptive pool sizing and tensor reshaping capabilities for memory pools."""

import time
import torch
import logging
from typing import List, Tuple, Optional, Dict, Any, Deque
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveMetrics:
    """Enhanced metrics for adaptive pool management."""
    
    # Window-based tracking
    window_size: int = 100
    num_windows: int = 10
    allocation_history: Deque[Dict[str, int]] = field(default_factory=lambda: deque(maxlen=10))
    current_window: Dict[str, int] = field(default_factory=lambda: {'allocations': 0, 'hits': 0, 'misses': 0, 'reshapes': 0})
    
    # Moving averages
    ema_miss_rate: float = 0.0
    ema_allocation_rate: float = 0.0
    ema_alpha: float = 0.2  # Smoothing factor for exponential moving average
    
    # Timing metrics
    last_window_time: float = field(default_factory=time.time)
    window_count: int = 0
    
    def record_allocation(self, hit: bool, reshaped: bool = False) -> None:
        """Record an allocation event."""
        self.current_window['allocations'] += 1
        if hit:
            self.current_window['hits'] += 1
        else:
            self.current_window['misses'] += 1
            
        if reshaped:
            self.current_window['reshapes'] += 1
            
        if self.current_window['allocations'] >= self.window_size:
            self._rotate_window()
    
    def _rotate_window(self) -> None:
        """Rotate to a new tracking window and update moving averages."""
        allocations = self.current_window['allocations']
        if allocations > 0:
            miss_rate = self.current_window['misses'] / allocations
            
            # Update exponential moving averages
            self.ema_miss_rate = (self.ema_alpha * miss_rate) + ((1 - self.ema_alpha) * self.ema_miss_rate)
            
            # Calculate allocation rate (allocations per second)
            current_time = time.time()
            window_duration = current_time - self.last_window_time
            if window_duration > 0:
                allocation_rate = allocations / window_duration
                self.ema_allocation_rate = (self.ema_alpha * allocation_rate) + ((1 - self.ema_alpha) * self.ema_allocation_rate)
            
            self.last_window_time = current_time
        
        # Store the completed window
        self.allocation_history.append(dict(self.current_window))
        self.current_window = {'allocations': 0, 'hits': 0, 'misses': 0, 'reshapes': 0}
        self.window_count += 1
    
    def get_recent_miss_rate(self, num_windows: int = 3) -> float:
        """Get miss rate over recent windows."""
        if not self.allocation_history:
            return 0.0
            
        recent_windows = list(self.allocation_history)[-num_windows:]
        total_allocations = sum(w['allocations'] for w in recent_windows)
        total_misses = sum(w['misses'] for w in recent_windows)
        
        return total_misses / total_allocations if total_allocations > 0 else 0.0
    
    def get_reshape_efficiency(self, num_windows: int = 5) -> float:
        """Get the percentage of hits that were due to reshaping."""
        if not self.allocation_history:
            return 0.0
            
        recent_windows = list(self.allocation_history)[-num_windows:]
        total_hits = sum(w['hits'] for w in recent_windows)
        total_reshapes = sum(w['reshapes'] for w in recent_windows)
        
        return total_reshapes / total_hits if total_hits > 0 else 0.0


class TensorCompatibilityChecker:
    """Checks tensor compatibility for shape reuse and reshaping."""
    
    @staticmethod
    def is_compatible(source_tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> bool:
        """
        Determines if a tensor can be reshaped to the target shape without reallocation.
        
        Args:
            source_tensor: The source tensor to potentially reshape
            target_shape: The desired shape
            
        Returns:
            True if reshaping is possible and efficient
        """
        # Check 1: Element count - target must have fewer or equal elements
        source_elements = source_tensor.numel()
        target_elements = torch.prod(torch.tensor(target_shape)).item()
        if target_elements > source_elements:
            return False
            
        # Check 2: Contiguity - reshaping requires contiguous memory
        if not source_tensor.is_contiguous():
            return False
            
        # Check 3: Reasonable waste factor - don't reuse if too inefficient
        waste_factor = (source_elements - target_elements) / target_elements
        if waste_factor > 2.0:  # Don't waste more than 2x the needed memory
            return False
            
        return True
    
    @staticmethod
    def find_best_match(
        available_tensors: List[torch.Tensor], 
        target_shape: Tuple[int, ...], 
        dtype: torch.dtype, 
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Finds the most efficient tensor to reuse from a list of available tensors.
        
        Args:
            available_tensors: List of tensors that could potentially be reused
            target_shape: The desired shape
            dtype: Required data type
            device: Required device
            
        Returns:
            The best tensor to reuse, or None if no suitable tensor found
        """
        compatible_tensors = []
        target_elements = torch.prod(torch.tensor(target_shape)).item()
        
        for tensor in available_tensors:
            # Basic requirements
            if tensor.dtype != dtype or tensor.device != device:
                continue
                
            if TensorCompatibilityChecker.is_compatible(tensor, target_shape):
                # Calculate waste factor (how much extra memory we're using)
                source_elements = tensor.numel()
                waste_factor = (source_elements - target_elements) / target_elements
                
                # Prefer tensors with better memory layout compatibility
                layout_score = TensorCompatibilityChecker._calculate_layout_score(
                    tensor.shape, target_shape)
                
                # Combined score (lower is better)
                combined_score = waste_factor + (1.0 - layout_score)
                compatible_tensors.append((tensor, combined_score))
        
        if not compatible_tensors:
            return None
            
        # Return tensor with best (lowest) score
        return min(compatible_tensors, key=lambda x: x[1])[0]
    
    @staticmethod
    def _calculate_layout_score(source_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> float:
        """
        Calculate a score for memory layout compatibility.
        Higher scores indicate better compatibility.
        """
        # Prefer keeping the last dimension unchanged for better memory locality
        if len(source_shape) > 0 and len(target_shape) > 0:
            if source_shape[-1] == target_shape[-1]:
                return 1.0
            elif len(source_shape) > 1 and len(target_shape) > 1:
                # Check if we're just changing the batch dimension
                if source_shape[1:] == target_shape[1:]:
                    return 0.8
        
        # General case - assign moderate score
        return 0.5


class AdaptivePoolSizer:
    """Manages adaptive pool sizing based on usage patterns and memory pressure."""
    
    def __init__(self, 
                 initial_size: int = 100,
                 min_size: int = 50,
                 max_size: int = 10000,
                 growth_factor: float = 1.5,
                 shrink_factor: float = 0.8):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        
        # Thresholds for adaptation
        self.high_pressure_threshold = 0.7  # 70% miss rate indicates high pressure
        self.low_pressure_threshold = 0.2   # 20% miss rate indicates low pressure
        
        # State tracking
        self.consecutive_high_pressure = 0
        self.consecutive_low_pressure = 0
        self.required_consecutive_signals = 3  # Require 3 consecutive signals before resizing
        
        # Memory pressure integration
        self.memory_pressure_threshold = 0.85
        self.under_memory_pressure = False
        
    def update(self, metrics: AdaptiveMetrics) -> Optional[int]:
        """
        Update pool size based on current metrics.
        
        Args:
            metrics: Current adaptive metrics
            
        Returns:
            New pool size if changed, None if no change
        """
        # Check system memory pressure
        self._update_memory_pressure()
        
        miss_rate = metrics.ema_miss_rate
        
        if miss_rate > self.high_pressure_threshold:
            self.consecutive_high_pressure += 1
            self.consecutive_low_pressure = 0
            
            if self.consecutive_high_pressure >= self.required_consecutive_signals:
                new_size = self._grow_pool()
                self.consecutive_high_pressure = 0
                logger.info(f"Adaptive pool growth triggered: {self.current_size} -> {new_size} "
                           f"(miss rate: {miss_rate:.3f})")
                return new_size
                
        elif miss_rate < self.low_pressure_threshold:
            self.consecutive_low_pressure += 1
            self.consecutive_high_pressure = 0
            
            # More conservative shrinking (2x the signals required)
            if self.consecutive_low_pressure >= self.required_consecutive_signals * 2:
                new_size = self._shrink_pool()
                self.consecutive_low_pressure = 0
                logger.info(f"Adaptive pool shrink triggered: {self.current_size} -> {new_size} "
                           f"(miss rate: {miss_rate:.3f})")
                return new_size
        else:
            # Reset counters when in the normal range
            self.consecutive_high_pressure = 0
            self.consecutive_low_pressure = 0
            
        return None
    
    def _grow_pool(self) -> int:
        """Grow the pool size."""
        if self.under_memory_pressure:
            # More conservative growth under memory pressure
            factor = min(self.growth_factor, 1.2)
        else:
            factor = self.growth_factor
            
        new_size = min(int(self.current_size * factor), self.max_size)
        self.current_size = new_size
        return new_size
        
    def _shrink_pool(self) -> int:
        """Shrink the pool size."""
        if self.under_memory_pressure:
            # More aggressive shrinking under memory pressure
            factor = min(self.shrink_factor, 0.6)
        else:
            factor = self.shrink_factor
            
        new_size = max(int(self.current_size * factor), self.min_size)
        self.current_size = new_size
        return new_size
    
    def _update_memory_pressure(self) -> None:
        """Update memory pressure state."""
        try:
            memory_pressure = self._get_system_memory_pressure()
            was_under_pressure = self.under_memory_pressure
            self.under_memory_pressure = memory_pressure > self.memory_pressure_threshold
            
            if self.under_memory_pressure and not was_under_pressure:
                logger.warning(f"System memory pressure detected: {memory_pressure:.1%}")
                # Adjust thresholds to be more conservative
                self.high_pressure_threshold = 0.8
                self.required_consecutive_signals = 5
            elif not self.under_memory_pressure and was_under_pressure:
                logger.info("System memory pressure relieved")
                # Restore normal thresholds
                self.high_pressure_threshold = 0.7
                self.required_consecutive_signals = 3
                
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            
    def _get_system_memory_pressure(self) -> float:
        """Returns a value between 0-1 indicating system memory pressure."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return 1.0 - (memory.available / memory.total)
        except ImportError:
            # Fallback when psutil is not available
            return 0.5  # Assume moderate pressure
        except Exception:
            return 0.5
    
    def get_sizing_stats(self) -> Dict[str, Any]:
        """Get current sizing statistics."""
        return {
            'current_size': self.current_size,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'under_memory_pressure': self.under_memory_pressure,
            'consecutive_high_pressure': self.consecutive_high_pressure,
            'consecutive_low_pressure': self.consecutive_low_pressure,
            'high_pressure_threshold': self.high_pressure_threshold,
            'low_pressure_threshold': self.low_pressure_threshold
        }


def create_adaptive_metrics(window_size: int = 100, num_windows: int = 10) -> AdaptiveMetrics:
    """Create an AdaptiveMetrics instance with specified configuration."""
    return AdaptiveMetrics(window_size=window_size, num_windows=num_windows)


def create_adaptive_pool_sizer(
    initial_size: int = 100,
    min_size: int = 50,
    max_size: int = 10000,
    growth_factor: float = 1.5,
    shrink_factor: float = 0.8
) -> AdaptivePoolSizer:
    """Create an AdaptivePoolSizer with specified configuration."""
    return AdaptivePoolSizer(
        initial_size=initial_size,
        min_size=min_size,
        max_size=max_size,
        growth_factor=growth_factor,
        shrink_factor=shrink_factor
    ) 