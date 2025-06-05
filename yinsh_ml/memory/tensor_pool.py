"""TensorPool implementation for efficient neural network tensor management."""

from typing import Dict, Tuple, Optional, List, Any, Union, DefaultDict
from collections import defaultdict
import threading
import torch
import logging
from dataclasses import dataclass, field

from .pool import MemoryPool, PoolConfig, PoolStatistics
from .adaptive import (
    AdaptiveMetrics, 
    TensorCompatibilityChecker, 
    AdaptivePoolSizer,
    create_adaptive_metrics,
    create_adaptive_pool_sizer
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TensorPoolConfig(PoolConfig):
    """Configuration for TensorPool with tensor-specific settings."""
    
    # Device management
    auto_device_selection: bool = True
    """Automatically select best available device (cuda > mps > cpu)"""
    
    max_memory_per_device_mb: int = 1024
    """Maximum memory to use per device in MB"""
    
    enable_memory_tracking: bool = True
    """Enable detailed memory usage tracking"""
    
    enable_ref_counting: bool = True
    """Enable reference counting for shared tensors"""
    
    # Adaptive features
    enable_adaptive_sizing: bool = True
    """Enable automatic pool size adjustment based on usage patterns"""
    
    enable_tensor_reshaping: bool = True
    """Enable tensor reuse through reshaping when shapes are compatible"""
    
    adaptive_window_size: int = 100
    """Number of operations per adaptive metrics window"""
    
    adaptive_min_pool_size: int = 50
    """Minimum pool size for adaptive sizing"""
    
    adaptive_max_pool_size: int = 10000
    """Maximum pool size for adaptive sizing"""
    
    max_waste_factor: float = 2.0
    """Maximum acceptable memory waste factor for tensor reshaping"""


@dataclass
class TensorPoolStatistics(PoolStatistics):
    """Extended statistics for TensorPool with tensor-specific metrics."""
    
    # Tensor-specific metrics
    tensor_allocations: int = 0
    tensor_deallocations: int = 0
    tensor_reshapes: int = 0
    reference_increments: int = 0
    reference_decrements: int = 0
    
    # Memory tracking
    total_memory_allocated_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    memory_by_device: Dict[str, float] = field(default_factory=dict)
    memory_by_shape: Dict[Tuple[int, ...], float] = field(default_factory=dict)
    
    # Shape analysis
    shape_frequency: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    most_common_shapes: List[Tuple[Tuple[int, ...], int]] = field(default_factory=list)
    
    # Adaptive metrics
    adaptive_growths: int = 0
    adaptive_shrinks: int = 0
    avg_miss_rate: float = 0.0
    reshape_efficiency: float = 0.0


# Type alias for tensor key
TensorKey = Tuple[Tuple[int, ...], torch.dtype, str]  # (shape, dtype, device_str)


class TensorPool:
    """
    Memory pool for PyTorch tensors with shape-based indexing, device management,
    and adaptive sizing capabilities.
    """
    
    def __init__(self, config: Optional[TensorPoolConfig] = None):
        """Initialize TensorPool with configuration."""
        self.config = config or TensorPoolConfig()
        
        # Tensor-specific storage
        self._tensor_pools: DefaultDict[TensorKey, List[torch.Tensor]] = defaultdict(list)
        """Maps tensor keys to lists of available tensors"""
        
        self._in_use_tensors: Dict[int, Tuple[TensorKey, int]] = {}
        """Maps tensor id to (key, ref_count)"""
        
        self._creation_times: Dict[int, float] = {}
        """Track when tensors were created for lifetime statistics"""
        
        # Thread safety
        self._tensor_lock = threading.RLock()
        
        # Device management
        self._default_device = self._detect_default_device() if self.config.auto_device_selection else torch.device('cpu')
        self._device_memory: Dict[str, float] = defaultdict(float)
        
        # Statistics
        self.statistics = TensorPoolStatistics()
        
        # Adaptive sizing and reshaping
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
            
        self.compatibility_checker = TensorCompatibilityChecker()
        
        logger.info(f"TensorPool initialized with device: {self._default_device}, "
                   f"adaptive_sizing: {self.config.enable_adaptive_sizing}, "
                   f"tensor_reshaping: {self.config.enable_tensor_reshaping}")
    
    def get(self, 
            shape: Union[List[int], Tuple[int, ...]], 
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            zero_init: bool = True) -> torch.Tensor:
        """
        Get a tensor from the pool with specified shape, dtype, and device.
        
        Args:
            shape: Desired tensor shape
            dtype: Tensor data type
            device: Target device (uses default if None)
            zero_init: Whether to zero-initialize the tensor
            
        Returns:
            Tensor with the requested specifications
        """
        shape = tuple(shape)
        device = device or self._default_device
        device_str = str(device)
        key = (shape, dtype, device_str)
        
        with self._tensor_lock:
            tensor = None
            was_reshaped = False
            hit = False
            
            # Try to find exact match first
            if key in self._tensor_pools and self._tensor_pools[key]:
                tensor = self._tensor_pools[key].pop()
                hit = True
                logger.debug(f"Exact match found for shape {shape}")
                
            # Try tensor reshaping if enabled and no exact match
            elif self.config.enable_tensor_reshaping:
                tensor = self._find_compatible_tensor(shape, dtype, device)
                if tensor is not None:
                    # Remove from its original pool
                    original_key = self._get_tensor_key(tensor)
                    if original_key in self._tensor_pools:
                        try:
                            self._tensor_pools[original_key].remove(tensor)
                        except ValueError:
                            logger.warning(f"Tensor not found in expected pool for key {original_key}")
                    
                    # Reshape the tensor
                    try:
                        tensor = tensor.view(shape)
                        was_reshaped = True
                        hit = True
                        self.statistics.tensor_reshapes += 1
                        logger.debug(f"Tensor reshaped from {tensor.shape} to {shape}")
                    except Exception as e:
                        logger.warning(f"Failed to reshape tensor: {e}")
                        tensor = None
            
            # Create new tensor if no reusable tensor found
            if tensor is None:
                try:
                    tensor = torch.zeros(shape, dtype=dtype, device=device) if zero_init else torch.empty(shape, dtype=dtype, device=device)
                    self._track_new_tensor_memory(tensor)
                    logger.debug(f"Created new tensor with shape {shape}")
                except Exception as e:
                    logger.error(f"Failed to create tensor {shape} on {device}: {e}")
                    raise
            
            # Zero-initialize if requested and tensor was reused
            if zero_init and hit and not was_reshaped:
                tensor.zero_()
            
            # Track usage
            tensor_id = id(tensor)
            ref_count = 1
            if self.config.enable_ref_counting:
                self._in_use_tensors[tensor_id] = (key, ref_count)
                self._creation_times[tensor_id] = torch.cuda.Event(enable_timing=True).record() if device.type == 'cuda' else None
            
            # Update statistics
            self.statistics.tensor_allocations += 1
            self.statistics.shape_frequency[shape] = self.statistics.shape_frequency.get(shape, 0) + 1
            
            # Update adaptive metrics
            if self.adaptive_metrics is not None:
                self.adaptive_metrics.record_allocation(hit=hit, reshaped=was_reshaped)
                
                # Check if we should adapt pool size
                new_size = self.adaptive_sizer.update(self.adaptive_metrics)
                if new_size is not None:
                    if new_size > self.adaptive_sizer.current_size:
                        self.statistics.adaptive_growths += 1
                    else:
                        self.statistics.adaptive_shrinks += 1
                    self._adapt_pool_sizes(new_size)
            
            return tensor
    
    def get_batch(self, 
                  batch_size: int,
                  shape: Union[List[int], Tuple[int, ...]], 
                  dtype: torch.dtype = torch.float32,
                  device: Optional[torch.device] = None,
                  zero_init: bool = True) -> torch.Tensor:
        """
        Get a batched tensor with shape (batch_size, *shape).
        
        Args:
            batch_size: Number of examples in the batch
            shape: Shape of individual examples
            dtype: Tensor data type
            device: Target device
            zero_init: Whether to zero-initialize the tensor
            
        Returns:
            Batched tensor with shape (batch_size, *shape)
        """
        batch_shape = (batch_size,) + tuple(shape)
        return self.get(batch_shape, dtype, device, zero_init)
    
    def release(self, tensor: torch.Tensor) -> None:
        """
        Return a tensor to the pool.
        
        Args:
            tensor: Tensor to return to the pool
        """
        if tensor is None:
            return
            
        with self._tensor_lock:
            tensor_id = id(tensor)
            
            if self.config.enable_ref_counting and tensor_id in self._in_use_tensors:
                key, ref_count = self._in_use_tensors[tensor_id]
                
                if ref_count > 1:
                    # Decrement reference count
                    self._in_use_tensors[tensor_id] = (key, ref_count - 1)
                    self.statistics.reference_decrements += 1
                    return
                else:
                    # Remove from tracking
                    del self._in_use_tensors[tensor_id]
                    if tensor_id in self._creation_times:
                        del self._creation_times[tensor_id]
            
            # Determine the key for this tensor
            key = self._get_tensor_key(tensor)
            
            # Check memory limits before returning to pool
            if self._is_within_memory_limits(key[2]):  # device_str
                # Return to appropriate pool
                self._tensor_pools[key].append(tensor)
                logger.debug(f"Tensor returned to pool with key {key}")
            else:
                logger.debug(f"Tensor not returned to pool due to memory limits for device {key[2]}")
            
            self.statistics.tensor_deallocations += 1
    
    def increment_ref(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Increment reference count for a tensor.
        
        Args:
            tensor: Tensor to increment reference count for
            
        Returns:
            The same tensor (for convenience)
        """
        if not self.config.enable_ref_counting:
            return tensor
            
        with self._tensor_lock:
            tensor_id = id(tensor)
            if tensor_id in self._in_use_tensors:
                key, ref_count = self._in_use_tensors[tensor_id]
                self._in_use_tensors[tensor_id] = (key, ref_count + 1)
                self.statistics.reference_increments += 1
            else:
                logger.warning(f"Attempted to increment reference for untracked tensor")
                
        return tensor
    
    def clear_device(self, device: Union[str, torch.device]) -> int:
        """
        Clear all pooled tensors for a specific device.
        
        Args:
            device: Device to clear tensors for
            
        Returns:
            Number of tensors cleared
        """
        device_str = str(device)
        
        with self._tensor_lock:
            cleared_count = 0
            keys_to_clear = []
            
            for key in self._tensor_pools.keys():
                if key[2] == device_str:  # device_str is the third element
                    keys_to_clear.append(key)
                    
            for key in keys_to_clear:
                cleared_count += len(self._tensor_pools[key])
                self._tensor_pools[key].clear()
                
            # Update memory tracking
            if device_str in self._device_memory:
                self._device_memory[device_str] = 0.0
                
            logger.info(f"Cleared {cleared_count} tensors for device {device_str}")
            return cleared_count
    
    def get_statistics(self) -> TensorPoolStatistics:
        """Get current pool statistics."""
        with self._tensor_lock:
            # Update derived statistics
            if self.adaptive_metrics:
                self.statistics.avg_miss_rate = self.adaptive_metrics.ema_miss_rate
                self.statistics.reshape_efficiency = self.adaptive_metrics.get_reshape_efficiency()
            
            # Update most common shapes
            sorted_shapes = sorted(self.statistics.shape_frequency.items(), 
                                 key=lambda x: x[1], reverse=True)
            self.statistics.most_common_shapes = sorted_shapes[:10]
            
            return self.statistics
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        with self._tensor_lock:
            return {
                'total_memory_mb': self.statistics.total_memory_allocated_mb,
                'peak_memory_mb': self.statistics.peak_memory_usage_mb,
                'memory_by_device': dict(self.statistics.memory_by_device),
                'memory_by_shape': dict(self.statistics.memory_by_shape),
                'current_pools_memory': self._calculate_current_pool_memory()
            }
    
    def cleanup_old_tensors(self, max_age_seconds: float = 300.0) -> int:
        """
        Clean up old, unused tensors from the pools.
        
        Args:
            max_age_seconds: Maximum age for tensors to remain in pool
            
        Returns:
            Number of tensors cleaned up
        """
        if not self.config.enable_memory_tracking:
            return 0
            
        with self._tensor_lock:
            cleaned_count = 0
            current_time = torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else None
            
            # For now, implement simple cleanup based on pool size
            # More sophisticated age-based cleanup would require tracking tensor ages
            for key, tensors in self._tensor_pools.items():
                if len(tensors) > 10:  # Keep only the 10 most recent tensors per shape
                    excess_tensors = tensors[:-10]
                    self._tensor_pools[key] = tensors[-10:]
                    cleaned_count += len(excess_tensors)
                    
            logger.info(f"Cleaned up {cleaned_count} old tensors")
            return cleaned_count
    
    def _find_compatible_tensor(self, 
                               shape: Tuple[int, ...], 
                               dtype: torch.dtype, 
                               device: torch.device) -> Optional[torch.Tensor]:
        """Find a compatible tensor that can be reshaped to the target shape."""
        device_str = str(device)
        
        # Collect all available tensors with matching dtype and device
        available_tensors = []
        for key, tensors in self._tensor_pools.items():
            key_shape, key_dtype, key_device_str = key
            if key_dtype == dtype and key_device_str == device_str and tensors:
                available_tensors.extend(tensors)
        
        # Find the best match
        best_tensor = self.compatibility_checker.find_best_match(
            available_tensors, shape, dtype, device)
            
        return best_tensor
    
    def _get_tensor_key(self, tensor: torch.Tensor) -> TensorKey:
        """Get the pool key for a tensor."""
        return (tuple(tensor.shape), tensor.dtype, str(tensor.device))
    
    def _track_new_tensor_memory(self, tensor: torch.Tensor) -> None:
        """Track memory usage for a newly created tensor."""
        if not self.config.enable_memory_tracking:
            return
            
        memory_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
        device_str = str(tensor.device)
        shape = tuple(tensor.shape)
        
        # Update total and peak memory
        self.statistics.total_memory_allocated_mb += memory_mb
        self.statistics.peak_memory_usage_mb = max(
            self.statistics.peak_memory_usage_mb,
            self.statistics.total_memory_allocated_mb
        )
        
        # Update per-device memory
        self._device_memory[device_str] += memory_mb
        self.statistics.memory_by_device[device_str] = self._device_memory[device_str]
        
        # Update per-shape memory
        self.statistics.memory_by_shape[shape] = self.statistics.memory_by_shape.get(shape, 0) + memory_mb
    
    def _is_within_memory_limits(self, device_str: str) -> bool:
        """Check if device memory usage is within configured limits."""
        if not self.config.enable_memory_tracking:
            return True
            
        current_memory = self._device_memory.get(device_str, 0.0)
        return current_memory < self.config.max_memory_per_device_mb
    
    def _adapt_pool_sizes(self, new_target_size: int) -> None:
        """Adapt pool sizes based on adaptive sizing decisions."""
        # This is a placeholder for more sophisticated pool size adaptation
        # In practice, this might involve growing/shrinking specific shape pools
        # based on their usage patterns
        logger.debug(f"Adapting pool sizes with target: {new_target_size}")
        
        # For now, we just log the adaptation event
        # More sophisticated implementation would:
        # 1. Identify which shape pools to grow/shrink
        # 2. Preallocate tensors for growing pools
        # 3. Remove excess tensors from shrinking pools
    
    def _calculate_current_pool_memory(self) -> Dict[str, float]:
        """Calculate current memory usage of all pools."""
        pool_memory = {}
        
        for key, tensors in self._tensor_pools.items():
            shape, dtype, device_str = key
            if tensors:
                # Calculate memory for one tensor and multiply by count
                sample_tensor = tensors[0]
                memory_per_tensor = sample_tensor.element_size() * sample_tensor.numel() / (1024 * 1024)
                total_memory = memory_per_tensor * len(tensors)
                
                pool_key = f"{shape}_{dtype}_{device_str}"
                pool_memory[pool_key] = total_memory
                
        return pool_memory
    
    def _detect_default_device(self) -> torch.device:
        """Detect the best available device for tensor allocation."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("MPS (Apple Silicon) acceleration available")
            return device
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for tensor operations")
            return device
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            with self._tensor_lock:
                total_tensors = sum(len(tensors) for tensors in self._tensor_pools.values())
                if total_tensors > 0:
                    logger.info(f"TensorPool cleanup: releasing {total_tensors} pooled tensors")
                self._tensor_pools.clear()
                self._in_use_tensors.clear()
                self._creation_times.clear()
        except Exception as e:
            logger.warning(f"Error during TensorPool cleanup: {e}")


# Factory functions
def create_tensor(shape: Union[List[int], Tuple[int, ...]], 
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 zero_init: bool = True) -> torch.Tensor:
    """
    Factory function to create a new tensor.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        device: Target device
        zero_init: Whether to zero-initialize
        
    Returns:
        New tensor
    """
    device = device or torch.device('cpu')
    if zero_init:
        return torch.zeros(shape, dtype=dtype, device=device)
    else:
        return torch.empty(shape, dtype=dtype, device=device)


def reset_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reset a tensor to zero values.
    
    Args:
        tensor: Tensor to reset
        
    Returns:
        The reset tensor
    """
    tensor.zero_()
    return tensor


def validate_tensor_reset(tensor: torch.Tensor) -> bool:
    """
    Validate that a tensor has been properly reset.
    
    Args:
        tensor: Tensor to validate
        
    Returns:
        True if tensor is properly reset
    """
    try:
        return torch.allclose(tensor, torch.zeros_like(tensor))
    except Exception:
        return False


def create_tensor_pool(initial_size: int = 100,
                      auto_device_selection: bool = True,
                      enable_adaptive_sizing: bool = True,
                      enable_tensor_reshaping: bool = True,
                      enable_ref_counting: bool = True,
                      enable_memory_tracking: bool = True) -> TensorPool:
    """
    Create a TensorPool with optimized defaults.
    
    Args:
        initial_size: Initial pool size
        auto_device_selection: Enable automatic device selection
        enable_adaptive_sizing: Enable adaptive pool sizing
        enable_tensor_reshaping: Enable tensor reshaping for reuse
        enable_ref_counting: Enable reference counting
        enable_memory_tracking: Enable memory usage tracking
        
    Returns:
        Configured TensorPool instance
    """
    config = TensorPoolConfig(
        initial_size=initial_size,
        auto_device_selection=auto_device_selection,
        enable_adaptive_sizing=enable_adaptive_sizing,
        enable_tensor_reshaping=enable_tensor_reshaping,
        enable_ref_counting=enable_ref_counting,
        enable_memory_tracking=enable_memory_tracking
    )
    
    return TensorPool(config) 