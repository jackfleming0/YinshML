"""Zero-copy tensor operations for efficient memory management."""

from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import warnings
from contextlib import contextmanager
import torch
import numpy as np
from dataclasses import dataclass, field

from .shared_memory import (
    SharedMemoryRegion, 
    SharedMemoryManager, 
    SharedMemoryConfig,
    get_shared_memory_manager
)
from .tensor_pool import TensorPool, TensorPoolConfig


@dataclass
class ZeroCopyConfig:
    """Configuration for zero-copy tensor operations."""
    
    # Shared memory settings
    enable_shared_memory_tensors: bool = True
    """Enable tensor allocation in shared memory regions"""
    
    shared_memory_page_size: int = 65536  # 64KB
    """Page size for shared memory allocations"""
    
    # In-place operation settings
    enable_inplace_operations: bool = True
    """Enable in-place tensor operations where safe"""
    
    inplace_threshold_mb: float = 1.0
    """Minimum tensor size in MB to consider for in-place operations"""
    
    # View operation settings
    enable_view_operations: bool = True
    """Enable tensor view operations instead of copying"""
    
    view_alignment_bytes: int = 64
    """Memory alignment for efficient view operations"""
    
    # Buffer management
    enable_persistent_buffers: bool = True
    """Enable persistent tensor buffers for common operations"""
    
    max_buffer_pool_size_mb: int = 512
    """Maximum size of buffer pool in MB"""
    
    buffer_reuse_threshold: float = 0.8
    """Threshold for buffer size compatibility (0.8 = 80% size match)"""


@dataclass
class ZeroCopyStatistics:
    """Statistics for zero-copy operations."""
    
    # Operation counts
    shared_memory_allocations: int = 0
    view_operations: int = 0
    inplace_operations: int = 0
    buffer_reuses: int = 0
    copy_avoided_count: int = 0
    
    # Memory savings
    total_memory_saved_mb: float = 0.0
    peak_memory_saved_mb: float = 0.0
    
    # Performance metrics
    avg_allocation_time_ms: float = 0.0
    avg_copy_avoidance_speedup: float = 0.0
    
    # Error tracking
    failed_view_operations: int = 0
    failed_inplace_operations: int = 0
    fallback_copies: int = 0


class ZeroCopyTensorFactory:
    """Factory for creating zero-copy tensors using shared memory."""
    
    def __init__(self, config: ZeroCopyConfig):
        """Initialize zero-copy tensor factory."""
        self.config = config
        self.statistics = ZeroCopyStatistics()
        self._lock = threading.RLock()
        
        # Initialize shared memory manager
        if self.config.enable_shared_memory_tensors:
            self.shared_memory_manager = get_shared_memory_manager()
        else:
            self.shared_memory_manager = None
            
        # Initialize tensor pool if persistent buffers are enabled
        if self.config.enable_persistent_buffers:
            tensor_pool_config = TensorPoolConfig(
                initial_size=min(100, self.config.max_buffer_pool_size_mb * 10),
                max_capacity=self.config.max_buffer_pool_size_mb * 10,
                enable_adaptive_sizing=True,
                enable_tensor_reshaping=True,
                enable_ref_counting=True
            )
            self.tensor_pool = TensorPool(tensor_pool_config)
        else:
            self.tensor_pool = None
    
    def create_shared_tensor(self, 
                           shape: Union[List[int], Tuple[int, ...]],
                           dtype: torch.dtype = torch.float32,
                           device: Optional[torch.device] = None,
                           name: Optional[str] = None) -> torch.Tensor:
        """
        Create a tensor backed by shared memory.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device (CPU only for shared memory)
            name: Optional name for the shared memory region
            
        Returns:
            Tensor backed by shared memory
        """
        if not self.config.enable_shared_memory_tensors or self.shared_memory_manager is None:
            # Fallback to regular tensor creation
            device = device or torch.device('cpu')
            return torch.zeros(shape, dtype=dtype, device=device)
        
        if device is not None and device.type != 'cpu':
            warnings.warn(f"Shared memory tensors only support CPU device, got {device}")
            device = torch.device('cpu')
        
        shape = tuple(shape)
        
        # Calculate memory requirements
        element_count = np.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = element_count * element_size
        
        # Align to page boundaries for efficiency
        aligned_bytes = ((total_bytes + self.config.shared_memory_page_size - 1) 
                        // self.config.shared_memory_page_size * self.config.shared_memory_page_size)
        
        with self._lock:
            try:
                # Create shared memory region
                region_name = name or f"tensor_{id(self)}_{len(shape)}D"
                region = self.shared_memory_manager.create_region(
                    name=region_name,
                    size=aligned_bytes
                )
                
                # For now, fall back to regular tensor and store the region reference
                # TODO: Implement proper shared memory tensor mapping
                tensor = torch.zeros(shape, dtype=dtype, device=torch.device('cpu'))
                
                # Store reference to shared memory region to prevent garbage collection
                tensor._shared_memory_region = region
                
                self.statistics.shared_memory_allocations += 1
                
                return tensor
                
            except Exception as e:
                self.statistics.fallback_copies += 1
                warnings.warn(f"Failed to create shared memory tensor: {e}, falling back to regular tensor")
                return torch.zeros(shape, dtype=dtype, device=torch.device('cpu'))
    
    def create_view_tensor(self, 
                          source: torch.Tensor,
                          shape: Union[List[int], Tuple[int, ...]],
                          offset: int = 0) -> Optional[torch.Tensor]:
        """
        Create a tensor view from an existing tensor without copying data.
        
        Args:
            source: Source tensor to create view from
            shape: Desired shape for the view
            offset: Byte offset into the source tensor
            
        Returns:
            Tensor view or None if view is not possible
        """
        if not self.config.enable_view_operations:
            return None
            
        try:
            shape = tuple(shape)
            
            # Check if view is possible
            source_elements = source.numel()
            target_elements = np.prod(shape)
            
            if target_elements > source_elements - offset:
                self.statistics.failed_view_operations += 1
                return None
            
            # Create storage view
            if offset > 0:
                # Create a view starting from the offset
                flat_view = source.flatten()[offset:offset + target_elements]
            else:
                flat_view = source.flatten()[:target_elements]
            
            # Reshape to target shape
            view_tensor = flat_view.view(shape)
            
            self.statistics.view_operations += 1
            self.statistics.copy_avoided_count += 1
            
            return view_tensor
            
        except Exception as e:
            self.statistics.failed_view_operations += 1
            return None
    
    def get_persistent_buffer(self,
                            shape: Union[List[int], Tuple[int, ...]],
                            dtype: torch.dtype = torch.float32,
                            device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get a persistent buffer tensor that can be reused across operations.
        
        Args:
            shape: Desired tensor shape
            dtype: Data type
            device: Target device
            
        Returns:
            Persistent buffer tensor
        """
        if not self.config.enable_persistent_buffers or self.tensor_pool is None:
            device = device or torch.device('cpu')
            return torch.zeros(shape, dtype=dtype, device=device)
        
        try:
            # Try to get from tensor pool
            tensor = self.tensor_pool.get(shape, dtype, device, zero_init=True)
            self.statistics.buffer_reuses += 1
            return tensor
            
        except Exception as e:
            self.statistics.fallback_copies += 1
            device = device or torch.device('cpu')
            return torch.zeros(shape, dtype=dtype, device=device)
    
    def release_buffer(self, tensor: torch.Tensor) -> None:
        """Release a persistent buffer back to the pool."""
        if self.tensor_pool is not None:
            self.tensor_pool.release(tensor)


class InPlaceOperations:
    """Utilities for safe in-place tensor operations."""
    
    def __init__(self, config: Optional[ZeroCopyConfig] = None):
        """Initialize in-place operations handler."""
        self.config = config or ZeroCopyConfig()
        self.statistics = ZeroCopyStatistics()
    
    def safe_add_(self, target: torch.Tensor, source: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Safely add tensors in-place if conditions are met."""
        if not self._can_operate_inplace(target, source):
            return target.add(source, alpha=alpha)
        
        try:
            target.add_(source, alpha=alpha)
            self.statistics.inplace_operations += 1
            self.statistics.copy_avoided_count += 1
            return target
        except Exception:
            self.statistics.failed_inplace_operations += 1
            return target.add(source, alpha=alpha)
    
    def safe_mul_(self, target: torch.Tensor, source: Union[torch.Tensor, float]) -> torch.Tensor:
        """Safely multiply tensors in-place if conditions are met."""
        if isinstance(source, torch.Tensor) and not self._can_operate_inplace(target, source):
            return target.mul(source)
        
        try:
            target.mul_(source)
            self.statistics.inplace_operations += 1
            self.statistics.copy_avoided_count += 1
            return target
        except Exception:
            self.statistics.failed_inplace_operations += 1
            if isinstance(source, torch.Tensor):
                return target.mul(source)
            else:
                return target.mul(source)
    
    def safe_copy_(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Safely copy tensor data in-place if conditions are met."""
        if not self._can_operate_inplace(target, source):
            return source.clone()
        
        try:
            target.copy_(source)
            self.statistics.inplace_operations += 1
            self.statistics.copy_avoided_count += 1
            return target
        except Exception:
            self.statistics.failed_inplace_operations += 1
            return source.clone()
    
    def _can_operate_inplace(self, target: torch.Tensor, source: torch.Tensor) -> bool:
        """Check if in-place operation is safe and beneficial."""
        if not self.config.enable_inplace_operations:
            return False
        
        # Check size threshold
        target_size_mb = target.numel() * target.element_size() / (1024 * 1024)
        if target_size_mb < self.config.inplace_threshold_mb:
            return False
        
        # Check shape compatibility
        if target.shape != source.shape:
            return False
        
        # Check device compatibility
        if target.device != source.device:
            return False
        
        # Check dtype compatibility
        if target.dtype != source.dtype:
            return False
        
        # Check if tensors are contiguous
        if not (target.is_contiguous() and source.is_contiguous()):
            return False
        
        return True


class ZeroCopyBatchProcessor:
    """Optimized batch processing with zero-copy operations."""
    
    def __init__(self, 
                 tensor_factory: ZeroCopyTensorFactory,
                 inplace_ops: InPlaceOperations):
        """Initialize batch processor."""
        self.tensor_factory = tensor_factory
        self.inplace_ops = inplace_ops
        self.statistics = ZeroCopyStatistics()
    
    def create_batch_from_numpy(self,
                               arrays: List[np.ndarray],
                               dtype: torch.dtype = torch.float32,
                               device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create a batched tensor from numpy arrays with minimal copying.
        
        Args:
            arrays: List of numpy arrays to batch
            dtype: Target tensor dtype
            device: Target device
            
        Returns:
            Batched tensor
        """
        if not arrays:
            return torch.empty(0, dtype=dtype, device=device or torch.device('cpu'))
        
        # Validate array shapes are consistent
        sample_shape = arrays[0].shape
        for i, array in enumerate(arrays):
            if array.shape != sample_shape:
                raise ValueError(f"Inconsistent array shapes: {array.shape} vs {sample_shape}")
        
        # Get batch dimensions
        batch_size = len(arrays)
        batch_shape = (batch_size,) + sample_shape
        
        try:
            # Try to create batch using persistent buffer on CPU first
            batch_tensor = self.tensor_factory.get_persistent_buffer(
                batch_shape, dtype, torch.device('cpu')
            )
            
            # Fill the batch tensor efficiently
            for i, array in enumerate(arrays):
                # Convert numpy to tensor without copying if possible
                if array.dtype == np.dtype(str(dtype).replace('torch.', '')):
                    source_tensor = torch.from_numpy(array)
                else:
                    source_tensor = torch.tensor(array, dtype=dtype)
                
                # Copy into batch tensor slot
                batch_tensor[i].copy_(source_tensor)
            
            # Move to target device if needed
            if device is not None and batch_tensor.device != device:
                batch_tensor = batch_tensor.to(device)
            
            self.statistics.copy_avoided_count += 1
            return batch_tensor
            
        except Exception as e:
            self.statistics.fallback_copies += 1
            # Fallback to standard torch.stack
            tensors = [torch.tensor(arr, dtype=dtype, device=device or torch.device('cpu')) for arr in arrays]
            return torch.stack(tensors)


# Global instances for convenience
_default_config = ZeroCopyConfig()
_tensor_factory = ZeroCopyTensorFactory(_default_config)
_inplace_ops = InPlaceOperations(_default_config)
_batch_processor = ZeroCopyBatchProcessor(_tensor_factory, _inplace_ops)


# Convenience functions
def create_shared_tensor(shape: Union[List[int], Tuple[int, ...]],
                        dtype: torch.dtype = torch.float32,
                        device: Optional[torch.device] = None,
                        name: Optional[str] = None) -> torch.Tensor:
    """Create a tensor backed by shared memory."""
    return _tensor_factory.create_shared_tensor(shape, dtype, device, name)


def create_view_tensor(source: torch.Tensor,
                      shape: Union[List[int], Tuple[int, ...]],
                      offset: int = 0) -> Optional[torch.Tensor]:
    """Create a tensor view without copying data."""
    return _tensor_factory.create_view_tensor(source, shape, offset)


def get_persistent_buffer(shape: Union[List[int], Tuple[int, ...]],
                         dtype: torch.dtype = torch.float32,
                         device: Optional[torch.device] = None) -> torch.Tensor:
    """Get a persistent buffer tensor."""
    return _tensor_factory.get_persistent_buffer(shape, dtype, device)


def release_buffer(tensor: torch.Tensor) -> None:
    """Release a persistent buffer."""
    _tensor_factory.release_buffer(tensor)


def safe_add_(target: torch.Tensor, source: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Safely add tensors in-place."""
    return _inplace_ops.safe_add_(target, source, alpha)


def safe_mul_(target: torch.Tensor, source: Union[torch.Tensor, float]) -> torch.Tensor:
    """Safely multiply tensors in-place."""
    return _inplace_ops.safe_mul_(target, source)


def safe_copy_(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Safely copy tensor data in-place."""
    return _inplace_ops.safe_copy_(target, source)


def create_batch_from_numpy(arrays: List[np.ndarray],
                          dtype: torch.dtype = torch.float32,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a batched tensor from numpy arrays with minimal copying."""
    return _batch_processor.create_batch_from_numpy(arrays, dtype, device)


@contextmanager
def zero_copy_context(config: Optional[ZeroCopyConfig] = None):
    """Context manager for zero-copy operations with custom configuration."""
    global _tensor_factory, _inplace_ops, _batch_processor
    
    # Save current instances
    old_factory = _tensor_factory
    old_inplace = _inplace_ops
    old_batch = _batch_processor
    
    try:
        # Create new instances with custom config
        if config is not None:
            _tensor_factory = ZeroCopyTensorFactory(config)
            _inplace_ops = InPlaceOperations(config)
            _batch_processor = ZeroCopyBatchProcessor(_tensor_factory, _inplace_ops)
        
        yield {
            'factory': _tensor_factory,
            'inplace': _inplace_ops,
            'batch_processor': _batch_processor
        }
        
    finally:
        # Restore original instances
        _tensor_factory = old_factory
        _inplace_ops = old_inplace
        _batch_processor = old_batch


def get_zero_copy_statistics() -> Dict[str, ZeroCopyStatistics]:
    """Get statistics from all zero-copy components."""
    return {
        'tensor_factory': _tensor_factory.statistics,
        'inplace_operations': _inplace_ops.statistics,
        'batch_processor': _batch_processor.statistics
    } 