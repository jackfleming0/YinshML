"""Zero-copy GPU utilities and optimized host-device transfers."""

from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
import torch
import numpy as np

from .zero_copy import ZeroCopyConfig, ZeroCopyStatistics
from .tensor_pool import TensorPool, TensorPoolConfig


@dataclass
class GPUTransferConfig:
    """Configuration for GPU transfer optimizations."""
    
    # Memory pinning
    enable_pinned_memory: bool = True
    """Enable pinned memory for faster CPU-GPU transfers"""
    
    pinned_memory_pool_size_mb: int = 256
    """Size of pinned memory pool in MB"""
    
    auto_pin_threshold_mb: float = 1.0
    """Minimum tensor size to auto-pin memory"""
    
    # Asynchronous transfers
    enable_async_transfers: bool = True
    """Enable asynchronous GPU transfers"""
    
    transfer_streams: int = 2
    """Number of CUDA streams for async transfers"""
    
    # Batch transfers
    enable_batch_transfers: bool = True
    """Batch multiple small transfers into larger ones"""
    
    batch_transfer_timeout_ms: float = 1.0
    """Max time to wait for batching transfers"""
    
    batch_size_threshold: int = 8
    """Min number of tensors to trigger batch transfer"""
    
    # Placement optimization
    enable_smart_placement: bool = True
    """Automatically optimize tensor device placement"""
    
    placement_history_size: int = 1000
    """Number of operations to track for placement decisions"""
    
    cpu_gpu_switch_penalty: float = 0.1
    """Cost penalty for switching between CPU and GPU"""


@dataclass
class GPUTransferStatistics(ZeroCopyStatistics):
    """Statistics for GPU transfer optimizations."""
    
    # Transfer metrics
    host_to_device_transfers: int = 0
    device_to_host_transfers: int = 0
    async_transfers: int = 0
    sync_transfers: int = 0
    batched_transfers: int = 0
    
    # Memory metrics
    pinned_memory_allocations: int = 0
    pinned_memory_reuses: int = 0
    total_transfer_time_ms: float = 0.0
    avg_transfer_speed_gbps: float = 0.0
    
    # Placement metrics
    optimal_placements: int = 0
    suboptimal_placements: int = 0
    placement_switches: int = 0


class ZeroCopyTensorWrapper:
    """Wrapper for tensors with zero-copy tracking and optimization."""
    
    def __init__(self, 
                 tensor: torch.Tensor,
                 name: Optional[str] = None,
                 track_copies: bool = True):
        """
        Initialize tensor wrapper.
        
        Args:
            tensor: The wrapped tensor
            name: Optional name for debugging
            track_copies: Whether to track copy operations
        """
        self._tensor = tensor
        self.name = name or f"tensor_{id(tensor)}"
        self.track_copies = track_copies
        
        # Copy tracking
        self.copy_count = 0
        self.last_copy_time = None
        self.access_pattern = deque(maxlen=100)  # Track recent operations
        
        # Device transfer tracking
        self.transfer_count = 0
        self.last_device = tensor.device
        
        # Memory pinning state
        self.is_pinned = tensor.is_pinned() if tensor.device.type == 'cpu' else False
    
    @property
    def tensor(self) -> torch.Tensor:
        """Get the underlying tensor."""
        self.access_pattern.append(('access', time.time()))
        return self._tensor
    
    def clone(self, name_suffix: str = "_clone") -> 'ZeroCopyTensorWrapper':
        """Create a tracked clone of the tensor."""
        if self.track_copies:
            self.copy_count += 1
            self.last_copy_time = time.time()
            self.access_pattern.append(('clone', time.time()))
        
        cloned_tensor = self._tensor.clone()
        new_name = self.name + name_suffix if self.name else None
        return ZeroCopyTensorWrapper(cloned_tensor, new_name, self.track_copies)
    
    def to(self, device: Union[str, torch.device], non_blocking: bool = True) -> 'ZeroCopyTensorWrapper':
        """Transfer tensor to device with tracking."""
        device = torch.device(device) if isinstance(device, str) else device
        
        if device != self.last_device:
            self.transfer_count += 1
            self.last_device = device
            self.access_pattern.append(('transfer', time.time(), str(device)))
        
        transferred_tensor = self._tensor.to(device, non_blocking=non_blocking)
        return ZeroCopyTensorWrapper(transferred_tensor, self.name, self.track_copies)
    
    def pin_memory(self) -> 'ZeroCopyTensorWrapper':
        """Pin tensor memory for faster transfers."""
        if self._tensor.device.type == 'cpu' and not self.is_pinned:
            pinned_tensor = self._tensor.pin_memory()
            wrapper = ZeroCopyTensorWrapper(pinned_tensor, self.name, self.track_copies)
            wrapper.is_pinned = True
            return wrapper
        return self
    
    def get_copy_stats(self) -> Dict[str, Any]:
        """Get copy tracking statistics."""
        return {
            'name': self.name,
            'copy_count': self.copy_count,
            'transfer_count': self.transfer_count,
            'last_copy_time': self.last_copy_time,
            'is_pinned': self.is_pinned,
            'current_device': str(self._tensor.device),
            'access_count': len(self.access_pattern)
        }


class PinnedMemoryPool:
    """Pool of pinned memory tensors for fast CPU-GPU transfers."""
    
    def __init__(self, config: GPUTransferConfig):
        """Initialize pinned memory pool."""
        self.config = config
        self.statistics = GPUTransferStatistics()
        
        # Pool storage
        self._pinned_tensors: Dict[Tuple[int, ...], List[torch.Tensor]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Memory tracking
        self._total_pinned_mb = 0.0
        self._max_pool_size_mb = config.pinned_memory_pool_size_mb
    
    def get_pinned_tensor(self, 
                         shape: Union[List[int], Tuple[int, ...]],
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get a pinned memory tensor.
        
        Args:
            shape: Desired tensor shape
            dtype: Tensor data type
            
        Returns:
            Pinned memory tensor
        """
        if not self.config.enable_pinned_memory:
            return torch.zeros(shape, dtype=dtype)
        
        shape = tuple(shape)
        
        with self._lock:
            # Try to reuse existing pinned tensor
            if shape in self._pinned_tensors and self._pinned_tensors[shape]:
                tensor = self._pinned_tensors[shape].pop()
                tensor.zero_()  # Clear data
                self.statistics.pinned_memory_reuses += 1
                return tensor
            
            # Create new pinned tensor if within memory limits
            tensor_size_mb = np.prod(shape) * torch.tensor([], dtype=dtype).element_size() / (1024 * 1024)
            
            if self._total_pinned_mb + tensor_size_mb <= self._max_pool_size_mb:
                try:
                    tensor = torch.zeros(shape, dtype=dtype).pin_memory()
                    self._total_pinned_mb += tensor_size_mb
                    self.statistics.pinned_memory_allocations += 1
                    return tensor
                except Exception as e:
                    warnings.warn(f"Failed to allocate pinned memory: {e}")
            
            # Fallback to regular tensor
            return torch.zeros(shape, dtype=dtype)
    
    def return_pinned_tensor(self, tensor: torch.Tensor) -> None:
        """Return a pinned tensor to the pool."""
        if not tensor.is_pinned():
            return
        
        shape = tuple(tensor.shape)
        
        with self._lock:
            # Only keep reasonable number of tensors per shape
            if len(self._pinned_tensors[shape]) < 10:
                self._pinned_tensors[shape].append(tensor)
    
    def cleanup(self) -> None:
        """Clean up pinned memory pool."""
        with self._lock:
            self._pinned_tensors.clear()
            self._total_pinned_mb = 0.0


class AsyncTransferManager:
    """Manages asynchronous tensor transfers between devices."""
    
    def __init__(self, config: GPUTransferConfig):
        """Initialize async transfer manager."""
        self.config = config
        self.statistics = GPUTransferStatistics()
        
        # CUDA streams for async transfers
        self.streams = []
        if torch.cuda.is_available() and config.enable_async_transfers:
            for _ in range(config.transfer_streams):
                self.streams.append(torch.cuda.Stream())
        
        self._current_stream_idx = 0
        self._lock = threading.RLock()
        
        # Batch transfer queue
        self._transfer_queue = deque()
        self._batch_timer = None
    
    def async_transfer(self, 
                      tensor: torch.Tensor,
                      target_device: Union[str, torch.device],
                      callback: Optional[callable] = None) -> torch.Tensor:
        """
        Perform asynchronous tensor transfer.
        
        Args:
            tensor: Tensor to transfer
            target_device: Target device
            callback: Optional callback when transfer completes
            
        Returns:
            Tensor on target device
        """
        target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        
        if not self.config.enable_async_transfers or not self.streams or tensor.device == target_device:
            # Synchronous fallback
            result = tensor.to(target_device, non_blocking=False)
            self.statistics.sync_transfers += 1
            if callback:
                callback(result)
            return result
        
        with self._lock:
            # Select stream in round-robin fashion
            stream = self.streams[self._current_stream_idx]
            self._current_stream_idx = (self._current_stream_idx + 1) % len(self.streams)
        
        with torch.cuda.stream(stream):
            start_time = time.time()
            result = tensor.to(target_device, non_blocking=True)
            
            # Record transfer stats
            transfer_time = (time.time() - start_time) * 1000  # ms
            self.statistics.total_transfer_time_ms += transfer_time
            self.statistics.async_transfers += 1
            
            if target_device.type == 'cuda':
                self.statistics.host_to_device_transfers += 1
            else:
                self.statistics.device_to_host_transfers += 1
            
            if callback:
                # Schedule callback on stream completion
                torch.cuda.current_stream().wait_stream(stream)
                callback(result)
        
        return result
    
    def batch_transfer(self, 
                      tensors: List[torch.Tensor],
                      target_device: Union[str, torch.device]) -> List[torch.Tensor]:
        """
        Perform batched tensor transfers.
        
        Args:
            tensors: List of tensors to transfer
            target_device: Target device
            
        Returns:
            List of transferred tensors
        """
        if not self.config.enable_batch_transfers or len(tensors) < 2:
            # Individual transfers
            return [self.async_transfer(t, target_device) for t in tensors]
        
        target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        
        # Group tensors by source device
        device_groups = defaultdict(list)
        for i, tensor in enumerate(tensors):
            device_groups[tensor.device].append((i, tensor))
        
        results = [None] * len(tensors)
        
        for source_device, tensor_list in device_groups.items():
            if source_device == target_device:
                # No transfer needed
                for idx, tensor in tensor_list:
                    results[idx] = tensor
                continue
            
            # Batch transfer for this device group
            if len(tensor_list) >= self.config.batch_size_threshold:
                with self._lock:
                    stream = self.streams[self._current_stream_idx] if self.streams else None
                    if stream:
                        self._current_stream_idx = (self._current_stream_idx + 1) % len(self.streams)
                
                if stream:
                    with torch.cuda.stream(stream):
                        for idx, tensor in tensor_list:
                            results[idx] = tensor.to(target_device, non_blocking=True)
                        self.statistics.batched_transfers += 1
                else:
                    # Fallback to sync transfers
                    for idx, tensor in tensor_list:
                        results[idx] = tensor.to(target_device, non_blocking=False)
            else:
                # Individual transfers for small batches
                for idx, tensor in tensor_list:
                    results[idx] = self.async_transfer(tensor, target_device)
        
        return results


class SmartPlacementOptimizer:
    """Optimizes tensor device placement based on usage patterns."""
    
    def __init__(self, config: GPUTransferConfig):
        """Initialize placement optimizer."""
        self.config = config
        self.statistics = GPUTransferStatistics()
        
        # Track operation patterns
        self.operation_history = deque(maxlen=config.placement_history_size)
        self.tensor_usage = defaultdict(lambda: {'cpu_ops': 0, 'gpu_ops': 0, 'transfers': 0})
        
        self._lock = threading.RLock()
    
    def record_operation(self, 
                        tensor_id: str,
                        operation: str,
                        device: torch.device) -> None:
        """Record tensor operation for placement optimization."""
        with self._lock:
            self.operation_history.append({
                'tensor_id': tensor_id,
                'operation': operation,
                'device': device,
                'timestamp': time.time()
            })
            
            # Update usage stats
            usage = self.tensor_usage[tensor_id]
            if device.type == 'cpu':
                usage['cpu_ops'] += 1
            else:
                usage['gpu_ops'] += 1
    
    def suggest_placement(self, tensor_id: str, current_device: torch.device) -> torch.device:
        """
        Suggest optimal device placement for a tensor.
        
        Args:
            tensor_id: Unique identifier for the tensor
            current_device: Current device of the tensor
            
        Returns:
            Suggested optimal device
        """
        if not self.config.enable_smart_placement:
            return current_device
        
        with self._lock:
            usage = self.tensor_usage[tensor_id]
            
            # Simple heuristic: prefer device with more operations
            cpu_score = usage['cpu_ops']
            gpu_score = usage['gpu_ops']
            
            # Apply transfer penalty
            if current_device.type == 'cpu':
                cpu_score += self.config.cpu_gpu_switch_penalty
            else:
                gpu_score += self.config.cpu_gpu_switch_penalty
            
            # Suggest device based on scores
            if gpu_score > cpu_score and torch.cuda.is_available():
                suggested = torch.device('cuda')
            else:
                suggested = torch.device('cpu')
            
            # Track placement decisions
            if suggested == current_device:
                self.statistics.optimal_placements += 1
            else:
                self.statistics.suboptimal_placements += 1
                self.statistics.placement_switches += 1
            
            return suggested
    
    def get_placement_stats(self) -> Dict[str, Any]:
        """Get placement optimization statistics."""
        with self._lock:
            total_tensors = len(self.tensor_usage)
            avg_cpu_ops = np.mean([usage['cpu_ops'] for usage in self.tensor_usage.values()]) if total_tensors > 0 else 0
            avg_gpu_ops = np.mean([usage['gpu_ops'] for usage in self.tensor_usage.values()]) if total_tensors > 0 else 0
            
            return {
                'tracked_tensors': total_tensors,
                'avg_cpu_operations': avg_cpu_ops,
                'avg_gpu_operations': avg_gpu_ops,
                'optimal_placements': self.statistics.optimal_placements,
                'suboptimal_placements': self.statistics.suboptimal_placements,
                'placement_switches': self.statistics.placement_switches
            }


class ZeroCopyGPUManager:
    """Comprehensive GPU optimization manager."""
    
    def __init__(self, config: Optional[GPUTransferConfig] = None):
        """Initialize GPU optimization manager."""
        self.config = config or GPUTransferConfig()
        self.statistics = GPUTransferStatistics()
        
        # Initialize components
        self.pinned_memory_pool = PinnedMemoryPool(self.config)
        self.async_transfer_manager = AsyncTransferManager(self.config)
        self.placement_optimizer = SmartPlacementOptimizer(self.config)
        
        # Tensor tracking
        self.tracked_tensors: Dict[str, ZeroCopyTensorWrapper] = {}
        self._lock = threading.RLock()
    
    def create_tensor(self, 
                     shape: Union[List[int], Tuple[int, ...]],
                     dtype: torch.dtype = torch.float32,
                     device: Union[str, torch.device] = 'cpu',
                     pin_memory: bool = False,
                     name: Optional[str] = None) -> ZeroCopyTensorWrapper:
        """
        Create an optimized tensor with zero-copy tracking.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            pin_memory: Whether to use pinned memory
            name: Optional tensor name
            
        Returns:
            Wrapped tensor with optimizations
        """
        device = torch.device(device) if isinstance(device, str) else device
        
        # Determine if we should use pinned memory
        tensor_size_mb = np.prod(shape) * torch.tensor([], dtype=dtype).element_size() / (1024 * 1024)
        use_pinned = (pin_memory or 
                     (device.type == 'cpu' and tensor_size_mb >= self.config.auto_pin_threshold_mb))
        
        if use_pinned and device.type == 'cpu':
            tensor = self.pinned_memory_pool.get_pinned_tensor(shape, dtype)
        else:
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        
        # Create wrapper
        wrapper = ZeroCopyTensorWrapper(tensor, name, track_copies=True)
        
        # Track tensor
        tensor_id = name or f"tensor_{id(tensor)}"
        with self._lock:
            self.tracked_tensors[tensor_id] = wrapper
        
        return wrapper
    
    def transfer_tensor(self, 
                       wrapper: ZeroCopyTensorWrapper,
                       target_device: Union[str, torch.device],
                       async_transfer: bool = True) -> ZeroCopyTensorWrapper:
        """
        Transfer tensor to target device with optimizations.
        
        Args:
            wrapper: Tensor wrapper to transfer
            target_device: Target device
            async_transfer: Whether to use async transfer
            
        Returns:
            Transferred tensor wrapper
        """
        target_device = torch.device(target_device) if isinstance(target_device, str) else target_device
        
        # Record operation for placement optimization
        tensor_id = wrapper.name or f"tensor_{id(wrapper.tensor)}"
        self.placement_optimizer.record_operation(tensor_id, 'transfer', target_device)
        
        # Perform transfer
        if async_transfer and self.config.enable_async_transfers:
            transferred_tensor = self.async_transfer_manager.async_transfer(
                wrapper.tensor, target_device
            )
        else:
            transferred_tensor = wrapper.tensor.to(target_device)
        
        # Return new wrapper
        return ZeroCopyTensorWrapper(transferred_tensor, wrapper.name, wrapper.track_copies)
    
    def cleanup_tensor(self, wrapper: ZeroCopyTensorWrapper) -> None:
        """Clean up tensor and return resources to pools."""
        if wrapper.is_pinned:
            self.pinned_memory_pool.return_pinned_tensor(wrapper.tensor)
        
        # Remove from tracking
        tensor_id = wrapper.name or f"tensor_{id(wrapper.tensor)}"
        with self._lock:
            if tensor_id in self.tracked_tensors:
                del self.tracked_tensors[tensor_id]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            'gpu_transfer_stats': {
                'host_to_device_transfers': self.statistics.host_to_device_transfers,
                'device_to_host_transfers': self.statistics.device_to_host_transfers,
                'async_transfers': self.async_transfer_manager.statistics.async_transfers,
                'sync_transfers': self.async_transfer_manager.statistics.sync_transfers,
                'batched_transfers': self.async_transfer_manager.statistics.batched_transfers,
                'avg_transfer_speed_gbps': self.statistics.avg_transfer_speed_gbps
            },
            'pinned_memory_stats': {
                'pinned_memory_allocations': self.pinned_memory_pool.statistics.pinned_memory_allocations,
                'pinned_memory_reuses': self.pinned_memory_pool.statistics.pinned_memory_reuses,
                'total_pinned_mb': self.pinned_memory_pool._total_pinned_mb
            },
            'placement_stats': self.placement_optimizer.get_placement_stats(),
            'tracked_tensors': len(self.tracked_tensors)
        }
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        self.pinned_memory_pool.cleanup()
        with self._lock:
            self.tracked_tensors.clear()


# Global manager instance
_gpu_manager = ZeroCopyGPUManager()


def create_optimized_tensor(shape: Union[List[int], Tuple[int, ...]],
                          dtype: torch.dtype = torch.float32,
                          device: Union[str, torch.device] = 'cpu',
                          pin_memory: bool = False,
                          name: Optional[str] = None) -> ZeroCopyTensorWrapper:
    """Create an optimized tensor with zero-copy tracking."""
    return _gpu_manager.create_tensor(shape, dtype, device, pin_memory, name)


def transfer_optimized(wrapper: ZeroCopyTensorWrapper,
                      target_device: Union[str, torch.device],
                      async_transfer: bool = True) -> ZeroCopyTensorWrapper:
    """Transfer tensor with optimizations."""
    return _gpu_manager.transfer_tensor(wrapper, target_device, async_transfer)


def cleanup_optimized_tensor(wrapper: ZeroCopyTensorWrapper) -> None:
    """Clean up optimized tensor."""
    _gpu_manager.cleanup_tensor(wrapper)


@contextmanager
def gpu_optimization_context(config: Optional[GPUTransferConfig] = None):
    """Context manager for GPU optimizations with custom configuration."""
    global _gpu_manager
    
    # Save current manager
    old_manager = _gpu_manager
    
    try:
        # Create new manager with custom config
        if config is not None:
            _gpu_manager = ZeroCopyGPUManager(config)
        
        yield _gpu_manager
        
    finally:
        # Cleanup and restore
        if config is not None:
            _gpu_manager.cleanup()
        _gpu_manager = old_manager


def get_gpu_optimization_stats() -> Dict[str, Any]:
    """Get GPU optimization statistics."""
    return _gpu_manager.get_comprehensive_stats() 