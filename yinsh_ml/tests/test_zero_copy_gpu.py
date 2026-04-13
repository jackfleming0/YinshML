"""Tests for zero-copy GPU utilities and optimizations."""

import unittest
import numpy as np
import torch
import time
import threading
from unittest.mock import patch, MagicMock

from yinsh_ml.memory.zero_copy_gpu import (
    GPUTransferConfig,
    GPUTransferStatistics,
    ZeroCopyTensorWrapper,
    PinnedMemoryPool,
    AsyncTransferManager,
    SmartPlacementOptimizer,
    ZeroCopyGPUManager,
    create_optimized_tensor,
    transfer_optimized,
    cleanup_optimized_tensor,
    gpu_optimization_context,
    get_gpu_optimization_stats
)


class TestGPUTransferConfig(unittest.TestCase):
    """Test GPU transfer configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GPUTransferConfig()
        
        self.assertTrue(config.enable_pinned_memory)
        self.assertTrue(config.enable_async_transfers)
        self.assertTrue(config.enable_batch_transfers)
        self.assertTrue(config.enable_smart_placement)
        self.assertEqual(config.pinned_memory_pool_size_mb, 256)
        self.assertEqual(config.transfer_streams, 2)
        self.assertEqual(config.batch_size_threshold, 8)
        self.assertEqual(config.placement_history_size, 1000)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPUTransferConfig(
            enable_pinned_memory=False,
            pinned_memory_pool_size_mb=128,
            transfer_streams=4,
            batch_size_threshold=16
        )
        
        self.assertFalse(config.enable_pinned_memory)
        self.assertEqual(config.pinned_memory_pool_size_mb, 128)
        self.assertEqual(config.transfer_streams, 4)
        self.assertEqual(config.batch_size_threshold, 16)


class TestZeroCopyTensorWrapper(unittest.TestCase):
    """Test tensor wrapper with zero-copy tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tensor = torch.randn(10, 10)
        self.wrapper = ZeroCopyTensorWrapper(self.tensor, "test_tensor", track_copies=True)
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.name, "test_tensor")
        self.assertTrue(self.wrapper.track_copies)
        self.assertEqual(self.wrapper.copy_count, 0)
        self.assertEqual(self.wrapper.transfer_count, 0)
        self.assertEqual(self.wrapper.last_device, self.tensor.device)
    
    def test_tensor_access(self):
        """Test tensor access tracking."""
        initial_access_count = len(self.wrapper.access_pattern)
        
        # Access tensor
        _ = self.wrapper.tensor
        
        # Should track access
        self.assertGreater(len(self.wrapper.access_pattern), initial_access_count)
    
    def test_clone_tracking(self):
        """Test clone operation tracking."""
        initial_copy_count = self.wrapper.copy_count
        
        cloned_wrapper = self.wrapper.clone()
        
        # Should track copy
        self.assertEqual(self.wrapper.copy_count, initial_copy_count + 1)
        self.assertIsNotNone(self.wrapper.last_copy_time)
        
        # Cloned wrapper should be separate
        self.assertEqual(cloned_wrapper.name, "test_tensor_clone")
        self.assertEqual(cloned_wrapper.copy_count, 0)  # Fresh wrapper
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_transfer_tracking(self):
        """Test device transfer tracking."""
        initial_transfer_count = self.wrapper.transfer_count
        
        # Transfer to GPU
        gpu_wrapper = self.wrapper.to('cuda')
        
        # Should track transfer
        self.assertEqual(self.wrapper.transfer_count, initial_transfer_count + 1)
        self.assertEqual(gpu_wrapper.name, self.wrapper.name)
        self.assertEqual(str(gpu_wrapper.last_device), 'cuda:0')
    
    def test_pinned_memory_detection(self):
        """Test pinned memory detection."""
        # CPU tensor
        cpu_tensor = torch.randn(5, 5)
        cpu_wrapper = ZeroCopyTensorWrapper(cpu_tensor)
        self.assertFalse(cpu_wrapper.is_pinned)
        
        # Pinned tensor
        pinned_tensor = torch.randn(5, 5).pin_memory()
        pinned_wrapper = ZeroCopyTensorWrapper(pinned_tensor)
        self.assertTrue(pinned_wrapper.is_pinned)
    
    def test_copy_statistics(self):
        """Test copy statistics collection."""
        # Perform operations
        self.wrapper.clone()
        _ = self.wrapper.tensor
        
        stats = self.wrapper.get_copy_stats()
        
        self.assertEqual(stats['name'], 'test_tensor')
        self.assertGreaterEqual(stats['copy_count'], 1)
        self.assertGreaterEqual(stats['access_count'], 1)
        self.assertIn('current_device', stats)


class TestPinnedMemoryPool(unittest.TestCase):
    """Test pinned memory pool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPUTransferConfig(
            enable_pinned_memory=True,
            pinned_memory_pool_size_mb=64
        )
        self.pool = PinnedMemoryPool(self.config)
    
    def test_get_pinned_tensor(self):
        """Test getting pinned tensors from pool."""
        shape = (32, 32)
        dtype = torch.float32
        
        tensor = self.pool.get_pinned_tensor(shape, dtype)
        
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, dtype)
        
        # Should be pinned if enabled
        if self.config.enable_pinned_memory:
            self.assertTrue(tensor.is_pinned())
    
    def test_tensor_reuse(self):
        """Test tensor reuse from pool."""
        shape = (16, 16)
        dtype = torch.float32
        
        # Get tensor
        tensor1 = self.pool.get_pinned_tensor(shape, dtype)
        
        # Return to pool
        self.pool.return_pinned_tensor(tensor1)
        
        # Get again - should potentially reuse
        tensor2 = self.pool.get_pinned_tensor(shape, dtype)
        
        self.assertEqual(tensor2.shape, shape)
        self.assertEqual(tensor2.dtype, dtype)
    
    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        # Request large tensor that exceeds pool limit
        large_shape = (2048, 2048)  # ~16MB for float32
        
        tensor = self.pool.get_pinned_tensor(large_shape, torch.float32)
        
        # Should still get a tensor (may not be pinned if limit exceeded)
        self.assertEqual(tensor.shape, large_shape)
        self.assertEqual(tensor.dtype, torch.float32)
    
    def test_disabled_pinned_memory(self):
        """Test behavior when pinned memory is disabled."""
        disabled_config = GPUTransferConfig(enable_pinned_memory=False)
        disabled_pool = PinnedMemoryPool(disabled_config)
        
        tensor = disabled_pool.get_pinned_tensor((10, 10), torch.float32)
        
        self.assertEqual(tensor.shape, (10, 10))
        self.assertFalse(tensor.is_pinned())
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        initial_allocations = self.pool.statistics.pinned_memory_allocations
        
        # Get pinned tensor
        self.pool.get_pinned_tensor((8, 8), torch.float32)
        
        # Should track allocation
        self.assertGreaterEqual(
            self.pool.statistics.pinned_memory_allocations, 
            initial_allocations
        )
    
    def test_pool_cleanup(self):
        """Test pool cleanup."""
        # Get some tensors
        for _ in range(3):
            tensor = self.pool.get_pinned_tensor((4, 4), torch.float32)
            self.pool.return_pinned_tensor(tensor)
        
        # Cleanup
        self.pool.cleanup()
        
        # Pool should be empty
        self.assertEqual(len(self.pool._pinned_tensors), 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestAsyncTransferManager(unittest.TestCase):
    """Test async transfer manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPUTransferConfig(
            enable_async_transfers=True,
            transfer_streams=2
        )
        self.manager = AsyncTransferManager(self.config)
    
    def test_async_transfer(self):
        """Test asynchronous tensor transfer."""
        tensor = torch.randn(64, 64)
        
        # Transfer to GPU
        gpu_tensor = self.manager.async_transfer(tensor, 'cuda')
        
        self.assertEqual(gpu_tensor.device.type, 'cuda')
        self.assertEqual(gpu_tensor.shape, tensor.shape)
        torch.testing.assert_close(gpu_tensor.cpu(), tensor, rtol=1e-5, atol=1e-6)
    
    def test_batch_transfer(self):
        """Test batch tensor transfers."""
        tensors = [torch.randn(32, 32) for _ in range(8)]
        
        # Batch transfer to GPU
        gpu_tensors = self.manager.batch_transfer(tensors, 'cuda')
        
        self.assertEqual(len(gpu_tensors), len(tensors))
        
        for i, (cpu_tensor, gpu_tensor) in enumerate(zip(tensors, gpu_tensors)):
            self.assertEqual(gpu_tensor.device.type, 'cuda')
            self.assertEqual(gpu_tensor.shape, cpu_tensor.shape)
            torch.testing.assert_close(gpu_tensor.cpu(), cpu_tensor, rtol=1e-5, atol=1e-6)
    
    def test_same_device_transfer(self):
        """Test transfer when source and target devices are the same."""
        tensor = torch.randn(16, 16, device='cuda')
        
        # Transfer to same device
        result = self.manager.async_transfer(tensor, 'cuda')
        
        # Should be the same tensor or a no-op
        self.assertEqual(result.device.type, 'cuda')
        self.assertEqual(result.shape, tensor.shape)
    
    def test_disabled_async_transfers(self):
        """Test behavior when async transfers are disabled."""
        disabled_config = GPUTransferConfig(enable_async_transfers=False)
        disabled_manager = AsyncTransferManager(disabled_config)
        
        tensor = torch.randn(32, 32)
        gpu_tensor = disabled_manager.async_transfer(tensor, 'cuda')
        
        # Should still work, but use synchronous transfer
        self.assertEqual(gpu_tensor.device.type, 'cuda')
        torch.testing.assert_close(gpu_tensor.cpu(), tensor, rtol=1e-5, atol=1e-6)
    
    def test_statistics_tracking(self):
        """Test transfer statistics tracking."""
        initial_transfers = self.manager.statistics.host_to_device_transfers
        
        tensor = torch.randn(16, 16)
        self.manager.async_transfer(tensor, 'cuda')
        
        # Should track transfer
        self.assertGreater(
            self.manager.statistics.host_to_device_transfers, 
            initial_transfers
        )
    
    def test_callback_functionality(self):
        """Test callback execution on transfer completion."""
        tensor = torch.randn(32, 32)
        callback_called = threading.Event()
        result_tensor = None
        
        def transfer_callback(transferred_tensor):
            nonlocal result_tensor
            result_tensor = transferred_tensor
            callback_called.set()
        
        # Transfer with callback
        gpu_tensor = self.manager.async_transfer(tensor, 'cuda', callback=transfer_callback)
        
        # Wait for callback (with timeout)
        callback_called.wait(timeout=5.0)
        
        # Callback should have been called
        self.assertTrue(callback_called.is_set())
        self.assertIsNotNone(result_tensor)


class TestSmartPlacementOptimizer(unittest.TestCase):
    """Test smart placement optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPUTransferConfig(
            enable_smart_placement=True,
            placement_history_size=100
        )
        self.optimizer = SmartPlacementOptimizer(self.config)
    
    def test_operation_recording(self):
        """Test operation recording."""
        tensor_id = "test_tensor"
        device = torch.device('cpu')
        
        initial_history_size = len(self.optimizer.operation_history)
        
        self.optimizer.record_operation(tensor_id, "compute", device)
        
        # Should record operation
        self.assertGreater(len(self.optimizer.operation_history), initial_history_size)
        
        # Check usage stats
        usage = self.optimizer.tensor_usage[tensor_id]
        self.assertGreater(usage['cpu_ops'], 0)
    
    def test_placement_suggestion(self):
        """Test placement suggestions."""
        tensor_id = "test_tensor"
        current_device = torch.device('cpu')
        
        # Record some CPU operations
        for _ in range(5):
            self.optimizer.record_operation(tensor_id, "compute", torch.device('cpu'))
        
        # Get placement suggestion
        suggested_device = self.optimizer.suggest_placement(tensor_id, current_device)
        
        # Should suggest a valid device
        self.assertIsInstance(suggested_device, torch.device)
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_placement_preference(self):
        """Test GPU placement preference."""
        tensor_id = "gpu_tensor"
        current_device = torch.device('cpu')
        
        # Record many GPU operations
        for _ in range(10):
            self.optimizer.record_operation(tensor_id, "compute", torch.device('cuda'))
        
        suggested_device = self.optimizer.suggest_placement(tensor_id, current_device)
        
        # Should prefer GPU if available and more GPU operations
        if torch.cuda.is_available():
            self.assertEqual(suggested_device.type, 'cuda')
    
    def test_disabled_smart_placement(self):
        """Test behavior when smart placement is disabled."""
        disabled_config = GPUTransferConfig(enable_smart_placement=False)
        disabled_optimizer = SmartPlacementOptimizer(disabled_config)
        
        tensor_id = "test_tensor"
        current_device = torch.device('cpu')
        
        suggested_device = disabled_optimizer.suggest_placement(tensor_id, current_device)
        
        # Should return current device when disabled
        self.assertEqual(suggested_device, current_device)
    
    def test_placement_statistics(self):
        """Test placement statistics collection."""
        tensor_id = "test_tensor"
        
        # Record operations and get suggestions
        for i in range(10):
            device = torch.device('cpu') if i % 2 == 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.optimizer.record_operation(tensor_id, "compute", device)
            self.optimizer.suggest_placement(tensor_id, device)
        
        stats = self.optimizer.get_placement_stats()
        
        self.assertIn('tracked_tensors', stats)
        self.assertIn('optimal_placements', stats)
        self.assertIn('suboptimal_placements', stats)
        self.assertGreaterEqual(stats['tracked_tensors'], 1)


class TestZeroCopyGPUManager(unittest.TestCase):
    """Test comprehensive GPU manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPUTransferConfig(
            pinned_memory_pool_size_mb=32,
            transfer_streams=1
        )
        self.manager = ZeroCopyGPUManager(self.config)
    
    def test_create_tensor(self):
        """Test optimized tensor creation."""
        shape = (16, 16)
        dtype = torch.float32
        device = 'cpu'
        
        wrapper = self.manager.create_tensor(shape, dtype, device, name="test")
        
        self.assertEqual(wrapper.tensor.shape, shape)
        self.assertEqual(wrapper.tensor.dtype, dtype)
        self.assertEqual(wrapper.tensor.device.type, device)
        self.assertEqual(wrapper.name, "test")
    
    def test_create_pinned_tensor(self):
        """Test pinned tensor creation."""
        shape = (8, 8)
        
        wrapper = self.manager.create_tensor(shape, pin_memory=True, device='cpu')
        
        if self.config.enable_pinned_memory:
            self.assertTrue(wrapper.tensor.is_pinned())
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_transfer_tensor(self):
        """Test optimized tensor transfer."""
        wrapper = self.manager.create_tensor((16, 16), device='cpu', name="transfer_test")
        
        # Transfer to GPU
        gpu_wrapper = self.manager.transfer_tensor(wrapper, 'cuda')
        
        self.assertEqual(gpu_wrapper.tensor.device.type, 'cuda')
        self.assertEqual(gpu_wrapper.name, wrapper.name)
    
    def test_tensor_tracking(self):
        """Test tensor tracking."""
        initial_count = len(self.manager.tracked_tensors)
        
        wrapper = self.manager.create_tensor((8, 8), name="tracked")
        
        # Should track tensor
        self.assertEqual(len(self.manager.tracked_tensors), initial_count + 1)
        self.assertIn("tracked", self.manager.tracked_tensors)
    
    def test_cleanup_tensor(self):
        """Test tensor cleanup."""
        wrapper = self.manager.create_tensor((8, 8), pin_memory=True, name="cleanup_test")
        
        initial_count = len(self.manager.tracked_tensors)
        
        self.manager.cleanup_tensor(wrapper)
        
        # Should remove from tracking
        self.assertEqual(len(self.manager.tracked_tensors), initial_count - 1)
        self.assertNotIn("cleanup_test", self.manager.tracked_tensors)
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics collection."""
        # Perform various operations
        wrapper = self.manager.create_tensor((16, 16), pin_memory=True)
        
        if torch.cuda.is_available():
            self.manager.transfer_tensor(wrapper, 'cuda')
        
        stats = self.manager.get_comprehensive_stats()
        
        self.assertIn('gpu_transfer_stats', stats)
        self.assertIn('pinned_memory_stats', stats)
        self.assertIn('placement_stats', stats)
        self.assertIn('tracked_tensors', stats)
    
    def test_manager_cleanup(self):
        """Test manager cleanup."""
        # Create some tensors
        for i in range(3):
            self.manager.create_tensor((4, 4), name=f"test_{i}")
        
        initial_count = len(self.manager.tracked_tensors)
        self.assertGreater(initial_count, 0)
        
        # Cleanup
        self.manager.cleanup()
        
        # Should clean up tracking
        self.assertEqual(len(self.manager.tracked_tensors), 0)


class TestGlobalFunctions(unittest.TestCase):
    """Test global GPU optimization functions."""
    
    def test_create_optimized_tensor_global(self):
        """Test global create_optimized_tensor function."""
        wrapper = create_optimized_tensor((8, 8), dtype=torch.float32, device='cpu')
        
        self.assertEqual(wrapper.tensor.shape, (8, 8))
        self.assertEqual(wrapper.tensor.dtype, torch.float32)
        self.assertEqual(wrapper.tensor.device.type, 'cpu')
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_transfer_optimized_global(self):
        """Test global transfer_optimized function."""
        wrapper = create_optimized_tensor((16, 16), device='cpu')
        
        gpu_wrapper = transfer_optimized(wrapper, 'cuda')
        
        self.assertEqual(gpu_wrapper.tensor.device.type, 'cuda')
    
    def test_cleanup_optimized_tensor_global(self):
        """Test global cleanup_optimized_tensor function."""
        wrapper = create_optimized_tensor((8, 8), pin_memory=True)
        
        # Should not raise exception
        cleanup_optimized_tensor(wrapper)
    
    def test_get_gpu_optimization_stats_global(self):
        """Test global get_gpu_optimization_stats function."""
        # Perform some operations
        wrapper = create_optimized_tensor((16, 16))
        
        stats = get_gpu_optimization_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('gpu_transfer_stats', stats)


class TestGPUOptimizationContext(unittest.TestCase):
    """Test GPU optimization context manager."""
    
    def test_context_manager(self):
        """Test GPU optimization context manager."""
        custom_config = GPUTransferConfig(
            pinned_memory_pool_size_mb=16,
            transfer_streams=1
        )
        
        with gpu_optimization_context(custom_config) as manager:
            self.assertIsInstance(manager, ZeroCopyGPUManager)
            self.assertEqual(manager.config.pinned_memory_pool_size_mb, 16)
            
            # Test creating tensor within context
            wrapper = manager.create_tensor((8, 8))
            self.assertEqual(wrapper.tensor.shape, (8, 8))
    
    def test_context_manager_cleanup(self):
        """Test that context manager cleans up properly."""
        custom_config = GPUTransferConfig(pinned_memory_pool_size_mb=16)
        
        with gpu_optimization_context(custom_config) as manager:
            wrapper = manager.create_tensor((8, 8), name="context_test")
            initial_count = len(manager.tracked_tensors)
            self.assertGreater(initial_count, 0)
        
        # After context, manager should be cleaned up
        # (We can't easily test this directly due to global state)
        
    def test_context_manager_default_config(self):
        """Test context manager with default configuration."""
        with gpu_optimization_context() as manager:
            # Should use existing global manager
            self.assertIsInstance(manager, ZeroCopyGPUManager)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for GPU optimizations."""
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_pinned_vs_unpinned_transfer_performance(self):
        """Benchmark pinned vs unpinned memory transfer performance."""
        size = (512, 512)
        
        # Create tensors
        regular_tensor = torch.randn(size)
        pinned_tensor = torch.randn(size).pin_memory()
        
        # Warm up GPU
        for _ in range(3):
            regular_tensor.to('cuda')
            pinned_tensor.to('cuda')
        
        # Benchmark regular transfer
        start_time = time.time()
        for _ in range(10):
            gpu_tensor = regular_tensor.to('cuda')
            cpu_tensor = gpu_tensor.to('cpu')
        regular_time = time.time() - start_time
        
        # Benchmark pinned transfer
        start_time = time.time()
        for _ in range(10):
            gpu_tensor = pinned_tensor.to('cuda', non_blocking=True)
            cpu_tensor = gpu_tensor.to('cpu')
        pinned_time = time.time() - start_time
        
        print(f"\nMemory Transfer Performance:")
        print(f"Regular memory: {regular_time:.4f}s")
        print(f"Pinned memory: {pinned_time:.4f}s")
        print(f"Speedup: {regular_time/pinned_time:.2f}x")
        
        # Pinned memory should be faster or competitive
        self.assertLess(pinned_time, regular_time * 1.5)
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_vs_individual_transfer_performance(self):
        """Benchmark batch vs individual transfer performance."""
        tensors = [torch.randn(64, 64) for _ in range(20)]
        
        config = GPUTransferConfig(enable_batch_transfers=True)
        manager = AsyncTransferManager(config)
        
        # Warm up
        for tensor in tensors[:3]:
            tensor.to('cuda')
        
        # Benchmark individual transfers
        start_time = time.time()
        for tensor in tensors:
            gpu_tensor = tensor.to('cuda')
        individual_time = time.time() - start_time
        
        # Benchmark batch transfers
        start_time = time.time()
        gpu_tensors = manager.batch_transfer(tensors, 'cuda')
        batch_time = time.time() - start_time
        
        print(f"\nBatch Transfer Performance:")
        print(f"Individual transfers: {individual_time:.4f}s")
        print(f"Batch transfers: {batch_time:.4f}s")
        print(f"Speedup: {individual_time/batch_time:.2f}x")
        
        # Batch transfers should be competitive
        self.assertLess(batch_time, individual_time * 2.0)


if __name__ == '__main__':
    unittest.main(verbosity=2) 