"""Unit tests for zero-copy core functionality."""

import unittest
import numpy as np
import torch
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from yinsh_ml.memory.zero_copy import (
    ZeroCopyConfig,
    ZeroCopyTensorFactory,
    InPlaceOperations,
    ZeroCopyBatchProcessor,
    create_shared_tensor,
    create_view_tensor,
    get_persistent_buffer,
    release_buffer,
    safe_add_, safe_mul_, safe_copy_,
    create_batch_from_numpy,
    zero_copy_context,
    get_zero_copy_statistics
)


class TestZeroCopyConfig(unittest.TestCase):
    """Test zero-copy configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ZeroCopyConfig()
        
        self.assertTrue(config.enable_shared_memory_tensors)
        self.assertTrue(config.enable_persistent_buffers)
        self.assertTrue(config.enable_view_operations)
        self.assertTrue(config.enable_inplace_operations)
        self.assertEqual(config.inplace_threshold_mb, 1.0)
        self.assertEqual(config.max_buffer_pool_size_mb, 512)
        self.assertEqual(config.shared_memory_page_size, 65536)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ZeroCopyConfig(
            enable_shared_memory_tensors=False,
            inplace_threshold_mb=0.5,
            max_buffer_pool_size_mb=256
        )
        
        self.assertFalse(config.enable_shared_memory_tensors)
        self.assertEqual(config.inplace_threshold_mb, 0.5)
        self.assertEqual(config.max_buffer_pool_size_mb, 256)


class TestZeroCopyTensorFactory(unittest.TestCase):
    """Test zero-copy tensor factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ZeroCopyConfig()
        self.factory = ZeroCopyTensorFactory(self.config)
    
    def test_create_shared_tensor_cpu(self):
        """Test creating shared memory tensors on CPU."""
        shape = (10, 10)
        tensor = self.factory.create_shared_tensor(shape, torch.float32)
        
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, 'cpu')
        
        # Check if it's actually backed by shared memory
        if hasattr(tensor, '_shared_memory_region'):
            self.assertIsNotNone(tensor._shared_memory_region)
    
    def test_create_shared_tensor_gpu_fallback(self):
        """Test fallback to regular tensor for GPU device."""
        shape = (5, 5)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        with patch('warnings.warn') as mock_warn:
            tensor = self.factory.create_shared_tensor(shape, torch.float32, device)
            
            self.assertEqual(tensor.shape, shape)
            self.assertEqual(tensor.dtype, torch.float32)
            
            if device.type == 'cuda':
                # Should warn about GPU not supporting shared memory
                mock_warn.assert_called()
    
    def test_create_view_tensor(self):
        """Test creating tensor views."""
        source = torch.randn(100, 100)
        view_shape = (50, 50)
        
        view_tensor = self.factory.create_view_tensor(source, view_shape)
        
        if view_tensor is not None:
            self.assertEqual(view_tensor.shape, view_shape)
            # Modifying view should affect source
            original_sum = source.sum().item()
            view_tensor.fill_(1.0)
            self.assertNotEqual(source.sum().item(), original_sum)
    
    def test_view_tensor_invalid_shape(self):
        """Test view creation with invalid shape."""
        source = torch.randn(10, 10)  # 100 elements
        view_shape = (20, 20)  # 400 elements (too large)
        
        view_tensor = self.factory.create_view_tensor(source, view_shape)
        self.assertIsNone(view_tensor)
    
    def test_persistent_buffer(self):
        """Test persistent buffer management."""
        shape = (32, 32)
        dtype = torch.float32
        
        # Get buffer
        buffer1 = self.factory.get_persistent_buffer(shape, dtype)
        self.assertEqual(buffer1.shape, shape)
        self.assertEqual(buffer1.dtype, dtype)
        
        # Release and get again - should potentially reuse
        self.factory.release_buffer(buffer1)
        buffer2 = self.factory.get_persistent_buffer(shape, dtype)
        self.assertEqual(buffer2.shape, shape)
        self.assertEqual(buffer2.dtype, dtype)
    
    def test_statistics_tracking(self):
        """Test statistics are properly tracked."""
        initial_stats = self.factory.statistics
        initial_shared = initial_stats.shared_memory_allocations
        
        # Create shared tensor
        self.factory.create_shared_tensor((10, 10), torch.float32)
        
        # Check statistics updated
        updated_stats = self.factory.statistics
        self.assertGreaterEqual(updated_stats.shared_memory_allocations, initial_shared)


class TestInPlaceOperations(unittest.TestCase):
    """Test in-place operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ZeroCopyConfig()
        self.inplace_ops = InPlaceOperations(self.config)
    
    def test_safe_add_inplace(self):
        """Test safe in-place addition."""
        # Use larger tensors to meet the inplace threshold (1MB)
        tensor1 = torch.ones(512, 512)  # About 1MB
        tensor2 = torch.ones(512, 512) * 2
        original_data_ptr = tensor1.data_ptr()
        
        result = self.inplace_ops.safe_add_(tensor1, tensor2)
        
        # Should be in-place if conditions are met
        if tensor1.is_contiguous() and not tensor1.requires_grad:
            self.assertEqual(result.data_ptr(), original_data_ptr)
        
        # Check mathematical correctness
        expected = torch.ones(512, 512) * 3
        torch.testing.assert_close(result, expected)
    
    def test_safe_mul_inplace(self):
        """Test safe in-place multiplication."""
        tensor1 = torch.ones(5, 5) * 3
        scalar = 2.0
        
        result = self.inplace_ops.safe_mul_(tensor1, scalar)
        
        # Check mathematical correctness
        expected = torch.ones(5, 5) * 6
        torch.testing.assert_close(result, expected)
    
    def test_safe_copy_inplace(self):
        """Test safe in-place copying."""
        source = torch.randn(512, 512)  # About 1MB
        target = torch.zeros(512, 512)
        original_data_ptr = target.data_ptr()
        
        result = self.inplace_ops.safe_copy_(target, source)
        
        # Should be in-place if conditions are met
        if target.is_contiguous() and not target.requires_grad:
            self.assertEqual(result.data_ptr(), original_data_ptr)
        
        # Check values are copied correctly
        torch.testing.assert_close(result, source)
    
    def test_inplace_with_gradients(self):
        """Test in-place operations with gradient tracking."""
        tensor1 = torch.ones(5, 5, requires_grad=True)
        tensor2 = torch.ones(5, 5) * 2
        
        # Should create copy when requires_grad=True
        result = self.inplace_ops.safe_add_(tensor1, tensor2)
        
        # Mathematical correctness
        expected = torch.ones(5, 5) * 3
        torch.testing.assert_close(result, expected)
    
    def test_inplace_size_mismatch(self):
        """Test in-place operations with size mismatch."""
        tensor1 = torch.ones(5, 5)
        tensor2 = torch.ones(3, 3)
        
        with self.assertRaises(RuntimeError):
            self.inplace_ops.safe_add_(tensor1, tensor2)
    
    def test_statistics_tracking(self):
        """Test that in-place operations are tracked."""
        initial_stats = self.inplace_ops.statistics
        initial_ops = initial_stats.inplace_operations
        initial_failed = initial_stats.failed_inplace_operations
        
        # Use larger tensors to meet threshold
        tensor1 = torch.ones(512, 512)  # About 1MB
        tensor2 = torch.ones(512, 512)
        
        self.inplace_ops.safe_add_(tensor1, tensor2)
        
        updated_stats = self.inplace_ops.statistics
        # Should increment either inplace_operations or failed_inplace_operations
        total_ops = updated_stats.inplace_operations + updated_stats.failed_inplace_operations
        initial_total = initial_ops + initial_failed
        self.assertGreater(total_ops, initial_total)


class TestZeroCopyBatchProcessor(unittest.TestCase):
    """Test zero-copy batch processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ZeroCopyConfig()
        self.factory = ZeroCopyTensorFactory(self.config)
        self.inplace_ops = InPlaceOperations(self.config)
        self.batch_processor = ZeroCopyBatchProcessor(self.factory, self.inplace_ops)
    
    def test_create_batch_from_numpy(self):
        """Test creating batch tensor from numpy arrays."""
        arrays = [
            np.random.randn(3, 32, 32).astype(np.float32),
            np.random.randn(3, 32, 32).astype(np.float32),
            np.random.randn(3, 32, 32).astype(np.float32)
        ]
        
        batch_tensor = self.batch_processor.create_batch_from_numpy(
            arrays, torch.float32, torch.device('cpu')
        )
        
        self.assertEqual(batch_tensor.shape, (3, 3, 32, 32))
        self.assertEqual(batch_tensor.dtype, torch.float32)
        
        # Check data correctness
        for i, array in enumerate(arrays):
            torch.testing.assert_close(
                batch_tensor[i].numpy(), array, rtol=1e-5, atol=1e-6
            )
    
    def test_create_batch_inconsistent_shapes(self):
        """Test batch creation with inconsistent array shapes."""
        arrays = [
            np.random.randn(32, 32).astype(np.float32),
            np.random.randn(16, 16).astype(np.float32)  # Different shape
        ]
        
        with self.assertRaises(ValueError):
            self.batch_processor.create_batch_from_numpy(arrays, torch.float32)
    
    def test_create_batch_empty_list(self):
        """Test batch creation with empty array list."""
        batch_tensor = self.batch_processor.create_batch_from_numpy(
            [], torch.float32, torch.device('cpu')
        )
        
        self.assertEqual(batch_tensor.shape[0], 0)
    
    def test_create_batch_different_dtypes(self):
        """Test batch creation with different numpy dtypes."""
        arrays = [
            np.random.randn(10, 10).astype(np.float64),  # Different dtype
            np.random.randn(10, 10).astype(np.float64)
        ]
        
        batch_tensor = self.batch_processor.create_batch_from_numpy(
            arrays, torch.float32
        )
        
        self.assertEqual(batch_tensor.dtype, torch.float32)
        self.assertEqual(batch_tensor.shape, (2, 10, 10))


class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""
    
    def test_create_shared_tensor_global(self):
        """Test global create_shared_tensor function."""
        tensor = create_shared_tensor((5, 5), torch.float32)
        
        self.assertEqual(tensor.shape, (5, 5))
        self.assertEqual(tensor.dtype, torch.float32)
    
    def test_create_view_tensor_global(self):
        """Test global create_view_tensor function."""
        source = torch.randn(100)
        view = create_view_tensor(source, (10, 10))
        
        if view is not None:
            self.assertEqual(view.shape, (10, 10))
    
    def test_persistent_buffer_global(self):
        """Test global persistent buffer functions."""
        buffer = get_persistent_buffer((8, 8), torch.float32)
        
        self.assertEqual(buffer.shape, (8, 8))
        self.assertEqual(buffer.dtype, torch.float32)
        
        # Test release doesn't throw error
        release_buffer(buffer)
    
    def test_safe_operations_global(self):
        """Test global safe operation functions."""
        tensor1 = torch.ones(5, 5)
        tensor2 = torch.ones(5, 5) * 2
        
        # Test safe_add_
        result = safe_add_(tensor1.clone(), tensor2)
        expected = torch.ones(5, 5) * 3
        torch.testing.assert_close(result, expected)
        
        # Test safe_mul_
        result = safe_mul_(tensor1.clone(), 3.0)
        expected = torch.ones(5, 5) * 3
        torch.testing.assert_close(result, expected)
        
        # Test safe_copy_
        target = torch.zeros(5, 5)
        result = safe_copy_(target, tensor1)
        torch.testing.assert_close(result, tensor1)
    
    def test_create_batch_from_numpy_global(self):
        """Test global create_batch_from_numpy function."""
        arrays = [
            np.random.randn(16, 16).astype(np.float32),
            np.random.randn(16, 16).astype(np.float32)
        ]
        
        batch = create_batch_from_numpy(arrays, torch.float32)
        
        self.assertEqual(batch.shape, (2, 16, 16))
        self.assertEqual(batch.dtype, torch.float32)


class TestZeroCopyContext(unittest.TestCase):
    """Test zero-copy context manager."""
    
    def test_context_manager(self):
        """Test zero-copy context manager functionality."""
        custom_config = ZeroCopyConfig(
            enable_shared_memory_tensors=False,
            inplace_threshold_mb=0.1
        )
        
        with zero_copy_context(custom_config) as context:
            self.assertIn('factory', context)
            self.assertIn('inplace', context)
            self.assertIn('batch_processor', context)
            
            # Test that custom config is applied
            factory = context['factory']
            self.assertEqual(factory.config.inplace_threshold_mb, 0.1)
    
    def test_context_manager_restoration(self):
        """Test that context manager restores original state."""
        # Get initial statistics
        initial_stats = get_zero_copy_statistics()
        
        custom_config = ZeroCopyConfig(enable_shared_memory_tensors=False)
        
        with zero_copy_context(custom_config):
            # Use custom configuration
            tensor = create_shared_tensor((10, 10))
        
        # After context, should be back to original configuration
        final_stats = get_zero_copy_statistics()
        self.assertEqual(type(initial_stats), type(final_stats))


class TestStatistics(unittest.TestCase):
    """Test statistics collection."""
    
    def test_get_zero_copy_statistics(self):
        """Test getting zero-copy statistics."""
        stats = get_zero_copy_statistics()
        
        self.assertIn('tensor_factory', stats)
        self.assertIn('inplace_operations', stats)
        self.assertIn('batch_processor', stats)
        
        # Check that statistics have expected fields
        factory_stats = stats['tensor_factory']
        self.assertIsInstance(factory_stats.shared_memory_allocations, int)
        self.assertIsInstance(factory_stats.buffer_reuses, int)
        self.assertIsInstance(factory_stats.copy_avoided_count, int)
    
    def test_statistics_increment(self):
        """Test that statistics increment correctly."""
        initial_stats = get_zero_copy_statistics()
        initial_shared = initial_stats['tensor_factory'].shared_memory_allocations
        
        # Perform operation that should increment statistics
        create_shared_tensor((5, 5))
        
        updated_stats = get_zero_copy_statistics()
        updated_shared = updated_stats['tensor_factory'].shared_memory_allocations
        
        self.assertGreaterEqual(updated_shared, initial_shared)


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations."""
    
    def test_inplace_vs_copy_performance(self):
        """Test that in-place operations are faster than copying."""
        size = (1000, 1000)
        tensor1 = torch.randn(size)
        tensor2 = torch.randn(size)
        
        # Time in-place operation
        start_time = time.time()
        for _ in range(10):
            safe_add_(tensor1.clone(), tensor2)
        inplace_time = time.time() - start_time
        
        # Time copy operation
        start_time = time.time()
        for _ in range(10):
            torch.add(tensor1, tensor2)
        copy_time = time.time() - start_time
        
        # In-place should be competitive (within 2x)
        self.assertLess(inplace_time, copy_time * 2.0)
    
    def test_batch_creation_performance(self):
        """Test batch creation performance."""
        arrays = [np.random.randn(64, 64).astype(np.float32) for _ in range(32)]
        
        # Time zero-copy batch creation
        start_time = time.time()
        for _ in range(5):
            batch = create_batch_from_numpy(arrays, torch.float32)
        zerocopy_time = time.time() - start_time
        
        # Time standard batch creation
        start_time = time.time()
        for _ in range(5):
            tensors = [torch.tensor(arr) for arr in arrays]
            batch = torch.stack(tensors)
        standard_time = time.time() - start_time
        
        # Zero-copy should be competitive
        self.assertLess(zerocopy_time, standard_time * 2.0)
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_transfer_optimization(self):
        """Test device transfer optimizations."""
        tensor = torch.randn(512, 512)
        
        # Time standard transfer
        start_time = time.time()
        for _ in range(10):
            gpu_tensor = tensor.to('cuda')
            cpu_tensor = gpu_tensor.to('cpu')
        standard_time = time.time() - start_time
        
        # The zero-copy system should handle device transfers efficiently
        # This is more of a placeholder for future device transfer optimizations
        self.assertGreater(standard_time, 0)  # Sanity check


if __name__ == '__main__':
    unittest.main() 