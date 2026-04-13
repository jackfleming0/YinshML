"""Tests for TensorPool implementation."""

import pytest
import torch
import threading
import time
from unittest.mock import patch

from yinsh_ml.memory.tensor_pool import (
    TensorPool,
    TensorPoolConfig,
    TensorPoolStatistics,
    TensorKey,
    create_tensor,
    reset_tensor,
    validate_tensor_reset,
    create_tensor_pool
)


class TestTensorKey:
    """Test TensorKey functionality."""
    
    def test_tensor_key_creation(self):
        """Test TensorKey creation and properties."""
        shape = (2, 3, 4)
        dtype = torch.float32
        device = torch.device('cpu')
        
        key = TensorKey(shape, dtype, device)
        
        assert key.shape == shape
        assert key.dtype == dtype
        assert key.device == device
    
    def test_tensor_key_hashing(self):
        """Test TensorKey hashing for use in dictionaries."""
        key1 = TensorKey((2, 3), torch.float32, torch.device('cpu'))
        key2 = TensorKey((2, 3), torch.float32, torch.device('cpu'))
        key3 = TensorKey((2, 4), torch.float32, torch.device('cpu'))
        
        # Same keys should have same hash
        assert hash(key1) == hash(key2)
        
        # Different keys should have different hashes (usually)
        assert hash(key1) != hash(key3)
        
        # Keys should work in sets and dicts
        key_set = {key1, key2, key3}
        assert len(key_set) == 2  # key1 and key2 are the same
    
    def test_tensor_key_equality(self):
        """Test TensorKey equality comparison."""
        key1 = TensorKey((2, 3), torch.float32, torch.device('cpu'))
        key2 = TensorKey((2, 3), torch.float32, torch.device('cpu'))
        key3 = TensorKey((2, 3), torch.float16, torch.device('cpu'))
        
        assert key1 == key2
        assert key1 != key3
        assert key1 != "not a key"
    
    def test_tensor_key_size_bytes(self):
        """Test size calculation for tensor keys."""
        # float32 tensors: 4 bytes per element
        key_float32 = TensorKey((2, 3), torch.float32, torch.device('cpu'))
        assert key_float32.size_bytes == 2 * 3 * 4  # 24 bytes
        
        # float16 tensors: 2 bytes per element
        key_float16 = TensorKey((10, 10), torch.float16, torch.device('cpu'))
        assert key_float16.size_bytes == 10 * 10 * 2  # 200 bytes


class TestTensorFactoryFunctions:
    """Test tensor factory and utility functions."""
    
    def test_create_tensor(self):
        """Test tensor creation function."""
        shape = (3, 4, 5)
        dtype = torch.float32
        device = torch.device('cpu')
        
        tensor = create_tensor(shape, dtype, device)
        
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device == device
        assert torch.allclose(tensor, torch.zeros_like(tensor))
    
    def test_reset_tensor(self):
        """Test tensor reset function."""
        tensor = torch.randn(2, 3, 4)
        
        # Ensure tensor has non-zero values
        assert not torch.allclose(tensor, torch.zeros_like(tensor))
        
        # Reset and verify
        reset_result = reset_tensor(tensor)
        
        assert reset_result is tensor  # Should return same tensor
        assert torch.allclose(tensor, torch.zeros_like(tensor))
    
    def test_validate_tensor_reset(self):
        """Test tensor validation function."""
        # Zero tensor should validate as reset
        zero_tensor = torch.zeros(2, 3)
        assert validate_tensor_reset(zero_tensor)
        
        # Non-zero tensor should not validate
        non_zero_tensor = torch.ones(2, 3)
        assert not validate_tensor_reset(non_zero_tensor)


class TestTensorPoolConfig:
    """Test TensorPoolConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TensorPoolConfig()
        
        assert config.auto_device_selection is True
        assert config.enable_ref_counting is True
        assert config.enable_memory_tracking is True
        assert config.max_memory_per_device_mb == 1024
        assert config.gc_threshold_mb == 100
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TensorPoolConfig(
            initial_size=20,
            auto_device_selection=False,
            enable_ref_counting=False,
            max_memory_per_device_mb=512
        )
        
        assert config.initial_size == 20
        assert config.auto_device_selection is False
        assert config.enable_ref_counting is False
        assert config.max_memory_per_device_mb == 512


class TestTensorPoolStatistics:
    """Test TensorPoolStatistics functionality."""
    
    def test_statistics_initialization(self):
        """Test statistics object initialization."""
        stats = TensorPoolStatistics()
        
        assert isinstance(stats.device_allocations, dict)
        assert isinstance(stats.shape_frequency, dict)
        assert isinstance(stats.dtype_frequency, dict)
        assert stats.total_ref_increments == 0
        assert stats.active_refs == 0
        assert stats.memory_reuse_ratio == 0.0


class TestTensorPool:
    """Test TensorPool functionality."""
    
    def test_pool_initialization(self):
        """Test pool initialization."""
        config = TensorPoolConfig(initial_size=5)
        pool = TensorPool(config)
        
        assert pool.config == config
        assert isinstance(pool._tensor_stats, TensorPoolStatistics)
    
    def test_device_detection(self):
        """Test automatic device detection."""
        config = TensorPoolConfig(auto_device_selection=True)
        pool = TensorPool(config)
        
        # Should detect some device
        assert pool._default_device is not None
        assert isinstance(pool._default_device, torch.device)
    
    def test_get_tensor_basic(self):
        """Test basic tensor allocation."""
        pool = TensorPool()
        
        # Get a tensor
        shape = (2, 3, 4)
        tensor = pool.get_tensor(shape)
        
        assert tensor.shape == shape
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.zeros_like(tensor))
    
    def test_get_tensor_with_params(self):
        """Test tensor allocation with specific parameters."""
        pool = TensorPool()
        
        shape = (5, 6)
        dtype = torch.float16
        device = torch.device('cpu')
        
        tensor = pool.get_tensor(shape, dtype, device)
        
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device == device
    
    def test_tensor_reuse(self):
        """Test that tensors are reused from the pool."""
        pool = TensorPool()
        shape = (3, 3)
        
        # Get and return a tensor
        tensor1 = pool.get_tensor(shape)
        tensor1.fill_(42)  # Mark it with a value
        pool.return_tensor(tensor1)
        
        # Get another tensor with same shape
        tensor2 = pool.get_tensor(shape)
        
        # Should be the same tensor object, but reset to zeros
        assert tensor2 is tensor1
        assert torch.allclose(tensor2, torch.zeros_like(tensor2))
    
    def test_batch_tensor_allocation(self):
        """Test batch tensor allocation."""
        pool = TensorPool()
        
        batch_size = 8
        shape = (3, 4, 5)
        
        batch_tensor = pool.get_batch_tensors(batch_size, shape)
        
        expected_shape = (batch_size,) + shape
        assert batch_tensor.shape == expected_shape
    
    def test_reference_counting(self):
        """Test reference counting functionality."""
        config = TensorPoolConfig(enable_ref_counting=True)
        pool = TensorPool(config)
        
        tensor = pool.get_tensor((2, 2))
        
        # Increment reference
        ref_tensor = pool.increment_ref(tensor)
        assert ref_tensor is tensor
        
        # Return once - should still be in use due to ref count
        pool.return_tensor(tensor)
        stats = pool.get_statistics()
        assert stats.current_in_use == 1
        
        # Return again - should now be released
        pool.return_tensor(tensor)
        stats = pool.get_statistics()
        assert stats.current_in_use == 0
    
    def test_reference_counting_disabled(self):
        """Test behavior when reference counting is disabled."""
        config = TensorPoolConfig(enable_ref_counting=False)
        pool = TensorPool(config)
        
        tensor = pool.get_tensor((2, 2))
        
        # Increment ref should do nothing when disabled
        ref_tensor = pool.increment_ref(tensor)
        assert ref_tensor is tensor
        
        # Single return should release immediately
        pool.return_tensor(tensor)
        stats = pool.get_statistics()
        assert stats.current_in_use == 0
    
    def test_device_clearing(self):
        """Test clearing tensors for specific devices."""
        pool = TensorPool()
        device = torch.device('cpu')
        
        # Allocate and return some tensors
        tensors = []
        for i in range(3):
            tensor = pool.get_tensor((i+1, i+1), device=device)
            tensors.append(tensor)
        
        for tensor in tensors:
            pool.return_tensor(tensor)
        
        # Verify tensors are pooled
        memory_info = pool.get_memory_info()
        assert memory_info['total_pooled_tensors'] > 0
        
        # Clear device
        pool.clear_device(device)
        
        # Verify tensors are cleared
        memory_info = pool.get_memory_info()
        assert memory_info['total_pooled_tensors'] == 0
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        config = TensorPoolConfig(enable_memory_tracking=True)
        pool = TensorPool(config)
        
        # Allocate various tensors
        shapes = [(10, 10), (5, 5, 5), (100,)]
        tensors = []
        
        for shape in shapes:
            tensor = pool.get_tensor(shape)
            tensors.append(tensor)
        
        stats = pool.get_statistics()
        
        # Should track allocations and memory usage
        assert stats.total_allocations == len(shapes)
        assert stats.current_in_use == len(shapes)
        assert len(stats.device_memory_usage_mb) > 0
        
        # Return tensors
        for tensor in tensors:
            pool.return_tensor(tensor)
        
        stats = pool.get_statistics()
        assert stats.current_in_use == 0
        assert stats.total_returns == len(shapes)
    
    def test_shape_and_dtype_frequency(self):
        """Test tracking of shape and dtype frequency."""
        pool = TensorPool()
        
        # Request same shape multiple times
        shape = (4, 4)
        for _ in range(5):
            tensor = pool.get_tensor(shape)
            pool.return_tensor(tensor)
        
        # Request different dtype
        for _ in range(3):
            tensor = pool.get_tensor(shape, dtype=torch.float16)
            pool.return_tensor(tensor)
        
        stats = pool.get_statistics()
        
        assert stats.shape_frequency[shape] == 8  # 5 + 3
        assert stats.dtype_frequency['torch.float32'] == 5
        assert stats.dtype_frequency['torch.float16'] == 3
    
    def test_memory_info(self):
        """Test detailed memory information retrieval."""
        pool = TensorPool()
        
        # Allocate various tensors
        tensor1 = pool.get_tensor((10, 10))
        tensor2 = pool.get_tensor((5, 5, 5))
        pool.return_tensor(tensor1)
        
        info = pool.get_memory_info()
        
        assert 'total_pooled_tensors' in info
        assert 'total_in_use' in info
        assert 'unique_shapes' in info
        assert 'device_breakdown' in info
        assert 'shape_distribution' in info
        assert 'dtype_distribution' in info
        
        assert info['total_pooled_tensors'] == 1  # Only tensor1 returned
        assert info['total_in_use'] == 1  # tensor2 still in use
        assert info['unique_shapes'] == 2  # Two different shapes used
    
    def test_pool_optimization(self):
        """Test pool optimization functionality."""
        pool = TensorPool()
        
        # This should run without errors
        pool.optimize_pools()
        
        # TODO: Add more specific optimization tests when optimization logic is enhanced
    
    def test_statistics_calculation(self):
        """Test statistics calculation methods."""
        pool = TensorPool()
        
        # Initially should have zero hit rate and reuse ratio
        stats = pool.get_statistics()
        assert stats.hit_rate == 0.0
        assert stats.memory_reuse_ratio == 0.0
        
        # Allocate and return to create some hits
        tensor = pool.get_tensor((3, 3))
        pool.return_tensor(tensor)
        
        # Get same tensor again (should be a hit)
        tensor2 = pool.get_tensor((3, 3))
        pool.return_tensor(tensor2)
        
        stats = pool.get_statistics()
        assert stats.hit_rate > 0.0  # Should have some hit rate now
        assert stats.memory_reuse_ratio > 0.0  # Should have some reuse
    
    def test_thread_safety(self):
        """Test thread safety of tensor pool operations."""
        pool = TensorPool()
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    tensor = pool.get_tensor((5, 5))
                    time.sleep(0.001)  # Small delay to increase contention
                    pool.return_tensor(tensor)
                    results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 operations each
        
        # Pool should be consistent
        stats = pool.get_statistics()
        assert stats.current_in_use == 0  # All tensors should be returned
    
    def test_convenience_aliases(self):
        """Test convenience alias methods."""
        pool = TensorPool()
        
        # Test get alias
        tensor = pool.get((3, 3))
        assert tensor.shape == (3, 3)
        
        # Test put alias
        pool.put(tensor)  # Should not raise error
        
        stats = pool.get_statistics()
        assert stats.current_in_use == 0
    
    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        pool = TensorPool()
        
        # Try to return tensor not from pool
        external_tensor = torch.zeros(2, 2)
        
        # Should handle gracefully (just warn, don't error)
        pool.return_tensor(external_tensor)
        
        # Try to increment ref for external tensor
        with patch('yinsh_ml.memory.tensor_pool.logger') as mock_logger:
            result = pool.increment_ref(external_tensor)
            assert result is external_tensor
            # Should have logged warning
            mock_logger.warning.assert_called()


class TestCreateTensorPool:
    """Test the create_tensor_pool convenience function."""
    
    def test_create_with_defaults(self):
        """Test creating pool with default settings."""
        pool = create_tensor_pool()
        
        assert isinstance(pool, TensorPool)
        assert pool.config.initial_size == 10
        assert pool.config.auto_device_selection is True
        assert pool.config.enable_ref_counting is True
        assert pool.config.enable_memory_tracking is True
    
    def test_create_with_custom_settings(self):
        """Test creating pool with custom settings."""
        pool = create_tensor_pool(
            initial_size=20,
            auto_device_selection=False,
            enable_ref_counting=False,
            enable_memory_tracking=False
        )
        
        assert isinstance(pool, TensorPool)
        assert pool.config.initial_size == 20
        assert pool.config.auto_device_selection is False
        assert pool.config.enable_ref_counting is False
        assert pool.config.enable_memory_tracking is False


class TestTensorPoolIntegration:
    """Integration tests for TensorPool with various scenarios."""
    
    def test_neural_network_simulation(self):
        """Test TensorPool with neural network-like tensor usage patterns."""
        pool = TensorPool()
        
        # Simulate typical neural network tensor shapes
        batch_size = 32
        
        # Input tensors (game states)
        input_tensors = []
        for _ in range(batch_size):
            tensor = pool.get_tensor((6, 11, 11))  # Typical game state shape
            input_tensors.append(tensor)
        
        # Feature map tensors (intermediate layers)
        feature_tensors = []
        for _ in range(batch_size):
            tensor = pool.get_tensor((256, 11, 11))  # Feature maps
            feature_tensors.append(tensor)
        
        # Output tensors
        policy_tensors = []
        value_tensors = []
        for _ in range(batch_size):
            policy = pool.get_tensor((7395,))  # Policy output
            value = pool.get_tensor((1,))  # Value output
            policy_tensors.append(policy)
            value_tensors.append(value)
        
        # Return all tensors
        for tensor in input_tensors + feature_tensors + policy_tensors + value_tensors:
            pool.return_tensor(tensor)
        
        stats = pool.get_statistics()
        
        # Should have tracked all allocations
        expected_total = batch_size * 4  # 4 types of tensors per batch item
        assert stats.total_allocations == expected_total
        assert stats.current_in_use == 0
        assert stats.total_returns == expected_total
        
        # Check memory info
        info = pool.get_memory_info()
        assert info['unique_shapes'] == 4  # 4 different tensor shapes
        assert info['total_pooled_tensors'] == expected_total
    
    def test_mcts_simulation(self):
        """Test TensorPool with MCTS-like usage patterns."""
        pool = TensorPool()
        
        # Simulate MCTS tree search with many state evaluations
        tensors_in_use = []
        
        # Build up tree (simulating holding many states)
        for i in range(50):
            state_tensor = pool.get_tensor((6, 11, 11))
            policy_tensor = pool.get_tensor((7395,))
            
            # Some tensors shared (increment ref)
            if i % 10 == 0:
                shared_state = pool.increment_ref(state_tensor)
                tensors_in_use.append(shared_state)
            
            tensors_in_use.extend([state_tensor, policy_tensor])
        
        stats_mid = pool.get_statistics()
        assert stats_mid.current_in_use > 0
        assert stats_mid.total_ref_increments >= 5  # At least 5 ref increments
        
        # Return all tensors (simulating tree cleanup)
        for tensor in tensors_in_use:
            pool.return_tensor(tensor)
        
        stats_final = pool.get_statistics()
        assert stats_final.current_in_use == 0
        
        # Should have good reuse ratio
        assert stats_final.memory_reuse_ratio > 0.0
    
    def test_device_switching_simulation(self):
        """Test TensorPool with device switching scenarios."""
        pool = TensorPool()
        
        # Allocate tensors on CPU
        cpu_tensors = []
        for _ in range(5):
            tensor = pool.get_tensor((10, 10), device=torch.device('cpu'))
            cpu_tensors.append(tensor)
        
        # Return CPU tensors
        for tensor in cpu_tensors:
            pool.return_tensor(tensor)
        
        # If CUDA available, test CUDA tensors
        if torch.cuda.is_available():
            cuda_tensors = []
            for _ in range(3):
                tensor = pool.get_tensor((10, 10), device=torch.device('cuda'))
                cuda_tensors.append(tensor)
            
            for tensor in cuda_tensors:
                pool.return_tensor(tensor)
        
        info = pool.get_memory_info()
        
        # Should track tensors by device
        assert 'cpu' in str(info['device_breakdown'])
        
        # Clear CPU device
        pool.clear_device(torch.device('cpu'))
        
        info_after_clear = pool.get_memory_info()
        # CPU tensors should be cleared
        assert info_after_clear['total_pooled_tensors'] < info['total_pooled_tensors']
    
    def test_mixed_dtype_usage(self):
        """Test TensorPool with mixed data types."""
        pool = TensorPool()
        
        # Allocate tensors with different dtypes
        float32_tensor = pool.get_tensor((5, 5), dtype=torch.float32)
        float16_tensor = pool.get_tensor((5, 5), dtype=torch.float16)
        int8_tensor = pool.get_tensor((5, 5), dtype=torch.int8)
        
        # Return them
        pool.return_tensor(float32_tensor)
        pool.return_tensor(float16_tensor)
        pool.return_tensor(int8_tensor)
        
        stats = pool.get_statistics()
        
        # Should track all dtypes
        assert 'torch.float32' in stats.dtype_frequency
        assert 'torch.float16' in stats.dtype_frequency
        assert 'torch.int8' in stats.dtype_frequency
        
        # Get same shapes again - should reuse appropriately
        float32_tensor2 = pool.get_tensor((5, 5), dtype=torch.float32)
        float16_tensor2 = pool.get_tensor((5, 5), dtype=torch.float16)
        
        # Should be the same objects (reused)
        assert float32_tensor2 is float32_tensor
        assert float16_tensor2 is float16_tensor
        
        pool.return_tensor(float32_tensor2)
        pool.return_tensor(float16_tensor2) 