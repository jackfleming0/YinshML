"""Unit tests for MemoryPool base class."""

import unittest
import threading
import time
from unittest.mock import Mock, patch
from collections import deque

from yinsh_ml.memory.pool import MemoryPool, PooledObject, PoolStatistics
from yinsh_ml.memory.config import PoolConfig, GrowthPolicy


class TestObject:
    """Simple test object for pool testing."""
    
    def __init__(self, value: int = 0):
        self.value = value
        self.reset_called = False
        
    def reset(self):
        """Reset method for testing reset functionality."""
        self.value = 0
        self.reset_called = True


class TestPoolStatistics(unittest.TestCase):
    """Test PoolStatistics functionality."""
    
    def test_initial_state(self):
        """Test initial statistics state."""
        stats = PoolStatistics()
        self.assertEqual(stats.allocations, 0)
        self.assertEqual(stats.deallocations, 0)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 0)
        self.assertEqual(stats.hit_rate, 0.0)
        self.assertEqual(stats.miss_rate, 0.0)
        
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = PoolStatistics()
        stats.hits = 8
        stats.misses = 2
        self.assertEqual(stats.hit_rate, 80.0)
        self.assertEqual(stats.miss_rate, 20.0)
        
    def test_reset(self):
        """Test statistics reset."""
        stats = PoolStatistics()
        stats.hits = 10
        stats.misses = 5
        stats.reset()
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 0)


class TestPoolConfig(unittest.TestCase):
    """Test PoolConfig validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = PoolConfig(
            initial_size=5,
            max_capacity=100,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=10
        )
        self.assertEqual(config.initial_size, 5)
        self.assertEqual(config.max_capacity, 100)
        
    def test_invalid_initial_size(self):
        """Test invalid initial size."""
        with self.assertRaises(ValueError):
            PoolConfig(initial_size=-1)
            
    def test_invalid_max_capacity(self):
        """Test invalid max capacity."""
        with self.assertRaises(ValueError):
            PoolConfig(initial_size=10, max_capacity=5)
            
    def test_invalid_growth_factor(self):
        """Test invalid growth factor."""
        with self.assertRaises(ValueError):
            PoolConfig(growth_factor=0)
            
    def test_invalid_cleanup_threshold(self):
        """Test invalid cleanup threshold."""
        with self.assertRaises(ValueError):
            PoolConfig(cleanup_threshold=1.5)


class TestMemoryPool(unittest.TestCase):
    """Test MemoryPool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory_func = lambda: TestObject()
        self.reset_func = lambda obj: obj.reset()
        
    def test_basic_pool_operations(self):
        """Test basic get and return operations."""
        config = PoolConfig(
            initial_size=3,
            factory_func=self.factory_func,
            reset_func=self.reset_func
        )
        pool = MemoryPool[TestObject](config)
        
        # Pool should be pre-populated
        self.assertEqual(pool.size(), 3)
        
        # Get an object
        obj = pool.get()
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(pool.size(), 2)
        
        # Return the object
        pool.return_obj(obj)
        self.assertEqual(pool.size(), 3)
        self.assertTrue(obj.reset_called)
        
    def test_pool_without_factory(self):
        """Test pool behavior without factory function."""
        config = PoolConfig(initial_size=0)
        pool = MemoryPool[TestObject](config)
        
        # Should raise error when trying to get from empty pool
        with self.assertRaises(RuntimeError):
            pool.get()
            
    def test_pool_with_factory_no_preallocation(self):
        """Test pool with factory but no pre-allocation."""
        config = PoolConfig(
            initial_size=0,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        self.assertEqual(pool.size(), 0)
        
        # Should create new object when pool is empty
        obj = pool.get()
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(pool.size(), 0)
        
    def test_max_capacity_limit(self):
        """Test maximum capacity enforcement."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=2,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        # Get both objects
        obj1 = pool.get()
        obj2 = pool.get()
        self.assertEqual(pool.size(), 0)
        
        # Return both objects
        pool.return_obj(obj1)
        pool.return_obj(obj2)
        self.assertEqual(pool.size(), 2)
        
        # Try to return a third object - should be discarded
        obj3 = TestObject()
        pool.return_obj(obj3)
        self.assertEqual(pool.size(), 2)
        
    def test_context_manager_borrow(self):
        """Test context manager borrowing."""
        config = PoolConfig(
            initial_size=1,
            factory_func=self.factory_func,
            reset_func=self.reset_func
        )
        pool = MemoryPool[TestObject](config)
        
        initial_size = pool.size()
        
        with pool.borrow() as obj:
            self.assertIsInstance(obj, TestObject)
            self.assertEqual(pool.size(), initial_size - 1)
            obj.value = 42
            
        # Object should be returned and reset
        self.assertEqual(pool.size(), initial_size)
        
    def test_pooled_object_wrapper(self):
        """Test PooledObject wrapper."""
        config = PoolConfig(
            initial_size=1,
            factory_func=self.factory_func,
            reset_func=self.reset_func
        )
        pool = MemoryPool[TestObject](config)
        
        initial_size = pool.size()
        
        with pool.get_pooled() as obj:
            self.assertIsInstance(obj, TestObject)
            self.assertEqual(pool.size(), initial_size - 1)
            obj.value = 42
            
        # Object should be returned and reset
        self.assertEqual(pool.size(), initial_size)
        
    def test_manual_release(self):
        """Test manual release of pooled object."""
        config = PoolConfig(
            initial_size=1,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        pooled_obj = pool.get_pooled()
        self.assertEqual(pool.size(), 0)
        
        pooled_obj.release()
        self.assertEqual(pool.size(), 1)
        
        # Second release should be safe
        pooled_obj.release()
        self.assertEqual(pool.size(), 1)
        
    def test_resize_operations(self):
        """Test pool resizing."""
        config = PoolConfig(
            initial_size=2,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        self.assertEqual(pool.size(), 2)
        
        # Grow the pool
        pool.resize(5)
        self.assertEqual(pool.size(), 5)
        
        # Shrink the pool
        pool.resize(3)
        self.assertEqual(pool.size(), 3)
        
    def test_clear_operation(self):
        """Test pool clearing."""
        config = PoolConfig(
            initial_size=3,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        self.assertEqual(pool.size(), 3)
        pool.clear()
        self.assertEqual(pool.size(), 0)
        self.assertTrue(pool.is_empty())
        
    def test_statistics_tracking(self):
        """Test statistics collection."""
        config = PoolConfig(
            initial_size=2,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        stats = pool.get_statistics()
        self.assertIsNotNone(stats)
        self.assertEqual(stats.total_created, 2)  # Pre-allocated objects
        
        # Get object from pool (hit)
        obj1 = pool.get()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.allocations, 1)
        
        # Get another object from pool (hit)
        obj2 = pool.get()
        self.assertEqual(stats.hits, 2)
        
        # Get third object - should create new one (miss)
        obj3 = pool.get()
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.total_created, 3)
        
        # Return objects
        pool.return_obj(obj1)
        pool.return_obj(obj2)
        pool.return_obj(obj3)
        self.assertEqual(stats.deallocations, 3)
        
        # Check hit rate
        self.assertAlmostEqual(stats.hit_rate, 66.67, places=1)
        
    def test_statistics_disabled(self):
        """Test pool with statistics disabled."""
        config = PoolConfig(
            initial_size=1,
            factory_func=self.factory_func,
            enable_statistics=False
        )
        pool = MemoryPool[TestObject](config)
        
        self.assertIsNone(pool.get_statistics())
        
    def test_exception_handling_in_factory(self):
        """Test exception handling during object creation."""
        def failing_factory():
            raise ValueError("Factory failed")
            
        config = PoolConfig(
            initial_size=0,
            factory_func=failing_factory
        )
        pool = MemoryPool[TestObject](config)
        
        with self.assertRaises(RuntimeError):
            pool.get()
            
    def test_exception_handling_in_reset(self):
        """Test exception handling during object reset."""
        def failing_reset(obj):
            raise ValueError("Reset failed")
            
        config = PoolConfig(
            initial_size=1,
            factory_func=self.factory_func,
            reset_func=failing_reset
        )
        pool = MemoryPool[TestObject](config)
        
        obj = pool.get()
        initial_size = pool.size()
        
        # Return should not add object back to pool if reset fails
        pool.return_obj(obj)
        self.assertEqual(pool.size(), initial_size)  # Should not increase
        
    def test_thread_safety(self):
        """Test thread safety of pool operations."""
        config = PoolConfig(
            initial_size=10,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(50):
                    obj = pool.get()
                    time.sleep(0.001)  # Simulate work
                    pool.return_obj(obj)
                    results.append(1)
            except Exception as e:
                errors.append(e)
                
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 250)  # 5 threads * 50 operations
        
        # Pool should be back to original size
        self.assertEqual(pool.size(), 10)
        
        # Statistics should be consistent
        stats = pool.get_statistics()
        self.assertEqual(stats.allocations, 250)
        self.assertEqual(stats.deallocations, 250)
        
    def test_return_none_object(self):
        """Test returning None object."""
        config = PoolConfig(initial_size=0, factory_func=self.factory_func)
        pool = MemoryPool[TestObject](config)
        
        # Should handle None gracefully
        pool.return_obj(None)
        self.assertEqual(pool.size(), 0)
        
    def test_repr(self):
        """Test string representation."""
        config = PoolConfig(
            initial_size=5,
            max_capacity=100,
            growth_policy=GrowthPolicy.LINEAR,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        repr_str = repr(pool)
        self.assertIn("MemoryPool", repr_str)
        self.assertIn("size=5", repr_str)
        self.assertIn("max_capacity=100", repr_str)
        self.assertIn("policy=linear", repr_str)


if __name__ == '__main__':
    unittest.main() 