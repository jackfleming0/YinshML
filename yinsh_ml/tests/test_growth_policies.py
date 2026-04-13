"""Unit tests for memory pool growth policies."""

import unittest
import warnings
import threading
import time
from unittest.mock import Mock

from yinsh_ml.memory.pool import MemoryPool, PoolStatistics
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


class TestGrowthPolicies(unittest.TestCase):
    """Test growth policy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory_func = lambda: TestObject()
        self.reset_func = lambda obj: obj.reset()
        
    def test_fixed_policy_no_growth(self):
        """Test that FIXED policy prevents automatic growth."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=10,
            growth_policy=GrowthPolicy.FIXED,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        # Empty the pool
        obj1 = pool.get()
        obj2 = pool.get()
        self.assertEqual(pool.size(), 0)
        
        # Get another object - should create new one, not grow pool
        obj3 = pool.get()
        self.assertEqual(pool.size(), 0)  # Pool should still be empty
        
        # Verify statistics
        stats = pool.get_statistics()
        self.assertEqual(stats.hits, 2)  # First two from pool
        self.assertEqual(stats.misses, 1)  # Third created new
        self.assertEqual(stats.auto_growths, 0)  # No automatic growth
        
    def test_linear_growth_policy(self):
        """Test LINEAR growth policy behavior."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=20,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=3,  # Grow by 3 each time
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        # Empty the pool
        obj1 = pool.get()
        obj2 = pool.get()
        self.assertEqual(pool.size(), 0)
        
        # Get another object - should trigger growth
        obj3 = pool.get()
        self.assertEqual(pool.size(), 2)  # Should have grown by 3, minus 1 taken
        
        # Verify statistics
        stats = pool.get_statistics()
        self.assertEqual(stats.auto_growths, 1)
        
        # Test manual growth
        initial_size = pool.size()
        success = pool.grow()
        self.assertTrue(success)
        self.assertEqual(pool.size(), initial_size + 3)  # Should grow by growth_factor
        
    def test_exponential_growth_policy(self):
        """Test EXPONENTIAL growth policy behavior."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=50,
            growth_policy=GrowthPolicy.EXPONENTIAL,
            growth_factor=2,  # Double each time
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        # Empty the pool
        obj1 = pool.get()
        obj2 = pool.get()
        self.assertEqual(pool.size(), 0)
        
        # Get another object - should trigger exponential growth
        # The pool will grow from 0 to max(0+1, 0*2) = 1, then 1 object is taken
        obj3 = pool.get()
        self.assertEqual(pool.size(), 0)  # Pool grew by 1, then 1 was taken
        
        # Verify statistics show auto-growth occurred
        stats = pool.get_statistics()
        self.assertEqual(stats.auto_growths, 1)
        
        # Test manual growth from current state
        initial_size = pool.size()
        success = pool.grow()
        self.assertTrue(success)
        # From size 0, exponential growth should create max(0+1, 0*2) = 1 object
        self.assertEqual(pool.size(), 1)
        
    def test_growth_respects_max_capacity(self):
        """Test that growth respects max_capacity limits."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=5,  # Small capacity
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=10,  # Large growth factor
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        # Empty the pool
        pool.get()
        pool.get()
        
        # Try to trigger growth - should be limited by max_capacity
        pool.get()
        self.assertLessEqual(pool.size(), 5)  # Should not exceed max_capacity
        
    def test_manual_resize_with_growth_policy(self):
        """Test manual resize using growth policy."""
        config = PoolConfig(
            initial_size=3,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=5,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        initial_size = pool.size()
        
        # Resize without specifying size - should use growth policy
        pool.resize()
        expected_size = initial_size + 5  # Linear growth by growth_factor
        self.assertEqual(pool.size(), expected_size)
        
    def test_grow_method_with_different_policies(self):
        """Test the grow() method with different policies."""
        # Test FIXED policy
        config_fixed = PoolConfig(
            initial_size=2,
            growth_policy=GrowthPolicy.FIXED,
            factory_func=self.factory_func
        )
        pool_fixed = MemoryPool[TestObject](config_fixed)
        self.assertFalse(pool_fixed.grow())  # Should not grow
        
        # Test LINEAR policy
        config_linear = PoolConfig(
            initial_size=2,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=3,
            factory_func=self.factory_func
        )
        pool_linear = MemoryPool[TestObject](config_linear)
        initial_size = pool_linear.size()
        self.assertTrue(pool_linear.grow())
        self.assertEqual(pool_linear.size(), initial_size + 3)
        
        # Test EXPONENTIAL policy
        config_exp = PoolConfig(
            initial_size=2,
            growth_policy=GrowthPolicy.EXPONENTIAL,
            growth_factor=2,
            factory_func=self.factory_func
        )
        pool_exp = MemoryPool[TestObject](config_exp)
        initial_size = pool_exp.size()
        self.assertTrue(pool_exp.grow())
        self.assertEqual(pool_exp.size(), initial_size * 2)
        
    def test_growth_without_factory_function(self):
        """Test growth behavior when no factory function is provided."""
        # Capture warnings during config creation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = PoolConfig(
                initial_size=0,  # Start empty
                growth_policy=GrowthPolicy.LINEAR,
                growth_factor=5,
                factory_func=None  # No factory
            )
            # Check that warning was issued
            self.assertTrue(len(w) > 0)
            self.assertIn("factory_func", str(w[0].message))
        
        # Create pool with the config
        pool = MemoryPool[TestObject](config)
        
        # Growth should not work without factory
        self.assertFalse(pool.grow())
        
    def test_capacity_edge_cases(self):
        """Test edge cases with capacity limits."""
        # Test pool at max capacity
        config = PoolConfig(
            initial_size=3,
            max_capacity=3,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=5,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        # Pool is already at max capacity
        self.assertFalse(pool.grow())  # Should not be able to grow
        
    def test_statistics_tracking_for_growth(self):
        """Test that statistics properly track growth operations."""
        config = PoolConfig(
            initial_size=1,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=2,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        stats = pool.get_statistics()
        initial_auto_growths = stats.auto_growths
        initial_manual_resizes = stats.manual_resizes
        
        # Trigger automatic growth
        pool.get()  # Take the one object
        pool.get()  # Should trigger auto-growth
        
        self.assertEqual(stats.auto_growths, initial_auto_growths + 1)
        
        # Trigger manual resize
        pool.resize(10)
        self.assertEqual(stats.manual_resizes, initial_manual_resizes + 1)
        
    def test_concurrent_growth(self):
        """Test growth behavior under concurrent access."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=50,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=5,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        pool = MemoryPool[TestObject](config)
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    obj = pool.get()
                    time.sleep(0.001)  # Simulate work
                    pool.return_obj(obj)
                    results.append(1)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads that will exhaust and trigger growth
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 50)  # 5 threads * 10 operations
        
        # Pool should have grown during concurrent access
        stats = pool.get_statistics()
        self.assertGreater(stats.auto_growths, 0)
        
    def test_invalid_growth_configurations(self):
        """Test validation of invalid growth configurations."""
        # LINEAR with growth_factor < 1
        with self.assertRaises(ValueError):
            PoolConfig(
                growth_policy=GrowthPolicy.LINEAR,
                growth_factor=0
            )
        
        # EXPONENTIAL with growth_factor <= 1
        with self.assertRaises(ValueError):
            PoolConfig(
                growth_policy=GrowthPolicy.EXPONENTIAL,
                growth_factor=1
            )
        
        # Test warnings for aggressive growth factors
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Large linear growth factor
            PoolConfig(
                growth_policy=GrowthPolicy.LINEAR,
                growth_factor=150
            )
            self.assertTrue(any("very large" in str(warning.message) for warning in w))
            
            # Aggressive exponential growth factor
            PoolConfig(
                growth_policy=GrowthPolicy.EXPONENTIAL,
                growth_factor=5
            )
            self.assertTrue(any("very aggressive" in str(warning.message) for warning in w))
            
    def test_growth_calculation_edge_cases(self):
        """Test edge cases in growth calculation."""
        config = PoolConfig(
            initial_size=0,
            growth_policy=GrowthPolicy.LINEAR,
            growth_factor=3,
            factory_func=self.factory_func
        )
        pool = MemoryPool[TestObject](config)
        
        # Test growth from empty pool
        new_size = pool._calculate_growth_size(0)
        self.assertEqual(new_size, 3)  # Should grow by growth_factor
        
        # Test exponential growth from size 1
        config.growth_policy = GrowthPolicy.EXPONENTIAL
        config.growth_factor = 2
        pool.config = config
        
        new_size = pool._calculate_growth_size(1)
        self.assertEqual(new_size, 2)  # 1 * 2 = 2
        
        # Test with max_capacity constraint
        config.max_capacity = 5
        new_size = pool._calculate_growth_size(4)
        self.assertEqual(new_size, 5)  # Should be limited by max_capacity


if __name__ == '__main__':
    unittest.main() 