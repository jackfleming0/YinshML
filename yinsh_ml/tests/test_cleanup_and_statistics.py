"""Unit tests for memory pool cleanup and statistics functionality."""

import unittest
import threading
import time
from unittest.mock import Mock, patch

from yinsh_ml.memory.pool import MemoryPool, PoolStatistics, TrackedObject
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
    """Test enhanced PoolStatistics functionality."""
    
    def test_enhanced_statistics_initialization(self):
        """Test that enhanced statistics are properly initialized."""
        stats = PoolStatistics()
        
        # Basic stats
        self.assertEqual(stats.allocations, 0)
        self.assertEqual(stats.deallocations, 0)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 0)
        
        # Enhanced cleanup stats
        self.assertEqual(stats.cleanup_events, 0)
        self.assertEqual(stats.objects_cleaned, 0)
        self.assertEqual(stats.timeout_removals, 0)
        self.assertEqual(stats.leaked_objects, 0)
        
        # Lifetime tracking
        self.assertEqual(len(stats.object_lifetimes), 0)
        self.assertEqual(stats.min_lifetime, float('inf'))
        self.assertEqual(stats.max_lifetime, 0.0)
        
    def test_lifetime_recording(self):
        """Test object lifetime recording functionality."""
        stats = PoolStatistics()
        
        stats.record_object_lifetime(1.5)
        stats.record_object_lifetime(2.0)
        stats.record_object_lifetime(0.5)
        
        self.assertEqual(len(stats.object_lifetimes), 3)
        self.assertEqual(stats.min_lifetime, 0.5)
        self.assertEqual(stats.max_lifetime, 2.0)
        self.assertAlmostEqual(stats.avg_object_lifetime, 1.33, places=1)
        
    def test_cleanup_efficiency_calculation(self):
        """Test cleanup efficiency calculation."""
        stats = PoolStatistics()
        
        stats.total_created = 100
        stats.objects_cleaned = 25
        
        self.assertEqual(stats.cleanup_efficiency, 25.0)
        
        # Test edge case with no objects created
        stats.total_created = 0
        self.assertEqual(stats.cleanup_efficiency, 0.0)


class TestTrackedObject(unittest.TestCase):
    """Test TrackedObject wrapper functionality."""
    
    def test_tracked_object_creation(self):
        """Test TrackedObject creation and basic properties."""
        test_obj = TestObject(42)
        tracked = TrackedObject(test_obj)
        
        self.assertEqual(tracked.obj, test_obj)
        self.assertEqual(tracked.use_count, 0)
        self.assertGreater(tracked.created_time, 0)
        self.assertEqual(tracked.last_used_time, tracked.created_time)
        
    def test_mark_used_functionality(self):
        """Test mark_used updates usage statistics."""
        test_obj = TestObject()
        tracked = TrackedObject(test_obj)
        
        initial_time = tracked.last_used_time
        time.sleep(0.01)  # Small delay to ensure time difference
        
        tracked.mark_used()
        
        self.assertEqual(tracked.use_count, 1)
        self.assertGreater(tracked.last_used_time, initial_time)
        
    def test_idle_time_calculation(self):
        """Test idle time calculation."""
        test_obj = TestObject()
        tracked = TrackedObject(test_obj)
        
        # Should have minimal idle time initially
        self.assertLess(tracked.get_idle_time(), 0.1)
        
        # Wait and check idle time increases
        time.sleep(0.05)
        idle_time = tracked.get_idle_time()
        self.assertGreaterEqual(idle_time, 0.05)
        
    def test_lifetime_calculation(self):
        """Test lifetime calculation."""
        test_obj = TestObject()
        tracked = TrackedObject(test_obj)
        
        time.sleep(0.05)
        lifetime = tracked.get_lifetime()
        self.assertGreaterEqual(lifetime, 0.05)


class TestCleanupMechanisms(unittest.TestCase):
    """Test pool cleanup mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory_func = lambda: TestObject()
        self.reset_func = lambda obj: obj.reset()
        
    def test_basic_cleanup(self):
        """Test basic cleanup functionality."""
        config = PoolConfig(
            initial_size=3,
            factory_func=self.factory_func,
            reset_func=self.reset_func,
            enable_statistics=True
        )
        
        pool = MemoryPool(config)
        
        # Use all objects to reset their timestamps
        objs = [pool.get() for _ in range(3)]
        for obj in objs:
            pool.return_obj(obj)
            
        # Wait for objects to become idle
        time.sleep(0.1)
        
        # Cleanup with short idle time
        cleaned = pool.cleanup(threshold_percent=0.0, min_idle_time_sec=0.05)
        
        # Should have cleaned up objects
        self.assertGreater(cleaned, 0)
        self.assertLess(pool.size(), 3)
        
        # Check statistics
        stats = pool.get_statistics()
        self.assertGreater(stats.cleanup_events, 0)
        self.assertEqual(stats.objects_cleaned, cleaned)
        
    def test_cleanup_threshold_respect(self):
        """Test that cleanup respects usage threshold."""
        config = PoolConfig(
            initial_size=2,
            max_capacity=10,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        
        pool = MemoryPool(config)
        
        # Use objects to set timestamps
        objs = [pool.get() for _ in range(2)]
        for obj in objs:
            pool.return_obj(obj)
            
        time.sleep(0.1)
        
        # Cleanup with high threshold (should not clean up)
        cleaned = pool.cleanup(threshold_percent=50.0, min_idle_time_sec=0.05)
        
        # Should not have cleaned anything due to high threshold
        self.assertEqual(cleaned, 0)
        self.assertEqual(pool.size(), 2)
        
    def test_cleanup_by_timeout(self):
        """Test timeout-based cleanup."""
        config = PoolConfig(
            initial_size=3,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        
        pool = MemoryPool(config)
        
        # Use objects to set timestamps
        objs = [pool.get() for _ in range(3)]
        for obj in objs:
            pool.return_obj(obj)
            
        time.sleep(0.1)
        
        # Cleanup by timeout
        removed = pool.cleanup_by_timeout(0.05)
        
        self.assertGreater(removed, 0)
        
        # Check statistics
        stats = pool.get_statistics()
        self.assertEqual(stats.timeout_removals, removed)
        
    def test_background_cleanup_scheduler(self):
        """Test background cleanup scheduler."""
        config = PoolConfig(
            initial_size=2,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        
        pool = MemoryPool(config)
        
        # Start scheduler with short interval
        pool.start_cleanup_scheduler(
            interval_sec=0.1,
            threshold_percent=0.0,
            min_idle_time_sec=0.05
        )
        
        # Use objects
        objs = [pool.get() for _ in range(2)]
        for obj in objs:
            pool.return_obj(obj)
            
        # Wait for background cleanup
        time.sleep(0.2)
        
        # Should have performed cleanup
        stats = pool.get_statistics()
        self.assertGreater(stats.cleanup_events, 0)
        
        # Stop scheduler
        pool.stop_cleanup_scheduler()
        
    def test_auto_cleanup_configuration(self):
        """Test automatic cleanup initialization via configuration."""
        config = PoolConfig(
            initial_size=2,
            factory_func=self.factory_func,
            enable_statistics=True,
            auto_cleanup=True,
            cleanup_interval=0.1,
            cleanup_threshold=0.0,
            object_timeout=0.05
        )
        
        pool = MemoryPool(config)
        
        # Use objects
        objs = [pool.get() for _ in range(2)]
        for obj in objs:
            pool.return_obj(obj)
            
        # Wait for automatic cleanup
        time.sleep(0.2)
        
        # Should have performed cleanup automatically
        stats = pool.get_statistics()
        self.assertGreater(stats.cleanup_events, 0)
        
        # Cleanup on context exit
        pool.stop_cleanup_scheduler()


class TestMemoryLeakDetection(unittest.TestCase):
    """Test memory leak detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory_func = lambda: TestObject()
        
    def test_no_leak_detection(self):
        """Test normal operation shows no leaks."""
        config = PoolConfig(
            initial_size=2,
            factory_func=self.factory_func,
            enable_statistics=True
        )
        
        pool = MemoryPool(config)
        
        # Normal use pattern
        obj1 = pool.get()
        obj2 = pool.get()
        pool.return_obj(obj1)
        pool.return_obj(obj2)
        
        leak_info = pool.check_for_leaks()
        
        self.assertFalse(leak_info["potential_leak_detected"])
        self.assertEqual(leak_info["leak_percentage"], 0.0)
        
    def test_leak_detection_disabled_without_statistics(self):
        """Test leak detection is disabled when statistics are off."""
        config = PoolConfig(
            initial_size=2,
            factory_func=self.factory_func,
            enable_statistics=False
        )
        
        pool = MemoryPool(config)
        leak_info = pool.check_for_leaks()
        
        self.assertTrue(leak_info["leak_detection_disabled"])
        
    def test_context_manager_functionality(self):
        """Test MemoryPool as context manager."""
        config = PoolConfig(
            initial_size=1,
            factory_func=self.factory_func,
            enable_statistics=True,
            auto_cleanup=True,
            cleanup_interval=0.1
        )
        
        with MemoryPool(config) as pool:
            obj = pool.get()
            pool.return_obj(obj)
            
            # Pool should be functional within context
            self.assertEqual(pool.size(), 1)
            
        # Cleanup thread should be stopped after context exit
        # This is verified by the pool being properly cleaned up


class TestStatisticsIntegration(unittest.TestCase):
    """Test integration of statistics with pool operations."""
    
    def test_comprehensive_statistics_tracking(self):
        """Test that all statistics are properly tracked during operations."""
        config = PoolConfig(
            initial_size=2,
            factory_func=lambda: TestObject(),
            reset_func=lambda obj: obj.reset(),
            enable_statistics=True
        )
        
        pool = MemoryPool(config)
        stats = pool.get_statistics()
        
        # Initial state
        self.assertEqual(stats.total_created, 2)  # Pre-allocated
        self.assertEqual(stats.allocations, 0)
        self.assertEqual(stats.deallocations, 0)
        
        # Get objects (should be hits)
        obj1 = pool.get()
        obj2 = pool.get()
        
        self.assertEqual(stats.allocations, 2)
        self.assertEqual(stats.hits, 2)
        self.assertEqual(stats.misses, 0)
        
        # Get third object (should be miss, creates new)
        obj3 = pool.get()
        
        self.assertEqual(stats.allocations, 3)
        self.assertEqual(stats.hits, 2)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.total_created, 3)
        
        # Return objects
        pool.return_obj(obj1)
        pool.return_obj(obj2)
        pool.return_obj(obj3)
        
        self.assertEqual(stats.deallocations, 3)
        self.assertEqual(pool.size(), 3)
        
        # Test cleanup statistics
        time.sleep(0.1)
        cleaned = pool.cleanup(threshold_percent=0.0, min_idle_time_sec=0.05)
        
        self.assertGreater(cleaned, 0)
        self.assertEqual(stats.objects_cleaned, cleaned)
        self.assertEqual(stats.cleanup_events, 1)


if __name__ == '__main__':
    unittest.main() 