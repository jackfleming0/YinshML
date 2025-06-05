"""Core memory pool implementation for object reuse and memory management."""

import threading
import logging
import time
import warnings
import weakref
from collections import deque
from typing import TypeVar, Generic, Optional, Callable, Any, ContextManager
from contextlib import contextmanager

from .config import PoolConfig, GrowthPolicy

# Type variable for pooled objects
T = TypeVar('T')

logger = logging.getLogger(__name__)


class PoolStatistics:
    """Statistics tracking for memory pool performance."""
    
    def __init__(self):
        self.allocations = 0
        self.deallocations = 0
        self.hits = 0  # Objects retrieved from pool
        self.misses = 0  # Objects created because pool was empty
        self.peak_size = 0
        self.total_created = 0
        self.creation_time = 0.0
        self.reset_time = 0.0
        self.auto_growths = 0  # Number of automatic pool growths
        self.manual_resizes = 0  # Number of manual resize operations
        
        # Enhanced cleanup and lifecycle tracking
        self.cleanup_events = 0  # Number of cleanup operations performed
        self.objects_cleaned = 0  # Total objects removed during cleanup
        self.timeout_removals = 0  # Objects removed due to timeout
        self.last_cleanup_time = 0.0  # Timestamp of last cleanup
        self.cleanup_duration = 0.0  # Total time spent in cleanup operations
        self.leaked_objects = 0  # Estimated number of leaked objects
        
        # Object lifecycle tracking
        self.object_lifetimes = deque(maxlen=1000)  # Recent object lifetime samples
        self.min_lifetime = float('inf')
        self.max_lifetime = 0.0
        
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage."""
        total_requests = self.hits + self.misses
        return (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate as percentage."""
        return 100.0 - self.hit_rate
    
    @property
    def avg_object_lifetime(self) -> float:
        """Calculate average object lifetime in seconds."""
        return sum(self.object_lifetimes) / len(self.object_lifetimes) if self.object_lifetimes else 0.0
    
    @property
    def cleanup_efficiency(self) -> float:
        """Calculate cleanup efficiency as percentage of objects cleaned vs total created."""
        return (self.objects_cleaned / self.total_created * 100) if self.total_created > 0 else 0.0
        
    def record_object_lifetime(self, lifetime: float):
        """Record an object's lifetime for statistics."""
        self.object_lifetimes.append(lifetime)
        self.min_lifetime = min(self.min_lifetime, lifetime)
        self.max_lifetime = max(self.max_lifetime, lifetime)
        
    def reset(self):
        """Reset all statistics."""
        self.__init__()


class TrackedObject:
    """Wrapper for objects in the pool to track usage and lifetime."""
    
    def __init__(self, obj: T):
        self.obj = obj
        self.created_time = time.time()
        self.last_used_time = self.created_time
        self.use_count = 0
        
    def mark_used(self):
        """Mark object as recently used."""
        self.last_used_time = time.time()
        self.use_count += 1
        
    def get_idle_time(self) -> float:
        """Get time since object was last used."""
        return time.time() - self.last_used_time
        
    def get_lifetime(self) -> float:
        """Get total lifetime of the object."""
        return time.time() - self.created_time


class PooledObject(Generic[T]):
    """Context manager wrapper for automatic object return to pool."""
    
    def __init__(self, obj: T, pool: 'MemoryPool[T]'):
        self.obj = obj
        self._pool = pool
        self._returned = False
        
    def __enter__(self) -> T:
        return self.obj
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._returned:
            self._pool.return_obj(self.obj)
            self._returned = True
            
    def release(self):
        """Manually return object to pool."""
        if not self._returned:
            self._pool.return_obj(self.obj)
            self._returned = True


class MemoryPool(Generic[T]):
    """Thread-safe memory pool for object reuse and memory management.
    
    This class provides a generic memory pool implementation that can be used
    to reduce memory allocation overhead by reusing objects. It supports:
    - Thread-safe operations using RLock
    - Configurable growth policies
    - Statistics tracking
    - Context manager support for automatic object return
    - Exception handling with fallback to direct creation
    - Automatic pool growth based on demand and policy
    - Object lifetime tracking and cleanup mechanisms
    - Memory leak detection and prevention
    """
    
    def __init__(self, config: PoolConfig):
        """Initialize the memory pool.
        
        Args:
            config: Pool configuration specifying behavior and limits
        """
        self.config = config
        self._pool: deque[TrackedObject[T]] = deque()
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._statistics = PoolStatistics() if config.enable_statistics else None
        self._shutdown = False
        self._cleanup_thread: Optional[threading.Thread] = None
        self._created_objects = weakref.WeakSet()  # Track all created objects for leak detection
        
        # Pre-allocate initial objects if factory function is provided
        if config.factory_func and config.initial_size > 0:
            self._preallocate(config.initial_size)
            
        # Start automatic cleanup if configured
        if (config.auto_cleanup and 
            config.cleanup_interval > 0 and 
            config.enable_statistics):
            self.start_cleanup_scheduler(
                interval_sec=config.cleanup_interval,
                threshold_percent=config.cleanup_threshold * 100,
                min_idle_time_sec=config.object_timeout or 60.0
            )
            
    def __del__(self):
        """Destructor to ensure cleanup thread is stopped."""
        try:
            self.stop_cleanup_scheduler()
        except Exception:
            # Ignore cleanup errors during destruction
            pass
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup thread is stopped."""
        self.stop_cleanup_scheduler()
        
    def _preallocate(self, count: int):
        """Pre-allocate objects to populate the pool."""
        if not self.config.factory_func:
            return
            
        start_time = time.time()
        for _ in range(count):
            try:
                obj = self.config.factory_func()
                tracked_obj = TrackedObject(obj)
                self._pool.append(tracked_obj)
                self._created_objects.add(tracked_obj)
                if self._statistics:
                    self._statistics.total_created += 1
            except Exception as e:
                logger.warning(f"Failed to pre-allocate object: {e}")
                break
                
        if self._statistics:
            self._statistics.creation_time += time.time() - start_time
            self._statistics.peak_size = max(self._statistics.peak_size, len(self._pool))
            
    def _calculate_growth_size(self, current_size: int) -> int:
        """Calculate new pool size based on growth policy.
        
        Args:
            current_size: Current pool size
            
        Returns:
            New pool size according to the growth policy
        """
        if self.config.growth_policy == GrowthPolicy.FIXED:
            return current_size  # Fixed policy doesn't grow
        
        max_capacity = self.config.max_capacity or float('inf')
        
        if self.config.growth_policy == GrowthPolicy.LINEAR:
            # Linear growth adds a fixed amount
            growth_amount = max(1, int(self.config.growth_factor))
            new_size = current_size + growth_amount
            return int(min(new_size, max_capacity))
        
        elif self.config.growth_policy == GrowthPolicy.EXPONENTIAL:
            # Exponential growth multiplies by the growth factor
            new_size = max(current_size + 1, int(current_size * self.config.growth_factor))
            return int(min(new_size, max_capacity))
        
        else:
            raise ValueError(f"Unknown growth policy: {self.config.growth_policy}")
    
    def _try_auto_grow(self) -> bool:
        """Attempt to automatically grow the pool if policy allows.
        
        Returns:
            True if pool was grown, False otherwise
        """
        if self.config.growth_policy == GrowthPolicy.FIXED:
            return False
            
        if not self.config.factory_func:
            return False
            
        current_size = len(self._pool)
        max_capacity = self.config.max_capacity or float('inf')
        
        if current_size >= max_capacity:
            return False
            
        new_size = self._calculate_growth_size(current_size)
        if new_size > current_size:
            growth_amount = new_size - current_size
            self._preallocate(growth_amount)
            
            if self._statistics:
                self._statistics.auto_growths += 1
                
            logger.debug(f"Pool auto-grew from {current_size} to {len(self._pool)} objects")
            return True
            
        return False
            
    def get(self) -> T:
        """Get an object from the pool.
        
        Returns:
            An object from the pool, or a newly created one if pool is empty
            
        Raises:
            RuntimeError: If no factory function is provided and pool is empty
        """
        with self._lock:
            if self._statistics:
                self._statistics.allocations += 1
                
            # Try to get from pool first
            if self._pool:
                tracked_obj = self._pool.popleft()
                tracked_obj.mark_used()  # Update usage statistics
                if self._statistics:
                    self._statistics.hits += 1
                return tracked_obj.obj
                
            # Pool is empty - try to auto-grow if policy allows
            auto_grew = self._try_auto_grow()
            if auto_grew and self._pool:
                # Pool grew successfully, try again
                tracked_obj = self._pool.popleft()
                tracked_obj.mark_used()  # Update usage statistics
                if self._statistics:
                    self._statistics.hits += 1
                return tracked_obj.obj
                    
            # Pool is still empty, create new object directly
            if self._statistics:
                self._statistics.misses += 1
                
            if not self.config.factory_func:
                raise RuntimeError("Pool is empty and no factory function provided")
                
            try:
                start_time = time.time()
                obj = self.config.factory_func()
                tracked_obj = TrackedObject(obj)
                tracked_obj.mark_used()  # Mark as used immediately
                self._created_objects.add(tracked_obj)
                
                if self._statistics:
                    self._statistics.creation_time += time.time() - start_time
                    self._statistics.total_created += 1
                return obj
            except Exception as e:
                logger.error(f"Failed to create object: {e}")
                raise RuntimeError(f"Object creation failed: {e}") from e
                
    def return_obj(self, obj: T):
        """Return an object to the pool.
        
        Args:
            obj: Object to return to the pool
        """
        if obj is None:
            return
            
        with self._lock:
            if self._statistics:
                self._statistics.deallocations += 1
                
            # Check capacity limits
            if (self.config.max_capacity is not None and 
                len(self._pool) >= self.config.max_capacity):
                # Pool is at capacity, discard the object
                return
                
            # Reset object if reset function is provided
            if self.config.reset_func:
                try:
                    start_time = time.time()
                    self.config.reset_func(obj)
                    if self._statistics:
                        self._statistics.reset_time += time.time() - start_time
                except Exception as e:
                    logger.warning(f"Failed to reset object: {e}")
                    # Don't return object to pool if reset failed
                    return
            
            # Wrap in TrackedObject and add to pool
            tracked_obj = TrackedObject(obj)
            tracked_obj.mark_used()  # Mark as recently used when returned
            self._created_objects.add(tracked_obj)
            self._pool.append(tracked_obj)
            
            if self._statistics:
                self._statistics.peak_size = max(self._statistics.peak_size, len(self._pool))
                
    def get_pooled(self) -> PooledObject[T]:
        """Get an object wrapped in a context manager for automatic return.
        
        Returns:
            PooledObject that automatically returns the object when done
        """
        obj = self.get()
        return PooledObject(obj, self)
        
    @contextmanager
    def borrow(self) -> ContextManager[T]:
        """Context manager for borrowing an object from the pool.
        
        Yields:
            An object from the pool that will be automatically returned
        """
        obj = self.get()
        try:
            yield obj
        finally:
            self.return_obj(obj)
            
    def resize(self, new_size: Optional[int] = None):
        """Resize the pool to a new target size or use growth policy.
        
        Args:
            new_size: Target size for the pool. If None, uses growth policy to determine size.
        """
        with self._lock:
            current_size = len(self._pool)
            
            if new_size is None:
                # Use growth policy to determine new size
                new_size = self._calculate_growth_size(current_size)
                
            if self._statistics:
                self._statistics.manual_resizes += 1
            
            if new_size > current_size:
                # Grow the pool
                if self.config.factory_func:
                    self._preallocate(new_size - current_size)
                    logger.debug(f"Pool manually resized from {current_size} to {len(self._pool)} objects")
            elif new_size < current_size:
                # Shrink the pool
                excess = current_size - new_size
                for _ in range(excess):
                    if self._pool:
                        self._pool.popleft()
                logger.debug(f"Pool shrunk from {current_size} to {len(self._pool)} objects")
                        
    def grow(self) -> bool:
        """Manually grow the pool using the configured growth policy.
        
        Returns:
            True if pool was grown, False if growth not possible or not allowed
        """
        with self._lock:
            if self.config.growth_policy == GrowthPolicy.FIXED:
                return False
                
            # Can't grow without a factory function
            if not self.config.factory_func:
                return False
                
            current_size = len(self._pool)
            max_capacity = self.config.max_capacity or float('inf')
            
            if current_size >= max_capacity:
                return False
                
            new_size = self._calculate_growth_size(current_size)
            if new_size > current_size:
                self.resize(new_size)
                return True
                
            return False
                
    def clear(self):
        """Clear all objects from the pool."""
        with self._lock:
            self._pool.clear()
            
    def size(self) -> int:
        """Get current number of objects in the pool."""
        with self._lock:
            return len(self._pool)
            
    def is_empty(self) -> bool:
        """Check if the pool is empty."""
        with self._lock:
            return len(self._pool) == 0
            
    def capacity(self) -> Optional[int]:
        """Get the maximum capacity of the pool."""
        return self.config.max_capacity
        
    def get_statistics(self) -> Optional[PoolStatistics]:
        """Get pool statistics if enabled."""
        return self._statistics
        
    def reset_statistics(self):
        """Reset pool statistics."""
        if self._statistics:
            self._statistics.reset()
            
    def __len__(self) -> int:
        """Return current pool size."""
        return self.size()
        
    def __repr__(self) -> str:
        """String representation of the pool."""
        with self._lock:
            return (f"MemoryPool(size={len(self._pool)}, "
                   f"max_capacity={self.config.max_capacity}, "
                   f"policy={self.config.growth_policy.value})")
    
    def cleanup(self, threshold_percent: float = 20.0, min_idle_time_sec: float = 60.0) -> int:
        """Release unused memory when pool usage is low.
        
        Args:
            threshold_percent: Pool usage percentage below which cleanup occurs
            min_idle_time_sec: Minimum time objects must be idle before removal
            
        Returns:
            Number of objects removed from the pool
        """
        if not self._statistics:
            return 0
            
        with self._lock:
            start_time = time.time()
            current_size = len(self._pool)
            
            if current_size == 0:
                return 0
            
            # For threshold calculation, use either max_capacity or a reasonable default
            max_capacity = self.config.max_capacity
            if max_capacity is None:
                # No capacity limit, so threshold should be based on actual pool size
                # For very low thresholds (< 10%), allow cleanup even of small pools
                if threshold_percent >= 10.0 and current_size < 5:
                    return 0  # Don't cleanup small pools unless threshold is very low
                capacity_for_threshold = max(current_size, self.config.initial_size * 2)
            else:
                capacity_for_threshold = max_capacity
            
            usage_percent = (current_size / capacity_for_threshold) * 100
            
            # Only cleanup if usage is below threshold
            if usage_percent >= threshold_percent:
                return 0
            
            # Find objects to remove based on idle time
            objects_to_remove = []
            for tracked_obj in self._pool:
                if tracked_obj.get_idle_time() > min_idle_time_sec:
                    objects_to_remove.append(tracked_obj)
            
            # Remove identified objects
            for tracked_obj in objects_to_remove:
                self._pool.remove(tracked_obj)
                # Record lifetime for statistics
                self._statistics.record_object_lifetime(tracked_obj.get_lifetime())
                
            cleanup_count = len(objects_to_remove)
            cleanup_duration = time.time() - start_time
            
            # Update statistics
            self._statistics.cleanup_events += 1
            self._statistics.objects_cleaned += cleanup_count
            self._statistics.last_cleanup_time = start_time
            self._statistics.cleanup_duration += cleanup_duration
            
            if cleanup_count > 0:
                logger.debug(f"Cleaned up {cleanup_count} idle objects in {cleanup_duration:.3f}s")
                
            return cleanup_count
    
    def cleanup_by_timeout(self, timeout_seconds: float) -> int:
        """Remove objects that haven't been used within the timeout period.
        
        Args:
            timeout_seconds: Maximum idle time before object removal
            
        Returns:
            Number of objects removed due to timeout
        """
        if not self._statistics:
            return 0
            
        with self._lock:
            objects_to_remove = []
            for tracked_obj in self._pool:
                if tracked_obj.get_idle_time() > timeout_seconds:
                    objects_to_remove.append(tracked_obj)
            
            # Remove timed-out objects
            for tracked_obj in objects_to_remove:
                self._pool.remove(tracked_obj)
                self._statistics.record_object_lifetime(tracked_obj.get_lifetime())
                
            timeout_count = len(objects_to_remove)
            self._statistics.timeout_removals += timeout_count
            
            if timeout_count > 0:
                logger.debug(f"Removed {timeout_count} objects due to timeout ({timeout_seconds}s)")
                
            return timeout_count
            
    def start_cleanup_scheduler(self, 
                              interval_sec: float = 300.0, 
                              threshold_percent: float = 20.0,
                              min_idle_time_sec: float = 60.0):
        """Start background thread for periodic cleanup.
        
        Args:
            interval_sec: Time between cleanup attempts
            threshold_percent: Pool usage threshold for cleanup
            min_idle_time_sec: Minimum idle time for object removal
        """
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("Cleanup scheduler already running")
            return
            
        def cleanup_worker():
            while not self._shutdown:
                try:
                    time.sleep(interval_sec)
                    if not self._shutdown:
                        self.cleanup(threshold_percent, min_idle_time_sec)
                except Exception as e:
                    logger.error(f"Cleanup worker error: {e}")
        
        self._shutdown = False
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker, 
            daemon=True, 
            name=f"MemoryPool-Cleanup-{id(self)}"
        )
        self._cleanup_thread.start()
        logger.debug("Started cleanup scheduler")
    
    def stop_cleanup_scheduler(self):
        """Stop the background cleanup thread."""
        self._shutdown = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
            if self._cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not stop within timeout")
            else:
                logger.debug("Stopped cleanup scheduler")
        
    def check_for_leaks(self) -> dict:
        """Detect potential memory leaks by analyzing object lifecycle.
        
        Returns:
            Dictionary with leak analysis data
        """
        if not self._statistics:
            return {"leak_detection_disabled": True}
            
        with self._lock:
            total_created = self._statistics.total_created
            available = len(self._pool)
            in_use = self._statistics.allocations - self._statistics.deallocations
            cleaned = self._statistics.objects_cleaned
            
            # Estimate live objects through weak references
            live_tracked_objects = len(self._created_objects)
            
            unaccounted = total_created - (available + cleaned)
            potential_leak = unaccounted > in_use
            
            leak_info = {
                "total_created": total_created,
                "available_in_pool": available,
                "estimated_in_use": in_use,
                "cleaned_up": cleaned,
                "live_tracked_objects": live_tracked_objects,
                "unaccounted_objects": unaccounted,
                "potential_leak_detected": potential_leak,
                "leak_percentage": (unaccounted / total_created * 100) if total_created > 0 else 0.0
            }
            
            if potential_leak:
                self._statistics.leaked_objects = unaccounted
                logger.warning(f"Potential memory leak detected: {unaccounted} unaccounted objects")
                
            return leak_info 