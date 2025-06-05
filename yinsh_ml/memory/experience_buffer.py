"""Memory-mapped experience buffer for efficient game experience storage."""

import os
import sys
import mmap
import struct
import time
import threading
import shutil
import logging
import math
import heapq
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, NamedTuple, Set
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict, deque
import numpy as np

# Platform-specific imports
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    import ctypes.wintypes
    
    # Windows API constants
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_ALWAYS = 4
    FILE_ATTRIBUTE_NORMAL = 0x80
    PAGE_READWRITE = 0x04
    FILE_MAP_ALL_ACCESS = 0xF001F
    
    # Windows API functions
    kernel32 = ctypes.windll.kernel32
    CreateFileW = kernel32.CreateFileW
    CreateFileMappingW = kernel32.CreateFileMappingW
    MapViewOfFile = kernel32.MapViewOfFile
    UnmapViewOfFile = kernel32.UnmapViewOfFile
    CloseHandle = kernel32.CloseHandle
    SetFilePointer = kernel32.SetFilePointer
    SetEndOfFile = kernel32.SetEndOfFile

logger = logging.getLogger(__name__)


class BufferError(Exception):
    """Base exception for memory-mapped buffer operations."""
    pass


class BufferLockTimeoutError(BufferError):
    """Raised when unable to acquire buffer lock within timeout."""
    pass


class BufferCorruptionError(BufferError):
    """Raised when buffer corruption is detected."""
    pass


class BufferPlatformError(BufferError):
    """Raised when platform-specific operations fail."""
    pass


class BufferEmptyError(BufferError):
    """Raised when trying to sample from empty buffer."""
    pass


class BufferBudgetExceededError(BufferError):
    """Raised when memory budget is exceeded and eviction fails."""
    pass


class ExperiencePhase(IntEnum):
    """Enumeration for game phases."""
    RING_PLACEMENT = 0
    MAIN_GAME = 1
    RING_REMOVAL = 2


@dataclass
class ExperienceRecord:
    """Single experience record structure."""
    state: np.ndarray      # Game state tensor
    policy: np.ndarray     # Move probability distribution
    value: float           # Game outcome value
    phase: ExperiencePhase # Game phase
    timestamp: float       # Record timestamp
    experience_id: Optional[str] = None  # Unique identifier
    priority: float = 1.0  # Priority for prioritized replay


@dataclass
class BufferStatistics:
    """Detailed buffer statistics."""
    capacity: int
    count: int
    utilization: float
    write_position: int
    read_position: int
    is_full: bool
    memory_usage_bytes: int
    memory_budget_bytes: int
    memory_budget_utilization: float
    experience_size_bytes: int
    phase_distribution: Dict[str, int]
    oldest_timestamp: float
    newest_timestamp: float
    hit_rate: float
    miss_rate: float
    eviction_count: int
    lru_evictions: int
    total_additions: int
    total_accesses: int

    
@dataclass
class BufferConfig:
    """Comprehensive buffer configuration."""
    # Memory budget settings
    max_memory_bytes: int = 1_000_000_000  # 1GB default
    eviction_headroom: float = 0.1         # 10% extra space when evicting
    file_size_limit: int = 100_000_000     # 100MB per file
    
    # Directory settings
    directory: str = "./buffer_storage"
    
    # Eviction policy
    eviction_policy: str = "lru"           # "lru", "fifo", or "random"
    lru_tracking_enabled: bool = True
    
    # Indexing settings
    enable_id_indexing: bool = True
    enable_timestamp_indexing: bool = True
    index_cleanup_interval: int = 1000     # Cleanup index every N operations
    
    # Prioritized replay settings
    enable_prioritized_replay: bool = False
    priority_alpha: float = 0.6            # How much prioritization to use
    priority_beta: float = 0.4             # Importance sampling correction
    priority_epsilon: float = 1e-6         # Small constant to avoid zero priorities
    
    # Statistics settings
    stats_tracking_window: int = 1000      # Number of operations to track
    enable_hit_miss_tracking: bool = True
    
    # Performance settings
    index_hash_size: int = 100000          # Hash table size for indexing
    lru_cleanup_batch_size: int = 100      # How many entries to clean at once
    
    def validate(self):
        """Validate configuration parameters."""
        if self.max_memory_bytes <= 0:
            raise ValueError("Memory budget must be positive")
        
        if self.eviction_policy not in ["lru", "fifo", "random"]:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")
        
        if not (0 <= self.eviction_headroom <= 1):
            raise ValueError("Eviction headroom must be between 0 and 1")
        
        if self.priority_alpha < 0 or self.priority_beta < 0:
            raise ValueError("Priority parameters must be non-negative")
        
        if self.stats_tracking_window <= 0:
            raise ValueError("Stats tracking window must be positive")


class MemoryBudgetManager:
    """Manages memory budget and eviction policies."""
    
    def __init__(self, config: BufferConfig):
        self.max_bytes = config.max_memory_bytes
        self.eviction_headroom = config.eviction_headroom
        self.eviction_policy = config.eviction_policy
        self._current_bytes = 0
        self._total_additions = 0
        self._total_evictions = 0
        self._lock = threading.RLock()
        
    def can_add(self, experience_size_bytes: int) -> bool:
        """Check if we can add an experience without exceeding budget."""
        with self._lock:
            return self._current_bytes + experience_size_bytes <= self.max_bytes
    
    def register_addition(self, experience_size_bytes: int) -> None:
        """Register addition of an experience."""
        with self._lock:
            self._current_bytes += experience_size_bytes
            self._total_additions += 1
    
    def register_removal(self, experience_size_bytes: int) -> None:
        """Register removal of an experience."""
        with self._lock:
            self._current_bytes = max(0, self._current_bytes - experience_size_bytes)
            self._total_evictions += 1
    
    def bytes_to_free(self, additional_bytes: int = 0) -> int:
        """Calculate bytes to free including headroom for new experiences."""
        with self._lock:
            needed = self._current_bytes + additional_bytes - self.max_bytes
            if needed <= 0:
                return 0
            return int(needed * (1 + self.eviction_headroom))
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            return {
                "current_bytes": self._current_bytes,
                "max_bytes": self.max_bytes,
                "utilization": self._current_bytes / self.max_bytes if self.max_bytes > 0 else 0,
                "total_additions": self._total_additions,
                "total_evictions": self._total_evictions,
                "eviction_policy": self.eviction_policy
            }


class LRUTracker:
    """Tracks experience access times for LRU eviction."""
    
    def __init__(self, cleanup_batch_size: int = 100):
        self._access_times: Dict[str, float] = {}
        self._access_heap: List[Tuple[float, str]] = []  # min-heap of (access_time, experience_id)
        self._cleanup_batch_size = cleanup_batch_size
        self._operations_since_cleanup = 0
        self._lock = threading.RLock()
        
    def register_access(self, experience_id: str) -> None:
        """Register an access to an experience."""
        with self._lock:
            access_time = time.time()
            
            # Update or add access time
            if experience_id in self._access_times:
                # Mark old entry in heap as stale (lazy deletion approach)
                pass
            
            self._access_times[experience_id] = access_time
            heapq.heappush(self._access_heap, (access_time, experience_id))
            
            self._operations_since_cleanup += 1
            if self._operations_since_cleanup >= self._cleanup_batch_size:
                self._cleanup_stale_entries()
    
    def get_lru_candidates(self, count: int) -> List[str]:
        """Get the least recently used experience IDs."""
        with self._lock:
            candidates = []
            temp_heap = []
            
            # Extract valid LRU candidates
            while len(candidates) < count and self._access_heap:
                access_time, exp_id = heapq.heappop(self._access_heap)
                
                # Check if this entry is still valid (not stale)
                if exp_id in self._access_times and self._access_times[exp_id] == access_time:
                    candidates.append(exp_id)
                    # Don't put back in heap since we're evicting
                else:
                    # Stale entry, skip it
                    continue
            
            # Restore non-selected valid entries to heap
            for item in temp_heap:
                heapq.heappush(self._access_heap, item)
            
            return candidates
    
    def remove_experience(self, experience_id: str) -> None:
        """Remove an experience from tracking."""
        with self._lock:
            if experience_id in self._access_times:
                del self._access_times[experience_id]
                # Leave stale entry in heap - will be cleaned up later
    
    def _cleanup_stale_entries(self) -> None:
        """Clean up stale entries from the heap."""
        with self._lock:
            # Rebuild heap with only valid entries
            valid_entries = []
            while self._access_heap:
                access_time, exp_id = heapq.heappop(self._access_heap)
                if exp_id in self._access_times and self._access_times[exp_id] == access_time:
                    valid_entries.append((access_time, exp_id))
            
            self._access_heap = valid_entries
            heapq.heapify(self._access_heap)
            self._operations_since_cleanup = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LRU tracking statistics."""
        with self._lock:
            return {
                "tracked_experiences": len(self._access_times),
                "heap_size": len(self._access_heap),
                "operations_since_cleanup": self._operations_since_cleanup
            }


class ExperienceIndex:
    """Efficient indexing system for O(1) access to experiences."""
    
    def __init__(self, config: BufferConfig):
        self.enable_id_indexing = config.enable_id_indexing
        self.enable_timestamp_indexing = config.enable_timestamp_indexing
        
        # ID-based index: experience_id -> buffer_position
        self._id_index: Dict[str, int] = {}
        
        # Timestamp-based index: timestamp -> experience_id (sorted)
        self._timestamp_index: List[Tuple[float, str]] = []
        self._timestamp_dirty = False
        
        # Reverse lookup: buffer_position -> (experience_id, timestamp)
        self._position_index: Dict[int, Tuple[str, float]] = {}
        
        self._lock = threading.RLock()
        self._operations_count = 0
        self.cleanup_interval = config.index_cleanup_interval
    
    def add(self, experience_id: str, timestamp: float, buffer_position: int) -> None:
        """Add an experience to the index."""
        with self._lock:
            if self.enable_id_indexing:
                self._id_index[experience_id] = buffer_position
            
            if self.enable_timestamp_indexing:
                self._timestamp_index.append((timestamp, experience_id))
                self._timestamp_dirty = True
            
            self._position_index[buffer_position] = (experience_id, timestamp)
            self._operations_count += 1
            
            if self._operations_count % self.cleanup_interval == 0:
                self._cleanup()
    
    def remove(self, experience_id: str = None, buffer_position: int = None) -> None:
        """Remove an experience from the index."""
        with self._lock:
            # Get missing info from what we have
            if experience_id is None and buffer_position is not None:
                if buffer_position in self._position_index:
                    experience_id, timestamp = self._position_index[buffer_position]
            elif buffer_position is None and experience_id is not None:
                if experience_id in self._id_index:
                    buffer_position = self._id_index[experience_id]
                    if buffer_position in self._position_index:
                        _, timestamp = self._position_index[buffer_position]
            
            if experience_id is not None and buffer_position is not None:
                # Remove from all indices
                if self.enable_id_indexing and experience_id in self._id_index:
                    del self._id_index[experience_id]
                
                if buffer_position in self._position_index:
                    del self._position_index[buffer_position]
                
                # Mark timestamp index as dirty - will be cleaned up later
                if self.enable_timestamp_indexing:
                    self._timestamp_dirty = True
    
    def get_position_by_id(self, experience_id: str) -> Optional[int]:
        """Get buffer position by experience ID."""
        if not self.enable_id_indexing:
            return None
        
        with self._lock:
            return self._id_index.get(experience_id)
    
    def get_position_by_timestamp(self, timestamp: float, approximate: bool = False) -> Optional[int]:
        """Get buffer position by timestamp."""
        if not self.enable_timestamp_indexing:
            return None
        
        with self._lock:
            self._ensure_timestamp_sorted()
            
            if not self._timestamp_index:
                return None
            
            if not approximate:
                # Exact match
                for ts, exp_id in self._timestamp_index:
                    if ts == timestamp:
                        return self._id_index.get(exp_id)
                return None
            else:
                # Find closest timestamp using binary search
                left, right = 0, len(self._timestamp_index) - 1
                closest_idx = 0
                min_diff = float('inf')
                
                while left <= right:
                    mid = (left + right) // 2
                    ts, exp_id = self._timestamp_index[mid]
                    diff = abs(ts - timestamp)
                    
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = mid
                    
                    if ts < timestamp:
                        left = mid + 1
                    else:
                        right = mid - 1
                
                _, exp_id = self._timestamp_index[closest_idx]
                return self._id_index.get(exp_id)
    
    def get_id_by_position(self, buffer_position: int) -> Optional[str]:
        """Get experience ID by buffer position."""
        with self._lock:
            if buffer_position in self._position_index:
                return self._position_index[buffer_position][0]
            return None
    
    def get_experiences_by_timestamp_range(self, start_ts: float, end_ts: float) -> List[str]:
        """Get experience IDs within a timestamp range."""
        if not self.enable_timestamp_indexing:
            return []
        
        with self._lock:
            self._ensure_timestamp_sorted()
            result = []
            
            for ts, exp_id in self._timestamp_index:
                if start_ts <= ts <= end_ts:
                    result.append(exp_id)
                elif ts > end_ts:
                    break
            
            return result
    
    def _ensure_timestamp_sorted(self) -> None:
        """Ensure timestamp index is sorted and clean."""
        if self._timestamp_dirty:
            # Remove entries that no longer exist in ID index
            valid_entries = [
                (ts, exp_id) for ts, exp_id in self._timestamp_index
                if exp_id in self._id_index
            ]
            self._timestamp_index = sorted(valid_entries)
            self._timestamp_dirty = False
    
    def _cleanup(self) -> None:
        """Periodic cleanup of stale index entries."""
        with self._lock:
            if self.enable_timestamp_indexing:
                self._ensure_timestamp_sorted()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        with self._lock:
            return {
                "id_index_size": len(self._id_index) if self.enable_id_indexing else 0,
                "timestamp_index_size": len(self._timestamp_index) if self.enable_timestamp_indexing else 0,
                "position_index_size": len(self._position_index),
                "timestamp_dirty": self._timestamp_dirty,
                "operations_count": self._operations_count
            }


class HitMissTracker:
    """Tracks hit/miss rates for buffer access patterns."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._recent_operations = deque(maxlen=window_size)
        self._total_hits = 0
        self._total_misses = 0
        self._total_operations = 0
        self._lock = threading.RLock()
    
    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._recent_operations.append(True)
            self._total_hits += 1
            self._total_operations += 1
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._recent_operations.append(False)
            self._total_misses += 1
            self._total_operations += 1
    
    def get_hit_rate(self) -> float:
        """Get recent hit rate (within window)."""
        with self._lock:
            if not self._recent_operations:
                return 0.0
            hits = sum(1 for hit in self._recent_operations if hit)
            return hits / len(self._recent_operations)
    
    def get_miss_rate(self) -> float:
        """Get recent miss rate (within window)."""
        return 1.0 - self.get_hit_rate()
    
    def get_total_stats(self) -> Dict[str, Any]:
        """Get total statistics."""
        with self._lock:
            total_ops = self._total_operations
            return {
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "total_operations": total_ops,
                "total_hit_rate": self._total_hits / total_ops if total_ops > 0 else 0.0,
                "total_miss_rate": self._total_misses / total_ops if total_ops > 0 else 0.0,
                "recent_hit_rate": self.get_hit_rate(),
                "recent_miss_rate": self.get_miss_rate()
            }
    
    
class BufferHeader(NamedTuple):
    """Memory-mapped buffer header structure."""
    magic_number: int
    version: int
    item_size: int
    capacity: int
    write_position: int
    read_position: int
    count: int
    flags: int


class MemoryMappedExperienceBuffer:
    """Memory-mapped experience buffer for efficient storage and retrieval."""
    
    # Buffer constants
    MAGIC_NUMBER = 0xE5B0FFED
    VERSION = 1
    HEADER_SIZE = 64
    HEADER_FORMAT = '<LHHLLLLH38x'  # Little-endian header format
    
    def __init__(
        self,
        file_path: str,
        capacity: int = 50000,
        state_shape: Tuple[int, ...] = (10, 11, 11),
        policy_size: int = 213,
        create_if_missing: bool = True,
        lock_timeout_ms: int = 1000,
        config: Optional[BufferConfig] = None
    ):
        """
        Initialize memory-mapped experience buffer.
        
        Args:
            file_path: Path to buffer file
            capacity: Maximum number of experiences to store
            state_shape: Shape of game state tensors
            policy_size: Size of policy vectors
            create_if_missing: Whether to create file if it doesn't exist
            lock_timeout_ms: Timeout for acquiring locks in milliseconds
            config: Advanced configuration options (optional)
        """
        self.file_path = Path(file_path)
        self.capacity = capacity
        self.state_shape = state_shape
        self.policy_size = policy_size
        self.lock_timeout_ms = lock_timeout_ms
        
        # Use provided config or create default
        self.config = config or BufferConfig()
        self.config.validate()
        
        # Calculate sizes
        self.state_size = int(np.prod(state_shape)) * 4  # float32
        self.policy_bytes = policy_size * 4  # float32
        self.value_size = 4  # float32
        self.phase_size = 1  # uint8
        self.timestamp_size = 8  # double
        self.priority_size = 4  # float32 for priority
        self.id_size = 36  # UUID string (36 chars)
        
        self.item_size = (
            self.state_size + self.policy_bytes + 
            self.value_size + self.phase_size + self.timestamp_size +
            self.priority_size + self.id_size
        )
        
        self.total_size = self.HEADER_SIZE + (self.item_size * capacity)
        
        # Threading primitives
        self._write_lock = threading.RLock()
        self._read_lock = threading.RLock()
        
        # Platform-specific handles
        self._mmap_buffer = None
        self._file_handle = None
        self._win_handles = None
        
        # Advanced memory management components
        self.budget_manager = MemoryBudgetManager(self.config)
        self.lru_tracker = LRUTracker(self.config.lru_cleanup_batch_size) if self.config.lru_tracking_enabled else None
        self.index = ExperienceIndex(self.config)
        self.hit_miss_tracker = HitMissTracker(self.config.stats_tracking_window) if self.config.enable_hit_miss_tracking else None
        
        # Sampling configuration
        self._sampling_temperature = 1.0  # Controls sampling uniformity
        self._phase_weights = {
            ExperiencePhase.RING_PLACEMENT: 1.0,
            ExperiencePhase.MAIN_GAME: 1.0,
            ExperiencePhase.RING_REMOVAL: 1.0
        }
        
        # Prioritized replay components
        self._priority_weights: Dict[str, float] = {}  # experience_id -> priority
        self._priority_sum = 0.0
        self._max_priority = 1.0
        
        # Initialize buffer
        try:
            self._initialize_buffer(create_if_missing)
            logger.info(f"Initialized memory-mapped buffer: {file_path}, "
                       f"capacity={capacity}, item_size={self.item_size}, "
                       f"budget={self.config.max_memory_bytes // 1024 // 1024}MB")
        except Exception as e:
            self._cleanup()
            raise BufferError(f"Failed to initialize buffer: {e}") from e
    
    def _initialize_buffer(self, create_if_missing: bool) -> None:
        """Initialize the memory-mapped buffer."""
        file_exists = self.file_path.exists()
        
        if not file_exists and not create_if_missing:
            raise BufferError(f"Buffer file does not exist: {self.file_path}")
        
        # Create directory if needed
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create or open memory-mapped file
        if sys.platform == 'win32':
            self._mmap_buffer = self._create_windows_mmap()
        else:
            self._mmap_buffer = self._create_unix_mmap()
        
        # Initialize header if new file
        if not file_exists or self._should_initialize_header():
            self._initialize_header()
        else:
            self._validate_header()
    
    def _create_unix_mmap(self) -> mmap.mmap:
        """Create memory-mapped file on Unix-like systems."""
        try:
            # Open file with read/write permissions
            fd = os.open(str(self.file_path), os.O_RDWR | os.O_CREAT, 0o600)
            self._file_handle = fd
            
            try:
                # Ensure file is correct size
                os.ftruncate(fd, self.total_size)
                
                # Create memory mapping
                buffer = mmap.mmap(fd, self.total_size, mmap.MAP_SHARED)
                return buffer
            except Exception:
                os.close(fd)
                raise
        except Exception as e:
            raise BufferPlatformError(f"Failed to create Unix mmap: {e}") from e
    
    def _create_windows_mmap(self) -> mmap.mmap:
        """Create memory-mapped file on Windows."""
        try:
            # Convert path to wide string
            path_wide = str(self.file_path).encode('utf-16le') + b'\x00\x00'
            
            # Create/open file
            file_handle = CreateFileW(
                path_wide,
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                0
            )
            
            if file_handle == -1:
                raise BufferPlatformError("Failed to create/open file on Windows")
            
            # Set file size
            SetFilePointer(file_handle, self.total_size, None, 0)
            SetEndOfFile(file_handle)
            
            # Create file mapping
            mapping_handle = CreateFileMappingW(
                file_handle,
                None,
                PAGE_READWRITE,
                0,
                self.total_size,
                None
            )
            
            if not mapping_handle:
                CloseHandle(file_handle)
                raise BufferPlatformError("Failed to create file mapping")
            
            # Map view of file
            view_ptr = MapViewOfFile(
                mapping_handle,
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                self.total_size
            )
            
            if not view_ptr:
                CloseHandle(mapping_handle)
                CloseHandle(file_handle)
                raise BufferPlatformError("Failed to map view of file")
            
            # Store handles for cleanup
            self._win_handles = (file_handle, mapping_handle, view_ptr)
            
            # Create Python mmap-like wrapper
            return self._create_windows_buffer_wrapper(view_ptr)
            
        except Exception as e:
            raise BufferPlatformError(f"Failed to create Windows mmap: {e}") from e
    
    def _create_windows_buffer_wrapper(self, view_ptr):
        """Create a Python buffer wrapper for Windows memory mapping."""
        # This creates a ctypes array that behaves like a memory buffer
        buffer_type = ctypes.c_byte * self.total_size
        buffer = buffer_type.from_address(view_ptr)
        
        # Wrap in a mmap-like interface
        class WindowsBufferWrapper:
            def __init__(self, buffer, size):
                self._buffer = buffer
                self._size = size
                self._closed = False
            
            def __getitem__(self, key):
                if isinstance(key, slice):
                    start, stop, step = key.indices(self._size)
                    return bytes(self._buffer[start:stop:step])
                return self._buffer[key]
            
            def __setitem__(self, key, value):
                if isinstance(key, slice):
                    start, stop, step = key.indices(self._size)
                    for i, v in enumerate(value):
                        if start + i < stop:
                            self._buffer[start + i] = v
                else:
                    self._buffer[key] = value
            
            def read(self, size=-1):
                if size == -1:
                    return bytes(self._buffer)
                return bytes(self._buffer[:size])
            
            def write(self, data):
                if isinstance(data, (bytes, bytearray)):
                    for i, byte in enumerate(data):
                        if i < self._size:
                            self._buffer[i] = byte
            
            def close(self):
                self._closed = True
            
            def flush(self):
                pass
            
            @property
            def closed(self):
                return self._closed
        
        return WindowsBufferWrapper(buffer, self.total_size)
    
    def _should_initialize_header(self) -> bool:
        """Check if header needs initialization."""
        try:
            header = self._read_header()
            return header.magic_number != self.MAGIC_NUMBER
        except:
            return True
    
    def _initialize_header(self) -> None:
        """Initialize buffer header with default values."""
        header_data = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC_NUMBER,    # magic_number
            self.VERSION,         # version
            self.item_size,       # item_size
            self.capacity,        # capacity
            0,                    # write_position
            0,                    # read_position
            0,                    # count
            0                     # flags
        )
        
        self._mmap_buffer[0:self.HEADER_SIZE] = header_data
        self._mmap_buffer.flush()
        logger.debug("Initialized buffer header")
    
    def _validate_header(self) -> None:
        """Validate existing buffer header."""
        try:
            header = self._read_header()
            
            if header.magic_number != self.MAGIC_NUMBER:
                raise BufferCorruptionError(f"Invalid magic number: {header.magic_number:x}")
            
            if header.version != self.VERSION:
                raise BufferCorruptionError(f"Unsupported version: {header.version}")
            
            if header.item_size != self.item_size:
                raise BufferCorruptionError(f"Item size mismatch: {header.item_size} != {self.item_size}")
            
            if header.capacity != self.capacity:
                raise BufferCorruptionError(f"Capacity mismatch: {header.capacity} != {self.capacity}")
            
            logger.debug("Buffer header validation passed")
            
        except BufferCorruptionError as e:
            # For configuration mismatches, don't attempt recovery
            if "mismatch" in str(e).lower():
                logger.error(f"Configuration mismatch detected: {e}")
                raise
            else:
                logger.error("Buffer corruption detected, attempting recovery")
                if not self._recover_from_corruption():
                    raise
    
    def _recover_from_corruption(self) -> bool:
        """Attempt to recover from buffer corruption."""
        try:
            # Backup corrupted file
            backup_path = self.file_path.with_suffix('.corrupted')
            shutil.copy2(self.file_path, backup_path)
            logger.info(f"Backed up corrupted buffer to {backup_path}")
            
            # Reinitialize header
            self._initialize_header()
            logger.info("Successfully recovered from buffer corruption")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover from corruption: {e}")
            return False
    
    def _read_header(self) -> BufferHeader:
        """Read buffer header."""
        header_bytes = self._mmap_buffer[0:self.HEADER_SIZE]
        header_data = struct.unpack(self.HEADER_FORMAT, header_bytes)
        
        return BufferHeader(
            magic_number=header_data[0],
            version=header_data[1],
            item_size=header_data[2],
            capacity=header_data[3],
            write_position=header_data[4],
            read_position=header_data[5],
            count=header_data[6],
            flags=header_data[7]
        )
    
    def _write_header(self, **updates) -> None:
        """Write updated header to buffer."""
        current = self._read_header()
        
        # Apply updates
        new_header = current._replace(**updates)
        
        header_data = struct.pack(
            self.HEADER_FORMAT,
            new_header.magic_number,
            new_header.version,
            new_header.item_size,
            new_header.capacity,
            new_header.write_position,
            new_header.read_position,
            new_header.count,
            new_header.flags
        )
        
        self._mmap_buffer[0:self.HEADER_SIZE] = header_data
        self._mmap_buffer.flush()
    
    def _acquire_write_lock(self, timeout_ms: Optional[int] = None) -> bool:
        """Acquire write lock with timeout."""
        timeout_ms = timeout_ms or self.lock_timeout_ms
        start_time = time.time()
        
        while (time.time() - start_time) < (timeout_ms / 1000):
            if self._write_lock.acquire(False):
                return True
            time.sleep(0.001)
        
        return False
    
    def _acquire_read_lock(self, timeout_ms: Optional[int] = None) -> bool:
        """Acquire read lock with timeout."""
        timeout_ms = timeout_ms or self.lock_timeout_ms
        start_time = time.time()
        
        while (time.time() - start_time) < (timeout_ms / 1000):
            if self._read_lock.acquire(False):
                return True
            time.sleep(0.001)
        
        return False
    
    def add_experience(self, experience: ExperienceRecord, experience_id: Optional[str] = None) -> str:
        """
        Add an experience to the buffer with advanced memory management.
        
        Args:
            experience: Experience record to add
            experience_id: Optional unique identifier (auto-generated if None)
            
        Returns:
            str: Experience ID of the added experience
            
        Raises:
            BufferBudgetExceededError: If memory budget exceeded and eviction fails
        """
        # Generate experience ID if not provided
        if experience_id is None:
            experience_id = str(uuid.uuid4())
        
        # Set experience ID in record
        if hasattr(experience, 'experience_id'):
            experience.experience_id = experience_id
        
        self._validate_experience(experience)
        
        # Calculate experience size in bytes
        experience_size = self.item_size
        
        with self._write_lock:
            # Check memory budget and evict if necessary
            if not self.budget_manager.can_add(experience_size):
                bytes_to_free = self.budget_manager.bytes_to_free(experience_size)
                evicted_count = self._evict_experiences(bytes_to_free)
                
                if evicted_count == 0 and not self.budget_manager.can_add(experience_size):
                    raise BufferBudgetExceededError(
                        f"Cannot add experience: budget exceeded and eviction failed. "
                        f"Current: {self.budget_manager._current_bytes}, "
                        f"Max: {self.budget_manager.max_bytes}, "
                        f"Needed: {experience_size}"
                    )
            
            # Serialize the experience
            data = self._serialize_experience(experience, experience_id)
            
            # Get current header
            header = self._read_header()
            
            # Store at write position
            record_offset = self.HEADER_SIZE + (header.write_position * self.item_size)
            
            # Handle circular buffer wraparound and eviction of old experience
            old_experience_id = None
            if header.count >= self.capacity:
                # We're about to overwrite an existing experience
                old_experience_id = self._get_experience_id_at_position(header.write_position)
                if old_experience_id:
                    self._remove_from_tracking(old_experience_id, header.write_position)
            
            # Write the experience data
            self._mmap_buffer[record_offset:record_offset + len(data)] = data
            
            # Update positions and count
            new_write_pos = (header.write_position + 1) % self.capacity
            new_count = min(header.count + 1, self.capacity)
            new_read_pos = header.read_position
            
            # In circular buffer, advance read position when buffer is full
            if new_count >= self.capacity and header.count < self.capacity:
                new_read_pos = new_write_pos
            elif new_count >= self.capacity:
                new_read_pos = (header.read_position + 1) % self.capacity
            
            # Update header
            self._write_header(
                write_position=new_write_pos,
                read_position=new_read_pos,
                count=new_count
            )
            
            # Update tracking systems
            self._add_to_tracking(experience_id, experience.timestamp, header.write_position, experience.priority)
            
            # Update memory budget
            self.budget_manager.register_addition(experience_size)
            
            logger.debug(f"Added experience {experience_id} at position {header.write_position}")
            
            return experience_id
    
    def _validate_experience(self, experience: ExperienceRecord) -> None:
        """Validate experience record before serialization."""
        # Check state shape
        state = np.asarray(experience.state, dtype=np.float32)
        if state.shape != self.state_shape:
            raise ValueError(f"State shape {state.shape} doesn't match expected {self.state_shape}")
        
        # Check policy size
        policy = np.asarray(experience.policy, dtype=np.float32)
        if policy.size != self.policy_size:
            raise ValueError(f"Policy size {policy.size} doesn't match expected {self.policy_size}")
    
    def _serialize_experience(self, experience: ExperienceRecord, experience_id: str) -> bytes:
        """Serialize experience record to bytes."""
        # Use validated arrays
        state_bytes = experience.state.astype(np.float32).tobytes()
        policy_bytes = experience.policy.astype(np.float32).tobytes()
        value_bytes = struct.pack('<f', experience.value)
        phase_bytes = struct.pack('<B', int(experience.phase))
        timestamp_bytes = struct.pack('<d', experience.timestamp)
        priority_bytes = struct.pack('<f', experience.priority)
        
        # Pad or truncate experience ID to fixed size
        id_bytes = experience_id.encode('utf-8')[:self.id_size].ljust(self.id_size, b'\x00')
        
        return (state_bytes + policy_bytes + value_bytes + 
                phase_bytes + timestamp_bytes + priority_bytes + id_bytes)
    
    def get_experience(self, index: int) -> Optional[ExperienceRecord]:
        """
        Get experience by index with advanced tracking.
        
        Args:
            index: Buffer index (0 to count-1)
            
        Returns:
            Experience record or None if not found
        """
        with self._read_lock:
            header = self._read_header()
            
            if index < 0 or index >= header.count:
                if self.hit_miss_tracker:
                    self.hit_miss_tracker.record_miss()
                return None
            
            # Calculate actual position in circular buffer
            actual_pos = (header.read_position + index) % self.capacity
            record_offset = self.HEADER_SIZE + (actual_pos * self.item_size)
            
            # Read experience data
            try:
                data = self._mmap_buffer[record_offset:record_offset + self.item_size]
                experience = self._deserialize_experience(data)
                
                # Update tracking systems
                if experience.experience_id and self.lru_tracker:
                    self.lru_tracker.register_access(experience.experience_id)
                
                if self.hit_miss_tracker:
                    self.hit_miss_tracker.record_hit()
                
                return experience
                
            except Exception as e:
                logger.error(f"Failed to deserialize experience at index {index}: {e}")
                if self.hit_miss_tracker:
                    self.hit_miss_tracker.record_miss()
                return None

    def get_experience_by_id(self, experience_id: str) -> Optional[ExperienceRecord]:
        """
        Get experience by ID using index for O(1) lookup.
        
        Args:
            experience_id: Unique experience identifier
            
        Returns:
            Experience record or None if not found
        """
        buffer_position = self.index.get_position_by_id(experience_id)
        if buffer_position is None:
            if self.hit_miss_tracker:
                self.hit_miss_tracker.record_miss()
            return None
        
        with self._read_lock:
            record_offset = self.HEADER_SIZE + (buffer_position * self.item_size)
            
            try:
                data = self._mmap_buffer[record_offset:record_offset + self.item_size]
                experience = self._deserialize_experience(data)
                
                # Update LRU tracking
                if self.lru_tracker:
                    self.lru_tracker.register_access(experience_id)
                
                if self.hit_miss_tracker:
                    self.hit_miss_tracker.record_hit()
                
                return experience
                
            except Exception as e:
                logger.error(f"Failed to deserialize experience {experience_id}: {e}")
                if self.hit_miss_tracker:
                    self.hit_miss_tracker.record_miss()
                return None

    def _deserialize_experience(self, data: bytes) -> ExperienceRecord:
        """Deserialize experience record from bytes."""
        offset = 0
        
        # Extract state tensor
        state_bytes = data[offset:offset + self.state_size]
        state = np.frombuffer(state_bytes, dtype=np.float32).reshape(self.state_shape)
        offset += self.state_size
        
        # Extract policy vector
        policy_bytes = data[offset:offset + self.policy_bytes]
        policy = np.frombuffer(policy_bytes, dtype=np.float32)
        offset += self.policy_bytes
        
        # Extract value
        value = struct.unpack('<f', data[offset:offset + self.value_size])[0]
        offset += self.value_size
        
        # Extract phase
        phase = ExperiencePhase(struct.unpack('<B', data[offset:offset + self.phase_size])[0])
        offset += self.phase_size
        
        # Extract timestamp
        timestamp = struct.unpack('<d', data[offset:offset + self.timestamp_size])[0]
        offset += self.timestamp_size
        
        # Extract priority
        priority = struct.unpack('<f', data[offset:offset + self.priority_size])[0]
        offset += self.priority_size
        
        # Extract experience ID
        id_bytes = data[offset:offset + self.id_size]
        experience_id = id_bytes.rstrip(b'\x00').decode('utf-8')
        
        return ExperienceRecord(
            state=state,
            policy=policy,
            value=value,
            phase=phase,
            timestamp=timestamp,
            experience_id=experience_id,
            priority=priority
        )

    def _add_to_tracking(self, experience_id: str, timestamp: float, buffer_position: int, priority: float) -> None:
        """Add experience to all tracking systems."""
        # Add to index
        self.index.add(experience_id, timestamp, buffer_position)
        
        # Add to LRU tracker
        if self.lru_tracker:
            self.lru_tracker.register_access(experience_id)
        
        # Add to priority tracking
        if self.config.enable_prioritized_replay:
            self._priority_weights[experience_id] = priority
            self._priority_sum += priority
            self._max_priority = max(self._max_priority, priority)

    def _remove_from_tracking(self, experience_id: str, buffer_position: int) -> None:
        """Remove experience from all tracking systems."""
        # Remove from index
        self.index.remove(experience_id=experience_id, buffer_position=buffer_position)
        
        # Remove from LRU tracker
        if self.lru_tracker:
            self.lru_tracker.remove_experience(experience_id)
        
        # Remove from priority tracking
        if self.config.enable_prioritized_replay and experience_id in self._priority_weights:
            old_priority = self._priority_weights.pop(experience_id)
            self._priority_sum = max(0, self._priority_sum - old_priority)

    def _get_experience_id_at_position(self, buffer_position: int) -> Optional[str]:
        """Get experience ID at a specific buffer position."""
        record_offset = self.HEADER_SIZE + (buffer_position * self.item_size)
        
        try:
            # Read only the ID portion (at the end of the record)
            id_offset = record_offset + self.item_size - self.id_size
            id_bytes = self._mmap_buffer[id_offset:id_offset + self.id_size]
            experience_id = id_bytes.rstrip(b'\x00').decode('utf-8')
            return experience_id if experience_id else None
        except Exception:
            return None

    def _evict_experiences(self, bytes_to_free: int) -> int:
        """
        Evict experiences based on configured policy.
        
        Args:
            bytes_to_free: Target number of bytes to free
            
        Returns:
            Number of experiences evicted
        """
        if bytes_to_free <= 0:
            return 0
            
        experiences_to_evict = max(1, bytes_to_free // self.item_size)
        evicted_count = 0
        
        if self.config.eviction_policy == "lru" and self.lru_tracker:
            # LRU eviction
            candidates = self.lru_tracker.get_lru_candidates(experiences_to_evict)
            for exp_id in candidates:
                if self._evict_experience_by_id(exp_id):
                    evicted_count += 1
                    if evicted_count * self.item_size >= bytes_to_free:
                        break
        
        elif self.config.eviction_policy == "fifo":
            # FIFO eviction (oldest first)
            header = self._read_header()
            for i in range(min(experiences_to_evict, header.count)):
                pos = (header.read_position + i) % self.capacity
                exp_id = self._get_experience_id_at_position(pos)
                if exp_id and self._evict_experience_by_id(exp_id):
                    evicted_count += 1
                    if evicted_count * self.item_size >= bytes_to_free:
                        break
        
        elif self.config.eviction_policy == "random":
            # Random eviction
            header = self._read_header()
            import random
            positions = list(range(header.count))
            random.shuffle(positions)
            
            for i in positions[:experiences_to_evict]:
                pos = (header.read_position + i) % self.capacity
                exp_id = self._get_experience_id_at_position(pos)
                if exp_id and self._evict_experience_by_id(exp_id):
                    evicted_count += 1
                    if evicted_count * self.item_size >= bytes_to_free:
                        break
        
        logger.debug(f"Evicted {evicted_count} experiences using {self.config.eviction_policy} policy")
        return evicted_count

    def _evict_experience_by_id(self, experience_id: str) -> bool:
        """
        Evict a specific experience by ID.
        
        Args:
            experience_id: ID of experience to evict
            
        Returns:
            True if evicted successfully
        """
        buffer_position = self.index.get_position_by_id(experience_id)
        if buffer_position is None:
            return False
        
        try:
            # Remove from tracking systems
            self._remove_from_tracking(experience_id, buffer_position)
            
            # Update memory budget
            self.budget_manager.register_removal(self.item_size)
            
            # Zero out the memory location (optional, for security)
            record_offset = self.HEADER_SIZE + (buffer_position * self.item_size)
            self._mmap_buffer[record_offset:record_offset + self.item_size] = b'\x00' * self.item_size
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to evict experience {experience_id}: {e}")
            return False

    def sample_batch(self, batch_size: int, phase_weights: Optional[Dict[str, float]] = None) -> List[ExperienceRecord]:
        """
        Sample random batch of experiences with optional phase weighting.
        
        Args:
            batch_size: Number of experiences to sample
            phase_weights: Optional weights for different game phases
            
        Returns:
            List of sampled experience records
        """
        if not self._acquire_read_lock():
            logger.warning("Failed to acquire read lock for sample_batch")
            return []
        
        try:
            header = self._read_header()
            
            if header.count == 0:
                return []
            
            # Get all available indices
            available_count = min(header.count, batch_size)
            
            if phase_weights is None:
                # Simple random sampling
                indices = np.random.choice(header.count, available_count, replace=False)
            else:
                # Weighted sampling by phase
                indices = self._weighted_sample_by_phase(
                    header.count, available_count, phase_weights
                )
            
            # Retrieve experiences
            experiences = []
            for idx in indices:
                exp = self.get_experience(idx)
                if exp is not None:
                    experiences.append(exp)
            
            return experiences
            
        except Exception as e:
            logger.error(f"Failed to sample batch: {e}")
            return []
        finally:
            self._read_lock.release()
    
    def _weighted_sample_by_phase(
        self, 
        total_count: int, 
        sample_size: int, 
        phase_weights: Dict[str, float]
    ) -> np.ndarray:
        """Sample indices with phase-based weighting and recency bias."""
        if total_count == 0:
            return np.array([], dtype=int)
            
        # Calculate phase distribution in current buffer
        phase_counts = defaultdict(int)
        timestamps = []
        phases = []
        
        for i in range(total_count):
            actual_pos = (self._read_header().read_position + i) % self.capacity
            record_offset = self.HEADER_SIZE + (actual_pos * self.item_size)
            
            # Extract phase and timestamp from record
            phase_offset = record_offset + self.state_size + self.policy_bytes + self.value_size
            phase_byte = self._mmap_buffer[phase_offset:phase_offset + 1]
            phase = ExperiencePhase(struct.unpack('<B', phase_byte)[0])
            
            timestamp_offset = phase_offset + self.phase_size
            timestamp_bytes = self._mmap_buffer[timestamp_offset:timestamp_offset + self.timestamp_size]
            timestamp = struct.unpack('<d', timestamp_bytes)[0]
            
            phase_counts[phase] += 1
            phases.append(phase)
            timestamps.append(timestamp)
        
        # Calculate inverse frequency weights to balance phases
        total = sum(phase_counts.values())
        inverse_freq_weights = {phase: total / max(count, 1) for phase, count in phase_counts.items()}
        
        # Apply user-provided weights
        combined_weights = {}
        for phase in ExperiencePhase:
            phase_name = phase.name
            user_weight = phase_weights.get(phase_name, phase_weights.get(phase.name.lower(), 1.0))
            freq_weight = inverse_freq_weights.get(phase, 1.0)
            combined_weights[phase] = user_weight * freq_weight
        
        # Calculate recency weights (more recent experiences get higher weight)
        if timestamps:
            max_timestamp = max(timestamps)
            min_timestamp = min(timestamps)
            timestamp_range = max_timestamp - min_timestamp if max_timestamp > min_timestamp else 1.0
            
            recency_weights = [
                1.0 + ((ts - min_timestamp) / timestamp_range) * 0.5  # Up to 50% bonus for newest
                for ts in timestamps
            ]
        else:
            recency_weights = [1.0] * total_count
        
        # Combine all weights
        final_weights = np.array([
            combined_weights[phases[i]] * recency_weights[i] * math.pow(recency_weights[i], 1/self._sampling_temperature)
            for i in range(total_count)
        ], dtype=np.float32)
        
        # Normalize and sample
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum()
            indices = np.random.choice(total_count, sample_size, replace=False, p=final_weights)
        else:
            indices = np.random.choice(total_count, sample_size, replace=False)
            
        return indices

    def sample_batch_optimized(
        self, 
        batch_size: int, 
        phase_weights: Optional[Dict[str, float]] = None,
        uniform: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimized batch sampling that returns numpy arrays directly.
        
        Args:
            batch_size: Number of experiences to sample
            phase_weights: Optional weights for different game phases
            uniform: If True, use uniform sampling; if False, use weighted sampling
            
        Returns:
            Tuple of (states, policies, values, phases) as numpy arrays
        """
        if not self._acquire_read_lock():
            logger.warning("Failed to acquire read lock for sample_batch_optimized")
            raise BufferLockTimeoutError("Cannot acquire read lock for sampling")
        
        try:
            header = self._read_header()
            
            if header.count == 0:
                raise BufferEmptyError("Cannot sample from empty buffer")
            
            # Select indices
            available_count = min(header.count, batch_size)
            
            if uniform or phase_weights is None:
                # Simple random sampling
                indices = np.random.choice(header.count, available_count, replace=False)
            else:
                # Weighted sampling by phase
                indices = self._weighted_sample_by_phase(
                    header.count, available_count, phase_weights
                )
            
            # Pre-allocate arrays for batch
            states = np.zeros((available_count,) + self.state_shape, dtype=np.float32)
            policies = np.zeros((available_count, self.policy_size), dtype=np.float32)
            values = np.zeros(available_count, dtype=np.float32)
            phases = np.zeros(available_count, dtype=np.uint8)
            
            # Batch read experiences - group by memory pages for efficiency
            page_size = 4096  # Typical system page size
            experiences_per_page = max(1, page_size // self.item_size)
            
            # Group indices by page to minimize page faults
            page_groups = defaultdict(list)
            for i, idx in enumerate(indices):
                actual_pos = (header.read_position + idx) % self.capacity
                page_num = actual_pos // experiences_per_page
                page_groups[page_num].append((i, actual_pos))
            
            # Read experiences page by page for better cache locality
            for page_indices in page_groups.values():
                for result_idx, actual_pos in sorted(page_indices, key=lambda x: x[1]):
                    record_offset = self.HEADER_SIZE + (actual_pos * self.item_size)
                    record_bytes = self._mmap_buffer[record_offset:record_offset + self.item_size]
                    
                    # Deserialize directly into pre-allocated arrays
                    offset = 0
                    
                    # State
                    state_bytes = record_bytes[offset:offset + self.state_size]
                    states[result_idx] = np.frombuffer(state_bytes, dtype=np.float32).reshape(self.state_shape)
                    offset += self.state_size
                    
                    # Policy
                    policy_bytes = record_bytes[offset:offset + self.policy_bytes]
                    policies[result_idx] = np.frombuffer(policy_bytes, dtype=np.float32)
                    offset += self.policy_bytes
                    
                    # Value
                    values[result_idx] = struct.unpack('<f', record_bytes[offset:offset + self.value_size])[0]
                    offset += self.value_size
                    
                    # Phase
                    phases[result_idx] = struct.unpack('<B', record_bytes[offset:offset + self.phase_size])[0]
            
            return states, policies, values, phases
            
        finally:
            self._read_lock.release()

    def add_experiences_batch(self, experiences: List[ExperienceRecord]) -> int:
        """
        Add multiple experiences efficiently in a single operation.
        
        Args:
            experiences: List of experience records to add
            
        Returns:
            Number of experiences successfully added
        """
        if not experiences:
            return 0
            
        # Validate all experiences before acquiring lock
        for exp in experiences:
            self._validate_experience(exp)
        
        if not self._acquire_write_lock():
            logger.warning("Failed to acquire write lock for add_experiences_batch")
            return 0
        
        try:
            header = self._read_header()
            added_count = 0
            
            for experience in experiences:
                write_pos = header.write_position
                
                # Calculate byte offset for this record
                record_offset = self.HEADER_SIZE + (write_pos * self.item_size)
                
                # Serialize and write experience
                serialized = self._serialize_experience(experience)
                self._mmap_buffer[record_offset:record_offset + len(serialized)] = serialized
                
                # Update positions
                new_write_pos = (write_pos + 1) % self.capacity
                new_count = min(header.count + 1, self.capacity)
                
                # If buffer is full, advance read position
                new_read_pos = header.read_position
                if header.count == self.capacity:
                    new_read_pos = (header.read_position + 1) % self.capacity
                
                # Update header for next iteration
                header = header._replace(
                    write_position=new_write_pos,
                    read_position=new_read_pos,
                    count=new_count
                )
                
                added_count += 1
            
            # Write final header state
            self._write_header(
                write_position=header.write_position,
                read_position=header.read_position,
                count=header.count
            )
            
            return added_count
            
        except Exception as e:
            logger.error(f"Failed to add batch experiences: {e}")
            return added_count
        finally:
            self._write_lock.release()

    def get_buffer_statistics(self) -> BufferStatistics:
        """Get comprehensive buffer statistics including advanced tracking metrics."""
        with self._read_lock:
            header = self._read_header()
            
            # Basic buffer stats
            capacity = self.capacity
            count = header.count
            utilization = count / capacity if capacity > 0 else 0.0
            is_full = count >= capacity
            
            # Memory stats
            memory_usage_bytes = self.total_size
            budget_stats = self.budget_manager.get_usage_stats()
            memory_budget_bytes = budget_stats["max_bytes"]
            memory_budget_utilization = budget_stats["utilization"]
            
            # Experience size
            experience_size_bytes = self.item_size
            
            # Phase distribution
            phase_distribution = defaultdict(int)
            oldest_timestamp = float('inf')
            newest_timestamp = 0.0
            
            if count > 0:
                # Sample experiences to get phase distribution and timestamps
                sample_size = min(1000, count)  # Sample up to 1000 for efficiency
                step = max(1, count // sample_size)
                
                for i in range(0, count, step):
                    try:
                        experience = self.get_experience(i)
                        if experience:
                            phase_name = experience.phase.name
                            phase_distribution[phase_name] += 1
                            oldest_timestamp = min(oldest_timestamp, experience.timestamp)
                            newest_timestamp = max(newest_timestamp, experience.timestamp)
                    except Exception as e:
                        logger.debug(f"Failed to sample experience at index {i}: {e}")
                        continue
                
                # Normalize phase distribution based on actual sample
                total_sampled = sum(phase_distribution.values())
                if total_sampled > 0:
                    for phase in phase_distribution:
                        phase_distribution[phase] = int(
                            (phase_distribution[phase] / total_sampled) * count
                        )
            
            if oldest_timestamp == float('inf'):
                oldest_timestamp = 0.0
            
            # Hit/miss rates
            hit_rate = 0.0
            miss_rate = 0.0
            if self.hit_miss_tracker:
                hit_miss_stats = self.hit_miss_tracker.get_total_stats()
                hit_rate = hit_miss_stats["recent_hit_rate"]
                miss_rate = hit_miss_stats["recent_miss_rate"]
            
            # Eviction counts
            eviction_count = budget_stats["total_evictions"]
            lru_evictions = eviction_count  # All evictions are currently LRU-based
            
            # Total operations
            total_additions = budget_stats["total_additions"]
            total_accesses = 0
            if self.hit_miss_tracker:
                total_accesses = self.hit_miss_tracker._total_operations
            
            return BufferStatistics(
                capacity=capacity,
                count=count,
                utilization=utilization,
                write_position=header.write_position,
                read_position=header.read_position,
                is_full=is_full,
                memory_usage_bytes=memory_usage_bytes,
                memory_budget_bytes=memory_budget_bytes,
                memory_budget_utilization=memory_budget_utilization,
                experience_size_bytes=experience_size_bytes,
                phase_distribution=dict(phase_distribution),
                oldest_timestamp=oldest_timestamp,
                newest_timestamp=newest_timestamp,
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                eviction_count=eviction_count,
                lru_evictions=lru_evictions,
                total_additions=total_additions,
                total_accesses=total_accesses
            )

    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics from all tracking systems."""
        stats = {}
        
        # Basic buffer statistics
        basic_stats = self.get_buffer_statistics()
        stats["buffer"] = {
            "capacity": basic_stats.capacity,
            "count": basic_stats.count,
            "utilization": basic_stats.utilization,
            "memory_budget_utilization": basic_stats.memory_budget_utilization,
            "hit_rate": basic_stats.hit_rate,
            "miss_rate": basic_stats.miss_rate
        }
        
        # Memory budget statistics
        stats["memory_budget"] = self.budget_manager.get_usage_stats()
        
        # LRU tracker statistics
        if self.lru_tracker:
            stats["lru_tracker"] = self.lru_tracker.get_stats()
        else:
            stats["lru_tracker"] = {"enabled": False}
        
        # Index statistics
        stats["index"] = self.index.get_stats()
        
        # Hit/miss tracker statistics
        if self.hit_miss_tracker:
            stats["hit_miss_tracker"] = self.hit_miss_tracker.get_total_stats()
        else:
            stats["hit_miss_tracker"] = {"enabled": False}
        
        # Prioritized replay statistics
        stats["prioritized_replay"] = {
            "enabled": self.config.enable_prioritized_replay,
            "priority_sum": self._priority_sum,
            "max_priority": self._max_priority,
            "tracked_priorities": len(self._priority_weights)
        }
        
        # Configuration
        stats["config"] = {
            "eviction_policy": self.config.eviction_policy,
            "lru_tracking_enabled": self.config.lru_tracking_enabled,
            "enable_id_indexing": self.config.enable_id_indexing,
            "enable_timestamp_indexing": self.config.enable_timestamp_indexing,
            "enable_prioritized_replay": self.config.enable_prioritized_replay,
            "enable_hit_miss_tracking": self.config.enable_hit_miss_tracking,
            "max_memory_mb": self.config.max_memory_bytes // 1024 // 1024
        }
        
        return stats

    def set_sampling_temperature(self, temperature: float) -> None:
        """
        Set sampling temperature for weighted sampling.
        
        Args:
            temperature: Controls sampling uniformity (1.0 = normal, >1.0 = more uniform, <1.0 = more peaked)
        """
        self._sampling_temperature = max(0.1, temperature)

    def set_phase_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for different game phases during sampling.
        
        Args:
            weights: Dictionary mapping phase names to weights
        """
        self._phase_weights = weights.copy()

    def get_oldest_experiences(self, count: int) -> List[ExperienceRecord]:
        """
        Get the oldest experiences in the buffer.
        
        Args:
            count: Number of oldest experiences to retrieve
            
        Returns:
            List of oldest experience records
        """
        if not self._acquire_read_lock():
            logger.warning("Failed to acquire read lock for get_oldest_experiences")
            return []
        
        try:
            header = self._read_header()
            
            if header.count == 0:
                return []
            
            actual_count = min(count, header.count)
            experiences = []
            
            for i in range(actual_count):
                exp = self.get_experience(i)  # This already uses the right indexing
                if exp is not None:
                    experiences.append(exp)
            
            return experiences
            
        finally:
            self._read_lock.release()

    def get_newest_experiences(self, count: int) -> List[ExperienceRecord]:
        """
        Get the newest experiences in the buffer.
        
        Args:
            count: Number of newest experiences to retrieve
            
        Returns:
            List of newest experience records
        """
        if not self._acquire_read_lock():
            logger.warning("Failed to acquire read lock for get_newest_experiences")
            return []
        
        try:
            header = self._read_header()
            
            if header.count == 0:
                return []
            
            actual_count = min(count, header.count)
            experiences = []
            
            # Start from the most recent experiences
            for i in range(actual_count):
                idx = header.count - 1 - i  # Count backwards from newest
                exp = self.get_experience(idx)
                if exp is not None:
                    experiences.append(exp)
            
            return experiences
            
        finally:
            self._read_lock.release()

    def compact_buffer(self) -> bool:
        """
        Compact the buffer by removing gaps and optimizing memory layout.
        This is useful for defragmentation after many overwrites.
        
        Returns:
            True if compaction was successful
        """
        if not self._acquire_write_lock():
            logger.warning("Failed to acquire write lock for compact_buffer")
            return False
        
        try:
            header = self._read_header()
            
            if header.count == 0:
                return True
            
            # Only compact if we're not using the full capacity efficiently
            if header.count >= self.capacity * 0.8:
                return True  # No need to compact
            
            logger.info(f"Compacting buffer with {header.count} experiences")
            
            # Read all experiences in order
            experiences = []
            for i in range(header.count):
                exp = self.get_experience(i)
                if exp is not None:
                    experiences.append(exp)
            
            # Clear the buffer
            self._write_header(
                write_position=0,
                read_position=0,
                count=0
            )
            
            # Add experiences back sequentially
            for exp in experiences:
                if not self.add_experience(exp):
                    logger.error("Failed to add experience during compaction")
                    return False
            
            logger.info(f"Buffer compaction completed, {len(experiences)} experiences restored")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compact buffer: {e}")
            return False
        finally:
            self._write_lock.release()
    
    def size(self) -> int:
        """Get current number of experiences in buffer."""
        try:
            header = self._read_header()
            return header.count
        except:
            return 0
    
    def capacity_used(self) -> float:
        """Get fraction of buffer capacity used."""
        try:
            header = self._read_header()
            return header.count / self.capacity
        except:
            return 0.0
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        if not self._acquire_write_lock():
            raise BufferLockTimeoutError("Failed to acquire write lock for clear")
        
        try:
            self._write_header(
                write_position=0,
                read_position=0,
                count=0
            )
            logger.info("Cleared experience buffer")
        finally:
            self._write_lock.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        try:
            header = self._read_header()
            return {
                'count': header.count,
                'capacity': self.capacity,
                'capacity_used': header.count / self.capacity,
                'write_position': header.write_position,
                'read_position': header.read_position,
                'item_size': self.item_size,
                'total_size_mb': self.total_size / (1024 * 1024),
                'file_path': str(self.file_path)
            }
        except:
            return {}
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Close memory mapping
            if self._mmap_buffer is not None:
                self._mmap_buffer.close()
                self._mmap_buffer = None
            
            # Platform-specific cleanup
            if sys.platform == 'win32' and self._win_handles:
                file_handle, mapping_handle, view_ptr = self._win_handles
                if view_ptr:
                    UnmapViewOfFile(view_ptr)
                if mapping_handle:
                    CloseHandle(mapping_handle)
                if file_handle:
                    CloseHandle(file_handle)
                self._win_handles = None
            elif self._file_handle is not None:
                os.close(self._file_handle)
                self._file_handle = None
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
    
    def __del__(self):
        """Destructor."""
        self._cleanup() 