"""Shared memory implementation for inter-process communication.

This module provides platform-agnostic shared memory regions for efficient
data sharing between processes without copying overhead.
"""

import os
import sys
import mmap
import struct
import threading
import tempfile
import logging
import time
import uuid
import weakref
from pathlib import Path
from typing import Optional, Union, Any, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SharedMemoryError(Exception):
    """Base exception for shared memory operations."""
    pass


class SharedMemoryAllocationError(SharedMemoryError):
    """Raised when shared memory allocation fails."""
    pass


class SharedMemoryAccessError(SharedMemoryError):
    """Raised when accessing shared memory fails."""
    pass


class SharedMemoryPlatform(Enum):
    """Supported platforms for shared memory."""
    UNIX = "unix"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


@dataclass
class SharedMemoryConfig:
    """Configuration for shared memory regions.
    
    Args:
        default_size: Default size for new shared memory regions in bytes
        alignment: Memory alignment boundary (must be power of 2)
        timeout_seconds: Timeout for acquiring shared memory regions
        cleanup_interval: Interval for cleaning up orphaned regions (seconds)
        enable_recovery: Whether to enable automatic recovery from corruption
        max_regions: Maximum number of shared memory regions to track
        temp_directory: Directory for shared memory files (Unix only)
    """
    default_size: int = 1024 * 1024  # 1MB default
    alignment: int = 64  # 64-byte alignment for cache line optimization
    timeout_seconds: float = 30.0
    cleanup_interval: float = 300.0  # 5 minutes
    enable_recovery: bool = True
    max_regions: int = 1000
    temp_directory: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.default_size <= 0:
            raise ValueError("default_size must be positive")
        
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError("alignment must be a positive power of 2")
            
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
            
        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be positive")
            
        if self.max_regions <= 0:
            raise ValueError("max_regions must be positive")


class SharedMemoryStatistics:
    """Statistics tracking for shared memory usage."""
    
    def __init__(self):
        self.regions_created = 0
        self.regions_destroyed = 0
        self.total_bytes_allocated = 0
        self.peak_bytes_allocated = 0
        self.current_bytes_allocated = 0
        self.allocation_failures = 0
        self.access_failures = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.orphaned_regions_cleaned = 0
        self.reference_count_errors = 0
        
        # Performance metrics
        self.allocation_times = []
        self.access_times = []
        
        self._lock = threading.Lock()
        
    def record_allocation(self, size: int, duration: float):
        """Record a successful allocation."""
        with self._lock:
            self.regions_created += 1
            self.total_bytes_allocated += size
            self.current_bytes_allocated += size
            self.peak_bytes_allocated = max(self.peak_bytes_allocated, self.current_bytes_allocated)
            self.allocation_times.append(duration)
            
    def record_deallocation(self, size: int):
        """Record a deallocation."""
        with self._lock:
            self.regions_destroyed += 1
            self.current_bytes_allocated = max(0, self.current_bytes_allocated - size)
            
    def record_allocation_failure(self):
        """Record an allocation failure."""
        with self._lock:
            self.allocation_failures += 1
            
    def record_access_failure(self):
        """Record an access failure."""
        with self._lock:
            self.access_failures += 1
            
    def record_recovery_attempt(self, successful: bool):
        """Record a recovery attempt."""
        with self._lock:
            self.recovery_attempts += 1
            if successful:
                self.successful_recoveries += 1
                
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of statistics."""
        with self._lock:
            avg_allocation_time = (
                sum(self.allocation_times) / len(self.allocation_times)
                if self.allocation_times else 0.0
            )
            
            return {
                'regions_created': self.regions_created,
                'regions_destroyed': self.regions_destroyed,
                'regions_active': self.regions_created - self.regions_destroyed,
                'total_bytes_allocated': self.total_bytes_allocated,
                'peak_bytes_allocated': self.peak_bytes_allocated,
                'current_bytes_allocated': self.current_bytes_allocated,
                'allocation_failures': self.allocation_failures,
                'access_failures': self.access_failures,
                'avg_allocation_time': avg_allocation_time,
                'recovery_success_rate': (
                    self.successful_recoveries / self.recovery_attempts * 100
                    if self.recovery_attempts > 0 else 0.0
                )
            }


class SharedMemoryRegion:
    """Cross-platform shared memory region with reference counting.
    
    This class provides a platform-agnostic interface for creating and managing
    shared memory regions that can be accessed by multiple processes.
    """
    
    # Memory layout constants
    HEADER_SIZE = 128  # Size of metadata header
    MAGIC_NUMBER = 0x53484D45  # "SHME" in little-endian
    VERSION = 1
    
    # Header layout: [magic:4][version:4][size:8][ref_count:4][created:8][last_access:8][alignment:4][reserved:88]
    HEADER_FORMAT = '<IIQIQQI88x'
    
    def __init__(self, name: str, size: int, config: SharedMemoryConfig, create: bool = True):
        """Initialize shared memory region.
        
        Args:
            name: Unique name for the shared memory region
            size: Size in bytes (will be aligned to config.alignment)
            config: Configuration for shared memory behavior
            create: Whether to create the region (True) or attach to existing (False)
        """
        self.name = name
        self.config = config
        self.platform = self._detect_platform()
        
        # Align size to configured boundary
        self.size = self._align_size(size + self.HEADER_SIZE)
        self.user_size = self.size - self.HEADER_SIZE
        
        # Platform-specific state
        self._file_handle = None
        self._mapping_handle = None
        self._mapped_view = None
        self._temp_file_path = None
        
        # Reference counting and synchronization
        self._local_ref_count = 0
        self._lock = threading.RLock()
        
        # Initialize the shared memory region
        try:
            if create:
                self._create_region()
            else:
                self._attach_region()
        except Exception as e:
            self._cleanup_resources()
            raise SharedMemoryAllocationError(f"Failed to initialize shared memory region '{name}': {e}")
            
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.close()
        except Exception:
            pass
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    @staticmethod
    def _detect_platform() -> SharedMemoryPlatform:
        """Detect the current platform for shared memory implementation."""
        if sys.platform.startswith('win'):
            return SharedMemoryPlatform.WINDOWS
        elif sys.platform in ('linux', 'darwin', 'freebsd', 'openbsd', 'netbsd'):
            return SharedMemoryPlatform.UNIX
        else:
            logger.warning(f"Unknown platform: {sys.platform}, using Unix implementation")
            return SharedMemoryPlatform.UNIX
            
    def _align_size(self, size: int) -> int:
        """Align size to configured boundary."""
        alignment = self.config.alignment
        return (size + alignment - 1) & ~(alignment - 1)
        
    def _create_region(self):
        """Create a new shared memory region."""
        start_time = time.time()
        
        if self.platform == SharedMemoryPlatform.UNIX:
            self._create_unix_region()
        elif self.platform == SharedMemoryPlatform.WINDOWS:
            self._create_windows_region()
        else:
            raise SharedMemoryError(f"Unsupported platform: {self.platform}")
            
        # Initialize header
        self._write_header(
            magic=self.MAGIC_NUMBER,
            version=self.VERSION,
            size=self.size,
            ref_count=1,
            created_time=int(time.time()),
            last_access_time=int(time.time()),
            alignment=self.config.alignment
        )
        
        duration = time.time() - start_time
        logger.debug(f"Created shared memory region '{self.name}' ({self.size} bytes) in {duration:.3f}s")
        
    def _attach_region(self):
        """Attach to an existing shared memory region."""
        start_time = time.time()
        
        if self.platform == SharedMemoryPlatform.UNIX:
            self._attach_unix_region()
        elif self.platform == SharedMemoryPlatform.WINDOWS:
            self._attach_windows_region()
        else:
            raise SharedMemoryError(f"Unsupported platform: {self.platform}")
            
        # Validate header and increment reference count
        self._validate_header()
        self._increment_ref_count()
        
        duration = time.time() - start_time
        logger.debug(f"Attached to shared memory region '{self.name}' in {duration:.3f}s")
        
    def _create_unix_region(self):
        """Create shared memory region using Unix mmap."""
        try:
            # Use /dev/shm if available (Linux), otherwise use temp directory
            if Path('/dev/shm').exists() and not self.config.temp_directory:
                shm_path = Path('/dev/shm') / f"yinsh_shm_{self.name}"
            else:
                temp_dir = self.config.temp_directory or tempfile.gettempdir()
                shm_path = Path(temp_dir) / f"yinsh_shm_{self.name}"
                
            self._temp_file_path = str(shm_path)
            
            # Create and truncate file
            with open(self._temp_file_path, 'w+b') as f:
                f.write(b'\x00' * self.size)
                f.flush()
                os.fsync(f.fileno())
                
            # Open file for mmap
            self._file_handle = open(self._temp_file_path, 'r+b')
            
            # Create memory mapping
            self._mapped_view = mmap.mmap(
                self._file_handle.fileno(),
                self.size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
        except Exception as e:
            self._cleanup_resources()
            raise SharedMemoryAllocationError(f"Unix mmap creation failed: {e}")
            
    def _attach_unix_region(self):
        """Attach to existing Unix shared memory region."""
        try:
            # Find the shared memory file
            if Path('/dev/shm').exists() and not self.config.temp_directory:
                shm_path = Path('/dev/shm') / f"yinsh_shm_{self.name}"
            else:
                temp_dir = self.config.temp_directory or tempfile.gettempdir()
                shm_path = Path(temp_dir) / f"yinsh_shm_{self.name}"
                
            self._temp_file_path = str(shm_path)
            
            if not shm_path.exists():
                raise SharedMemoryError(f"Shared memory region '{self.name}' does not exist")
                
            # Open existing file
            self._file_handle = open(self._temp_file_path, 'r+b')
            
            # Get file size
            file_size = os.path.getsize(self._temp_file_path)
            if file_size != self.size:
                # Adjust size to match existing region
                self.size = file_size
                self.user_size = self.size - self.HEADER_SIZE
                
            # Create memory mapping
            self._mapped_view = mmap.mmap(
                self._file_handle.fileno(),
                self.size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
        except Exception as e:
            self._cleanup_resources()
            raise SharedMemoryAccessError(f"Unix mmap attachment failed: {e}")
            
    def _create_windows_region(self):
        """Create shared memory region using Windows CreateFileMapping."""
        try:
            import ctypes
            from ctypes import wintypes
            
            # Windows API constants
            INVALID_HANDLE_VALUE = -1
            FILE_MAP_ALL_ACCESS = 0x000F001F
            PAGE_READWRITE = 0x04
            
            # Create file mapping
            kernel32 = ctypes.windll.kernel32
            
            self._mapping_handle = kernel32.CreateFileMappingW(
                INVALID_HANDLE_VALUE,  # Use page file
                None,                  # Default security
                PAGE_READWRITE,        # Read/write access
                0,                     # High-order DWORD of size
                self.size,             # Low-order DWORD of size
                f"yinsh_shm_{self.name}"  # Object name
            )
            
            if not self._mapping_handle:
                error = kernel32.GetLastError()
                raise SharedMemoryAllocationError(f"CreateFileMapping failed with error {error}")
                
            # Map view of file
            self._mapped_view = kernel32.MapViewOfFile(
                self._mapping_handle,
                FILE_MAP_ALL_ACCESS,
                0,  # High-order DWORD of offset
                0,  # Low-order DWORD of offset
                self.size
            )
            
            if not self._mapped_view:
                error = kernel32.GetLastError()
                raise SharedMemoryAllocationError(f"MapViewOfFile failed with error {error}")
                
        except ImportError:
            # Fallback to mmap with temporary file on Windows
            self._create_unix_region()
        except Exception as e:
            self._cleanup_resources()
            raise SharedMemoryAllocationError(f"Windows shared memory creation failed: {e}")
            
    def _attach_windows_region(self):
        """Attach to existing Windows shared memory region."""
        try:
            import ctypes
            
            # Windows API constants
            FILE_MAP_ALL_ACCESS = 0x000F001F
            
            kernel32 = ctypes.windll.kernel32
            
            # Open existing file mapping
            self._mapping_handle = kernel32.OpenFileMappingW(
                FILE_MAP_ALL_ACCESS,
                False,  # Do not inherit handle
                f"yinsh_shm_{self.name}"
            )
            
            if not self._mapping_handle:
                error = kernel32.GetLastError()
                raise SharedMemoryAccessError(f"OpenFileMapping failed with error {error}")
                
            # Map view of file
            self._mapped_view = kernel32.MapViewOfFile(
                self._mapping_handle,
                FILE_MAP_ALL_ACCESS,
                0,  # High-order DWORD of offset
                0,  # Low-order DWORD of offset
                0   # Map entire file
            )
            
            if not self._mapped_view:
                error = kernel32.GetLastError()
                raise SharedMemoryAccessError(f"MapViewOfFile failed with error {error}")
                
        except ImportError:
            # Fallback to mmap attachment
            self._attach_unix_region()
        except Exception as e:
            self._cleanup_resources()
            raise SharedMemoryAccessError(f"Windows shared memory attachment failed: {e}")
            
    def _write_header(self, magic: int, version: int, size: int, ref_count: int,
                     created_time: int, last_access_time: int, alignment: int):
        """Write header metadata to shared memory region."""
        with self._lock:
            if not self._mapped_view:
                raise SharedMemoryAccessError("Shared memory region not mapped")
                
            header_data = struct.pack(
                self.HEADER_FORMAT,
                magic, version, size, ref_count,
                created_time, last_access_time, alignment
            )
            
            if self.platform == SharedMemoryPlatform.WINDOWS and hasattr(self._mapped_view, '__setitem__'):
                # Windows ctypes pointer approach
                ctypes.memmove(self._mapped_view, header_data, len(header_data))
            else:
                # Unix mmap approach
                self._mapped_view[:len(header_data)] = header_data
                
    def _read_header(self) -> Tuple[int, int, int, int, int, int, int]:
        """Read header metadata from shared memory region."""
        with self._lock:
            if not self._mapped_view:
                raise SharedMemoryAccessError("Shared memory region not mapped")
                
            header_size = struct.calcsize(self.HEADER_FORMAT)
            
            if self.platform == SharedMemoryPlatform.WINDOWS and hasattr(self._mapped_view, '__getitem__'):
                # Windows ctypes pointer approach
                import ctypes
                header_data = (ctypes.c_char * header_size).from_address(self._mapped_view)
                header_bytes = bytes(header_data)
            else:
                # Unix mmap approach
                header_bytes = self._mapped_view[:header_size]
                
            return struct.unpack(self.HEADER_FORMAT, header_bytes)
            
    def _validate_header(self):
        """Validate shared memory region header."""
        try:
            magic, version, size, ref_count, created_time, last_access_time, alignment = self._read_header()
            
            if magic != self.MAGIC_NUMBER:
                raise SharedMemoryError(f"Invalid magic number: expected {self.MAGIC_NUMBER}, got {magic}")
                
            if version != self.VERSION:
                raise SharedMemoryError(f"Unsupported version: expected {self.VERSION}, got {version}")
                
            if size != self.size:
                logger.warning(f"Size mismatch: expected {self.size}, found {size}")
                
        except struct.error as e:
            raise SharedMemoryError(f"Header validation failed: {e}")
            
    def _increment_ref_count(self) -> int:
        """Atomically increment reference count."""
        with self._lock:
            try:
                # Read current header
                magic, version, size, ref_count, created_time, last_access_time, alignment = self._read_header()
                
                # Increment reference count
                new_ref_count = ref_count + 1
                
                # Write updated header
                self._write_header(magic, version, size, new_ref_count, created_time, int(time.time()), alignment)
                
                self._local_ref_count += 1
                return new_ref_count
                
            except Exception as e:
                raise SharedMemoryAccessError(f"Failed to increment reference count: {e}")
                
    def _decrement_ref_count(self) -> int:
        """Atomically decrement reference count."""
        with self._lock:
            try:
                # Read current header
                magic, version, size, ref_count, created_time, last_access_time, alignment = self._read_header()
                
                # Decrement reference count
                new_ref_count = max(0, ref_count - 1)
                
                # Write updated header
                self._write_header(magic, version, size, new_ref_count, created_time, int(time.time()), alignment)
                
                self._local_ref_count = max(0, self._local_ref_count - 1)
                return new_ref_count
                
            except Exception as e:
                logger.error(f"Failed to decrement reference count: {e}")
                return 0
                
    def _cleanup_resources(self):
        """Clean up platform-specific resources."""
        try:
            if self.platform == SharedMemoryPlatform.WINDOWS:
                if self._mapped_view:
                    try:
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        kernel32.UnmapViewOfFile(self._mapped_view)
                    except Exception:
                        pass
                    self._mapped_view = None
                    
                if self._mapping_handle:
                    try:
                        import ctypes
                        kernel32 = ctypes.windll.kernel32
                        kernel32.CloseHandle(self._mapping_handle)
                    except Exception:
                        pass
                    self._mapping_handle = None
            else:
                # Unix cleanup
                if self._mapped_view:
                    try:
                        self._mapped_view.close()
                    except Exception:
                        pass
                    self._mapped_view = None
                    
                if self._file_handle:
                    try:
                        self._file_handle.close()
                    except Exception:
                        pass
                    self._file_handle = None
                    
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
            
    def write_bytes(self, offset: int, data: bytes):
        """Write bytes to shared memory region.
        
        Args:
            offset: Offset from start of user data region (after header)
            data: Bytes to write
        """
        with self._lock:
            if not self._mapped_view:
                raise SharedMemoryAccessError("Shared memory region not mapped")
                
            if offset < 0 or offset + len(data) > self.user_size:
                raise ValueError(f"Write would exceed region bounds: offset={offset}, size={len(data)}, max={self.user_size}")
                
            actual_offset = self.HEADER_SIZE + offset
            
            if self.platform == SharedMemoryPlatform.WINDOWS and hasattr(self._mapped_view, '__setitem__'):
                # Windows ctypes approach
                import ctypes
                ctypes.memmove(ctypes.c_char_p(self._mapped_view + actual_offset), data, len(data))
            else:
                # Unix mmap approach
                self._mapped_view[actual_offset:actual_offset + len(data)] = data
                
    def read_bytes(self, offset: int, size: int) -> bytes:
        """Read bytes from shared memory region.
        
        Args:
            offset: Offset from start of user data region (after header)
            size: Number of bytes to read
            
        Returns:
            Bytes read from shared memory
        """
        with self._lock:
            if not self._mapped_view:
                raise SharedMemoryAccessError("Shared memory region not mapped")
                
            if offset < 0 or offset + size > self.user_size:
                raise ValueError(f"Read would exceed region bounds: offset={offset}, size={size}, max={self.user_size}")
                
            actual_offset = self.HEADER_SIZE + offset
            
            if self.platform == SharedMemoryPlatform.WINDOWS and hasattr(self._mapped_view, '__getitem__'):
                # Windows ctypes approach
                import ctypes
                data = (ctypes.c_char * size).from_address(self._mapped_view + actual_offset)
                return bytes(data)
            else:
                # Unix mmap approach
                return bytes(self._mapped_view[actual_offset:actual_offset + size])
                
    def get_reference_count(self) -> int:
        """Get current reference count from shared memory header."""
        try:
            _, _, _, ref_count, _, _, _ = self._read_header()
            return ref_count
        except Exception:
            return 0
            
    def close(self):
        """Close shared memory region and clean up resources."""
        with self._lock:
            if self._mapped_view is None:
                return  # Already closed
                
            try:
                # Decrement reference count
                ref_count = self._decrement_ref_count()
                
                # Clean up platform-specific resources
                self._cleanup_resources()
                
                # Remove temporary file if reference count is zero (Unix only)
                if (ref_count == 0 and 
                    self.platform == SharedMemoryPlatform.UNIX and 
                    self._temp_file_path and 
                    os.path.exists(self._temp_file_path)):
                    try:
                        os.unlink(self._temp_file_path)
                        logger.debug(f"Removed shared memory file: {self._temp_file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to remove shared memory file {self._temp_file_path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error during shared memory region close: {e}")
                
            finally:
                self._mapped_view = None


class SharedMemoryManager:
    """Manager for shared memory regions with automatic cleanup and monitoring."""
    
    def __init__(self, config: SharedMemoryConfig):
        """Initialize shared memory manager.
        
        Args:
            config: Configuration for shared memory behavior
        """
        self.config = config
        self.statistics = SharedMemoryStatistics()
        self._regions: Dict[str, SharedMemoryRegion] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
        # Start cleanup thread if configured
        if config.cleanup_interval > 0:
            self._start_cleanup_thread()
            
    def __del__(self):
        """Destructor to ensure cleanup thread is stopped."""
        self.shutdown()
        
    def create_region(self, name: str, size: int) -> SharedMemoryRegion:
        """Create a new shared memory region.
        
        Args:
            name: Unique name for the region
            size: Size in bytes for the region
            
        Returns:
            SharedMemoryRegion instance
        """
        with self._lock:
            if name in self._regions:
                raise SharedMemoryError(f"Shared memory region '{name}' already exists")
                
            if len(self._regions) >= self.config.max_regions:
                raise SharedMemoryError(f"Maximum number of regions ({self.config.max_regions}) exceeded")
                
            start_time = time.time()
            
            try:
                region = SharedMemoryRegion(name, size, self.config, create=True)
                self._regions[name] = region
                
                duration = time.time() - start_time
                self.statistics.record_allocation(size, duration)
                
                logger.info(f"Created shared memory region '{name}' ({size} bytes)")
                return region
                
            except Exception as e:
                self.statistics.record_allocation_failure()
                raise
                
    def attach_region(self, name: str, size: int) -> SharedMemoryRegion:
        """Attach to an existing shared memory region.
        
        Args:
            name: Name of the existing region
            size: Expected size of the region
            
        Returns:
            SharedMemoryRegion instance
        """
        with self._lock:
            if name in self._regions:
                return self._regions[name]
                
            start_time = time.time()
            
            try:
                region = SharedMemoryRegion(name, size, self.config, create=False)
                self._regions[name] = region
                
                duration = time.time() - start_time
                self.statistics.record_allocation(size, duration)
                
                logger.info(f"Attached to shared memory region '{name}'")
                return region
                
            except Exception as e:
                self.statistics.record_access_failure()
                raise
                
    def get_region(self, name: str) -> Optional[SharedMemoryRegion]:
        """Get an existing region by name.
        
        Args:
            name: Name of the region
            
        Returns:
            SharedMemoryRegion instance or None if not found
        """
        with self._lock:
            return self._regions.get(name)
            
    def remove_region(self, name: str) -> bool:
        """Remove and close a shared memory region.
        
        Args:
            name: Name of the region to remove
            
        Returns:
            True if region was removed, False if not found
        """
        with self._lock:
            region = self._regions.pop(name, None)
            if region:
                try:
                    region.close()
                    self.statistics.record_deallocation(region.size)
                    logger.info(f"Removed shared memory region '{name}'")
                    return True
                except Exception as e:
                    logger.error(f"Error removing shared memory region '{name}': {e}")
                    
            return False
            
    def list_regions(self) -> Dict[str, Dict[str, Any]]:
        """List all managed regions with their metadata.
        
        Returns:
            Dictionary mapping region names to metadata
        """
        with self._lock:
            result = {}
            for name, region in self._regions.items():
                try:
                    result[name] = {
                        'size': region.size,
                        'user_size': region.user_size,
                        'reference_count': region.get_reference_count(),
                        'platform': region.platform.value,
                        'alignment': region.config.alignment
                    }
                except Exception as e:
                    result[name] = {'error': str(e)}
                    
            return result
            
    def cleanup_orphaned_regions(self) -> int:
        """Clean up orphaned shared memory regions.
        
        Returns:
            Number of regions cleaned up
        """
        cleaned_count = 0
        
        with self._lock:
            regions_to_remove = []
            
            for name, region in self._regions.items():
                try:
                    ref_count = region.get_reference_count()
                    if ref_count <= 1:  # Only our reference remains
                        regions_to_remove.append(name)
                except Exception:
                    # Region is likely corrupted or inaccessible
                    regions_to_remove.append(name)
                    
            for name in regions_to_remove:
                if self.remove_region(name):
                    cleaned_count += 1
                    
        if cleaned_count > 0:
            self.statistics.orphaned_regions_cleaned += cleaned_count
            logger.info(f"Cleaned up {cleaned_count} orphaned shared memory regions")
            
        return cleaned_count
        
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
            
        self._shutdown = False
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Started shared memory cleanup thread")
        
    def _cleanup_worker(self):
        """Background worker for cleaning up orphaned regions."""
        while not self._shutdown:
            try:
                time.sleep(self.config.cleanup_interval)
                if not self._shutdown:
                    self.cleanup_orphaned_regions()
            except Exception as e:
                logger.error(f"Error in shared memory cleanup worker: {e}")
                
    def shutdown(self):
        """Shutdown manager and clean up all resources."""
        with self._lock:
            if self._shutdown:
                return
                
            self._shutdown = True
            
            # Stop cleanup thread
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5.0)
                
            # Close all regions
            regions_to_close = list(self._regions.keys())
            for name in regions_to_close:
                self.remove_region(name)
                
            logger.info("Shared memory manager shutdown complete")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about shared memory usage.
        
        Returns:
            Dictionary containing statistics
        """
        stats = self.statistics.get_summary()
        
        with self._lock:
            stats['active_regions'] = len(self._regions)
            stats['regions_by_platform'] = {}
            
            for region in self._regions.values():
                platform = region.platform.value
                if platform not in stats['regions_by_platform']:
                    stats['regions_by_platform'][platform] = 0
                stats['regions_by_platform'][platform] += 1
                
        return stats


# Global shared memory manager instance
_global_manager: Optional[SharedMemoryManager] = None
_manager_lock = threading.Lock()


def get_shared_memory_manager(config: Optional[SharedMemoryConfig] = None) -> SharedMemoryManager:
    """Get global shared memory manager instance.
    
    Args:
        config: Configuration for manager (only used on first call)
        
    Returns:
        Global SharedMemoryManager instance
    """
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            if config is None:
                config = SharedMemoryConfig()
            _global_manager = SharedMemoryManager(config)
            
        return _global_manager


def shutdown_shared_memory():
    """Shutdown global shared memory manager."""
    global _global_manager
    
    with _manager_lock:
        if _global_manager:
            _global_manager.shutdown()
            _global_manager = None


# Convenience functions for common operations
def create_shared_region(name: str, size: int, config: Optional[SharedMemoryConfig] = None) -> SharedMemoryRegion:
    """Create a shared memory region.
    
    Args:
        name: Unique name for the region
        size: Size in bytes
        config: Optional configuration (uses default if None)
        
    Returns:
        SharedMemoryRegion instance
    """
    manager = get_shared_memory_manager(config)
    return manager.create_region(name, size)


def attach_shared_region(name: str, size: int, config: Optional[SharedMemoryConfig] = None) -> SharedMemoryRegion:
    """Attach to an existing shared memory region.
    
    Args:
        name: Name of existing region
        size: Expected size
        config: Optional configuration (uses default if None)
        
    Returns:
        SharedMemoryRegion instance
    """
    manager = get_shared_memory_manager(config)
    return manager.attach_region(name, size)


def remove_shared_region(name: str) -> bool:
    """Remove a shared memory region.
    
    Args:
        name: Name of region to remove
        
    Returns:
        True if removed, False if not found
    """
    manager = get_shared_memory_manager()
    return manager.remove_region(name) 