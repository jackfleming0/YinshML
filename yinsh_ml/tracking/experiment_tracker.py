"""
Experiment tracker module for YinshML.

Provides a high-level singleton interface for experiment management,
metadata capture, and metric logging with automatic reproducibility features.
"""

import threading
import logging
import json
import os
import sys
import platform
import subprocess
import socket
import pkg_resources
import queue
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from .utils import DatabaseConnectionManager, initialize_database, create_experiment as utils_create_experiment
    from .database import ExperimentDatabase
except ImportError:
    from utils import DatabaseConnectionManager, initialize_database, create_experiment as utils_create_experiment
    from database import ExperimentDatabase

logger = logging.getLogger(__name__)


class ExperimentTrackerError(Exception):
    """Base exception for ExperimentTracker operations."""
    pass


class ExperimentNotFoundError(ExperimentTrackerError):
    """Raised when an experiment is not found."""
    pass


class ExperimentConfigurationError(ExperimentTrackerError):
    """Raised when experiment configuration is invalid."""
    pass


# =================================================================
# Asynchronous Logging Infrastructure
# =================================================================

class AsyncLogEntryType(Enum):
    """Types of asynchronous log entries."""
    METRIC = "metric"
    PARAMETER = "parameter"
    STATUS = "status"
    TAG_ADD = "tag_add"
    TAG_REMOVE = "tag_remove"
    NOTE = "note"


class AsyncLogEntry(ABC):
    """Base class for asynchronous log entries."""
    
    def __init__(self, experiment_id: int, timestamp: datetime, 
                 entry_type: AsyncLogEntryType = None, attempt_count: int = 0):
        self.experiment_id = experiment_id
        self.timestamp = timestamp
        self.entry_type = entry_type
        self.attempt_count = attempt_count
    
    @abstractmethod
    def execute(self, tracker: 'ExperimentTracker') -> bool:
        """
        Execute the log operation synchronously.
        
        Args:
            tracker: ExperimentTracker instance
            
        Returns:
            bool: True if successful, False if should retry
        """
        pass


@dataclass
class MetricLogEntry(AsyncLogEntry):
    """Asynchronous metric log entry."""
    experiment_id: int
    timestamp: datetime
    metric_name: str
    value: float
    iteration: Optional[int] = None
    attempt_count: int = 0
    
    def __post_init__(self):
        super().__init__(self.experiment_id, self.timestamp, AsyncLogEntryType.METRIC, self.attempt_count)
    
    def execute(self, tracker: 'ExperimentTracker') -> bool:
        """Execute metric logging synchronously."""
        try:
            # Use the internal sync method to avoid recursive async calls
            tracker._log_metric_sync(self.experiment_id, self.metric_name, 
                                   self.value, self.iteration, self.timestamp)
            return True
        except Exception as e:
            logger.warning(f"Failed to log metric {self.metric_name} for experiment {self.experiment_id}: {e}")
            return False


@dataclass
class ParameterLogEntry(AsyncLogEntry):
    """Asynchronous parameter log entry."""
    experiment_id: int
    timestamp: datetime
    param_name: str
    value: Any
    attempt_count: int = 0
    
    def __post_init__(self):
        super().__init__(self.experiment_id, self.timestamp, AsyncLogEntryType.PARAMETER, self.attempt_count)
    
    def execute(self, tracker: 'ExperimentTracker') -> bool:
        """Execute parameter logging synchronously."""
        try:
            tracker._log_parameter_sync(self.experiment_id, self.param_name, self.value)
            return True
        except Exception as e:
            logger.warning(f"Failed to log parameter {self.param_name} for experiment {self.experiment_id}: {e}")
            return False


@dataclass
class StatusUpdateEntry(AsyncLogEntry):
    """Asynchronous status update entry."""
    experiment_id: int
    timestamp: datetime
    status: str
    attempt_count: int = 0
    
    def __post_init__(self):
        super().__init__(self.experiment_id, self.timestamp, AsyncLogEntryType.STATUS, self.attempt_count)
    
    def execute(self, tracker: 'ExperimentTracker') -> bool:
        """Execute status update synchronously."""
        try:
            tracker._update_experiment_status_sync(self.experiment_id, self.status)
            return True
        except Exception as e:
            logger.warning(f"Failed to update status for experiment {self.experiment_id}: {e}")
            return False


@dataclass
class TagOperationEntry(AsyncLogEntry):
    """Asynchronous tag operation entry."""
    experiment_id: int
    timestamp: datetime
    tags: List[str]
    operation: str  # 'add' or 'remove'
    attempt_count: int = 0
    
    def __post_init__(self):
        entry_type = AsyncLogEntryType.TAG_ADD if self.operation == 'add' else AsyncLogEntryType.TAG_REMOVE
        super().__init__(self.experiment_id, self.timestamp, entry_type, self.attempt_count)
    
    def execute(self, tracker: 'ExperimentTracker') -> bool:
        """Execute tag operation synchronously."""
        try:
            if self.operation == 'add':
                tracker._add_tags_sync(self.experiment_id, self.tags)
            else:
                tracker._remove_tags_sync(self.experiment_id, self.tags)
            return True
        except Exception as e:
            logger.warning(f"Failed to {self.operation} tags for experiment {self.experiment_id}: {e}")
            return False


@dataclass
class NoteEntry(AsyncLogEntry):
    """Asynchronous note entry."""
    experiment_id: int
    timestamp: datetime
    note: str
    attempt_count: int = 0
    
    def __post_init__(self):
        super().__init__(self.experiment_id, self.timestamp, AsyncLogEntryType.NOTE, self.attempt_count)
    
    def execute(self, tracker: 'ExperimentTracker') -> bool:
        """Execute note addition synchronously."""
        try:
            tracker._add_note_sync(self.experiment_id, self.note)
            return True
        except Exception as e:
            logger.warning(f"Failed to add note for experiment {self.experiment_id}: {e}")
            return False


class AsyncLogger:
    """
    Asynchronous logger for experiment tracking operations.
    
    Manages a background thread that processes logging operations
    from a queue to minimize performance impact on the main thread.
    """
    
    def __init__(self, max_queue_size: int = 10000, flush_interval: float = 1.0,
                 batch_size: int = 100, max_retries: int = 3):
        """
        Initialize the async logger.
        
        Args:
            max_queue_size: Maximum number of items in queue
            flush_interval: Interval between queue processing (seconds)
            batch_size: Number of items to process per batch
            max_retries: Maximum retry attempts for failed operations
        """
        self.max_queue_size = max_queue_size
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Queue and threading
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread = None
        self._shutdown_event = threading.Event()
        self._running = False
        
        # Performance monitoring
        self._stats = {
            'items_processed': 0,
            'items_failed': 0,
            'queue_full_count': 0,
            'total_processing_time': 0.0,
            'max_queue_size_seen': 0
        }
        self._stats_lock = threading.Lock()
        
    def start(self, tracker: 'ExperimentTracker'):
        """
        Start the async logging worker thread.
        
        Args:
            tracker: ExperimentTracker instance for processing operations
        """
        if self._running:
            logger.warning("AsyncLogger already running")
            return
            
        self._running = True
        self._shutdown_event.clear()
        self._tracker = tracker
        
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncLogger-Worker",
            daemon=True
        )
        self._worker_thread.start()
        
        logger.info(f"AsyncLogger started with queue_size={self.max_queue_size}, "
                   f"flush_interval={self.flush_interval}s, batch_size={self.batch_size}")
    
    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the async logging worker and process remaining items.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            bool: True if shutdown completed within timeout
        """
        if not self._running:
            return True
            
        logger.info("Stopping AsyncLogger...")
        self._shutdown_event.set()
        
        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            
        self._running = False
        
        # Report final stats
        remaining_items = self._queue.qsize()
        if remaining_items > 0:
            logger.warning(f"AsyncLogger stopped with {remaining_items} items remaining in queue")
        
        stats = self.get_stats()
        logger.info(f"AsyncLogger stopped. Final stats: {stats}")
        
        return not (self._worker_thread and self._worker_thread.is_alive())
    
    def add_entry(self, entry: AsyncLogEntry) -> bool:
        """
        Add an entry to the async logging queue.
        
        Args:
            entry: AsyncLogEntry to process
            
        Returns:
            bool: True if added successfully, False if queue is full
        """
        if not self._running:
            logger.warning("AsyncLogger not running, cannot add entry")
            return False
            
        try:
            # Set timestamp if not already set
            if not entry.timestamp:
                entry.timestamp = datetime.now()
                
            self._queue.put_nowait(entry)
            
            # Update stats
            with self._stats_lock:
                current_size = self._queue.qsize()
                self._stats['max_queue_size_seen'] = max(
                    self._stats['max_queue_size_seen'], current_size
                )
            
            return True
            
        except queue.Full:
            with self._stats_lock:
                self._stats['queue_full_count'] += 1
            
            logger.warning(f"AsyncLogger queue is full (size: {self.max_queue_size}), "
                          f"dropping entry: {entry}")
            return False
    
    def flush(self, timeout: float = 5.0) -> bool:
        """
        Flush all pending items in the queue.
        
        Args:
            timeout: Maximum time to wait for flush
            
        Returns:
            bool: True if all items were processed
        """
        if not self._running:
            return True
            
        start_time = time.time()
        initial_size = self._queue.qsize()
        
        # Wait for queue to empty or timeout
        while self._queue.qsize() > 0 and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        final_size = self._queue.qsize()
        success = final_size == 0
        
        if success:
            logger.debug(f"AsyncLogger flush completed, processed {initial_size} items")
        else:
            logger.warning(f"AsyncLogger flush timed out, {final_size} items remaining")
            
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
            stats['current_queue_size'] = self._queue.qsize()
            stats['is_running'] = self._running
            
            # Calculate average processing time
            if stats['items_processed'] > 0:
                stats['avg_processing_time_ms'] = (
                    stats['total_processing_time'] / stats['items_processed']
                ) * 1000
            else:
                stats['avg_processing_time_ms'] = 0.0
                
        return stats
    
    def _worker_loop(self):
        """Main worker loop for processing queue items."""
        logger.debug("AsyncLogger worker thread started")
        
        retry_queue = []
        
        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                # Process regular queue items
                items_processed = self._process_batch_from_queue()
                
                # Process retry items
                if retry_queue:
                    retry_queue = self._process_retry_queue(retry_queue)
                
                # If no items were processed, wait before next iteration
                if items_processed == 0 and not retry_queue:
                    self._shutdown_event.wait(self.flush_interval)
                    
            except Exception as e:
                logger.error(f"Error in AsyncLogger worker loop: {e}")
                time.sleep(0.1)  # Brief pause before continuing
        
        # Process any remaining retry items
        if retry_queue:
            logger.info(f"Processing {len(retry_queue)} remaining retry items during shutdown")
            self._process_retry_queue(retry_queue, final_attempt=True)
        
        logger.debug("AsyncLogger worker thread ended")
    
    def _process_batch_from_queue(self) -> int:
        """Process a batch of items from the main queue."""
        items_processed = 0
        retry_items = []
        
        # Get batch of items
        batch = []
        for _ in range(self.batch_size):
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break
        
        if not batch:
            return 0
        
        # Process batch
        start_time = time.time()
        
        for entry in batch:
            try:
                success = entry.execute(self._tracker)
                
                if success:
                    with self._stats_lock:
                        self._stats['items_processed'] += 1
                    items_processed += 1
                else:
                    # Add to retry queue if not exceeded max retries
                    entry.attempt_count += 1
                    if entry.attempt_count <= self.max_retries:
                        retry_items.append(entry)
                    else:
                        with self._stats_lock:
                            self._stats['items_failed'] += 1
                        logger.error(f"Failed to process entry after {self.max_retries} attempts: {entry}")
                        
            except Exception as e:
                logger.error(f"Unexpected error processing entry {entry}: {e}")
                with self._stats_lock:
                    self._stats['items_failed'] += 1
        
        # Update processing time stats
        processing_time = time.time() - start_time
        with self._stats_lock:
            self._stats['total_processing_time'] += processing_time
        
        # Add retry items back for later processing
        for item in retry_items:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                logger.error(f"Queue full, dropping retry item: {item}")
                with self._stats_lock:
                    self._stats['items_failed'] += 1
        
        return items_processed
    
    def _process_retry_queue(self, retry_queue: List[AsyncLogEntry], 
                           final_attempt: bool = False) -> List[AsyncLogEntry]:
        """Process retry queue with exponential backoff."""
        remaining_retries = []
        
        for entry in retry_queue:
            # Calculate backoff delay
            backoff_delay = min(2 ** (entry.attempt_count - 1), 30)  # Max 30 seconds
            
            # Skip if not enough time has passed (unless final attempt)
            if not final_attempt:
                time_since_attempt = (datetime.now() - entry.timestamp).total_seconds()
                if time_since_attempt < backoff_delay:
                    remaining_retries.append(entry)
                    continue
            
            try:
                success = entry.execute(self._tracker)
                
                if success:
                    with self._stats_lock:
                        self._stats['items_processed'] += 1
                else:
                    entry.attempt_count += 1
                    if entry.attempt_count <= self.max_retries and not final_attempt:
                        remaining_retries.append(entry)
                    else:
                        with self._stats_lock:
                            self._stats['items_failed'] += 1
                        logger.error(f"Failed to process retry entry after {self.max_retries} attempts: {entry}")
                        
            except Exception as e:
                logger.error(f"Unexpected error processing retry entry {entry}: {e}")
                with self._stats_lock:
                    self._stats['items_failed'] += 1
        
        return remaining_retries


class ExperimentTracker:
    """
    Singleton experiment tracker for comprehensive ML experiment management.
    
    Provides a high-level interface for:
    - Creating and managing experiments
    - Automatic metadata capture (git, system, configuration)
    - Metric and parameter logging
    - Experiment status management
    - Asynchronous logging capabilities
    
    Uses singleton pattern to ensure consistent state across the application.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = None, config: Dict[str, Any] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ExperimentTracker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize the ExperimentTracker singleton.
        
        Args:
            db_path: Path to the database file. If None, uses default location.
            config: Configuration dictionary for tracker settings.
        """
        if not self._initialized:
            self._initialize(db_path, config)
    
    def _initialize(self, db_path: str = None, config: Dict[str, Any] = None):
        """Internal initialization method called only once."""
        # Default configuration
        self._config = {
            'auto_capture_git': True,
            'auto_capture_system': True,
            'auto_capture_environment': True,
            'async_logging': True,
            'default_experiment_status': 'running',
            'log_level': 'INFO',
            'flush_interval': 5.0,  # seconds
            'queue_size_limit': 1000,
        }
        
        # Update with provided config
        if config:
            self._config.update(config)
        
        # Set up database path
        if db_path is None:
            # Default to a tracking directory in the current working directory
            db_path = Path.cwd() / 'experiments' / 'tracking.db'
        
        self._db_path = str(db_path)
        
        # Initialize database
        try:
            self._database = initialize_database(self._db_path)
            self._connection_manager = DatabaseConnectionManager(self._db_path)
        except Exception as e:
            raise ExperimentConfigurationError(f"Failed to initialize database: {e}")
        
        # Active experiments tracking
        self._active_experiments = {}  # experiment_id -> experiment_data
        self._experiment_lock = threading.Lock()
        
        # Asynchronous logging
        self._async_logger = None
        self._async_enabled = self._config.get('enable_async_logging', True)
        
        # Initialize async logging if enabled
        if self._async_enabled:
            self._initialize_async_logging()
        
        self._initialized = True
        logger.info(f"ExperimentTracker initialized successfully (async_logging: {self._async_enabled})")
    
    @classmethod
    def get_instance(cls, db_path: str = None, config: Dict[str, Any] = None):
        """
        Get the singleton instance of ExperimentTracker.
        
        Args:
            db_path: Database path (only used if instance doesn't exist)
            config: Configuration dictionary (only used if instance doesn't exist)
            
        Returns:
            ExperimentTracker: The singleton instance
        """
        return cls(db_path, config)
    
    def configure(self, **kwargs):
        """
        Update configuration settings.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        with self._lock:
            self._config.update(kwargs)
            logger.debug(f"Configuration updated: {kwargs}")
    
    def get_config(self, key: str = None) -> Union[Dict[str, Any], Any]:
        """
        Get configuration value(s).
        
        Args:
            key: Specific configuration key. If None, returns all config.
            
        Returns:
            Configuration value or entire configuration dictionary
        """
        if key is None:
            return self._config.copy()
        return self._config.get(key)
    
    # =================================================================
    # Metadata Capture Methods
    # =================================================================
    
    def _capture_git_metadata(self) -> Dict[str, Any]:
        """
        Capture git repository metadata.
        
        Returns:
            Dict containing git information
        """
        git_info = {
            'commit': 'unknown',
            'branch': 'unknown', 
            'working_directory_clean': True,
            'remote_url': None,
            'commit_message': None,
            'author': None,
            'timestamp': None
        }
        
        if not self._config.get('auto_capture_git', True):
            logger.debug("Git metadata capture disabled")
            return git_info
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode != 0:
                logger.warning("Not in a git repository")
                return git_info
            
            # Get commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['commit'] = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check working directory status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['working_directory_clean'] = len(result.stdout.strip()) == 0
            
            # Get remote URL
            result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
            
            # Get commit message
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%s'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['commit_message'] = result.stdout.strip()
            
            # Get author
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%an <%ae>'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['author'] = result.stdout.strip()
            
            # Get commit timestamp
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%ai'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['timestamp'] = result.stdout.strip()
            
            logger.debug(f"Captured git metadata: {git_info}")
            
        except FileNotFoundError:
            logger.warning("Git command not found - git metadata unavailable")
        except Exception as e:
            logger.warning(f"Failed to capture git metadata: {e}")
        
        return git_info
    
    def _capture_system_metadata(self) -> Dict[str, Any]:
        """
        Capture system and hardware metadata.
        
        Returns:
            Dict containing system information
        """
        system_info = {}
        
        if not self._config.get('auto_capture_system', True):
            logger.debug("System metadata capture disabled")
            return system_info
        
        try:
            # Basic system information
            system_info.update({
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'hostname': socket.gethostname(),
                'python_version': sys.version,
                'python_executable': sys.executable,
                'working_directory': str(Path.cwd()),
            })
            
            # CPU information
            try:
                import psutil
                system_info.update({
                    'cpu_count': psutil.cpu_count(),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'memory_total': psutil.virtual_memory().total,
                    'memory_available': psutil.virtual_memory().available,
                })
            except ImportError:
                logger.debug("psutil not available for detailed system info")
            
            logger.debug(f"Captured system metadata with {len(system_info)} fields")
            
        except Exception as e:
            logger.warning(f"Failed to capture system metadata: {e}")
        
        return system_info
    
    def _capture_environment_metadata(self) -> Dict[str, Any]:
        """
        Capture environment and dependency metadata.
        
        Returns:
            Dict containing environment information
        """
        env_info = {}
        
        if not self._config.get('auto_capture_environment', True):
            logger.debug("Environment metadata capture disabled")
            return env_info
        
        try:
            # Environment variables (filtered for relevant ones)
            relevant_env_vars = [
                'PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'TF_CPP_MIN_LOG_LEVEL', 'PYTORCH_CUDA_ALLOC_CONF'
            ]
            
            env_vars = {}
            for var in relevant_env_vars:
                value = os.environ.get(var)
                if value is not None:
                    env_vars[var] = value
            
            env_info['environment_variables'] = env_vars
            
            # Installed packages
            try:
                installed_packages = {}
                for pkg in pkg_resources.working_set:
                    installed_packages[pkg.project_name] = pkg.version
                
                env_info['installed_packages'] = installed_packages
                logger.debug(f"Captured {len(installed_packages)} installed packages")
                
            except Exception as e:
                logger.warning(f"Failed to capture package information: {e}")
            
            # PyTorch/CUDA information if available
            try:
                import torch
                env_info['torch'] = {
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                    'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
                if torch.cuda.is_available():
                    env_info['torch']['device_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            except ImportError:
                pass
            
            logger.debug(f"Captured environment metadata with {len(env_info)} categories")
            
        except Exception as e:
            logger.warning(f"Failed to capture environment metadata: {e}")
        
        return env_info
    
    def _create_configuration_snapshot(self, user_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a snapshot of current configuration state.
        
        Args:
            user_config: User-provided configuration to include
            
        Returns:
            Dict containing configuration snapshot
        """
        config_snapshot = {
            'tracker_config': self._config.copy(),
            'timestamp': datetime.now().isoformat(),
        }
        
        if user_config:
            config_snapshot['user_config'] = user_config
        
        return config_snapshot
    
    # =================================================================
    # Core Interface Methods
    # =================================================================
    
    def create_experiment(self, name: str, description: str = None, 
                         tags: List[str] = None, config: Dict[str, Any] = None) -> int:
        """
        Create a new experiment with automatic metadata capture.
        
        Args:
            name: Experiment name
            description: Optional experiment description
            tags: Optional list of tags for organization
            config: Optional experiment configuration
            
        Returns:
            int: Unique experiment ID
            
        Raises:
            ExperimentConfigurationError: If experiment creation fails
        """
        try:
            logger.info(f"Creating new experiment: {name}")
            
            # Capture metadata
            git_metadata = self._capture_git_metadata()
            system_metadata = self._capture_system_metadata()
            environment_metadata = self._capture_environment_metadata()
            config_snapshot = self._create_configuration_snapshot(config)
            
            # Prepare data for database storage
            git_commit = git_metadata.get('commit', 'unknown')
            git_branch = git_metadata.get('branch', 'unknown')
            
            # Combine all metadata into environment JSON
            combined_environment = {
                'system': system_metadata,
                'environment': environment_metadata,
                'git': git_metadata,
            }
            
            # Use description as notes if provided
            notes = description
            
            # Get default status from config
            status = self._config.get('default_experiment_status', 'running')
            
            # Create experiment using existing utility function
            # Note: utils_create_experiment has @with_transaction decorator, 
            # so it manages its own connection and transaction
            experiment_id = utils_create_experiment(
                name=name,
                git_commit=git_commit,
                git_branch=git_branch,
                config=config_snapshot,
                environment=combined_environment,
                status=status,
                notes=notes,
                tags=tags
            )
            
            # Add to active experiments tracking
            experiment_data = {
                'id': experiment_id,
                'name': name,
                'status': status,
                'created_at': datetime.now(),
                'tags': tags or [],
                'metrics_count': 0,
                'last_metric_time': None,
            }
            
            with self._experiment_lock:
                self._active_experiments[experiment_id] = experiment_data
            
            logger.info(f"Successfully created experiment {experiment_id}: {name}")
            return experiment_id
            
        except Exception as e:
            error_msg = f"Failed to create experiment '{name}': {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def log_metric(self, experiment_id: int, metric_name: str, 
                   value: float, iteration: int = None, 
                   timestamp: datetime = None) -> None:
        """
        Log a metric value for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            value: Metric value
            iteration: Optional iteration number (defaults to auto-increment)
            timestamp: Optional timestamp (defaults to current time)
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ExperimentConfigurationError: If logging fails
        """
        # Try async logging first if enabled
        if self._async_enabled:
            entry = MetricLogEntry(
                experiment_id=experiment_id,
                timestamp=timestamp or datetime.now(),
                metric_name=metric_name,
                value=value,
                iteration=iteration
            )
            
            if self._log_async_if_enabled(entry):
                logger.debug(f"Queued async metric {metric_name}={value} for experiment {experiment_id}")
                return
            else:
                logger.warning("Failed to queue async metric, falling back to synchronous logging")
        
        # Fallback to synchronous logging
        self._log_metric_sync(experiment_id, metric_name, value, iteration, timestamp)
    
    def log_parameter(self, experiment_id: int, param_name: str, 
                     value: Any) -> None:
        """
        Log a parameter for an experiment.
        
        Parameters are stored as special metrics with a 'param:' prefix to distinguish
        them from regular metrics.
        
        Args:
            experiment_id: Experiment ID
            param_name: Parameter name
            value: Parameter value (will be converted to string for storage)
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ExperimentConfigurationError: If logging fails
        """
        # Try async logging first if enabled
        if self._async_enabled:
            entry = ParameterLogEntry(
                experiment_id=experiment_id,
                timestamp=datetime.now(),
                param_name=param_name,
                value=value
            )
            
            if self._log_async_if_enabled(entry):
                logger.debug(f"Queued async parameter {param_name}={value} for experiment {experiment_id}")
                return
            else:
                logger.warning("Failed to queue async parameter, falling back to synchronous logging")
        
        # Fallback to synchronous logging
        self._log_parameter_sync(experiment_id, param_name, value)
    
    def update_experiment_status(self, experiment_id: int, status: str) -> None:
        """
        Update experiment status.
        
        Args:
            experiment_id: Experiment ID
            status: New status ('running', 'completed', 'failed', 'paused')
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ExperimentConfigurationError: If status update fails
        """
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(status, str) or not status.strip():
                raise ExperimentConfigurationError("Status must be a non-empty string")
            
            valid_statuses = ['running', 'completed', 'failed', 'paused', 'cancelled']
            if status not in valid_statuses:
                logger.warning(f"Status '{status}' not in recommended values: {valid_statuses}")
            
            # Update status using existing utility function
            from .utils import update_experiment_status as utils_update_status
            utils_update_status(experiment_id, status)
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    self._active_experiments[experiment_id]['status'] = status
                    self._active_experiments[experiment_id]['status_updated_at'] = datetime.now()
                    
                    # If experiment is completed, failed, or cancelled, it's no longer active
                    if status in ['completed', 'failed', 'cancelled']:
                        logger.info(f"Experiment {experiment_id} is no longer active (status: {status})")
            
            logger.info(f"Updated experiment {experiment_id} status to '{status}'")
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            else:
                error_msg = f"Failed to update status for experiment {experiment_id}: {e}"
                logger.error(error_msg)
                raise ExperimentConfigurationError(error_msg)
    
    def add_tags(self, experiment_id: int, tags: List[str]) -> None:
        """
        Add tags to an experiment.
        
        Args:
            experiment_id: Experiment ID
            tags: List of tags to add
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ExperimentConfigurationError: If tag addition fails
        """
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(tags, list):
                raise ExperimentConfigurationError("Tags must be provided as a list")
            
            if not tags:
                logger.debug("No tags to add")
                return
            
            # Validate all tags are strings
            for tag in tags:
                if not isinstance(tag, str) or not tag.strip():
                    raise ExperimentConfigurationError("All tags must be non-empty strings")
            
            # Add tags using existing utility function
            from .utils import transaction_context, add_tags_to_experiment
            
            with transaction_context() as conn:
                add_tags_to_experiment(conn, experiment_id, tags)
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    exp_data = self._active_experiments[experiment_id]
                    current_tags = set(exp_data.get('tags', []))
                    current_tags.update(tags)
                    exp_data['tags'] = list(current_tags)
            
            logger.info(f"Added {len(tags)} tags to experiment {experiment_id}: {tags}")
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            else:
                error_msg = f"Failed to add tags to experiment {experiment_id}: {e}"
                logger.error(error_msg)
                raise ExperimentConfigurationError(error_msg)
    
    def remove_tags(self, experiment_id: int, tags: List[str]) -> None:
        """
        Remove tags from an experiment.
        
        Args:
            experiment_id: Experiment ID
            tags: List of tags to remove
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ExperimentConfigurationError: If tag removal fails
        """
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(tags, list):
                raise ExperimentConfigurationError("Tags must be provided as a list")
            
            if not tags:
                logger.debug("No tags to remove")
                return
            
            # Remove tags using database operations
            from .utils import transaction_context
            
            with transaction_context() as conn:
                cursor = conn.cursor()
                for tag in tags:
                    if not isinstance(tag, str) or not tag.strip():
                        continue  # Skip invalid tags
                    
                    cursor.execute("""
                        DELETE FROM tags 
                        WHERE experiment_id = ? AND tag = ?
                    """, (experiment_id, tag.strip()))
                
                # Check if experiment exists
                cursor.execute("SELECT id FROM experiments WHERE id = ?", (experiment_id,))
                if not cursor.fetchone():
                    raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    exp_data = self._active_experiments[experiment_id]
                    current_tags = set(exp_data.get('tags', []))
                    current_tags.difference_update(tags)
                    exp_data['tags'] = list(current_tags)
            
            logger.info(f"Removed {len(tags)} tags from experiment {experiment_id}: {tags}")
            
        except ExperimentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to remove tags from experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete experiment data including metrics and tags.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dict containing experiment data or None if not found
            
        Raises:
            ExperimentConfigurationError: If experiment_id is invalid
        """
        try:
            # Validate input
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            # Flush async queue to ensure data consistency
            if self._async_enabled and self._async_logger:
                self._async_logger.flush(timeout=2.0)
            
            # Get base experiment data
            from .utils import get_experiment_by_id, get_experiment_metrics, get_experiment_tags
            
            experiment = get_experiment_by_id(experiment_id)
            if not experiment:
                return None
            
            # Enrich with metrics and tags
            experiment['metrics'] = get_experiment_metrics(experiment_id)
            experiment['tags'] = get_experiment_tags(experiment_id)
            
            # Add computed fields
            experiment['metric_count'] = len(experiment['metrics'])
            experiment['tag_count'] = len(experiment['tags'])
            
            logger.debug(f"Retrieved experiment {experiment_id} with {experiment['metric_count']} metrics")
            return experiment
            
        except ExperimentConfigurationError:
            raise
        except Exception as e:
            error_msg = f"Failed to retrieve experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def get_metric_history(self, experiment_id: int, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get the history of a specific metric for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            
        Returns:
            List of metric records with timestamps and values, ordered by iteration
            
        Raises:
            ExperimentConfigurationError: If inputs are invalid
            ExperimentNotFoundError: If experiment doesn't exist
        """
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ExperimentConfigurationError("Metric name must be a non-empty string")
            
            # Flush async queue to ensure data consistency
            if self._async_enabled and self._async_logger:
                self._async_logger.flush(timeout=2.0)
            
            # Check if experiment exists
            from .utils import get_experiment_by_id, get_experiment_metrics
            
            experiment = get_experiment_by_id(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            # Get metric history
            metrics = get_experiment_metrics(experiment_id, metric_name.strip())
            
            # Convert timestamps to datetime objects for consistency
            for metric in metrics:
                if 'timestamp' in metric and isinstance(metric['timestamp'], str):
                    try:
                        metric['timestamp'] = datetime.fromisoformat(metric['timestamp'])
                    except:
                        pass  # Keep original if conversion fails
            
            logger.debug(f"Retrieved {len(metrics)} records for metric '{metric_name}' in experiment {experiment_id}")
            return metrics
            
        except (ExperimentConfigurationError, ExperimentNotFoundError):
            raise
        except Exception as e:
            error_msg = f"Failed to retrieve metric history for '{metric_name}' in experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def query_experiments(self, status: str = None, tags: List[str] = None, 
                         start_date: date = None, end_date: date = None,
                         limit: int = None, offset: int = 0, 
                         include_metrics: bool = False, include_tags: bool = True) -> List[Dict[str, Any]]:
        """
        Query experiments based on filters.
        
        Args:
            status: Filter by experiment status
            tags: Filter by tags (experiment must have ALL specified tags)
            start_date: Filter experiments created after this date
            end_date: Filter experiments created before this date
            limit: Maximum number of results
            offset: Number of results to skip
            include_metrics: Whether to include full metrics in results
            include_tags: Whether to include tags in results
            
        Returns:
            List of matching experiments with optional enriched data
            
        Raises:
            ExperimentConfigurationError: If filter parameters are invalid
        """
        try:
            # Validate inputs
            if status is not None and (not isinstance(status, str) or not status.strip()):
                raise ExperimentConfigurationError("Status filter must be a non-empty string")
            
            if tags is not None and not isinstance(tags, list):
                raise ExperimentConfigurationError("Tags filter must be a list of strings")
            
            if limit is not None and (not isinstance(limit, int) or limit <= 0):
                raise ExperimentConfigurationError("Limit must be a positive integer")
            
            if not isinstance(offset, int) or offset < 0:
                raise ExperimentConfigurationError("Offset must be a non-negative integer")
            
            # Flush async queue to ensure data consistency
            if self._async_enabled and self._async_logger:
                self._async_logger.flush(timeout=2.0)
            
            # Use existing query function from utils
            from .utils import query_experiments as query_experiments_db
            
            experiments = query_experiments_db(
                status=status.strip() if status else None,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset
            )
            
            # Enrich results if requested
            if include_metrics or include_tags:
                from .utils import get_experiment_metrics, get_experiment_tags
                
                for exp in experiments:
                    exp_id = exp['id']
                    
                    if include_tags:
                        exp['tags'] = get_experiment_tags(exp_id)
                        exp['tag_count'] = len(exp['tags'])
                    
                    if include_metrics:
                        exp['metrics'] = get_experiment_metrics(exp_id)
                        exp['metric_count'] = len(exp['metrics'])
            
            logger.debug(f"Query returned {len(experiments)} experiments")
            return experiments
            
        except ExperimentConfigurationError:
            raise
        except Exception as e:
            error_msg = f"Failed to query experiments: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    # =================================================================
    # Data Export and Analysis Methods
    # =================================================================
    
    def export_experiment_data(self, experiment_id: int, format: str = 'json', 
                              include_metrics: bool = True, include_config: bool = True,
                              file_path: str = None) -> Union[str, Dict[str, Any]]:
        """
        Export experiment data in specified format.
        
        Args:
            experiment_id: Experiment ID to export
            format: Export format ('json' or 'csv')
            include_metrics: Whether to include metrics data
            include_config: Whether to include configuration data
            file_path: Optional file path to save export (returns data if None)
            
        Returns:
            Exported data as string/dict, or file path if saved to file
            
        Raises:
            ExperimentConfigurationError: If inputs are invalid
            ExperimentNotFoundError: If experiment doesn't exist
        """
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if format not in ['json', 'csv']:
                raise ExperimentConfigurationError("Format must be 'json' or 'csv'")
            
            # Get experiment data
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            # Prepare export data
            export_data = {
                'id': experiment['id'],
                'name': experiment['name'],
                'description': experiment.get('description', ''),
                'status': experiment['status'],
                'timestamp': experiment['timestamp'],
                'git_commit': experiment['git_commit'],
                'git_branch': experiment['git_branch'],
                'notes': experiment.get('notes', ''),
                'tags': experiment.get('tags', [])
            }
            
            if include_config:
                export_data['config'] = experiment.get('config', {})
                export_data['environment'] = experiment.get('environment', {})
            
            if include_metrics:
                export_data['metrics'] = experiment.get('metrics', [])
            
            # Handle different export formats
            if format == 'json':
                import json
                json_data = json.dumps(export_data, indent=2, default=str)
                
                if file_path:
                    with open(file_path, 'w') as f:
                        f.write(json_data)
                    logger.info(f"Exported experiment {experiment_id} to {file_path}")
                    return file_path
                else:
                    return export_data
            
            elif format == 'csv':
                import csv
                import io
                
                output = io.StringIO()
                
                if include_metrics and experiment.get('metrics'):
                    # Export metrics as CSV
                    fieldnames = ['metric_name', 'value', 'iteration', 'timestamp']
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for metric in experiment['metrics']:
                        writer.writerow({
                            'metric_name': metric['metric_name'],
                            'value': metric['metric_value'],
                            'iteration': metric.get('iteration', ''),
                            'timestamp': metric['timestamp']
                        })
                else:
                    # Export experiment metadata as CSV
                    fieldnames = ['field', 'value']
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for key, value in export_data.items():
                        if key != 'metrics':
                            writer.writerow({'field': key, 'value': str(value)})
                
                csv_data = output.getvalue()
                output.close()
                
                if file_path:
                    with open(file_path, 'w') as f:
                        f.write(csv_data)
                    logger.info(f"Exported experiment {experiment_id} to {file_path}")
                    return file_path
                else:
                    return csv_data
            
        except (ExperimentConfigurationError, ExperimentNotFoundError):
            raise
        except Exception as e:
            error_msg = f"Failed to export experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def get_metric_statistics(self, experiment_id: int, metric_name: str) -> Dict[str, Any]:
        """
        Calculate summary statistics for a specific metric.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric to analyze
            
        Returns:
            Dictionary containing statistical summaries (min, max, mean, std, etc.)
            
        Raises:
            ExperimentConfigurationError: If inputs are invalid
            ExperimentNotFoundError: If experiment doesn't exist
        """
        try:
            # Get metric history
            metrics = self.get_metric_history(experiment_id, metric_name)
            
            if not metrics:
                return {
                    'count': 0,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'first_value': None,
                    'last_value': None,
                    'total_change': None
                }
            
            # Extract values
            values = [float(m['metric_value']) for m in metrics]
            
            # Calculate statistics
            import statistics
            stats = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'first_value': values[0],
                'last_value': values[-1],
                'total_change': values[-1] - values[0]
            }
            
            # Calculate standard deviation if we have more than one value
            if len(values) > 1:
                stats['std'] = statistics.stdev(values)
                stats['median'] = statistics.median(values)
            else:
                stats['std'] = 0.0
                stats['median'] = values[0]
            
            logger.debug(f"Calculated statistics for metric '{metric_name}' in experiment {experiment_id}")
            return stats
            
        except (ExperimentConfigurationError, ExperimentNotFoundError):
            raise
        except Exception as e:
            error_msg = f"Failed to calculate statistics for metric '{metric_name}' in experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def compare_experiments(self, experiment_ids: List[int], metric_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare metrics across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_names: Optional list of specific metrics to compare (all if None)
            
        Returns:
            Dictionary containing comparison data and statistics
            
        Raises:
            ExperimentConfigurationError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(experiment_ids, list) or len(experiment_ids) < 2:
                raise ExperimentConfigurationError("Must provide at least 2 experiment IDs")
            
            if not all(isinstance(exp_id, int) and exp_id > 0 for exp_id in experiment_ids):
                raise ExperimentConfigurationError("All experiment IDs must be positive integers")
            
            # Get experiment data
            experiments = {}
            all_metric_names = set()
            
            for exp_id in experiment_ids:
                exp = self.get_experiment(exp_id)
                if not exp:
                    raise ExperimentNotFoundError(f"Experiment {exp_id} not found")
                
                experiments[exp_id] = exp
                # Collect all unique metric names
                exp_metrics = {m['metric_name'] for m in exp.get('metrics', [])}
                all_metric_names.update(exp_metrics)
            
            # Filter metric names if specified
            if metric_names:
                if not isinstance(metric_names, list):
                    raise ExperimentConfigurationError("Metric names must be a list")
                all_metric_names = set(metric_names).intersection(all_metric_names)
            
            # Build comparison data
            comparison = {
                'experiment_ids': experiment_ids,
                'experiment_names': {exp_id: exp['name'] for exp_id, exp in experiments.items()},
                'metrics': {},
                'summary': {}
            }
            
            # Compare each metric
            for metric_name in all_metric_names:
                metric_comparison = {
                    'values': {},
                    'statistics': {},
                    'best_experiment': None,
                    'worst_experiment': None
                }
                
                final_values = {}
                
                for exp_id in experiment_ids:
                    metrics = [m for m in experiments[exp_id].get('metrics', []) 
                             if m['metric_name'] == metric_name]
                    
                    if metrics:
                        # Get the final (last) value for this metric
                        final_value = metrics[-1]['metric_value']
                        final_values[exp_id] = final_value
                        
                        # Get full statistics
                        stats = self.get_metric_statistics(exp_id, metric_name)
                        metric_comparison['values'][exp_id] = final_value
                        metric_comparison['statistics'][exp_id] = stats
                
                # Find best and worst (assuming higher is better)
                if final_values:
                    best_exp = max(final_values.keys(), key=lambda k: final_values[k])
                    worst_exp = min(final_values.keys(), key=lambda k: final_values[k])
                    metric_comparison['best_experiment'] = best_exp
                    metric_comparison['worst_experiment'] = worst_exp
                
                comparison['metrics'][metric_name] = metric_comparison
            
            # Add overall summary
            comparison['summary'] = {
                'total_experiments': len(experiment_ids),
                'common_metrics': len(all_metric_names),
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Compared {len(experiment_ids)} experiments across {len(all_metric_names)} metrics")
            return comparison
            
        except (ExperimentConfigurationError, ExperimentNotFoundError):
            raise
        except Exception as e:
            error_msg = f"Failed to compare experiments: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def get_visualization_data(self, experiment_id: int, metric_name: str, 
                              plot_type: str = 'line') -> Dict[str, Any]:
        """
        Get data formatted for common visualization libraries.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric to visualize
            plot_type: Type of plot ('line', 'scatter', 'histogram')
            
        Returns:
            Dictionary containing data formatted for plotting libraries
            
        Raises:
            ExperimentConfigurationError: If inputs are invalid
        """
        try:
            # Validate inputs
            if plot_type not in ['line', 'scatter', 'histogram']:
                raise ExperimentConfigurationError("Plot type must be 'line', 'scatter', or 'histogram'")
            
            # Get metric history
            metrics = self.get_metric_history(experiment_id, metric_name)
            
            if not metrics:
                return {
                    'plot_type': plot_type,
                    'data': [],
                    'x_values': [],
                    'y_values': [],
                    'labels': [],
                    'title': f"{metric_name} - No Data Available"
                }
            
            # Extract data
            iterations = [m.get('iteration', i) for i, m in enumerate(metrics)]
            values = [float(m['metric_value']) for m in metrics]
            timestamps = [m['timestamp'] for m in metrics]
            
            # Format for different plot types
            plot_data = {
                'plot_type': plot_type,
                'metric_name': metric_name,
                'experiment_id': experiment_id,
                'title': f"{metric_name} - Experiment {experiment_id}",
                'xlabel': 'Iteration' if plot_type != 'histogram' else 'Value',
                'ylabel': 'Value' if plot_type != 'histogram' else 'Frequency'
            }
            
            if plot_type in ['line', 'scatter']:
                plot_data.update({
                    'x_values': iterations,
                    'y_values': values,
                    'timestamps': timestamps,
                    'data': list(zip(iterations, values, timestamps))
                })
            elif plot_type == 'histogram':
                plot_data.update({
                    'values': values,
                    'data': values,
                    'bins': min(20, len(set(values)))  # Reasonable number of bins
                })
            
            logger.debug(f"Prepared {plot_type} visualization data for metric '{metric_name}' in experiment {experiment_id}")
            return plot_data
            
        except (ExperimentConfigurationError, ExperimentNotFoundError):
            raise
        except Exception as e:
            error_msg = f"Failed to prepare visualization data: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    # =================================================================
    # Experiment Lifecycle Management Methods
    # =================================================================
    
    def start_experiment(self, experiment_id: int) -> None:
        """
        Start an experiment (set status to 'running').
        
        Args:
            experiment_id: Experiment ID
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
        """
        self.update_experiment_status(experiment_id, 'running')
        logger.info(f"Started experiment {experiment_id}")
    
    def pause_experiment(self, experiment_id: int) -> None:
        """
        Pause an experiment (set status to 'paused').
        
        Args:
            experiment_id: Experiment ID
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
        """
        self.update_experiment_status(experiment_id, 'paused')
        logger.info(f"Paused experiment {experiment_id}")
    
    def resume_experiment(self, experiment_id: int) -> None:
        """
        Resume a paused experiment (set status to 'running').
        
        Args:
            experiment_id: Experiment ID
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
        """
        self.update_experiment_status(experiment_id, 'running')
        logger.info(f"Resumed experiment {experiment_id}")
    
    def complete_experiment(self, experiment_id: int) -> None:
        """
        Mark an experiment as completed (set status to 'completed').
        
        Args:
            experiment_id: Experiment ID
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
        """
        self.update_experiment_status(experiment_id, 'completed')
        logger.info(f"Completed experiment {experiment_id}")
    
    def fail_experiment(self, experiment_id: int, error_message: str = None) -> None:
        """
        Mark an experiment as failed (set status to 'failed').
        
        Args:
            experiment_id: Experiment ID
            error_message: Optional error message to log as a note
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
        """
        self.update_experiment_status(experiment_id, 'failed')
        
        if error_message:
            self.add_note(experiment_id, f"Failed: {error_message}")
        
        logger.info(f"Failed experiment {experiment_id}")
    
    def add_note(self, experiment_id: int, note: str) -> None:
        """
        Add a note/observation to an experiment.
        
        Notes are appended to the experiment's notes field with timestamps.
        
        Args:
            experiment_id: Experiment ID
            note: Note text to add
            
        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
            ExperimentConfigurationError: If note addition fails
        """
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(note, str) or not note.strip():
                raise ExperimentConfigurationError("Note must be a non-empty string")
            
            # Get current experiment to append to existing notes
            from .utils import get_experiment_by_id, transaction_context
            
            experiment = get_experiment_by_id(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            # Create timestamped note
            timestamp = datetime.now().isoformat()
            new_note = f"[{timestamp}] {note.strip()}"
            
            # Append to existing notes
            current_notes = experiment.get('notes', '') or ''
            if current_notes:
                updated_notes = f"{current_notes}\n{new_note}"
            else:
                updated_notes = new_note
            
            # Update notes in database
            with transaction_context() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE experiments SET notes = ? WHERE id = ?
                """, (updated_notes, experiment_id))
                
                if cursor.rowcount == 0:
                    raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            logger.debug(f"Added note to experiment {experiment_id}: {note}")
            
        except ExperimentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to add note to experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    # =================================================================
    # Asynchronous Logging Implementation
    # =================================================================
    
    def _initialize_async_logging(self):
        """Initialize asynchronous logging capabilities."""
        try:
            # Get configuration parameters
            max_queue_size = self._config.get('async_max_queue_size', 10000)
            flush_interval = self._config.get('async_flush_interval', 1.0)
            batch_size = self._config.get('async_batch_size', 100)
            max_retries = self._config.get('async_max_retries', 3)
            
            # Create async logger
            self._async_logger = AsyncLogger(
                max_queue_size=max_queue_size,
                flush_interval=flush_interval,
                batch_size=batch_size,
                max_retries=max_retries
            )
            
            # Start the worker thread
            self._async_logger.start(self)
            
            logger.info(f"Async logging initialized with queue_size={max_queue_size}, "
                       f"flush_interval={flush_interval}s, batch_size={batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize async logging: {e}")
            self._async_enabled = False
            self._async_logger = None
    
    def start_async_logging(self):
        """Start asynchronous logging worker."""
        if not self._async_enabled:
            self._async_enabled = True
            self._initialize_async_logging()
        elif self._async_logger:
            logger.warning("Async logging already running")
        else:
            self._initialize_async_logging()
    
    def stop_async_logging(self, timeout: float = 10.0) -> bool:
        """
        Stop asynchronous logging worker and flush remaining items.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            bool: True if shutdown completed within timeout
        """
        if not self._async_logger:
            return True
            
        logger.info("Stopping async logging...")
        success = self._async_logger.stop(timeout=timeout)
        
        if success:
            self._async_logger = None
            self._async_enabled = False
            logger.info("Async logging stopped successfully")
        else:
            logger.warning("Async logging stop timed out")
            
        return success
    
    def flush_async_queue(self, timeout: float = 5.0) -> bool:
        """
        Flush all pending async log items.
        
        Args:
            timeout: Maximum time to wait for flush
            
        Returns:
            bool: True if all items were processed
        """
        if not self._async_logger:
            return True
            
        return self._async_logger.flush(timeout=timeout)
    
    def get_async_stats(self) -> Dict[str, Any]:
        """
        Get asynchronous logging performance statistics.
        
        Returns:
            Dict containing async logging stats
        """
        if not self._async_logger:
            return {'async_enabled': False}
            
        stats = self._async_logger.get_stats()
        stats['async_enabled'] = True
        return stats
    
    def _log_async_if_enabled(self, entry: AsyncLogEntry) -> bool:
        """
        Add entry to async queue if async logging is enabled.
        
        Args:
            entry: AsyncLogEntry to process
            
        Returns:
            bool: True if added to async queue, False if should use sync
        """
        if not self._async_enabled or not self._async_logger:
            return False
            
        return self._async_logger.add_entry(entry)
    
    # =================================================================
    # Synchronous Implementation Methods (for async entry execution)
    # =================================================================
    
    def _log_metric_sync(self, experiment_id: int, metric_name: str, 
                        value: float, iteration: int = None, 
                        timestamp: datetime = None) -> None:
        """Internal synchronous metric logging (used by async entries)."""
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ExperimentConfigurationError("Metric name must be a non-empty string")
            
            if not isinstance(value, (int, float)):
                raise ExperimentConfigurationError("Metric value must be numeric")
            
            # Set timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now()
            
            # Auto-increment iteration if not provided
            if iteration is None:
                from .utils import get_experiment_metrics
                try:
                    existing_metrics = get_experiment_metrics(experiment_id, metric_name)
                    if existing_metrics:
                        iteration = max(m.get('iteration', 0) for m in existing_metrics) + 1
                    else:
                        iteration = 0
                except Exception:
                    iteration = 0
            
            # Log the metric using existing utility function
            from .utils import add_metric_to_experiment
            add_metric_to_experiment(experiment_id, metric_name, value, iteration, timestamp)
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    exp_data = self._active_experiments[experiment_id]
                    exp_data['metrics_count'] += 1
                    exp_data['last_metric_time'] = timestamp.isoformat()
            
            logger.debug(f"Logged metric {metric_name}={value} for experiment {experiment_id}")
            
        except Exception as e:
            error_msg = f"Failed to log metric {metric_name} for experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def _log_parameter_sync(self, experiment_id: int, param_name: str, value: Any) -> None:
        """Internal synchronous parameter logging (used by async entries)."""
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(param_name, str) or not param_name.strip():
                raise ExperimentConfigurationError("Parameter name must be a non-empty string")
            
            # Encode parameter name to distinguish from metrics
            full_param_name = f"param:{param_name}"
            
            # Convert value to numeric if possible, otherwise use hash
            if isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                # For non-numeric parameters, store as hash of string representation
                value_str = json.dumps(value, default=str, sort_keys=True)
                numeric_value = float(hash(value_str) % (10**10))  # Keep reasonable range
            
            # Store as a metric with special prefix
            from .utils import add_metric_to_experiment
            add_metric_to_experiment(experiment_id, full_param_name, numeric_value, iteration=0)
            
            logger.debug(f"Logged parameter {param_name}={value} for experiment {experiment_id}")
            
        except Exception as e:
            error_msg = f"Failed to log parameter {param_name} for experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def _update_experiment_status_sync(self, experiment_id: int, status: str) -> None:
        """Internal synchronous status update (used by async entries)."""
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(status, str) or not status.strip():
                raise ExperimentConfigurationError("Status must be a non-empty string")
            
            # Update status using existing utility function
            from .utils import update_experiment_status as utils_update_status
            utils_update_status(experiment_id, status)
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    self._active_experiments[experiment_id]['status'] = status
            
            logger.debug(f"Updated experiment {experiment_id} status to {status}")
            
        except Exception as e:
            error_msg = f"Failed to update status for experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def _add_tags_sync(self, experiment_id: int, tags: List[str]) -> None:
        """Internal synchronous tag addition (used by async entries)."""
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(tags, list) or not tags:
                raise ExperimentConfigurationError("Tags must be a non-empty list")
            
            # Validate and clean tags
            valid_tags = []
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    valid_tags.append(tag.strip())
            
            if not valid_tags:
                logger.warning(f"No valid tags to add for experiment {experiment_id}")
                return
            
            # Add tags using existing utility function
            from .utils import transaction_context, add_tags_to_experiment
            
            with transaction_context() as conn:
                add_tags_to_experiment(conn, experiment_id, valid_tags)
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    exp_data = self._active_experiments[experiment_id]
                    current_tags = set(exp_data.get('tags', []))
                    current_tags.update(valid_tags)
                    exp_data['tags'] = list(current_tags)
            
            logger.debug(f"Added {len(valid_tags)} tags to experiment {experiment_id}")
            
        except Exception as e:
            error_msg = f"Failed to add tags for experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def _remove_tags_sync(self, experiment_id: int, tags: List[str]) -> None:
        """Internal synchronous tag removal (used by async entries)."""
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(tags, list) or not tags:
                raise ExperimentConfigurationError("Tags must be a non-empty list")
            
            # Remove tags using database operations
            from .utils import transaction_context
            
            with transaction_context() as conn:
                cursor = conn.cursor()
                for tag in tags:
                    if not isinstance(tag, str) or not tag.strip():
                        continue  # Skip invalid tags
                    
                    cursor.execute("""
                        DELETE FROM experiment_tags 
                        WHERE experiment_id = ? AND tag = ?
                    """, (experiment_id, tag.strip()))
            
            # Update active experiment tracking
            with self._experiment_lock:
                if experiment_id in self._active_experiments:
                    exp_data = self._active_experiments[experiment_id]
                    current_tags = set(exp_data.get('tags', []))
                    current_tags.difference_update(tags)
                    exp_data['tags'] = list(current_tags)
            
            logger.debug(f"Removed {len(tags)} tags from experiment {experiment_id}")
            
        except Exception as e:
            error_msg = f"Failed to remove tags for experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    def _add_note_sync(self, experiment_id: int, note: str) -> None:
        """Internal synchronous note addition (used by async entries)."""
        try:
            # Validate inputs
            if not isinstance(experiment_id, int) or experiment_id <= 0:
                raise ExperimentConfigurationError("Experiment ID must be a positive integer")
            
            if not isinstance(note, str) or not note.strip():
                raise ExperimentConfigurationError("Note must be a non-empty string")
            
            # Get current experiment to append to existing notes
            from .utils import get_experiment_by_id, transaction_context
            
            experiment = get_experiment_by_id(experiment_id)
            if not experiment:
                raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            # Create timestamped note
            timestamp = datetime.now().isoformat()
            new_note = f"[{timestamp}] {note.strip()}"
            
            # Append to existing notes
            current_notes = experiment.get('notes', '') or ''
            if current_notes:
                updated_notes = f"{current_notes}\n{new_note}"
            else:
                updated_notes = new_note
            
            # Update notes in database
            with transaction_context() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE experiments SET notes = ? WHERE id = ?
                """, (updated_notes, experiment_id))
                
                if cursor.rowcount == 0:
                    raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")
            
            logger.debug(f"Added note to experiment {experiment_id}: {note}")
            
        except ExperimentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to add note to experiment {experiment_id}: {e}"
            logger.error(error_msg)
            raise ExperimentConfigurationError(error_msg)
    
    # =================================================================
    # Utility Methods
    # =================================================================
    
    def is_active_experiment(self, experiment_id: int) -> bool:
        """
        Check if an experiment is currently active (being tracked).
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            bool: True if experiment is active
        """
        with self._experiment_lock:
            return experiment_id in self._active_experiments
    
    def get_active_experiments(self) -> List[int]:
        """
        Get list of currently active experiment IDs.
        
        Returns:
            List of active experiment IDs
        """
        with self._experiment_lock:
            return list(self._active_experiments.keys())
    
    def get_database_path(self) -> str:
        """Get the current database path."""
        return self._db_path
    
    def shutdown(self):
        """
        Shutdown the tracker and cleanup resources.
        Should be called when the application is shutting down.
        """
        logger.info("Shutting down ExperimentTracker")
        
        # Stop async logging if running
        if hasattr(self, '_async_logger') and self._async_logger:
            try:
                self.stop_async_logging()
            except Exception as e:
                logger.error(f"Error stopping async logging: {e}")
        
        # Close database connections
        if hasattr(self, '_connection_manager'):
            self._connection_manager.close_all_connections()
        
        # Clear active experiments
        with self._experiment_lock:
            self._active_experiments.clear()
        
        logger.info("ExperimentTracker shutdown complete")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            self.shutdown()
        except Exception as e:
            # Log error but don't raise during destruction
            logger.error(f"Error during ExperimentTracker cleanup: {e}") 