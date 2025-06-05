"""
Memory Monitoring Service

This module provides real-time memory monitoring capabilities for the YinshML system.
It tracks system memory, process memory, and GPU memory usage with configurable 
intervals and provides historical data for analysis.
"""

import threading
import time
import logging
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import psutil
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Import the new event and logging systems
from .events import (
    MemoryEventManager, MemoryEvent, EventType, EventSeverity, 
    EventFilter
)
from .logging import StructuredMemoryLogger
from .metrics import MemoryMetricsCollector


class MemoryType(Enum):
    """Types of memory being monitored."""
    SYSTEM = "system"
    PROCESS = "process"
    GPU = "gpu"


class MemoryPressureLevel(Enum):
    """Memory pressure levels in order of severity."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class MemoryThreshold:
    """Defines a memory threshold that can be percentage or absolute bytes based."""
    percentage: Optional[float] = None  # e.g., 0.8 for 80%
    absolute_bytes: Optional[int] = None  # e.g., 8_000_000_000 for 8GB
    
    def __post_init__(self):
        """Validate threshold configuration."""
        if self.percentage is not None and (self.percentage < 0 or self.percentage > 1):
            raise ValueError("Percentage threshold must be between 0 and 1")
        if self.percentage is None and self.absolute_bytes is None:
            raise ValueError("Either percentage or absolute_bytes must be specified")
            
    def check_threshold(self, current_bytes: int, total_bytes: int) -> bool:
        """Check if the current memory usage exceeds this threshold."""
        if self.percentage is not None:
            current_percentage = current_bytes / total_bytes if total_bytes > 0 else 0
            return current_percentage >= self.percentage
        elif self.absolute_bytes is not None:
            return current_bytes >= self.absolute_bytes
        return False


@dataclass
class ThresholdConfig:
    """Configuration for memory pressure thresholds."""
    system_memory: Dict[MemoryPressureLevel, MemoryThreshold] = field(default_factory=dict)
    process_memory: Dict[MemoryPressureLevel, MemoryThreshold] = field(default_factory=dict)
    gpu_memory: Optional[Dict[MemoryPressureLevel, MemoryThreshold]] = None
    hysteresis_percent: float = 0.05  # 5% deadband to prevent oscillation
    
    def __post_init__(self):
        """Set default thresholds if not provided."""
        if not self.system_memory:
            self.system_memory = {
                MemoryPressureLevel.WARNING: MemoryThreshold(percentage=0.8),
                MemoryPressureLevel.CRITICAL: MemoryThreshold(percentage=0.9),
                MemoryPressureLevel.EMERGENCY: MemoryThreshold(percentage=0.95)
            }
            
        if not self.process_memory:
            self.process_memory = {
                MemoryPressureLevel.WARNING: MemoryThreshold(percentage=0.7),
                MemoryPressureLevel.CRITICAL: MemoryThreshold(percentage=0.85),
                MemoryPressureLevel.EMERGENCY: MemoryThreshold(percentage=0.95)
            }
            
        if self.gpu_memory is None:
            self.gpu_memory = {
                MemoryPressureLevel.WARNING: MemoryThreshold(percentage=0.8),
                MemoryPressureLevel.CRITICAL: MemoryThreshold(percentage=0.9),
                MemoryPressureLevel.EMERGENCY: MemoryThreshold(percentage=0.95)
            }
            
        if self.hysteresis_percent < 0 or self.hysteresis_percent > 0.2:
            raise ValueError("Hysteresis percent must be between 0 and 0.2 (20%)")


@dataclass
class MemoryPressureTransition:
    """Records a transition between memory pressure levels."""
    timestamp: datetime
    memory_type: MemoryType
    from_level: MemoryPressureLevel
    to_level: MemoryPressureLevel
    current_usage: float  # Current usage percentage
    trigger_threshold: Optional[float] = None  # Threshold that triggered the transition


class MemoryPressureState:
    """Manages memory pressure state with thread-safe transitions."""
    
    def __init__(self):
        """Initialize the memory pressure state manager."""
        self._states = {
            MemoryType.SYSTEM: MemoryPressureLevel.NORMAL,
            MemoryType.PROCESS: MemoryPressureLevel.NORMAL,
            MemoryType.GPU: MemoryPressureLevel.NORMAL
        }
        self._last_transitions = {
            MemoryType.SYSTEM: datetime.now(),
            MemoryType.PROCESS: datetime.now(),
            MemoryType.GPU: datetime.now()
        }
        self._transition_history: List[MemoryPressureTransition] = []
        self._lock = threading.RLock()
        self._max_history_size = 100
        
    def transition_to(self, memory_type: MemoryType, new_level: MemoryPressureLevel,
                     current_usage: float, trigger_threshold: Optional[float] = None) -> bool:
        """
        Transition to a new memory pressure level for the specified memory type.
        
        Args:
            memory_type: Type of memory (system, process, GPU)
            new_level: New pressure level to transition to
            current_usage: Current memory usage percentage
            trigger_threshold: Threshold that triggered this transition
            
        Returns:
            True if state changed, False if no change
        """
        with self._lock:
            current_level = self._states[memory_type]
            
            if new_level != current_level:
                # Record the transition
                timestamp = datetime.now()
                transition = MemoryPressureTransition(
                    timestamp=timestamp,
                    memory_type=memory_type,
                    from_level=current_level,
                    to_level=new_level,
                    current_usage=current_usage,
                    trigger_threshold=trigger_threshold
                )
                
                # Update state
                self._states[memory_type] = new_level
                self._last_transitions[memory_type] = timestamp
                self._transition_history.append(transition)
                
                # Limit history size
                if len(self._transition_history) > self._max_history_size:
                    self._transition_history = self._transition_history[-self._max_history_size:]
                
                return True
            return False
            
    def get_current_level(self, memory_type: MemoryType) -> MemoryPressureLevel:
        """Get the current pressure level for a memory type."""
        with self._lock:
            return self._states[memory_type]
            
    def get_overall_level(self) -> MemoryPressureLevel:
        """Get the highest pressure level across all memory types."""
        with self._lock:
            return max(self._states.values(), key=lambda x: x.value)
            
    def get_last_transition_time(self, memory_type: MemoryType) -> datetime:
        """Get the timestamp of the last transition for a memory type."""
        with self._lock:
            return self._last_transitions[memory_type]
            
    def get_transition_history(self, max_entries: Optional[int] = None) -> List[MemoryPressureTransition]:
        """Get the transition history."""
        with self._lock:
            history = list(self._transition_history)
            
        if max_entries is not None and len(history) > max_entries:
            return history[-max_entries:]
        return history
        
    def clear_history(self) -> None:
        """Clear the transition history."""
        with self._lock:
            self._transition_history.clear()


@dataclass
class MemoryMetrics:
    """Snapshot of memory usage at a specific time."""
    timestamp: float
    
    # System memory (in bytes)
    system_total: int
    system_available: int
    system_used: int
    system_percentage: float
    
    # Process memory (in bytes)
    process_rss: int  # Resident Set Size
    process_vms: int  # Virtual Memory Size
    process_percentage: float
    
    # GPU memory (in bytes, per device)
    gpu_memory: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'system': {
                'total': self.system_total,
                'available': self.system_available,
                'used': self.system_used,
                'percentage': self.system_percentage
            },
            'process': {
                'rss': self.process_rss,
                'vms': self.process_vms,
                'percentage': self.process_percentage
            },
            'gpu': self.gpu_memory
        }


@dataclass
class MonitorConfig:
    """Configuration for the memory monitor."""
    interval_seconds: float = 5.0  # Monitoring interval
    history_size: int = 720  # Keep 1 hour of history at 5-second intervals
    enable_gpu_monitoring: bool = True
    enable_detailed_process_info: bool = True
    enable_threshold_detection: bool = True
    enable_event_system: bool = True
    enable_structured_logging: bool = True
    enable_metrics_collection: bool = True
    threshold_config: Optional[ThresholdConfig] = None
    log_level: str = "INFO"
    log_directory: str = "./logs"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.interval_seconds <= 0:
            raise ValueError("Monitoring interval must be positive")
        if self.history_size < 1:
            raise ValueError("History size must be at least 1")
        if self.enable_threshold_detection and self.threshold_config is None:
            self.threshold_config = ThresholdConfig()


def load_threshold_config_from_env() -> ThresholdConfig:
    """Load threshold configuration from environment variables."""
    config = ThresholdConfig()
    
    # System memory thresholds
    if os.environ.get("MEMORY_SYSTEM_WARNING_THRESHOLD"):
        config.system_memory[MemoryPressureLevel.WARNING] = MemoryThreshold(
            percentage=float(os.environ["MEMORY_SYSTEM_WARNING_THRESHOLD"])
        )
    if os.environ.get("MEMORY_SYSTEM_CRITICAL_THRESHOLD"):
        config.system_memory[MemoryPressureLevel.CRITICAL] = MemoryThreshold(
            percentage=float(os.environ["MEMORY_SYSTEM_CRITICAL_THRESHOLD"])
        )
    if os.environ.get("MEMORY_SYSTEM_EMERGENCY_THRESHOLD"):
        config.system_memory[MemoryPressureLevel.EMERGENCY] = MemoryThreshold(
            percentage=float(os.environ["MEMORY_SYSTEM_EMERGENCY_THRESHOLD"])
        )
        
    # Process memory thresholds
    if os.environ.get("MEMORY_PROCESS_WARNING_THRESHOLD"):
        config.process_memory[MemoryPressureLevel.WARNING] = MemoryThreshold(
            percentage=float(os.environ["MEMORY_PROCESS_WARNING_THRESHOLD"])
        )
    if os.environ.get("MEMORY_PROCESS_CRITICAL_THRESHOLD"):
        config.process_memory[MemoryPressureLevel.CRITICAL] = MemoryThreshold(
            percentage=float(os.environ["MEMORY_PROCESS_CRITICAL_THRESHOLD"])
        )
    if os.environ.get("MEMORY_PROCESS_EMERGENCY_THRESHOLD"):
        config.process_memory[MemoryPressureLevel.EMERGENCY] = MemoryThreshold(
            percentage=float(os.environ["MEMORY_PROCESS_EMERGENCY_THRESHOLD"])
        )
        
    # Hysteresis
    if os.environ.get("MEMORY_HYSTERESIS_PERCENT"):
        config.hysteresis_percent = float(os.environ["MEMORY_HYSTERESIS_PERCENT"])
        
    return config


class MemoryMonitor:
    """
    Real-time memory monitoring service that runs in a background thread.
    
    Collects system memory, process memory, and GPU memory metrics at regular
    intervals and maintains a rolling history for analysis. Supports configurable
    thresholds for memory pressure detection with comprehensive event notification
    and structured logging.
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize the memory monitor.
        
        Args:
            config: Configuration for monitoring behavior
        """
        self.config = config or MonitorConfig()
        self._setup_logging()
        
        # Threading control
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Data storage
        self._metrics_history: deque[MemoryMetrics] = deque(maxlen=self.config.history_size)
        self._current_metrics: Optional[MemoryMetrics] = None
        
        # Pressure state management
        self._pressure_state = MemoryPressureState()
        self._pressure_callbacks: List[Callable[[MemoryPressureTransition], None]] = []
        
        # Event system
        if self.config.enable_event_system:
            self.event_manager = MemoryEventManager(max_history_size=1000)
            # Subscribe to all events for logging
            self.event_manager.subscribe(self._handle_memory_event)
        else:
            self.event_manager = None
            
        # Structured logging
        if self.config.enable_structured_logging:
            self.structured_logger = StructuredMemoryLogger(
                log_dir=self.config.log_directory,
                log_level=getattr(logging, self.config.log_level.upper()),
                enable_json=True,
                enable_csv=True
            )
        else:
            self.structured_logger = None
            
        # Metrics collection
        if self.config.enable_metrics_collection:
            self.metrics_collector = MemoryMetricsCollector(
                collection_interval=self.config.interval_seconds * 2,  # Collect less frequently
                max_samples_per_metric=self.config.history_size
            )
            # Register custom collectors
            self._register_metric_collectors()
            self.metrics_collector.start_collection()
        else:
            self.metrics_collector = None
        
        # Process and GPU initialization
        self._process = psutil.Process()
        self._gpu_available = self._initialize_gpu_monitoring()
        
        # Statistics tracking
        self._collection_count = 0
        self._collection_errors = 0
        self._last_collection_time = 0.0
        self._pressure_events_count = 0
        
        self.logger.info(f"MemoryMonitor initialized with {self.config.interval_seconds}s interval")
        if self.config.enable_threshold_detection:
            self.logger.info("Memory pressure detection enabled")
        if self.config.enable_event_system:
            self.logger.info("Event notification system enabled")
        if self.config.enable_structured_logging:
            self.logger.info("Structured logging enabled")
        if self.config.enable_metrics_collection:
            self.logger.info("Metrics collection enabled")
        
    def _setup_logging(self) -> None:
        """Set up logging for the monitor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
    def _register_metric_collectors(self) -> None:
        """Register custom metric collectors."""
        if not self.metrics_collector:
            return
            
        # System memory collectors
        self.metrics_collector.register_custom_collector(
            "system_total_mb",
            lambda: psutil.virtual_memory().total / (1024 * 1024)
        )
        self.metrics_collector.register_custom_collector(
            "system_used_mb", 
            lambda: psutil.virtual_memory().used / (1024 * 1024)
        )
        self.metrics_collector.register_custom_collector(
            "system_available_mb",
            lambda: psutil.virtual_memory().available / (1024 * 1024)
        )
        self.metrics_collector.register_custom_collector(
            "system_usage_percent",
            lambda: psutil.virtual_memory().percent
        )
        
        # Process memory collectors
        self.metrics_collector.register_custom_collector(
            "process_rss_mb",
            lambda: self._process.memory_info().rss / (1024 * 1024)
        )
        self.metrics_collector.register_custom_collector(
            "process_vms_mb",
            lambda: self._process.memory_info().vms / (1024 * 1024)
        )
        self.metrics_collector.register_custom_collector(
            "process_usage_percent",
            lambda: self._process.memory_percent()
        )
        
    def _handle_memory_event(self, event: MemoryEvent) -> None:
        """Handle memory events by logging and collecting metrics."""
        # Log the event using structured logger
        if self.structured_logger:
            self.structured_logger.log_event(event)
            
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_event_metrics(event)
            
        # Call legacy pressure callbacks if it's a pressure event
        if event.event_type in [EventType.PRESSURE_WARNING, EventType.PRESSURE_CRITICAL, 
                              EventType.PRESSURE_EMERGENCY, EventType.PRESSURE_NORMAL]:
            # Convert to legacy format for backward compatibility
            transition = MemoryPressureTransition(
                timestamp=datetime.fromtimestamp(event.timestamp),
                memory_type=MemoryType(event.memory_type),
                from_level=MemoryPressureLevel.NORMAL,  # Would need to track this properly
                to_level=self._pressure_state.get_current_level(MemoryType(event.memory_type)),
                current_usage=event.memory_percentage
            )
            
            for callback in self._pressure_callbacks:
                try:
                    callback(transition)
                except Exception as e:
                    self.logger.error(f"Error in pressure callback: {e}")
        
    def _initialize_gpu_monitoring(self) -> bool:
        """Initialize GPU monitoring if available."""
        if not self.config.enable_gpu_monitoring:
            return False
            
        gpu_available = False
        
        # Try PyTorch CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_available = True
            self.logger.info(f"GPU monitoring enabled: {torch.cuda.device_count()} CUDA devices")
            
        # Try NVML for more detailed GPU info
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                gpu_available = True
                device_count = pynvml.nvmlDeviceGetCount()
                self.logger.info(f"NVML monitoring enabled: {device_count} devices")
            except Exception as e:
                self.logger.warning(f"NVML initialization failed: {e}")
                
        if not gpu_available:
            self.logger.info("GPU monitoring disabled: No compatible GPU libraries found")
            
        return gpu_available
        
    def _collect_system_memory(self) -> Tuple[int, int, int, float]:
        """Collect system memory metrics."""
        mem = psutil.virtual_memory()
        return mem.total, mem.available, mem.used, mem.percent
        
    def _collect_process_memory(self) -> Tuple[int, int, float]:
        """Collect current process memory metrics."""
        if self.config.enable_detailed_process_info:
            # Get detailed memory info
            mem_info = self._process.memory_info()
            mem_percent = self._process.memory_percent()
            return mem_info.rss, mem_info.vms, mem_percent
        else:
            # Basic memory info only
            mem_info = self._process.memory_info()
            mem_percent = self._process.memory_percent()
            return mem_info.rss, mem_info.vms, mem_percent
            
    def _collect_gpu_memory(self) -> Dict[int, Dict[str, int]]:
        """Collect GPU memory metrics for all available devices."""
        gpu_memory = {}
        
        if not self._gpu_available:
            return gpu_memory
            
        # PyTorch CUDA memory info
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                try:
                    # Get memory info for this device
                    total = torch.cuda.get_device_properties(device_id).total_memory
                    allocated = torch.cuda.memory_allocated(device_id)
                    cached = torch.cuda.memory_reserved(device_id)
                    free = total - allocated
                    
                    gpu_memory[device_id] = {
                        'total': total,
                        'allocated': allocated,
                        'cached': cached,
                        'free': free,
                        'percentage': (allocated / total) * 100 if total > 0 else 0.0
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to collect GPU {device_id} memory: {e}")
                    
        # NVML memory info (more detailed)
        if NVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for device_id in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Update or add GPU memory info
                    if device_id not in gpu_memory:
                        gpu_memory[device_id] = {}
                        
                    gpu_memory[device_id].update({
                        'total': mem_info.total,
                        'used': mem_info.used,
                        'free': mem_info.free,
                        'percentage': (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0.0
                    })
                    
            except Exception as e:
                self.logger.warning(f"NVML GPU memory collection failed: {e}")
                
        return gpu_memory
        
    def _check_memory_pressure(self, metrics: MemoryMetrics) -> None:
        """Check memory metrics against thresholds and update pressure state."""
        if not self.config.enable_threshold_detection or not self.config.threshold_config:
            return
            
        threshold_config = self.config.threshold_config
        
        # Check system memory pressure
        self._check_single_memory_pressure(
            MemoryType.SYSTEM,
            metrics.system_used,
            metrics.system_total,
            threshold_config.system_memory,
            threshold_config.hysteresis_percent
        )
        
        # Check process memory pressure
        system_memory = psutil.virtual_memory()
        process_used_bytes = metrics.process_rss
        self._check_single_memory_pressure(
            MemoryType.PROCESS,
            process_used_bytes,
            system_memory.total,  # Use system total as reference
            threshold_config.process_memory,
            threshold_config.hysteresis_percent
        )
        
        # Check GPU memory pressure
        if self._gpu_available and threshold_config.gpu_memory:
            for device_id, gpu_info in metrics.gpu_memory.items():
                gpu_used = gpu_info.get('used', gpu_info.get('allocated', 0))
                gpu_total = gpu_info.get('total', 0)
                if gpu_total > 0:
                    self._check_single_memory_pressure(
                        MemoryType.GPU,
                        gpu_used,
                        gpu_total,
                        threshold_config.gpu_memory,
                        threshold_config.hysteresis_percent,
                        device_id=device_id
                    )
                    
    def _check_single_memory_pressure(self, memory_type: MemoryType, used_bytes: int,
                                    total_bytes: int, thresholds: Dict[MemoryPressureLevel, MemoryThreshold],
                                    hysteresis_percent: float, device_id: Optional[int] = None) -> None:
        """Check memory pressure for a single memory type."""
        if total_bytes <= 0:
            return
            
        current_percentage = used_bytes / total_bytes
        current_level = self._pressure_state.get_current_level(memory_type)
        
        # Determine new pressure level with hysteresis
        new_level = self._determine_pressure_level(
            current_percentage, 
            current_level, 
            thresholds, 
            hysteresis_percent
        )
        
        # Check for state transition
        if self._pressure_state.transition_to(memory_type, new_level, current_percentage * 100):
            self._pressure_events_count += 1
            
            # Create and publish memory event
            if self.event_manager:
                event_type_map = {
                    MemoryPressureLevel.NORMAL: EventType.PRESSURE_NORMAL,
                    MemoryPressureLevel.WARNING: EventType.PRESSURE_WARNING,
                    MemoryPressureLevel.CRITICAL: EventType.PRESSURE_CRITICAL,
                    MemoryPressureLevel.EMERGENCY: EventType.PRESSURE_EMERGENCY
                }
                
                severity_map = {
                    MemoryPressureLevel.NORMAL: EventSeverity.INFO,
                    MemoryPressureLevel.WARNING: EventSeverity.WARNING,
                    MemoryPressureLevel.CRITICAL: EventSeverity.CRITICAL,
                    MemoryPressureLevel.EMERGENCY: EventSeverity.EMERGENCY
                }
                
                event = MemoryEvent(
                    timestamp=time.time(),
                    event_type=event_type_map[new_level],
                    severity=severity_map[new_level],
                    memory_usage_bytes=used_bytes,
                    memory_total_bytes=total_bytes,
                    memory_percentage=current_percentage * 100,
                    memory_type=memory_type.value,
                    device_id=device_id,
                    details={
                        'previous_level': current_level.name,
                        'new_level': new_level.name,
                        'hysteresis_percent': hysteresis_percent
                    }
                )
                
                self.event_manager.publish(event)
            
            # Log the transition
            device_info = f" (GPU {device_id})" if device_id is not None else ""
            self.logger.warning(
                f"Memory pressure transition: {memory_type.value}{device_info} "
                f"{current_level.name} -> {new_level.name} "
                f"({current_percentage * 100:.1f}% usage)"
            )
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_pressure_transition(
                    memory_type.value, current_level.value, new_level.value
                )
                    
    def _determine_pressure_level(self, current_percentage: float, current_level: MemoryPressureLevel,
                                thresholds: Dict[MemoryPressureLevel, MemoryThreshold],
                                hysteresis_percent: float) -> MemoryPressureLevel:
        """Determine the appropriate pressure level with hysteresis."""
        # Sort levels by severity (highest first)
        sorted_levels = sorted(
            [level for level in thresholds.keys() if level != MemoryPressureLevel.NORMAL], 
            key=lambda x: x.value, 
            reverse=True
        )
        
        # Check if we should increase pressure level
        for level in sorted_levels:
            if level.value > current_level.value:
                threshold = thresholds[level]
                if threshold.percentage and current_percentage >= threshold.percentage:
                    return level
                    
        # Check if we should decrease pressure level (with hysteresis)
        if current_level != MemoryPressureLevel.NORMAL:
            # Find the threshold for current level
            current_threshold = thresholds.get(current_level)
            if current_threshold and current_threshold.percentage:
                # Apply hysteresis - require lower value to transition down
                hysteresis_threshold = current_threshold.percentage - hysteresis_percent
                if current_percentage < hysteresis_threshold:
                    # Find the appropriate lower level
                    for level in reversed(sorted_levels):
                        if level.value < current_level.value:
                            level_threshold = thresholds.get(level)
                            if level_threshold and level_threshold.percentage:
                                if current_percentage >= level_threshold.percentage - hysteresis_percent:
                                    return level
                    # If no appropriate level found, return to normal
                    return MemoryPressureLevel.NORMAL
                    
        return current_level
        
    def _collect_metrics(self) -> MemoryMetrics:
        """Collect all memory metrics at current time."""
        timestamp = time.time()
        
        try:
            # Collect system memory
            sys_total, sys_available, sys_used, sys_percent = self._collect_system_memory()
            
            # Collect process memory  
            proc_rss, proc_vms, proc_percent = self._collect_process_memory()
            
            # Collect GPU memory
            gpu_memory = self._collect_gpu_memory()
            
            metrics = MemoryMetrics(
                timestamp=timestamp,
                system_total=sys_total,
                system_available=sys_available,
                system_used=sys_used,
                system_percentage=sys_percent,
                process_rss=proc_rss,
                process_vms=proc_vms,
                process_percentage=proc_percent,
                gpu_memory=gpu_memory
            )
            
            self._collection_count += 1
            self._last_collection_time = timestamp
            
            return metrics
            
        except Exception as e:
            self._collection_errors += 1
            self.logger.error(f"Error collecting memory metrics: {e}")
            
            # Publish collection error event
            if self.event_manager:
                error_event = MemoryEvent(
                    timestamp=timestamp,
                    event_type=EventType.COLLECTION_ERROR,
                    severity=EventSeverity.WARNING,
                    memory_usage_bytes=0,
                    memory_total_bytes=0,
                    memory_percentage=0.0,
                    memory_type="system",
                    details={'error': str(e)}
                )
                self.event_manager.publish(error_event)
            
            raise
            
    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in the background thread."""
        self.logger.info("Memory monitoring thread started")
        
        # Publish monitor started event
        if self.event_manager:
            start_event = MemoryEvent(
                timestamp=time.time(),
                event_type=EventType.MONITOR_STARTED,
                severity=EventSeverity.INFO,
                memory_usage_bytes=0,
                memory_total_bytes=0,
                memory_percentage=0.0,
                memory_type="system",
                details={'interval_seconds': self.config.interval_seconds}
            )
            self.event_manager.publish(start_event)
        
        while not self._stop_event.is_set():
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                
                # Store metrics with thread safety
                with self._lock:
                    self._current_metrics = metrics
                    self._metrics_history.append(metrics)
                    
                # Check memory pressure thresholds
                if self.config.enable_threshold_detection:
                    self._check_memory_pressure(metrics)
                    
                # Log periodic summary
                if self._collection_count % 60 == 0:  # Every 5 minutes at 5-second intervals
                    self._log_summary(metrics)
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            # Wait for next collection or stop signal
            self._stop_event.wait(self.config.interval_seconds)
        
        # Publish monitor stopped event
        if self.event_manager:
            stop_event = MemoryEvent(
                timestamp=time.time(),
                event_type=EventType.MONITOR_STOPPED,
                severity=EventSeverity.INFO,
                memory_usage_bytes=0,
                memory_total_bytes=0,
                memory_percentage=0.0,
                memory_type="system",
                details={'uptime_seconds': time.time() - (self._last_collection_time - 
                                                         (self._collection_count * self.config.interval_seconds))}
            )
            self.event_manager.publish(stop_event)
            
        self.logger.info("Memory monitoring thread stopped")
        
    def _log_summary(self, metrics: MemoryMetrics) -> None:
        """Log a summary of current memory usage."""
        # Log using structured logger if available
        if self.structured_logger:
            gpu_usage = {}
            for device_id, gpu_info in metrics.gpu_memory.items():
                gpu_usage[device_id] = gpu_info.get('used', gpu_info.get('allocated', 0))
                
            self.structured_logger.log_memory_summary(
                metrics.system_used,
                metrics.system_total,
                metrics.process_rss,
                gpu_usage if gpu_usage else None
            )
        
        # Legacy logging
        self.logger.info(
            f"Memory Summary - System: {metrics.system_percentage:.1f}% "
            f"({self._format_bytes(metrics.system_used)}/{self._format_bytes(metrics.system_total)}), "
            f"Process: {metrics.process_percentage:.1f}% ({self._format_bytes(metrics.process_rss)})"
        )
        
        # Log GPU memory if available
        if metrics.gpu_memory:
            for device_id, gpu_info in metrics.gpu_memory.items():
                gpu_percent = gpu_info.get('percentage', 0.0)
                gpu_used = gpu_info.get('used', gpu_info.get('allocated', 0))
                gpu_total = gpu_info.get('total', 0)
                self.logger.info(
                    f"GPU {device_id}: {gpu_percent:.1f}% "
                    f"({self._format_bytes(gpu_used)}/{self._format_bytes(gpu_total)})"
                )
                
        # Log pressure state if threshold detection is enabled
        if self.config.enable_threshold_detection:
            system_level = self._pressure_state.get_current_level(MemoryType.SYSTEM)
            process_level = self._pressure_state.get_current_level(MemoryType.PROCESS)
            gpu_level = self._pressure_state.get_current_level(MemoryType.GPU)
            overall_level = self._pressure_state.get_overall_level()
            
            if overall_level != MemoryPressureLevel.NORMAL:
                self.logger.info(
                    f"Memory Pressure - System: {system_level.name}, "
                    f"Process: {process_level.name}, GPU: {gpu_level.name}, "
                    f"Overall: {overall_level.name}"
                )
                
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes value in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"
        
    def add_pressure_callback(self, callback: Callable[[MemoryPressureTransition], None]) -> None:
        """Add a callback to be notified of memory pressure transitions."""
        self._pressure_callbacks.append(callback)
        
    def remove_pressure_callback(self, callback: Callable[[MemoryPressureTransition], None]) -> None:
        """Remove a pressure callback."""
        if callback in self._pressure_callbacks:
            self._pressure_callbacks.remove(callback)
            
    def get_pressure_state(self) -> MemoryPressureState:
        """Get the current memory pressure state manager."""
        return self._pressure_state
        
    def subscribe_to_events(self, 
                           callback: Callable[[MemoryEvent], None],
                           event_filter: Optional[EventFilter] = None) -> Optional[str]:
        """Subscribe to memory events."""
        if not self.event_manager:
            return None
        return self.event_manager.subscribe(callback, event_filter)
        
    def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Unsubscribe from memory events."""
        if not self.event_manager:
            return False
        return self.event_manager.unsubscribe(subscription_id)
        
    def start(self) -> None:
        """Start the memory monitoring service."""
        with self._lock:
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                self.logger.warning("Monitor is already running")
                return
                
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="MemoryMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
        self.logger.info("Memory monitor started")
        
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the memory monitoring service.
        
        Args:
            timeout: Maximum time to wait for thread to stop
            
        Returns:
            True if stopped successfully, False if timeout occurred
        """
        with self._lock:
            if self._monitor_thread is None or not self._monitor_thread.is_alive():
                self.logger.info("Monitor is not running")
                return True
                
            self._stop_event.set()
            
        # Wait for thread to stop
        if self._monitor_thread:
            self._monitor_thread.join(timeout)
            if self._monitor_thread.is_alive():
                self.logger.warning(f"Monitor thread did not stop within {timeout}s")
                return False
        
        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.stop_collection()
            
        # Close structured logger
        if self.structured_logger:
            self.structured_logger.close()
                
        self.logger.info("Memory monitor stopped")
        return True
        
    def get_current_metrics(self) -> Optional[MemoryMetrics]:
        """Get the most recent memory metrics."""
        with self._lock:
            return self._current_metrics
            
    def get_metrics_history(self, max_entries: Optional[int] = None) -> List[MemoryMetrics]:
        """
        Get historical memory metrics.
        
        Args:
            max_entries: Maximum number of entries to return (None for all)
            
        Returns:
            List of memory metrics in chronological order
        """
        with self._lock:
            history = list(self._metrics_history)
            
        if max_entries is not None and len(history) > max_entries:
            return history[-max_entries:]
        return history
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            current_time = time.time()
            uptime = current_time - (self._last_collection_time - 
                                   (self._collection_count * self.config.interval_seconds))
            
            stats = {
                'uptime_seconds': uptime,
                'collections_total': self._collection_count,
                'collection_errors': self._collection_errors,
                'pressure_events_total': self._pressure_events_count,
                'last_collection_time': self._last_collection_time,
                'history_size': len(self._metrics_history),
                'max_history_size': self.config.history_size,
                'gpu_monitoring_enabled': self._gpu_available,
                'threshold_detection_enabled': self.config.enable_threshold_detection,
                'event_system_enabled': self.config.enable_event_system,
                'structured_logging_enabled': self.config.enable_structured_logging,
                'metrics_collection_enabled': self.config.enable_metrics_collection,
                'interval_seconds': self.config.interval_seconds
            }
            
            # Add pressure state information
            if self.config.enable_threshold_detection:
                stats['pressure_state'] = {
                    'system': self._pressure_state.get_current_level(MemoryType.SYSTEM).name,
                    'process': self._pressure_state.get_current_level(MemoryType.PROCESS).name,
                    'gpu': self._pressure_state.get_current_level(MemoryType.GPU).name,
                    'overall': self._pressure_state.get_overall_level().name
                }
            
            # Add event system statistics
            if self.event_manager:
                stats['event_system'] = self.event_manager.get_statistics()
                
            # Add metrics collection statistics
            if self.metrics_collector:
                stats['metrics_collection'] = self.metrics_collector.get_statistics()
                
            # Add logging statistics
            if self.structured_logger:
                stats['structured_logging'] = self.structured_logger.get_log_statistics()
                
            return stats
            
    def clear_history(self) -> None:
        """Clear the metrics history."""
        with self._lock:
            self._metrics_history.clear()
        self._pressure_state.clear_history()
        
        if self.event_manager:
            self.event_manager.clear_history()
            
        if self.metrics_collector:
            self.metrics_collector.clear_metrics()
            
        self.logger.info("Memory metrics history cleared")
        
    def export_prometheus_metrics(self) -> Optional[str]:
        """Export metrics in Prometheus format."""
        if not self.metrics_collector:
            return None
        return self.metrics_collector.export_prometheus()
        
    def export_events_json(self, filepath: str) -> int:
        """Export events to JSON file."""
        if not self.event_manager:
            return 0
        return self.event_manager.export_events_json(filepath)
        
    def export_events_csv(self, filepath: str) -> int:
        """Export events to CSV file."""
        if not self.event_manager:
            return 0
        return self.event_manager.export_events_csv(filepath)
        
    def is_running(self) -> bool:
        """Check if the monitor is currently running."""
        with self._lock:
            return (self._monitor_thread is not None and 
                   self._monitor_thread.is_alive() and 
                   not self._stop_event.is_set())
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_monitor_thread') and self._monitor_thread:
                self.stop(timeout=1.0)
        except Exception:
            pass  # Ignore errors during cleanup 