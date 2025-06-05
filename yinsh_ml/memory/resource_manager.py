"""
Dynamic Batch Size and Worker Pool Manager

This module provides adaptive resource management that responds to memory pressure
events by dynamically adjusting batch sizes, worker counts, and other resource
allocation parameters to maintain system stability and performance.
"""

import time
import threading
import logging
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
import numpy as np

# Import adaptive monitoring components
from .adaptive_monitor import (
    AdaptiveMemoryMonitor, MemoryPrediction, ResourceType, PredictionAccuracy
)
from .monitor import MemoryPressureLevel, MemoryMetrics
from .events import MemoryEvent, EventType, EventSeverity

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of resource scaling."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Strategies for resource scaling."""
    CONSERVATIVE = "conservative"  # Small, gradual changes
    MODERATE = "moderate"         # Balanced approach
    AGGRESSIVE = "aggressive"     # Large, rapid changes


@dataclass
class BatchSizeConfig:
    """Configuration for dynamic batch sizing."""
    min_batch_size: int = 8
    max_batch_size: int = 512
    default_batch_size: int = 32
    scaling_factor: float = 1.5
    min_scaling_factor: float = 0.7
    memory_efficiency_target: float = 0.8  # Target memory utilization
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_batch_size <= 0:
            raise ValueError("Minimum batch size must be positive")
        if self.max_batch_size < self.min_batch_size:
            raise ValueError("Maximum batch size must be >= minimum batch size")
        if not (0.1 <= self.scaling_factor <= 3.0):
            raise ValueError("Scaling factor must be between 0.1 and 3.0")
        if not (0.5 <= self.memory_efficiency_target <= 0.95):
            raise ValueError("Memory efficiency target must be between 0.5 and 0.95")


@dataclass
class WorkerPoolConfig:
    """Configuration for dynamic worker pool."""
    min_workers: int = 1
    max_workers: int = 8
    default_workers: int = 4
    scaling_threshold: float = 0.75  # Memory threshold to trigger scaling
    scale_up_delay: float = 30.0     # Seconds to wait before scaling up
    scale_down_delay: float = 60.0   # Seconds to wait before scaling down
    worker_memory_overhead: int = 100_000_000  # Estimated memory per worker (100MB)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_workers <= 0:
            raise ValueError("Minimum workers must be positive")
        if self.max_workers < self.min_workers:
            raise ValueError("Maximum workers must be >= minimum workers")
        if not (0.1 <= self.scaling_threshold <= 0.9):
            raise ValueError("Scaling threshold must be between 0.1 and 0.9")


@dataclass
class ResourceAllocation:
    """Current resource allocation state."""
    batch_size: int
    worker_count: int
    estimated_memory_usage: int
    efficiency_score: float
    last_updated: float
    scaling_direction: ScalingDirection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'batch_size': self.batch_size,
            'worker_count': self.worker_count,
            'estimated_memory_usage': self.estimated_memory_usage,
            'efficiency_score': self.efficiency_score,
            'last_updated': self.last_updated,
            'scaling_direction': self.scaling_direction.value
        }


@dataclass
class ScalingEvent:
    """Records a resource scaling event for analysis."""
    timestamp: float
    resource_type: str  # "batch_size" or "worker_count"
    old_value: int
    new_value: int
    trigger: str  # What triggered the scaling
    memory_pressure: MemoryPressureLevel
    predicted_improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'resource_type': self.resource_type,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'trigger': self.trigger,
            'memory_pressure': self.memory_pressure.name,
            'predicted_improvement': self.predicted_improvement
        }


class BatchSizeCalculator:
    """Calculates optimal batch sizes based on memory constraints."""
    
    def __init__(self, config: BatchSizeConfig):
        """Initialize batch size calculator."""
        self.config = config
        self.memory_per_sample_history: deque = deque(maxlen=50)
        self.performance_history: deque = deque(maxlen=100)
        self._lock = threading.RLock()
        
    def calculate_optimal_batch_size(self, 
                                   available_memory: int,
                                   current_batch_size: int,
                                   memory_per_sample: int,
                                   current_pressure: MemoryPressureLevel) -> Tuple[int, float]:
        """Calculate optimal batch size given memory constraints.
        
        Args:
            available_memory: Available memory in bytes
            current_batch_size: Current batch size
            memory_per_sample: Estimated memory per sample in bytes
            current_pressure: Current memory pressure level
            
        Returns:
            Tuple of (optimal_batch_size, confidence_score)
        """
        with self._lock:
            # Record memory per sample for trend analysis
            if memory_per_sample > 0:
                self.memory_per_sample_history.append(memory_per_sample)
            
            # Get average memory per sample from recent history
            if self.memory_per_sample_history:
                avg_memory_per_sample = np.mean(list(self.memory_per_sample_history))
            else:
                avg_memory_per_sample = memory_per_sample or 10_000_000  # 10MB default
            
            # Calculate theoretical maximum batch size
            if avg_memory_per_sample > 0:
                theoretical_max = int(available_memory * self.config.memory_efficiency_target / avg_memory_per_sample)
                theoretical_max = max(1, theoretical_max)  # At least 1
            else:
                theoretical_max = self.config.max_batch_size
            
            # Apply pressure-based adjustments
            pressure_multiplier = self._get_pressure_multiplier(current_pressure)
            pressure_adjusted_max = int(theoretical_max * pressure_multiplier)
            
            # Apply configuration constraints
            optimal_batch_size = max(
                self.config.min_batch_size,
                min(self.config.max_batch_size, pressure_adjusted_max)
            )
            
            # Implement gradual scaling to prevent thrashing
            if current_batch_size > 0:
                optimal_batch_size = self._apply_gradual_scaling(current_batch_size, optimal_batch_size)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence()
            
            return optimal_batch_size, confidence
    
    def _get_pressure_multiplier(self, pressure: MemoryPressureLevel) -> float:
        """Get batch size multiplier based on memory pressure."""
        multipliers = {
            MemoryPressureLevel.NORMAL: 1.0,
            MemoryPressureLevel.WARNING: 0.8,
            MemoryPressureLevel.CRITICAL: 0.6,
            MemoryPressureLevel.EMERGENCY: 0.4
        }
        return multipliers.get(pressure, 0.5)
    
    def _apply_gradual_scaling(self, current: int, target: int) -> int:
        """Apply gradual scaling to prevent thrashing."""
        if current == target:
            return target
            
        # Calculate maximum change (25% of current value)
        max_change = max(1, int(current * 0.25))
        
        if target > current:
            # Scaling up
            return min(target, current + max_change)
        else:
            # Scaling down
            return max(target, current - max_change)
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in batch size calculation."""
        if len(self.memory_per_sample_history) < 5:
            return 0.3  # Low confidence with little data
        
        # Calculate coefficient of variation for memory per sample
        values = list(self.memory_per_sample_history)
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
        
        # Convert CV to confidence (lower variation = higher confidence)
        confidence = max(0.1, min(0.9, 1.0 - cv))
        return confidence
    
    def record_performance(self, batch_size: int, throughput: float, memory_usage: int) -> None:
        """Record performance metrics for batch size optimization."""
        with self._lock:
            self.performance_history.append({
                'timestamp': time.time(),
                'batch_size': batch_size,
                'throughput': throughput,
                'memory_usage': memory_usage,
                'efficiency': throughput / memory_usage if memory_usage > 0 else 0
            })


class WorkerPoolManager:
    """Manages dynamic worker pool scaling based on memory constraints."""
    
    def __init__(self, config: WorkerPoolConfig):
        """Initialize worker pool manager."""
        self.config = config
        self.current_workers = config.default_workers
        self.last_scale_time = 0.0
        self.scaling_events: deque = deque(maxlen=100)
        self.worker_utilization_history: deque = deque(maxlen=50)
        self._lock = threading.RLock()
        
    def calculate_optimal_workers(self, 
                                available_memory: int,
                                current_workers: int,
                                current_pressure: MemoryPressureLevel,
                                workload_demand: float = 1.0) -> Tuple[int, str]:
        """Calculate optimal number of workers.
        
        Args:
            available_memory: Available memory in bytes
            current_workers: Current number of workers
            current_pressure: Current memory pressure level
            workload_demand: Workload demand factor (0.0 to 2.0)
            
        Returns:
            Tuple of (optimal_workers, reasoning)
        """
        with self._lock:
            current_time = time.time()
            
            # Calculate memory-based maximum workers
            memory_based_max = max(1, int(available_memory / self.config.worker_memory_overhead))
            
            # Apply pressure-based constraints
            pressure_adjusted_max = self._apply_pressure_constraints(memory_based_max, current_pressure)
            
            # Apply workload demand adjustments
            demand_adjusted = max(1, int(pressure_adjusted_max * workload_demand))
            
            # Apply configuration limits
            optimal_workers = max(
                self.config.min_workers,
                min(self.config.max_workers, demand_adjusted)
            )
            
            # Check scaling delays to prevent thrashing
            if optimal_workers != current_workers:
                can_scale, delay_reason = self._can_scale_now(optimal_workers > current_workers, current_time)
                if not can_scale:
                    return current_workers, delay_reason
            
            # Apply gradual scaling
            if current_workers > 0:
                optimal_workers = self._apply_gradual_worker_scaling(current_workers, optimal_workers)
            
            # Generate reasoning
            reasoning = self._generate_scaling_reasoning(
                current_workers, optimal_workers, current_pressure, workload_demand
            )
            
            return optimal_workers, reasoning
    
    def _apply_pressure_constraints(self, max_workers: int, pressure: MemoryPressureLevel) -> int:
        """Apply memory pressure constraints to worker count."""
        pressure_multipliers = {
            MemoryPressureLevel.NORMAL: 1.0,
            MemoryPressureLevel.WARNING: 0.9,
            MemoryPressureLevel.CRITICAL: 0.7,
            MemoryPressureLevel.EMERGENCY: 0.5
        }
        multiplier = pressure_multipliers.get(pressure, 0.5)
        return max(1, int(max_workers * multiplier))
    
    def _can_scale_now(self, scaling_up: bool, current_time: float) -> Tuple[bool, str]:
        """Check if scaling is allowed based on delay constraints."""
        time_since_last_scale = current_time - self.last_scale_time
        
        if scaling_up:
            required_delay = self.config.scale_up_delay
            if time_since_last_scale < required_delay:
                return False, f"Scale-up delay: {required_delay - time_since_last_scale:.1f}s remaining"
        else:
            required_delay = self.config.scale_down_delay
            if time_since_last_scale < required_delay:
                return False, f"Scale-down delay: {required_delay - time_since_last_scale:.1f}s remaining"
        
        return True, "Scaling allowed"
    
    def _apply_gradual_worker_scaling(self, current: int, target: int) -> int:
        """Apply gradual scaling for worker count."""
        if current == target:
            return target
        
        # For workers, scale by at most 1 at a time to be conservative
        if target > current:
            return current + 1
        else:
            return current - 1
    
    def _generate_scaling_reasoning(self, current: int, target: int, 
                                  pressure: MemoryPressureLevel, demand: float) -> str:
        """Generate human-readable reasoning for scaling decision."""
        if current == target:
            return f"Maintaining {current} workers (pressure: {pressure.name}, demand: {demand:.1f})"
        elif target > current:
            return f"Scaling up from {current} to {target} workers (demand: {demand:.1f})"
        else:
            return f"Scaling down from {current} to {target} workers (pressure: {pressure.name})"
    
    def record_scaling_event(self, old_workers: int, new_workers: int, 
                           trigger: str, pressure: MemoryPressureLevel) -> None:
        """Record a worker scaling event."""
        with self._lock:
            event = ScalingEvent(
                timestamp=time.time(),
                resource_type="worker_count",
                old_value=old_workers,
                new_value=new_workers,
                trigger=trigger,
                memory_pressure=pressure,
                predicted_improvement=abs(new_workers - old_workers) / max(old_workers, 1)
            )
            self.scaling_events.append(event)
            self.last_scale_time = event.timestamp
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get statistics about worker scaling events."""
        with self._lock:
            if not self.scaling_events:
                return {'total_events': 0}
            
            events = list(self.scaling_events)
            scale_ups = sum(1 for e in events if e.new_value > e.old_value)
            scale_downs = sum(1 for e in events if e.new_value < e.old_value)
            
            return {
                'total_events': len(events),
                'scale_ups': scale_ups,
                'scale_downs': scale_downs,
                'avg_change_magnitude': np.mean([abs(e.new_value - e.old_value) for e in events]),
                'recent_events': [e.to_dict() for e in events[-5:]]
            }


class DynamicResourceManager:
    """Main resource manager that coordinates batch sizing and worker pool management."""
    
    def __init__(self, 
                 memory_monitor: AdaptiveMemoryMonitor,
                 batch_config: Optional[BatchSizeConfig] = None,
                 worker_config: Optional[WorkerPoolConfig] = None,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.MODERATE):
        """Initialize dynamic resource manager.
        
        Args:
            memory_monitor: Adaptive memory monitor for pressure events
            batch_config: Batch size configuration
            worker_config: Worker pool configuration
            scaling_strategy: Scaling strategy to use
        """
        self.memory_monitor = memory_monitor
        self.batch_config = batch_config or BatchSizeConfig()
        self.worker_config = worker_config or WorkerPoolConfig()
        self.scaling_strategy = scaling_strategy
        
        # Resource managers
        self.batch_calculator = BatchSizeCalculator(self.batch_config)
        self.worker_manager = WorkerPoolManager(self.worker_config)
        
        # Current state
        self.current_allocation = ResourceAllocation(
            batch_size=self.batch_config.default_batch_size,
            worker_count=self.worker_config.default_workers,
            estimated_memory_usage=0,
            efficiency_score=0.0,
            last_updated=time.time(),
            scaling_direction=ScalingDirection.STABLE
        )
        
        # Telemetry and monitoring
        self.allocation_history: deque = deque(maxlen=200)
        self.performance_metrics: deque = deque(maxlen=100)
        self.adjustment_events: deque = deque(maxlen=100)
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Register with memory monitor
        self._register_with_monitor()
        
        logger.info(f"Initialized DynamicResourceManager with {scaling_strategy.value} scaling")
    
    def _register_with_monitor(self) -> None:
        """Register callbacks with the memory monitor."""
        # Register for memory pressure predictions
        self.memory_monitor.register_resource_allocator(
            ResourceType.TRAINING_PIPELINE,
            self,
            self._handle_memory_prediction
        )
        
        # Subscribe to memory pressure events
        self.memory_monitor.subscribe_to_events(
            self._handle_memory_event,
            event_filter=None  # Listen to all events
        )
    
    def _handle_memory_prediction(self, prediction: MemoryPrediction) -> None:
        """Handle memory pressure predictions."""
        try:
            if prediction.confidence in [PredictionAccuracy.HIGH, PredictionAccuracy.MEDIUM]:
                logger.info(f"Received memory prediction: {prediction.predicted_pressure.name} "
                          f"in {prediction.time_until_prediction():.1f}s")
                
                # Proactively adjust resources based on prediction
                self._proactive_adjustment(prediction)
                
        except Exception as e:
            logger.error(f"Error handling memory prediction: {e}")
    
    def _handle_memory_event(self, event: MemoryEvent) -> None:
        """Handle memory pressure events."""
        try:
            if event.event_type in [EventType.PRESSURE_WARNING, EventType.PRESSURE_CRITICAL, 
                                  EventType.PRESSURE_EMERGENCY]:
                logger.warning(f"Memory pressure event: {event.event_type.value}")
                
                # Reactive adjustment based on current pressure
                self._reactive_adjustment(event)
                
        except Exception as e:
            logger.error(f"Error handling memory event: {e}")
    
    def _proactive_adjustment(self, prediction: MemoryPrediction) -> None:
        """Make proactive resource adjustments based on predictions."""
        with self._lock:
            current_metrics = self.memory_monitor.get_current_metrics()
            if not current_metrics:
                return
            
            # Calculate adjustments with prediction in mind
            time_to_pressure = prediction.time_until_prediction()
            urgency_factor = max(0.1, min(1.0, 60.0 / max(time_to_pressure, 1.0)))  # More urgent if sooner
            
            new_allocation = self._calculate_resource_allocation(
                current_metrics, 
                prediction.predicted_pressure,
                urgency_factor=urgency_factor
            )
            
            if self._should_apply_allocation(new_allocation):
                self._apply_allocation(new_allocation, f"Proactive (prediction: {prediction.predicted_pressure.name})")
    
    def _reactive_adjustment(self, event: MemoryEvent) -> None:
        """Make reactive resource adjustments based on current pressure."""
        with self._lock:
            current_metrics = self.memory_monitor.get_current_metrics()
            if not current_metrics:
                return
            
            # Determine pressure level from event
            pressure_mapping = {
                EventType.PRESSURE_WARNING: MemoryPressureLevel.WARNING,
                EventType.PRESSURE_CRITICAL: MemoryPressureLevel.CRITICAL,
                EventType.PRESSURE_EMERGENCY: MemoryPressureLevel.EMERGENCY
            }
            pressure_level = pressure_mapping.get(event.event_type, MemoryPressureLevel.NORMAL)
            
            new_allocation = self._calculate_resource_allocation(
                current_metrics, 
                pressure_level,
                urgency_factor=1.0  # Maximum urgency for reactive adjustments
            )
            
            if self._should_apply_allocation(new_allocation):
                self._apply_allocation(new_allocation, f"Reactive ({event.event_type.value})")
    
    def _calculate_resource_allocation(self, 
                                     metrics: MemoryMetrics,
                                     pressure_level: MemoryPressureLevel,
                                     urgency_factor: float = 1.0) -> ResourceAllocation:
        """Calculate optimal resource allocation."""
        # Calculate available memory
        available_memory = metrics.system_available
        
        # Estimate current memory per sample (simplified)
        current_memory_per_sample = self._estimate_memory_per_sample()
        
        # Calculate optimal batch size
        optimal_batch, batch_confidence = self.batch_calculator.calculate_optimal_batch_size(
            available_memory=available_memory,
            current_batch_size=self.current_allocation.batch_size,
            memory_per_sample=current_memory_per_sample,
            current_pressure=pressure_level
        )
        
        # Calculate workload demand (simplified)
        workload_demand = self._estimate_workload_demand()
        
        # Calculate optimal worker count
        optimal_workers, worker_reasoning = self.worker_manager.calculate_optimal_workers(
            available_memory=available_memory,
            current_workers=self.current_allocation.worker_count,
            current_pressure=pressure_level,
            workload_demand=workload_demand
        )
        
        # Estimate memory usage and efficiency
        estimated_memory = optimal_batch * current_memory_per_sample * optimal_workers
        efficiency_score = min(1.0, batch_confidence * (available_memory - estimated_memory) / available_memory)
        
        # Determine scaling direction
        batch_change = optimal_batch - self.current_allocation.batch_size
        worker_change = optimal_workers - self.current_allocation.worker_count
        
        if batch_change > 0 or worker_change > 0:
            scaling_direction = ScalingDirection.UP
        elif batch_change < 0 or worker_change < 0:
            scaling_direction = ScalingDirection.DOWN
        else:
            scaling_direction = ScalingDirection.STABLE
        
        return ResourceAllocation(
            batch_size=optimal_batch,
            worker_count=optimal_workers,
            estimated_memory_usage=estimated_memory,
            efficiency_score=efficiency_score,
            last_updated=time.time(),
            scaling_direction=scaling_direction
        )
    
    def _estimate_memory_per_sample(self) -> int:
        """Estimate memory usage per training sample."""
        if self.batch_calculator.memory_per_sample_history:
            return int(np.mean(list(self.batch_calculator.memory_per_sample_history)))
        else:
            # Default estimate based on typical game state + neural network overhead
            return 10_000_000  # 10MB per sample (conservative estimate)
    
    def _estimate_workload_demand(self) -> float:
        """Estimate current workload demand factor."""
        # This could be based on queue sizes, throughput targets, etc.
        # For now, return a moderate demand
        return 1.0
    
    def _should_apply_allocation(self, new_allocation: ResourceAllocation) -> bool:
        """Determine if new allocation should be applied."""
        # Don't apply if no significant change
        batch_change = abs(new_allocation.batch_size - self.current_allocation.batch_size)
        worker_change = abs(new_allocation.worker_count - self.current_allocation.worker_count)
        
        if batch_change == 0 and worker_change == 0:
            return False
        
        # Apply minimum change thresholds
        min_batch_change = max(1, int(self.current_allocation.batch_size * 0.1))  # 10% minimum
        if batch_change > 0 and batch_change < min_batch_change:
            new_allocation.batch_size = self.current_allocation.batch_size
        
        # Always apply worker changes (they're already gradual)
        
        return True
    
    def _apply_allocation(self, allocation: ResourceAllocation, trigger: str) -> None:
        """Apply new resource allocation."""
        with self._lock:
            old_allocation = self.current_allocation
            
            # Record scaling events
            if allocation.batch_size != old_allocation.batch_size:
                batch_event = ScalingEvent(
                    timestamp=time.time(),
                    resource_type="batch_size",
                    old_value=old_allocation.batch_size,
                    new_value=allocation.batch_size,
                    trigger=trigger,
                    memory_pressure=self.memory_monitor.get_pressure_state().get_overall_level(),
                    predicted_improvement=allocation.efficiency_score - old_allocation.efficiency_score
                )
                self.adjustment_events.append(batch_event)
            
            if allocation.worker_count != old_allocation.worker_count:
                self.worker_manager.record_scaling_event(
                    old_allocation.worker_count,
                    allocation.worker_count,
                    trigger,
                    self.memory_monitor.get_pressure_state().get_overall_level()
                )
            
            # Update current allocation
            self.current_allocation = allocation
            self.allocation_history.append(allocation)
            
            # Log the change
            logger.info(f"Applied resource allocation: batch_size={allocation.batch_size}, "
                       f"workers={allocation.worker_count}, trigger='{trigger}'")
    
    def get_current_allocation(self) -> ResourceAllocation:
        """Get current resource allocation."""
        with self._lock:
            return self.current_allocation
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get current resource allocation recommendations."""
        allocation = self.get_current_allocation()
        return {
            'batch_size': allocation.batch_size,
            'worker_count': allocation.worker_count,
            'estimated_memory_usage_bytes': allocation.estimated_memory_usage,
            'efficiency_score': allocation.efficiency_score,
            'scaling_direction': allocation.scaling_direction.value,
            'last_updated': allocation.last_updated
        }
    
    def record_performance(self, batch_size: int, worker_count: int, 
                         throughput: float, memory_usage: int) -> None:
        """Record performance metrics for optimization."""
        with self._lock:
            self.performance_metrics.append({
                'timestamp': time.time(),
                'batch_size': batch_size,
                'worker_count': worker_count,
                'throughput': throughput,
                'memory_usage': memory_usage,
                'efficiency': throughput / memory_usage if memory_usage > 0 else 0
            })
            
            # Also record with batch calculator
            self.batch_calculator.record_performance(batch_size, throughput, memory_usage)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about resource management."""
        with self._lock:
            return {
                'current_allocation': self.current_allocation.to_dict(),
                'scaling_strategy': self.scaling_strategy.value,
                'batch_sizing': {
                    'config': {
                        'min_batch_size': self.batch_config.min_batch_size,
                        'max_batch_size': self.batch_config.max_batch_size,
                        'default_batch_size': self.batch_config.default_batch_size
                    },
                    'memory_per_sample_samples': len(self.batch_calculator.memory_per_sample_history),
                    'performance_samples': len(self.batch_calculator.performance_history)
                },
                'worker_management': {
                    'config': {
                        'min_workers': self.worker_config.min_workers,
                        'max_workers': self.worker_config.max_workers,
                        'default_workers': self.worker_config.default_workers
                    },
                    'scaling_stats': self.worker_manager.get_scaling_statistics()
                },
                'adjustment_events': len(self.adjustment_events),
                'allocation_history_size': len(self.allocation_history),
                'performance_metrics_size': len(self.performance_metrics),
                'recent_adjustments': [e.to_dict() for e in list(self.adjustment_events)[-5:]]
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the resource manager.
        
        Returns:
            Dictionary containing current resource allocation status
        """
        with self._lock:
            return {
                'monitoring_active': getattr(self, '_running', False),
                'current_allocation': self.current_allocation.to_dict(),
                'last_scaling_event': self.adjustment_events[-1].to_dict() if self.adjustment_events else None,
                'batch_calculator': {
                    'memory_samples': len(self.batch_calculator.memory_per_sample_history),
                    'performance_samples': len(self.batch_calculator.performance_history),
                },
                'worker_manager': {
                    'scaling_events': len(self.worker_manager.scaling_events),
                    'last_scaling_time': self.worker_manager.last_scale_time
                },
                'total_adjustments': len(self.adjustment_events),
                'allocation_history_size': len(self.allocation_history)
            }
    
    def start_monitoring(self) -> None:
        """Start the resource monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._running = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="ResourceManagerMonitoring",
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info("Started resource manager monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the resource monitoring thread."""
        self._running = False
        logger.info("Stopped resource manager monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for periodic adjustments."""
        while self._running:
            try:
                # Periodic health check and adjustment
                current_metrics = self.memory_monitor.get_current_metrics()
                if current_metrics:
                    current_pressure = self.memory_monitor.get_pressure_state().get_overall_level()
                    
                    # Only adjust if we're not in normal pressure state
                    if current_pressure != MemoryPressureLevel.NORMAL:
                        new_allocation = self._calculate_resource_allocation(
                            current_metrics, current_pressure, urgency_factor=0.5
                        )
                        
                        if self._should_apply_allocation(new_allocation):
                            self._apply_allocation(new_allocation, "Periodic monitoring")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in resource manager monitoring loop: {e}")
                time.sleep(30)  # Shorter sleep on error 