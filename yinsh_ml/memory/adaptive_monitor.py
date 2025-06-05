"""
Adaptive Memory Monitoring and Pressure Detection System

This module extends the base memory monitoring system with advanced adaptive capabilities
for resource management, including predictive pressure forecasting, resource-specific
tracking, and adaptive threshold management.
"""

import time
import threading
import logging
import weakref
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json

# Import existing memory monitoring infrastructure
from .monitor import (
    MemoryMonitor, MemoryMetrics, MemoryPressureLevel, MemoryType,
    ThresholdConfig, MemoryThreshold, MonitorConfig
)
from .events import (
    MemoryEventManager, MemoryEvent, EventType, EventSeverity, EventFilter
)

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources that consume memory."""
    MCTS_SIMULATION = "mcts_simulation"
    NEURAL_NETWORK = "neural_network"
    EXPERIENCE_BUFFER = "experience_buffer"
    GAME_STATE_POOL = "game_state_pool"
    TENSOR_POOL = "tensor_pool"
    TRAINING_PIPELINE = "training_pipeline"
    BACKGROUND_TASKS = "background_tasks"
    SYSTEM_OVERHEAD = "system_overhead"


class PredictionAccuracy(Enum):
    """Confidence levels for memory pressure predictions."""
    HIGH = "high"           # >90% confidence
    MEDIUM = "medium"       # 70-90% confidence
    LOW = "low"            # 50-70% confidence
    UNCERTAIN = "uncertain" # <50% confidence


@dataclass
class MemoryPrediction:
    """Represents a prediction of future memory pressure."""
    timestamp: float
    predicted_time: float  # When the prediction is for
    predicted_pressure: MemoryPressureLevel
    confidence: PredictionAccuracy
    contributing_factors: List[str]
    predicted_usage_bytes: int
    current_trend: float  # Bytes per second growth rate
    
    def time_until_prediction(self) -> float:
        """Get seconds until the predicted time."""
        return max(0, self.predicted_time - time.time())
    
    def is_expired(self) -> bool:
        """Check if this prediction has expired."""
        return time.time() > self.predicted_time


@dataclass 
class ResourceMemoryUsage:
    """Tracks memory usage for a specific resource type."""
    resource_type: ResourceType
    current_usage_bytes: int
    peak_usage_bytes: int
    allocation_count: int
    deallocation_count: int
    growth_rate_bytes_per_sec: float
    last_updated: float
    
    def update_usage(self, new_usage_bytes: int) -> None:
        """Update usage statistics."""
        old_usage = self.current_usage_bytes
        self.current_usage_bytes = new_usage_bytes
        self.peak_usage_bytes = max(self.peak_usage_bytes, new_usage_bytes)
        self.last_updated = time.time()
        
        # Simple growth rate calculation
        time_delta = 1.0  # Assume 1 second for simplicity
        self.growth_rate_bytes_per_sec = (new_usage_bytes - old_usage) / time_delta


class MemoryTrendAnalyzer:
    """Analyzes memory usage trends to predict future pressure events."""
    
    def __init__(self, history_window_seconds: int = 300):
        """Initialize trend analyzer.
        
        Args:
            history_window_seconds: How far back to look for trend analysis
        """
        self.history_window_seconds = history_window_seconds
        self.usage_history: deque = deque(maxlen=1000)  # (timestamp, usage_bytes)
        self.trend_history: deque = deque(maxlen=100)   # (timestamp, trend_slope)
        self._lock = threading.RLock()
        
    def add_measurement(self, timestamp: float, usage_bytes: int) -> None:
        """Add a new memory usage measurement."""
        with self._lock:
            self.usage_history.append((timestamp, usage_bytes))
            
            # Calculate trend if we have enough data
            if len(self.usage_history) >= 10:
                trend = self._calculate_current_trend()
                self.trend_history.append((timestamp, trend))
    
    def _calculate_current_trend(self) -> float:
        """Calculate the current memory usage trend in bytes per second."""
        if len(self.usage_history) < 10:
            return 0.0
            
        # Use linear regression on recent data points
        recent_data = list(self.usage_history)[-20:]  # Last 20 data points
        
        timestamps = np.array([point[0] for point in recent_data])
        usages = np.array([point[1] for point in recent_data])
        
        # Normalize timestamps to prevent numerical issues
        timestamps = timestamps - timestamps[0]
        
        # Linear regression: y = mx + b
        n = len(timestamps)
        sum_x = np.sum(timestamps)
        sum_y = np.sum(usages)
        sum_xy = np.sum(timestamps * usages)
        sum_x2 = np.sum(timestamps ** 2)
        
        # Calculate slope (trend in bytes per second)
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def predict_pressure_event(self, 
                             current_usage: int,
                             total_memory: int,
                             pressure_thresholds: Dict[MemoryPressureLevel, int],
                             prediction_horizon_seconds: int = 300) -> Optional[MemoryPrediction]:
        """Predict when the next memory pressure event might occur.
        
        Args:
            current_usage: Current memory usage in bytes
            total_memory: Total available memory in bytes
            pressure_thresholds: Thresholds for each pressure level
            prediction_horizon_seconds: How far into the future to predict
            
        Returns:
            MemoryPrediction if a pressure event is predicted, None otherwise
        """
        with self._lock:
            if len(self.usage_history) < 5:
                return None
                
            current_trend = self._calculate_current_trend()
            
            # If trend is not increasing, no pressure event predicted
            if current_trend <= 0:
                return None
                
            # Find the next pressure threshold that would be crossed
            current_time = time.time()
            predicted_pressure = None
            predicted_time = None
            
            # Check each pressure level threshold
            for pressure_level in [MemoryPressureLevel.WARNING, MemoryPressureLevel.CRITICAL, MemoryPressureLevel.EMERGENCY]:
                if pressure_level not in pressure_thresholds:
                    continue
                    
                threshold_bytes = pressure_thresholds[pressure_level]
                
                # Skip if we're already past this threshold
                if current_usage >= threshold_bytes:
                    continue
                    
                # Calculate time to reach this threshold
                bytes_to_threshold = threshold_bytes - current_usage
                if current_trend > 0:
                    time_to_threshold = bytes_to_threshold / current_trend
                    
                    # Only predict within our horizon
                    if 0 < time_to_threshold <= prediction_horizon_seconds:
                        predicted_pressure = pressure_level
                        predicted_time = current_time + time_to_threshold
                        break
            
            if predicted_pressure is None:
                return None
                
            # Calculate confidence based on trend stability
            confidence = self._calculate_prediction_confidence(current_trend)
            
            # Identify contributing factors
            contributing_factors = self._identify_contributing_factors(current_trend)
            
            return MemoryPrediction(
                timestamp=current_time,
                predicted_time=predicted_time,
                predicted_pressure=predicted_pressure,
                confidence=confidence,
                contributing_factors=contributing_factors,
                predicted_usage_bytes=int(current_usage + current_trend * (predicted_time - current_time)),
                current_trend=current_trend
            )
    
    def _calculate_prediction_confidence(self, current_trend: float) -> PredictionAccuracy:
        """Calculate confidence in prediction based on trend stability."""
        if len(self.trend_history) < 5:
            return PredictionAccuracy.UNCERTAIN
            
        # Calculate variance in recent trends
        recent_trends = [trend for _, trend in list(self.trend_history)[-10:]]
        trend_variance = np.var(recent_trends)
        trend_mean = np.mean(recent_trends)
        
        # Calculate coefficient of variation
        if abs(trend_mean) > 1e-6:
            cv = np.sqrt(trend_variance) / abs(trend_mean)
        else:
            cv = float('inf')
            
        # Assign confidence based on stability
        if cv < 0.1:
            return PredictionAccuracy.HIGH
        elif cv < 0.3:
            return PredictionAccuracy.MEDIUM
        elif cv < 0.6:
            return PredictionAccuracy.LOW
        else:
            return PredictionAccuracy.UNCERTAIN
    
    def _identify_contributing_factors(self, current_trend: float) -> List[str]:
        """Identify factors contributing to the current memory trend."""
        factors = []
        
        if current_trend > 0:
            if current_trend > 1_000_000:  # > 1MB/sec
                factors.append("rapid_memory_growth")
            elif current_trend > 100_000:  # > 100KB/sec
                factors.append("moderate_memory_growth")
            else:
                factors.append("slow_memory_growth")
        
        # Add more sophisticated factor identification here
        # This could include analysis of specific resource usage patterns
        
        return factors


class AdaptiveThresholdManager:
    """Manages adaptive threshold adjustment based on system behavior."""
    
    def __init__(self, base_config: ThresholdConfig):
        """Initialize with base threshold configuration."""
        self.base_config = base_config
        self.current_config = ThresholdConfig(
            system_memory=base_config.system_memory.copy(),
            process_memory=base_config.process_memory.copy(),
            gpu_memory=base_config.gpu_memory.copy() if base_config.gpu_memory else None,
            hysteresis_percent=base_config.hysteresis_percent
        )
        self.adaptation_history: deque = deque(maxlen=100)
        self._lock = threading.RLock()
        
    def adapt_thresholds(self, 
                        recent_pressure_events: List[Any],
                        system_volatility: float,
                        average_usage_pattern: Dict[str, float]) -> bool:
        """Adapt thresholds based on recent system behavior.
        
        Args:
            recent_pressure_events: Recent pressure level transitions
            system_volatility: Measure of how volatile memory usage has been
            average_usage_pattern: Average usage patterns for different memory types
            
        Returns:
            True if thresholds were adjusted, False otherwise
        """
        with self._lock:
            adjustments_made = False
            
            # Analyze recent pressure events
            if len(recent_pressure_events) > 10:  # Too many events - thresholds may be too sensitive
                adjustments_made |= self._make_thresholds_less_sensitive()
            elif len(recent_pressure_events) == 0:  # No events - may be too conservative
                adjustments_made |= self._make_thresholds_more_sensitive()
            
            # Adjust based on volatility
            if system_volatility > 0.3:  # High volatility
                adjustments_made |= self._increase_hysteresis()
            elif system_volatility < 0.1:  # Low volatility
                adjustments_made |= self._decrease_hysteresis()
            
            if adjustments_made:
                self.adaptation_history.append({
                    'timestamp': time.time(),
                    'reason': f'volatility_{system_volatility:.2f}_events_{len(recent_pressure_events)}',
                    'config_snapshot': self._get_config_snapshot()
                })
            
            return adjustments_made
    
    def _make_thresholds_less_sensitive(self) -> bool:
        """Increase thresholds to make them less sensitive."""
        adjustments_made = False
        
        for memory_type in [self.current_config.system_memory, self.current_config.process_memory]:
            for level, threshold in memory_type.items():
                if threshold.percentage is not None:
                    new_percentage = min(0.95, threshold.percentage + 0.02)  # Increase by 2%
                    if new_percentage != threshold.percentage:
                        threshold.percentage = new_percentage
                        adjustments_made = True
        
        return adjustments_made
    
    def _make_thresholds_more_sensitive(self) -> bool:
        """Decrease thresholds to make them more sensitive."""
        adjustments_made = False
        
        for memory_type in [self.current_config.system_memory, self.current_config.process_memory]:
            for level, threshold in memory_type.items():
                if threshold.percentage is not None:
                    new_percentage = max(0.5, threshold.percentage - 0.02)  # Decrease by 2%
                    if new_percentage != threshold.percentage:
                        threshold.percentage = new_percentage
                        adjustments_made = True
        
        return adjustments_made
    
    def _increase_hysteresis(self) -> bool:
        """Increase hysteresis to reduce oscillation."""
        new_hysteresis = min(0.15, self.current_config.hysteresis_percent + 0.01)
        if new_hysteresis != self.current_config.hysteresis_percent:
            self.current_config.hysteresis_percent = new_hysteresis
            return True
        return False
    
    def _decrease_hysteresis(self) -> bool:
        """Decrease hysteresis for more responsive thresholds."""
        new_hysteresis = max(0.02, self.current_config.hysteresis_percent - 0.01)
        if new_hysteresis != self.current_config.hysteresis_percent:
            self.current_config.hysteresis_percent = new_hysteresis
            return True
        return False
    
    def _get_config_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current configuration."""
        return {
            'system_thresholds': {
                level.name: {'percentage': threshold.percentage, 'absolute_bytes': threshold.absolute_bytes}
                for level, threshold in self.current_config.system_memory.items()
            },
            'hysteresis_percent': self.current_config.hysteresis_percent
        }


class AdaptiveMemoryMonitor(MemoryMonitor):
    """Extended memory monitor with adaptive capabilities for resource management."""
    
    def __init__(self, 
                 config: Optional[MonitorConfig] = None,
                 prediction_horizon_seconds: int = 300,
                 resource_tracking_enabled: bool = True):
        """Initialize adaptive memory monitor.
        
        Args:
            config: Base monitor configuration
            prediction_horizon_seconds: How far ahead to predict pressure events
            resource_tracking_enabled: Whether to track individual resource usage
        """
        super().__init__(config)
        
        self.prediction_horizon = prediction_horizon_seconds
        self.resource_tracking_enabled = resource_tracking_enabled
        
        # Adaptive components
        self.trend_analyzer = MemoryTrendAnalyzer(history_window_seconds=prediction_horizon_seconds)
        self.threshold_manager = AdaptiveThresholdManager(self.config.threshold_config)
        
        # Resource tracking
        self.resource_usage: Dict[ResourceType, ResourceMemoryUsage] = {}
        self.resource_allocators: Dict[ResourceType, Set[weakref.ref]] = defaultdict(set)
        self.resource_callbacks: Dict[ResourceType, List[Callable]] = defaultdict(list)
        
        # Prediction tracking
        self.current_predictions: List[MemoryPrediction] = []
        self.prediction_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = {
            'training_throughput': deque(maxlen=50),
            'allocation_latency': deque(maxlen=100),
            'pressure_correlation': deque(maxlen=100)
        }
        
        # Threading for adaptive features
        self._adaptive_lock = threading.RLock()
        self._prediction_thread: Optional[threading.Thread] = None
        self._adaptation_thread: Optional[threading.Thread] = None
        
        logger.info(f"Initialized AdaptiveMemoryMonitor with {prediction_horizon_seconds}s prediction horizon")
    
    def register_resource_allocator(self, 
                                  resource_type: ResourceType,
                                  allocator_ref: Any,
                                  callback: Optional[Callable[[MemoryPrediction], None]] = None) -> None:
        """Register a resource allocator for tracking and notifications.
        
        Args:
            resource_type: Type of resource being allocated
            allocator_ref: Reference to the allocator object
            callback: Callback for pressure predictions affecting this resource
        """
        with self._adaptive_lock:
            # Use weak reference to avoid circular dependencies
            weak_ref = weakref.ref(allocator_ref)
            self.resource_allocators[resource_type].add(weak_ref)
            
            # Initialize resource usage tracking
            if resource_type not in self.resource_usage:
                self.resource_usage[resource_type] = ResourceMemoryUsage(
                    resource_type=resource_type,
                    current_usage_bytes=0,
                    peak_usage_bytes=0,
                    allocation_count=0,
                    deallocation_count=0,
                    growth_rate_bytes_per_sec=0.0,
                    last_updated=time.time()
                )
            
            if callback:
                self.resource_callbacks[resource_type].append(callback)
            
            logger.debug(f"Registered {resource_type.value} allocator")
    
    def update_resource_usage(self, 
                            resource_type: ResourceType,
                            current_usage_bytes: int,
                            allocation_delta: int = 0) -> None:
        """Update usage statistics for a specific resource.
        
        Args:
            resource_type: Type of resource
            current_usage_bytes: Current total usage in bytes
            allocation_delta: Change in allocation count (positive for allocations, negative for deallocations)
        """
        with self._adaptive_lock:
            if resource_type not in self.resource_usage:
                # Create a dummy allocator class that can have weak references
                class DummyAllocator:
                    def __init__(self, resource_type):
                        self.resource_type = resource_type
                    
                    def __repr__(self):
                        return f"DummyAllocator({self.resource_type})"
                        
                self.register_resource_allocator(resource_type, DummyAllocator(resource_type))
            
            usage = self.resource_usage[resource_type]
            usage.update_usage(current_usage_bytes)
            
            if allocation_delta > 0:
                usage.allocation_count += allocation_delta
            elif allocation_delta < 0:
                usage.deallocation_count += abs(allocation_delta)
    
    def get_resource_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of resource usage across all tracked resources."""
        with self._adaptive_lock:
            summary = {}
            total_usage = 0
            
            for resource_type, usage in self.resource_usage.items():
                summary[resource_type.value] = {
                    'current_bytes': usage.current_usage_bytes,
                    'peak_bytes': usage.peak_usage_bytes,
                    'allocations': usage.allocation_count,
                    'deallocations': usage.deallocation_count,
                    'growth_rate_bps': usage.growth_rate_bytes_per_sec,
                    'last_updated': usage.last_updated
                }
                total_usage += usage.current_usage_bytes
            
            summary['total_tracked_usage_bytes'] = total_usage
            summary['tracking_enabled'] = self.resource_tracking_enabled
            summary['num_tracked_resources'] = len(self.resource_usage)
            
            return summary
    
    def start_adaptive_monitoring(self) -> None:
        """Start the adaptive monitoring threads."""
        if self._prediction_thread is None or not self._prediction_thread.is_alive():
            self._prediction_thread = threading.Thread(
                target=self._prediction_loop,
                name="AdaptiveMemoryPrediction",
                daemon=True
            )
            self._prediction_thread.start()
            
        if self._adaptation_thread is None or not self._adaptation_thread.is_alive():
            self._adaptation_thread = threading.Thread(
                target=self._adaptation_loop,
                name="AdaptiveMemoryThresholds", 
                daemon=True
            )
            self._adaptation_thread.start()
            
        logger.info("Started adaptive monitoring threads")
    
    def stop_adaptive_monitoring(self) -> None:
        """Stop the adaptive monitoring threads."""
        # The threads are daemon threads so they'll stop when the main process stops
        logger.info("Stopped adaptive monitoring threads")
    
    def _prediction_loop(self) -> None:
        """Background thread that generates memory pressure predictions."""
        while self.is_running():
            try:
                current_metrics = self.get_current_metrics()
                if current_metrics:
                    self._update_predictions(current_metrics)
                
                time.sleep(30)  # Update predictions every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _adaptation_loop(self) -> None:
        """Background thread that adapts thresholds based on system behavior."""
        while self.is_running():
            try:
                self._adapt_thresholds_if_needed()
                time.sleep(120)  # Adapt every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                time.sleep(300)  # Wait longer on error
    
    def _update_predictions(self, metrics: MemoryMetrics) -> None:
        """Update memory pressure predictions based on current metrics."""
        with self._adaptive_lock:
            # Add current usage to trend analyzer
            self.trend_analyzer.add_measurement(metrics.timestamp, metrics.system_used)
            
            # Calculate pressure thresholds in bytes
            pressure_thresholds = {}
            for level, threshold in self.threshold_manager.current_config.system_memory.items():
                if threshold.percentage is not None:
                    pressure_thresholds[level] = int(metrics.system_total * threshold.percentage)
                elif threshold.absolute_bytes is not None:
                    pressure_thresholds[level] = threshold.absolute_bytes
            
            # Generate prediction
            prediction = self.trend_analyzer.predict_pressure_event(
                current_usage=metrics.system_used,
                total_memory=metrics.system_total,
                pressure_thresholds=pressure_thresholds,
                prediction_horizon_seconds=self.prediction_horizon
            )
            
            # Clean up expired predictions
            current_time = time.time()
            self.current_predictions = [p for p in self.current_predictions if not p.is_expired()]
            
            # Add new prediction if we have one
            if prediction:
                self.current_predictions.append(prediction)
                self.prediction_history.append(prediction)
                
                # Notify resource allocators
                self._notify_resource_allocators(prediction)
                
                logger.info(f"Generated pressure prediction: {prediction.predicted_pressure.name} "
                          f"in {prediction.time_until_prediction():.1f}s "
                          f"(confidence: {prediction.confidence.value})")
    
    def _adapt_thresholds_if_needed(self) -> None:
        """Adapt thresholds based on recent system behavior."""
        with self._adaptive_lock:
            # Get recent pressure events
            recent_events = []
            if hasattr(self, 'pressure_state'):
                recent_events = [
                    t for t in self.pressure_state.get_transition_history(max_entries=20)
                    if t.timestamp > datetime.now() - timedelta(minutes=30)
                ]
            
            # Calculate system volatility
            volatility = self._calculate_system_volatility()
            
            # Get average usage patterns
            usage_patterns = self._calculate_usage_patterns()
            
            # Attempt threshold adaptation
            adapted = self.threshold_manager.adapt_thresholds(
                recent_pressure_events=recent_events,
                system_volatility=volatility,
                average_usage_pattern=usage_patterns
            )
            
            if adapted:
                # Update the base monitor's threshold config
                self.config.threshold_config = self.threshold_manager.current_config
                logger.info(f"Adapted memory thresholds (volatility: {volatility:.2f})")
    
    def _calculate_system_volatility(self) -> float:
        """Calculate a measure of system memory usage volatility."""
        metrics_history = self.get_metrics_history(max_entries=50)
        if len(metrics_history) < 10:
            return 0.0
            
        # Calculate coefficient of variation for memory usage percentages
        percentages = [m.system_percentage for m in metrics_history]
        mean_percentage = np.mean(percentages)
        std_percentage = np.std(percentages)
        
        return std_percentage / mean_percentage if mean_percentage > 0 else 0.0
    
    def _calculate_usage_patterns(self) -> Dict[str, float]:
        """Calculate average usage patterns for different memory types."""
        metrics_history = self.get_metrics_history(max_entries=100)
        if not metrics_history:
            return {}
            
        return {
            'system_avg_percentage': np.mean([m.system_percentage for m in metrics_history]),
            'process_avg_percentage': np.mean([m.process_percentage for m in metrics_history]),
            'system_peak_percentage': np.max([m.system_percentage for m in metrics_history]),
        }
    
    def _notify_resource_allocators(self, prediction: MemoryPrediction) -> None:
        """Notify registered resource allocators about a prediction."""
        for resource_type, callbacks in self.resource_callbacks.items():
            for callback in callbacks:
                try:
                    callback(prediction)
                except Exception as e:
                    logger.error(f"Error notifying {resource_type.value} callback: {e}")
    
    def get_current_predictions(self) -> List[MemoryPrediction]:
        """Get current active predictions."""
        with self._adaptive_lock:
            return [p for p in self.current_predictions if not p.is_expired()]
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction accuracy and performance."""
        with self._adaptive_lock:
            if not self.prediction_history:
                return {'total_predictions': 0}
            
            predictions = list(self.prediction_history)
            
            # Calculate accuracy metrics
            total_predictions = len(predictions)
            high_confidence = sum(1 for p in predictions if p.confidence == PredictionAccuracy.HIGH)
            accurate_predictions = 0  # Would need to track actual vs predicted
            
            # Calculate average prediction horizon
            avg_horizon = np.mean([p.time_until_prediction() for p in predictions if not p.is_expired()])
            
            return {
                'total_predictions': total_predictions,
                'high_confidence_predictions': high_confidence,
                'high_confidence_rate': high_confidence / total_predictions if total_predictions > 0 else 0,
                'average_prediction_horizon_seconds': avg_horizon,
                'active_predictions': len(self.current_predictions),
                'prediction_types': {
                    level.name: sum(1 for p in predictions if p.predicted_pressure == level)
                    for level in MemoryPressureLevel
                }
            }
    
    def record_performance_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric for correlation analysis."""
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name].append((time.time(), value))
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including base monitor stats and adaptive features."""
        base_stats = self.get_statistics()
        
        adaptive_stats = {
            'adaptive_features': {
                'prediction_horizon_seconds': self.prediction_horizon,
                'resource_tracking_enabled': self.resource_tracking_enabled,
                'predictions': self.get_prediction_statistics(),
                'resource_usage': self.get_resource_usage_summary(),
                'threshold_adaptations': len(self.threshold_manager.adaptation_history),
                'performance_metrics': {
                    name: len(values) for name, values in self.performance_metrics.items()
                }
            }
        }
        
        # Merge the statistics
        return {**base_stats, **adaptive_stats}
    
    def start(self) -> None:
        """Start the adaptive memory monitor."""
        super().start()
        self.start_adaptive_monitoring()
    
    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the adaptive memory monitor."""
        self.stop_adaptive_monitoring()
        return super().stop(timeout) 