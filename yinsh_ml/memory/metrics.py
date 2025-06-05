"""
Memory Metrics Collection System

This module provides metrics collection, aggregation, and export capabilities
for memory monitoring, with support for various monitoring backends.
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from statistics import mean, median, stdev
import math

from .events import MemoryEvent, EventSeverity, EventType


@dataclass
class MetricPoint:
    """Represents a single metric measurement at a point in time."""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class MetricSeries:
    """A time series of metric points for a specific metric."""
    name: str
    description: str
    unit: str
    metric_type: str  # "gauge", "counter", "histogram"
    points: deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.points.append(point)
        
    def get_latest(self) -> Optional[MetricPoint]:
        """Get the most recent metric point."""
        return self.points[-1] if self.points else None
        
    def get_range(self, start_time: float, end_time: float) -> List[MetricPoint]:
        """Get metric points within a time range."""
        return [
            point for point in self.points
            if start_time <= point.timestamp <= end_time
        ]
        
    def calculate_statistics(self, window_seconds: int = 300) -> Dict[str, float]:
        """Calculate statistics for recent points."""
        cutoff_time = time.time() - window_seconds
        recent_points = [
            point for point in self.points
            if point.timestamp >= cutoff_time
        ]
        
        if not recent_points:
            return {}
            
        values = [point.value for point in recent_points]
        
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': mean(values),
            'median': median(values),
            'latest': values[-1]
        }
        
        if len(values) > 1:
            stats['stddev'] = stdev(values)
        else:
            stats['stddev'] = 0.0
            
        return stats


class MetricsAggregator:
    """Aggregates and analyzes metric data."""
    
    def __init__(self):
        """Initialize the aggregator."""
        self._lock = threading.RLock()
        
    def calculate_percentiles(self, values: List[float], 
                            percentiles: List[float] = [50, 90, 95, 99]) -> Dict[str, float]:
        """Calculate percentiles for a list of values."""
        if not values:
            return {}
            
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        result = {}
        for p in percentiles:
            if p < 0 or p > 100:
                continue
                
            if p == 0:
                result[f'p{p:g}'] = sorted_values[0]
            elif p == 100:
                result[f'p{p:g}'] = sorted_values[-1]
            else:
                index = (p / 100) * (n - 1)
                if index.is_integer():
                    result[f'p{p:g}'] = sorted_values[int(index)]
                else:
                    lower = sorted_values[int(math.floor(index))]
                    upper = sorted_values[int(math.ceil(index))]
                    result[f'p{p:g}'] = lower + (upper - lower) * (index - math.floor(index))
                    
        return result
        
    def calculate_rate(self, points: List[MetricPoint], window_seconds: int = 60) -> float:
        """Calculate rate of change for counter metrics."""
        if len(points) < 2:
            return 0.0
            
        # Get points within the window
        cutoff_time = time.time() - window_seconds
        recent_points = [p for p in points if p.timestamp >= cutoff_time]
        
        if len(recent_points) < 2:
            return 0.0
            
        # Sort by timestamp
        recent_points.sort(key=lambda p: p.timestamp)
        
        # Calculate rate (assuming counter always increases)
        first_point = recent_points[0]
        last_point = recent_points[-1]
        
        time_diff = last_point.timestamp - first_point.timestamp
        value_diff = last_point.value - first_point.value
        
        if time_diff <= 0:
            return 0.0
            
        return value_diff / time_diff
        
    def calculate_moving_average(self, points: List[MetricPoint], 
                               window_points: int = 10) -> Optional[float]:
        """Calculate moving average for recent points."""
        if len(points) < window_points:
            recent_points = points
        else:
            recent_points = points[-window_points:]
            
        if not recent_points:
            return None
            
        return mean([p.value for p in recent_points])


class MemoryMetricsCollector:
    """
    Collects, stores, and exports memory metrics with support for
    various monitoring backends and real-time analysis.
    """
    
    def __init__(self, 
                 collection_interval: float = 30.0,
                 max_samples_per_metric: int = 1000,
                 enable_aggregation: bool = True):
        """
        Initialize the metrics collector.
        
        Args:
            collection_interval: How often to collect metrics (seconds)
            max_samples_per_metric: Maximum number of samples to keep per metric
            enable_aggregation: Whether to enable metric aggregation
        """
        self.collection_interval = collection_interval
        self.max_samples_per_metric = max_samples_per_metric
        self.enable_aggregation = enable_aggregation
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Metrics storage
        self._metrics: Dict[str, MetricSeries] = {}
        self._custom_collectors: Dict[str, Callable[[], Union[int, float]]] = {}
        
        # Aggregation
        self.aggregator = MetricsAggregator() if enable_aggregation else None
        
        # Statistics
        self._collection_count = 0
        self._collection_errors = 0
        self._last_collection_time = 0.0
        
        # Initialize built-in metrics
        self._init_builtin_metrics()
        
    def _init_builtin_metrics(self) -> None:
        """Initialize built-in memory metrics."""
        # System memory metrics
        self.register_metric(
            "memory_system_total_bytes",
            "Total system memory in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_system_used_bytes",
            "Used system memory in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_system_available_bytes",
            "Available system memory in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_system_usage_percent",
            "System memory usage percentage",
            "percent",
            "gauge"
        )
        
        # Process memory metrics
        self.register_metric(
            "memory_process_rss_bytes",
            "Process resident set size in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_process_vms_bytes",
            "Process virtual memory size in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_process_usage_percent",
            "Process memory usage percentage",
            "percent",
            "gauge"
        )
        
        # GPU memory metrics (will be populated per device)
        self.register_metric(
            "memory_gpu_total_bytes",
            "Total GPU memory in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_gpu_used_bytes",
            "Used GPU memory in bytes",
            "bytes",
            "gauge"
        )
        
        self.register_metric(
            "memory_gpu_usage_percent",
            "GPU memory usage percentage",
            "percent",
            "gauge"
        )
        
        # Pressure metrics
        self.register_metric(
            "memory_pressure_level",
            "Current memory pressure level (0=normal, 1=warning, 2=critical, 3=emergency)",
            "level",
            "gauge"
        )
        
        self.register_metric(
            "memory_pressure_transitions_total",
            "Total number of memory pressure transitions",
            "count",
            "counter"
        )
        
        # Event metrics
        self.register_metric(
            "memory_events_total",
            "Total number of memory events",
            "count",
            "counter"
        )
        
        self.register_metric(
            "memory_collection_errors_total",
            "Total number of collection errors",
            "count",
            "counter"
        )
        
    def register_metric(self, name: str, description: str, unit: str, metric_type: str) -> None:
        """Register a new metric series."""
        with self._lock:
            self._metrics[name] = MetricSeries(
                name=name,
                description=description,
                unit=unit,
                metric_type=metric_type,
                points=deque(maxlen=self.max_samples_per_metric)
            )
            
    def register_custom_collector(self, name: str, 
                                collector_func: Callable[[], Union[int, float]]) -> None:
        """Register a custom metric collector function."""
        self._custom_collectors[name] = collector_func
        
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].add_point(value, labels)
            else:
                # Auto-register unknown metrics as gauges
                self.register_metric(name, f"Auto-registered metric: {name}", "unknown", "gauge")
                self._metrics[name].add_point(value, labels)
                
    def record_event_metrics(self, event: MemoryEvent) -> None:
        """Record metrics from a memory event."""
        # Record memory usage metrics
        labels = {
            'memory_type': event.memory_type,
            'component': event.source_component
        }
        
        if event.device_id is not None:
            labels['device_id'] = str(event.device_id)
            
        # Record event counter
        self.record_metric("memory_events_total", 1, labels)
        
        # Record memory usage for this event
        if event.memory_type == "system":
            self.record_metric("memory_system_used_bytes", event.memory_usage_bytes)
            self.record_metric("memory_system_total_bytes", event.memory_total_bytes)
            self.record_metric("memory_system_usage_percent", event.memory_percentage)
        elif event.memory_type == "process":
            self.record_metric("memory_process_rss_bytes", event.memory_usage_bytes, labels)
            self.record_metric("memory_process_usage_percent", event.memory_percentage, labels)
        elif event.memory_type == "gpu":
            gpu_labels = {**labels, 'device_id': str(event.device_id or 0)}
            self.record_metric("memory_gpu_used_bytes", event.memory_usage_bytes, gpu_labels)
            self.record_metric("memory_gpu_total_bytes", event.memory_total_bytes, gpu_labels)
            self.record_metric("memory_gpu_usage_percent", event.memory_percentage, gpu_labels)
            
    def record_pressure_transition(self, memory_type: str, 
                                 from_level: int, to_level: int) -> None:
        """Record a memory pressure transition."""
        labels = {'memory_type': memory_type, 'from_level': str(from_level), 'to_level': str(to_level)}
        self.record_metric("memory_pressure_transitions_total", 1, labels)
        self.record_metric("memory_pressure_level", to_level, {'memory_type': memory_type})
        
    def start_collection(self) -> None:
        """Start background metric collection."""
        with self._lock:
            if self._collection_thread and self._collection_thread.is_alive():
                return
                
            self._stop_event.clear()
            self._collection_thread = threading.Thread(
                target=self._collection_loop,
                name="MetricsCollector",
                daemon=True
            )
            self._collection_thread.start()
            
    def stop_collection(self, timeout: float = 5.0) -> bool:
        """Stop background metric collection."""
        self._stop_event.set()
        
        if self._collection_thread:
            self._collection_thread.join(timeout)
            return not self._collection_thread.is_alive()
        return True
        
    def _collection_loop(self) -> None:
        """Background collection loop."""
        while not self._stop_event.is_set():
            try:
                self._collect_custom_metrics()
                self._collection_count += 1
                self._last_collection_time = time.time()
            except Exception as e:
                self._collection_errors += 1
                print(f"Error in metrics collection: {e}")
                
            self._stop_event.wait(self.collection_interval)
            
    def _collect_custom_metrics(self) -> None:
        """Collect metrics from custom collector functions."""
        for name, collector_func in self._custom_collectors.items():
            try:
                value = collector_func()
                self.record_metric(name, value)
            except Exception as e:
                print(f"Error collecting custom metric {name}: {e}")
                self.record_metric("memory_collection_errors_total", 1, {'metric': name})
                
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        with self._lock:
            return self._metrics.get(name)
            
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all metric series."""
        with self._lock:
            return self._metrics.copy()
            
    def get_latest_values(self) -> Dict[str, Union[int, float]]:
        """Get the latest value for each metric."""
        with self._lock:
            result = {}
            for name, series in self._metrics.items():
                latest = series.get_latest()
                if latest:
                    result[name] = latest.value
            return result
            
    def export_prometheus(self, metric_prefix: str = "yinsh_ml") -> str:
        """
        Export metrics in Prometheus text format.
        
        Args:
            metric_prefix: Prefix for metric names
            
        Returns:
            Prometheus format metrics string
        """
        lines = []
        
        with self._lock:
            for name, series in self._metrics.items():
                if not series.points:
                    continue
                    
                latest = series.get_latest()
                if not latest:
                    continue
                    
                prometheus_name = f"{metric_prefix}_{name}"
                
                # Add HELP and TYPE comments
                lines.append(f"# HELP {prometheus_name} {series.description}")
                lines.append(f"# TYPE {prometheus_name} {series.metric_type}")
                
                # Add metric with labels
                if latest.labels:
                    label_str = ",".join([f'{k}="{v}"' for k, v in latest.labels.items()])
                    lines.append(f"{prometheus_name}{{{label_str}}} {latest.value} {int(latest.timestamp * 1000)}")
                else:
                    lines.append(f"{prometheus_name} {latest.value} {int(latest.timestamp * 1000)}")
                    
        return "\n".join(lines)
        
    def export_json(self, 
                   include_history: bool = True,
                   max_points_per_metric: Optional[int] = None) -> Dict[str, Any]:
        """
        Export metrics as JSON.
        
        Args:
            include_history: Whether to include historical data
            max_points_per_metric: Maximum points to include per metric
            
        Returns:
            JSON-serializable dictionary
        """
        result = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'collection_stats': {
                'collection_count': self._collection_count,
                'collection_errors': self._collection_errors,
                'last_collection_time': self._last_collection_time
            },
            'metrics': {}
        }
        
        with self._lock:
            for name, series in self._metrics.items():
                metric_data = {
                    'name': series.name,
                    'description': series.description,
                    'unit': series.unit,
                    'type': series.metric_type,
                    'point_count': len(series.points)
                }
                
                if series.points:
                    latest = series.get_latest()
                    metric_data['latest_value'] = latest.value
                    metric_data['latest_timestamp'] = latest.timestamp
                    
                    # Include statistics if aggregation is enabled
                    if self.aggregator:
                        metric_data['statistics'] = series.calculate_statistics()
                        
                    # Include history if requested
                    if include_history:
                        points_to_include = series.points
                        if max_points_per_metric and len(points_to_include) > max_points_per_metric:
                            points_to_include = list(points_to_include)[-max_points_per_metric:]
                        metric_data['history'] = [point.to_dict() for point in points_to_include]
                        
                result['metrics'][name] = metric_data
                
        return result
        
    def export_csv(self, filepath: str, metric_names: Optional[List[str]] = None) -> int:
        """
        Export metrics to CSV file.
        
        Args:
            filepath: Output file path
            metric_names: Specific metrics to export (None for all)
            
        Returns:
            Number of data points exported
        """
        import csv
        
        with self._lock:
            metrics_to_export = self._metrics
            if metric_names:
                metrics_to_export = {name: series for name, series in self._metrics.items() 
                                   if name in metric_names}
                
            # Collect all data points
            all_points = []
            for name, series in metrics_to_export.items():
                for point in series.points:
                    all_points.append({
                        'metric_name': name,
                        'timestamp': point.timestamp,
                        'datetime': datetime.fromtimestamp(point.timestamp).isoformat(),
                        'value': point.value,
                        'labels': json.dumps(point.labels) if point.labels else ""
                    })
                    
            # Sort by timestamp
            all_points.sort(key=lambda x: x['timestamp'])
            
            # Write to CSV
            if all_points:
                with open(filepath, 'w', newline='') as f:
                    fieldnames = ['metric_name', 'timestamp', 'datetime', 'value', 'labels']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_points)
                    
            return len(all_points)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        with self._lock:
            return {
                'collection_interval': self.collection_interval,
                'max_samples_per_metric': self.max_samples_per_metric,
                'collection_count': self._collection_count,
                'collection_errors': self._collection_errors,
                'last_collection_time': self._last_collection_time,
                'total_metrics': len(self._metrics),
                'custom_collectors': len(self._custom_collectors),
                'total_data_points': sum(len(series.points) for series in self._metrics.values()),
                'is_collecting': self._collection_thread and self._collection_thread.is_alive()
            }
            
    def clear_metrics(self, metric_names: Optional[List[str]] = None) -> None:
        """Clear metric data."""
        with self._lock:
            if metric_names:
                for name in metric_names:
                    if name in self._metrics:
                        self._metrics[name].points.clear()
            else:
                for series in self._metrics.values():
                    series.points.clear() 