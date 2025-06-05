"""
Memory Management Metrics Collection and Export System

This module provides comprehensive metrics collection for memory pools,
allocation patterns, and system performance monitoring.
"""

import time
import threading
import json
import csv
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Container for memory management metrics."""
    timestamp: float
    
    # Pool metrics
    pool_utilization: Dict[str, float]  # pool_name -> utilization %
    pool_hit_rates: Dict[str, float]    # pool_name -> hit rate %
    pool_sizes: Dict[str, int]          # pool_name -> current size
    pool_max_sizes: Dict[str, int]      # pool_name -> max size
    
    # Allocation metrics
    allocations_per_second: float
    deallocations_per_second: float
    allocation_latency_p50: float       # microseconds
    allocation_latency_p95: float       # microseconds
    allocation_latency_p99: float       # microseconds
    
    # Memory metrics
    total_memory_used: int              # bytes
    fragmentation_index: float          # 0.0 to 1.0
    memory_pressure: float              # 0.0 to 1.0
    
    # System metrics
    cpu_usage: float                    # percentage
    system_memory_usage: float          # percentage
    gc_collections: int                 # total GC collections
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


class MemoryMetricsCollector:
    """Collects and aggregates memory management metrics."""
    
    def __init__(self, memory_components=None, collection_interval: float = 5.0, history_size: int = 1000):
        self.memory_components = memory_components
        self.collection_interval = collection_interval
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics = MemoryMetrics(
            timestamp=time.time(),
            pool_utilization={},
            pool_hit_rates={},
            pool_sizes={},
            pool_max_sizes={},
            allocations_per_second=0.0,
            deallocations_per_second=0.0,
            allocation_latency_p50=0.0,
            allocation_latency_p95=0.0,
            allocation_latency_p99=0.0,
            total_memory_used=0,
            fragmentation_index=0.0,
            memory_pressure=0.0,
            cpu_usage=0.0,
            system_memory_usage=0.0,
            gc_collections=0
        )
        
        # Collection state
        self.running = False
        self._is_collecting = False
        self.collection_thread = None
        self.lock = threading.RLock()
        
        # Registered collectors
        self.pool_collectors: Dict[str, Callable] = {}
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Performance tracking
        self.allocation_times: deque = deque(maxlen=1000)
        self.last_allocation_count = 0
        self.last_deallocation_count = 0
        self.last_gc_count = 0
        
        # Auto-register pool collectors if memory_components provided
        if memory_components:
            self._setup_pool_collectors(memory_components)
    
    def _setup_pool_collectors(self, memory_components):
        """Set up automatic pool collectors from memory components."""
        try:
            # Game state pool collector
            if hasattr(memory_components, 'game_state_pool'):
                def game_state_collector():
                    pool = memory_components.game_state_pool
                    try:
                        stats = pool.get_statistics()
                        return {
                            'utilization': stats.utilization if hasattr(stats, 'utilization') else 0.0,
                            'hit_rate': stats.hit_rate if hasattr(stats, 'hit_rate') else 0.0,
                            'current_size': pool.size(),
                            'max_size': getattr(pool.config, 'max_capacity', 1000) or 1000
                        }
                    except Exception as e:
                        logger.debug(f"Error getting game state pool stats: {e}")
                        return {
                            'utilization': 0.0,
                            'hit_rate': 0.0,
                            'current_size': pool.size() if hasattr(pool, 'size') else 0,
                            'max_size': 1000
                        }
                
                self.register_pool_collector('game_state_pool', game_state_collector)
            
            # Tensor pool collector
            if hasattr(memory_components, 'tensor_pool'):
                def tensor_pool_collector():
                    pool = memory_components.tensor_pool
                    try:
                        stats = pool.get_statistics()
                        total_tensors = sum(len(shape_pool._pool) for shape_pool in pool._shape_pools.values())
                        return {
                            'utilization': min(total_tensors / 100.0, 1.0),  # Estimate
                            'hit_rate': stats.hit_rate if hasattr(stats, 'hit_rate') else 0.0,
                            'current_size': total_tensors,
                            'max_size': 1000  # Default
                        }
                    except Exception as e:
                        logger.debug(f"Error getting tensor pool stats: {e}")
                        return {
                            'utilization': 0.0,
                            'hit_rate': 0.0,
                            'current_size': 0,
                            'max_size': 1000
                        }
                
                self.register_pool_collector('tensor_pool', tensor_pool_collector)
                
        except Exception as e:
            logger.warning(f"Error setting up pool collectors: {e}")
    
    def register_pool_collector(self, pool_name: str, collector_func: Callable) -> None:
        """Register a function to collect metrics from a specific pool."""
        with self.lock:
            self.pool_collectors[pool_name] = collector_func
            logger.info(f"Registered pool collector for {pool_name}")
    
    def register_custom_collector(self, metric_name: str, collector_func: Callable) -> None:
        """Register a custom metric collector function."""
        with self.lock:
            self.custom_collectors[metric_name] = collector_func
            logger.info(f"Registered custom collector for {metric_name}")
    
    def record_allocation_time(self, duration_microseconds: float) -> None:
        """Record an allocation timing measurement."""
        with self.lock:
            self.allocation_times.append(duration_microseconds)
    
    def start_collection(self) -> None:
        """Start periodic metrics collection."""
        if self.running:
            logger.warning("Metrics collection already running")
            return
            
        self.running = True
        self._is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="MemoryMetricsCollector",
            daemon=True
        )
        self.collection_thread.start()
        logger.info("Started memory metrics collection")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self.running:
            return
            
        self.running = False
        self._is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped memory metrics collection")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in background thread."""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(1.0)  # Brief pause before retrying
    
    def _collect_metrics(self) -> None:
        """Collect all metrics and update current state."""
        with self.lock:
            timestamp = time.time()
            
            # Collect pool metrics
            pool_utilization = {}
            pool_hit_rates = {}
            pool_sizes = {}
            pool_max_sizes = {}
            
            for pool_name, collector in self.pool_collectors.items():
                try:
                    stats = collector()
                    if stats:
                        pool_utilization[pool_name] = stats.get('utilization', 0.0)
                        pool_hit_rates[pool_name] = stats.get('hit_rate', 0.0)
                        pool_sizes[pool_name] = stats.get('current_size', 0)
                        pool_max_sizes[pool_name] = stats.get('max_size', 0)
                except Exception as e:
                    logger.warning(f"Error collecting metrics from pool {pool_name}: {e}")
            
            # Calculate allocation performance metrics
            allocation_latencies = list(self.allocation_times) if self.allocation_times else [0.0]
            allocation_latencies.sort()
            n = len(allocation_latencies)
            
            allocation_latency_p50 = allocation_latencies[int(n * 0.5)] if n > 0 else 0.0
            allocation_latency_p95 = allocation_latencies[int(n * 0.95)] if n > 0 else 0.0
            allocation_latency_p99 = allocation_latencies[int(n * 0.99)] if n > 0 else 0.0
            
            # Calculate rates (simplified - would need proper tracking in production)
            allocations_per_second = len(self.allocation_times) / self.collection_interval
            deallocations_per_second = allocations_per_second * 0.95  # Estimate
            
            # Collect system metrics
            cpu_usage, system_memory_usage, gc_collections = self._collect_system_metrics()
            
            # Calculate derived metrics
            total_memory_used = sum(pool_sizes.values()) * 1024  # Estimate in bytes
            fragmentation_index = self._calculate_fragmentation_index(pool_utilization)
            memory_pressure = min(system_memory_usage / 100.0, 1.0)
            
            # Collect custom metrics
            custom_metrics = {}
            for metric_name, collector in self.custom_collectors.items():
                try:
                    custom_metrics[metric_name] = collector()
                except Exception as e:
                    logger.warning(f"Error collecting custom metric {metric_name}: {e}")
            
            # Create new metrics object
            metrics = MemoryMetrics(
                timestamp=timestamp,
                pool_utilization=pool_utilization,
                pool_hit_rates=pool_hit_rates,
                pool_sizes=pool_sizes,
                pool_max_sizes=pool_max_sizes,
                allocations_per_second=allocations_per_second,
                deallocations_per_second=deallocations_per_second,
                allocation_latency_p50=allocation_latency_p50,
                allocation_latency_p95=allocation_latency_p95,
                allocation_latency_p99=allocation_latency_p99,
                total_memory_used=total_memory_used,
                fragmentation_index=fragmentation_index,
                memory_pressure=memory_pressure,
                cpu_usage=cpu_usage,
                system_memory_usage=system_memory_usage,
                gc_collections=gc_collections,
                custom_metrics=custom_metrics
            )
            
            # Update current metrics and add to history
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
    
    def _collect_system_metrics(self) -> tuple:
        """Collect system-level metrics."""
        try:
            import psutil
            import gc
            
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            system_memory_usage = memory.percent
            gc_collections = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
            
            return cpu_usage, system_memory_usage, gc_collections
        except ImportError:
            logger.warning("psutil not available, using fallback system metrics")
            return 0.0, 0.0, 0
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
            return 0.0, 0.0, 0
    
    def _calculate_fragmentation_index(self, pool_utilization: Dict[str, float]) -> float:
        """Calculate overall memory fragmentation index."""
        if not pool_utilization:
            return 0.0
        
        # Simple fragmentation estimate based on pool utilization variance
        utilizations = list(pool_utilization.values())
        if not utilizations:
            return 0.0
        
        avg_utilization = sum(utilizations) / len(utilizations)
        variance = sum((u - avg_utilization) ** 2 for u in utilizations) / len(utilizations)
        
        # Normalize to 0-1 range (higher variance = more fragmentation)
        return min(variance / 0.25, 1.0)  # 0.25 is max expected variance
    
    def get_current_metrics(self) -> MemoryMetrics:
        """Get the most recent metrics."""
        with self.lock:
            return self.current_metrics
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[MemoryMetrics]:
        """Get metrics history, optionally filtered by time duration."""
        with self.lock:
            if duration_seconds is None:
                return list(self.metrics_history)
            
            cutoff_time = time.time() - duration_seconds
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_summary_stats(self, duration_seconds: float = 3600) -> Dict[str, Any]:
        """Get summary statistics over a time period."""
        history = self.get_metrics_history(duration_seconds)
        if not history:
            return {}
        
        # Calculate averages and trends
        avg_cpu = sum(m.cpu_usage for m in history) / len(history)
        avg_memory = sum(m.system_memory_usage for m in history) / len(history)
        avg_fragmentation = sum(m.fragmentation_index for m in history) / len(history)
        
        # Pool utilization trends
        pool_trends = defaultdict(list)
        for metrics in history:
            for pool_name, utilization in metrics.pool_utilization.items():
                pool_trends[pool_name].append(utilization)
        
        pool_avg_utilization = {
            pool: sum(values) / len(values) 
            for pool, values in pool_trends.items()
        }
        
        return {
            'period_seconds': duration_seconds,
            'sample_count': len(history),
            'avg_cpu_usage': avg_cpu,
            'avg_memory_usage': avg_memory,
            'avg_fragmentation_index': avg_fragmentation,
            'pool_avg_utilization': pool_avg_utilization,
            'latest_timestamp': history[-1].timestamp if history else 0
        }


class MetricsExporter:
    """Exports metrics in various formats for monitoring systems."""
    
    def __init__(self, metrics_collector: MemoryMetricsCollector):
        self.collector = metrics_collector
    
    def export_prometheus(self, include_labels: bool = True) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.collector.get_current_metrics()
        lines = []
        
        # Pool utilization metrics
        for pool_name, utilization in metrics.pool_utilization.items():
            if include_labels:
                lines.append(f'memory_pool_utilization{{pool="{pool_name}"}} {utilization}')
            else:
                lines.append(f'memory_pool_utilization_{pool_name} {utilization}')
        
        # Pool hit rates
        for pool_name, hit_rate in metrics.pool_hit_rates.items():
            if include_labels:
                lines.append(f'memory_pool_hit_rate{{pool="{pool_name}"}} {hit_rate}')
            else:
                lines.append(f'memory_pool_hit_rate_{pool_name} {hit_rate}')
        
        # System metrics
        lines.extend([
            f'memory_allocations_per_second {metrics.allocations_per_second}',
            f'memory_allocation_latency_p50 {metrics.allocation_latency_p50}',
            f'memory_allocation_latency_p95 {metrics.allocation_latency_p95}',
            f'memory_allocation_latency_p99 {metrics.allocation_latency_p99}',
            f'memory_fragmentation_index {metrics.fragmentation_index}',
            f'memory_pressure {metrics.memory_pressure}',
            f'system_cpu_usage {metrics.cpu_usage}',
            f'system_memory_usage {metrics.system_memory_usage}',
        ])
        
        # Custom metrics
        for metric_name, value in metrics.custom_metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f'memory_custom_{metric_name} {value}')
        
        return '\n'.join(lines)
    
    def export_json(self, include_history: bool = False, 
                   history_duration: Optional[float] = None) -> str:
        """Export metrics as JSON."""
        if include_history:
            history = self.collector.get_metrics_history(history_duration)
            data = {
                'current': asdict(self.collector.get_current_metrics()),
                'history': [asdict(m) for m in history],
                'summary': self.collector.get_summary_stats(history_duration or 3600)
            }
        else:
            data = asdict(self.collector.get_current_metrics())
        
        return json.dumps(data, indent=2)
    
    def export_csv(self, filepath: str, duration_seconds: float = 3600) -> None:
        """Export metrics history to CSV file."""
        history = self.collector.get_metrics_history(duration_seconds)
        if not history:
            logger.warning("No metrics history to export")
            return
        
        with open(filepath, 'w', newline='') as csvfile:
            # Get all possible field names from the first metrics object
            fieldnames = [
                'timestamp', 'allocations_per_second', 'deallocations_per_second',
                'allocation_latency_p50', 'allocation_latency_p95', 'allocation_latency_p99',
                'total_memory_used', 'fragmentation_index', 'memory_pressure',
                'cpu_usage', 'system_memory_usage', 'gc_collections'
            ]
            
            # Add pool-specific fields
            if history:
                for pool_name in history[0].pool_utilization.keys():
                    fieldnames.extend([
                        f'pool_{pool_name}_utilization',
                        f'pool_{pool_name}_hit_rate',
                        f'pool_{pool_name}_size'
                    ])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in history:
                row = {
                    'timestamp': metrics.timestamp,
                    'allocations_per_second': metrics.allocations_per_second,
                    'deallocations_per_second': metrics.deallocations_per_second,
                    'allocation_latency_p50': metrics.allocation_latency_p50,
                    'allocation_latency_p95': metrics.allocation_latency_p95,
                    'allocation_latency_p99': metrics.allocation_latency_p99,
                    'total_memory_used': metrics.total_memory_used,
                    'fragmentation_index': metrics.fragmentation_index,
                    'memory_pressure': metrics.memory_pressure,
                    'cpu_usage': metrics.cpu_usage,
                    'system_memory_usage': metrics.system_memory_usage,
                    'gc_collections': metrics.gc_collections
                }
                
                # Add pool-specific data
                for pool_name in metrics.pool_utilization.keys():
                    row[f'pool_{pool_name}_utilization'] = metrics.pool_utilization.get(pool_name, 0)
                    row[f'pool_{pool_name}_hit_rate'] = metrics.pool_hit_rates.get(pool_name, 0)
                    row[f'pool_{pool_name}_size'] = metrics.pool_sizes.get(pool_name, 0)
                
                writer.writerow(row)
        
        logger.info(f"Exported {len(history)} metrics records to {filepath}")
    
    def export_grafana_dashboard(self) -> Dict[str, Any]:
        """Export Grafana dashboard configuration."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "YinshML Memory Management",
                "tags": ["memory", "yinsh_ml", "performance"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "Memory Pool Utilization",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "memory_pool_utilization",
                                "legendFormat": "{{pool}} Utilization"
                            }
                        ],
                        "yAxes": [
                            {"label": "Utilization %", "max": 100, "min": 0}
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Allocation Latency",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "memory_allocation_latency_p50",
                                "legendFormat": "P50 Latency"
                            },
                            {
                                "expr": "memory_allocation_latency_p95", 
                                "legendFormat": "P95 Latency"
                            },
                            {
                                "expr": "memory_allocation_latency_p99",
                                "legendFormat": "P99 Latency"
                            }
                        ],
                        "yAxes": [
                            {"label": "Latency (μs)", "min": 0}
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Memory Fragmentation",
                        "type": "singlestat",
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "memory_fragmentation_index",
                                "legendFormat": "Fragmentation"
                            }
                        ],
                        "thresholds": "0.15,0.25",
                        "colorBackground": True
                    },
                    {
                        "id": 4,
                        "title": "Memory Pressure",
                        "type": "gauge",
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
                        "targets": [
                            {
                                "expr": "memory_pressure",
                                "legendFormat": "Pressure"
                            }
                        ],
                        "options": {
                            "thresholds": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 0.7},
                                {"color": "red", "value": 0.9}
                            ]
                        }
                    }
                ]
            }
        }
        
        return dashboard 