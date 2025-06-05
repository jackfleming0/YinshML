"""
Core benchmarking framework for performance testing.

This module provides the fundamental infrastructure for running performance
benchmarks with memory management system components.
"""

import gc
import json
import logging
import os
import psutil
import sys
import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import warnings

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark performance metrics."""
    
    # Timing metrics (nanoseconds for precision)
    duration_ns: int = 0
    setup_duration_ns: int = 0
    teardown_duration_ns: int = 0
    
    # Memory metrics
    memory_peak_mb: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_deallocated_mb: float = 0.0
    
    # System metrics
    cpu_percent: float = 0.0
    gc_collections: int = 0
    gc_objects_before: int = 0
    gc_objects_after: int = 0
    
    # Custom metrics (benchmark-specific)
    custom_metrics: Dict[str, Union[int, float, str]] = field(default_factory=dict)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'timing': {
                'duration_ms': self.duration_ns / 1_000_000,
                'setup_duration_ms': self.setup_duration_ns / 1_000_000,
                'teardown_duration_ms': self.teardown_duration_ns / 1_000_000,
            },
            'memory': {
                'peak_mb': self.memory_peak_mb,
                'start_mb': self.memory_start_mb,
                'end_mb': self.memory_end_mb,
                'allocated_mb': self.memory_allocated_mb,
                'deallocated_mb': self.memory_deallocated_mb,
                'net_change_mb': self.memory_end_mb - self.memory_start_mb,
            },
            'system': {
                'cpu_percent': self.cpu_percent,
                'gc_collections': self.gc_collections,
                'gc_objects_before': self.gc_objects_before,
                'gc_objects_after': self.gc_objects_after,
                'gc_objects_change': self.gc_objects_after - self.gc_objects_before,
            },
            'custom': self.custom_metrics,
            'errors': self.errors,
            'warnings': self.warnings
        }


@dataclass
class BenchmarkResult:
    """Results from running a benchmark case."""
    
    name: str
    description: str
    iterations: int
    timestamp: datetime
    
    # Aggregate metrics
    avg_metrics: BenchmarkMetrics
    min_metrics: BenchmarkMetrics
    max_metrics: BenchmarkMetrics
    
    # Per-iteration metrics
    iteration_metrics: List[BenchmarkMetrics] = field(default_factory=list)
    
    # Configuration and metadata
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistical summaries of the benchmark results."""
        if not self.iteration_metrics:
            return {}
        
        durations_ms = [m.duration_ns / 1_000_000 for m in self.iteration_metrics]
        memory_peaks = [m.memory_peak_mb for m in self.iteration_metrics]
        cpu_usages = [m.cpu_percent for m in self.iteration_metrics]
        
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            values = sorted(values)
            n = len(values)
            return {
                'mean': sum(values) / n,
                'median': values[n // 2],
                'min': values[0],
                'max': values[-1],
                'p95': values[int(0.95 * n)],
                'p99': values[int(0.99 * n)],
                'std_dev': (sum((x - sum(values)/n)**2 for x in values) / n) ** 0.5
            }
        
        return {
            'duration_ms': calculate_stats(durations_ms),
            'memory_peak_mb': calculate_stats(memory_peaks),
            'cpu_percent': calculate_stats(cpu_usages),
            'iterations': len(self.iteration_metrics),
            'success_rate': 1.0 - (len([m for m in self.iteration_metrics if m.errors]) / len(self.iteration_metrics))
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'iterations': self.iterations,
            'timestamp': self.timestamp.isoformat(),
            'statistics': self.calculate_statistics(),
            'config': self.config,
            'environment': self.environment,
            'avg_metrics': self.avg_metrics.to_dict(),
            'min_metrics': self.min_metrics.to_dict(),
            'max_metrics': self.max_metrics.to_dict(),
            'iteration_count': len(self.iteration_metrics)
        }


class SystemMonitor:
    """Monitors system resources during benchmark execution."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize system monitor.
        
        Args:
            sampling_interval: How often to sample system metrics (seconds)
        """
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Get process handle
        self.process = psutil.Process()
    
    def start_monitoring(self) -> None:
        """Start monitoring system resources."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._samples.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated results."""
        if not self._monitoring:
            return {}
            
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        with self._lock:
            samples = self._samples.copy()
        
        if not samples:
            return {}
        
        # Aggregate samples
        cpu_values = [s['cpu_percent'] for s in samples]
        memory_values = [s['memory_mb'] for s in samples]
        
        return {
            'cpu_percent_avg': sum(cpu_values) / len(cpu_values),
            'cpu_percent_max': max(cpu_values),
            'memory_mb_avg': sum(memory_values) / len(memory_values),
            'memory_mb_max': max(memory_values),
            'sample_count': len(samples)
        }
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                sample = {
                    'timestamp': time.time(),
                    'cpu_percent': self.process.cpu_percent(),
                    'memory_mb': self.process.memory_info().rss / 1024 / 1024
                }
                
                with self._lock:
                    self._samples.append(sample)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.warning(f"Error sampling system metrics: {e}")
            
            time.sleep(self.sampling_interval)


@contextmanager
def memory_profiler():
    """Context manager for memory profiling using tracemalloc."""
    if tracemalloc.is_tracing():
        # Already tracing, just yield current stats
        start_snapshot = tracemalloc.take_snapshot()
        try:
            yield start_snapshot
        finally:
            pass
    else:
        # Start tracing
        tracemalloc.start()
        try:
            start_snapshot = tracemalloc.take_snapshot()
            yield start_snapshot
        finally:
            tracemalloc.stop()


class BenchmarkCase(ABC):
    """Abstract base class for benchmark test cases."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize benchmark case.
        
        Args:
            name: Unique name for this benchmark
            description: Human-readable description
        """
        self.name = name
        self.description = description or name
        self._metrics: Optional[BenchmarkMetrics] = None
        self._system_monitor = SystemMonitor()
        
    @abstractmethod
    def setup(self) -> None:
        """Set up benchmark environment. Called once before iterations."""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Clean up benchmark environment. Called once after iterations."""
        pass
    
    @abstractmethod
    def run_iteration(self) -> Dict[str, Any]:
        """
        Run a single benchmark iteration.
        
        Returns:
            Dictionary of custom metrics for this iteration
        """
        pass
    
    def run_with_monitoring(self) -> BenchmarkMetrics:
        """Run a single iteration with full monitoring."""
        metrics = BenchmarkMetrics()
        
        # Pre-execution state
        gc.collect()  # Clean start
        gc_stats_before = gc.get_stats()
        gc_count_before = sum(stat['collections'] for stat in gc_stats_before)
        gc_objects_before = len(gc.get_objects())
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Start system monitoring
        self._system_monitor.start_monitoring()
        
        # Start memory profiling
        with memory_profiler() as start_snapshot:
            start_time = time.perf_counter_ns()
            
            try:
                # Run the actual benchmark iteration
                custom_metrics = self.run_iteration()
                metrics.custom_metrics.update(custom_metrics or {})
                
            except Exception as e:
                metrics.errors.append(f"Execution error: {str(e)}")
                logger.exception(f"Error in benchmark {self.name}")
            
            end_time = time.perf_counter_ns()
        
        # Stop monitoring and collect results
        system_stats = self._system_monitor.stop_monitoring()
        
        # Post-execution state
        memory_after = process.memory_info().rss / 1024 / 1024
        gc_stats_after = gc.get_stats()
        gc_count_after = sum(stat['collections'] for stat in gc_stats_after)
        gc_objects_after = len(gc.get_objects())
        
        # Calculate memory usage from tracemalloc
        try:
            end_snapshot = tracemalloc.take_snapshot()
            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            
            total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
            total_deallocated = sum(abs(stat.size_diff) for stat in top_stats if stat.size_diff < 0)
            
            metrics.memory_allocated_mb = total_allocated / 1024 / 1024
            metrics.memory_deallocated_mb = total_deallocated / 1024 / 1024
            
        except Exception as e:
            metrics.warnings.append(f"Memory profiling error: {str(e)}")
        
        # Populate metrics
        metrics.duration_ns = end_time - start_time
        metrics.memory_start_mb = memory_before
        metrics.memory_end_mb = memory_after
        metrics.memory_peak_mb = system_stats.get('memory_mb_max', memory_after)
        metrics.cpu_percent = system_stats.get('cpu_percent_avg', 0.0)
        metrics.gc_collections = gc_count_after - gc_count_before
        metrics.gc_objects_before = gc_objects_before
        metrics.gc_objects_after = gc_objects_after
        
        return metrics
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for this benchmark."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'process_id': os.getpid(),
        }


class BenchmarkSuite:
    """Collection of benchmark cases with execution orchestration."""
    
    def __init__(self, 
                 cases: List[BenchmarkCase],
                 iterations: int = 100,
                 warmup_iterations: int = 10,
                 cooldown_seconds: float = 1.0):
        """
        Initialize benchmark suite.
        
        Args:
            cases: List of benchmark cases to run
            iterations: Number of iterations per case
            warmup_iterations: Number of warmup runs (not counted)
            cooldown_seconds: Time to wait between cases
        """
        self.cases = cases
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.cooldown_seconds = cooldown_seconds
        self.results: List[BenchmarkResult] = []
    
    def run(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> List[BenchmarkResult]:
        """
        Run all benchmark cases in the suite.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of benchmark results
        """
        self.results.clear()
        total_cases = len(self.cases)
        
        for case_idx, case in enumerate(self.cases):
            if progress_callback:
                progress = case_idx / total_cases
                progress_callback(f"Running {case.name}", progress)
            
            logger.info(f"Running benchmark: {case.name}")
            result = self._run_case(case)
            self.results.append(result)
            
            # Cooldown between cases
            if case_idx < total_cases - 1:
                time.sleep(self.cooldown_seconds)
        
        if progress_callback:
            progress_callback("Completed", 1.0)
        
        return self.results
    
    def _run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single benchmark case."""
        iteration_metrics: List[BenchmarkMetrics] = []
        
        try:
            # Setup
            setup_start = time.perf_counter_ns()
            case.setup()
            setup_duration = time.perf_counter_ns() - setup_start
            
            # Warmup iterations
            logger.debug(f"Running {self.warmup_iterations} warmup iterations")
            for _ in range(self.warmup_iterations):
                try:
                    case.run_iteration()
                except Exception as e:
                    logger.warning(f"Warmup iteration failed: {e}")
            
            # Measured iterations
            logger.debug(f"Running {self.iterations} measured iterations")
            for i in range(self.iterations):
                try:
                    metrics = case.run_with_monitoring()
                    metrics.setup_duration_ns = setup_duration if i == 0 else 0
                    iteration_metrics.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Iteration {i} failed: {e}")
                    # Create error metrics
                    error_metrics = BenchmarkMetrics()
                    error_metrics.errors.append(f"Iteration {i}: {str(e)}")
                    iteration_metrics.append(error_metrics)
            
            # Teardown
            teardown_start = time.perf_counter_ns()
            case.teardown()
            teardown_duration = time.perf_counter_ns() - teardown_start
            
            # Update last metrics with teardown duration
            if iteration_metrics:
                iteration_metrics[-1].teardown_duration_ns = teardown_duration
            
        except Exception as e:
            logger.exception(f"Benchmark case {case.name} failed during setup/teardown")
            # Create minimal error result
            error_metrics = BenchmarkMetrics()
            error_metrics.errors.append(f"Setup/teardown error: {str(e)}")
            iteration_metrics = [error_metrics]
        
        # Calculate aggregate metrics
        if iteration_metrics:
            avg_metrics, min_metrics, max_metrics = self._calculate_aggregates(iteration_metrics)
        else:
            avg_metrics = min_metrics = max_metrics = BenchmarkMetrics()
        
        return BenchmarkResult(
            name=case.name,
            description=case.description,
            iterations=len(iteration_metrics),
            timestamp=datetime.now(),
            avg_metrics=avg_metrics,
            min_metrics=min_metrics,
            max_metrics=max_metrics,
            iteration_metrics=iteration_metrics,
            environment=case.get_environment_info()
        )
    
    def _calculate_aggregates(self, metrics: List[BenchmarkMetrics]) -> Tuple[BenchmarkMetrics, BenchmarkMetrics, BenchmarkMetrics]:
        """Calculate aggregate metrics from iterations."""
        if not metrics:
            return BenchmarkMetrics(), BenchmarkMetrics(), BenchmarkMetrics()
        
        # Filter out error iterations for aggregation
        valid_metrics = [m for m in metrics if not m.errors]
        if not valid_metrics:
            return metrics[0], metrics[0], metrics[0]
        
        def safe_avg(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0
        
        def safe_min(values: List[float]) -> float:
            return min(values) if values else 0.0
        
        def safe_max(values: List[float]) -> float:
            return max(values) if values else 0.0
        
        # Extract values for aggregation
        durations = [m.duration_ns for m in valid_metrics]
        memory_peaks = [m.memory_peak_mb for m in valid_metrics]
        memory_allocated = [m.memory_allocated_mb for m in valid_metrics]
        cpu_percents = [m.cpu_percent for m in valid_metrics]
        gc_collections = [m.gc_collections for m in valid_metrics]
        
        avg_metrics = BenchmarkMetrics(
            duration_ns=int(safe_avg(durations)),
            memory_peak_mb=safe_avg(memory_peaks),
            memory_allocated_mb=safe_avg(memory_allocated),
            cpu_percent=safe_avg(cpu_percents),
            gc_collections=int(safe_avg(gc_collections))
        )
        
        min_metrics = BenchmarkMetrics(
            duration_ns=int(safe_min(durations)),
            memory_peak_mb=safe_min(memory_peaks),
            memory_allocated_mb=safe_min(memory_allocated),
            cpu_percent=safe_min(cpu_percents),
            gc_collections=int(safe_min(gc_collections))
        )
        
        max_metrics = BenchmarkMetrics(
            duration_ns=int(safe_max(durations)),
            memory_peak_mb=safe_max(memory_peaks),
            memory_allocated_mb=safe_max(memory_allocated),
            cpu_percent=safe_max(cpu_percents),
            gc_collections=int(safe_max(gc_collections))
        )
        
        return avg_metrics, min_metrics, max_metrics


class BenchmarkRunner:
    """High-level interface for running benchmarks with configuration."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(output_dir, "benchmark.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_suite(self, suite: BenchmarkSuite, name: str = "benchmark") -> str:
        """
        Run a benchmark suite and save results.
        
        Args:
            suite: Benchmark suite to run
            name: Name for the result files
            
        Returns:
            Path to the results file
        """
        logger.info(f"Starting benchmark suite: {name}")
        start_time = datetime.now()
        
        def progress_callback(message: str, progress: float):
            logger.info(f"{message} ({progress:.1%})")
        
        results = suite.run(progress_callback)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Create summary
        summary = {
            'suite_name': name,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'case_count': len(results),
            'total_iterations': sum(r.iterations for r in results),
            'results': [r.to_dict() for r in results]
        }
        
        # Save results
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"{name}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Benchmark suite completed in {duration}")
        logger.info(f"Results saved to: {results_file}")
        
        return results_file 