#!/usr/bin/env python3
"""
Comprehensive stress testing for YinshML experiment tracking system.

This script simulates large-scale usage patterns to identify performance
bottlenecks and validate system behavior under load.
"""

import os
import sys
import time
import json
import random
import threading
import multiprocessing
import psutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# Add the parent directory to the path to import yinsh_ml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.tracking import ExperimentTracker
from yinsh_ml.tracking.database import ExperimentDatabase
from yinsh_ml.tracking.utils import get_connection


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    num_experiments: int = 1000
    num_concurrent_threads: int = 10
    metrics_per_experiment: int = 100
    iterations_per_experiment: int = 50
    concurrent_readers: int = 5
    test_duration_minutes: int = 10
    memory_monitoring_interval: float = 1.0
    database_path: str = "stress_test.db"
    cleanup_after_test: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during stress testing."""
    timestamp: str
    memory_usage_mb: float
    cpu_percent: float
    database_size_mb: float
    active_connections: int
    experiments_created: int
    metrics_logged: int
    avg_response_time_ms: float
    errors_count: int


@dataclass
class StressTestResults:
    """Results from stress testing."""
    config: StressTestConfig
    start_time: str
    end_time: str
    total_duration_seconds: float
    experiments_created: int
    metrics_logged: int
    total_operations: int
    operations_per_second: float
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    database_final_size_mb: float
    errors: List[str]
    performance_timeline: List[PerformanceMetrics]
    success_rate: float


class StressTestMonitor:
    """Monitors system performance during stress testing."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.performance_data: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent()
                
                # Get database size
                db_size_mb = 0
                if Path(self.config.database_path).exists():
                    db_size_mb = Path(self.config.database_path).stat().st_size / (1024 * 1024)
                
                # Count active connections (simplified)
                active_connections = 1  # Placeholder - would need more sophisticated tracking
                
                metrics = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    memory_usage_mb=memory_mb,
                    cpu_percent=cpu_percent,
                    database_size_mb=db_size_mb,
                    active_connections=active_connections,
                    experiments_created=0,  # Will be updated by test runner
                    metrics_logged=0,       # Will be updated by test runner
                    avg_response_time_ms=0.0,  # Will be updated by test runner
                    errors_count=0          # Will be updated by test runner
                )
                
                self.performance_data.append(metrics)
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
            
            time.sleep(self.config.memory_monitoring_interval)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics."""
        if not self.performance_data:
            return {}
        
        memory_values = [m.memory_usage_mb for m in self.performance_data]
        cpu_values = [m.cpu_percent for m in self.performance_data]
        
        return {
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'peak_cpu_percent': max(cpu_values),
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'final_db_size_mb': self.performance_data[-1].database_size_mb if self.performance_data else 0
        }


class ExperimentWorker:
    """Worker for creating experiments and logging metrics."""
    
    def __init__(self, worker_id: int, config: StressTestConfig):
        self.worker_id = worker_id
        self.config = config
        self.tracker = None
        self.errors: List[str] = []
        self.operations_count = 0
        self.response_times: List[float] = []
    
    def setup(self):
        """Set up the worker."""
        try:
            self.tracker = ExperimentTracker(db_path=self.config.database_path)
        except Exception as e:
            self.errors.append(f"Worker {self.worker_id} setup failed: {e}")
    
    def create_experiment_with_metrics(self, exp_index: int) -> Tuple[int, int]:
        """Create an experiment and log metrics."""
        experiments_created = 0
        metrics_logged = 0
        
        try:
            start_time = time.time()
            
            # Create experiment
            exp_name = f"stress_test_exp_{self.worker_id}_{exp_index}"
            config = {
                'model_type': random.choice(['cnn', 'rnn', 'transformer']),
                'learning_rate': random.uniform(0.001, 0.1),
                'batch_size': random.choice([16, 32, 64, 128]),
                'epochs': random.randint(10, 100),
                'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
                'worker_id': self.worker_id,
                'stress_test': True
            }
            
            exp_id = self.tracker.create_experiment(
                name=exp_name,
                config=config,
                tags=[f'worker_{self.worker_id}', 'stress_test']
            )
            experiments_created = 1
            
            # Log metrics for multiple iterations
            for iteration in range(self.config.iterations_per_experiment):
                metrics = {
                    'loss': random.uniform(0.1, 2.0) * (1 - iteration / self.config.iterations_per_experiment),
                    'accuracy': random.uniform(0.5, 0.99) * (iteration / self.config.iterations_per_experiment + 0.1),
                    'learning_rate': config['learning_rate'] * (0.99 ** iteration),
                    'batch_time': random.uniform(0.1, 1.0),
                    'memory_usage': random.uniform(100, 1000),
                    'gpu_utilization': random.uniform(50, 95)
                }
                
                for metric_name, metric_value in metrics.items():
                    self.tracker.log_metric(exp_id, metric_name, metric_value, iteration=iteration)
                metrics_logged += len(metrics)
                
                # Simulate some processing time
                time.sleep(random.uniform(0.001, 0.01))
            
            # Complete the experiment
            final_status = random.choice(['completed', 'completed', 'completed', 'failed'])  # 75% success rate
            self.tracker.update_experiment_status(exp_id, final_status)
            
            # Record response time
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            self.response_times.append(response_time)
            self.operations_count += 1
            
        except Exception as e:
            self.errors.append(f"Worker {self.worker_id} experiment {exp_index} failed: {e}")
        
        return experiments_created, metrics_logged
    
    def run_read_operations(self, duration_seconds: float):
        """Run continuous read operations for testing concurrent access."""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                # Random read operations
                operation = random.choice(['list_experiments', 'get_experiment', 'get_metrics'])
                
                if operation == 'list_experiments':
                    experiments = self.tracker.query_experiments(limit=50)
                    
                elif operation == 'get_experiment':
                    # Try to get a random experiment ID from 1-100
                    exp_id = random.randint(1, 100)
                    try:
                        exp_data = self.tracker.get_experiment(exp_id)
                    except:
                        pass  # Ignore if experiment doesn't exist
                    
                elif operation == 'get_metrics':
                    # Try to get metrics for a random experiment
                    exp_id = random.randint(1, 100)
                    try:
                        metrics = self.tracker.get_metric_history(exp_id, 'loss')
                    except:
                        pass  # Ignore if experiment doesn't exist
                
                self.operations_count += 1
                time.sleep(random.uniform(0.01, 0.1))  # Simulate read delay
                
            except Exception as e:
                self.errors.append(f"Worker {self.worker_id} read operation failed: {e}")
            
            time.sleep(random.uniform(0.001, 0.01))


class StressTestRunner:
    """Main stress test runner."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.monitor = StressTestMonitor(config)
        self.results = None
        self.memory_samples = []
        
    def setup_test_environment(self):
        """Set up the test environment."""
        # Remove existing test database
        if Path(self.config.database_path).exists():
            Path(self.config.database_path).unlink()
        
        # Create fresh database
        tracker = ExperimentTracker(db_path=self.config.database_path)
        
        print(f"Test environment set up with database: {self.config.database_path}")
    
    def start_memory_monitoring(self):
        """Start monitoring memory usage."""
        self.monitoring = True
        self.memory_samples = []
        
        def monitor():
            process = psutil.Process()
            while self.monitoring:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                time.sleep(1.0)
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def stop_memory_monitoring(self):
        """Stop monitoring memory usage."""
        self.monitoring = False
    
    def run_stress_test(self) -> StressTestResults:
        """Run the complete stress test."""
        print("Starting comprehensive stress test...")
        print(f"Configuration: {self.config.num_experiments} experiments, "
              f"{self.config.num_concurrent_threads} threads, "
              f"{self.config.metrics_per_experiment} metrics per experiment")
        
        start_time = datetime.now()
        self.setup_test_environment()
        self.start_memory_monitoring()
        
        total_experiments = 0
        total_metrics = 0
        all_errors = []
        
        try:
            # Phase 1: Concurrent experiment creation
            print("\nPhase 1: Creating experiments with concurrent threads...")
            exp_results = self._run_experiment_creation()
            total_experiments += exp_results[0]
            total_metrics += exp_results[1]
            all_errors.extend(exp_results[2])
            
            # Phase 2: Concurrent read operations
            print("\nPhase 2: Running concurrent read operations...")
            read_results = self._run_concurrent_reads()
            all_errors.extend(read_results)
            
            # Phase 3: Mixed workload
            print("\nPhase 3: Running mixed read/write workload...")
            mixed_results = self._run_mixed_workload()
            total_experiments += mixed_results[0]
            total_metrics += mixed_results[1]
            all_errors.extend(mixed_results[2])
            
        finally:
            # Stop monitoring
            self.stop_memory_monitoring()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate performance metrics
        performance_summary = self.monitor.get_performance_summary()
        total_operations = total_experiments + total_metrics
        operations_per_second = total_operations / duration if duration > 0 else 0
        success_rate = 1.0 - (len(all_errors) / max(total_operations, 1))
        
        # Memory statistics
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        
        # Create results
        self.results = StressTestResults(
            config=self.config,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=duration,
            experiments_created=total_experiments,
            metrics_logged=total_metrics,
            total_operations=total_operations,
            operations_per_second=operations_per_second,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            peak_cpu_percent=performance_summary.get('peak_cpu_percent', 0),
            avg_cpu_percent=performance_summary.get('avg_cpu_percent', 0),
            database_final_size_mb=performance_summary.get('final_db_size_mb', 0),
            errors=all_errors,
            performance_timeline=self.monitor.performance_data,
            success_rate=success_rate
        )
        
        return self.results
    
    def _run_experiment_creation(self) -> Tuple[int, int, List[str]]:
        """Run concurrent experiment creation."""
        experiments_per_thread = self.config.num_experiments // self.config.num_concurrent_threads
        total_experiments = 0
        total_metrics = 0
        all_errors = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_concurrent_threads) as executor:
            # Submit tasks
            futures = []
            for thread_id in range(self.config.num_concurrent_threads):
                worker = ExperimentWorker(thread_id, self.config)
                worker.setup()
                
                future = executor.submit(self._worker_create_experiments, worker, experiments_per_thread)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    exp_count, metric_count, errors = future.result()
                    total_experiments += exp_count
                    total_metrics += metric_count
                    all_errors.extend(errors)
                except Exception as e:
                    all_errors.append(f"Thread execution failed: {e}")
        
        return total_experiments, total_metrics, all_errors
    
    def _worker_create_experiments(self, worker: ExperimentWorker, num_experiments: int) -> Tuple[int, int, List[str]]:
        """Worker function for creating experiments."""
        total_experiments = 0
        total_metrics = 0
        
        for i in range(num_experiments):
            exp_count, metric_count = worker.create_experiment_with_metrics(i)
            total_experiments += exp_count
            total_metrics += metric_count
            
            # Progress reporting
            if (i + 1) % 50 == 0:
                print(f"Worker {worker.worker_id}: Created {i + 1}/{num_experiments} experiments")
        
        return total_experiments, total_metrics, worker.errors
    
    def _run_concurrent_reads(self) -> List[str]:
        """Run concurrent read operations."""
        all_errors = []
        duration = 30  # 30 seconds of concurrent reads
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_readers) as executor:
            futures = []
            for reader_id in range(self.config.concurrent_readers):
                worker = ExperimentWorker(f"reader_{reader_id}", self.config)
                worker.setup()
                
                future = executor.submit(worker.run_read_operations, duration)
                futures.append((future, worker))
            
            # Wait for completion
            for future, worker in futures:
                try:
                    future.result()
                    all_errors.extend(worker.errors)
                except Exception as e:
                    all_errors.append(f"Reader thread failed: {e}")
        
        return all_errors
    
    def _run_mixed_workload(self) -> Tuple[int, int, List[str]]:
        """Run mixed read/write workload."""
        # Create additional experiments while running reads
        additional_experiments = 100
        total_experiments = 0
        total_metrics = 0
        all_errors = []
        
        # Start background readers
        reader_futures = []
        with ThreadPoolExecutor(max_workers=self.config.concurrent_readers + 2) as executor:
            # Start readers
            for reader_id in range(self.config.concurrent_readers):
                worker = ExperimentWorker(f"mixed_reader_{reader_id}", self.config)
                worker.setup()
                future = executor.submit(worker.run_read_operations, 60)  # 1 minute
                reader_futures.append((future, worker))
            
            # Start writers
            for writer_id in range(2):  # 2 concurrent writers
                worker = ExperimentWorker(f"mixed_writer_{writer_id}", self.config)
                worker.setup()
                future = executor.submit(self._worker_create_experiments, worker, additional_experiments // 2)
                reader_futures.append((future, worker))
            
            # Collect results
            for future, worker in reader_futures:
                try:
                    result = future.result()
                    if isinstance(result, tuple):  # Writer result
                        exp_count, metric_count, errors = result
                        total_experiments += exp_count
                        total_metrics += metric_count
                        all_errors.extend(errors)
                    else:  # Reader result
                        all_errors.extend(worker.errors)
                except Exception as e:
                    all_errors.append(f"Mixed workload thread failed: {e}")
        
        return total_experiments, total_metrics, all_errors
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.results:
            return "No test results available"
        
        report = []
        report.append("=" * 80)
        report.append("YINSHML EXPERIMENT TRACKING - STRESS TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Test configuration
        report.append("TEST CONFIGURATION:")
        report.append(f"  Experiments Created: {self.results.experiments_created:,}")
        report.append(f"  Metrics Logged: {self.results.metrics_logged:,}")
        report.append(f"  Concurrent Threads: {self.config.num_concurrent_threads}")
        report.append(f"  Test Duration: {self.results.total_duration_seconds:.2f} seconds")
        report.append("")
        
        # Performance results
        report.append("PERFORMANCE RESULTS:")
        report.append(f"  Operations per Second: {self.results.operations_per_second:.2f}")
        report.append(f"  Success Rate: {self.results.success_rate:.2%}")
        report.append(f"  Total Operations: {self.results.total_operations:,}")
        report.append("")
        
        # Resource usage
        report.append("RESOURCE USAGE:")
        report.append(f"  Peak Memory: {self.results.peak_memory_mb:.2f} MB")
        report.append(f"  Average Memory: {self.results.avg_memory_mb:.2f} MB")
        report.append(f"  Peak CPU: {self.results.peak_cpu_percent:.1f}%")
        report.append(f"  Average CPU: {self.results.avg_cpu_percent:.1f}%")
        report.append(f"  Final Database Size: {self.results.database_final_size_mb:.2f} MB")
        report.append("")
        
        # Error analysis
        if self.results.errors:
            report.append("ERRORS ENCOUNTERED:")
            error_counts = {}
            for error in self.results.errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_counts.items()):
                report.append(f"  {error_type}: {count} occurrences")
            report.append("")
        
        # Performance assessment
        report.append("PERFORMANCE ASSESSMENT:")
        
        # Throughput assessment
        if self.results.operations_per_second > 1000:
            report.append("  ✅ EXCELLENT: Operations per second > 1000")
        elif self.results.operations_per_second > 500:
            report.append("  ✅ GOOD: Operations per second > 500")
        elif self.results.operations_per_second > 100:
            report.append("  ⚠️  ACCEPTABLE: Operations per second > 100")
        else:
            report.append("  ❌ POOR: Operations per second < 100")
        
        # Memory assessment
        if self.results.peak_memory_mb < 500:
            report.append("  ✅ EXCELLENT: Peak memory usage < 500 MB")
        elif self.results.peak_memory_mb < 1000:
            report.append("  ✅ GOOD: Peak memory usage < 1 GB")
        elif self.results.peak_memory_mb < 2000:
            report.append("  ⚠️  ACCEPTABLE: Peak memory usage < 2 GB")
        else:
            report.append("  ❌ HIGH: Peak memory usage > 2 GB")
        
        # Success rate assessment
        if self.results.success_rate > 0.99:
            report.append("  ✅ EXCELLENT: Success rate > 99%")
        elif self.results.success_rate > 0.95:
            report.append("  ✅ GOOD: Success rate > 95%")
        elif self.results.success_rate > 0.90:
            report.append("  ⚠️  ACCEPTABLE: Success rate > 90%")
        else:
            report.append("  ❌ POOR: Success rate < 90%")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_results(self, output_path: str = "stress_test_results.json"):
        """Save detailed results to JSON file."""
        if not self.results:
            return
        
        # Convert results to dictionary for JSON serialization
        results_dict = asdict(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {output_path}")
    
    def cleanup(self):
        """Clean up test artifacts."""
        if self.config.cleanup_after_test:
            if Path(self.config.database_path).exists():
                Path(self.config.database_path).unlink()
                print(f"Cleaned up test database: {self.config.database_path}")


def main():
    """Main function to run stress tests."""
    # Configuration for stress testing
    config = StressTestConfig(
        num_experiments=1200,  # Exceed the 1000+ requirement
        num_concurrent_threads=12,
        metrics_per_experiment=150,
        iterations_per_experiment=75,
        concurrent_readers=8,
        test_duration_minutes=15,
        memory_monitoring_interval=0.5,
        database_path="stress_test_experiments.db",
        cleanup_after_test=True
    )
    
    print("YinshML Experiment Tracking - Comprehensive Stress Test")
    print("=" * 60)
    
    # Run the stress test
    runner = StressTestRunner(config)
    
    try:
        results = runner.run_stress_test()
        
        # Generate and display report
        report = runner.generate_report()
        print(report)
        
        # Save detailed results
        runner.save_detailed_results("stress_test_results.json")
        
        # Performance validation
        print("\nPERFORMANCE VALIDATION:")
        if results.operations_per_second > 500 and results.success_rate > 0.95:
            print("✅ STRESS TEST PASSED: System performs well under load")
            return 0
        else:
            print("❌ STRESS TEST FAILED: Performance issues detected")
            return 1
            
    except Exception as e:
        print(f"❌ STRESS TEST ERROR: {e}")
        return 1
    
    finally:
        runner.cleanup()


if __name__ == "__main__":
    exit(main()) 