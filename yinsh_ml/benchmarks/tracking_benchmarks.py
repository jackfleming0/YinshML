"""
Experiment tracking system performance benchmarks.

This module contains benchmark cases that measure the performance impact
of the experiment tracking system on training pipelines and individual
component operations.
"""

import logging
import time
import tempfile
import shutil
import torch
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

from ..tracking.experiment_tracker import ExperimentTracker
from ..tracking.database import create_database
from ..tracking import utils as tracking_utils
from ..tracking.config_serializer import ConfigurationSerializer
from .benchmark_framework import BenchmarkCase

logger = logging.getLogger(__name__)


class ExperimentTrackingBenchmark(BenchmarkCase):
    """Benchmark for end-to-end experiment tracking overhead."""
    
    def __init__(self,
                 num_experiments: int = 10,
                 metrics_per_experiment: int = 100,
                 config_complexity: str = "medium",
                 async_logging: bool = False):
        """
        Initialize experiment tracking benchmark.
        
        Args:
            num_experiments: Number of experiments to create per iteration
            metrics_per_experiment: Number of metrics to log per experiment
            config_complexity: Configuration complexity ("simple", "medium", "complex")
            async_logging: Whether to use async logging
        """
        async_suffix = "async" if async_logging else "sync"
        super().__init__(
            name=f"ExperimentTracking_{num_experiments}exp_{metrics_per_experiment}metrics_{config_complexity}_{async_suffix}",
            description=f"End-to-end tracking: {num_experiments} experiments, "
                       f"{metrics_per_experiment} metrics each, {config_complexity} config, "
                       f"async: {async_logging}"
        )
        
        self.num_experiments = num_experiments
        self.metrics_per_experiment = metrics_per_experiment
        self.config_complexity = config_complexity
        self.async_logging = async_logging
        
        self.temp_dir: Optional[Path] = None
        self.tracker: Optional[ExperimentTracker] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create temporary directory for database
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tracking_benchmark_"))
        db_path = self.temp_dir / "benchmark.db"
        
        # Initialize tracker with benchmark configuration
        tracker_config = {
            'async_logging': self.async_logging,
            'capture_git': False,  # Disable for consistent benchmarking
            'capture_environment': False,
            'capture_system': False,
            'tensorboard_enabled': False,  # Disable TensorBoard for pure tracking measurement
        }
        
        # Reset singleton to ensure clean state
        ExperimentTracker._instance = None
        self.tracker = ExperimentTracker(str(db_path), tracker_config)
        tracking_utils.set_database_path(str(db_path))
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.tracker:
            # Clean up tracker resources
            self.tracker = None
            ExperimentTracker._instance = None
        
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def _generate_config(self) -> Dict[str, Any]:
        """Generate configuration based on complexity level."""
        if self.config_complexity == "simple":
            return {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }
        elif self.config_complexity == "medium":
            return {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'model_config': {
                    'hidden_size': 256,
                    'num_layers': 4,
                    'dropout': 0.1
                },
                'data_config': {
                    'augmentation': True,
                    'normalization': 'batch',
                    'shuffle': True
                }
            }
        else:  # complex
            return {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'optimizer': {
                    'type': 'adam',
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'eps': 1e-8,
                    'weight_decay': 1e-4
                },
                'scheduler': {
                    'type': 'cosine',
                    'T_max': 100,
                    'eta_min': 1e-6,
                    'warmup_epochs': 10
                },
                'model_config': {
                    'architecture': 'transformer',
                    'hidden_size': 512,
                    'num_layers': 8,
                    'num_heads': 8,
                    'dropout': 0.1,
                    'layer_norm_eps': 1e-5,
                    'activation': 'gelu'
                },
                'data_config': {
                    'dataset': 'custom',
                    'augmentation': {
                        'rotation': True,
                        'flip': True,
                        'noise': 0.1
                    },
                    'normalization': {
                        'type': 'batch',
                        'momentum': 0.1,
                        'eps': 1e-5
                    },
                    'preprocessing': {
                        'resize': [224, 224],
                        'crop': 'center',
                        'normalize': True
                    }
                },
                'training_config': {
                    'gradient_clipping': 1.0,
                    'mixed_precision': True,
                    'accumulation_steps': 4,
                    'validation_frequency': 10
                }
            }
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.tracker:
            raise RuntimeError("Tracker not initialized")
        
        start_time = time.perf_counter_ns()
        
        # Track operations
        experiments_created = 0
        total_metrics_logged = 0
        total_status_updates = 0
        
        try:
            for i in range(self.num_experiments):
                # Create experiment
                config = self._generate_config()
                exp_id = self.tracker.create_experiment(
                    name=f"benchmark_exp_{i}",
                    description=f"Benchmark experiment {i}",
                    config=config,
                    tags=['benchmark', f'iteration_{i}']
                )
                experiments_created += 1
                
                # Log metrics
                for j in range(self.metrics_per_experiment):
                    # Simulate realistic metric patterns
                    epoch = j // 10
                    step = j
                    
                    # Training metrics
                    self.tracker.log_metric(exp_id, "train_loss", 
                                          np.random.exponential(0.5) + 0.1, step)
                    self.tracker.log_metric(exp_id, "train_accuracy", 
                                          min(0.95, 0.5 + np.random.exponential(0.3)), step)
                    
                    # Validation metrics (less frequent)
                    if j % 10 == 0:
                        self.tracker.log_metric(exp_id, "val_loss", 
                                              np.random.exponential(0.6) + 0.15, step)
                        self.tracker.log_metric(exp_id, "val_accuracy", 
                                              min(0.92, 0.45 + np.random.exponential(0.25)), step)
                    
                    total_metrics_logged += 2 if j % 10 != 0 else 4
                
                # Update experiment status
                if i % 3 == 0:
                    self.tracker.update_experiment_status(exp_id, "completed")
                    total_status_updates += 1
                elif i % 3 == 1:
                    self.tracker.update_experiment_status(exp_id, "running")
                    total_status_updates += 1
                
        except Exception as e:
            logger.warning(f"Benchmark iteration failed: {e}")
        
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        
        # Get database statistics
        db_stats = tracking_utils.get_database_stats()
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'experiments_created': experiments_created,
            'total_metrics_logged': total_metrics_logged,
            'total_status_updates': total_status_updates,
            'experiments_per_sec': experiments_created / duration_seconds if duration_seconds > 0 else 0,
            'metrics_per_sec': total_metrics_logged / duration_seconds if duration_seconds > 0 else 0,
            'avg_metrics_per_experiment': total_metrics_logged / experiments_created if experiments_created > 0 else 0,
            'config_complexity': self.config_complexity,
            'async_logging': self.async_logging,
            'db_experiment_count': db_stats.get('experiment_count', 0),
            'db_metric_count': db_stats.get('metric_count', 0),
        }
        
        return metrics


class DatabaseOperationsBenchmark(BenchmarkCase):
    """Benchmark for database operation performance."""
    
    def __init__(self,
                 num_queries: int = 1000,
                 query_type: str = "mixed",
                 data_size: str = "medium"):
        """
        Initialize database operations benchmark.
        
        Args:
            num_queries: Number of database queries per iteration
            query_type: Type of queries ("read", "write", "mixed")
            data_size: Size of test data ("small", "medium", "large")
        """
        super().__init__(
            name=f"DatabaseOps_{num_queries}queries_{query_type}_{data_size}",
            description=f"Database operations: {num_queries} {query_type} queries, "
                       f"{data_size} data size"
        )
        
        self.num_queries = num_queries
        self.query_type = query_type
        self.data_size = data_size
        
        self.temp_dir: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self.experiment_ids: List[str] = []
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create temporary directory and database
        self.temp_dir = Path(tempfile.mkdtemp(prefix="db_benchmark_"))
        self.db_path = self.temp_dir / "benchmark.db"
        
        # Create database and populate with test data
        database = create_database(self.db_path)
        tracking_utils.set_database_path(str(self.db_path))
        
        # Create test experiments based on data size
        num_experiments = {"small": 10, "medium": 100, "large": 1000}[self.data_size]
        
        # Reset tracker singleton
        ExperimentTracker._instance = None
        tracker = ExperimentTracker(str(self.db_path), {'async_logging': False})
        
        self.experiment_ids = []
        for i in range(num_experiments):
            exp_id = tracker.create_experiment(
                name=f"test_exp_{i}",
                description=f"Test experiment {i}",
                config={'param': i, 'value': i * 0.1}
            )
            self.experiment_ids.append(exp_id)
            
            # Add some metrics
            for j in range(10):
                tracker.log_metric(exp_id, "metric_a", i + j * 0.1, j)
                tracker.log_metric(exp_id, "metric_b", (i + j) * 0.05, j)
    
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        
        ExperimentTracker._instance = None
        
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        start_time = time.perf_counter_ns()
        
        read_operations = 0
        write_operations = 0
        
        try:
            for i in range(self.num_queries):
                if self.query_type == "read" or (self.query_type == "mixed" and i % 3 != 0):
                    # Read operations
                    if i % 4 == 0:
                        # Get experiment by ID
                        exp_id = np.random.choice(self.experiment_ids)
                        tracking_utils.get_experiment_by_id(exp_id)
                    elif i % 4 == 1:
                        # Get experiment metrics
                        exp_id = np.random.choice(self.experiment_ids)
                        tracking_utils.get_experiment_metrics(exp_id)
                    elif i % 4 == 2:
                        # List experiments
                        tracking_utils.list_experiments(limit=10)
                    else:
                        # Get database stats
                        tracking_utils.get_database_stats()
                    
                    read_operations += 1
                
                if self.query_type == "write" or (self.query_type == "mixed" and i % 3 == 0):
                    # Write operations (create new experiment)
                    ExperimentTracker._instance = None
                    tracker = ExperimentTracker(str(self.db_path), {'async_logging': False})
                    
                    exp_id = tracker.create_experiment(
                        name=f"bench_exp_{i}",
                        config={'iteration': i}
                    )
                    tracker.log_metric(exp_id, "bench_metric", i * 0.1, 0)
                    
                    write_operations += 1
                    
        except Exception as e:
            logger.warning(f"Database benchmark iteration failed: {e}")
        
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        total_operations = read_operations + write_operations
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'total_operations': total_operations,
            'read_operations': read_operations,
            'write_operations': write_operations,
            'operations_per_sec': total_operations / duration_seconds if duration_seconds > 0 else 0,
            'avg_operation_time_ms': (total_duration_ns / 1_000_000) / total_operations if total_operations > 0 else 0,
            'query_type': self.query_type,
            'data_size': self.data_size,
        }
        
        return metrics


class MetricLoggingBenchmark(BenchmarkCase):
    """Benchmark for metric logging throughput."""
    
    def __init__(self,
                 num_metrics: int = 10000,
                 batch_size: int = 1,
                 async_logging: bool = False):
        """
        Initialize metric logging benchmark.
        
        Args:
            num_metrics: Number of metrics to log per iteration
            batch_size: Number of metrics to log in each batch
            async_logging: Whether to use async logging
        """
        async_suffix = "async" if async_logging else "sync"
        super().__init__(
            name=f"MetricLogging_{num_metrics}metrics_batch{batch_size}_{async_suffix}",
            description=f"Metric logging: {num_metrics} metrics, batch size {batch_size}, "
                       f"async: {async_logging}"
        )
        
        self.num_metrics = num_metrics
        self.batch_size = batch_size
        self.async_logging = async_logging
        
        self.temp_dir: Optional[Path] = None
        self.tracker: Optional[ExperimentTracker] = None
        self.experiment_id: Optional[str] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        # Create temporary directory for database
        self.temp_dir = Path(tempfile.mkdtemp(prefix="metric_benchmark_"))
        db_path = self.temp_dir / "benchmark.db"
        
        # Initialize tracker
        tracker_config = {
            'async_logging': self.async_logging,
            'capture_git': False,
            'capture_environment': False,
            'capture_system': False,
            'tensorboard_enabled': False,
        }
        
        ExperimentTracker._instance = None
        self.tracker = ExperimentTracker(str(db_path), tracker_config)
        tracking_utils.set_database_path(str(db_path))
        
        # Create a single experiment for metric logging
        self.experiment_id = self.tracker.create_experiment(
            name="metric_benchmark",
            description="Benchmark experiment for metric logging",
            config={'benchmark': True}
        )
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        if self.tracker:
            self.tracker = None
            ExperimentTracker._instance = None
        
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.tracker or not self.experiment_id:
            raise RuntimeError("Tracker or experiment not initialized")
        
        start_time = time.perf_counter_ns()
        
        metrics_logged = 0
        batches_processed = 0
        
        try:
            for i in range(0, self.num_metrics, self.batch_size):
                batch_start = time.perf_counter_ns()
                
                # Log batch of metrics
                for j in range(min(self.batch_size, self.num_metrics - i)):
                    metric_idx = i + j
                    
                    # Simulate different metric types
                    if metric_idx % 4 == 0:
                        self.tracker.log_metric(self.experiment_id, "loss", 
                                              np.random.exponential(0.5), metric_idx)
                    elif metric_idx % 4 == 1:
                        self.tracker.log_metric(self.experiment_id, "accuracy", 
                                              np.random.uniform(0.5, 0.95), metric_idx)
                    elif metric_idx % 4 == 2:
                        self.tracker.log_metric(self.experiment_id, "learning_rate", 
                                              0.001 * (0.99 ** (metric_idx // 100)), metric_idx)
                    else:
                        self.tracker.log_metric(self.experiment_id, "batch_time", 
                                              np.random.normal(0.1, 0.02), metric_idx)
                    
                    metrics_logged += 1
                
                batches_processed += 1
                
        except Exception as e:
            logger.warning(f"Metric logging benchmark failed: {e}")
        
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'metrics_logged': metrics_logged,
            'batches_processed': batches_processed,
            'metrics_per_sec': metrics_logged / duration_seconds if duration_seconds > 0 else 0,
            'avg_metric_time_us': (total_duration_ns / 1000) / metrics_logged if metrics_logged > 0 else 0,
            'batch_size': self.batch_size,
            'async_logging': self.async_logging,
        }
        
        return metrics


class ConfigSerializationBenchmark(BenchmarkCase):
    """Benchmark for configuration serialization performance."""
    
    def __init__(self,
                 num_configs: int = 1000,
                 config_complexity: str = "medium"):
        """
        Initialize configuration serialization benchmark.
        
        Args:
            num_configs: Number of configurations to serialize per iteration
            config_complexity: Configuration complexity ("simple", "medium", "complex")
        """
        super().__init__(
            name=f"ConfigSerialization_{num_configs}configs_{config_complexity}",
            description=f"Config serialization: {num_configs} configs, {config_complexity} complexity"
        )
        
        self.num_configs = num_configs
        self.config_complexity = config_complexity
        self.serializer: Optional[ConfigurationSerializer] = None
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        self.serializer = ConfigurationSerializer()
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        self.serializer = None
    
    def _generate_config(self, index: int) -> Dict[str, Any]:
        """Generate configuration based on complexity level."""
        if self.config_complexity == "simple":
            return {
                'learning_rate': 0.001 + index * 0.0001,
                'batch_size': 32 + index % 64,
                'epochs': 100 + index % 50
            }
        elif self.config_complexity == "medium":
            return {
                'learning_rate': 0.001 + index * 0.0001,
                'batch_size': 32 + index % 64,
                'epochs': 100 + index % 50,
                'optimizer': ['adam', 'sgd', 'rmsprop'][index % 3],
                'model_config': {
                    'hidden_size': 256 + index % 512,
                    'num_layers': 4 + index % 8,
                    'dropout': 0.1 + (index % 10) * 0.01
                },
                'data_config': {
                    'augmentation': index % 2 == 0,
                    'normalization': ['batch', 'layer', 'instance'][index % 3],
                    'shuffle': True
                }
            }
        else:  # complex
            return {
                'learning_rate': 0.001 + index * 0.0001,
                'batch_size': 32 + index % 64,
                'epochs': 100 + index % 50,
                'optimizer': {
                    'type': ['adam', 'sgd', 'rmsprop'][index % 3],
                    'beta1': 0.9 + (index % 10) * 0.001,
                    'beta2': 0.999 - (index % 10) * 0.0001,
                    'weight_decay': 1e-4 + (index % 10) * 1e-5
                },
                'model_config': {
                    'architecture': ['transformer', 'cnn', 'rnn'][index % 3],
                    'hidden_size': 256 + index % 512,
                    'num_layers': 4 + index % 8,
                    'dropout': 0.1 + (index % 10) * 0.01,
                    'activation': ['relu', 'gelu', 'swish'][index % 3],
                    'layer_configs': [
                        {'type': 'linear', 'size': 512 + i * 64}
                        for i in range(index % 5 + 1)
                    ]
                },
                'data_config': {
                    'dataset': f'dataset_{index % 10}',
                    'preprocessing': {
                        'resize': [224 + index % 32, 224 + index % 32],
                        'normalize': True,
                        'augmentation': {
                            'rotation': index % 2 == 0,
                            'flip': index % 3 == 0,
                            'noise': (index % 10) * 0.01
                        }
                    }
                },
                'training_config': {
                    'gradient_clipping': 1.0 + (index % 10) * 0.1,
                    'mixed_precision': index % 2 == 0,
                    'accumulation_steps': 1 + index % 8,
                    'checkpointing': {
                        'frequency': 10 + index % 20,
                        'keep_best': True,
                        'metric': 'val_accuracy'
                    }
                }
            }
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        if not self.serializer:
            raise RuntimeError("Serializer not initialized")
        
        start_time = time.perf_counter_ns()
        
        configs_serialized = 0
        configs_deserialized = 0
        total_serialized_size = 0
        
        serialized_configs = []
        
        try:
            # Serialization phase
            serialize_start = time.perf_counter_ns()
            
            for i in range(self.num_configs):
                config = self._generate_config(i)
                # Use the actual API method
                comprehensive_config = self.serializer.capture_comprehensive_configuration(
                    user_config=config,
                    include_environment=False,
                    include_system_config=False
                )
                serialized = json.dumps(comprehensive_config)
                serialized_configs.append(comprehensive_config)
                total_serialized_size += len(serialized)
                configs_serialized += 1
            
            serialize_end = time.perf_counter_ns()
            
            # Deserialization phase
            deserialize_start = time.perf_counter_ns()
            
            for comprehensive_config in serialized_configs:
                # Simulate deserialization by reconstructing config
                reconstructed = self.serializer.reconstruct_configuration(comprehensive_config)
                configs_deserialized += 1
            
            deserialize_end = time.perf_counter_ns()
            
        except Exception as e:
            logger.warning(f"Config serialization benchmark failed: {e}")
            serialize_end = deserialize_start = deserialize_end = time.perf_counter_ns()
        
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        serialize_duration_ns = serialize_end - serialize_start
        deserialize_duration_ns = deserialize_end - deserialize_start
        
        duration_seconds = total_duration_ns / 1_000_000_000
        serialize_seconds = serialize_duration_ns / 1_000_000_000
        deserialize_seconds = deserialize_duration_ns / 1_000_000_000
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'serialize_duration_ns': serialize_duration_ns,
            'deserialize_duration_ns': deserialize_duration_ns,
            'configs_serialized': configs_serialized,
            'configs_deserialized': configs_deserialized,
            'total_serialized_size_bytes': total_serialized_size,
            'avg_serialized_size_bytes': total_serialized_size / configs_serialized if configs_serialized > 0 else 0,
            'serialize_configs_per_sec': configs_serialized / serialize_seconds if serialize_seconds > 0 else 0,
            'deserialize_configs_per_sec': configs_deserialized / deserialize_seconds if deserialize_seconds > 0 else 0,
            'total_configs_per_sec': (configs_serialized + configs_deserialized) / duration_seconds if duration_seconds > 0 else 0,
            'config_complexity': self.config_complexity,
        }
        
        return metrics


@contextmanager
def training_simulation_context(use_tracking: bool = True, 
                               tracking_config: Optional[Dict[str, Any]] = None):
    """
    Context manager for training simulation with optional tracking.
    
    Args:
        use_tracking: Whether to enable experiment tracking
        tracking_config: Configuration for tracking system
    
    Yields:
        Tuple of (tracker, experiment_id) if tracking enabled, else (None, None)
    """
    temp_dir = None
    tracker = None
    experiment_id = None
    
    try:
        if use_tracking:
            # Set up tracking
            temp_dir = Path(tempfile.mkdtemp(prefix="training_sim_"))
            db_path = temp_dir / "training.db"
            
            config = tracking_config or {
                'async_logging': False,
                'capture_git': False,
                'capture_environment': False,
                'capture_system': False,
                'tensorboard_enabled': False,
            }
            
            ExperimentTracker._instance = None
            tracker = ExperimentTracker(str(db_path), config)
            tracking_utils.set_database_path(str(db_path))
            
            experiment_id = tracker.create_experiment(
                name="training_simulation",
                description="Simulated training for performance measurement",
                config={'simulation': True}
            )
        
        yield tracker, experiment_id
        
    finally:
        if tracker:
            ExperimentTracker._instance = None
        
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


class TrainingPipelineOverheadBenchmark(BenchmarkCase):
    """Benchmark for measuring tracking overhead on training pipeline."""
    
    def __init__(self,
                 num_epochs: int = 10,
                 steps_per_epoch: int = 100,
                 use_tracking: bool = True,
                 tracking_frequency: int = 1):
        """
        Initialize training pipeline overhead benchmark.
        
        Args:
            num_epochs: Number of training epochs to simulate
            steps_per_epoch: Number of steps per epoch
            use_tracking: Whether to enable tracking
            tracking_frequency: How often to log metrics (every N steps)
        """
        tracking_suffix = "with_tracking" if use_tracking else "no_tracking"
        super().__init__(
            name=f"TrainingOverhead_{num_epochs}epochs_{steps_per_epoch}steps_{tracking_suffix}_freq{tracking_frequency}",
            description=f"Training pipeline: {num_epochs} epochs, {steps_per_epoch} steps/epoch, "
                       f"tracking: {use_tracking}, frequency: {tracking_frequency}"
        )
        
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.use_tracking = use_tracking
        self.tracking_frequency = tracking_frequency
        
    def setup(self) -> None:
        """Set up the benchmark environment."""
        pass  # Setup handled in context manager
        
    def teardown(self) -> None:
        """Clean up the benchmark environment."""
        pass  # Cleanup handled in context manager
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single benchmark iteration."""
        start_time = time.perf_counter_ns()
        
        total_steps = 0
        metrics_logged = 0
        
        with training_simulation_context(use_tracking=self.use_tracking) as (tracker, experiment_id):
            try:
                for epoch in range(self.num_epochs):
                    epoch_start = time.perf_counter_ns()
                    
                    for step in range(self.steps_per_epoch):
                        step_start = time.perf_counter_ns()
                        
                        # Simulate training step computation
                        # Create some tensors to simulate model forward/backward
                        batch_size = 32
                        input_tensor = torch.randn(batch_size, 128)
                        target_tensor = torch.randn(batch_size, 10)
                        
                        # Simulate forward pass
                        weight = torch.randn(10, 128, requires_grad=True)
                        output = torch.nn.functional.linear(input_tensor, weight)
                        loss = torch.nn.functional.mse_loss(output, target_tensor)
                        
                        # Simulate backward pass
                        loss.backward()
                        
                        # Simulate optimizer step
                        time.sleep(0.001)  # Small delay to simulate computation
                        
                        total_steps += 1
                        
                        # Log metrics if tracking enabled and frequency matches
                        if (tracker and experiment_id and 
                            step % self.tracking_frequency == 0):
                            
                            global_step = epoch * self.steps_per_epoch + step
                            
                            tracker.log_metric(experiment_id, "train_loss", 
                                             float(loss.item()), global_step)
                            tracker.log_metric(experiment_id, "learning_rate", 
                                             0.001 * (0.99 ** epoch), global_step)
                            
                            if step % 10 == 0:  # Less frequent metrics
                                tracker.log_metric(experiment_id, "epoch", 
                                                 float(epoch), global_step)
                                tracker.log_metric(experiment_id, "step_time", 
                                                 (time.perf_counter_ns() - step_start) / 1_000_000, 
                                                 global_step)
                            
                            metrics_logged += 2 if step % 10 != 0 else 4
                    
                    # End of epoch logging
                    if tracker and experiment_id:
                        epoch_duration = (time.perf_counter_ns() - epoch_start) / 1_000_000
                        tracker.log_metric(experiment_id, "epoch_duration_ms", 
                                         epoch_duration, epoch * self.steps_per_epoch)
                        metrics_logged += 1
                        
            except Exception as e:
                logger.warning(f"Training simulation failed: {e}")
        
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        total_duration_ns = end_time - start_time
        duration_seconds = total_duration_ns / 1_000_000_000
        
        metrics = {
            'total_duration_ns': total_duration_ns,
            'total_steps': total_steps,
            'metrics_logged': metrics_logged,
            'steps_per_sec': total_steps / duration_seconds if duration_seconds > 0 else 0,
            'avg_step_time_ms': (total_duration_ns / 1_000_000) / total_steps if total_steps > 0 else 0,
            'tracking_overhead_per_step_us': (metrics_logged * 100) if self.use_tracking else 0,  # Rough estimate
            'use_tracking': self.use_tracking,
            'tracking_frequency': self.tracking_frequency,
            'num_epochs': self.num_epochs,
            'steps_per_epoch': self.steps_per_epoch,
        }
        
        return metrics 