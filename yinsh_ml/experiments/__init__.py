"""
Experiment tracking and management for YinshML hyperparameter tuning.

This module provides:
- Typed experiment configuration (ExperimentConfig)
- SQLite-backed experiment database (ExperimentDB)
- Unified metrics aggregation (MetricsAggregator)
- Experiment runner with observability (run_experiment)
"""

from .experiment_config import (
    ExperimentConfig,
    TrainingConfig,
    OptimizerConfig,
    MCTSConfig,
    TemperatureConfig,
    ValueHeadConfig,
    load_config,
    validate_config,
)

from .experiment_db import (
    ExperimentDB,
    ExperimentRecord,
    IterationMetrics,
)

from .metrics_aggregator import (
    MetricsAggregator,
    MetricsBackend,
    JSONMetricsBackend,
    SQLiteMetricsBackend,
    ConsoleMetricsBackend,
)

from .experiment_runner import (
    run_experiment,
    ExperimentRunner,
)

__all__ = [
    # Config
    'ExperimentConfig',
    'TrainingConfig',
    'OptimizerConfig',
    'MCTSConfig',
    'TemperatureConfig',
    'ValueHeadConfig',
    'load_config',
    'validate_config',
    # Database
    'ExperimentDB',
    'ExperimentRecord',
    'IterationMetrics',
    # Metrics
    'MetricsAggregator',
    'MetricsBackend',
    'JSONMetricsBackend',
    'SQLiteMetricsBackend',
    'ConsoleMetricsBackend',
    # Runner
    'run_experiment',
    'ExperimentRunner',
]
