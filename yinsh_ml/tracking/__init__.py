"""
Experiment tracking module for YinshML.

This module provides comprehensive experiment tracking, reproducibility,
and analysis capabilities for machine learning experiments.
"""

# Lazy imports to avoid dependency issues
__all__ = [
    'ExperimentTracker',  # High-level singleton interface
    'ExperimentDatabase', 'create_database',
    'initialize_database', 'create_experiment', 'add_metric_to_experiment', 
    'add_metrics_bulk', 'query_experiments', 'get_experiment_by_id',
    'get_experiment_metrics', 'get_experiment_tags', 'update_experiment_status',
    'delete_experiment', 'get_database_stats', 'close_all_connections'
]

def __getattr__(name):
    if name == 'ExperimentTracker':
        from .experiment_tracker import ExperimentTracker
        return ExperimentTracker
    elif name == 'ExperimentDatabase':
        from .database import ExperimentDatabase
        return ExperimentDatabase
    elif name == 'create_database':
        from .database import create_database
        return create_database
    elif name in ['initialize_database', 'create_experiment', 'add_metric_to_experiment', 
                  'add_metrics_bulk', 'query_experiments', 'get_experiment_by_id',
                  'get_experiment_metrics', 'get_experiment_tags', 'update_experiment_status',
                  'delete_experiment', 'get_database_stats', 'close_all_connections']:
        from . import utils
        return getattr(utils, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 