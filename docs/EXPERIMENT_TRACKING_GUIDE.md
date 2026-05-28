# YinshML Experiment Tracking System - Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Performance & Optimization](#performance--optimization)
8. [Data Management](#data-management)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

The YinshML Experiment Tracking System is a comprehensive, high-performance solution for managing machine learning experiments. It provides automatic metadata capture, efficient metric logging, and robust data management capabilities.

### Key Features

- **High Performance**: 1,690+ operations/second with 100% reliability
- **Automatic Metadata Capture**: Git, system, and environment information
- **Asynchronous Logging**: Non-blocking metric and parameter logging
- **TensorBoard Integration**: Seamless visualization support
- **Data Management**: Retention policies, backup, and export capabilities
- **Thread-Safe**: Concurrent experiment management
- **SQLite Backend**: Optimized for performance and reliability

### Performance Characteristics

Based on comprehensive stress testing:
- **Throughput**: 1,690+ operations/second
- **Reliability**: 100% success rate under load
- **Memory Efficiency**: <2GB for 1,300+ concurrent experiments
- **Scalability**: Handles 12+ concurrent threads

## Installation & Setup

### Requirements

- Python 3.8+
- SQLite 3.24+
- Git (for metadata capture)
- Optional: TensorBoard for visualization

### Installation

```bash
# Install YinshML
pip install yinsh-ml

# Or install from source
git clone https://github.com/your-org/yinsh-ml.git
cd yinsh-ml
pip install -e .
```

### Basic Configuration

```python
from yinsh_ml.tracking import ExperimentTracker

# Initialize with default settings
tracker = ExperimentTracker()

# Or with custom configuration
tracker = ExperimentTracker(
    db_path="experiments.db",
    config={
        'async_logging': True,
        'tensorboard_logging': True,
        'auto_git_capture': True,
        'max_queue_size': 10000
    }
)
```

### Environment Variables

Configure behavior via environment variables:

```bash
# Database configuration
export YINSH_DB_PATH="experiments.db"

# Async logging settings
export YINSH_ASYNC_LOGGING=true
export YINSH_MAX_QUEUE_SIZE=10000
export YINSH_FLUSH_INTERVAL=1.0

# TensorBoard integration
export YINSH_TENSORBOARD_LOGGING=true
export YINSH_TENSORBOARD_LOG_DIR="./tensorboard_logs"

# Git metadata capture
export YINSH_AUTO_GIT_CAPTURE=true
```

## Quick Start

### Basic Experiment

```python
from yinsh_ml.tracking import ExperimentTracker
import random

# Initialize tracker
tracker = ExperimentTracker()

# Create experiment
exp_id = tracker.create_experiment(
    name="my_first_experiment",
    description="Testing the tracking system",
    tags=["tutorial", "test"],
    config={
        'model_type': 'neural_network',
        'learning_rate': 0.001,
        'batch_size': 32
    }
)

# Log metrics during training
for epoch in range(10):
    loss = random.uniform(0.1, 1.0) * (1 - epoch/10)
    accuracy = random.uniform(0.5, 0.95) * (epoch/10 + 0.1)
    
    tracker.log_metric(exp_id, 'loss', loss, iteration=epoch)
    tracker.log_metric(exp_id, 'accuracy', accuracy, iteration=epoch)

# Complete experiment
tracker.update_experiment_status(exp_id, 'completed')

print(f"Experiment {exp_id} completed successfully!")
```

### Advanced Usage with Context Manager

```python
from yinsh_ml.tracking import ExperimentTracker
from contextlib import contextmanager

@contextmanager
def experiment_context(name, config, tags=None):
    tracker = ExperimentTracker()
    exp_id = tracker.create_experiment(name=name, config=config, tags=tags)
    
    try:
        yield tracker, exp_id
        tracker.update_experiment_status(exp_id, 'completed')
    except Exception as e:
        tracker.update_experiment_status(exp_id, 'failed')
        tracker.add_note(exp_id, f"Failed with error: {str(e)}")
        raise

# Usage
with experiment_context("advanced_experiment", {'lr': 0.01}) as (tracker, exp_id):
    # Your training code here
    for i in range(100):
        tracker.log_metric(exp_id, 'loss', 1.0/(i+1), iteration=i)
```

## Core Concepts

### Experiments

An experiment represents a single training run or model evaluation. Each experiment has:

- **Unique ID**: Auto-generated integer identifier
- **Metadata**: Name, description, creation time, status
- **Configuration**: Model parameters, hyperparameters
- **Metrics**: Time-series data (loss, accuracy, etc.)
- **Tags**: Categorical labels for organization
- **Status**: pending, running, completed, failed, cancelled

### Metrics

Metrics are time-series data points logged during experiment execution:

```python
# Simple metric logging
tracker.log_metric(exp_id, 'loss', 0.5, iteration=10)

# Batch metric logging for efficiency
metrics = {'loss': 0.5, 'accuracy': 0.85, 'f1_score': 0.78}
for name, value in metrics.items():
    tracker.log_metric(exp_id, name, value, iteration=10)
```

### Configuration Management

Configurations are automatically captured and stored:

```python
config = {
    'model': {
        'type': 'transformer',
        'layers': 12,
        'hidden_size': 768
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    },
    'data': {
        'dataset': 'custom_dataset',
        'train_split': 0.8,
        'validation_split': 0.1
    }
}

exp_id = tracker.create_experiment(
    name="transformer_training",
    config=config
)
```

## API Reference

### ExperimentTracker Class

#### Initialization

```python
ExperimentTracker(db_path=None, config=None)
```

**Parameters:**
- `db_path` (str, optional): Path to SQLite database file
- `config` (dict, optional): Configuration dictionary

#### Core Methods

##### create_experiment()

```python
create_experiment(name, description=None, tags=None, config=None) -> int
```

Creates a new experiment and returns its ID.

**Parameters:**
- `name` (str): Experiment name
- `description` (str, optional): Experiment description
- `tags` (List[str], optional): List of tags
- `config` (dict, optional): Configuration dictionary

**Returns:** Experiment ID (int)

##### log_metric()

```python
log_metric(experiment_id, metric_name, value, iteration=None, timestamp=None)
```

Logs a metric value for an experiment.

**Parameters:**
- `experiment_id` (int): Experiment ID
- `metric_name` (str): Name of the metric
- `value` (float): Metric value
- `iteration` (int, optional): Training iteration/epoch
- `timestamp` (datetime, optional): Custom timestamp

##### log_parameter()

```python
log_parameter(experiment_id, param_name, value)
```

Logs a parameter value for an experiment.

**Parameters:**
- `experiment_id` (int): Experiment ID
- `param_name` (str): Parameter name
- `value` (Any): Parameter value

##### update_experiment_status()

```python
update_experiment_status(experiment_id, status)
```

Updates experiment status.

**Parameters:**
- `experiment_id` (int): Experiment ID
- `status` (str): New status ('pending', 'running', 'completed', 'failed', 'cancelled')

##### get_experiment()

```python
get_experiment(experiment_id) -> Dict[str, Any]
```

Retrieves experiment details.

**Parameters:**
- `experiment_id` (int): Experiment ID

**Returns:** Dictionary containing experiment data

##### query_experiments()

```python
query_experiments(status=None, tags=None, start_date=None, end_date=None, 
                 limit=None, offset=0, include_metrics=False, include_tags=True) -> List[Dict]
```

Queries experiments with filtering options.

**Parameters:**
- `status` (str, optional): Filter by status
- `tags` (List[str], optional): Filter by tags
- `start_date` (date, optional): Filter by start date
- `end_date` (date, optional): Filter by end date
- `limit` (int, optional): Maximum number of results
- `offset` (int): Result offset for pagination
- `include_metrics` (bool): Include metric data
- `include_tags` (bool): Include tag data

**Returns:** List of experiment dictionaries

##### get_metric_history()

```python
get_metric_history(experiment_id, metric_name) -> List[Dict[str, Any]]
```

Retrieves metric history for an experiment.

**Parameters:**
- `experiment_id` (int): Experiment ID
- `metric_name` (str): Name of the metric

**Returns:** List of metric data points

##### export_experiment_data()

```python
export_experiment_data(experiment_id, format='json', include_metrics=True, 
                      include_config=True, file_path=None) -> Union[str, Dict]
```

Exports experiment data to file or returns as dictionary.

**Parameters:**
- `experiment_id` (int): Experiment ID
- `format` (str): Export format ('json', 'csv')
- `include_metrics` (bool): Include metric data
- `include_config` (bool): Include configuration data
- `file_path` (str, optional): Output file path

**Returns:** File path (str) or data dictionary

### Data Management Classes

#### DataRetentionManager

```python
from yinsh_ml.tracking.data_management import DataRetentionManager

manager = DataRetentionManager(tracker)

# Add retention rules
manager.add_rule(
    name="cleanup_failed",
    policy_type="status",
    policy_config={"status": "failed"},
    action="delete",
    priority=1
)

# Execute cleanup
results = manager.execute_cleanup(dry_run=False)
```

#### DataBackupManager

```python
from yinsh_ml.tracking.data_management import DataBackupManager

backup_manager = DataBackupManager(tracker)

# Export single experiment
result = backup_manager.export_experiment(
    experiment_id=123,
    output_path="experiment_123.json",
    include_metrics=True
)
```

## Usage Examples

### Training Loop Integration

```python
import torch
import torch.nn as nn
from yinsh_ml.tracking import ExperimentTracker

def train_model(model, train_loader, val_loader, config):
    tracker = ExperimentTracker()
    
    # Create experiment
    exp_id = tracker.create_experiment(
        name=f"training_{config['model_name']}",
        description="Model training with PyTorch",
        config=config,
        tags=["pytorch", "training", config['model_name']]
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    try:
        for epoch in range(config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Log batch metrics
                if batch_idx % 100 == 0:
                    tracker.log_metric(exp_id, 'batch_loss', loss.item(), 
                                     iteration=epoch * len(train_loader) + batch_idx)
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)
            
            # Log epoch metrics
            tracker.log_metric(exp_id, 'train_loss', train_loss, iteration=epoch)
            tracker.log_metric(exp_id, 'train_accuracy', train_acc, iteration=epoch)
            tracker.log_metric(exp_id, 'val_loss', val_loss, iteration=epoch)
            tracker.log_metric(exp_id, 'val_accuracy', val_acc, iteration=epoch)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
        tracker.update_experiment_status(exp_id, 'completed')
        
    except Exception as e:
        tracker.update_experiment_status(exp_id, 'failed')
        tracker.add_note(exp_id, f"Training failed: {str(e)}")
        raise
    
    return exp_id
```

### Hyperparameter Tuning

```python
from itertools import product
from yinsh_ml.tracking import ExperimentTracker

def hyperparameter_search():
    tracker = ExperimentTracker()
    
    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    hidden_sizes = [128, 256, 512]
    
    best_exp_id = None
    best_accuracy = 0.0
    
    for lr, batch_size, hidden_size in product(learning_rates, batch_sizes, hidden_sizes):
        config = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'epochs': 50
        }
        
        exp_id = tracker.create_experiment(
            name=f"hyperparam_search_lr{lr}_bs{batch_size}_hs{hidden_size}",
            description="Hyperparameter search experiment",
            config=config,
            tags=["hyperparameter_search", "grid_search"]
        )
        
        try:
            # Train model with these hyperparameters
            final_accuracy = train_with_config(config, tracker, exp_id)
            
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_exp_id = exp_id
            
            tracker.update_experiment_status(exp_id, 'completed')
            
        except Exception as e:
            tracker.update_experiment_status(exp_id, 'failed')
            tracker.add_note(exp_id, f"Failed: {str(e)}")
    
    # Tag best experiment
    if best_exp_id:
        tracker.add_tags(best_exp_id, ['best_model'])
        print(f"Best experiment: {best_exp_id} with accuracy: {best_accuracy:.4f}")
    
    return best_exp_id
```

### Experiment Comparison

```python
def compare_experiments(exp_ids, metric_names=['loss', 'accuracy']):
    tracker = ExperimentTracker()
    
    comparison_data = tracker.compare_experiments(exp_ids, metric_names)
    
    print("Experiment Comparison:")
    print("=" * 50)
    
    for exp_id in exp_ids:
        exp_data = tracker.get_experiment(exp_id)
        print(f"\nExperiment {exp_id}: {exp_data['name']}")
        print(f"Status: {exp_data['status']}")
        print(f"Tags: {', '.join(exp_data.get('tags', []))}")
        
        for metric in metric_names:
            if metric in comparison_data['metrics']:
                final_value = comparison_data['metrics'][metric].get(str(exp_id), {}).get('final')
                if final_value is not None:
                    print(f"{metric}: {final_value:.4f}")
    
    # Find best performing experiment
    best_exp = None
    best_accuracy = -1
    
    for exp_id in exp_ids:
        accuracy = comparison_data['metrics'].get('accuracy', {}).get(str(exp_id), {}).get('final')
        if accuracy and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_exp = exp_id
    
    if best_exp:
        print(f"\nBest performing experiment: {best_exp} (accuracy: {best_accuracy:.4f})")
    
    return comparison_data
```

## Performance & Optimization

### Performance Characteristics

Based on comprehensive stress testing with 1,300 experiments and 585,000 metrics:

- **Throughput**: 1,690+ operations/second
- **Memory Usage**: Peak 1.97GB, Average 701MB
- **Success Rate**: 100% reliability
- **Concurrency**: Handles 12+ concurrent threads

### Optimization Strategies

#### 1. Asynchronous Logging

Enable async logging for high-throughput scenarios:

```python
tracker = ExperimentTracker(config={
    'async_logging': True,
    'max_queue_size': 10000,
    'flush_interval': 1.0,
    'batch_size': 100
})

# Start async logging
tracker.start_async_logging()

# Your training code here...

# Ensure all data is written before shutdown
tracker.flush_async_queue(timeout=10.0)
tracker.stop_async_logging()
```

#### 2. Batch Operations

Group related operations for better performance:

```python
# Instead of individual metric calls
for metric_name, value in metrics.items():
    tracker.log_metric(exp_id, metric_name, value, iteration=epoch)

# Use batch logging when available
# (Note: Implement batch_log_metrics for even better performance)
```

#### 3. Database Optimization

The system automatically applies SQLite optimizations:

- WAL mode for concurrent access
- Memory mapping for faster I/O
- Optimized cache settings
- Comprehensive indexing

#### 4. Memory Management

Monitor memory usage in long-running processes:

```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    # Force garbage collection if needed
    if memory_mb > 1000:  # 1GB threshold
        gc.collect()

# Call periodically during training
```

### Performance Tuning Configuration

```python
# High-performance configuration
high_perf_config = {
    # Async logging settings
    'async_logging': True,
    'max_queue_size': 20000,
    'flush_interval': 0.5,
    'batch_size': 200,
    'max_retries': 3,
    
    # Database settings
    'db_cache_size': 100000,  # 100MB cache
    'db_synchronous': 'NORMAL',
    'db_journal_mode': 'WAL',
    'db_temp_store': 'MEMORY',
    
    # TensorBoard settings
    'tensorboard_logging': True,
    'tensorboard_flush_interval': 30,
    
    # Metadata capture
    'auto_git_capture': True,
    'capture_system_info': True,
    'capture_environment': True
}

tracker = ExperimentTracker(config=high_perf_config)
```

## Data Management

### Retention Policies

Implement automated data cleanup:

```python
from yinsh_ml.tracking.data_management import DataRetentionManager, RetentionRule

manager = DataRetentionManager(tracker)

# Clean up failed experiments older than 30 days
manager.add_rule(RetentionRule(
    name="cleanup_old_failed",
    policy_type="age",
    policy_config={"days": 30},
    conditions={"status": "failed"},
    action="delete",
    priority=1
))

# Archive completed experiments older than 90 days
manager.add_rule(RetentionRule(
    name="archive_old_completed",
    policy_type="age", 
    policy_config={"days": 90},
    conditions={"status": "completed"},
    action="archive",
    priority=2
))

# Keep only top 100 experiments by accuracy
manager.add_rule(RetentionRule(
    name="keep_top_performers",
    policy_type="count",
    policy_config={"max_count": 100, "sort_by": "accuracy", "sort_order": "desc"},
    action="delete",
    priority=3
))

# Execute cleanup
results = manager.execute_cleanup(dry_run=False)
print(f"Cleaned up {results.experiments_processed} experiments")
```

### Backup and Export

```python
from yinsh_ml.tracking.data_management import DataBackupManager

backup_manager = DataBackupManager(tracker)

# Export single experiment
backup_manager.export_experiment(
    experiment_id=123,
    output_path="experiment_123.json",
    include_metrics=True,
    compress=True
)

# Backup multiple experiments
experiment_ids = [1, 2, 3, 4, 5]
for exp_id in experiment_ids:
    backup_manager.export_experiment(
        experiment_id=exp_id,
        output_path=f"backups/experiment_{exp_id}.json.gz",
        compress=True
    )
```

## Troubleshooting

### Common Issues

#### 1. Database Lock Errors

**Problem**: `database is locked` errors during concurrent access.

**Solution**: 
- Ensure WAL mode is enabled (default)
- Reduce concurrent thread count
- Add retry logic with exponential backoff

```python
import time
import random

def retry_operation(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(random.uniform(0.1, 0.5) * (2 ** attempt))
                continue
            raise
```

#### 2. Memory Issues

**Problem**: High memory usage during long training runs.

**Solution**:
- Enable async logging to reduce memory pressure
- Implement periodic garbage collection
- Monitor memory usage and adjust batch sizes

```python
# Memory-efficient configuration
tracker = ExperimentTracker(config={
    'async_logging': True,
    'max_queue_size': 5000,  # Smaller queue
    'flush_interval': 0.5,   # More frequent flushes
})
```

#### 3. Slow Query Performance

**Problem**: Slow experiment queries with large datasets.

**Solution**:
- Use appropriate filters (status, tags, date ranges)
- Implement pagination for large result sets
- Consider database maintenance

```python
# Efficient querying
experiments = tracker.query_experiments(
    status='completed',
    tags=['production'],
    limit=50,
    include_metrics=False  # Exclude metrics for faster queries
)
```

#### 4. TensorBoard Integration Issues

**Problem**: TensorBoard logs not appearing or corrupted.

**Solution**:
- Verify TensorBoard log directory permissions
- Ensure proper flush intervals
- Check for disk space issues

```python
# Explicit TensorBoard flushing
tracker.flush_tensorboard()

# Check TensorBoard configuration
tb_config = tracker.get_tensorboard_config()
print(f"TensorBoard log dir: {tb_config['log_dir']}")
```

### Debugging Tools

#### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('yinsh_ml.tracking')
logger.setLevel(logging.DEBUG)
```

#### Performance Monitoring

```python
def monitor_tracker_performance(tracker):
    """Monitor tracker performance metrics."""
    if hasattr(tracker, 'get_async_stats'):
        stats = tracker.get_async_stats()
        print(f"Async queue size: {stats.get('queue_size', 0)}")
        print(f"Items processed: {stats.get('items_processed', 0)}")
        print(f"Items failed: {stats.get('items_failed', 0)}")
        print(f"Processing time: {stats.get('total_processing_time', 0):.2f}s")
```

#### Database Health Check

```python
def check_database_health(tracker):
    """Check database health and performance."""
    db_path = tracker.get_database_path()
    
    # Check file size
    import os
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"Database size: {size_mb:.2f} MB")
    
    # Check experiment count
    experiments = tracker.query_experiments(limit=1)
    total_count = len(tracker.query_experiments())
    print(f"Total experiments: {total_count}")
    
    # Check for active experiments
    active = tracker.get_active_experiments()
    print(f"Active experiments: {len(active)}")
```

## Best Practices

### 1. Experiment Organization

```python
# Use descriptive names and consistent tagging
exp_id = tracker.create_experiment(
    name="resnet50_imagenet_lr0.001_bs32_aug",
    description="ResNet-50 training on ImageNet with data augmentation",
    tags=["resnet", "imagenet", "production", "v2.1"],
    config=config
)
```

### 2. Configuration Management

```python
# Store complete configuration for reproducibility
config = {
    'model': {
        'architecture': 'resnet50',
        'pretrained': True,
        'num_classes': 1000
    },
    'training': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'optimizer': 'adam',
        'weight_decay': 1e-4
    },
    'data': {
        'dataset': 'imagenet',
        'augmentation': True,
        'normalize': True,
        'train_split': 0.8
    },
    'environment': {
        'python_version': '3.8.10',
        'pytorch_version': '1.9.0',
        'cuda_version': '11.1'
    }
}
```

### 3. Metric Logging Strategy

```python
# Log metrics at appropriate intervals
def log_training_metrics(tracker, exp_id, epoch, batch_idx, total_batches):
    # Log every 100 batches during training
    if batch_idx % 100 == 0:
        tracker.log_metric(exp_id, 'batch_loss', loss.item(), 
                         iteration=epoch * total_batches + batch_idx)
    
    # Log epoch-level metrics
    if batch_idx == total_batches - 1:
        tracker.log_metric(exp_id, 'epoch_loss', epoch_loss, iteration=epoch)
        tracker.log_metric(exp_id, 'epoch_accuracy', epoch_acc, iteration=epoch)
        tracker.log_metric(exp_id, 'learning_rate', current_lr, iteration=epoch)
```

### 4. Error Handling

```python
def robust_experiment_tracking(config):
    tracker = ExperimentTracker()
    exp_id = None
    
    try:
        exp_id = tracker.create_experiment(
            name=config['name'],
            config=config
        )
        
        # Training code here
        result = train_model(config, tracker, exp_id)
        
        tracker.update_experiment_status(exp_id, 'completed')
        return result
        
    except KeyboardInterrupt:
        if exp_id:
            tracker.update_experiment_status(exp_id, 'cancelled')
            tracker.add_note(exp_id, "Cancelled by user")
        raise
        
    except Exception as e:
        if exp_id:
            tracker.update_experiment_status(exp_id, 'failed')
            tracker.add_note(exp_id, f"Failed with error: {str(e)}")
        raise
        
    finally:
        # Ensure async queue is flushed
        if hasattr(tracker, 'flush_async_queue'):
            tracker.flush_async_queue(timeout=10.0)
```

### 5. Resource Management

```python
# Use context managers for automatic cleanup
class ExperimentContext:
    def __init__(self, name, config, tags=None):
        self.tracker = ExperimentTracker()
        self.name = name
        self.config = config
        self.tags = tags
        self.exp_id = None
    
    def __enter__(self):
        self.exp_id = self.tracker.create_experiment(
            name=self.name,
            config=self.config,
            tags=self.tags
        )
        return self.tracker, self.exp_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.exp_id:
            if exc_type is None:
                self.tracker.update_experiment_status(self.exp_id, 'completed')
            else:
                self.tracker.update_experiment_status(self.exp_id, 'failed')
                self.tracker.add_note(self.exp_id, f"Failed: {exc_val}")
        
        # Cleanup
        if hasattr(self.tracker, 'flush_async_queue'):
            self.tracker.flush_async_queue()

# Usage
with ExperimentContext("my_experiment", config) as (tracker, exp_id):
    # Training code here
    pass
```

---

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/yinsh-ml/issues)
- **Documentation**: [Full API documentation](https://yinsh-ml.readthedocs.io)
- **Examples**: [Additional examples and tutorials](https://github.com/your-org/yinsh-ml/tree/main/examples)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 