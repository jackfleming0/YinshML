# YinshML Experiment Tracking - Quick Setup Guide

## 5-Minute Setup

Get started with YinshML experiment tracking in just a few minutes!

## Prerequisites

- Python 3.8 or higher
- Git (optional, for metadata capture)

## Installation

```bash
# Install from the project directory
cd /path/to/YinshML
pip install -e .

# Or install required dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Import and Initialize

```python
from yinsh_ml.tracking import ExperimentTracker

# Create tracker (uses default database: experiments.db)
tracker = ExperimentTracker()
```

### 2. Create Your First Experiment

```python
# Create experiment
exp_id = tracker.create_experiment(
    name="my_first_experiment",
    description="Testing the tracking system",
    tags=["tutorial", "test"],
    config={
        'learning_rate': 0.001,
        'batch_size': 32,
        'model_type': 'neural_network'
    }
)

print(f"Created experiment {exp_id}")
```

### 3. Log Metrics During Training

```python
# Simulate training loop
for epoch in range(10):
    # Your training code here...
    loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
    accuracy = 0.5 + (epoch * 0.05)  # Simulated increasing accuracy
    
    # Log metrics
    tracker.log_metric(exp_id, 'loss', loss, iteration=epoch)
    tracker.log_metric(exp_id, 'accuracy', accuracy, iteration=epoch)
    
    print(f"Epoch {epoch}: Loss={loss:.3f}, Accuracy={accuracy:.3f}")
```

### 4. Complete the Experiment

```python
# Mark experiment as completed
tracker.update_experiment_status(exp_id, 'completed')
print("Experiment completed!")
```

## Complete Example

```python
from yinsh_ml.tracking import ExperimentTracker
import random
import time

def run_simple_experiment():
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Create experiment
    exp_id = tracker.create_experiment(
        name="simple_training_example",
        description="A simple training example",
        tags=["example", "tutorial"],
        config={
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'model': 'simple_nn'
        }
    )
    
    print(f"Started experiment {exp_id}")
    
    try:
        # Simulate training
        for epoch in range(10):
            # Simulate training metrics
            loss = random.uniform(0.1, 1.0) * (1 - epoch/10)
            accuracy = random.uniform(0.5, 0.95) * (epoch/10 + 0.1)
            learning_rate = 0.001 * (0.95 ** epoch)  # Decay
            
            # Log metrics
            tracker.log_metric(exp_id, 'loss', loss, iteration=epoch)
            tracker.log_metric(exp_id, 'accuracy', accuracy, iteration=epoch)
            tracker.log_metric(exp_id, 'learning_rate', learning_rate, iteration=epoch)
            
            print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={accuracy:.4f}")
            time.sleep(0.1)  # Simulate training time
        
        # Mark as completed
        tracker.update_experiment_status(exp_id, 'completed')
        print(f"✅ Experiment {exp_id} completed successfully!")
        
        return exp_id
        
    except Exception as e:
        # Mark as failed if something goes wrong
        tracker.update_experiment_status(exp_id, 'failed')
        print(f"❌ Experiment {exp_id} failed: {e}")
        raise

if __name__ == "__main__":
    run_simple_experiment()
```

## Viewing Your Results

### Query Experiments

```python
# Get all experiments
experiments = tracker.query_experiments()
for exp in experiments:
    print(f"ID: {exp['id']}, Name: {exp['name']}, Status: {exp['status']}")

# Get specific experiment details
exp_data = tracker.get_experiment(exp_id)
print(f"Experiment: {exp_data['name']}")
print(f"Config: {exp_data['config']}")
print(f"Status: {exp_data['status']}")
```

### View Metrics

```python
# Get metric history
loss_history = tracker.get_metric_history(exp_id, 'loss')
for point in loss_history:
    print(f"Iteration {point['iteration']}: {point['value']:.4f}")
```

### Export Data

```python
# Export experiment to JSON
tracker.export_experiment_data(
    experiment_id=exp_id,
    file_path=f"experiment_{exp_id}.json",
    include_metrics=True
)
print(f"Exported experiment {exp_id} to JSON")
```

## Configuration Options

### Basic Configuration

```python
# Custom database location
tracker = ExperimentTracker(db_path="my_experiments.db")

# With configuration
tracker = ExperimentTracker(
    db_path="experiments.db",
    config={
        'async_logging': True,
        'tensorboard_logging': True,
        'auto_git_capture': True
    }
)
```

### Environment Variables

```bash
# Set database path
export YINSH_DB_PATH="experiments.db"

# Enable async logging
export YINSH_ASYNC_LOGGING=true

# Enable TensorBoard integration
export YINSH_TENSORBOARD_LOGGING=true
export YINSH_TENSORBOARD_LOG_DIR="./tensorboard_logs"
```

## Next Steps

1. **Read the Full Guide**: Check out [EXPERIMENT_TRACKING_GUIDE.md](EXPERIMENT_TRACKING_GUIDE.md) for comprehensive documentation

2. **Performance Optimization**: See [PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md) for optimization tips

3. **Advanced Features**: Explore data management, retention policies, and backup options

4. **Integration**: Learn how to integrate with your existing ML workflows

## Common Issues

### Database Permissions
```bash
# Ensure write permissions for database directory
chmod 755 .
chmod 644 experiments.db  # if it exists
```

### Import Errors
```bash
# Install in development mode
pip install -e .

# Or install dependencies
pip install -r requirements.txt
```

### TensorBoard Integration
```bash
# Install TensorBoard if needed
pip install tensorboard

# View logs
tensorboard --logdir=./tensorboard_logs
```

## Getting Help

- **Documentation**: [docs/EXPERIMENT_TRACKING_GUIDE.md](EXPERIMENT_TRACKING_GUIDE.md)
- **Examples**: Check the `scripts/` directory for more examples
- **Issues**: Report bugs or ask questions in the project repository

## Performance Notes

The system is designed for high performance:
- **1,690+ operations/second** under load
- **100% reliability** in stress tests
- **Efficient memory usage** (<2GB for 1,300+ experiments)
- **Thread-safe** for concurrent access

You can start with the default configuration and optimize later based on your specific needs! 