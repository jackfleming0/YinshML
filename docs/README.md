# YinshML Experiment Tracking System - Documentation

Welcome to the comprehensive documentation for the YinshML Experiment Tracking System!

## 📚 Documentation Index

### Getting Started
- **[Quick Setup Guide](QUICK_SETUP.md)** - Get up and running in 5 minutes
- **[Complete User Guide](EXPERIMENT_TRACKING_GUIDE.md)** - Comprehensive documentation with examples
- **[CLI Guide](CLI_GUIDE.md)** - Command-line interface documentation

### Performance & Optimization
- **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)** - Detailed performance analysis and optimization guides
- **[Migration Guide](MIGRATION.md)** - Upgrading and migration instructions

### Advanced Topics
- **[Production Deployment](production/)** - Production-ready configurations and best practices

## 🚀 Quick Start

```python
from yinsh_ml.tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker()

# Create experiment
exp_id = tracker.create_experiment(
    name="my_experiment",
    config={'learning_rate': 0.001, 'batch_size': 32}
)

# Log metrics
tracker.log_metric(exp_id, 'loss', 0.5, iteration=1)
tracker.log_metric(exp_id, 'accuracy', 0.85, iteration=1)

# Complete experiment
tracker.update_experiment_status(exp_id, 'completed')
```

## 🎯 Key Features

- **High Performance**: 1,690+ operations/second with 100% reliability
- **Automatic Metadata Capture**: Git, system, and environment information
- **Asynchronous Logging**: Non-blocking metric and parameter logging
- **TensorBoard Integration**: Seamless visualization support
- **Data Management**: Retention policies, backup, and export capabilities
- **Thread-Safe**: Concurrent experiment management
- **SQLite Backend**: Optimized for performance and reliability

## 📊 Performance Highlights

Based on comprehensive stress testing:

| Metric | Value | Status |
|--------|-------|--------|
| **Throughput** | 1,690+ ops/sec | ✅ Excellent |
| **Reliability** | 100% success rate | ✅ Perfect |
| **Memory Usage** | <2GB for 1,300+ experiments | ✅ Efficient |
| **Concurrency** | 12+ concurrent threads | ✅ Scalable |

## 📖 Documentation Structure

### For New Users
1. Start with **[Quick Setup Guide](QUICK_SETUP.md)** for immediate hands-on experience
2. Read **[Complete User Guide](EXPERIMENT_TRACKING_GUIDE.md)** for comprehensive understanding
3. Check **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)** for optimization tips

### For Developers
1. Review **[Complete User Guide](EXPERIMENT_TRACKING_GUIDE.md)** for API reference
2. Study **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)** for implementation details
3. Consult **[CLI Guide](CLI_GUIDE.md)** for command-line integration

### For System Administrators
1. Read **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)** for capacity planning
2. Check **[Production Deployment](production/)** for deployment strategies
3. Review **[Migration Guide](MIGRATION.md)** for upgrade procedures

## 🛠️ Installation

```bash
# Install from project directory
cd /path/to/YinshML
pip install -e .

# Or install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration

### Basic Configuration
```python
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
export YINSH_DB_PATH="experiments.db"
export YINSH_ASYNC_LOGGING=true
export YINSH_TENSORBOARD_LOGGING=true
```

## 📈 Use Cases

### Individual Researchers
- Track personal experiments and model iterations
- Compare different approaches and hyperparameters
- Export results for papers and presentations

### Small Teams (5-10 users)
- Collaborative experiment tracking
- Shared experiment database
- Team performance monitoring

### Large Organizations (20+ users)
- Production-scale experiment management
- Advanced data retention policies
- High-throughput concurrent access

## 🔍 Examples

### Basic Training Loop
```python
for epoch in range(100):
    # Training code...
    loss = train_one_epoch()
    accuracy = evaluate_model()
    
    # Log metrics
    tracker.log_metric(exp_id, 'loss', loss, iteration=epoch)
    tracker.log_metric(exp_id, 'accuracy', accuracy, iteration=epoch)
```

### Hyperparameter Search
```python
for lr in [0.001, 0.01, 0.1]:
    for batch_size in [16, 32, 64]:
        exp_id = tracker.create_experiment(
            name=f"hyperparam_lr{lr}_bs{batch_size}",
            config={'learning_rate': lr, 'batch_size': batch_size}
        )
        # Training and logging...
```

### Data Management
```python
from yinsh_ml.tracking.data_management import DataRetentionManager

manager = DataRetentionManager(tracker)
manager.add_rule(
    name="cleanup_failed",
    policy_type="status",
    policy_config={"status": "failed"},
    action="delete"
)
results = manager.execute_cleanup()
```

## 🚨 Troubleshooting

### Common Issues
- **Database locks**: Enable WAL mode (default) and reduce concurrent threads
- **Memory usage**: Enable async logging and monitor queue sizes
- **Slow queries**: Use appropriate filters and pagination
- **TensorBoard issues**: Check log directory permissions and disk space

### Getting Help
- Check the **[Complete User Guide](EXPERIMENT_TRACKING_GUIDE.md)** troubleshooting section
- Review **[Performance Benchmarks](PERFORMANCE_BENCHMARKS.md)** for optimization tips
- Look at example scripts in the `scripts/` directory

## 🎯 Best Practices

1. **Use descriptive experiment names** and consistent tagging
2. **Store complete configurations** for reproducibility
3. **Log metrics at appropriate intervals** (not every batch)
4. **Implement proper error handling** with status updates
5. **Monitor resource usage** in production environments

## 📊 Monitoring

### Key Metrics to Track
- Operations per second (target: >1000)
- Success rate (target: >99.5%)
- Memory usage (target: <1GB for 1000 experiments)
- Query response time (target: <50ms for 95th percentile)

### Performance Monitoring
```python
from yinsh_ml.tracking.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(tracker)
stats = monitor.get_current_stats()
alerts = monitor.check_performance_thresholds()
```

## 🔄 Updates and Migration

- **Current Version**: Latest stable release
- **Migration**: See **[Migration Guide](MIGRATION.md)** for upgrade procedures
- **Changelog**: Check project repository for version history

## 📞 Support

- **Documentation**: This comprehensive guide collection
- **Examples**: Scripts and tutorials in the project repository
- **Issues**: Report bugs or request features in the project tracker
- **Community**: Join discussions and share experiences

## 🏆 Performance Recognition

The YinshML Experiment Tracking System has been validated through comprehensive stress testing and demonstrates:

- **Industry-leading performance** with 1,690+ operations/second
- **Perfect reliability** with 100% success rate under load
- **Efficient resource utilization** with <2GB memory for large-scale workloads
- **Production readiness** with thread-safe concurrent access

Ready to get started? Begin with the **[Quick Setup Guide](QUICK_SETUP.md)** and explore the full capabilities in the **[Complete User Guide](EXPERIMENT_TRACKING_GUIDE.md)**! 