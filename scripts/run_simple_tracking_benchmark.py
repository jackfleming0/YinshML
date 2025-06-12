#!/usr/bin/env python3
"""
Simple tracking performance benchmark.

Measures the overhead of experiment tracking on training pipelines.
"""

import time
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.tracking.experiment_tracker import ExperimentTracker
from yinsh_ml.tracking import utils as tracking_utils


def simulate_training_step():
    """Simulate a single training step."""
    # Create tensors
    batch_size = 32
    input_tensor = torch.randn(batch_size, 128)
    target_tensor = torch.randn(batch_size, 10)
    
    # Forward pass
    weight = torch.randn(10, 128, requires_grad=True)
    output = torch.nn.functional.linear(input_tensor, weight)
    loss = torch.nn.functional.mse_loss(output, target_tensor)
    
    # Backward pass
    loss.backward()
    
    # Small delay to simulate computation
    time.sleep(0.001)
    
    return float(loss.item())


def benchmark_without_tracking(num_epochs=5, steps_per_epoch=50):
    """Benchmark training without tracking."""
    print("Running baseline (no tracking)...")
    
    start_time = time.perf_counter()
    
    total_steps = 0
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            loss = simulate_training_step()
            total_steps += 1
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Baseline: {duration:.3f}s for {total_steps} steps ({duration/total_steps*1000:.2f}ms per step)")
    return duration, total_steps


def benchmark_with_tracking(num_epochs=5, steps_per_epoch=50, tracking_frequency=1):
    """Benchmark training with tracking enabled."""
    print(f"Running with tracking (frequency={tracking_frequency})...")
    
    # Setup tracking
    temp_dir = Path(tempfile.mkdtemp(prefix="tracking_benchmark_"))
    db_path = temp_dir / "benchmark.db"
    
    try:
        # Initialize tracker with minimal overhead configuration
        tracker_config = {
            'async_logging': False,
            'capture_git': False,
            'capture_environment': False,
            'capture_system': False,
            'tensorboard_enabled': False,
        }
        
        ExperimentTracker._instance = None
        tracker = ExperimentTracker(str(db_path), tracker_config)
        tracking_utils.set_database_path(str(db_path))
        
        # Create experiment
        exp_id = tracker.create_experiment(
            name="benchmark_experiment",
            description="Performance benchmark",
            config={'num_epochs': num_epochs, 'steps_per_epoch': steps_per_epoch}
        )
        
        start_time = time.perf_counter()
        
        total_steps = 0
        metrics_logged = 0
        
        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                loss = simulate_training_step()
                total_steps += 1
                
                # Log metrics based on frequency
                if step % tracking_frequency == 0:
                    global_step = epoch * steps_per_epoch + step
                    tracker.log_metric(exp_id, "train_loss", loss, global_step)
                    tracker.log_metric(exp_id, "learning_rate", 0.001 * (0.99 ** epoch), global_step)
                    metrics_logged += 2
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        print(f"With tracking: {duration:.3f}s for {total_steps} steps ({duration/total_steps*1000:.2f}ms per step)")
        print(f"Metrics logged: {metrics_logged}")
        
        return duration, total_steps, metrics_logged
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        ExperimentTracker._instance = None


def main():
    """Run the benchmark comparison."""
    print("=== Experiment Tracking Performance Benchmark ===\n")
    
    # Configuration
    num_epochs = 5
    steps_per_epoch = 50
    
    # Run baseline
    baseline_duration, total_steps = benchmark_without_tracking(num_epochs, steps_per_epoch)
    
    print()
    
    # Run with tracking at different frequencies
    frequencies = [1, 10]
    
    for freq in frequencies:
        tracking_duration, _, metrics_logged = benchmark_with_tracking(
            num_epochs, steps_per_epoch, freq
        )
        
        # Calculate overhead
        overhead_pct = ((tracking_duration - baseline_duration) / baseline_duration) * 100
        overhead_ms = (tracking_duration - baseline_duration) / total_steps * 1000
        
        print(f"Overhead: {overhead_pct:+.1f}% ({overhead_ms:+.2f}ms per step)")
        print(f"Throughput: {metrics_logged / tracking_duration:.1f} metrics/sec")
        print()
    
    # Performance assessment
    print("=== Performance Assessment ===")
    
    # Test with frequency=10 (more realistic)
    realistic_duration, _, _ = benchmark_with_tracking(num_epochs, steps_per_epoch, 10)
    realistic_overhead = ((realistic_duration - baseline_duration) / baseline_duration) * 100
    
    if realistic_overhead < 5.0:
        print(f"✅ Target Met: {realistic_overhead:.1f}% overhead is below 5% target")
    else:
        print(f"❌ Target Missed: {realistic_overhead:.1f}% overhead exceeds 5% target")
    
    print(f"\nBaseline time per step: {baseline_duration/total_steps*1000:.2f}ms")
    print(f"Tracking time per step: {realistic_duration/total_steps*1000:.2f}ms")
    print(f"Absolute overhead: {(realistic_duration-baseline_duration)/total_steps*1000:.2f}ms per step")


if __name__ == "__main__":
    main() 