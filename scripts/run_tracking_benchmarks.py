#!/usr/bin/env python3
"""
Run experiment tracking performance benchmarks.

This script runs comprehensive benchmarks to measure the performance impact
of the experiment tracking system and generates detailed reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.benchmarks.benchmark_framework import BenchmarkSuite, BenchmarkRunner
from yinsh_ml.benchmarks.tracking_benchmarks import (
    ExperimentTrackingBenchmark,
    DatabaseOperationsBenchmark,
    MetricLoggingBenchmark,
    ConfigSerializationBenchmark,
    TrainingPipelineOverheadBenchmark
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_baseline_benchmarks():
    """Create benchmark cases for baseline performance measurement."""
    return [
        # Training pipeline overhead comparison
        TrainingPipelineOverheadBenchmark(
            num_epochs=5,
            steps_per_epoch=50,
            use_tracking=False,
            tracking_frequency=1
        ),
        TrainingPipelineOverheadBenchmark(
            num_epochs=5,
            steps_per_epoch=50,
            use_tracking=True,
            tracking_frequency=1
        ),
        TrainingPipelineOverheadBenchmark(
            num_epochs=5,
            steps_per_epoch=50,
            use_tracking=True,
            tracking_frequency=10
        ),
        
        # Configuration serialization performance
        ConfigSerializationBenchmark(
            num_configs=500,
            config_complexity="simple"
        ),
        ConfigSerializationBenchmark(
            num_configs=500,
            config_complexity="medium"
        ),
        ConfigSerializationBenchmark(
            num_configs=200,
            config_complexity="complex"
        ),
    ]


def create_component_benchmarks():
    """Create benchmark cases for individual component performance."""
    return [
        # Experiment tracking end-to-end
        ExperimentTrackingBenchmark(
            num_experiments=5,
            metrics_per_experiment=50,
            config_complexity="simple",
            async_logging=False
        ),
        ExperimentTrackingBenchmark(
            num_experiments=5,
            metrics_per_experiment=50,
            config_complexity="medium",
            async_logging=False
        ),
        ExperimentTrackingBenchmark(
            num_experiments=3,
            metrics_per_experiment=50,
            config_complexity="complex",
            async_logging=False
        ),
        
        # Async vs sync comparison
        ExperimentTrackingBenchmark(
            num_experiments=5,
            metrics_per_experiment=50,
            config_complexity="medium",
            async_logging=True
        ),
        
        # Database operations
        DatabaseOperationsBenchmark(
            num_queries=500,
            query_type="read",
            data_size="small"
        ),
        DatabaseOperationsBenchmark(
            num_queries=500,
            query_type="write",
            data_size="small"
        ),
        DatabaseOperationsBenchmark(
            num_queries=500,
            query_type="mixed",
            data_size="medium"
        ),
        
        # Metric logging throughput
        MetricLoggingBenchmark(
            num_metrics=5000,
            batch_size=1,
            async_logging=False
        ),
        MetricLoggingBenchmark(
            num_metrics=5000,
            batch_size=10,
            async_logging=False
        ),
        MetricLoggingBenchmark(
            num_metrics=5000,
            batch_size=1,
            async_logging=True
        ),
    ]


def create_scalability_benchmarks():
    """Create benchmark cases for scalability testing."""
    return [
        # Large-scale experiment tracking
        ExperimentTrackingBenchmark(
            num_experiments=20,
            metrics_per_experiment=200,
            config_complexity="medium",
            async_logging=False
        ),
        
        # High-throughput metric logging
        MetricLoggingBenchmark(
            num_metrics=20000,
            batch_size=1,
            async_logging=False
        ),
        MetricLoggingBenchmark(
            num_metrics=20000,
            batch_size=50,
            async_logging=True
        ),
        
        # Large database operations
        DatabaseOperationsBenchmark(
            num_queries=2000,
            query_type="mixed",
            data_size="large"
        ),
        
        # Complex training simulation
        TrainingPipelineOverheadBenchmark(
            num_epochs=10,
            steps_per_epoch=200,
            use_tracking=True,
            tracking_frequency=5
        ),
    ]


def analyze_results(results, output_dir: Path):
    """Analyze benchmark results and generate performance insights."""
    logger.info("Analyzing benchmark results...")
    
    # Group results by benchmark type
    training_results = []
    tracking_results = []
    database_results = []
    metric_results = []
    config_results = []
    
    for result in results:
        name = result.name
        if "TrainingOverhead" in name:
            training_results.append(result)
        elif "ExperimentTracking" in name:
            tracking_results.append(result)
        elif "DatabaseOps" in name:
            database_results.append(result)
        elif "MetricLogging" in name:
            metric_results.append(result)
        elif "ConfigSerialization" in name:
            config_results.append(result)
    
    # Generate analysis report
    analysis_file = output_dir / "performance_analysis.md"
    
    with open(analysis_file, 'w') as f:
        f.write("# Experiment Tracking Performance Analysis\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # Training pipeline overhead analysis
        if training_results:
            f.write("## Training Pipeline Overhead\n\n")
            
            no_tracking = None
            with_tracking = []
            
            for result in training_results:
                if "no_tracking" in result.name:
                    no_tracking = result
                else:
                    with_tracking.append(result)
            
            if no_tracking and with_tracking:
                baseline_time = no_tracking.avg_metrics.to_dict()['timing']['duration_ms']
                f.write(f"**Baseline (no tracking)**: {baseline_time:.2f}ms per iteration\n\n")
                
                for tracked_result in with_tracking:
                    tracked_time = tracked_result.avg_metrics.to_dict()['timing']['duration_ms']
                    overhead_pct = ((tracked_time - baseline_time) / baseline_time) * 100
                    
                    freq = "1" if "freq1" in tracked_result.name else "10"
                    f.write(f"**With tracking (freq={freq})**: {tracked_time:.2f}ms "
                           f"({overhead_pct:+.1f}% overhead)\n")
                
                f.write("\n")
                
                # Check if we meet the <5% target
                min_overhead = min(
                    ((r.avg_metrics.to_dict()['timing']['duration_ms'] - baseline_time) / baseline_time) * 100
                    for r in with_tracking
                )
                
                if min_overhead < 5.0:
                    f.write("✅ **Target Met**: Minimum overhead is below 5% target\n\n")
                else:
                    f.write("❌ **Target Missed**: Minimum overhead exceeds 5% target\n\n")
        
        # Component performance analysis
        if tracking_results:
            f.write("## End-to-End Tracking Performance\n\n")
            
            for result in tracking_results:
                stats = result.calculate_statistics()
                custom_metrics = result.avg_metrics.to_dict()['custom']
                
                f.write(f"**{result.description}**\n")
                f.write(f"- Experiments/sec: {custom_metrics.get('experiments_per_sec', 0):.1f}\n")
                f.write(f"- Metrics/sec: {custom_metrics.get('metrics_per_sec', 0):.1f}\n")
                f.write(f"- Duration: {stats['duration_ms']['mean']:.2f}ms ± {stats['duration_ms']['std_dev']:.2f}ms\n")
                f.write(f"- Success rate: {stats['success_rate']*100:.1f}%\n\n")
        
        if metric_results:
            f.write("## Metric Logging Throughput\n\n")
            
            for result in metric_results:
                custom_metrics = result.avg_metrics.to_dict()['custom']
                
                f.write(f"**{result.description}**\n")
                f.write(f"- Throughput: {custom_metrics.get('metrics_per_sec', 0):.0f} metrics/sec\n")
                f.write(f"- Avg time per metric: {custom_metrics.get('avg_metric_time_us', 0):.1f}μs\n")
                f.write(f"- Batch size: {custom_metrics.get('batch_size', 1)}\n")
                f.write(f"- Async: {custom_metrics.get('async_logging', False)}\n\n")
        
        if database_results:
            f.write("## Database Operations Performance\n\n")
            
            for result in database_results:
                custom_metrics = result.avg_metrics.to_dict()['custom']
                
                f.write(f"**{result.description}**\n")
                f.write(f"- Operations/sec: {custom_metrics.get('operations_per_sec', 0):.1f}\n")
                f.write(f"- Avg operation time: {custom_metrics.get('avg_operation_time_ms', 0):.2f}ms\n")
                f.write(f"- Read ops: {custom_metrics.get('read_operations', 0)}\n")
                f.write(f"- Write ops: {custom_metrics.get('write_operations', 0)}\n\n")
        
        if config_results:
            f.write("## Configuration Serialization Performance\n\n")
            
            for result in config_results:
                custom_metrics = result.avg_metrics.to_dict()['custom']
                
                f.write(f"**{result.description}**\n")
                f.write(f"- Serialize: {custom_metrics.get('serialize_configs_per_sec', 0):.0f} configs/sec\n")
                f.write(f"- Deserialize: {custom_metrics.get('deserialize_configs_per_sec', 0):.0f} configs/sec\n")
                f.write(f"- Avg size: {custom_metrics.get('avg_serialized_size_bytes', 0):.0f} bytes\n")
                f.write(f"- Complexity: {custom_metrics.get('config_complexity', 'unknown')}\n\n")
        
        # Performance recommendations
        f.write("## Performance Recommendations\n\n")
        
        # Analyze metric logging performance
        best_metric_throughput = 0
        best_metric_config = None
        
        for result in metric_results:
            throughput = result.avg_metrics.to_dict()['custom'].get('metrics_per_sec', 0)
            if throughput > best_metric_throughput:
                best_metric_throughput = throughput
                best_metric_config = result
        
        if best_metric_config:
            custom = best_metric_config.avg_metrics.to_dict()['custom']
            f.write(f"- **Optimal metric logging**: {best_metric_throughput:.0f} metrics/sec "
                   f"(batch_size={custom.get('batch_size', 1)}, "
                   f"async={custom.get('async_logging', False)})\n")
        
        # Check if we meet throughput targets
        if best_metric_throughput >= 1000:
            f.write("- ✅ **Metric throughput target met**: >1000 metrics/sec achieved\n")
        else:
            f.write("- ❌ **Metric throughput target missed**: <1000 metrics/sec\n")
        
        # Database performance recommendations
        fast_db_ops = [r for r in database_results 
                      if r.avg_metrics.to_dict()['custom'].get('avg_operation_time_ms', 100) < 10]
        
        if fast_db_ops:
            f.write("- ✅ **Database performance target met**: <10ms average operation time\n")
        else:
            f.write("- ❌ **Database performance target missed**: >10ms average operation time\n")
        
        f.write("\n")
    
    logger.info(f"Performance analysis saved to {analysis_file}")
    return analysis_file


def main():
    """Main function to run tracking benchmarks."""
    parser = argparse.ArgumentParser(description="Run experiment tracking performance benchmarks")
    parser.add_argument("--suite", choices=["baseline", "component", "scalability", "all"],
                       default="baseline", help="Benchmark suite to run")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup iterations")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--name", type=str, default="tracking_performance",
                       help="Name for the benchmark run")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Running {args.suite} benchmark suite...")
    logger.info(f"Iterations: {args.iterations}, Warmup: {args.warmup}")
    logger.info(f"Output directory: {output_dir}")
    
    # Select benchmark cases
    if args.suite == "baseline":
        benchmark_cases = create_baseline_benchmarks()
    elif args.suite == "component":
        benchmark_cases = create_component_benchmarks()
    elif args.suite == "scalability":
        benchmark_cases = create_scalability_benchmarks()
    else:  # all
        benchmark_cases = (create_baseline_benchmarks() + 
                          create_component_benchmarks() + 
                          create_scalability_benchmarks())
    
    logger.info(f"Running {len(benchmark_cases)} benchmark cases...")
    
    # Create and run benchmark suite
    suite = BenchmarkSuite(
        cases=benchmark_cases,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        cooldown_seconds=1.0
    )
    
    runner = BenchmarkRunner(output_dir=str(output_dir))
    
    try:
        results_file = runner.run_suite(suite, name=args.name)
        logger.info(f"Benchmark results saved to {results_file}")
        
        # Load results for analysis
        import json
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Convert back to BenchmarkResult objects for analysis
        from yinsh_ml.benchmarks.benchmark_framework import BenchmarkResult, BenchmarkMetrics
        from datetime import datetime
        
        results = []
        for result_data in results_data['results']:
            # Create mock objects for analysis
            avg_metrics = type('BenchmarkMetrics', (), {})()
            avg_metrics.to_dict = lambda: result_data['avg_metrics']
            
            result = type('BenchmarkResult', (), {})()
            result.name = result_data['name']
            result.description = result_data['description']
            result.avg_metrics = avg_metrics
            result.calculate_statistics = lambda: result_data['statistics']
            results.append(result)
        
        # Generate analysis
        analysis_file = analyze_results(results, output_dir)
        
        logger.info("Benchmark run completed successfully!")
        logger.info(f"Results: {results_file}")
        logger.info(f"Analysis: {analysis_file}")
        
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 