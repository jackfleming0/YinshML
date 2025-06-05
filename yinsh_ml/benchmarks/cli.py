"""
Command-line interface for running YinshML benchmarks.

This module provides a simple CLI for executing various benchmark suites
and generating reports.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .benchmark_framework import BenchmarkRunner
from .scenarios import StandardScenarios, ScalingScenarios, StressTestScenarios, ComparisonScenarios
from .reporters import HTMLReporter, JSONReporter, CSVReporter

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_benchmark_suite(suite_name: str, iterations: int):
    """Get a benchmark suite by name."""
    suite_map = {
        # Standard scenarios
        'memory_pool': lambda: StandardScenarios.create_memory_pool_suite(iterations),
        'training_pipeline': lambda: StandardScenarios.create_training_pipeline_suite(iterations),
        'quick_validation': lambda: StandardScenarios.create_quick_validation_suite(),
        'comprehensive': lambda: StandardScenarios.create_comprehensive_suite(iterations),
        
        # Scaling scenarios
        'pool_size_scaling': lambda: ScalingScenarios.create_pool_size_scaling_suite(),
        'allocation_scaling': lambda: ScalingScenarios.create_allocation_scaling_suite(),
        'mcts_scaling': lambda: ScalingScenarios.create_mcts_scaling_suite(),
        
        # Stress test scenarios
        'memory_pressure': lambda: StressTestScenarios.create_memory_pressure_suite(),
        'endurance': lambda: StressTestScenarios.create_endurance_suite(),
        'concurrency_stress': lambda: StressTestScenarios.create_concurrency_stress_suite(),
        
        # Comparison scenarios
        'pooled_vs_unpooled': lambda: ComparisonScenarios.create_pooled_vs_unpooled_suite(),
        'configuration_comparison': lambda: ComparisonScenarios.create_configuration_comparison_suite(),
    }
    
    if suite_name not in suite_map:
        available = ', '.join(suite_map.keys())
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")
    
    return suite_map[suite_name]()


def generate_reports(results, output_dir: str, formats: List[str]) -> None:
    """Generate reports in specified formats."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for format_name in formats:
        if format_name == 'html':
            reporter = HTMLReporter()
            filepath = os.path.join(output_dir, f'benchmark_report_{timestamp}.html')
            reporter.export(results, filepath)
            print(f"HTML report generated: {filepath}")
            
        elif format_name == 'json':
            reporter = JSONReporter()
            filepath = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
            reporter.export(results, filepath)
            print(f"JSON results saved: {filepath}")
            
        elif format_name == 'csv':
            reporter = CSVReporter()
            filepath = os.path.join(output_dir, f'benchmark_data_{timestamp}.csv')
            reporter.export(results, filepath)
            print(f"CSV data exported: {filepath}")
            
        else:
            logger.warning(f"Unknown report format: {format_name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='YinshML Memory Management Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick validation benchmarks
  python -m yinsh_ml.benchmarks.cli --suite quick_validation
  
  # Run memory pool benchmarks with 100 iterations
  python -m yinsh_ml.benchmarks.cli --suite memory_pool --iterations 100
  
  # Run comprehensive suite and generate all report formats
  python -m yinsh_ml.benchmarks.cli --suite comprehensive --output ./reports --formats html json csv
  
  # Compare pooled vs unpooled performance
  python -m yinsh_ml.benchmarks.cli --suite pooled_vs_unpooled --iterations 50
  
Available benchmark suites:
  Standard Scenarios:
    - memory_pool: Test memory pool allocation patterns
    - training_pipeline: Test training components with memory management
    - quick_validation: Fast validation suite for CI/CD
    - comprehensive: Complete benchmark suite
    
  Scaling Scenarios:
    - pool_size_scaling: Test different pool sizes
    - allocation_scaling: Test different allocation counts
    - mcts_scaling: Test MCTS simulation scaling
    
  Stress Test Scenarios:
    - memory_pressure: Apply memory pressure to test limits
    - endurance: Long-running endurance tests
    - concurrency_stress: Test concurrent access patterns
    
  Comparison Scenarios:
    - pooled_vs_unpooled: Compare memory management on/off
    - configuration_comparison: Compare different configurations
        """
    )
    
    parser.add_argument(
        '--suite', '-s',
        type=str,
        required=True,
        help='Benchmark suite to run'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=20,
        help='Number of iterations per benchmark (default: 20)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./benchmark_results',
        help='Output directory for reports (default: ./benchmark_results)'
    )
    
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['html', 'json', 'csv'],
        default=['html'],
        help='Report formats to generate (default: html)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel execution (experimental)'
    )
    
    parser.add_argument(
        '--warmup-iterations',
        type=int,
        help='Override warmup iterations for all benchmarks'
    )
    
    parser.add_argument(
        '--cooldown-seconds',
        type=float,
        help='Override cooldown seconds for all benchmarks'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Skip cleanup after benchmarks (for debugging)'
    )
    
    parser.add_argument(
        '--list-suites',
        action='store_true',
        help='List available benchmark suites and exit'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # List suites and exit if requested
    if args.list_suites:
        print("Available benchmark suites:")
        suite_descriptions = {
            'memory_pool': 'Test memory pool allocation patterns',
            'training_pipeline': 'Test training components with memory management',
            'quick_validation': 'Fast validation suite for CI/CD',
            'comprehensive': 'Complete benchmark suite',
            'pool_size_scaling': 'Test different pool sizes',
            'allocation_scaling': 'Test different allocation counts',
            'mcts_scaling': 'Test MCTS simulation scaling',
            'memory_pressure': 'Apply memory pressure to test limits',
            'endurance': 'Long-running endurance tests',
            'concurrency_stress': 'Test concurrent access patterns',
            'pooled_vs_unpooled': 'Compare memory management on/off',
            'configuration_comparison': 'Compare different configurations'
        }
        
        for suite_name, description in suite_descriptions.items():
            print(f"  {suite_name}: {description}")
        return 0
    
    try:
        # Get benchmark suite
        print(f"Loading benchmark suite: {args.suite}")
        suite = get_benchmark_suite(args.suite, args.iterations)
        
        # Override suite parameters if specified
        if args.warmup_iterations is not None:
            suite.warmup_iterations = args.warmup_iterations
        if args.cooldown_seconds is not None:
            suite.cooldown_seconds = args.cooldown_seconds
        
        print(f"Benchmark suite loaded with {len(suite.cases)} test cases")
        print(f"Iterations per test: {suite.iterations}")
        print(f"Warmup iterations: {suite.warmup_iterations}")
        print(f"Cooldown time: {suite.cooldown_seconds}s")
        
        # Create benchmark runner
        runner = BenchmarkRunner(output_dir=args.output)
        
        # Run benchmarks
        print("\nStarting benchmark execution...")
        print("=" * 60)
        
        results_file_path = runner.run_suite(suite)
        results = suite.results  # Get results from the suite object
        
        print("=" * 60)
        print(f"Benchmark execution completed!")
        print(f"Total benchmarks run: {len(results)}")
        
        # Generate summary statistics
        if results:
            total_iterations = sum(r.iterations for r in results)
            avg_duration = sum(r.avg_metrics.duration_ns for r in results) / len(results) / 1_000_000
            total_duration = sum(r.avg_metrics.duration_ns * r.iterations for r in results) / 1_000_000_000
            
            print(f"Total iterations executed: {total_iterations:,}")
            print(f"Average benchmark duration: {avg_duration:.2f} ms")
            print(f"Total execution time: {total_duration:.2f} seconds")
        
        # Generate reports
        print(f"\nGenerating reports in formats: {', '.join(args.formats)}")
        generate_reports(results, args.output, args.formats)
        
        print(f"\nAll reports saved to: {os.path.abspath(args.output)}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark execution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 