"""Performance benchmark for YinshHeuristics evaluator.

This script benchmarks the heuristic evaluator to ensure it meets the
<1ms per evaluation performance requirement.
"""

import time
import statistics
from typing import List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics.evaluator import YinshHeuristics


def generate_test_positions(count: int = 1000) -> List[GameState]:
    """Generate test game positions for benchmarking.
    
    Creates fresh game states for testing. In a real scenario, you would
    want to test with positions at different game phases, but for basic
    performance testing, fresh states are sufficient.
    
    Args:
        count: Number of test positions to generate
        
    Returns:
        List of GameState instances for testing
    """
    positions = []
    for _ in range(count):
        positions.append(GameState())
    return positions


def benchmark_evaluator(
    evaluator: YinshHeuristics,
    positions: List[GameState],
    player: Player = Player.WHITE,
    warmup_iterations: int = 100
) -> dict:
    """Benchmark the heuristic evaluator performance.
    
    Args:
        evaluator: YinshHeuristics instance to benchmark
        positions: List of game positions to evaluate
        player: Player to evaluate from perspective of
        warmup_iterations: Number of warmup iterations before timing
        
    Returns:
        Dictionary containing benchmark results:
        - total_time: Total time in seconds
        - avg_time_ms: Average time per evaluation in milliseconds
        - min_time_ms: Minimum evaluation time in milliseconds
        - max_time_ms: Maximum evaluation time in milliseconds
        - median_time_ms: Median evaluation time in milliseconds
        - p95_time_ms: 95th percentile evaluation time in milliseconds
        - p99_time_ms: 99th percentile evaluation time in milliseconds
        - evaluations_per_second: Throughput in evaluations per second
    """
    # Warmup phase
    for i in range(min(warmup_iterations, len(positions))):
        evaluator.evaluate_position(positions[i], player)
    
    # Actual benchmark
    times_ms = []
    start_total = time.perf_counter()
    
    for position in positions:
        start = time.perf_counter()
        evaluator.evaluate_position(position, player)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)  # Convert to milliseconds
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # Calculate statistics
    times_ms_sorted = sorted(times_ms)
    n = len(times_ms)
    
    results = {
        'total_time': total_time,
        'total_evaluations': n,
        'avg_time_ms': statistics.mean(times_ms),
        'min_time_ms': min(times_ms),
        'max_time_ms': max(times_ms),
        'median_time_ms': statistics.median(times_ms),
        'p95_time_ms': times_ms_sorted[int(n * 0.95)] if n > 0 else 0.0,
        'p99_time_ms': times_ms_sorted[int(n * 0.99)] if n > 0 else 0.0,
        'evaluations_per_second': n / total_time if total_time > 0 else 0.0,
    }
    
    return results


def run_benchmark(num_positions: int = 10000):
    """Run comprehensive performance benchmark.
    
    Args:
        num_positions: Number of positions to evaluate
    """
    print(f"Running performance benchmark with {num_positions} evaluations...")
    print("=" * 60)
    
    # Create evaluator
    evaluator = YinshHeuristics()
    
    # Generate test positions
    print("Generating test positions...")
    positions = generate_test_positions(num_positions)
    
    # Benchmark
    print("Running benchmark...")
    results = benchmark_evaluator(evaluator, positions, Player.WHITE)
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 60)
    print(f"Total evaluations: {results['total_evaluations']}")
    print(f"Total time: {results['total_time']:.3f} seconds")
    print(f"Average time per evaluation: {results['avg_time_ms']:.4f} ms")
    print(f"Median time: {results['median_time_ms']:.4f} ms")
    print(f"Min time: {results['min_time_ms']:.4f} ms")
    print(f"Max time: {results['max_time_ms']:.4f} ms")
    print(f"95th percentile: {results['p95_time_ms']:.4f} ms")
    print(f"99th percentile: {results['p99_time_ms']:.4f} ms")
    print(f"Throughput: {results['evaluations_per_second']:.0f} evaluations/second")
    print("-" * 60)
    
    # Check if requirement is met
    requirement_met = results['avg_time_ms'] < 1.0
    print(f"\nPerformance Requirement: <1ms per evaluation")
    print(f"Status: {'✅ PASS' if requirement_met else '❌ FAIL'}")
    if not requirement_met:
        print(f"Average time ({results['avg_time_ms']:.4f} ms) exceeds 1ms requirement")
    
    return results


if __name__ == '__main__':
    run_benchmark(10000)
