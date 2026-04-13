"""Comprehensive performance benchmarking for batch evaluation.

This script benchmarks the batch evaluation implementation, measuring:
- Throughput (evaluations per second)
- Memory usage patterns
- Scaling behavior with different batch sizes
- Correctness validation
- Optimal batch size recommendations
"""

import time
import tracemalloc
import statistics
from typing import List, Dict, Tuple
import json
from pathlib import Path

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics import YinshHeuristics


class BatchEvaluationBenchmark:
    """Comprehensive benchmarking suite for batch evaluation."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.evaluator = YinshHeuristics()
        self.results: Dict[str, any] = {}
    
    def generate_test_positions(
        self,
        count: int,
        diverse: bool = True
    ) -> Tuple[List[GameState], List[Player]]:
        """Generate test game positions.
        
        Args:
            count: Number of positions to generate
            diverse: If True, creates diverse positions (different phases/players)
            
        Returns:
            Tuple of (game_states, players) lists
        """
        game_states = []
        players = []
        
        for i in range(count):
            gs = GameState()
            # For diverse positions, alternate players
            player = Player.WHITE if i % 2 == 0 else Player.BLACK
            game_states.append(gs)
            players.append(player)
        
        return game_states, players
    
    def benchmark_throughput(
        self,
        batch_sizes: List[int],
        iterations: int = 100
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark throughput for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per batch size
            
        Returns:
            Dictionary mapping batch_size to performance metrics
        """
        print("\n" + "="*70)
        print("THROUGHPUT BENCHMARKING")
        print("="*70)
        
        results = {}
        max_batch = max(batch_sizes)
        game_states, players = self.generate_test_positions(max_batch, diverse=True)
        
        # Warmup
        print("\nWarming up...")
        for _ in range(10):
            self.evaluator.evaluate_batch(game_states[:10], players[:10])
        
        print(f"\nBenchmarking batch sizes: {batch_sizes}")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Benchmark individual evaluation
            individual_times = []
            for i in range(batch_size):
                for _ in range(iterations):
                    start = time.perf_counter()
                    self.evaluator.evaluate_position(game_states[i], players[i])
                    end = time.perf_counter()
                    individual_times.append((end - start) * 1000.0)  # ms
            
            total_individual_time = sum(individual_times)
            avg_individual_time = statistics.mean(individual_times)
            
            # Benchmark batch evaluation
            batch_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                self.evaluator.evaluate_batch(
                    game_states[:batch_size],
                    players[:batch_size]
                )
                end = time.perf_counter()
                batch_times.append((end - start) * 1000.0)  # ms
            
            avg_batch_time = statistics.mean(batch_times)
            total_batch_time = sum(batch_times)
            
            # Calculate metrics
            evaluations_per_second_batch = (batch_size * iterations) / (total_batch_time / 1000.0)
            evaluations_per_second_individual = (batch_size * iterations) / (total_individual_time / 1000.0)
            speedup = total_individual_time / total_batch_time if total_batch_time > 0 else 0
            time_per_eval_batch = avg_batch_time / batch_size
            time_per_eval_individual = avg_individual_time
            
            results[batch_size] = {
                'avg_batch_time_ms': avg_batch_time,
                'total_batch_time_ms': total_batch_time,
                'avg_individual_time_ms': avg_individual_time,
                'total_individual_time_ms': total_individual_time,
                'evaluations_per_second_batch': evaluations_per_second_batch,
                'evaluations_per_second_individual': evaluations_per_second_individual,
                'speedup': speedup,
                'time_per_eval_batch_ms': time_per_eval_batch,
                'time_per_eval_individual_ms': time_per_eval_individual,
            }
            
            print(f"  Batch avg time: {avg_batch_time:.4f} ms")
            print(f"  Individual total time: {total_individual_time:.4f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Throughput (batch): {evaluations_per_second_batch:.0f} eval/s")
            print(f"  Throughput (individual): {evaluations_per_second_individual:.0f} eval/s")
            print(f"  Time per eval (batch): {time_per_eval_batch:.4f} ms")
            print(f"  Time per eval (individual): {time_per_eval_individual:.4f} ms")
        
        self.results['throughput'] = results
        return results
    
    def benchmark_memory(
        self,
        batch_sizes: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark memory usage for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping batch_size to memory metrics
        """
        print("\n" + "="*70)
        print("MEMORY PROFILING")
        print("="*70)
        
        if not hasattr(tracemalloc, 'start'):
            print("\nWarning: tracemalloc not available, skipping memory profiling")
            return {}
        
        results = {}
        max_batch = max(batch_sizes)
        game_states, players = self.generate_test_positions(max_batch, diverse=False)
        
        print(f"\nProfiling memory for batch sizes: {batch_sizes}")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            tracemalloc.start()
            
            # Evaluate batch
            self.evaluator.evaluate_batch(
                game_states[:batch_size],
                players[:batch_size]
            )
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[batch_size] = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024,
                'current_kb': current / 1024,
                'peak_kb': peak / 1024,
            }
            
            print(f"\nBatch size: {batch_size}")
            print(f"  Current memory: {results[batch_size]['current_mb']:.2f} MB")
            print(f"  Peak memory: {results[batch_size]['peak_mb']:.2f} MB")
            print(f"  Memory per position: {results[batch_size]['peak_kb'] / batch_size:.2f} KB")
        
        # Calculate scaling factor
        if len(results) >= 2:
            sizes = sorted(results.keys())
            first_size = sizes[0]
            last_size = sizes[-1]
            ratio = results[last_size]['peak_mb'] / results[first_size]['peak_mb'] if results[first_size]['peak_mb'] > 0 else float('inf')
            expected_ratio = last_size / first_size
            
            print(f"\nScaling Analysis:")
            print(f"  Size ratio: {last_size}/{first_size} = {last_size/first_size:.1f}x")
            print(f"  Memory ratio: {ratio:.2f}x")
            print(f"  Expected (linear): {expected_ratio:.1f}x")
            print(f"  Scaling efficiency: {(expected_ratio / ratio * 100):.1f}%")
        
        self.results['memory'] = results
        return results
    
    def validate_correctness(
        self,
        batch_sizes: List[int],
        num_tests: int = 100
    ) -> Dict[str, any]:
        """Validate correctness of batch evaluation.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_tests: Number of correctness tests per batch size
            
        Returns:
            Dictionary with correctness validation results
        """
        print("\n" + "="*70)
        print("CORRECTNESS VALIDATION")
        print("="*70)
        
        max_batch = max(batch_sizes)
        all_correct = True
        max_error = 0.0
        total_tests = 0
        errors = []
        
        print(f"\nValidating correctness for batch sizes: {batch_sizes}")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            batch_correct = True
            
            for test_num in range(num_tests):
                game_states, players = self.generate_test_positions(batch_size, diverse=True)
                
                # Get batch results
                batch_results = self.evaluator.evaluate_batch(game_states, players)
                
                # Get individual results
                individual_results = [
                    self.evaluator.evaluate_position(gs, p)
                    for gs, p in zip(game_states, players)
                ]
                
                # Compare results
                for i, (batch_result, individual_result) in enumerate(
                    zip(batch_results, individual_results)
                ):
                    total_tests += 1
                    error = abs(batch_result - individual_result)
                    max_error = max(max_error, error)
                    
                    if error > 1e-10:  # Allow for floating point precision
                        batch_correct = False
                        all_correct = False
                        errors.append({
                            'batch_size': batch_size,
                            'test_num': test_num,
                            'position': i,
                            'batch_result': batch_result,
                            'individual_result': individual_result,
                            'error': error
                        })
            
            status = "✅ PASS" if batch_correct else "❌ FAIL"
            print(f"  {status} ({num_tests} tests)")
        
        results = {
            'all_correct': all_correct,
            'max_error': max_error,
            'total_tests': total_tests,
            'num_errors': len(errors),
            'errors': errors[:10]  # Store first 10 errors for debugging
        }
        
        print(f"\nSummary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Errors: {len(errors)}")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Status: {'✅ PASS' if all_correct else '❌ FAIL'}")
        
        self.results['correctness'] = results
        return results
    
    def analyze_optimal_batch_size(self) -> Dict[str, any]:
        """Analyze results to recommend optimal batch size.
        
        Returns:
            Dictionary with recommendations
        """
        if 'throughput' not in self.results:
            return {}
        
        throughput = self.results['throughput']
        
        # Find batch size with best throughput
        best_throughput = 0
        best_batch_size = 1
        
        for batch_size, metrics in throughput.items():
            if metrics['evaluations_per_second_batch'] > best_throughput:
                best_throughput = metrics['evaluations_per_second_batch']
                best_batch_size = batch_size
        
        # Find batch size with best speedup
        best_speedup = 0
        best_speedup_batch_size = 1
        
        for batch_size, metrics in throughput.items():
            if metrics['speedup'] > best_speedup:
                best_speedup = metrics['speedup']
                best_speedup_batch_size = batch_size
        
        # Find sweet spot (good balance of speedup and efficiency)
        # Prefer batch sizes with speedup > 1.0 and reasonable batch size
        sweet_spot_candidates = [
            (bs, m['speedup']) for bs, m in throughput.items()
            if m['speedup'] > 1.0 and bs >= 10
        ]
        
        if sweet_spot_candidates:
            sweet_spot_candidates.sort(key=lambda x: x[1], reverse=True)
            sweet_spot_batch_size = sweet_spot_candidates[0][0]
        else:
            sweet_spot_batch_size = best_batch_size
        
        recommendations = {
            'best_throughput_batch_size': best_batch_size,
            'best_throughput': best_throughput,
            'best_speedup_batch_size': best_speedup_batch_size,
            'best_speedup': best_speedup,
            'recommended_batch_size': sweet_spot_batch_size,
            'recommendations': {
                'small_batches': 'Use batch sizes 10-50 for interactive applications',
                'medium_batches': f'Use batch size {sweet_spot_batch_size} for balanced performance',
                'large_batches': 'Use batch sizes 100-500 for batch processing workloads',
                'very_large_batches': 'Use batch sizes 500-1000 for maximum throughput',
            }
        }
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive performance report.
        
        Args:
            output_file: Optional path to save report as JSON
            
        Returns:
            Report as formatted string
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("BATCH EVALUATION PERFORMANCE REPORT")
        report_lines.append("="*70)
        
        # Throughput summary
        if 'throughput' in self.results:
            report_lines.append("\nTHROUGHPUT SUMMARY")
            report_lines.append("-"*70)
            throughput = self.results['throughput']
            for batch_size in sorted(throughput.keys()):
                m = throughput[batch_size]
                report_lines.append(
                    f"Batch {batch_size:4d}: {m['speedup']:5.2f}x speedup, "
                    f"{m['evaluations_per_second_batch']:8.0f} eval/s, "
                    f"{m['time_per_eval_batch_ms']:6.4f} ms/eval"
                )
        
        # Memory summary
        if 'memory' in self.results:
            report_lines.append("\nMEMORY SUMMARY")
            report_lines.append("-"*70)
            memory = self.results['memory']
            for batch_size in sorted(memory.keys()):
                m = memory[batch_size]
                report_lines.append(
                    f"Batch {batch_size:4d}: {m['peak_mb']:6.2f} MB peak, "
                    f"{m['peak_kb']/batch_size:6.2f} KB/position"
                )
        
        # Correctness summary
        if 'correctness' in self.results:
            report_lines.append("\nCORRECTNESS SUMMARY")
            report_lines.append("-"*70)
            correctness = self.results['correctness']
            report_lines.append(f"Status: {'✅ PASS' if correctness['all_correct'] else '❌ FAIL'}")
            report_lines.append(f"Total tests: {correctness['total_tests']}")
            report_lines.append(f"Errors: {correctness['num_errors']}")
            report_lines.append(f"Max error: {correctness['max_error']:.2e}")
        
        # Recommendations
        if 'recommendations' in self.results:
            report_lines.append("\nRECOMMENDATIONS")
            report_lines.append("-"*70)
            rec = self.results['recommendations']
            report_lines.append(f"Best throughput: Batch size {rec['best_throughput_batch_size']} ({rec['best_throughput']:.0f} eval/s)")
            report_lines.append(f"Best speedup: Batch size {rec['best_speedup_batch_size']} ({rec['best_speedup']:.2f}x)")
            report_lines.append(f"Recommended: Batch size {rec['recommended_batch_size']}")
            report_lines.append("\nUse Cases:")
            for use_case, recommendation in rec['recommendations'].items():
                report_lines.append(f"  - {use_case.replace('_', ' ').title()}: {recommendation}")
        
        report_lines.append("\n" + "="*70)
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            
            # Also save JSON data
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nReport saved to: {output_file}")
            print(f"Data saved to: {json_file}")
        
        return report


def main():
    """Run comprehensive benchmarks."""
    print("Starting Batch Evaluation Performance Benchmarks...")
    print("This may take a few minutes...")
    
    benchmark = BatchEvaluationBenchmark()
    
    # Define batch sizes to test
    batch_sizes = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    # Run benchmarks
    benchmark.benchmark_throughput(batch_sizes, iterations=50)
    benchmark.benchmark_memory(batch_sizes)
    benchmark.validate_correctness(batch_sizes, num_tests=50)
    benchmark.analyze_optimal_batch_size()
    
    # Generate report
    report = benchmark.generate_report('yinsh_ml/heuristics/benchmark_report.txt')
    print("\n" + report)
    
    print("\n✅ Benchmarking complete!")


if __name__ == '__main__':
    main()

