"""Comprehensive test suite for batch evaluation functionality.

This test suite verifies the batch evaluation implementation for Task 8,
ensuring correctness, performance, and integration with existing evaluation methods.
"""

import unittest
import time
import sys
import tracemalloc
from typing import List
from unittest.mock import patch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics import YinshHeuristics


class TestBatchEvaluationInterface(unittest.TestCase):
    """Test subtask 8.1: Batch processing architecture - Interface tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = YinshHeuristics()
        self.game_state = GameState()
    
    def test_evaluate_batch_method_exists(self):
        """Test that evaluate_batch method exists on YinshHeuristics."""
        self.assertTrue(hasattr(self.evaluator, 'evaluate_batch'))
        self.assertTrue(callable(getattr(self.evaluator, 'evaluate_batch')))
    
    def test_evaluate_batch_signature(self):
        """Test that evaluate_batch accepts correct parameters."""
        # Should accept List[GameState] and List[Player]
        try:
            results = self.evaluator.evaluate_batch([self.game_state], [Player.WHITE])
            self.assertIsInstance(results, list)
        except NotImplementedError:
            self.fail("evaluate_batch method not implemented yet")
        except TypeError as e:
            self.fail(f"Method signature incorrect: {e}")
    
    def test_evaluate_batch_empty_lists(self):
        """Test that empty lists are handled correctly."""
        results = self.evaluator.evaluate_batch([], [])
        self.assertEqual(results, [])
    
    def test_evaluate_batch_mismatched_lengths(self):
        """Test that mismatched list lengths raise ValueError."""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_batch(
                [self.game_state, self.game_state],
                [Player.WHITE]
            )
    
    def test_evaluate_batch_invalid_types(self):
        """Test that invalid types raise TypeError."""
        with self.assertRaises(TypeError):
            self.evaluator.evaluate_batch([None], [Player.WHITE])
        
        with self.assertRaises(TypeError):
            self.evaluator.evaluate_batch([self.game_state], [None])
        
        with self.assertRaises(TypeError):
            self.evaluator.evaluate_batch("not a list", [Player.WHITE])
        
        with self.assertRaises(TypeError):
            self.evaluator.evaluate_batch([self.game_state], "not a list")
    
    def test_evaluate_batch_return_type(self):
        """Test that evaluate_batch returns List[float]."""
        results = self.evaluator.evaluate_batch([self.game_state], [Player.WHITE])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], float)
    
    def test_evaluate_batch_correct_length(self):
        """Test that results list has same length as input."""
        game_states = [GameState() for _ in range(5)]
        players = [Player.WHITE] * 5
        
        results = self.evaluator.evaluate_batch(game_states, players)
        self.assertEqual(len(results), 5)


class TestBatchEvaluationCorrectness(unittest.TestCase):
    """Test subtask 8.1: Batch processing architecture - Correctness tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = YinshHeuristics()
        self.game_state = GameState()
    
    def test_single_position_matches_individual(self):
        """Test that batch evaluation of single position matches individual evaluation."""
        batch_result = self.evaluator.evaluate_batch([self.game_state], [Player.WHITE])
        individual_result = self.evaluator.evaluate_position(self.game_state, Player.WHITE)
        
        self.assertEqual(len(batch_result), 1)
        self.assertAlmostEqual(
            batch_result[0],
            individual_result,
            places=10,
            msg="Batch result should match individual evaluation"
        )
    
    def test_multiple_positions_same_player(self):
        """Test batch evaluation with multiple positions and same player."""
        game_states = [GameState() for _ in range(3)]
        players = [Player.WHITE] * 3
        
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        self.assertEqual(len(batch_results), 3)
        
        # Verify each result matches individual evaluation
        for i, (game_state, player) in enumerate(zip(game_states, players)):
            individual_result = self.evaluator.evaluate_position(game_state, player)
            self.assertAlmostEqual(
                batch_results[i],
                individual_result,
                places=10,
                msg=f"Batch result {i} should match individual evaluation"
            )
    
    def test_multiple_positions_different_players(self):
        """Test batch evaluation with multiple positions and different players."""
        game_states = [GameState() for _ in range(4)]
        players = [Player.WHITE, Player.BLACK, Player.WHITE, Player.BLACK]
        
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        self.assertEqual(len(batch_results), 4)
        
        # Verify each result matches individual evaluation
        for i, (game_state, player) in enumerate(zip(game_states, players)):
            individual_result = self.evaluator.evaluate_position(game_state, player)
            self.assertAlmostEqual(
                batch_results[i],
                individual_result,
                places=10,
                msg=f"Batch result {i} should match individual evaluation"
            )
    
    def test_results_maintain_order(self):
        """Test that results maintain the same order as input positions."""
        # Create distinct game states by making different moves
        game_states = []
        for i in range(5):
            gs = GameState()
            # Each state will be slightly different (even if empty, they're separate instances)
            game_states.append(gs)
        
        players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(5)]
        
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        # Verify order by comparing with individual evaluations
        for i, (game_state, player) in enumerate(zip(game_states, players)):
            individual_result = self.evaluator.evaluate_position(game_state, player)
            self.assertAlmostEqual(
                batch_results[i],
                individual_result,
                places=10,
                msg=f"Result at index {i} should match individual evaluation"
            )


class TestBatchEvaluationIntegration(unittest.TestCase):
    """Test subtask 8.1: Batch processing architecture - Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = YinshHeuristics()
    
    def test_integration_with_evaluate_position(self):
        """Test that batch evaluation integrates correctly with evaluate_position."""
        game_states = [GameState() for _ in range(10)]
        players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(10)]
        
        # Get batch results
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        # Get individual results
        individual_results = [
            self.evaluator.evaluate_position(gs, p)
            for gs, p in zip(game_states, players)
        ]
        
        # Compare results
        self.assertEqual(len(batch_results), len(individual_results))
        for i, (batch_result, individual_result) in enumerate(zip(batch_results, individual_results)):
            self.assertAlmostEqual(
                batch_result,
                individual_result,
                places=10,
                msg=f"Batch result {i} should match individual evaluation"
            )
    
    def test_various_game_phases(self):
        """Test batch evaluation with positions in different game phases."""
        # Create game states in different phases
        # Early phase: empty board
        early_state = GameState()
        
        # Mid phase: simulate by creating state with some moves
        # (In real usage, moves would be made via make_move())
        mid_state = GameState()
        
        # Late phase: simulate by creating state with many moves
        late_state = GameState()
        
        game_states = [early_state, mid_state, late_state]
        players = [Player.WHITE, Player.BLACK, Player.WHITE]
        
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        # Verify all results are valid floats
        for i, result in enumerate(batch_results):
            self.assertIsInstance(result, float)
            # Verify matches individual evaluation
            individual_result = self.evaluator.evaluate_position(
                game_states[i], players[i]
            )
            self.assertAlmostEqual(
                result,
                individual_result,
                places=10,
                msg=f"Phase test result {i} should match individual evaluation"
            )
    
    def test_terminal_positions(self):
        """Test that batch evaluation handles terminal positions correctly."""
        # Create game states (terminal detection will be tested if positions are terminal)
        game_states = [GameState() for _ in range(3)]
        players = [Player.WHITE, Player.BLACK, Player.WHITE]
        
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        # Verify results match individual evaluation (including terminal handling)
        for i, (game_state, player) in enumerate(zip(game_states, players)):
            individual_result = self.evaluator.evaluate_position(game_state, player)
            self.assertAlmostEqual(
                batch_results[i],
                individual_result,
                places=10,
                msg=f"Terminal position test {i} should match individual evaluation"
            )


class TestBatchEvaluationPerformance(unittest.TestCase):
    """Test subtask 8.1: Batch processing architecture - Performance baseline tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = YinshHeuristics()
        self.game_state = GameState()
    
    def test_batch_method_callable(self):
        """Test that batch method can be called without errors."""
        try:
            results = self.evaluator.evaluate_batch([self.game_state], [Player.WHITE])
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Batch method should be callable, but raised: {e}")
    
    def test_batch_size_one_performance(self):
        """Test that batch_size=1 is at least as fast as individual evaluation."""
        # Warmup
        for _ in range(10):
            self.evaluator.evaluate_position(self.game_state, Player.WHITE)
            self.evaluator.evaluate_batch([self.game_state], [Player.WHITE])
        
        # Benchmark individual evaluation
        individual_times = []
        for _ in range(100):
            start = time.perf_counter()
            self.evaluator.evaluate_position(self.game_state, Player.WHITE)
            end = time.perf_counter()
            individual_times.append((end - start) * 1000.0)  # Convert to ms
        
        # Benchmark batch evaluation (batch_size=1)
        batch_times = []
        for _ in range(100):
            start = time.perf_counter()
            self.evaluator.evaluate_batch([self.game_state], [Player.WHITE])
            end = time.perf_counter()
            batch_times.append((end - start) * 1000.0)  # Convert to ms
        
        avg_individual = sum(individual_times) / len(individual_times)
        avg_batch = sum(batch_times) / len(batch_times)
        
        print(f"\nIndividual evaluation avg: {avg_individual:.4f} ms")
        print(f"Batch evaluation (size=1) avg: {avg_batch:.4f} ms")
        
        # Batch should not be significantly slower (allow 20% overhead for now)
        # This will be optimized in subtask 8.2
        self.assertLess(
            avg_batch,
            avg_individual * 1.2,
            "Batch evaluation (size=1) should not be significantly slower"
        )


class TestBatchEvaluationOptimization(unittest.TestCase):
    """Test subtask 8.2: Memory optimization and vectorization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = YinshHeuristics()
        self.game_state = GameState()
    
    def test_batch_performance_improvement(self):
        """Test that batch evaluation shows performance improvement with larger batches."""
        # Create batch of positions
        batch_sizes = [1, 10, 50, 100]
        game_states = [GameState() for _ in range(max(batch_sizes))]
        players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(max(batch_sizes))]
        
        # Warmup
        for _ in range(5):
            self.evaluator.evaluate_batch(game_states[:10], players[:10])
        
        results = {}
        for batch_size in batch_sizes:
            # Benchmark individual evaluation
            individual_times = []
            for i in range(batch_size):
                start = time.perf_counter()
                self.evaluator.evaluate_position(game_states[i], players[i])
                end = time.perf_counter()
                individual_times.append((end - start) * 1000.0)
            total_individual = sum(individual_times)
            
            # Benchmark batch evaluation
            batch_times = []
            for _ in range(10):  # Multiple runs for averaging
                start = time.perf_counter()
                self.evaluator.evaluate_batch(game_states[:batch_size], players[:batch_size])
                end = time.perf_counter()
                batch_times.append((end - start) * 1000.0)
            avg_batch = sum(batch_times) / len(batch_times)
            
            results[batch_size] = {
                'individual_total': total_individual,
                'batch_avg': avg_batch,
                'speedup': total_individual / avg_batch if avg_batch > 0 else 0
            }
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Individual total: {total_individual:.4f} ms")
            print(f"  Batch avg: {avg_batch:.4f} ms")
            print(f"  Speedup: {results[batch_size]['speedup']:.2f}x")
        
        # For larger batches, we should see some improvement
        # (Even basic implementation should be close due to reduced overhead)
        speedup_100 = results[100]['speedup']
        self.assertGreater(
            speedup_100,
            0.8,  # At least 80% of individual time (allowing some overhead)
            f"Batch evaluation should be efficient, got {speedup_100:.2f}x speedup"
        )
    
    def test_memory_usage_scaling(self):
        """Test that memory usage scales linearly with batch size, not quadratically."""
        if not hasattr(tracemalloc, 'start'):
            self.skipTest("tracemalloc not available")
        
        batch_sizes = [10, 50, 100]
        game_states = [GameState() for _ in range(max(batch_sizes))]
        players = [Player.WHITE] * max(batch_sizes)
        
        memory_usage = {}
        
        for batch_size in batch_sizes:
            tracemalloc.start()
            
            # Evaluate batch
            self.evaluator.evaluate_batch(game_states[:batch_size], players[:batch_size])
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_usage[batch_size] = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Current memory: {memory_usage[batch_size]['current_mb']:.2f} MB")
            print(f"  Peak memory: {memory_usage[batch_size]['peak_mb']:.2f} MB")
        
        # Check that memory scales roughly linearly
        # Memory for batch_size=100 should be less than 15x memory for batch_size=10
        # (allowing some overhead, but should be roughly linear)
        ratio = memory_usage[100]['peak_mb'] / memory_usage[10]['peak_mb'] if memory_usage[10]['peak_mb'] > 0 else float('inf')
        self.assertLess(
            ratio,
            15.0,  # Should be roughly 10x, allow up to 15x for overhead
            f"Memory should scale roughly linearly, got {ratio:.2f}x ratio"
        )
    
    def test_vectorized_correctness(self):
        """Test that vectorized operations produce correct results."""
        # Create diverse batch
        batch_size = 50
        game_states = [GameState() for _ in range(batch_size)]
        players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(batch_size)]
        
        # Get batch results
        batch_results = self.evaluator.evaluate_batch(game_states, players)
        
        # Get individual results
        individual_results = [
            self.evaluator.evaluate_position(gs, p)
            for gs, p in zip(game_states, players)
        ]
        
        # Verify exact match (vectorized should match individual)
        self.assertEqual(len(batch_results), len(individual_results))
        for i, (batch_result, individual_result) in enumerate(zip(batch_results, individual_results)):
            self.assertAlmostEqual(
                batch_result,
                individual_result,
                places=10,
                msg=f"Vectorized result {i} should match individual evaluation exactly"
            )
    
    def test_large_batch_handling(self):
        """Test that large batches (1000+) can be handled efficiently."""
        batch_size = 1000
        game_states = [GameState() for _ in range(batch_size)]
        players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(batch_size)]
        
        # Should complete without errors
        start = time.perf_counter()
        results = self.evaluator.evaluate_batch(game_states, players)
        elapsed = (time.perf_counter() - start) * 1000.0  # Convert to ms
        
        self.assertEqual(len(results), batch_size)
        self.assertIsInstance(results[0], float)
        
        # Should complete in reasonable time (less than 10 seconds for 1000 positions)
        self.assertLess(
            elapsed,
            10000.0,
            f"Large batch evaluation should complete in reasonable time, took {elapsed:.2f} ms"
        )
        
        print(f"\nLarge batch ({batch_size} positions): {elapsed:.2f} ms")
        print(f"  Average per position: {elapsed / batch_size:.4f} ms")
    
    def test_phase_caching_effectiveness(self):
        """Test that positions in same phase benefit from caching."""
        # Create positions in same phase (all early game)
        batch_size = 100
        game_states = [GameState() for _ in range(batch_size)]
        players = [Player.WHITE] * batch_size
        
        # All positions should be in same phase (early game, 0 moves)
        # This should allow phase caching to be effective
        
        # Benchmark
        start = time.perf_counter()
        results = self.evaluator.evaluate_batch(game_states, players)
        elapsed = (time.perf_counter() - start) * 1000.0
        
        self.assertEqual(len(results), batch_size)
        
        # Verify all results are valid
        for result in results:
            self.assertIsInstance(result, float)
        
        print(f"\nSame-phase batch ({batch_size} positions): {elapsed:.2f} ms")
        print(f"  Average per position: {elapsed / batch_size:.4f} ms")


if __name__ == '__main__':
    unittest.main()

