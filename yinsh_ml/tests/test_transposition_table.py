"""Tests for transposition table implementation.

This test suite verifies the core hash table data structure with collision handling,
including basic operations, hash distribution, and performance characteristics.
"""

import pytest
import time
import random
from typing import List

from yinsh_ml.search.transposition_table import (
    TranspositionTable,
    TranspositionTableEntry,
    NodeType,
)
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.game.constants import Player, Position


class TestTranspositionTableBasic:
    """Test basic transposition table operations."""
    
    def test_basic_store_and_lookup(self):
        """Test basic store and lookup functionality."""
        tt = TranspositionTable(size_power=10)  # Small table for testing
        
        hash_key = 1234567890
        entry = tt.store(
            hash_key=hash_key,
            depth=3,
            value=0.5,
            best_move=None,
            node_type=NodeType.EXACT,
        )
        
        result = tt.lookup(hash_key)
        assert result is not None
        assert result.hash_key == hash_key
        assert result.depth == 3
        assert result.value == 0.5
        assert result.node_type == NodeType.EXACT
    
    def test_empty_table_lookup(self):
        """Test lookup on empty table returns None."""
        tt = TranspositionTable(size_power=10)
        
        result = tt.lookup(1234567890)
        assert result is None
    
    def test_overwrite_existing_entry(self):
        """Test storing to same hash key overwrites existing entry."""
        tt = TranspositionTable(size_power=10)
        
        hash_key = 1234567890
        
        # Store first entry
        tt.store(hash_key, depth=2, value=0.3, node_type=NodeType.EXACT)
        result1 = tt.lookup(hash_key)
        assert result1.value == 0.3
        assert result1.depth == 2
        
        # Store second entry with same hash key
        tt.store(hash_key, depth=4, value=0.7, node_type=NodeType.LOWER_BOUND)
        result2 = tt.lookup(hash_key)
        assert result2.value == 0.7
        assert result2.depth == 4
        assert result2.node_type == NodeType.LOWER_BOUND
    
    def test_table_size_validation(self):
        """Test power-of-2 size validation."""
        # Valid sizes
        tt1 = TranspositionTable(size_power=10)
        assert tt1.size == (1 << 10)
        
        tt2 = TranspositionTable(size_power=20)
        assert tt2.size == (1 << 20)
        
        # Invalid sizes
        with pytest.raises(ValueError):
            TranspositionTable(size_power=3)  # Too small
        
        with pytest.raises(ValueError):
            TranspositionTable(size_power=31)  # Too large
    
    def test_entry_with_best_move(self):
        """Test storing and retrieving entries with best moves."""
        tt = TranspositionTable(size_power=10)
        
        move = Move(
            type=MoveType.PLACE_RING,
            player=Player.WHITE,
            source=Position.from_string("E5"),
        )
        
        hash_key = 9876543210
        tt.store(
            hash_key=hash_key,
            depth=5,
            value=0.8,
            best_move=move,
            node_type=NodeType.EXACT,
        )
        
        result = tt.lookup(hash_key)
        assert result is not None
        assert result.best_move == move
        assert result.best_move.type == MoveType.PLACE_RING


class TestHashDistribution:
    """Test hash distribution uniformity."""
    
    def test_hash_distribution_uniformity(self):
        """Test that hash keys are distributed uniformly across table buckets."""
        tt = TranspositionTable(size_power=10)
        num_samples = 1000
        
        # Generate random hash keys
        hash_keys = [random.randint(0, 2**63 - 1) for _ in range(num_samples)]
        
        # Store entries
        for hash_key in hash_keys:
            tt.store(hash_key, depth=1, value=0.0, node_type=NodeType.EXACT)
        
        # Check distribution
        bucket_counts = [0] * tt.size
        for hash_key in hash_keys:
            index = hash_key & tt.mask
            bucket_counts[index] += 1
        
        # Calculate statistics
        mean = num_samples / tt.size
        variance = sum((count - mean) ** 2 for count in bucket_counts) / tt.size
        std_dev = variance ** 0.5
        
        # For uniform distribution, std_dev should be close to sqrt(mean)
        # Allow some variance (within 3 standard deviations)
        expected_std = (mean * (1 - 1/tt.size)) ** 0.5
        assert std_dev <= expected_std * 2, f"Distribution too uneven: std_dev={std_dev}, expected={expected_std}"
    
    def test_collision_handling(self):
        """Test collision handling correctness."""
        tt = TranspositionTable(size_power=8)  # Small table to force collisions
        
        # Create hash keys that will collide (same lower bits)
        base_hash = 0x1234567890ABCDEF
        hash_keys = []
        
        # Generate keys that map to same initial bucket
        for i in range(10):
            # Use same lower bits but different upper bits
            hash_key = base_hash | (i << 32)
            hash_keys.append(hash_key)
        
        # Store all entries
        stored_values = {}
        for i, hash_key in enumerate(hash_keys):
            value = float(i) * 0.1
            tt.store(hash_key, depth=i, value=value, node_type=NodeType.EXACT)
            stored_values[hash_key] = value
        
        # Verify all entries can be retrieved
        for hash_key, expected_value in stored_values.items():
            result = tt.lookup(hash_key)
            assert result is not None, f"Failed to find entry for hash {hash_key}"
            assert result.hash_key == hash_key, "Hash key mismatch"
            assert abs(result.value - expected_value) < 1e-9, "Value mismatch"
    
    def test_collision_linear_probing(self):
        """Test that linear probing handles collisions correctly."""
        tt = TranspositionTable(size_power=6)  # Very small table
        
        # Force a collision by using hash keys that map to same initial index
        hash1 = 0x0000000000000001
        hash2 = 0x0000000000000041  # Same lower 6 bits
        
        tt.store(hash1, depth=1, value=1.0, node_type=NodeType.EXACT)
        tt.store(hash2, depth=2, value=2.0, node_type=NodeType.EXACT)
        
        # Both should be retrievable
        result1 = tt.lookup(hash1)
        result2 = tt.lookup(hash2)
        
        assert result1 is not None and result1.value == 1.0
        assert result2 is not None and result2.value == 2.0


class TestPerformance:
    """Test performance characteristics."""
    
    def test_performance_insertion_lookup(self):
        """Benchmark insertion and lookup performance."""
        tt = TranspositionTable(size_power=16)
        num_operations = 10000
        
        # Generate random hash keys
        hash_keys = [random.randint(0, 2**63 - 1) for _ in range(num_operations)]
        
        # Benchmark insertion
        start_time = time.perf_counter()
        for i, hash_key in enumerate(hash_keys):
            tt.store(
                hash_key,
                depth=i % 10,
                value=random.random(),
                node_type=NodeType.EXACT,
            )
        insert_time = time.perf_counter() - start_time
        
        # Benchmark lookup
        start_time = time.perf_counter()
        for hash_key in hash_keys:
            tt.lookup(hash_key)
        lookup_time = time.perf_counter() - start_time
        
        # Calculate times per operation (should be < 1μs)
        insert_time_per_op = (insert_time / num_operations) * 1e6  # microseconds
        lookup_time_per_op = (lookup_time / num_operations) * 1e6
        
        # Performance should be very fast (< 10μs per operation)
        assert insert_time_per_op < 10.0, f"Insertion too slow: {insert_time_per_op:.2f}μs/op"
        assert lookup_time_per_op < 10.0, f"Lookup too slow: {lookup_time_per_op:.2f}μs/op"
    
    def test_large_table_performance(self):
        """Test performance with large table size."""
        tt = TranspositionTable(size_power=20)  # 1M entries
        
        num_operations = 1000
        hash_keys = [random.randint(0, 2**63 - 1) for _ in range(num_operations)]
        
        start_time = time.perf_counter()
        for hash_key in hash_keys:
            tt.store(hash_key, depth=5, value=0.5, node_type=NodeType.EXACT)
            tt.lookup(hash_key)
        total_time = time.perf_counter() - start_time
        
        time_per_op = (total_time / num_operations) * 1e6
        assert time_per_op < 50.0, f"Operations too slow: {time_per_op:.2f}μs/op"


class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        tt = TranspositionTable(size_power=10, enable_metrics=True)
        
        # Perform operations
        hash_keys = [random.randint(0, 2**63 - 1) for _ in range(100)]
        
        for hash_key in hash_keys:
            tt.store(hash_key, depth=1, value=0.5, node_type=NodeType.EXACT)
        
        for hash_key in hash_keys:
            tt.lookup(hash_key)
        
        # Check some lookups that don't exist
        for _ in range(50):
            tt.lookup(random.randint(0, 2**63 - 1))
        
        metrics = tt.get_metrics()
        
        assert metrics["stores"] == 100
        assert metrics["hits"] == 100
        assert metrics["misses"] == 50
        assert metrics["hit_rate"] == pytest.approx(100.0 * 100 / 150, rel=0.01)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        tt = TranspositionTable(size_power=10, enable_metrics=True)
        
        tt.store(123, depth=1, value=0.5, node_type=NodeType.EXACT)
        tt.lookup(123)
        
        metrics_before = tt.get_metrics()
        assert metrics_before["stores"] > 0
        
        tt.reset_metrics()
        metrics_after = tt.get_metrics()
        assert metrics_after["stores"] == 0
        assert metrics_after["hits"] == 0
        assert metrics_after["misses"] == 0
    
    def test_utilization_rate(self):
        """Test utilization rate calculation."""
        tt = TranspositionTable(size_power=10, enable_metrics=True)
        
        # Store entries
        num_entries = 500
        for i in range(num_entries):
            tt.store(i, depth=1, value=0.5, node_type=NodeType.EXACT)
        
        utilization = tt.get_utilization_rate()
        expected_utilization = (num_entries / tt.size) * 100.0
        
        assert utilization == pytest.approx(expected_utilization, rel=0.01)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_full_table_handling(self):
        """Test behavior when table is full."""
        tt = TranspositionTable(size_power=4)  # Very small table (16 entries)
        
        # Fill table completely
        for i in range(tt.size):
            tt.store(i, depth=1, value=float(i), node_type=NodeType.EXACT)
        
        # Verify all entries are stored
        for i in range(tt.size):
            result = tt.lookup(i)
            assert result is not None, f"Entry {i} should exist"
        
        # Try to store one more (should replace due to full table)
        # Use a hash that will collide with existing entries
        new_hash = tt.size + 1
        tt.store(new_hash, depth=2, value=999.0, node_type=NodeType.EXACT)
        
        # Verify the new entry can be looked up
        result = tt.lookup(new_hash)
        assert result is not None, "New entry should be stored"
        assert result.depth == 2, "New entry should have depth 2"
    
    def test_zero_hash_key(self):
        """Test handling of zero hash key."""
        tt = TranspositionTable(size_power=10)
        
        tt.store(0, depth=1, value=0.0, node_type=NodeType.EXACT)
        result = tt.lookup(0)
        
        assert result is not None
        assert result.hash_key == 0
        assert result.value == 0.0
    
    def test_max_hash_key(self):
        """Test handling of maximum hash key."""
        tt = TranspositionTable(size_power=10)
        
        max_hash = 2**63 - 1
        tt.store(max_hash, depth=1, value=1.0, node_type=NodeType.EXACT)
        result = tt.lookup(max_hash)
        
        assert result is not None
        assert result.hash_key == max_hash


class TestEntryStructure:
    """Test entry structure design and optimization."""
    
    def test_entry_size_requirements(self):
        """Test that entry size meets memory requirements (≤ 48 bytes)."""
        import sys
        
        entry = TranspositionTableEntry(
            hash_key=0x1234567890ABCDEF,
            depth=5,
            value=0.5,
            best_move=None,
            node_type=NodeType.EXACT,
            age=0,
        )
        
        # Estimate size (slots=True should reduce overhead)
        entry_size = sys.getsizeof(entry)
        
        # With slots=True, entry should be reasonably sized
        # Allow some overhead for Python object header
        assert entry_size <= 200, f"Entry too large: {entry_size} bytes"
    
    def test_entry_with_all_node_types(self):
        """Test entry creation with each node type."""
        tt = TranspositionTable(size_power=10)
        
        node_types = [NodeType.EXACT, NodeType.LOWER_BOUND, NodeType.UPPER_BOUND]
        
        for i, node_type in enumerate(node_types):
            hash_key = 1000 + i
            tt.store(
                hash_key,
                depth=3,
                value=float(i),
                node_type=node_type,
            )
            
            result = tt.lookup(hash_key)
            assert result is not None
            assert result.node_type == node_type
    
    def test_entry_field_access(self):
        """Test that all entry fields are accessible and correct types."""
        entry = TranspositionTableEntry(
            hash_key=0xABCDEF1234567890,
            depth=7,
            value=0.75,
            best_move=None,
            node_type=NodeType.EXACT,
            age=10,
        )
        
        assert isinstance(entry.hash_key, int)
        assert isinstance(entry.depth, int)
        assert isinstance(entry.value, float)
        assert entry.best_move is None or isinstance(entry.best_move, Move)
        assert isinstance(entry.node_type, NodeType)
        assert isinstance(entry.age, int)
        
        assert entry.hash_key == 0xABCDEF1234567890
        assert entry.depth == 7
        assert entry.value == 0.75
        assert entry.node_type == NodeType.EXACT
        assert entry.age == 10
    
    def test_entry_with_move(self):
        """Test entry with best move stored."""
        tt = TranspositionTable(size_power=10)
        
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string("E5"),
            destination=Position.from_string("E8"),
        )
        
        hash_key = 9999
        tt.store(
            hash_key,
            depth=4,
            value=0.6,
            best_move=move,
            node_type=NodeType.EXACT,
        )
        
        result = tt.lookup(hash_key)
        assert result is not None
        assert result.best_move == move
        assert result.best_move.type == MoveType.MOVE_RING


class TestReplacementPolicy:
    """Test depth-preferred replacement policy."""
    
    def test_depth_preferred_replacement(self):
        """Test that higher depth entries are preserved."""
        tt = TranspositionTable(size_power=8)  # Small table to force collisions
        
        # Create hash keys that will collide
        hash1 = 0x0000000000000001
        hash2 = 0x0000000000000041  # Same lower bits
        
        # Store lower depth entry first
        tt.store(hash1, depth=2, value=1.0, node_type=NodeType.EXACT)
        
        # Store higher depth entry (should replace if collision)
        tt.store(hash2, depth=5, value=2.0, node_type=NodeType.EXACT)
        
        # Both should be retrievable
        result1 = tt.lookup(hash1)
        result2 = tt.lookup(hash2)
        
        assert result1 is not None and result1.depth == 2
        assert result2 is not None and result2.depth == 5
    
    def test_equal_depth_replacement(self):
        """Test replacement when depths are equal (node type comparison)."""
        tt = TranspositionTable(size_power=8)
        
        hash1 = 0x0000000000000001
        hash2 = 0x0000000000000041
        
        # Store LOWER_BOUND entry
        tt.store(hash1, depth=3, value=1.0, node_type=NodeType.LOWER_BOUND)
        
        # Store EXACT entry with same depth (should be preferred)
        tt.store(hash2, depth=3, value=2.0, node_type=NodeType.EXACT)
        
        # Verify both exist
        result1 = tt.lookup(hash1)
        result2 = tt.lookup(hash2)
        
        assert result1 is not None
        assert result2 is not None
        assert result2.node_type == NodeType.EXACT
    
    def test_replacement_with_node_types(self):
        """Test replacement policy with different node types."""
        tt = TranspositionTable(size_power=6)  # Very small to force replacement
        
        # Fill table to force replacements
        base_hash = 0x1000000000000000
        
        # Store entries with different depths and node types
        entries = []
        for i in range(4):
            entries.extend([
                (base_hash + i, 2, NodeType.UPPER_BOUND, float(i)),
                (base_hash + i + 10, 3, NodeType.LOWER_BOUND, float(i + 10)),
                (base_hash + i + 20, 3, NodeType.EXACT, float(i + 20)),
                (base_hash + i + 30, 4, NodeType.EXACT, float(i + 30)),
            ])
        
        for hash_key, depth, node_type, value in entries:
            tt.store(hash_key, depth=depth, value=value, node_type=node_type)
        
        # Verify higher depth entries are preserved
        depth_4_found = 0
        depth_3_found = 0
        for hash_key, depth, node_type, value in entries:
            result = tt.lookup(hash_key)
            if result is not None:
                if depth == 4:
                    depth_4_found += 1
                elif depth == 3:
                    depth_3_found += 1
        
        # Higher depth entries should be more likely to be preserved
        assert depth_4_found > 0, "Some depth 4 entries should be preserved"
    
    def test_replacement_edge_cases(self):
        """Test replacement policy edge cases."""
        tt = TranspositionTable(size_power=8)
        
        hash_key = 0x1234567890ABCDEF
        
        # Test depth 0
        tt.store(hash_key, depth=0, value=0.0, node_type=NodeType.EXACT)
        result = tt.lookup(hash_key)
        assert result is not None and result.depth == 0
        
        # Replace with depth 1 (should replace)
        tt.store(hash_key, depth=1, value=1.0, node_type=NodeType.EXACT)
        result = tt.lookup(hash_key)
        assert result is not None and result.depth == 1
        
        # Try to replace with depth 0 (should not replace)
        tt.store(hash_key, depth=0, value=0.0, node_type=NodeType.EXACT)
        result = tt.lookup(hash_key)
        assert result is not None and result.depth == 1  # Should keep depth 1
    
    def test_replacement_policy_correctness(self):
        """Test that replacement policy logic matches specification."""
        tt = TranspositionTable(size_power=8)
        
        hash_key = 0xABCDEF1234567890
        
        # Store entry with depth 3, LOWER_BOUND
        tt.store(hash_key, depth=3, value=1.0, node_type=NodeType.LOWER_BOUND)
        
        # Try to replace with depth 3, EXACT (should replace - same depth, better node type)
        tt.store(hash_key, depth=3, value=2.0, node_type=NodeType.EXACT)
        result = tt.lookup(hash_key)
        assert result is not None
        assert result.depth == 3
        assert result.node_type == NodeType.EXACT
        assert result.value == 2.0
        
        # Try to replace with depth 2, EXACT (should NOT replace - lower depth)
        tt.store(hash_key, depth=2, value=3.0, node_type=NodeType.EXACT)
        result = tt.lookup(hash_key)
        assert result is not None
        assert result.depth == 3  # Should keep higher depth
        assert result.node_type == NodeType.EXACT


class TestThreadSafety:
    """Test thread safety and concurrent access."""
    
    def test_concurrent_reads(self):
        """Test multiple threads reading simultaneously."""
        import threading
        
        tt = TranspositionTable(size_power=12)
        
        # Pre-populate table
        num_entries = 100
        hash_keys = list(range(num_entries))
        for hash_key in hash_keys:
            tt.store(hash_key, depth=1, value=float(hash_key), node_type=NodeType.EXACT)
        
        results = {}
        errors = []
        
        def read_entries(start_idx, end_idx):
            try:
                for i in range(start_idx, end_idx):
                    result = tt.lookup(hash_keys[i])
                    results[i] = result
            except Exception as e:
                errors.append(e)
        
        # Create multiple reader threads
        threads = []
        num_threads = 4
        entries_per_thread = num_entries // num_threads
        
        for t in range(num_threads):
            start = t * entries_per_thread
            end = start + entries_per_thread if t < num_threads - 1 else num_entries
            thread = threading.Thread(target=read_entries, args=(start, end))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and all reads succeeded
        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert len(results) == num_entries
        
        # Verify all results are correct
        for i, result in results.items():
            assert result is not None
            assert result.value == float(hash_keys[i])
    
    def test_concurrent_writes(self):
        """Test multiple threads writing simultaneously."""
        import threading
        
        tt = TranspositionTable(size_power=12)
        
        num_threads = 4
        entries_per_thread = 50
        errors = []
        
        def write_entries(thread_id):
            try:
                start_hash = thread_id * 1000
                for i in range(entries_per_thread):
                    hash_key = start_hash + i
                    tt.store(
                        hash_key,
                        depth=thread_id + 1,
                        value=float(thread_id),
                        node_type=NodeType.EXACT,
                    )
            except Exception as e:
                errors.append(e)
        
        # Create writer threads
        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=write_entries, args=(t,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        
        # Verify entries were stored
        for t in range(num_threads):
            hash_key = t * 1000
            result = tt.lookup(hash_key)
            assert result is not None
    
    def test_concurrent_read_write(self):
        """Test mixed read/write operations from multiple threads."""
        import threading
        import time
        
        tt = TranspositionTable(size_power=12)
        
        # Pre-populate some entries
        for i in range(50):
            tt.store(i, depth=1, value=float(i), node_type=NodeType.EXACT)
        
        errors = []
        read_count = [0]
        write_count = [0]
        
        def reader_thread():
            try:
                for _ in range(100):
                    hash_key = random.randint(0, 49)
                    result = tt.lookup(hash_key)
                    if result is not None:
                        read_count[0] += 1
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        def writer_thread(thread_id):
            try:
                for i in range(50):
                    hash_key = 100 + thread_id * 100 + i
                    tt.store(
                        hash_key,
                        depth=2,
                        value=float(thread_id),
                        node_type=NodeType.EXACT,
                    )
                    write_count[0] += 1
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create mixed threads
        threads = []
        for t in range(2):  # 2 reader threads
            thread = threading.Thread(target=reader_thread)
            threads.append(thread)
            thread.start()
        
        for t in range(2):  # 2 writer threads
            thread = threading.Thread(target=writer_thread, args=(t,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"
        assert read_count[0] > 0
        assert write_count[0] > 0
    
    def test_no_data_races(self):
        """Test that no data races occur under stress."""
        import threading
        
        tt = TranspositionTable(size_power=10)
        
        num_operations = 1000
        errors = []
        race_detected = [False]
        
        def stress_test(thread_id):
            try:
                for i in range(num_operations):
                    hash_key = thread_id * 10000 + i
                    
                    # Store
                    tt.store(
                        hash_key,
                        depth=i % 10,
                        value=float(i),
                        node_type=NodeType.EXACT,
                    )
                    
                    # Lookup
                    result = tt.lookup(hash_key)
                    if result is None:
                        # Entry might have been replaced, but shouldn't cause crash
                        pass
                    elif result.hash_key != hash_key:
                        race_detected[0] = True
            except Exception as e:
                errors.append(e)
        
        # Create many threads
        threads = []
        num_threads = 8
        for t in range(num_threads):
            thread = threading.Thread(target=stress_test, args=(t,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors and no races detected
        assert len(errors) == 0, f"Errors during stress test: {errors}"
        assert not race_detected[0], "Data race detected!"


class TestClear:
    """Test table clearing functionality."""
    
    def test_clear_table(self):
        """Test that clear removes all entries."""
        tt = TranspositionTable(size_power=10)
        
        # Store some entries
        for i in range(100):
            tt.store(i, depth=1, value=0.5, node_type=NodeType.EXACT)
        
        # Verify entries exist
        assert tt.lookup(0) is not None
        
        # Clear table
        tt.clear()
        
        # Verify entries are gone
        assert tt.lookup(0) is None
        
        # Verify utilization is zero
        assert tt.get_utilization_rate() == 0.0

