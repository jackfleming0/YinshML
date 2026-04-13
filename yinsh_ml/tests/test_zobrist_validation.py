"""Comprehensive statistical validation tests for Zobrist hashing."""

import random
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
import pytest
from scipy import stats

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import PieceType, Position, Player, VALID_POSITIONS
from yinsh_ml.game.zobrist import ZobristHasher, ZobristInitializer


def generate_random_board_state(
    num_pieces: int = None,
    seed: int = None,
) -> Board:
    """Generate a random valid board state."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    board = Board()
    if num_pieces is None:
        # Random number of pieces between 5 and 30
        num_pieces = random.randint(5, 30)

    # Collect all valid positions
    all_positions = []
    for col in sorted(VALID_POSITIONS.keys()):
        for row in sorted(VALID_POSITIONS[col]):
            all_positions.append(Position(col, row))

    # Randomly select positions and assign random pieces
    selected_positions = random.sample(all_positions, min(num_pieces, len(all_positions)))
    piece_types = [
        PieceType.WHITE_RING,
        PieceType.BLACK_RING,
        PieceType.WHITE_MARKER,
        PieceType.BLACK_MARKER,
    ]

    for pos in selected_positions:
        piece = random.choice(piece_types)
        board.place_piece(pos, piece)

    return board


def generate_board_variations(
    base_board: Board,
    num_variations: int = 100,
    seed: int = None,
) -> List[Board]:
    """Generate small variations of a base board for avalanche testing."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    variations = []
    all_positions = []
    for col in sorted(VALID_POSITIONS.keys()):
        for row in sorted(VALID_POSITIONS[col]):
            all_positions.append(Position(col, row))

    for _ in range(num_variations):
        variation = base_board.copy()
        # Make a small change: add, remove, or modify one piece
        pos = random.choice(all_positions)
        current_piece = variation.get_piece(pos)

        if current_piece is None:
            # Add a random piece
            piece_types = [
                PieceType.WHITE_RING,
                PieceType.BLACK_RING,
                PieceType.WHITE_MARKER,
                PieceType.BLACK_MARKER,
            ]
            variation.place_piece(pos, random.choice(piece_types))
        elif current_piece.is_marker():
            # Flip the marker or remove it
            if random.random() < 0.5:
                # Flip
                new_marker = (
                    PieceType.WHITE_MARKER if current_piece == PieceType.BLACK_MARKER
                    else PieceType.BLACK_MARKER
                )
                variation.remove_piece(pos)
                variation.place_piece(pos, new_marker)
            else:
                # Remove
                variation.remove_piece(pos)
        else:
            # Remove ring
            variation.remove_piece(pos)

        variations.append(variation)

    return variations


def analyze_hash_distribution(hashes: List[int], num_buckets: int = 256) -> Tuple[float, float]:
    """
    Perform chi-square test for uniform distribution.

    Returns:
        (chi2_statistic, p_value)
    """
    # Map hashes to buckets
    bucket_counts = np.zeros(num_buckets, dtype=int)
    for h in hashes:
        bucket = h % num_buckets
        bucket_counts[bucket] += 1

    # Expected count per bucket
    expected = len(hashes) / num_buckets

    # Chi-square test
    chi2, p_value = stats.chisquare(bucket_counts, f_exp=expected)

    return float(chi2), float(p_value)


def measure_collision_rate(hashes: List[int]) -> Tuple[int, float]:
    """
    Measure collision rate in hash set.

    Returns:
        (num_collisions, collision_rate)
    """
    unique_hashes = len(set(hashes))
    total_hashes = len(hashes)
    collisions = total_hashes - unique_hashes
    collision_rate = collisions / total_hashes if total_hashes > 0 else 0.0

    return collisions, collision_rate


def hamming_distance(a: int, b: int) -> int:
    """Calculate Hamming distance (number of differing bits) between two integers."""
    return bin(a ^ b).count("1")


def measure_avalanche_effect(
    base_hash: int,
    variation_hashes: List[int],
) -> Tuple[float, float, float]:
    """
    Measure avalanche effect: small board changes should cause large hash changes.

    Returns:
        (mean_hamming_distance, min_hamming_distance, max_hamming_distance)
    """
    distances = [hamming_distance(base_hash, var_hash) for var_hash in variation_hashes]
    return (
        float(np.mean(distances)),
        float(np.min(distances)),
        float(np.max(distances)),
    )


def bit_independence_test(hashes: List[int]) -> Tuple[float, float]:
    """
    Test bit independence: each bit should be independent.

    Returns:
        (mean_bit_ones_ratio, std_bit_ones_ratio)
    """
    # Convert hashes to binary arrays
    bits = np.array([[int(b) for b in format(h, "064b")] for h in hashes], dtype=int)

    # Calculate ratio of ones for each bit position
    bit_ones_ratios = np.mean(bits, axis=0)

    return float(np.mean(bit_ones_ratios)), float(np.std(bit_ones_ratios))


@pytest.mark.slow
def test_hash_distribution_uniformity():
    """Test that hash distribution is uniform using chi-square test."""
    initializer = ZobristInitializer(seed="distribution-test")
    hasher = ZobristHasher(initializer.table)

    # Generate 10,000 random board states
    num_samples = 10_000
    hashes = []
    for i in range(num_samples):
        board = generate_random_board_state(seed=i)
        hash_val = hasher.hash_board(board)
        hashes.append(hash_val)

    # Test with 256 buckets
    chi2, p_value = analyze_hash_distribution(hashes, num_buckets=256)

    # p-value should be > 0.05 (not significantly different from uniform)
    # Chi-square should be reasonable (not too high)
    assert p_value > 0.05, f"Hash distribution not uniform (p={p_value:.6f}, chi2={chi2:.2f})"
    assert chi2 < 300, f"Chi-square too high: {chi2:.2f} (expected < 300 for uniform distribution)"


@pytest.mark.slow
def test_collision_rate_large_sample():
    """Test collision rate over a large sample of random board states."""
    initializer = ZobristInitializer(seed="collision-test")
    hasher = ZobristHasher(initializer.table)

    # Generate 100,000 random board states
    num_samples = 100_000
    hashes = []
    for i in range(num_samples):
        board = generate_random_board_state(seed=i)
        hash_val = hasher.hash_board(board)
        hashes.append(hash_val)

    collisions, collision_rate = measure_collision_rate(hashes)

    # For 64-bit hashes and 100K samples, collision rate should be very low
    # Expected collisions ≈ n²/(2*2^64) ≈ 0.0000000005 for n=100K
    # In practice, we expect near-zero collisions
    assert collision_rate < 0.001, f"Collision rate too high: {collision_rate:.6f} ({collisions} collisions)"
    print(f"\nCollision rate: {collision_rate:.6f} ({collisions} collisions out of {num_samples})")


def test_avalanche_effect():
    """Test that small board changes cause large hash changes (avalanche effect)."""
    initializer = ZobristInitializer(seed="avalanche-test")
    hasher = ZobristHasher(initializer.table)

    # Create base board
    base_board = generate_random_board_state(seed=42, num_pieces=15)
    base_hash = hasher.hash_board(base_board)

    # Generate 100 variations
    variations = generate_board_variations(base_board, num_variations=100, seed=123)
    variation_hashes = [hasher.hash_board(var) for var in variations]

    mean_dist, min_dist, max_dist = measure_avalanche_effect(base_hash, variation_hashes)

    # For good avalanche effect:
    # - Mean Hamming distance should be around 32 (half of 64 bits)
    # - Min distance should be > 0 (no identical hashes for different boards)
    # - Max distance should be reasonably high (for single piece changes, ~40+ is good)
    assert min_dist > 0, "Some variations produced identical hashes (bad avalanche effect)"
    assert mean_dist > 25, f"Mean Hamming distance too low: {mean_dist:.2f} (expected ~32)"
    assert mean_dist < 40, f"Mean Hamming distance too high: {mean_dist:.2f} (expected ~32)"
    assert max_dist > 35, f"Max Hamming distance too low: {max_dist} (expected > 35 for good avalanche)"

    print(f"\nAvalanche effect: mean={mean_dist:.2f}, min={min_dist}, max={max_dist}")


def test_bit_independence():
    """Test that hash bits are independent."""
    initializer = ZobristInitializer(seed="bit-independence")
    hasher = ZobristHasher(initializer.table)

    # Generate 5,000 random board states
    num_samples = 5_000
    hashes = []
    for i in range(num_samples):
        board = generate_random_board_state(seed=i)
        hash_val = hasher.hash_board(board)
        hashes.append(hash_val)

    mean_ratio, std_ratio = bit_independence_test(hashes)

    # Each bit should be 1 approximately 50% of the time
    # Mean should be close to 0.5, std should be small
    assert 0.45 < mean_ratio < 0.55, f"Mean bit ratio not ~0.5: {mean_ratio:.4f}"
    assert std_ratio < 0.1, f"Bit ratio std too high: {std_ratio:.4f} (bits not independent)"

    print(f"\nBit independence: mean_ratio={mean_ratio:.4f}, std={std_ratio:.4f}")


def test_incremental_vs_full_performance():
    """Benchmark incremental updates vs full recomputation."""
    initializer = ZobristInitializer(seed="perf-test")
    hasher = ZobristHasher(initializer.table)

    # Create a board with several pieces
    board = generate_random_board_state(seed=100, num_pieces=20)
    initial_hash = hasher.hash_board(board)

    # Measure incremental update time
    num_updates = 1000
    start_time = time.perf_counter()
    current_hash = initial_hash
    for _ in range(num_updates):
        # Make a random update
        all_positions = []
        for col in sorted(VALID_POSITIONS.keys()):
            for row in sorted(VALID_POSITIONS[col]):
                all_positions.append(Position(col, row))
        pos = random.choice(all_positions)
        player = random.choice([Player.WHITE, Player.BLACK])
        current_hash = hasher.place_marker(pos, player, current_hash)
    incremental_time = time.perf_counter() - start_time

    # Measure full recomputation time
    start_time = time.perf_counter()
    for _ in range(num_updates):
        board = generate_random_board_state(seed=100 + _, num_pieces=20)
        hasher.hash_board(board)
    full_time = time.perf_counter() - start_time

    speedup = full_time / incremental_time if incremental_time > 0 else float("inf")

    # Incremental should be significantly faster
    assert speedup > 2.0, f"Incremental updates not faster: speedup={speedup:.2f}x"
    print(f"\nPerformance: incremental={incremental_time:.4f}s, full={full_time:.4f}s, speedup={speedup:.2f}x")


def test_hash_consistency_across_calls():
    """Test that hashing the same board multiple times produces identical results."""
    initializer = ZobristInitializer(seed="consistency")
    hasher = ZobristHasher(initializer.table)

    board = generate_random_board_state(seed=999, num_pieces=15)

    # Hash the same board multiple times
    hashes = [hasher.hash_board(board) for _ in range(100)]

    # All hashes should be identical
    assert len(set(hashes)) == 1, "Hash values not consistent across multiple calls"
    print(f"\nConsistency: All 100 calls produced hash {hashes[0]}")


@pytest.mark.slow
def test_collision_rate_different_seeds():
    """Test that different seeds produce different hash distributions."""
    hasher1 = ZobristHasher(seed="seed1")
    hasher2 = ZobristHasher(seed="seed2")

    # Generate same board states
    num_samples = 10_000
    hashes1 = []
    hashes2 = []
    for i in range(num_samples):
        board = generate_random_board_state(seed=i, num_pieces=10)
        hashes1.append(hasher1.hash_board(board))
        hashes2.append(hasher2.hash_board(board))

    # Hashes should be completely different (different seeds = different tables)
    overlap = len(set(hashes1) & set(hashes2))
    overlap_rate = overlap / num_samples

    # With different seeds, overlap should be minimal (just by chance)
    assert overlap_rate < 0.01, f"Too much overlap between different seeds: {overlap_rate:.4f}"
    print(f"\nSeed independence: {overlap} overlapping hashes out of {num_samples} ({overlap_rate:.4f})")


def test_deterministic_with_same_seed():
    """Test that same seed produces deterministic results."""
    hasher1 = ZobristHasher(seed="deterministic")
    hasher2 = ZobristHasher(seed="deterministic")

    # Generate same board states
    num_samples = 1000
    for i in range(num_samples):
        board = generate_random_board_state(seed=i, num_pieces=10)
        hash1 = hasher1.hash_board(board)
        hash2 = hasher2.hash_board(board)
        assert hash1 == hash2, f"Hashes differ for same seed at sample {i}"

    print(f"\nDeterminism: All {num_samples} samples produced identical hashes with same seed")

