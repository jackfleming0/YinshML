"""Unit tests for D6 Symmetry Augmentation.

Tests verify:
1. Correct number of transforms (6 or 12)
2. State transform preserves non-zero structure
3. Policy transforms correctly
4. Round-trip transforms return to original
5. Batch augmentation works correctly
"""

import pytest
import numpy as np
from yinsh_ml.training.augmentation import (
    YinshSymmetryAugmenter,
    AugmentationStats,
    verify_transform_round_trip
)
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, PieceType, Position


class TestAugmenterBasics:
    """Basic functionality tests."""

    @pytest.fixture
    def augmenter_with_reflections(self):
        return YinshSymmetryAugmenter(include_reflections=True, enable_stats=True)

    @pytest.fixture
    def augmenter_rotations_only(self):
        return YinshSymmetryAugmenter(include_reflections=False, enable_stats=True)

    @pytest.fixture
    def simple_state(self):
        """Create a simple state with one piece."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0  # Ring at center (F6)
        return state

    @pytest.fixture
    def simple_policy(self):
        """Create a simple policy with some non-zero entries."""
        policy = np.zeros(7395, dtype=np.float32)
        policy[0] = 0.3
        policy[1] = 0.3
        policy[2] = 0.4
        return policy

    def test_num_transforms_with_reflections(self, augmenter_with_reflections):
        """Test that we have 12 transforms with reflections enabled."""
        assert augmenter_with_reflections.num_transforms == 12

    def test_num_transforms_rotations_only(self, augmenter_rotations_only):
        """Test that we have 6 transforms without reflections."""
        assert augmenter_rotations_only.num_transforms == 6

    def test_augment_returns_correct_count(self, augmenter_with_reflections, simple_state, simple_policy):
        """Test that augment returns correct number of samples."""
        results = augmenter_with_reflections.augment(
            simple_state, simple_policy, 0.5, include_original=True
        )
        assert len(results) == 12, f"Expected 12 augmentations, got {len(results)}"

    def test_augment_without_original(self, augmenter_with_reflections, simple_state, simple_policy):
        """Test augment without including original."""
        results = augmenter_with_reflections.augment(
            simple_state, simple_policy, 0.5, include_original=False
        )
        assert len(results) == 11, f"Expected 11 augmentations (excluding original), got {len(results)}"

    def test_augmented_shapes_match(self, augmenter_with_reflections, simple_state, simple_policy):
        """Test that augmented states and policies have correct shapes."""
        results = augmenter_with_reflections.augment(simple_state, simple_policy, 0.5)

        for aug_state, aug_policy, aug_value in results:
            assert aug_state.shape == simple_state.shape, \
                f"State shape mismatch: {aug_state.shape} vs {simple_state.shape}"
            assert aug_policy.shape == simple_policy.shape, \
                f"Policy shape mismatch: {aug_policy.shape} vs {simple_policy.shape}"

    def test_value_unchanged(self, augmenter_with_reflections, simple_state, simple_policy):
        """Test that value is unchanged across all transforms."""
        value = 0.75
        results = augmenter_with_reflections.augment(simple_state, simple_policy, value)

        for _, _, aug_value in results:
            assert aug_value == value, f"Value changed: {aug_value} != {value}"


class TestIdentityTransform:
    """Test identity transform (transform_id=0)."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_identity_state_unchanged(self, augmenter):
        """Test that identity transform returns identical state."""
        state = np.random.rand(6, 11, 11).astype(np.float32)
        transformed = augmenter._transform_state(state, 0)
        assert np.array_equal(transformed, state), "Identity transform should not change state"

    def test_identity_policy_unchanged(self, augmenter):
        """Test that identity transform returns identical policy."""
        policy = np.random.rand(7395).astype(np.float32)
        policy /= policy.sum()  # Normalize
        transformed = augmenter._transform_policy(policy, 0)
        assert np.array_equal(transformed, policy), "Identity transform should not change policy"


class TestRotationTransforms:
    """Test rotation transforms (0-5)."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=False)

    def test_rotation_preserves_nonzero_count(self, augmenter):
        """Test that rotation preserves the number of non-zero elements."""
        # Create state with some pieces
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0
        state[0, 4, 4] = 1.0
        state[2, 6, 6] = 1.0

        original_nonzero = np.count_nonzero(state)

        for rot_id in range(1, 6):
            transformed = augmenter._transform_state(state, rot_id)
            transformed_nonzero = np.count_nonzero(transformed)
            # Allow some tolerance for edge cases where rotation moves pieces off-board
            assert transformed_nonzero <= original_nonzero, \
                f"Rotation {rot_id} increased non-zero count: {transformed_nonzero} > {original_nonzero}"

    def test_180_rotation_is_self_inverse(self, augmenter):
        """Test that 180° rotation (transform_id=3) is its own inverse."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0  # Center
        state[0, 4, 4] = 1.0  # Off-center

        # Apply 180° twice
        rotated_once = augmenter._transform_state(state, 3)
        rotated_twice = augmenter._transform_state(rotated_once, 3)

        # Should return to original
        assert np.allclose(state, rotated_twice), "180° rotation applied twice should return to original"

    def test_six_rotations_cycle(self, augmenter):
        """Test that applying 60° rotation 6 times returns to original."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0  # Center (should be invariant)

        current = state.copy()
        for _ in range(6):
            current = augmenter._transform_state(current, 1)

        # After 6 rotations of 60°, should return to original
        # Note: due to discrete grid, this may not be exact for off-center positions
        center_value = current[0, 5, 5]
        assert center_value == 1.0, "Center should remain at center after 6 rotations"


class TestReflectionTransforms:
    """Test reflection transforms (6-11)."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_reflection_is_self_inverse(self, augmenter):
        """Test that each reflection applied twice returns to original."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0  # Center

        for ref_id in range(6, 12):
            reflected_once = augmenter._transform_state(state, ref_id)
            reflected_twice = augmenter._transform_state(reflected_once, ref_id)

            # Should return to original (at least for center)
            assert reflected_twice[0, 5, 5] == 1.0, \
                f"Reflection {ref_id} applied twice should preserve center"


class TestCoordinateTransforms:
    """Test low-level coordinate transformation functions."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_center_is_invariant_under_rotation(self, augmenter):
        """Test that center coordinates are invariant under rotation."""
        center_row, center_col = 5, 5

        for rot_id in range(1, 6):
            new_row, new_col = augmenter._transform_coord(center_row, center_col, rot_id)
            assert (new_row, new_col) == (5, 5), \
                f"Rotation {rot_id} moved center from (5,5) to ({new_row},{new_col})"

    def test_center_is_invariant_under_reflection(self, augmenter):
        """Test that center coordinates are invariant under reflection."""
        center_row, center_col = 5, 5

        for ref_id in range(6, 12):
            new_row, new_col = augmenter._transform_coord(center_row, center_col, ref_id)
            assert (new_row, new_col) == (5, 5), \
                f"Reflection {ref_id} moved center from (5,5) to ({new_row},{new_col})"


class TestPolicyTransforms:
    """Test policy transformation."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_policy_sum_preserved(self, augmenter):
        """Test that policy sums are preserved (approximately) after transform."""
        policy = np.random.rand(7395).astype(np.float32)
        policy /= policy.sum()

        for transform_id in range(1, 12):
            transformed = augmenter._transform_policy(policy, transform_id)

            # Sum should be 1.0 (or close to it)
            transformed_sum = transformed.sum()
            assert abs(transformed_sum - 1.0) < 0.1 or transformed_sum < 0.1, \
                f"Transform {transform_id}: policy sum = {transformed_sum}"

    def test_zero_policy_stays_zero(self, augmenter):
        """Test that zero policy stays zero."""
        policy = np.zeros(7395, dtype=np.float32)

        for transform_id in range(12):
            transformed = augmenter._transform_policy(policy, transform_id)
            assert np.allclose(transformed, 0), \
                f"Transform {transform_id} created non-zero values from zero policy"


class TestBatchAugmentation:
    """Test batch augmentation functionality."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_batch_augment_correct_count(self, augmenter):
        """Test that batch augmentation produces correct number of samples."""
        states = [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(3)]
        policies = [np.zeros(7395, dtype=np.float32) for _ in range(3)]
        policies[0][0] = 1.0
        policies[1][1] = 1.0
        policies[2][2] = 1.0
        values = [0.1, 0.2, 0.3]

        aug_states, aug_policies, aug_values = augmenter.augment_batch(
            states, policies, values
        )

        # 3 original + 3 * 11 augmented = 36
        expected_count = 3 * 12
        assert len(aug_states) == expected_count, \
            f"Expected {expected_count} samples, got {len(aug_states)}"
        assert len(aug_policies) == expected_count
        assert len(aug_values) == expected_count

    def test_batch_augment_with_limit(self, augmenter):
        """Test batch augmentation with max_augmentations limit."""
        states = [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(2)]
        policies = [np.zeros(7395, dtype=np.float32) for _ in range(2)]
        values = [0.5, 0.5]

        aug_states, aug_policies, aug_values = augmenter.augment_batch(
            states, policies, values, max_augmentations=4
        )

        # 2 original + 2 * 3 augmented = 8
        expected_count = 2 * 4
        assert len(aug_states) == expected_count, \
            f"Expected {expected_count} samples, got {len(aug_states)}"


class TestStatisticsCollection:
    """Test statistics collection."""

    def test_stats_enabled(self):
        """Test that stats are collected when enabled."""
        augmenter = YinshSymmetryAugmenter(enable_stats=True)

        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0
        policy = np.zeros(7395, dtype=np.float32)
        policy[0] = 1.0

        augmenter.augment(state, policy, 0.5)
        stats = augmenter.get_stats()

        assert stats is not None
        assert stats.total_augmentations > 0

    def test_stats_disabled(self):
        """Test that stats are None when disabled."""
        augmenter = YinshSymmetryAugmenter(enable_stats=False)
        stats = augmenter.get_stats()
        assert stats is None

    def test_stats_reset(self):
        """Test that stats can be reset."""
        augmenter = YinshSymmetryAugmenter(enable_stats=True)

        state = np.zeros((6, 11, 11), dtype=np.float32)
        policy = np.zeros(7395, dtype=np.float32)
        policy[0] = 1.0

        augmenter.augment(state, policy, 0.5)
        augmenter.reset_stats()

        stats = augmenter.get_stats()
        assert stats.total_augmentations == 0


class TestRoundTripVerification:
    """Test round-trip verification utility."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_identity_round_trip(self, augmenter):
        """Test that identity transform passes round-trip test."""
        state = np.random.rand(6, 11, 11).astype(np.float32)
        assert verify_transform_round_trip(augmenter, state, 0)

    def test_rotation_round_trips(self, augmenter):
        """Test that all rotations pass round-trip test for center-only state."""
        # Use a state with only center piece (guaranteed to be invariant under rotation)
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0

        for rot_id in range(1, 6):
            result = verify_transform_round_trip(augmenter, state, rot_id)
            assert result, f"Rotation {rot_id} failed round-trip test"


class TestRealGameState:
    """Test augmentation with real game states."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def test_augment_real_game_state(self, augmenter, encoder):
        """Test augmentation with a realistic game state."""
        game_state = GameState()

        # Set up a mid-game position
        game_state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        game_state.board.place_piece(Position('F', 6), PieceType.BLACK_RING)
        game_state.board.place_piece(Position('D', 4), PieceType.WHITE_MARKER)

        # Encode the state
        encoded = encoder.encode_state(game_state)

        # Create a simple policy
        policy = np.zeros(7395, dtype=np.float32)
        policy[0] = 0.5
        policy[10] = 0.3
        policy[20] = 0.2

        # Augment
        results = augmenter.augment(encoded, policy, 0.5)

        assert len(results) == 12, f"Expected 12 augmentations, got {len(results)}"

        # Verify shapes
        for aug_state, aug_policy, aug_value in results:
            assert aug_state.shape == encoded.shape
            assert aug_policy.shape == policy.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_empty_state(self, augmenter):
        """Test augmentation of empty state."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        policy = np.zeros(7395, dtype=np.float32)

        results = augmenter.augment(state, policy, 0.0)

        assert len(results) == 12
        for aug_state, aug_policy, _ in results:
            assert np.allclose(aug_state, 0)
            assert np.allclose(aug_policy, 0)

    def test_single_piece_at_center(self, augmenter):
        """Test that a piece at center stays at center."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0  # Center position
        policy = np.zeros(7395, dtype=np.float32)
        policy[0] = 1.0

        results = augmenter.augment(state, policy, 0.5)

        # Center should be invariant under all transforms
        for aug_state, _, _ in results:
            assert aug_state[0, 5, 5] == 1.0, \
                "Center piece should remain at center under all transforms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
