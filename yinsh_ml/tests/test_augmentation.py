"""Unit tests for D2 symmetry augmentation.

Transform IDs (see YinshSymmetryAugmenter._TRANSFORMS):
  0 — identity
  1 — 180° rotation
  2 — diagonal swap
  3 — anti-diagonal swap

Each covers all 85 valid cells; all are order-2 and compose as the Klein 4-group.
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
from yinsh_ml.game.types import Move, MoveType, GamePhase


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
        """D2 group: identity + 180° + 2 reflections = 4 transforms."""
        assert augmenter_with_reflections.num_transforms == 4

    def test_num_transforms_rotations_only(self, augmenter_rotations_only):
        """C2 subgroup (rotations only): identity + 180° = 2 transforms."""
        assert augmenter_rotations_only.num_transforms == 2

    def test_augment_returns_correct_count(self, augmenter_with_reflections, simple_state, simple_policy):
        results = augmenter_with_reflections.augment(
            simple_state, simple_policy, 0.5, include_original=True
        )
        assert len(results) == 4

    def test_augment_without_original(self, augmenter_with_reflections, simple_state, simple_policy):
        results = augmenter_with_reflections.augment(
            simple_state, simple_policy, 0.5, include_original=False
        )
        assert len(results) == 3

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
        state = np.zeros((6, 11, 11), dtype=np.float32)
        policy = np.random.rand(7395).astype(np.float32)
        policy /= policy.sum()  # Normalize
        transformed = augmenter._transform_policy(state, policy, 0)
        assert np.array_equal(transformed, policy), "Identity transform should not change policy"


class TestD2GroupStructure:
    """All 4 D2 elements are order-2; together they form the Klein 4-group."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_every_transform_covers_all_85_cells(self, augmenter):
        """Every D2 transform must map the 85-cell valid set to itself."""
        for tid in range(augmenter.num_transforms):
            assert len(augmenter._coord_maps[tid]) == 85, \
                f"Transform {tid} covers {len(augmenter._coord_maps[tid])} cells (expected 85)"

    def test_every_transform_is_self_inverse(self, augmenter):
        """D2 has exponent 2: applying any transform twice is identity."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0   # center — invariant under all
        state[0, 4, 4] = 1.0
        state[2, 6, 6] = 1.0

        for tid in range(1, augmenter.num_transforms):
            once = augmenter._transform_state(state, tid)
            twice = augmenter._transform_state(once, tid)
            assert np.allclose(state, twice), \
                f"Transform {tid} applied twice did not recover original state"

    def test_180_equals_diag_composed_with_antidiag(self, augmenter):
        """Klein 4-group law: 180° = diagonal ∘ anti-diagonal (up to sign choice)."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 2, 3] = 1.0
        state[0, 7, 8] = 1.0

        via_180 = augmenter._transform_state(state, 1)
        via_composition = augmenter._transform_state(
            augmenter._transform_state(state, 2), 3
        )
        assert np.allclose(via_180, via_composition), \
            "T1 should equal T2 ∘ T3 (Klein 4-group closure)"


class TestCoordinateTransforms:
    """Verify the D2 coord maps put center + edge positions where expected."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_center_is_invariant_under_every_transform(self, augmenter):
        """(5, 5) is fixed under all 4 D2 transforms."""
        for tid in range(augmenter.num_transforms):
            assert augmenter._coord_maps[tid][(5, 5)] == (5, 5), \
                f"Transform {tid} moved center"

    def test_180_maps_corner_to_opposite_corner(self, augmenter):
        """A2 (row_idx=1, col_idx=0) must round-trip to K10 (row_idx=9, col_idx=10)."""
        assert augmenter._coord_maps[1][(1, 0)] == (9, 10)
        assert augmenter._coord_maps[1][(9, 10)] == (1, 0)

    def test_diagonal_swap_swaps_axes(self, augmenter):
        """Diagonal: (r, c) -> (c, r). A2 → B1."""
        assert augmenter._coord_maps[2][(1, 0)] == (0, 1)


class TestPhaseChannelPreservation:
    """Regression test for the previous _transform_state phase-channel bug:
    encode_state fills channel 5 uniformly across all 121 cells, but the old
    _transform_state zeroed the 36 off-board cells, shrinking the channel mean
    and causing decode_state to report the wrong GamePhase after any rotation."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    @pytest.mark.parametrize("phase", list(GamePhase))
    def test_decoded_phase_preserved_under_every_transform(self, augmenter, encoder, phase):
        gs = GameState()
        gs.phase = phase
        gs.current_player = Player.WHITE
        gs.board.place_piece(Position('F', 6), PieceType.WHITE_RING)
        state = encoder.encode_state(gs)

        for tid in range(augmenter.num_transforms):
            transformed = augmenter._transform_state(state, tid)
            decoded = encoder.decode_state(transformed)
            assert decoded.phase == phase, \
                f"T{tid}: decoded phase {decoded.phase} != original {phase}"


class TestPolicyTransforms:
    """Test policy transformation."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_policy_sum_preserved(self, augmenter):
        """After drop-and-renormalize, any surviving mass is normalized to sum 1.0."""
        state = np.zeros((6, 11, 11), dtype=np.float32)  # RING_PLACEMENT, all empty
        policy = np.random.rand(7395).astype(np.float32)
        policy /= policy.sum()

        for transform_id in range(1, augmenter.num_transforms):
            transformed = augmenter._transform_policy(state, policy, transform_id)

            transformed_sum = transformed.sum()
            assert abs(transformed_sum - 1.0) < 1e-5 or transformed_sum < 1e-6, \
                f"Transform {transform_id}: policy sum = {transformed_sum}"

    def test_zero_policy_stays_zero(self, augmenter):
        """Zero policy stays zero — no mass to redistribute."""
        state = np.zeros((6, 11, 11), dtype=np.float32)  # RING_PLACEMENT, all empty
        policy = np.zeros(7395, dtype=np.float32)

        for transform_id in range(augmenter.num_transforms):
            transformed = augmenter._transform_policy(state, policy, transform_id)
            assert np.allclose(transformed, 0), \
                f"Transform {transform_id} created non-zero values from zero policy"


class TestBatchAugmentation:
    """Test batch augmentation functionality."""

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    def test_batch_augment_correct_count(self, augmenter):
        """Batch augmentation emits num_transforms samples per original."""
        states = [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(3)]
        policies = [np.zeros(7395, dtype=np.float32) for _ in range(3)]
        policies[0][0] = 1.0
        policies[1][1] = 1.0
        policies[2][2] = 1.0
        values = [0.1, 0.2, 0.3]

        aug_states, aug_policies, aug_values = augmenter.augment_batch(
            states, policies, values
        )

        expected_count = 3 * augmenter.num_transforms
        assert len(aug_states) == expected_count
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

    def test_all_transforms_round_trip(self, augmenter):
        """Center-only state is invariant under every D2 transform and round-trips."""
        state = np.zeros((6, 11, 11), dtype=np.float32)
        state[0, 5, 5] = 1.0

        for tid in range(1, augmenter.num_transforms):
            result = verify_transform_round_trip(augmenter, state, tid)
            assert result, f"Transform {tid} failed round-trip test"


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

        assert len(results) == augmenter.num_transforms

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

        assert len(results) == augmenter.num_transforms
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


class TestPolicyGeometricCorrectness:
    """Verify the transformed policy points at the geometrically-correct move in
    the transformed state, across all 3 non-identity D2 transforms."""

    NON_IDENTITY_TRANSFORMS = (1, 2, 3)  # 180°, diag, anti-diag

    @pytest.fixture
    def augmenter(self):
        return YinshSymmetryAugmenter(include_reflections=True)

    @pytest.fixture
    def encoder(self):
        return StateEncoder()

    def _ring_placement_state(self):
        gs = GameState()
        gs.phase = GamePhase.RING_PLACEMENT
        return gs

    def _main_game_state(self):
        """MAIN_GAME with rings placed on positions inside the 180°-safe region."""
        gs = GameState()
        gs.phase = GamePhase.MAIN_GAME
        for col, row in [('E', 5), ('F', 7), ('G', 5)]:
            gs.board.place_piece(Position(col, row), PieceType.WHITE_RING)
        for col, row in [('E', 8), ('F', 4), ('G', 8)]:
            gs.board.place_piece(Position(col, row), PieceType.BLACK_RING)
        gs.current_player = Player.WHITE
        gs.rings_placed = {Player.WHITE: 5, Player.BLACK: 5}
        return gs

    def _ring_removal_state(self):
        """RING_REMOVAL with WHITE to remove. Ring counts honor decode_state's
        heuristic `current=WHITE iff white_rings <= black_rings`, which in side-
        normalized encoding means the *current player's rings* (ch0) must not
        outnumber the opponent's (ch1) after decoding."""
        gs = GameState()
        gs.phase = GamePhase.RING_REMOVAL
        gs.current_player = Player.WHITE
        gs.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        for col, row in [('F', 6), ('G', 7)]:
            gs.board.place_piece(Position(col, row), PieceType.BLACK_RING)
        gs.rings_placed = {Player.WHITE: 5, Player.BLACK: 5}
        return gs

    def _apply_coord_transform_to_gs(self, gs, coord_map):
        """Manually rebuild a GameState by routing every piece position through the
        coord_map — used to cross-check the augmented state."""
        rotated = GameState()
        rotated.phase = gs.phase
        rotated.current_player = gs.current_player
        rotated.rings_placed = dict(gs.rings_placed)

        def rot(pos):
            row_idx = pos.row - 1
            col_idx = ord(pos.column) - ord('A')
            new_row_idx, new_col_idx = coord_map[(row_idx, col_idx)]
            return Position(chr(ord('A') + new_col_idx), new_row_idx + 1)

        for piece_type in (PieceType.WHITE_RING, PieceType.BLACK_RING,
                           PieceType.WHITE_MARKER, PieceType.BLACK_MARKER):
            for pos in gs.board.get_pieces_positions(piece_type):
                rotated.board.place_piece(rot(pos), piece_type)
        return rotated

    def test_place_ring_round_trip_all_d2(self, augmenter, encoder):
        """PLACE_RING under every D2 transform: exact round-trip for every valid position."""
        gs = self._ring_placement_state()
        state = encoder.encode_state(gs)
        valid_moves = gs.get_valid_moves()

        for tid in self.NON_IDENTITY_TRANSFORMS:
            for move in valid_moves:
                policy = np.zeros(7395, dtype=np.float32)
                policy[encoder.move_to_index(move)] = 1.0
                aug_policy = augmenter._transform_policy(state, policy, tid)

                expected_move = augmenter._transform_move(move, augmenter._coord_maps[tid])
                assert expected_move is not None
                expected_idx = encoder.move_to_index(expected_move)
                assert aug_policy[expected_idx] == pytest.approx(1.0, abs=1e-5), \
                    f"T{tid}: PLACE_RING {move.source} mass not at expected idx"

    def test_move_ring_round_trip_all_d2(self, augmenter, encoder):
        """MOVE_RING under every D2 transform — exercises the hashed path."""
        gs = self._main_game_state()
        state = encoder.encode_state(gs)
        valid_moves = gs.get_valid_moves()
        assert valid_moves

        for tid in self.NON_IDENTITY_TRANSFORMS:
            for move in valid_moves[:10]:
                policy = np.zeros(7395, dtype=np.float32)
                policy[encoder.move_to_index(move)] = 1.0
                aug_policy = augmenter._transform_policy(state, policy, tid)

                expected_move = augmenter._transform_move(move, augmenter._coord_maps[tid])
                assert expected_move is not None
                expected_idx = encoder.move_to_index(expected_move)
                assert aug_policy[expected_idx] == pytest.approx(1.0, abs=1e-5), \
                    f"T{tid}: MOVE_RING {move.source}→{move.destination} mass not at expected idx"

    def test_remove_ring_round_trip_all_d2(self, augmenter, encoder):
        """REMOVE_RING under every D2 transform."""
        gs = self._ring_removal_state()
        state = encoder.encode_state(gs)
        valid_moves = gs.get_valid_moves()

        for tid in self.NON_IDENTITY_TRANSFORMS:
            for move in valid_moves:
                policy = np.zeros(7395, dtype=np.float32)
                policy[encoder.move_to_index(move)] = 1.0
                aug_policy = augmenter._transform_policy(state, policy, tid)

                expected_move = augmenter._transform_move(move, augmenter._coord_maps[tid])
                assert expected_move is not None
                expected_idx = encoder.move_to_index(expected_move)
                assert aug_policy[expected_idx] == pytest.approx(1.0, abs=1e-5), \
                    f"T{tid}: REMOVE_RING {move.source} mass not at expected idx"

    def test_every_transform_is_policy_self_inverse(self, augmenter, encoder):
        """Every D2 element has order 2: applying the same transform twice recovers
        the original policy, given the correctly-rotated intermediate state."""
        gs = self._main_game_state()
        state = encoder.encode_state(gs)
        move = gs.get_valid_moves()[0]
        original_idx = encoder.move_to_index(move)
        policy = np.zeros(7395, dtype=np.float32)
        policy[original_idx] = 1.0

        for tid in self.NON_IDENTITY_TRANSFORMS:
            rotated_gs = self._apply_coord_transform_to_gs(gs, augmenter._coord_maps[tid])
            rotated_state = encoder.encode_state(rotated_gs)

            once_policy = augmenter._transform_policy(state, policy, tid)
            twice_policy = augmenter._transform_policy(rotated_state, once_policy, tid)
            assert np.argmax(twice_policy) == original_idx, \
                f"T{tid} applied twice did not recover policy peak"
            assert twice_policy[original_idx] == pytest.approx(1.0, abs=1e-5)

    def test_invalid_indices_dropped_and_renormalized(self, augmenter, encoder):
        """Mass at an invalid-move index is dropped; remaining mass renormalizes to 1.0."""
        gs = self._ring_placement_state()
        state = encoder.encode_state(gs)
        valid_move = next(m for m in gs.get_valid_moves()
                          if m.source.column == 'F' and m.source.row == 6)
        valid_idx = encoder.move_to_index(valid_move)

        invalid_idx = encoder.move_ring_range[0] + 7  # MOVE_RING slot under PLACEMENT phase
        assert invalid_idx != valid_idx

        policy = np.zeros(7395, dtype=np.float32)
        policy[valid_idx] = 0.6
        policy[invalid_idx] = 0.4

        aug_policy = augmenter._transform_policy(state, policy, 1)
        assert aug_policy.sum() == pytest.approx(1.0, abs=1e-5)
        expected_move = augmenter._transform_move(valid_move, augmenter._coord_maps[1])
        assert aug_policy[encoder.move_to_index(expected_move)] == pytest.approx(1.0, abs=1e-5)

    def test_augmented_policy_mass_on_valid_moves_only_all_d2(self, augmenter, encoder):
        """Mass lands only at indices valid in the transformed state, for every D2 transform."""
        gs = self._main_game_state()
        state = encoder.encode_state(gs)
        valid_moves = gs.get_valid_moves()
        policy = np.zeros(7395, dtype=np.float32)
        for m in valid_moves:
            policy[encoder.move_to_index(m)] = 1.0
        policy /= policy.sum()

        for tid in self.NON_IDENTITY_TRANSFORMS:
            rotated_gs = self._apply_coord_transform_to_gs(gs, augmenter._coord_maps[tid])
            aug_policy = augmenter._transform_policy(state, policy, tid)
            rotated_valid_indices = {encoder.move_to_index(m)
                                     for m in rotated_gs.get_valid_moves()}
            nonzero = set(np.nonzero(aug_policy)[0].tolist())
            stray = nonzero - rotated_valid_indices
            assert not stray, \
                f"T{tid}: augmented policy has mass at indices not valid in rotated state: {stray}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
