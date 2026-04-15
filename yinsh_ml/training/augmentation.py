"""D2 symmetry augmentation for YINSH training data.

The YINSH board has D2 (Klein 4-group) symmetry, not D6 as originally assumed.
Column sizes [4,7,8,9,10,9,10,9,8,7,4] are symmetric under 180° rotation and
under two axes (diagonal and anti-diagonal), but NOT under 60° rotation —
verify: A column has 4 cells while G has 10, so a 60° rotation would need to
map a column with 4 cells somewhere, and no such image column exists.

The 4 true symmetries of the board:
  T0 — identity:          (row, col) -> (row, col)
  T1 — 180° rotation:     (row, col) -> (10-row, 10-col)
  T2 — diagonal swap:     (row, col) -> (col, row)
  T3 — anti-diagonal:     (row, col) -> (10-col, 10-row)

All 4 cover the full 85-cell valid set (verified empirically).
Composition law: T1 ∘ T2 = T3 (and permutations).

This yields a 4× data multiplier, not the 12× previously advertised.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import logging
from dataclasses import dataclass, field

from ..game.constants import Position, is_valid_position, VALID_POSITIONS
from ..game.types import Move, MoveType
from ..utils.encoding import StateEncoder

logger = logging.getLogger(__name__)


@dataclass
class AugmentationStats:
    """Statistics for debugging augmentation behavior."""
    total_augmentations: int = 0
    rotations_applied: int = 0
    reflections_applied: int = 0
    invalid_transforms_skipped: int = 0


class YinshSymmetryAugmenter:
    """D2 symmetry augmentation for YINSH training data.

    The board's Klein 4-group symmetry {identity, 180°, diag, anti-diag} is the
    largest group that maps the 85-cell valid-position set to itself. Applies the
    geometric transform to state channels and builds a correct valid-move-based
    permutation for the policy (see _transform_policy_with_base).

    Value is invariant under these symmetries, so it passes through unchanged.
    """

    BOARD_SIZE = 11

    def __init__(self,
                 include_reflections: bool = True,
                 state_encoder: Optional[StateEncoder] = None,
                 enable_stats: bool = False):
        """Initialize the augmenter.

        Args:
            include_reflections: If True, include the 2 reflections (total 4 transforms
                incl. identity + 180°). If False, C2 only: {identity, 180°}.
            state_encoder: Optional StateEncoder for policy transforms.
            enable_stats: If True, collect statistics for debugging.
        """
        self.include_reflections = include_reflections
        self.state_encoder = state_encoder or StateEncoder()
        self.enable_stats = enable_stats
        self._stats = AugmentationStats() if enable_stats else None

        self.num_transforms = 4 if include_reflections else 2

        self._coord_maps = self._precompute_coord_maps()

        logger.info(f"YinshSymmetryAugmenter initialized: {self.num_transforms} transforms "
                   f"(reflections={'enabled' if include_reflections else 'disabled'})")

    # Transform IDs (index into _coord_maps):
    #   0 — identity
    #   1 — 180° rotation
    #   2 — diagonal swap     (axis row = col)
    #   3 — anti-diagonal     (axis row + col = 10)
    # With include_reflections=False, only 0 and 1 are used.
    _TRANSFORMS = (
        lambda r, c: (r, c),
        lambda r, c: (10 - r, 10 - c),
        lambda r, c: (c, r),
        lambda r, c: (10 - c, 10 - r),
    )

    def _precompute_coord_maps(self) -> List[Dict[Tuple[int, int], Tuple[int, int]]]:
        """Precompute (row_idx, col_idx) -> (new_row_idx, new_col_idx) for every valid
        board position, for each of the D2 transforms."""
        coord_maps = []
        for tid in range(self.num_transforms):
            transform = self._TRANSFORMS[tid]
            coord_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for col_idx in range(self.BOARD_SIZE):
                for row_idx in range(self.BOARD_SIZE):
                    pos = Position(chr(ord('A') + col_idx), row_idx + 1)
                    if not is_valid_position(pos):
                        continue
                    new_row_idx, new_col_idx = transform(row_idx, col_idx)
                    coord_map[(row_idx, col_idx)] = (new_row_idx, new_col_idx)
            coord_maps.append(coord_map)
        return coord_maps

    def augment(self,
                state: np.ndarray,
                policy: np.ndarray,
                value: float,
                include_original: bool = True
                ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate all symmetric versions of a training sample.

        Args:
            state: Game state tensor of shape (C, 11, 11)
            policy: Policy tensor of shape (7395,) or (num_moves,)
            value: Position value (unchanged across transforms)
            include_original: If True, include the original (transform_id=0) in output

        Returns:
            List of (augmented_state, augmented_policy, value) tuples.
            Length is num_transforms if include_original, else num_transforms - 1.
        """
        results = []

        # Decode state and enumerate valid moves *once* — they're the same across
        # all 12 transforms of this sample. Caches valid_moves + their old_idx values
        # so each per-transform policy build only does coord_map + re-encode work.
        try:
            base_moves = self._base_move_encoding(state)
        except Exception:
            base_moves = None

        start_idx = 0 if include_original else 1

        for transform_id in range(start_idx, self.num_transforms):
            try:
                aug_state = self._transform_state(state, transform_id)
                aug_policy = self._transform_policy_with_base(
                    policy, transform_id, base_moves
                )
                results.append((aug_state, aug_policy, value))

                if self.enable_stats:
                    self._stats.total_augmentations += 1
                    # In D2: T1 is a rotation, T2/T3 are reflections.
                    if transform_id == 1:
                        self._stats.rotations_applied += 1
                    else:
                        self._stats.reflections_applied += 1

            except Exception as e:
                logger.warning(f"Failed to apply transform {transform_id}: {e}")
                if self.enable_stats:
                    self._stats.invalid_transforms_skipped += 1
                continue

        return results

    def _base_move_encoding(self, state: np.ndarray) -> List[Tuple[Move, int]]:
        """Decode state once, enumerate valid moves, and pair each with its old policy
        index. Shared across all 12 transforms of a sample."""
        game_state = self.state_encoder.decode_state(state)
        out: List[Tuple[Move, int]] = []
        for move in game_state.get_valid_moves():
            try:
                old_idx = self.state_encoder.move_to_index(move)
            except Exception:
                continue
            out.append((move, old_idx))
        return out

    def _transform_policy_with_base(self,
                                    policy: np.ndarray,
                                    transform_id: int,
                                    base_moves: Optional[List[Tuple[Move, int]]]
                                    ) -> np.ndarray:
        """Variant of _transform_policy that reuses a precomputed (move, old_idx) list."""
        if transform_id == 0:
            return policy.copy()
        if base_moves is None:
            raise ValueError("State did not decode to usable valid moves")

        coord_map = self._coord_maps[transform_id]
        transformed_policy = np.zeros_like(policy)
        any_mapped = False

        for move, old_idx in base_moves:
            transformed = self._transform_move(move, coord_map)
            if transformed is None:
                continue
            try:
                new_idx = self.state_encoder.move_to_index(transformed)
            except Exception:
                continue
            if 0 <= new_idx < len(transformed_policy):
                transformed_policy[new_idx] += policy[old_idx]
                any_mapped = True

        if not any_mapped:
            raise ValueError(f"No valid-move mapping for transform {transform_id}")

        policy_sum = transformed_policy.sum()
        if policy_sum > 1e-6:
            transformed_policy /= policy_sum

        return transformed_policy

    def _transform_state(self, state: np.ndarray, transform_id: int) -> np.ndarray:
        """Apply geometric transform to all channels of the state.

        The coord_map covers the 85 valid board positions. The remaining 36 cells
        of the 11×11 grid are *off-board* — they never hold pieces, but the encoder
        uses them for channels that are spatially-uniform (e.g. the phase channel
        broadcasts a scalar to every cell). We start from `state.copy()` so those
        off-board cells retain their source value; the coord_map writes then
        overwrite the 85 on-board cells with the rotated values.
        """
        if transform_id == 0:
            return state.copy()

        transformed = state.copy()
        coord_map = self._coord_maps[transform_id]

        num_channels = state.shape[0]
        for (old_row, old_col), (new_row, new_col) in coord_map.items():
            for c in range(num_channels):
                transformed[c, new_row, new_col] = state[c, old_row, old_col]

        return transformed

    def _transform_policy(self,
                          state: np.ndarray,
                          policy: np.ndarray,
                          transform_id: int) -> np.ndarray:
        """Transform policy distribution under the geometric symmetry.

        Builds a valid-move permutation by forward-encoding each valid move on the
        original state and its geometric image. Mass at invalid indices is dropped;
        remaining mass is renormalized to preserve distribution total.

        Raises ValueError if the permutation cannot be built (e.g. state doesn't
        decode), so the caller can skip this transform rather than emit a
        mistransformed sample.
        """
        if transform_id == 0:
            return policy.copy()

        permutation = self._build_index_permutation(state, transform_id)
        if not permutation:
            raise ValueError(
                f"Empty permutation for transform {transform_id}; "
                "likely terminal state or decode failure"
            )

        transformed_policy = np.zeros_like(policy)
        for old_idx, new_idx in permutation.items():
            if 0 <= new_idx < len(transformed_policy):
                transformed_policy[new_idx] += policy[old_idx]

        policy_sum = transformed_policy.sum()
        if policy_sum > 1e-6:
            transformed_policy /= policy_sum

        return transformed_policy

    def _build_index_permutation(self,
                                 state: np.ndarray,
                                 transform_id: int) -> Dict[int, int]:
        """Build {old_idx -> new_idx} by forward-encoding each valid move and its image.

        This avoids inverting the encoder's lossy move-hash: we only encode forward,
        which is deterministic. Only indices with a valid-move preimage are included,
        so mass at invalid indices in the original state is dropped during the
        transform (renormalization restores the sum).
        """
        coord_map = self._coord_maps[transform_id]

        game_state = self.state_encoder.decode_state(state)
        valid_moves = game_state.get_valid_moves()

        permutation: Dict[int, int] = {}
        for move in valid_moves:
            try:
                old_idx = self.state_encoder.move_to_index(move)
            except Exception:
                continue

            transformed = self._transform_move(move, coord_map)
            if transformed is None:
                continue

            try:
                new_idx = self.state_encoder.move_to_index(transformed)
            except Exception:
                continue

            permutation[old_idx] = new_idx

        return permutation

    def _transform_move(self,
                        move: Move,
                        coord_map: Dict[Tuple[int, int], Tuple[int, int]]
                        ) -> Optional[Move]:
        """Route every Position on a Move through the coord_map; return None if any
        target position is off-board / off-hex."""
        if move.type == MoveType.PLACE_RING:
            new_source = self._transform_position(move.source, coord_map)
            if new_source is None:
                return None
            return Move(type=move.type, player=move.player, source=new_source)

        if move.type == MoveType.MOVE_RING:
            new_source = self._transform_position(move.source, coord_map)
            new_dest = self._transform_position(move.destination, coord_map)
            if new_source is None or new_dest is None:
                return None
            return Move(type=move.type, player=move.player,
                        source=new_source, destination=new_dest)

        if move.type == MoveType.REMOVE_RING:
            new_source = self._transform_position(move.source, coord_map)
            if new_source is None:
                return None
            return Move(type=move.type, player=move.player, source=new_source)

        if move.type == MoveType.REMOVE_MARKERS:
            new_markers = []
            for marker in (move.markers or ()):
                m = self._transform_position(marker, coord_map)
                if m is None:
                    return None
                new_markers.append(m)
            return Move(type=move.type, player=move.player,
                        markers=tuple(new_markers))

        return None

    def _transform_position(self,
                            pos: Position,
                            coord_map: Dict[Tuple[int, int], Tuple[int, int]]
                            ) -> Optional[Position]:
        """Apply the geometric transform to a Position. Returns None if the image is
        not a valid hex-board position."""
        row_idx = pos.row - 1
        col_idx = ord(pos.column) - ord('A')
        mapped = coord_map.get((row_idx, col_idx))
        if mapped is None:
            return None
        new_row_idx, new_col_idx = mapped
        return Position(chr(ord('A') + new_col_idx), new_row_idx + 1)

    def augment_batch(self,
                      states: List[np.ndarray],
                      policies: List[np.ndarray],
                      values: List[float],
                      max_augmentations: Optional[int] = None
                      ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Augment a batch of training samples.

        Args:
            states: List of state tensors
            policies: List of policy tensors
            values: List of values
            max_augmentations: Maximum number of augmentations per sample.
                              If None, generate all transforms.

        Returns:
            Tuple of (augmented_states, augmented_policies, augmented_values)
        """
        aug_states = []
        aug_policies = []
        aug_values = []

        n_transforms = min(self.num_transforms, max_augmentations) if max_augmentations else self.num_transforms

        for state, policy, value in zip(states, policies, values):
            # Always include original
            aug_states.append(state)
            aug_policies.append(policy)
            aug_values.append(value)

            # Add augmentations (excluding identity)
            augmented = self.augment(state, policy, value, include_original=False)

            # Possibly limit number of augmentations
            if max_augmentations and len(augmented) > max_augmentations - 1:
                # Randomly sample augmentations
                indices = np.random.choice(len(augmented), max_augmentations - 1, replace=False)
                augmented = [augmented[i] for i in indices]

            for aug_state, aug_policy, aug_value in augmented:
                aug_states.append(aug_state)
                aug_policies.append(aug_policy)
                aug_values.append(aug_value)

        return aug_states, aug_policies, aug_values

    def get_stats(self) -> Optional[AugmentationStats]:
        """Get augmentation statistics (if enabled)."""
        return self._stats

    def reset_stats(self):
        """Reset statistics counters."""
        if self._stats:
            self._stats = AugmentationStats()


def verify_transform_round_trip(augmenter: YinshSymmetryAugmenter,
                                state: np.ndarray,
                                transform_id: int) -> bool:
    """Verify that applying a D2 transform twice recovers the original state.

    Every element of the Klein 4-group has order 2, so each transform is its
    own inverse.
    """
    if transform_id == 0:
        return True
    forward = augmenter._transform_state(state, transform_id)
    roundtrip = augmenter._transform_state(forward, transform_id)
    return np.allclose(state, roundtrip, atol=1e-6)


def test_augmentation_basic():
    """Basic test of augmentation functionality."""
    augmenter = YinshSymmetryAugmenter(include_reflections=True, enable_stats=True)

    # Create a simple test state
    state = np.zeros((6, 11, 11), dtype=np.float32)
    state[0, 5, 5] = 1.0  # Ring at center
    state[2, 4, 4] = 1.0  # Marker near center

    policy = np.zeros(7395, dtype=np.float32)
    policy[0] = 0.5
    policy[1] = 0.5

    value = 0.5

    # Generate augmentations
    augmented = augmenter.augment(state, policy, value)

    print(f"Generated {len(augmented)} augmentations")
    print(f"Stats: {augmenter.get_stats()}")

    # Verify shapes
    for aug_state, aug_policy, aug_value in augmented:
        assert aug_state.shape == state.shape, f"State shape mismatch: {aug_state.shape}"
        assert aug_policy.shape == policy.shape, f"Policy shape mismatch: {aug_policy.shape}"
        assert aug_value == value, f"Value changed: {aug_value}"

    print("Basic augmentation test passed!")
    return True


if __name__ == '__main__':
    test_augmentation_basic()
