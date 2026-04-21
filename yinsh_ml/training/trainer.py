"""Training implementation for YINSH ML model."""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time
from collections import deque, defaultdict
import random


from ..utils.metrics_logger import MetricsLogger, EpochMetrics
from ..utils.enhanced_metrics import EnhancedMetricsCollector
from ..network.wrapper import NetworkWrapper
from .ema import EMAShadow
from yinsh_ml.utils.value_head_metrics import ValueHeadMetrics


# Setup logger
logger = logging.getLogger(__name__)


class GameExperience:
    """Stores game states and outcomes for training, with optimized memory usage."""

    def __init__(self, max_size: int = 100000, subsample_long_games: bool = True,
                 enable_augmentation: bool = False, max_augmentations: int = 12):
        """Initialize GameExperience replay buffer.

        Args:
            max_size: Maximum number of samples to store (default: 100000, increased from 10000 for better training)
            subsample_long_games: Whether to subsample games >100 moves
            enable_augmentation: If True, apply D6 symmetry augmentation to training data
            max_augmentations: Maximum number of augmented samples per original (1-12)
        """
        self.states = deque(maxlen=max_size)
        self.move_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.phases = deque(maxlen=max_size)
        # move_numbers: per-sample outer-game move index. Used by the search-
        # consistency probe (`YinshTrainer._search_consistency_step`) to drive
        # `MCTS.search(state, move_number)` correctly — `decode_state` cannot
        # recover the move number from the encoded tensor, so we have to track
        # it alongside. Default 0 for old buffer files loaded without this
        # field.
        self.move_numbers = deque(maxlen=max_size)
        self.max_size = max_size  # Store for logging
        self.subsample_long_games = subsample_long_games

        # Augmentation settings
        self.enable_augmentation = enable_augmentation
        self.max_augmentations = min(max_augmentations, 12)
        self._augmenter = None
        self._augmentation_stats = {'total_original': 0, 'total_augmented': 0}

        if enable_augmentation:
            try:
                from .augmentation import YinshSymmetryAugmenter
                self._augmenter = YinshSymmetryAugmenter(
                    include_reflections=True,
                    enable_stats=False  # Disable stats for production
                )
                logger.info(f"Augmentation enabled: up to {self.max_augmentations}x data expansion")
            except ImportError as e:
                logger.warning(f"Could not import augmenter: {e}. Augmentation disabled.")
                self.enable_augmentation = False

        # Memory monitoring
        self._total_states_added = 0
        self._memory_check_interval = 500
        self._last_memory_check = 0

    def add_game_experience(self,
                            states: list,
                            policies: list,
                            values: list,
                            final_white_score: int = None,
                            final_black_score: int = None,
                            discount_factor: float = 1.0,
                            move_numbers: Optional[List[int]] = None):
        """
        Add a completed game to the replay buffer with memory optimization.

        Fix #1: Now accepts MCTS root values directly instead of computing from game outcome.

        Args:
            states:             List of game states (numpy arrays).
            policies:           List of move probability distributions.
            values:             List of MCTS root values (position-specific training targets).
            final_white_score:  (Optional) White's final score - only used for logging.
            final_black_score:  (Optional) Black's final score - only used for logging.
            discount_factor:    (Deprecated) No longer used - values are already position-specific.
            move_numbers:       (Optional) Per-position outer-game move index.
                                Defaults to ``range(len(states))`` — correct since
                                callers pass states in move order. Tracked so the
                                search-consistency probe can drive MCTS with the
                                correct move number (``decode_state`` can't recover
                                it from the encoded tensor).
        """
        # Safety check
        assert len(states) == len(policies), "Mismatch: states vs. policies!"
        assert len(states) == len(values), "Mismatch: states vs. values!"
        if move_numbers is None:
            move_numbers = list(range(len(states)))
        else:
            assert len(states) == len(move_numbers), "Mismatch: states vs. move_numbers!"

        # Memory optimization: Subsample very long games
        if self.subsample_long_games and len(states) > 100:
            # For very long games, keep early, late, and sample middle positions
            keep_early = 20  # Keep first 20 moves
            keep_late = 20  # Keep last 20 moves

            # If game is long enough to need subsampling
            if len(states) > (keep_early + keep_late + 10):
                # Take positions from the middle
                middle_indices = list(range(keep_early, len(states) - keep_late))
                # Sample up to 30 positions from middle
                num_to_sample = min(30, len(middle_indices))
                if num_to_sample > 0:
                    sampled_indices = sorted(random.sample(middle_indices, num_to_sample))
                    keep_indices = list(range(keep_early)) + sampled_indices + list(
                        range(len(states) - keep_late, len(states)))

                    # Create subsampled game
                    states = [states[i] for i in keep_indices]
                    policies = [policies[i] for i in keep_indices]
                    values = [values[i] for i in keep_indices]  # Fix #1: Also subsample values
                    move_numbers = [move_numbers[i] for i in keep_indices]

                    logger.debug(f"[Memory Opt] Reduced long game from {len(states)} → {len(keep_indices)} states")

        # Helper to decode phases from channel 5
        def decode_phase(state: np.ndarray) -> str:
            phase_channel = state[5]  # 5th channel for 'GAME_PHASE'
            avg = np.mean(np.abs(phase_channel))
            if avg < 0.2:
                phase = "RING_PLACEMENT"
            elif avg < 0.6:
                phase = "MAIN_GAME"
            else:
                phase = "RING_REMOVAL"
            return phase

        # Fix #1: No longer compute normalized_margin - values are already position-specific from MCTS
        # Keep score computation for logging only
        if final_white_score is not None and final_black_score is not None:
            score_diff = final_white_score - final_black_score
            normalized_margin = score_diff / 3.0
        else:
            normalized_margin = 0.0  # Unknown outcome

        # Precompute phases
        phases = [decode_phase(s) for s in states]

        # Iterate over each move/state in the completed game
        original_count = 0
        augmented_count = 0

        for idx, (state, policy, value, phase, mn) in enumerate(zip(states, policies, values, phases, move_numbers)):
            # Fix #1: Use MCTS root value directly (position-specific, not game outcome)
            self.states.append(state)
            # MEMORY: Store policy as float16 to reduce buffer memory by 50%
            policy_f16 = np.asarray(policy, dtype=np.float16)
            self.move_probs.append(policy_f16)
            self.values.append(float(value))  # MCTS root value for this position
            self.phases.append(phase)
            self.move_numbers.append(int(mn))
            original_count += 1

            # Apply augmentation if enabled
            if self.enable_augmentation and self._augmenter is not None:
                try:
                    # Generate augmented samples (excluding original which was already added)
                    augmented = self._augmenter.augment(
                        state, policy, value, include_original=False
                    )

                    # Limit number of augmentations to save memory
                    if len(augmented) > self.max_augmentations - 1:
                        # Randomly sample subset
                        indices = random.sample(range(len(augmented)), self.max_augmentations - 1)
                        augmented = [augmented[i] for i in indices]

                    # Add augmented samples with same phase
                    for aug_state, aug_policy, aug_value in augmented:
                        self.states.append(aug_state)
                        # MEMORY: Store augmented policy as float16 too
                        aug_policy_f16 = np.asarray(aug_policy, dtype=np.float16)
                        self.move_probs.append(aug_policy_f16)
                        self.values.append(float(aug_value))
                        self.phases.append(phase)
                        # Augmented samples inherit move_number from the original
                        # — D2 transforms preserve the position's outer-game time.
                        self.move_numbers.append(int(mn))
                        augmented_count += 1

                except Exception as e:
                    logger.debug(f"Augmentation failed for state {idx}: {e}")

            # Track total states for memory monitoring
            self._total_states_added += 1

        # Update augmentation stats
        if self.enable_augmentation:
            self._augmentation_stats['total_original'] += original_count
            self._augmentation_stats['total_augmented'] += augmented_count

        # Log information about the replay buffer (debug level to avoid spam)
        aug_info = f", augmented={augmented_count}" if self.enable_augmentation else ""
        logger.debug(f"[Replay Buffer] Added game with scores W={final_white_score}, B={final_black_score}, "
                     f"margin={normalized_margin:.3f}, original={original_count}{aug_info}. Replay size={self.size()}")

        # Periodically check memory usage
        if self._total_states_added - self._last_memory_check > self._memory_check_interval:
            self._monitor_memory()
            self._last_memory_check = self._total_states_added

    def _monitor_memory(self):
        """Monitor memory usage and take action if needed."""
        try:
            import psutil
            import sys

            # Check overall process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            # Estimate buffer size
            if self.states:
                sample_state = self.states[0]
                sample_policy = self.move_probs[0]
                # Rough estimate of per-item memory
                state_size = sys.getsizeof(sample_state) + sample_state.nbytes
                policy_size = sys.getsizeof(sample_policy) + sample_policy.nbytes
                item_size = state_size + policy_size + 16  # Extra for values/phases
                buffer_mb = (item_size * len(self.states)) / (1024 * 1024)

                logger.debug(
                    f"[Memory Monitor] Process: {memory_mb:.1f}MB, Buffer: ~{buffer_mb:.1f}MB, States: {len(self.states)}")

                # If memory is getting high, force garbage collection
                # NOTE: Buffer reduction disabled - was causing training plateau by destroying data
                # The 6GB threshold was too aggressive for modern Macs with 16GB+ RAM
                # If memory issues occur, increase max_buffer_size in config instead
                if memory_mb > 12000:  # 12GB threshold (raised from 4GB)
                    # Memory pools handle cleanup automatically
                    pass

                    # Only reduce buffer in extreme cases (16GB+)
                    if memory_mb > 16000:  # 16GB threshold (raised from 6GB)
                        current_size = len(self.states)
                        target_size = max(5000, int(current_size * 0.85))  # Reduce by only 15% (was 30%)

                        # Create new smaller deques and keep newest data
                        start_idx = max(0, current_size - target_size)
                        self.states = deque(list(self.states)[start_idx:], maxlen=self.max_size)
                        self.move_probs = deque(list(self.move_probs)[start_idx:], maxlen=self.max_size)
                        self.values = deque(list(self.values)[start_idx:], maxlen=self.max_size)
                        self.phases = deque(list(self.phases)[start_idx:], maxlen=self.max_size)
                        self.move_numbers = deque(list(self.move_numbers)[start_idx:], maxlen=self.max_size)

                        logger.warning(f"[Memory Manager] Reduced buffer from {current_size} to {len(self.states)} states (extreme memory pressure)")
        except Exception as e:
            logger.debug(f"[Memory Monitor] Error checking memory: {e}")

    def save_buffer(self, path: str, compress: bool = True):
        """Save the replay buffer to disk with compression option."""
        import pickle

        save_data = {
            'states': list(self.states),
            'move_probs': list(self.move_probs),
            'values': list(self.values),
            'phases': list(self.phases),
            'move_numbers': list(self.move_numbers),
        }

        mode = 'wb'
        protocol = pickle.HIGHEST_PROTOCOL

        if compress:
            try:
                import gzip
                path = path if path.endswith('.gz') else path + '.gz'
                with gzip.open(path, mode) as f:
                    pickle.dump(save_data, f, protocol=protocol)
                logger.info(f"[Replay Buffer] Saved compressed buffer to {path}. Size: {self.size()}")
                return
            except ImportError:
                logger.warning("[Replay Buffer] gzip module not available, saving uncompressed")

        # Regular save if compression fails or is disabled
        with open(path, mode) as f:
            pickle.dump(save_data, f, protocol=protocol)
        logger.info(f"[Replay Buffer] Saved to {path}. Size: {self.size()}")

    def load_buffer(self, path: str):
        """Load the replay buffer from disk, handling compressed files."""
        import pickle

        try:
            # Try to detect gzip compression
            if path.endswith('.gz'):
                import gzip
                with gzip.open(path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Try normal file loading
                with open(path, 'rb') as f:
                    data = pickle.load(f)

            # Restore buffer data
            # BUGFIX: Use self.max_size instead of self.states.maxlen to respect new buffer size config
            self.states = deque(data.get('states', []), maxlen=self.max_size)
            self.move_probs = deque(data.get('move_probs', []), maxlen=self.max_size)
            self.values = deque(data.get('values', []), maxlen=self.max_size)
            loaded_phases = data.get('phases', None)

            # Handle phases if they exist
            if loaded_phases is None or len(loaded_phases) != len(self.states):
                default_phase = "MAIN_GAME"
                self.phases = deque([default_phase] * len(self.states), maxlen=self.max_size)
            else:
                self.phases = deque(loaded_phases, maxlen=self.max_size)

            # Backward compat: older buffers don't carry move_numbers. Default
            # to 0 so the search-consistency probe sees a single peak-noise
            # epsilon_mix value across all loaded samples — degraded but safe.
            loaded_move_numbers = data.get('move_numbers', None)
            if loaded_move_numbers is None or len(loaded_move_numbers) != len(self.states):
                self.move_numbers = deque([0] * len(self.states), maxlen=self.max_size)
            else:
                self.move_numbers = deque(loaded_move_numbers, maxlen=self.max_size)

            logger.info(f"[Replay Buffer] Loaded from {path}. Size: {self.size()}")

        except Exception as e:
            logger.error(f"[Replay Buffer] Failed to load from {path}: {e}")

    def size(self) -> int:
        """Return the current size of the replay buffer."""
        return len(self.states)

    def get_augmentation_stats(self) -> Dict[str, int]:
        """Get statistics about augmentation."""
        return {
            'enabled': self.enable_augmentation,
            'max_augmentations': self.max_augmentations,
            **self._augmentation_stats,
            'expansion_ratio': (
                (self._augmentation_stats['total_augmented'] + self._augmentation_stats['total_original']) /
                max(1, self._augmentation_stats['total_original'])
            ) if self._augmentation_stats['total_original'] > 0 else 1.0
        }

    def sample_batch(
            self,
            batch_size: int,
            phase_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of experiences, weighting certain phases more/less.
        """
        n = len(self.states)
        if n == 0:
            raise ValueError("Replay buffer is empty!")

        # Default weights if none are provided
        if phase_weights is None:
            phase_weights = {
                "RING_PLACEMENT": 1.0,
                "MAIN_GAME": 1.0,
                "RING_REMOVAL": 1.0
            }

        # Initialize equal probabilities for each sample
        p = np.ones(n, dtype=np.float32)  # Use float32 to save memory

        # Apply phase-based weighting
        for i, phase in enumerate(self.phases):
            p[i] *= phase_weights.get(phase, 1.0)

        # Normalize to make a proper probability distribution
        p = p / p.sum()

        # Sample without replacement (faster with numpy choice)
        indices = np.random.choice(n, batch_size, replace=False, p=p)

        # Create tensors from selected indices
        states = torch.stack([torch.from_numpy(self.states[i]).float() for i in indices])
        probs = torch.stack([torch.from_numpy(self.move_probs[i]).float() for i in indices])
        values = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)

        # Optional debug: Count phases in batch (debug level only)
        if logger.isEnabledFor(logging.DEBUG) and random.random() < 0.05:
            phase_counts = {"RING_PLACEMENT": 0, "MAIN_GAME": 0, "RING_REMOVAL": 0}
            for idx in indices:
                phase_counts[self.phases[idx]] += 1
            logger.debug(f"[Batch Stats] Phase distribution: {phase_counts}")

        return states, probs, values.unsqueeze(1)


class YinshTrainer:
    """Handles the training of the YINSH neural network."""

    def __init__(self,
                 network: NetworkWrapper,
                 device: Optional[str] = None,
                 batch_size: int = 256,
                 l2_reg: float = 0.0,
                 metrics_logger: Optional[MetricsLogger] = None,
                 value_head_lr_factor: float = 5.0,
                 value_loss_weights: Tuple[float, float] = (0.5, 0.5),
                 replay_buffer_path: Optional[str] = None,
                 max_buffer_size: int = 10000,
                 discrimination_weight: float = 0.5,
                 enable_augmentation: bool = False,
                 max_augmentations: int = 12,
                 ema_decay: Optional[float] = None,
                 soft_value_target_sigma: float = 0.5,
                 base_lr: float = 1e-3,
                 lr_schedule: str = 'cosine',
                 warmup_epochs: int = 0,
                 total_epochs: int = 200,
                 enable_autocast: bool = True,
                 enable_search_consistency: bool = False,
                 search_consistency_weight: float = 0.1,
                 search_consistency_value_weight: float = 1.0,
                 search_consistency_every_k_steps: int = 10,
                 search_consistency_long_sims: int = 64,
                 search_consistency_batch_size: int = 32,
                 search_consistency_warmup_iters: int = 3):
        """
        Initialize the trainer.

        Args:
            network: NetworkWrapper instance
            device: Device to train on ('cuda', 'mps', or 'cpu')
            l2_reg: L2 regularization coefficient
            batch_size: Batch size for training, passed from config via trainingsupervisor
            metrics_logger: Optional MetricsLogger instance
            value_head_lr_factor: Factor to multiply base lr for value head
            value_loss_weights: Weights for combining MSE and CE loss in value head
            replay_buffer_path: Path to save/load replay buffer
            max_buffer_size: Maximum number of samples in replay buffer (default 10000 = ~100 games)
            discrimination_weight: Weight for discrimination loss term (encourages value spread)
            enable_augmentation: If True, apply D6 symmetry augmentation to training data
            max_augmentations: Maximum augmented samples per original (1-12, default 12)
            ema_decay: If set, maintain an EMA shadow of network weights updated
                after every optimizer step. The supervisor swaps this copy in
                when writing the per-iteration checkpoint so the tournament
                eval target is the smoothed weights, not the single-step-noisy
                live weights. Typical values: 0.999 (AlphaZero-ish cadence).
                None disables the shadow entirely.
            soft_value_target_sigma: Standard deviation (in value-class widths)
                of the Gaussian smoothing applied to discrete value targets
                before cross-entropy. With 7 classes spanning score diffs
                {-3..+3}, σ=0.5 spreads ~38% of the mass onto adjacent classes,
                teaching the value head that "outcome +2 and outcome +3 are
                neighbors, not arbitrary labels." Set to 0 to use hard one-hot
                targets (old behavior) for A/B testing.
            base_lr: Base learning rate for the policy head's Adam optimizer.
                The value head uses `base_lr * value_head_lr_factor`. Previously
                the optimizer was hardcoded to lr=1e-3 at construction and the
                supervisor overwrote `param_groups[0]['lr']` post-init — which
                worked by accident for StepLR (captured base_lrs never mattered)
                but is incompatible with cosine decay (the scheduler grabs
                base_lrs at construction, so an overwrite would leave them
                pointing at the wrong base). Threading the LR through here
                keeps scheduler state coherent with actual optimizer LR.
            lr_schedule: 'cosine' (default) for linear-warmup + cosine-annealing
                down to zero across `total_epochs`; 'step' for the legacy
                StepLR(step_size=10, gamma=0.9). Cosine gives smooth decay with
                no cliffs; StepLR is kept for A/B.
            warmup_epochs: Linear warmup from 0.1·base_lr to base_lr over the
                first N epochs, then cosine decay takes over for the remaining
                `total_epochs - warmup_epochs`. 0 disables warmup entirely.
            total_epochs: Horizon for the cosine annealing curve. Usually
                `num_iterations × epochs_per_iteration`. The supervisor passes
                this in so the curve is scaled to the full training run, not to
                a single iteration's handful of epochs.
            enable_autocast: If True, wrap forward passes and loss computations
                in `torch.autocast(device_type=...)`. MPS uses bf16 (no grad
                scaler needed); CUDA also uses bf16 by default under autocast.
                CE-family losses are auto-promoted to fp32 internally so value-
                head gradient precision is preserved. CPU disables autocast
                regardless — bf16 ops on CPU don't win wall-clock back.
            enable_search_consistency: Track B §5 probe. When True, every K
                training batches the trainer runs long-search MCTS on a
                sampled batch of replay-buffer positions and trains the
                network to match the long-search policy + value targets
                (KL on policy, MSE on value). The hypothesis is that the
                0.104 discrimination plateau is a training-signal problem —
                if MCTS-distillation pushes past it, the AZ recipe is
                sufficient. Default False so existing training is unaffected.
            search_consistency_weight: Multiplier on the policy-distillation
                CE term (KL(π_long || softmax(network_logits))). Scales
                relative to the main policy/value losses.
            search_consistency_value_weight: Multiplier on the value-
                distillation MSE term ((v_long − v_pred)^2).
            search_consistency_every_k_steps: Apply the consistency step
                every K `train_step` calls. K=10 ≈ +30-50% wall-clock at
                K=10/batch=32/long_sims=64.
            search_consistency_long_sims: MCTS sim budget for the
                "teacher" search whose visit distribution + root value
                become the distillation target. Higher = stronger target,
                quadratically more expensive.
            search_consistency_batch_size: Number of replay-buffer
                positions sampled per consistency step. Each requires one
                long-search MCTS — the dominant cost of the probe.
            search_consistency_warmup_iters: Skip the consistency step
                until iteration N. Early iterations have a noisy network
                whose long-search outputs are themselves unreliable
                targets; warming up lets the policy stabilize first.
        """
        self.state_encoder = network.state_encoder
        self.batch_size = batch_size

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        self.network = network
        self.network.network = self.network.network.to(self.device)
        self.metrics_logger = metrics_logger

        # EMA shadow (optional). Must be built *after* the network is on the
        # target device so the shadow tensors live alongside the live weights —
        # otherwise `update()` would copy cross-device every step.
        self.ema: Optional[EMAShadow] = None
        if ema_decay is not None:
            self.ema = EMAShadow(self.network.network, decay=float(ema_decay))

        self.value_loss_weights = value_loss_weights
        self.value_head_lr_factor = value_head_lr_factor
        self.discrimination_weight = discrimination_weight
        self.soft_value_target_sigma = float(soft_value_target_sigma)
        # Search-consistency probe (Track B §5). All knobs read from config
        # via `mode_settings` in the supervisor. Off by default so existing
        # training is unaffected; flip `enable_search_consistency` to true to
        # arm the probe. The probe-side MCTS instance is built lazily on the
        # first call to `_search_consistency_step` because constructing it at
        # __init__ time would log MCTS init banners during pure-supervised
        # runs that never use it.
        self.enable_search_consistency = bool(enable_search_consistency)
        self.search_consistency_weight = float(search_consistency_weight)
        self.search_consistency_value_weight = float(search_consistency_value_weight)
        self.search_consistency_every_k_steps = int(search_consistency_every_k_steps)
        self.search_consistency_long_sims = int(search_consistency_long_sims)
        self.search_consistency_batch_size = int(search_consistency_batch_size)
        self.search_consistency_warmup_iters = int(search_consistency_warmup_iters)
        self._sc_mcts = None  # lazy
        self._sc_step_counter = 0
        self._sc_loss_history: list = []  # rolling, capped
        # LR schedule state. `_global_epoch` is incremented once per
        # `train_epoch()` and drives scheduler fast-forward on reinit.
        self.lr_schedule = lr_schedule
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self._base_policy_lr = float(base_lr)
        self._base_value_lr = float(base_lr) * float(value_head_lr_factor)
        self._global_epoch = 0

        # Autocast: bf16 forward + loss on MPS/CUDA. CE-family losses auto-
        # promote to fp32 under autocast, so value-head gradient precision is
        # preserved without a GradScaler. CPU disables — bf16 on CPU doesn't
        # buy wall-clock back. MPS autocast requires PyTorch 2.3+; on older
        # builds `torch.autocast(device_type='mps')` raises RuntimeError at
        # context construction. Probe and fall back gracefully rather than
        # crashing mid-training.
        self._autocast_device = self.device.type  # 'cuda' | 'mps' | 'cpu'
        self._autocast_enabled = bool(enable_autocast) and self._autocast_device in ('cuda', 'mps')
        if self._autocast_enabled:
            try:
                # Probe with the exact args `_autocast()` will use at train
                # time. bf16 is both safer (no GradScaler) and the only MPS
                # autocast dtype that was stable in our target PyTorch range.
                with torch.autocast(
                    device_type=self._autocast_device,
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    pass
            except RuntimeError as e:
                # This PyTorch build doesn't support autocast on this device.
                # Historical: PyTorch ≤2.2 + MPS rejected device_type='mps'
                # entirely. We now require ≥2.7.1 so this branch is mostly
                # belt-and-suspenders, but it still catches surprise builds
                # (e.g. custom wheels, WASM targets).
                self._autocast_enabled = False
                self._autocast_unsupported_reason = str(e)

        # Separate out parameters for policy vs. value heads
        value_params = [p for n, p in self.network.network.named_parameters()
                        if 'value_head' in n]
        policy_params = [p for n, p in self.network.network.named_parameters()
                         if 'value_head' not in n]

        self.policy_optimizer = optim.Adam(
            policy_params,
            lr=self._base_policy_lr,
            weight_decay=1e-4
        )

        self.value_optimizer = optim.SGD(
            value_params,
            lr=self._base_value_lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Initialize logger early for logging during initialization
        self.logger = logging.getLogger("YinshTrainer")
        self.logger.setLevel(logging.INFO)

        # Memory optimization: Cap replay buffer size to prevent unbounded growth
        # 10,000 samples = ~100 games at 100 moves each (reasonable for training)
        self.experience = GameExperience(
            max_size=max_buffer_size,
            enable_augmentation=enable_augmentation,
            max_augmentations=max_augmentations
        )
        self.logger.info(f"Replay buffer capped at {max_buffer_size:,} samples")
        if enable_augmentation:
            self.logger.info(f"Augmentation enabled: up to {max_augmentations}x data expansion")
        if replay_buffer_path is not None:
            from os import path as osp
            if osp.exists(replay_buffer_path):
                self.experience.load_buffer(replay_buffer_path)
            else:
                self.logger.info(f"[Replay Buffer] File '{replay_buffer_path}' not found. Starting with empty buffer.")

        # Schedulers: cosine-annealing with linear warmup by default; StepLR
        # kept as the A/B fallback. See `_build_schedulers` for the math and
        # for the reason we fast-forward on reinit instead of preserving
        # scheduler objects across optimizer reconstructions.
        self._build_schedulers(resume_epoch=0)
        self.logger.info(
            f"Schedulers: {self.lr_schedule} "
            f"(warmup_epochs={self.warmup_epochs}, total_epochs={self.total_epochs})"
        )
        self.logger.info(f"Initial LRs: Policy={self.policy_optimizer.param_groups[0]['lr']:.2e}, "
                         f"Value={self.value_optimizer.param_groups[0]['lr']:.2e}")
        if self._autocast_enabled:
            self.logger.info(
                f"Autocast: ENABLED (device_type={self._autocast_device}, dtype=bf16)"
            )
        elif getattr(self, '_autocast_unsupported_reason', None):
            self.logger.warning(
                f"Autocast: DISABLED — device_type={self._autocast_device!r} is not "
                f"supported by this PyTorch build ({self._autocast_unsupported_reason}). "
                f"Training will run in fp32. On MPS, upgrade to PyTorch ≥ 2.7 to enable."
            )
        else:
            self.logger.info(f"Autocast: DISABLED (device_type={self._autocast_device})")

        self.l2_reg = l2_reg
        # Logger already initialized earlier for use during __init__

        self.current_iteration = 0
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.learning_rates = {'policy': [], 'value': []}
        self.value_accuracies = []
        self.move_accuracies = []
        self.temperature = 1.0
        self.value_metrics = ValueHeadMetrics()

        self.metrics_logger = metrics_logger

    def _get_phase_weight(self, phase: str, iteration: int) -> float:
        """Get sampling weight for specific game phase.

        Args:
            phase: Game phase name
            iteration: Current training iteration

        Returns:
            Weight multiplier for this phase
        """
        if phase == "RING_REMOVAL":
            return 2.0  # Double weight for struggling phase
        elif phase == "RING_PLACEMENT":
            return 1.0 + iteration * 0.0125  # Keep existing schedule
        return 1.0  # Default weight for main game

    def _smooth_policy_targets(self, targets: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        n_classes = targets.shape[1]
        uniform = torch.ones_like(targets) / n_classes
        return (1 - epsilon) * targets + epsilon * uniform

    @staticmethod
    def _summarize_ece(diag: dict) -> Optional[str]:
        """Compute Expected Calibration Error from per-bin accumulators and
        render a compact reliability-diagram sparkline. Returns a one-line
        string like ``ECE=0.0421, reliability=▁▂▃▃▄▅▅▆▇█ bin_counts=[...]`` or
        None if no samples accumulated yet.

        The sparkline glyphs encode ``acc − conf`` per bin (well-calibrated
        bins render near the middle of the glyph range). A heavy-left /
        heavy-right sparkline means the head is over- / under-confident
        respectively."""
        bin_correct = diag.get('ece_bin_correct')
        bin_conf = diag.get('ece_bin_conf')
        bin_count = diag.get('ece_bin_count')
        if bin_correct is None or bin_count is None or bin_conf is None:
            return None
        total = bin_count.sum().item()
        if total <= 0:
            return None

        # ECE = weighted |acc - conf| across bins with nonzero count.
        ece = 0.0
        per_bin_gap = []  # (acc - conf) per bin; used for the sparkline.
        for b in range(bin_count.numel()):
            n = bin_count[b].item()
            if n > 0:
                acc = bin_correct[b].item() / n
                conf = bin_conf[b].item() / n
                ece += (n / total) * abs(acc - conf)
                per_bin_gap.append(acc - conf)
            else:
                per_bin_gap.append(None)

        # Sparkline: map (acc - conf) ∈ [-1, 1] onto 8 glyphs. Empty bins
        # render as a space so the reader sees the sparse regions.
        glyphs = '▁▂▃▄▅▆▇█'
        def _glyph(x):
            if x is None:
                return ' '
            # clamp + scale to [0, len(glyphs)-1]
            x = max(-1.0, min(1.0, x))
            idx = int((x + 1.0) / 2.0 * (len(glyphs) - 1))
            return glyphs[idx]
        spark = ''.join(_glyph(g) for g in per_bin_gap)
        counts = '[' + ','.join(str(int(bin_count[b].item())) for b in range(bin_count.numel())) + ']'
        return f"ECE={ece:.4f}, reliability={spark} bin_counts={counts}"

    def _autocast(self):
        """Autocast context manager. Always returns a fresh context (can't be
        reused across `with` blocks). When disabled — either by the
        `enable_autocast=False` flag, the CPU device gate, or the init-time
        probe failing on unsupported builds — this returns a real null-context
        so the `with self._autocast():` call path stays uniform. We can't lean
        on `torch.autocast(enabled=False, device_type=...)` here: older
        PyTorch versions (and MPS on ≤2.2) validate `device_type` *before*
        checking `enabled`, so the call raises on builds without MPS-autocast
        support even when enabled=False.

        Uses bf16 explicitly. MPS autocast's default is fp16 (PyTorch 2.7+),
        which without a `GradScaler` risks gradient underflow. bf16 has the
        same dynamic range as fp32 with a ~half-precision mantissa, so
        gradients don't underflow and we don't need GradScaler at all.
        """
        if not self._autocast_enabled:
            import contextlib
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self._autocast_device,
            dtype=torch.bfloat16,
            enabled=True,
        )

    def _build_schedulers(self, resume_epoch: int = 0) -> None:
        """(Re)construct policy + value schedulers against the current
        optimizers, then fast-forward to `resume_epoch` so LR state survives
        the per-iteration `_reinitialize_optimizers` reset in the supervisor.

        Optimizer LRs are reset to `_base_policy_lr` / `_base_value_lr` first
        because PyTorch schedulers capture `base_lrs` at construction time
        from `optimizer.param_groups[i]['lr']` — if the optimizer was already
        decayed, cosine would anchor its curve at the wrong peak. Stepping
        `resume_epoch` times forward puts us back where we were.
        """
        import warnings

        for pg in self.policy_optimizer.param_groups:
            pg['lr'] = self._base_policy_lr
        for pg in self.value_optimizer.param_groups:
            pg['lr'] = self._base_value_lr

        if self.lr_schedule == 'cosine':
            cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
            if self.warmup_epochs > 0:
                policy_warm = optim.lr_scheduler.LinearLR(
                    self.policy_optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                policy_cos = optim.lr_scheduler.CosineAnnealingLR(
                    self.policy_optimizer, T_max=cosine_epochs,
                )
                self.policy_scheduler = optim.lr_scheduler.SequentialLR(
                    self.policy_optimizer,
                    schedulers=[policy_warm, policy_cos],
                    milestones=[self.warmup_epochs],
                )
                value_warm = optim.lr_scheduler.LinearLR(
                    self.value_optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                value_cos = optim.lr_scheduler.CosineAnnealingLR(
                    self.value_optimizer, T_max=cosine_epochs,
                )
                self.value_scheduler = optim.lr_scheduler.SequentialLR(
                    self.value_optimizer,
                    schedulers=[value_warm, value_cos],
                    milestones=[self.warmup_epochs],
                )
            else:
                self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.policy_optimizer, T_max=self.total_epochs,
                )
                self.value_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.value_optimizer, T_max=self.total_epochs,
                )
        elif self.lr_schedule == 'step':
            # Legacy StepLR for A/B. Cliffs every 10 epochs at γ=0.9 per the
            # historical CyclicLR-instability note.
            self.policy_scheduler = optim.lr_scheduler.StepLR(
                self.policy_optimizer, step_size=10, gamma=0.9,
            )
            self.value_scheduler = optim.lr_scheduler.StepLR(
                self.value_optimizer, step_size=10, gamma=0.9,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule!r}")

        # Fast-forward. Suppress PyTorch's "step() called before optimizer.step()"
        # warning — we're rebuilding, not actually training.
        if resume_epoch > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(resume_epoch):
                    self.policy_scheduler.step()
                    self.value_scheduler.step()

    def train_step(self,
                   batch_size: int,
                   phase_weights: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, Dict]:
        """Training step with separate policy and value optimization."""
        if len(self.experience.states) < batch_size:
            return 0.0, 0.0, 0.0, {'top_1_accuracy': 0.0, 'top_3_accuracy': 0.0, 'top_5_accuracy': 0.0}

        self.network.network.train()

        # Pull a batch from replay
        states, target_probs, target_values = self.experience.sample_batch(
            batch_size,
            phase_weights=phase_weights  # <--- pass in your dictionary
        )

        # Debug print every 20 iterations (debug level)
        if self.current_iteration % 20 == 0 and self.logger.isEnabledFor(logging.DEBUG):
            device_before = states.device
            self.logger.debug(f"Training tensors device before transfer: {device_before}")

        states = states.to(self.device)
        target_probs = target_probs.to(self.device)
        target_values = target_values.to(self.device)
        target_values_flat = target_values.view(-1)

        # Debug print every 20 iterations (debug level)
        if self.current_iteration % 20 == 0 and self.logger.isEnabledFor(logging.DEBUG):
            device_after = states.device
            self.logger.debug(f"Training tensors device after transfer: {device_after}")
            self.logger.debug(f"Network device: {next(self.network.network.parameters()).device}")

        # Forward pass. Autocast wraps forward + loss so CE-family ops (log_softmax,
        # cross_entropy) auto-promote to fp32 internally and the heavy conv/linear
        # work runs in bf16. No GradScaler: MPS autocast uses bf16 (no underflow);
        # we also stay in bf16 on CUDA by default, which similarly doesn't need it.
        with self._autocast():
            pred_logits, pred_values = self.network.network(states)

        # Monitor / log value head metrics
        value_metrics = self._monitor_value_head(
            pred_values,
            target_values_flat,
            log_activations=(self.current_iteration % 10 == 0)
        )
        self._log_value_head_metrics(value_metrics)

        # MEMORY FIX: Clear value head activations after logging to prevent tensor accumulation
        # The activation hooks store tensors in a dict that grows without bound
        if hasattr(self.network.network, 'value_head_activations'):
            self.network.network.value_head_activations.clear()

        # MEMORY FIX: Disable per-sample evaluation logging to prevent tensor accumulation
        # This loop processes 256 samples per batch × 34 batches × 4 epochs = ~35K evaluations
        # Each evaluation holds tensor references that accumulate
        # Comment out to prevent memory leak - can re-enable for detailed debugging if needed
        # for i in range(states.size(0)):
        #     game_state = self.state_encoder.decode_state(states[i].cpu().numpy())
        #     self.value_metrics.record_evaluation(
        #         state=game_state,
        #         value_pred=pred_values[i].detach().cpu().numpy(),
        #         policy_probs=F.softmax(pred_logits[i], dim=-1).detach().cpu().numpy(),
        #         chosen_move=None,
        #         temperature=self.temperature,
        #         actual_outcome=target_values[i].detach().cpu().numpy()
        #     )

        # --- Policy Optimization ---
        self.policy_optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with self._autocast():
                scaled_logits = pred_logits / self.temperature
                log_probs = F.log_softmax(scaled_logits, dim=1)
                policy_loss = -(target_probs * log_probs).sum(dim=1).mean()

                # L2 regularization for policy
                if self.l2_reg > 0:
                    l2_loss = 0
                    for name, param in self.network.network.named_parameters():
                        if 'value_head' not in name:
                            l2_loss += torch.norm(param)
                    policy_loss = policy_loss + self.l2_reg * l2_loss

            policy_loss_val = float(policy_loss.item())  # for logging
            # Per-batch policy entropy. `log_probs` is fp32 under autocast (F.log_softmax
            # is on the fp32-promote list for CUDA; we `.float()`-cast elsewhere for MPS).
            # Low entropy means the policy has collapsed onto a narrow subset of moves;
            # `train_epoch` watches the epoch-over-epoch trend for sudden drops.
            with torch.no_grad():
                # Entropy H = -Σ p·log(p), averaged over the batch. Using log_probs
                # directly avoids recomputing softmax + log.
                policy_probs = log_probs.exp()
                batch_entropy = -(policy_probs * log_probs).sum(dim=1).mean()
                self.last_policy_entropy = float(batch_entropy.item())
            # Backward runs outside autocast — PyTorch autograd handles the
            # dtype bookkeeping via the saved-tensor pattern.
            policy_loss.backward()

            # Clip gradients
            policy_params = [p for n, p in self.network.network.named_parameters()
                             if 'value_head' not in n]
            torch.nn.utils.clip_grad_norm_(policy_params, max_norm=1.0)

            self.policy_optimizer.step()
            if self.ema is not None:
                self.ema.update()
            # NOTE: Scheduler stepping moved to train_epoch() - step once per epoch, not per batch

        # --- Value Optimization ---
        self.value_optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Recompute (because we did backward above). Autocast wraps the
            # second forward + its CE loss below — the full loss computation
            # is inside `with self._autocast():` at the top of the block.
            with self._autocast():
                _, pred_values = self.network.network(states)

            # Classification-based value head (AlphaZero approach)
            # Uses cross-entropy loss on discrete outcome distribution
            # This naturally encourages confident predictions and avoids MSE's variance minimization bias

            if hasattr(self.network.network, 'value_mode') and self.network.network.value_mode == 'classification':
                # Get the logits that were stored during forward pass
                if hasattr(self.network.network, '_value_logits'):
                    value_logits = self.network.network._value_logits

                    # Convert continuous target values to discrete class labels
                    # Map target ∈ [-1, 1] to class ∈ {0, 1, 2, 3, 4, 5, 6}
                    # For 7 classes representing: {-3, -2, -1, 0, +1, +2, +3} score differences
                    num_value_classes = self.network.network.num_value_classes
                    target_normalized = (target_values_flat + 1.0) / 2.0 * (num_value_classes - 1)
                    target_class = torch.round(target_normalized).long().clamp(0, num_value_classes - 1)

                    # CE math runs in fp32 — explicit `.float()` cast defends
                    # against autocast promotion rules differing by device:
                    # CUDA promotes `log_softmax` to fp32 automatically, MPS/CPU
                    # don't (the promotion list is per-device). Promoting at
                    # the boundary guarantees platform-independent precision for
                    # value-head gradients regardless of runtime hardware.
                    value_logits_fp32 = value_logits.float()
                    if self.soft_value_target_sigma > 0:
                        # Gaussian soft targets: mass centered on the (unrounded)
                        # target spreads onto neighboring classes. Encodes ordinal
                        # structure — outcome +2 is closer to +3 than to -2 — which
                        # hard one-hot CE throws away.
                        class_indices = torch.arange(
                            num_value_classes, device=value_logits.device, dtype=target_normalized.dtype
                        )
                        # dist_sq[i, k] = (k - target_normalized[i])^2
                        dist_sq = (class_indices.unsqueeze(0) - target_normalized.unsqueeze(1)) ** 2
                        target_dist = torch.exp(-dist_sq / (2.0 * self.soft_value_target_sigma ** 2))
                        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)
                        log_probs = F.log_softmax(value_logits_fp32, dim=1)
                        ce_loss = -(target_dist * log_probs).sum(dim=1).mean()
                    else:
                        # Hard one-hot CE (old behavior). Kept for A/B testing.
                        ce_loss = F.cross_entropy(value_logits_fp32, target_class)

                    # Track metrics for monitoring
                    batch_variance = torch.var(pred_values)

                    # FIXED DISCRIMINATION LOSS: Encourage confident (high absolute value) predictions
                    # Previous bug: Used batch variance (variance across positions)
                    # Fixed approach: Encourage high |value| predictions (confidence)
                    # Logic: Confident predictions (far from 0) naturally discriminate:
                    #   - Good moves → high positive values
                    #   - Bad moves → high negative values
                    #   - Uncertain/equal moves → near zero
                    # Loss: Penalize predictions close to zero (low confidence)
                    discrimination_weight = getattr(self, 'discrimination_weight', 0.5)

                    # Compute mean absolute value (measure of confidence)
                    mean_abs_value = torch.mean(torch.abs(pred_values))

                    # Loss: Maximize confidence (minimize its negative)
                    # We want high |values|, so minimize -mean_abs_value
                    discrimination_loss = -discrimination_weight * mean_abs_value

                    # Total value loss = classification accuracy + confidence incentive
                    value_loss = ce_loss + discrimination_loss

                    with torch.no_grad():
                        # Compute prediction accuracy for classification
                        pred_class = torch.argmax(value_logits, dim=-1)
                        value_accuracy = (pred_class == target_class).float().mean()

                        # DIAGNOSTIC FIX: Store real value head metrics for epoch summary
                        # These persist across batches and are summarized in train_epoch
                        ECE_N_BINS = 10
                        if not hasattr(self, '_epoch_value_diagnostics'):
                            self._epoch_value_diagnostics = {
                                'mean_abs_values': [],
                                'pred_class_counts': torch.zeros(self.network.network.num_value_classes),
                                'target_class_counts': torch.zeros(self.network.network.num_value_classes),
                                'batch_variances': [],
                                # ECE accumulators: per-bin sums of correctness, confidence,
                                # and count. Summed across batches; ECE computed at epoch end
                                # as Σ_b (n_b / N) · |acc_b − conf_b|.
                                'ece_bin_correct': torch.zeros(ECE_N_BINS),
                                'ece_bin_conf': torch.zeros(ECE_N_BINS),
                                'ece_bin_count': torch.zeros(ECE_N_BINS),
                            }

                        self._epoch_value_diagnostics['mean_abs_values'].append(mean_abs_value.item())
                        self._epoch_value_diagnostics['batch_variances'].append(batch_variance.item())

                        # Accumulate class histograms
                        for c in range(self.network.network.num_value_classes):
                            self._epoch_value_diagnostics['pred_class_counts'][c] += (pred_class == c).sum().item()
                            self._epoch_value_diagnostics['target_class_counts'][c] += (target_class == c).sum().item()

                        # ECE bin accumulation. confidence = max_softmax_prob per sample.
                        # Binning uses fp32 logits to match the CE-loss dtype (avoids
                        # bf16-precision drift in the probability tails that matter for
                        # calibration).
                        value_probs = F.softmax(value_logits_fp32, dim=1)
                        confidences, preds = value_probs.max(dim=1)
                        correct = (preds == target_class).float()
                        bin_idx = (confidences * ECE_N_BINS).long().clamp(0, ECE_N_BINS - 1)
                        for b in range(ECE_N_BINS):
                            mask = (bin_idx == b)
                            if mask.any():
                                self._epoch_value_diagnostics['ece_bin_correct'][b] += correct[mask].sum().cpu()
                                self._epoch_value_diagnostics['ece_bin_conf'][b] += confidences[mask].sum().cpu()
                                self._epoch_value_diagnostics['ece_bin_count'][b] += mask.sum().cpu()

                    # Log for diagnostics
                    if not hasattr(self, '_batch_counter'):
                        self._batch_counter = 0
                    self._batch_counter += 1
                    if self._batch_counter % 10 == 0:
                        self.logger.debug(f"Batch {self._batch_counter}: Confidence={mean_abs_value:.4f}, "
                                        f"CE={ce_loss:.4f}, Disc={discrimination_loss:.4f}, "
                                        f"Total={value_loss:.4f}, Accuracy={value_accuracy:.3f}")
                else:
                    # Fallback if logits not available
                    raise RuntimeError("Classification mode requires logits but none were found")

            else:
                # Legacy regression mode with MSE + variance penalty (kept for backward compatibility)
                value_loss_mse = F.mse_loss(pred_values.view(-1), target_values_flat)
                batch_variance = torch.var(pred_values)
                variance_weight = 1.5
                variance_penalty = variance_weight * torch.exp(-batch_variance * 10)
                value_loss = value_loss_mse + variance_penalty

                if not hasattr(self, '_batch_counter'):
                    self._batch_counter = 0
                self._batch_counter += 1
                if self._batch_counter % 10 == 0:
                    self.logger.debug(f"Batch {self._batch_counter}: Value variance={batch_variance:.4f}, "
                                    f"MSE={value_loss_mse:.4f}, Penalty={variance_penalty:.4f}, "
                                    f"Total={value_loss:.4f}")

            # OLD HYBRID LOSS (removed for Phase 1.5):
            # - Combined MSE (regression) + BCE (classification)
            # - Caused training/inference mismatch: trained for both objectives,
            #   but only regression used during play
            # - Also had sigmoid(tanh(x)) compression issue
            # value_loss_mse = F.mse_loss(pred_values, target_values)
            # target_outcomes = (target_values > 0).long()
            # value_probs = torch.sigmoid(pred_values)
            # value_loss_ce = F.binary_cross_entropy(value_probs, target_outcomes.float())
            # value_loss = (self.value_loss_weights[0] * value_loss_mse +
            #               self.value_loss_weights[1] * value_loss_ce)

            if self.l2_reg > 0:
                l2_loss = 0
                for name, param in self.network.network.named_parameters():
                    if 'value_head' in name:
                        l2_loss += torch.norm(param)
                value_loss = value_loss + (self.l2_reg * 2) * l2_loss

            value_loss_val = float(value_loss.item())

            # Store variance for epoch-level tracking
            self.last_value_variance = float(batch_variance.item())

            value_loss.backward()

            # Clip value head gradients
            value_params = [p for n, p in self.network.network.named_parameters()
                            if 'value_head' in n]
            torch.nn.utils.clip_grad_norm_(value_params, max_norm=0.5)

            self.value_optimizer.step()
            if self.ema is not None:
                self.ema.update()
            # NOTE: Scheduler stepping moved to train_epoch() - step once per epoch, not per batch

        # Debugging prints (debug level only)
        if self.current_iteration % 10 == 0 and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Value Head Gradients:")
            for name, param in self.network.network.named_parameters():
                if 'value_head' in name and param.grad is not None:
                    self.logger.debug(f"  {name}: {param.grad.data.norm(2).item():.6f}")

        # Metrics / logging
        with torch.no_grad():
            # Value accuracy computation depends on mode
            if hasattr(self.network.network, 'value_mode') and self.network.network.value_mode == 'classification':
                # For classification mode: use the accuracy already computed above
                if hasattr(self.network.network, '_value_logits'):
                    value_logits = self.network.network._value_logits
                    # Ensure target_values is 1D
                    target_values_flat = target_values.view(-1) if target_values.dim() > 1 else target_values
                    target_normalized = (target_values_flat + 1.0) / 2.0 * (self.network.network.num_value_classes - 1)
                    target_class = torch.round(target_normalized).long().clamp(0, self.network.network.num_value_classes - 1)
                    pred_class = torch.argmax(value_logits, dim=-1)
                    value_accuracy = (pred_class == target_class).float().mean().item()
                else:
                    value_accuracy = 0.0
            else:
                # For regression mode: binary accuracy (sign prediction)
                pred_outcomes = (pred_values.view(-1) > 0).float()
                true_outcomes = (target_values_flat > 0).float()
                value_accuracy = (pred_outcomes == true_outcomes).float().mean().item()

            pred_moves = torch.argmax(pred_logits, dim=1)
            true_moves = torch.argmax(target_probs, dim=1)
            top1_acc = (pred_moves == true_moves).float().mean().item()

            _, pred_top3 = torch.topk(pred_logits, k=3, dim=1)
            top3_acc = torch.any(pred_top3 == true_moves.unsqueeze(1), dim=1).float().mean().item()

            _, pred_top5 = torch.topk(pred_logits, k=5, dim=1)
            top5_acc = torch.any(pred_top5 == true_moves.unsqueeze(1), dim=1).float().mean().item()

            move_accuracies = {
                'top_1_accuracy': top1_acc,
                'top_3_accuracy': top3_acc,
                'top_5_accuracy': top5_acc
            }

            # MEMORY FIX: Track LR with bounded history
            MAX_LR_HISTORY = 100
            self.learning_rates['policy'].append(self.policy_optimizer.param_groups[0]['lr'])
            self.learning_rates['value'].append(self.value_optimizer.param_groups[0]['lr'])
            if len(self.learning_rates['policy']) > MAX_LR_HISTORY:
                self.learning_rates['policy'] = self.learning_rates['policy'][-MAX_LR_HISTORY:]
            if len(self.learning_rates['value']) > MAX_LR_HISTORY:
                self.learning_rates['value'] = self.learning_rates['value'][-MAX_LR_HISTORY:]

        self.current_iteration += 1

        # MEMORY FIX: Explicit tensor cleanup after training step
        # Move all tensors back to CPU and delete references to free GPU memory
        del states, target_probs, target_values, pred_logits, pred_values

        # MEMORY FIX: Clear stored tensors to prevent accumulation
        if hasattr(self.network.network, '_value_logits'):
            self.network.network._value_logits = None

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return (policy_loss_val, value_loss_val, value_accuracy, move_accuracies)

    def _build_consistency_mcts(self):
        """Lazily build the pure-neural MCTS used by the search-consistency
        probe. Subtree reuse is disabled (each sampled position is independent
        — there is no game continuity between samples). The MCTS shares the
        live network so its targets always reflect the current weights.
        """
        from .self_play import MCTS  # local import to avoid circular at module load
        return MCTS(
            network=self.network,
            evaluation_mode="pure_neural",
            heuristic_evaluator=None,
            num_simulations=self.search_consistency_long_sims,
            late_simulations=self.search_consistency_long_sims,
            simulation_switch_ply=10_000,  # never switch
            enable_subtree_reuse=False,
            # Disable root Dirichlet noise for distillation searches —
            # we want clean targets, not exploration noise.
            epsilon_mix_start=0.0,
            epsilon_mix_end=0.0,
            epsilon_mix_taper_moves=0,
            # Greedy temperature so visit-count → π is sharp, not flattened.
            initial_temp=1.0,
            final_temp=1.0,
            annealing_steps=1,
        )

    def _search_consistency_step(self) -> Optional[Dict[str, float]]:
        """One distillation step: long-search MCTS targets → network.

        For each sampled replay-buffer position:
          1. Decode tensor → GameState (move_number is recovered from the
             `move_numbers` deque, since `decode_state` can't reconstruct it).
          2. Run long-search MCTS with the live network as the leaf evaluator.
             Visit-distribution → π_long. `mcts.last_root_value` → v_long.
          3. Forward the network on the encoded state with grad enabled to
             produce (policy_logits, v_pred).
          4. Loss = λ_π · CE(π_long, softmax(logits)) + λ_v · MSE(v_long, v_pred).
             Both targets are detached — gradients flow only through the
             network's direct forward pass (the MCTS-internal evaluations are
             numpy and don't propagate). This is policy/value distillation
             from a stronger (longer-search) target into the network.

        Returns a small stats dict on a step taken, or None when skipped.
        """
        if not self.enable_search_consistency:
            return None
        if self.experience.size() < self.search_consistency_batch_size:
            return None
        if self.current_iteration < self.search_consistency_warmup_iters:
            return None

        if self._sc_mcts is None:
            self._sc_mcts = self._build_consistency_mcts()
        mcts = self._sc_mcts
        # Re-bind the MCTS's network reference to the *current* trainer
        # network on every step. The supervisor's `_reset_network_objects`
        # (every 3 iters at iter % 3 == 0, supervisor.py:1251-1252) swaps
        # `self.trainer.network` for a freshly-loaded NetworkWrapper —
        # without this rebind, the cached MCTS holds a stale reference to
        # the OLD wrapper whose tensor pool / device state may have been
        # torn down, and every `predict_from_state` raises. Also keep the
        # encoder in sync (cheap, defends against any future encoder swap).
        mcts.network = self.network
        mcts.state_encoder = self.network.state_encoder
        # Always start each consistency step with a fresh tree — sampled
        # positions have no game continuity.
        mcts.reset_tree()

        n = self.experience.size()
        sample_size = min(self.search_consistency_batch_size, n)
        indices = np.random.choice(n, size=sample_size, replace=False)

        # Force eval mode for the MCTS-distillation searches so the
        # subsequent training-mode forward (later in this method) is the
        # ONLY path that touches BN with `training=True`. The MCTS leaf
        # evaluator calls `predict()` which sets eval internally, but we
        # belt-and-suspenders here so any future code path that bypasses
        # `predict` still gets a clean eval-mode network for the search.
        self.network.network.eval()

        encoder = self.state_encoder
        target_pis = []
        target_vs = []
        states_for_forward = []
        skipped = 0
        # Per-cause counters so when nothing trains we can see why.
        skip_decode = 0
        skip_terminal = 0
        skip_search_exc = 0
        skip_degenerate_pi = 0

        # Wrap the whole search loop in no_grad so even if a code path inside
        # MCTS bypasses the no_grad blocks in `_evaluate_state` /
        # `predict_from_state` / `predict`, BN's train-mode batch-size-1 check
        # is still skipped. (PyTorch BN's check fires on `self.training`
        # alone — `is_grad_enabled()` is *not* part of it — so this no_grad
        # is purely about ensuring no autograd graph is built; the eval()
        # forcing above is what protects BN.)
        with torch.no_grad():
            for idx in indices:
                state_np = self.experience.states[idx]
                move_no = int(self.experience.move_numbers[idx])
                try:
                    game_state = encoder.decode_state(state_np)
                except Exception as e:
                    self.logger.debug(f"[SC] decode_state failed for idx={idx}: {e}")
                    skipped += 1
                    skip_decode += 1
                    continue
                if game_state.is_terminal():
                    # Terminal states have no MCTS distribution to learn — skip.
                    skipped += 1
                    skip_terminal += 1
                    continue
                try:
                    # Use `search_batch` (not `search`): it accumulates leaf
                    # evaluations and forwards them as a single
                    # `predict_batch(states_to_evaluate)` call. The serial
                    # `search` calls `predict_from_state(state)` per leaf which
                    # forwards a single-position tensor. With BatchNorm1d in
                    # the value head, a batch_size=1 forward in train mode
                    # raises ValueError. Even though `predict()` calls
                    # `eval()` first, the `nn.BatchNorm1d` at value_head[8]
                    # has been observed to fire the error from iter 1 onwards
                    # in real training runs — likely an interaction with
                    # MPS bf16 autocast state from the preceding train_step.
                    # Batched evaluation (batch_size > 1) bypasses the trap.
                    pi_long = mcts.search_batch(
                        game_state, move_number=move_no, batch_size=32
                    )
                except Exception as e:
                    self.logger.debug(f"[SC] MCTS search failed for idx={idx}: {e}")
                    skipped += 1
                    skip_search_exc += 1
                    # Capture the first exception's type+message for the all-skipped
                    # info log. Subsequent exceptions of the same step share this.
                    if not hasattr(self, '_sc_last_exc'):
                        self._sc_last_exc = ''
                    if not self._sc_last_exc:
                        self._sc_last_exc = f"{type(e).__name__}: {e}"
                    mcts.reset_tree()
                    continue
                v_long = float(getattr(mcts, 'last_root_value', 0.0))
                mcts.reset_tree()  # independent positions

                # MCTS returned a normalized visit distribution; defend against
                # the all-zero degenerate case (pre-existing MCTS warning path).
                if not np.isfinite(pi_long).all() or pi_long.sum() <= 0:
                    skipped += 1
                    skip_degenerate_pi += 1
                    continue

                target_pis.append(pi_long.astype(np.float32))
                target_vs.append(v_long)
                states_for_forward.append(state_np)

        if not states_for_forward:
            # Surface the breakdown so the per-epoch warning has actionable
            # context (which skip cause dominated). Single INFO line per
            # all-skipped step is fine — they're rare in healthy runs.
            exc_info = getattr(self, '_sc_last_exc', '')
            exc_suffix = f" first_exc=[{exc_info}]" if exc_info else ""
            self.logger.info(
                f"[SC] step skipped all {sample_size} samples "
                f"(decode={skip_decode}, terminal={skip_terminal}, "
                f"search_exc={skip_search_exc}, degenerate_pi={skip_degenerate_pi})"
                f"{exc_suffix}"
            )
            self._sc_last_exc = ''  # reset for next step
            return {'skipped': float(skipped), 'samples': 0.0}

        # Stack into batched tensors. State tensors come from the buffer
        # already encoded; forward expects fp32 on the trainer device.
        states_tensor = torch.stack(
            [torch.from_numpy(s).float() for s in states_for_forward]
        ).to(self.device)
        pi_target = torch.from_numpy(np.stack(target_pis)).to(self.device)  # detached by construction
        v_target = torch.tensor(target_vs, dtype=torch.float32, device=self.device)

        self.network.network.train()
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        with self._autocast():
            pred_logits, pred_values = self.network.network(states_tensor)
        # Promote CE math to fp32 (same pattern as the main value-head loss).
        log_probs = F.log_softmax(pred_logits.float(), dim=1)
        # Cross-entropy with the long-search distribution as soft target.
        # Equivalent to KL(π_long || softmax(logits)) up to a constant in π_long.
        policy_consistency_loss = -(pi_target * log_probs).sum(dim=1).mean()
        value_consistency_loss = F.mse_loss(pred_values.view(-1).float(), v_target)

        total = (
            self.search_consistency_weight * policy_consistency_loss
            + self.search_consistency_value_weight * value_consistency_loss
        )
        total.backward()

        # Reuse the same grad clipping bounds as the main loops.
        policy_params = [p for n, p in self.network.network.named_parameters()
                         if 'value_head' not in n]
        value_params = [p for n, p in self.network.network.named_parameters()
                        if 'value_head' in n]
        torch.nn.utils.clip_grad_norm_(policy_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(value_params, max_norm=0.5)

        self.policy_optimizer.step()
        self.value_optimizer.step()
        if self.ema is not None:
            self.ema.update()

        # Clear cached value logits to mirror the main train_step's hygiene.
        if hasattr(self.network.network, '_value_logits'):
            self.network.network._value_logits = None
        if hasattr(self.network.network, 'value_head_activations'):
            self.network.network.value_head_activations.clear()

        stats = {
            'samples': float(len(states_for_forward)),
            'skipped': float(skipped),
            'skip_decode': float(skip_decode),
            'skip_terminal': float(skip_terminal),
            'skip_search_exc': float(skip_search_exc),
            'skip_degenerate_pi': float(skip_degenerate_pi),
            'policy_loss': float(policy_consistency_loss.item()),
            'value_loss': float(value_consistency_loss.item()),
            'total_loss': float(total.item()),
        }
        # Keep a small rolling history for epoch-end logging.
        self._sc_loss_history.append(stats)
        if len(self._sc_loss_history) > 200:
            self._sc_loss_history = self._sc_loss_history[-200:]
        return stats

    def clear_training_state(self):
        """Clear accumulated training state to prevent memory leaks.

        Call this after training is complete to free memory from:
        - Loss history lists
        - Learning rate history
        - Value metrics accumulation
        """
        # Keep only recent history instead of clearing completely
        # This allows the supervisor to still report final stats
        MAX_KEEP = 10
        if len(self.policy_losses) > MAX_KEEP:
            self.policy_losses = self.policy_losses[-MAX_KEEP:]
        if len(self.value_losses) > MAX_KEEP:
            self.value_losses = self.value_losses[-MAX_KEEP:]
        if len(self.total_losses) > MAX_KEEP:
            self.total_losses = self.total_losses[-MAX_KEEP:]
        if len(self.value_accuracies) > MAX_KEEP:
            self.value_accuracies = self.value_accuracies[-MAX_KEEP:]

        # Clear learning rate history (can grow large)
        if len(self.learning_rates['policy']) > MAX_KEEP:
            self.learning_rates['policy'] = self.learning_rates['policy'][-MAX_KEEP:]
        if len(self.learning_rates['value']) > MAX_KEEP:
            self.learning_rates['value'] = self.learning_rates['value'][-MAX_KEEP:]

        # Clear optimizer state to free momentum/Adam buffers
        # They'll be rebuilt on next training
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)

        # Force garbage collection
        import gc
        gc.collect()

        # Clear GPU cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _monitor_value_head(self, pred_values: torch.Tensor,
                            target_values: torch.Tensor,
                            log_activations: bool = False) -> Dict:
        """Monitor value head performance and distributions."""
        with torch.no_grad():
            # Normalize to 1D to avoid broadcasting warnings ([B] vs [B, 1]).
            pred_values = pred_values.view(-1)
            target_values = target_values.view(-1)
            metrics = {}
            pre_tanh = pred_values

            metrics['pre_tanh'] = {
                'mean': float(pre_tanh.mean()),
                'std': float(pre_tanh.std()),
                'min': float(pre_tanh.min()),
                'max': float(pre_tanh.max()),
                'saturated_pct': float((torch.abs(pre_tanh) > 0.99).float().mean() * 100)
            }

            if log_activations:
                activations = self.network.network.value_head_activations
                for name, activation in activations.items():
                    metrics[f'layer_{name}'] = {
                        'mean': float(activation.mean()),
                        'std': float(activation.std()),
                        'zeros_pct': float((activation == 0).float().mean() * 100)
                    }

            value_confidence = torch.abs(pred_values)
            high_confidence = (value_confidence > 0.8).float().mean()
            metrics['predictions'] = {
                'mean_confidence': float(value_confidence.mean()),
                'high_confidence_pct': float(high_confidence * 100),
                'mean': float(pred_values.mean()),
                'std': float(pred_values.std())
            }

            metrics['targets'] = {
                'mean': float(target_values.mean()),
                'std': float(target_values.std()),
                'positive_pct': float((target_values > 0).float().mean() * 100)
            }

            pred_signs = torch.sign(pred_values)
            target_signs = torch.sign(target_values)
            metrics['alignment'] = {
                'sign_match_pct': float((pred_signs == target_signs).float().mean() * 100),
                'mse': float(F.mse_loss(pred_values, target_values)),
                'mae': float(F.l1_loss(pred_values, target_values))
            }

            return metrics

    def _log_value_head_metrics(self, metrics: Dict):
        """Log value head metrics. Only logs in debug mode to avoid output spam."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return  # Skip detailed logging unless in debug mode

        self.logger.debug("\nValue Head Analysis:")
        self.logger.debug("Pre-tanh Activations:")
        self.logger.debug(f"  Range: [{metrics['pre_tanh']['min']:.3f}, {metrics['pre_tanh']['max']:.3f}]")
        self.logger.debug(f"  Distribution: {metrics['pre_tanh']['mean']:.3f} ± {metrics['pre_tanh']['std']:.3f}")
        self.logger.debug(f"  Saturated: {metrics['pre_tanh']['saturated_pct']:.1f}%")

        self.logger.debug("\nPredictions:")
        self.logger.debug(f"  Confidence: {metrics['predictions']['mean_confidence']:.3f}")
        self.logger.debug(f"  High Confidence: {metrics['predictions']['high_confidence_pct']:.1f}%")
        self.logger.debug(f"  Distribution: {metrics['predictions']['mean']:.3f} ± {metrics['predictions']['std']:.3f}")

        self.logger.debug("\nAlignment with Targets:")
        self.logger.debug(f"  Sign Match: {metrics['alignment']['sign_match_pct']:.1f}%")
        self.logger.debug(f"  MSE: {metrics['alignment']['mse']:.3f}")
        self.logger.debug(f"  MAE: {metrics['alignment']['mae']:.3f}")

        if any(k.startswith('layer_') for k in metrics.keys()):
            self.logger.debug("\nLayer-wise Activations:")
            for name, layer_metrics in metrics.items():
                if name.startswith('layer_'):
                    self.logger.debug(f"  {name}:")
                    self.logger.debug(f"    Mean: {layer_metrics['mean']:.3f}")
                    self.logger.debug(f"    Std: {layer_metrics['std']:.3f}")
                    self.logger.debug(f"    Zeros: {layer_metrics['zeros_pct']:.1f}%")

    def train_epoch(self, batch_size: int, batches_per_epoch: int,
                    phase_weights: Optional[Dict[str, float]] = None) -> Dict:
        """Train for one epoch and return comprehensive stats.

        Args:
            batch_size: Number of samples per batch
            batches_per_epoch: Number of batches to train
            phase_weights: Optional dict mapping phase names to sampling weights
                          e.g., {'RING_PLACEMENT': 1.0, 'MAIN_GAME': 2.0, 'RING_REMOVAL': 0.5}
        """
        stats_accum = {
            'policy_loss': 0,
            'value_loss': 0,
            'value_accuracy': 0,
            'policy_entropy': 0,
            'move_accuracies': defaultdict(float),
            'gradient_norm': 0,
            'loss_improvement': 0
        }

        prev_total_loss = float('inf')
        current_policy_lr = float(self.policy_optimizer.param_groups[0]['lr'])
        current_value_lr = float(self.value_optimizer.param_groups[0]['lr'])

        stats_accum['learning_rates'] = {
            'policy': current_policy_lr,
            'value': current_value_lr
        }

        # Track best value accuracy for early stopping
        self.best_value_accuracy = getattr(self, 'best_value_accuracy', 0.0)
        self.value_accuracy_patience = getattr(self, 'value_accuracy_patience', 0)
        self.current_iteration = getattr(self, 'current_iteration', 0) + 1

        # Keep track of value head metrics for this epoch
        value_confidences = []
        value_sign_matches = []
        value_variances = []  # Track variance to monitor collapse

        # Use provided phase_weights or defaults
        if phase_weights is None:
            phase_weights = {
                "RING_PLACEMENT": 1.0,
                "MAIN_GAME": 1.0,
                "RING_REMOVAL": 1.0
            }

        # Snapshot consistency-loss history length so we can report just this
        # epoch's contribution at the bottom (and not stale stats across runs).
        sc_history_start = len(self._sc_loss_history)
        sc_attempts_this_epoch = 0  # gate fired (regardless of internal skips)

        for batch in range(batches_per_epoch):
            p_loss, v_loss, v_acc, move_accs = self.train_step(
                batch_size,
                phase_weights=phase_weights
            )

            # Search-consistency probe (Track B §5). Gated by `enable_search_
            # consistency` and a warmup-iter check inside the step itself; this
            # block is just the every-K-batches cadence. Counter advances on
            # every batch (not just on enabled steps) so disabling/re-enabling
            # at runtime doesn't reset the cadence.
            self._sc_step_counter += 1
            if (
                self.enable_search_consistency
                and self._sc_step_counter % max(1, self.search_consistency_every_k_steps) == 0
            ):
                sc_attempts_this_epoch += 1
                self._search_consistency_step()

            # Track value head metrics
            value_confidences.append(self._get_current_value_confidence())
            value_sign_matches.append(v_acc)

            # Track variance (from last train_step)
            if hasattr(self, 'last_value_variance'):
                value_variances.append(self.last_value_variance)

            self.logger.debug(f"Batch {batch} losses - Policy: {p_loss:.4f}, Value: {v_loss:.4f}")

            stats_accum['policy_loss'] += float(p_loss)
            stats_accum['value_loss'] += float(v_loss)
            stats_accum['value_accuracy'] += float(v_acc)

            for k, v in move_accs.items():
                stats_accum['move_accuracies'][k] += v

            # Calculate gradient norm
            total_norm = 0.0
            for p in self.network.network.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            stats_accum['gradient_norm'] += np.sqrt(total_norm)

            if hasattr(self, 'last_policy_entropy'):
                stats_accum['policy_entropy'] += self.last_policy_entropy

        # Average stats
        for key in ['policy_loss', 'value_loss', 'value_accuracy', 'policy_entropy', 'gradient_norm']:
            stats_accum[key] /= batches_per_epoch

        for k in stats_accum['move_accuracies']:
            stats_accum['move_accuracies'][k] /= batches_per_epoch

        # Add average value confidence and variance
        stats_accum['value_confidence'] = np.mean(value_confidences) if value_confidences else 0.0
        stats_accum['value_variance'] = np.mean(value_variances) if value_variances else 0.0

        # MEMORY FIX: Track losses for supervisor reporting
        # Keep only recent history to prevent unbounded growth
        # These lists grow with every epoch: 4 epochs/iter × many iterations = leak
        # Solution: Keep only last 20 epochs worth of history
        MAX_HISTORY = 20
        self.policy_losses.append(stats_accum['policy_loss'])
        self.value_losses.append(stats_accum['value_loss'])
        if len(self.policy_losses) > MAX_HISTORY:
            self.policy_losses = self.policy_losses[-MAX_HISTORY:]
        if len(self.value_losses) > MAX_HISTORY:
            self.value_losses = self.value_losses[-MAX_HISTORY:]

        # Track value head improvement with LR floor to prevent collapse lock-in
        VALUE_LR_FLOOR = 1e-4  # Minimum value LR to prevent decay to zero
        current_value_accuracy = stats_accum['value_accuracy']
        if current_value_accuracy > self.best_value_accuracy:
            self.best_value_accuracy = current_value_accuracy
            self.value_accuracy_patience = 0
            self.logger.info(f"New best value accuracy: {current_value_accuracy:.2%}")
        else:
            self.value_accuracy_patience += 1
            # If accuracy hasn't improved for several epochs, adjust learning rate (with floor)
            if self.value_accuracy_patience >= 3:
                current_lr = self.value_optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * 0.7, VALUE_LR_FLOOR)
                if new_lr < current_lr:
                    self.logger.warning(f"Value accuracy not improving for {self.value_accuracy_patience} epochs. "
                                        f"Reducing value LR: {current_value_lr:.2e} -> {new_lr:.2e}")
                    for param_group in self.value_optimizer.param_groups:
                        param_group['lr'] = new_lr
                else:
                    self.logger.info(f"Value LR at floor ({VALUE_LR_FLOOR:.0e}), not reducing further")
                # Reset patience
                self.value_accuracy_patience = 0

        current_total_loss = stats_accum['policy_loss'] + stats_accum['value_loss']
        if prev_total_loss != float('inf') and prev_total_loss != 0:
            stats_accum['loss_improvement'] = (prev_total_loss - current_total_loss) / (prev_total_loss + 1e-8)
        else:
            stats_accum['loss_improvement'] = 0.0

        metrics = EpochMetrics(
            policy_loss=max(stats_accum['policy_loss'], 1e-8),
            value_loss=max(stats_accum['value_loss'], 1e-8),
            value_accuracy=stats_accum['value_accuracy'],
            move_accuracies=dict(stats_accum['move_accuracies']),
            learning_rates={
                'policy': current_policy_lr,
                'value': current_value_lr
            },
            gradient_norm=max(stats_accum['gradient_norm'], 1e-8),
            loss_improvement=stats_accum['loss_improvement']
        )

        if self.metrics_logger is not None:
            self.metrics_logger.log_training(metrics)

        self.logger.info(
            f"\n{'=' * 20} Epoch Summary {'=' * 20}\n"
            f"Policy: loss={metrics.policy_loss:.4f}, lr={metrics.learning_rates['policy']:.2e}\n"
            f"Value:  loss={metrics.value_loss:.4f}, acc={metrics.value_accuracy:.2%}, "
            f"conf={stats_accum['value_confidence']:.3f}, var={stats_accum['value_variance']:.4f}, "  # Added variance
            f"lr={metrics.learning_rates['value']:.2e}\n"
            f"Moves:  acc={metrics.move_accuracies['top_1_accuracy']:.2%}, "
            f"top3={metrics.move_accuracies['top_3_accuracy']:.2%}\n"
            f"Grad norm: {metrics.gradient_norm:.2e}, Loss improvement: {metrics.loss_improvement:.2%}\n"
            f"{'=' * 50}"
        )

        # DIAGNOSTIC: Log value head class distribution histograms
        if hasattr(self, '_epoch_value_diagnostics') and self._epoch_value_diagnostics:
            diag = self._epoch_value_diagnostics
            num_classes = len(diag['pred_class_counts'])

            # Normalize to percentages
            pred_total = diag['pred_class_counts'].sum().item()
            target_total = diag['target_class_counts'].sum().item()

            if pred_total > 0 and target_total > 0:
                pred_pct = [f"{100*diag['pred_class_counts'][i].item()/pred_total:.1f}" for i in range(num_classes)]
                target_pct = [f"{100*diag['target_class_counts'][i].item()/target_total:.1f}" for i in range(num_classes)]

                # Mean confidence (mean abs value)
                mean_conf = np.mean(diag['mean_abs_values']) if diag['mean_abs_values'] else 0.0

                # `mean_abs_value` is the value-head "discrimination" metric
                # tracked in RESEARCH_LOG.md (0.104 ceiling). Logged with the
                # explicit `discrimination=` alias so the §5 search-consistency
                # bake-off can grep for plateau-break events without parsing a
                # second name. The two are identical by construction.
                self.logger.info(
                    f"[Value Diagnostics] discrimination={mean_conf:.4f} "
                    f"(mean_abs_value={mean_conf:.4f})\n"
                    f"  Target class %: [{', '.join(target_pct)}] (classes 0-{num_classes-1})\n"
                    f"  Pred class %:   [{', '.join(pred_pct)}]"
                )

            # Expected Calibration Error + reliability-diagram sparkline.
            # ECE = Σ_b (n_b / N) · |acc_b − conf_b|. Sparkline shows per-bin
            # (acc − conf) as a compact glyph so drift is visible at a glance.
            if 'ece_bin_count' in diag:
                ece_report = self._summarize_ece(diag)
                if ece_report is not None:
                    self.logger.info(f"[Value Diagnostics] {ece_report}")

            # Reset diagnostics for next epoch
            self._epoch_value_diagnostics = {
                'mean_abs_values': [],
                'pred_class_counts': torch.zeros(num_classes),
                'target_class_counts': torch.zeros(num_classes),
                'batch_variances': [],
                'ece_bin_correct': torch.zeros(diag['ece_bin_count'].numel() if 'ece_bin_count' in diag else 10),
                'ece_bin_conf': torch.zeros(diag['ece_bin_count'].numel() if 'ece_bin_count' in diag else 10),
                'ece_bin_count': torch.zeros(diag['ece_bin_count'].numel() if 'ece_bin_count' in diag else 10),
            }

        # Policy entropy + collapse detection. Low entropy = policy has collapsed
        # onto a narrow subset of moves — a known failure mode of this codebase.
        # Track a rolling history, flag sudden drops >50% below the 3-epoch mean.
        current_entropy = stats_accum.get('policy_entropy', 0.0)
        if current_entropy > 0:
            self._policy_entropy_history = getattr(self, '_policy_entropy_history', [])
            if len(self._policy_entropy_history) >= 3:
                recent_mean = float(np.mean(self._policy_entropy_history[-3:]))
                if recent_mean > 0.1 and current_entropy < 0.5 * recent_mean:
                    self.logger.warning(
                        f"[Value Diagnostics] POLICY ENTROPY COLLAPSE: "
                        f"{current_entropy:.3f} < 0.5 × 3-epoch avg ({recent_mean:.3f})"
                    )
            self._policy_entropy_history.append(current_entropy)
            # Bound memory: keep most-recent 20 epochs.
            self._policy_entropy_history = self._policy_entropy_history[-20:]
            self.logger.info(
                f"[Value Diagnostics] policy_entropy={current_entropy:.3f} "
                f"(rolling N={len(self._policy_entropy_history)})"
            )

        # Search-consistency probe summary for this epoch. We log the per-
        # epoch averages of the policy/value distillation losses so the
        # discrimination-plateau hypothesis can be evaluated against the
        # standard `[Value Diagnostics]` lines side-by-side.
        if self.enable_search_consistency and len(self._sc_loss_history) > sc_history_start:
            this_epoch = self._sc_loss_history[sc_history_start:]
            steps = len(this_epoch)
            avg_p = float(np.mean([s['policy_loss'] for s in this_epoch]))
            avg_v = float(np.mean([s['value_loss'] for s in this_epoch]))
            avg_total = float(np.mean([s['total_loss'] for s in this_epoch]))
            avg_samples = float(np.mean([s['samples'] for s in this_epoch]))
            avg_skipped = float(np.mean([s['skipped'] for s in this_epoch]))
            self.logger.info(
                f"[SearchConsistency] steps={steps} "
                f"policy_distill={avg_p:.4f} value_distill={avg_v:.4f} "
                f"total={avg_total:.4f} samples/step={avg_samples:.1f} "
                f"skipped/step={avg_skipped:.2f}"
            )
        elif self.enable_search_consistency:
            # Probe armed but produced no logged steps this epoch. Three
            # distinct reasons — surface them so a config bug doesn't hide
            # behind the same diagnostic as legitimate warmup behavior.
            if sc_attempts_this_epoch == 0:
                # Gate cadence didn't fire — usually short-epoch + large k.
                reason = "every_k_steps gate not hit this epoch"
            elif self.current_iteration < self.search_consistency_warmup_iters:
                reason = f"warmup ({self.current_iteration}/{self.search_consistency_warmup_iters} iters)"
            elif self.experience.size() < self.search_consistency_batch_size:
                reason = (
                    f"buffer too small "
                    f"({self.experience.size()}<{self.search_consistency_batch_size})"
                )
            else:
                # Gate hit, not in warmup, buffer big enough — but every
                # attempt skipped all positions internally (decode_state
                # failures or all-terminal samples). Bump severity because
                # this means the loss is silently no-op'ing.
                self.logger.warning(
                    f"[SearchConsistency] step fired {sc_attempts_this_epoch}× "
                    f"but every attempt produced 0 distillation samples — "
                    f"all positions skipped (terminal or decode_state failed). "
                    f"Re-enable DEBUG logging to inspect the per-position skip cause."
                )
                reason = None
            if reason is not None:
                self.logger.info(f"[SearchConsistency] no steps this epoch ({reason})")

        # Step schedulers once per epoch (not per batch!)
        # This ensures LR decay happens at a reasonable rate
        self.policy_scheduler.step()
        self.value_scheduler.step()
        # Track global epoch across the whole training run so the supervisor
        # can rebuild cosine state correctly after each per-iteration optimizer
        # reset (see `_build_schedulers`).
        self._global_epoch += 1

        return vars(metrics)

    def _get_current_value_confidence(self) -> float:
        """Get the current average value head confidence from real batch data.

        Returns mean(abs(pred_values)) averaged across recent batches.
        This measures how far from 0 the value predictions are (decisiveness).
        """
        if not hasattr(self, '_epoch_value_diagnostics'):
            return 0.0

        mean_abs_values = self._epoch_value_diagnostics.get('mean_abs_values', [])
        if not mean_abs_values:
            return 0.0

        return float(np.mean(mean_abs_values))

    def save_checkpoint(self, path: str, epoch: int):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.network.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),  # or separate if needed
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'total_losses': self.total_losses
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.network.load_state_dict(checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # If you have separate value_optimizer, load that too
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.total_losses = checkpoint.get('total_losses', [])
        self.logger.info(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']

    def evaluate_position(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate a single position."""
        self.network.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.network.network(state_tensor)
            return (
                torch.softmax(policy, dim=1).squeeze().cpu().numpy(),
                value.item()
            )

    def add_game_experience(self, states: list, policies: list, values: list,
                            final_white_score: int = None, final_black_score: int = None):
        """
        Delegating method for backward compatibility.

        Fix #1: Updated to accept MCTS root values instead of game outcome.

        Args:
            states:   List of game state numpy arrays.
            policies: List of move probability distributions.
            values:   List of MCTS root values (position-specific training targets).
            final_white_score: (Optional) White's final score - only used for logging.
            final_black_score: (Optional) Black's final score - only used for logging.
        """
        self.experience.add_game_experience(
            states,
            policies,
            values,
            final_white_score,
            final_black_score
        )

    def _calculate_accuracies(self,
                              pred_values: torch.Tensor,
                              target_values: torch.Tensor,
                              pred_logits: torch.Tensor,
                              target_probs: torch.Tensor) -> Dict:
        """
        Calculate value and move prediction accuracies.
        """
        with torch.no_grad():
            pred_outcomes = (pred_values > 0).float()
            true_outcomes = (target_values > 0).float()
            value_accuracy = float((pred_outcomes == true_outcomes).float().mean().item())

            pred_moves = torch.argmax(pred_logits, dim=1)
            true_moves = torch.argmax(target_probs, dim=1)
            move_accuracy = float((pred_moves == true_moves).float().mean().item())

            k = 5
            _, pred_top_k = torch.topk(pred_logits, k, dim=1)
            _, true_top_k = torch.topk(target_probs, k, dim=1)
            top_k_accuracy = float(torch.any(
                pred_top_k == true_moves.unsqueeze(1),
                dim=1
            ).float().mean().item())

            return {
                'value_accuracy': value_accuracy,
                'move_accuracy': move_accuracy,
                'top_k_accuracy': top_k_accuracy
            }
