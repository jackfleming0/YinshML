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
                            discount_factor: float = 1.0):
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
        """
        # Safety check
        assert len(states) == len(policies), "Mismatch: states vs. policies!"
        assert len(states) == len(values), "Mismatch: states vs. values!"

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

        for idx, (state, policy, value, phase) in enumerate(zip(states, policies, values, phases)):
            # Fix #1: Use MCTS root value directly (position-specific, not game outcome)
            self.states.append(state)
            # MEMORY: Store policy as float16 to reduce buffer memory by 50%
            policy_f16 = np.asarray(policy, dtype=np.float16)
            self.move_probs.append(policy_f16)
            self.values.append(float(value))  # MCTS root value for this position
            self.phases.append(phase)
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
            'phases': list(self.phases)
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
                 max_augmentations: int = 12):
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

        self.value_loss_weights = value_loss_weights
        self.value_head_lr_factor = value_head_lr_factor
        self.discrimination_weight = discrimination_weight

        # Separate out parameters for policy vs. value heads
        value_params = [p for n, p in self.network.network.named_parameters()
                        if 'value_head' in n]
        policy_params = [p for n, p in self.network.network.named_parameters()
                         if 'value_head' not in n]

        self.policy_optimizer = optim.Adam(
            policy_params,
            lr=0.001,
            weight_decay=1e-4
        )

        self.value_optimizer = optim.SGD(
            value_params,
            lr=0.0001 * value_head_lr_factor,
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

        # Schedulers: Simple StepLR to avoid LR instability
        # Previous CyclicLR caused dramatic LR drops (1e-3 → 1e-5) causing training instability
        # StepLR provides gentle, predictable decay
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer,
            step_size=10,  # Decay every 10 epochs
            gamma=0.9  # Multiply LR by 0.9
        )

        self.value_scheduler = optim.lr_scheduler.StepLR(
            self.value_optimizer,
            step_size=10,
            gamma=0.9
        )

        self.logger.info(f"Schedulers: StepLR (step_size=10, gamma=0.9)")
        self.logger.info(f"Initial LRs: Policy={self.policy_optimizer.param_groups[0]['lr']:.2e}, "
                         f"Value={self.value_optimizer.param_groups[0]['lr']:.2e}")

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

        # Forward pass
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
            # CRITICAL FIX: Remove retain_graph=True to prevent memory leak
            # The graph is no longer needed after this backward pass
            policy_loss.backward()

            # Clip gradients
            policy_params = [p for n, p in self.network.network.named_parameters()
                             if 'value_head' not in n]
            torch.nn.utils.clip_grad_norm_(policy_params, max_norm=1.0)

            self.policy_optimizer.step()
            # NOTE: Scheduler stepping moved to train_epoch() - step once per epoch, not per batch

        # --- Value Optimization ---
        self.value_optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Recompute (because we did backward above)
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
                    target_normalized = (target_values_flat + 1.0) / 2.0 * (self.network.network.num_value_classes - 1)
                    target_class = torch.round(target_normalized).long().clamp(0, self.network.network.num_value_classes - 1)

                    # Cross-entropy loss encourages confident predictions
                    ce_loss = F.cross_entropy(value_logits, target_class)

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
                        if not hasattr(self, '_epoch_value_diagnostics'):
                            self._epoch_value_diagnostics = {
                                'mean_abs_values': [],
                                'pred_class_counts': torch.zeros(self.network.network.num_value_classes),
                                'target_class_counts': torch.zeros(self.network.network.num_value_classes),
                                'batch_variances': []
                            }

                        self._epoch_value_diagnostics['mean_abs_values'].append(mean_abs_value.item())
                        self._epoch_value_diagnostics['batch_variances'].append(batch_variance.item())

                        # Accumulate class histograms
                        for c in range(self.network.network.num_value_classes):
                            self._epoch_value_diagnostics['pred_class_counts'][c] += (pred_class == c).sum().item()
                            self._epoch_value_diagnostics['target_class_counts'][c] += (target_class == c).sum().item()

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

        for batch in range(batches_per_epoch):
            p_loss, v_loss, v_acc, move_accs = self.train_step(
                batch_size,
                phase_weights=phase_weights
            )

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

                self.logger.info(
                    f"[Value Diagnostics] mean_abs_value={mean_conf:.4f}\n"
                    f"  Target class %: [{', '.join(target_pct)}] (classes 0-{num_classes-1})\n"
                    f"  Pred class %:   [{', '.join(pred_pct)}]"
                )

            # Reset diagnostics for next epoch
            self._epoch_value_diagnostics = {
                'mean_abs_values': [],
                'pred_class_counts': torch.zeros(num_classes),
                'target_class_counts': torch.zeros(num_classes),
                'batch_variances': []
            }

        # Step schedulers once per epoch (not per batch!)
        # This ensures LR decay happens at a reasonable rate
        self.policy_scheduler.step()
        self.value_scheduler.step()

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
