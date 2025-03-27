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
    """Stores game states and outcomes for training, with persistence support."""

    def __init__(self, max_size: int = 100000):
        self.states = deque(maxlen=max_size)
        self.move_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.phases = deque(maxlen=max_size)

    def add_game_experience(self,
                            states: list,
                            policies: list,
                            final_white_score: int,
                            final_black_score: int,
                            discount_factor: float = 1.0):
        """
        Add a completed game to the replay buffer using a continuous
        margin-based value in [-1, +1] normalized by max score = 3.

        Args:
            states:             List of game states (numpy arrays).
            policies:           List of move probability distributions.
            final_white_score:  White's final score (0 to 3).
            final_black_score:  Black's final score (0 to 3).
            discount_factor:    Optional discount factor, e.g. 0.99, applied per move
                                (from the end of the game backward). Default = 1.0 (no discount).
        """
        # Safety check
        assert len(states) == len(policies), "Mismatch: states vs. policies!"

        # Helper to decode phases from channel 5, if needed
        def decode_phase(state: np.ndarray) -> str:
            phase_channel = state[5]  # 5th channel for 'GAME_PHASE'
            # Use absolute value to ignore current player sign.
            avg = np.mean(np.abs(phase_channel))
            # Adjust thresholds to match your observed distribution.
            if avg < 0.2:
                phase = "RING_PLACEMENT"
            elif avg < 0.6:
                phase = "MAIN_GAME"
            else:
                phase = "RING_REMOVAL"
            print(f"[DEBUG] decode_phase: avg={avg:.3f}")
            print(f"[DEBUG] Classified phase: {phase}")
            return phase

        # Compute the normalized margin in [-1, +1]
        score_diff = final_white_score - final_black_score  # e.g. +2 if 3–1
        normalized_margin = score_diff / 3.0                # e.g. +0.666..., range is [-1, +1]

        # Precompute phases
        phases = [decode_phase(s) for s in states]
        game_length = len(states)

        # Iterate over each move/state in the completed game
        for idx, (state, policy, phase) in enumerate(zip(states, policies, phases)):
            # If you want to discount earlier moves more heavily, you can do:
            # (game_length - 1 - idx) as the exponent for discount_factor
            discounted_value = normalized_margin * (discount_factor ** (game_length - 1 - idx))

            self.states.append(state)
            self.move_probs.append(policy)
            self.values.append(discounted_value)
            self.phases.append(phase)

        print(f"[Replay Buffer] Added game with final scores W={final_white_score}, B={final_black_score}, "
              f"margin={normalized_margin:.3f}. Replay size={self.size()}")

    def save_buffer(self, path: str):
        """Save the replay buffer to disk using pickle."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'states': list(self.states),
                'move_probs': list(self.move_probs),
                'values': list(self.values),
                'phases': list(self.phases)  # Save phases along with other data
            }, f)
        print(f"[Replay Buffer] Saved to {path}. Current size: {self.size()}")

    def load_buffer(self, path: str):
        """Load the replay buffer from disk using pickle."""
        import pickle
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.states = deque(data.get('states', []), maxlen=self.states.maxlen)
            self.move_probs = deque(data.get('move_probs', []), maxlen=self.move_probs.maxlen)
            self.values = deque(data.get('values', []), maxlen=self.values.maxlen)
            # If phases are not present, default to "MAIN_GAME" for each state.
            loaded_phases = data.get('phases', None)
            if loaded_phases is None or len(loaded_phases) != len(self.states):
                default_phase = "MAIN_GAME"
                self.phases = deque([default_phase] * len(self.states), maxlen=self.states.maxlen)
            else:
                self.phases = deque(loaded_phases, maxlen=self.states.maxlen)
            print(f"[Replay Buffer] Loaded from {path}. Current size: {self.size()}")
        except Exception as e:
            print(f"[Replay Buffer] Failed to load from {path}: {e}")

    def size(self) -> int:
        """Return the current size of the replay buffer."""
        return len(self.states)

    def sample_batch(
            self,
            batch_size: int,
            phase_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of experiences, weighting certain phases more/less.

        Args:
            batch_size: Number of samples to draw.
            phase_weights: Dictionary of phase -> weight multipliers.
                           Example: {"RING_PLACEMENT": 0.5, "MAIN_GAME": 2.0, "RING_REMOVAL": 0.5}
                           If None, defaults to 1.0 for everything.
        Returns:
            (states, move_probs, values) as tensors.
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
        p = np.ones(n, dtype=np.float64)

        # Apply phase-based weighting
        for i, phase in enumerate(self.phases):
            p[i] *= phase_weights.get(phase, 1.0)

        # Normalize to make a proper probability distribution
        p = p / p.sum()

        # Sample without replacement
        indices = np.random.choice(n, batch_size, replace=False, p=p)

        states = torch.stack([torch.from_numpy(self.states[i]).float() for i in indices])
        probs = torch.stack([torch.from_numpy(self.move_probs[i]).float() for i in indices])
        values = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)

        # Optional debug: Log how many states of each phase are in this batch
        phase_counts = {"RING_PLACEMENT": 0, "MAIN_GAME": 0, "RING_REMOVAL": 0}
        for idx in indices:
            phase_counts[self.phases[idx]] += 1
        print(f"[DEBUG] Sampled Batch Phase Counts: {phase_counts}")

        return states, probs, values.unsqueeze(1)


class YinshTrainer:
    """Handles the training of the YINSH neural network."""

    def __init__(self,
                 network: NetworkWrapper,
                 device: Optional[str] = None,
                 l2_reg: float = 0.0,
                 metrics_logger: Optional[MetricsLogger] = None,
                 value_head_lr_factor: float = 5.0,
                 value_loss_weights: Tuple[float, float] = (0.5, 0.5),
                 replay_buffer_path: Optional[str] = None,):
        """
        Initialize the trainer.

        Args:
            network: NetworkWrapper instance
            device: Device to train on ('cuda', 'mps', or 'cpu')
            l2_reg: L2 regularization coefficient
            metrics_logger: Optional MetricsLogger instance
            value_head_lr_factor: Factor to multiply base lr for value head
            value_loss_weights: Weights for combining MSE and CE loss in value head
        """
        self.state_encoder = network.state_encoder

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

        self.experience = GameExperience()
        if replay_buffer_path is not None:
            from os import path as osp
            if osp.exists(replay_buffer_path):
                self.experience.load_buffer(replay_buffer_path)
            else:
                print(f"[Replay Buffer] File '{replay_buffer_path}' not found. Starting with empty buffer.")

        # Schedulers
        # self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.policy_optimizer,
        #     T_max=1000,
        #     eta_min=1e-5
        # )

        self.policy_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.policy_optimizer,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=500,
            mode='triangular2',
            cycle_momentum=False  # if using Adam, usually no momentum cycling is needed
        )

        self.value_scheduler = optim.lr_scheduler.CyclicLR(
            self.value_optimizer,
            base_lr=1e-5,
            max_lr=1e-4,
            step_size_up=500,
            mode='triangular2',
            cycle_momentum=True
        )

        self.l2_reg = l2_reg
        self.logger = logging.getLogger("YinshTrainer")
        self.logger.setLevel(logging.DEBUG)

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

        # Debug print every 20 iterations
        if self.current_iteration % 20 == 0:
            device_before = states.device
            print(f"Training tensors device before transfer: {device_before}")

        states = states.to(self.device)
        target_probs = target_probs.to(self.device)
        target_values = target_values.to(self.device)

        # Debug print every 20 iterations
        if self.current_iteration % 20 == 0:
            device_after = states.device
            print(f"Training tensors device after transfer: {device_after}")
            print(f"Network device: {next(self.network.network.parameters()).device}")

        # Forward pass
        pred_logits, pred_values = self.network.network(states)

        # Monitor / log value head metrics
        value_metrics = self._monitor_value_head(
            pred_values,
            target_values,
            log_activations=(self.current_iteration % 10 == 0)
        )
        self._log_value_head_metrics(value_metrics)

        # Additional logging per-sample if needed
        for i in range(states.size(0)):
            game_state = self.state_encoder.decode_state(states[i].cpu().numpy())
            self.value_metrics.record_evaluation(
                state=game_state,
                value_pred=pred_values[i].detach().cpu().numpy(),
                policy_probs=F.softmax(pred_logits[i], dim=-1).detach().cpu().numpy(),
                chosen_move=None,
                temperature=self.temperature,
                actual_outcome=target_values[i].detach().cpu().numpy()
            )

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
            policy_loss.backward(retain_graph=True)

            # Clip gradients
            policy_params = [p for n, p in self.network.network.named_parameters()
                             if 'value_head' not in n]
            torch.nn.utils.clip_grad_norm_(policy_params, max_norm=1.0)

            self.policy_optimizer.step()
            self.policy_scheduler.step()

        # --- Value Optimization ---
        self.value_optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Recompute (because we did backward above)
            _, pred_values = self.network.network(states)

            # MSE component
            value_loss_mse = F.mse_loss(pred_values, target_values)

            # Optionally keep BCE for sign classification
            target_outcomes = (target_values > 0).long()
            value_probs = torch.sigmoid(pred_values)  # interpret as "win" probability
            value_loss_ce = F.binary_cross_entropy(value_probs, target_outcomes.float())

            # Weighted combination
            value_loss = (self.value_loss_weights[0] * value_loss_mse +
                          self.value_loss_weights[1] * value_loss_ce)

            if self.l2_reg > 0:
                l2_loss = 0
                for name, param in self.network.network.named_parameters():
                    if 'value_head' in name:
                        l2_loss += torch.norm(param)
                value_loss = value_loss + (self.l2_reg * 2) * l2_loss

            value_loss_val = float(value_loss.item())
            value_loss.backward()

            # Clip value head gradients
            value_params = [p for n, p in self.network.network.named_parameters()
                            if 'value_head' in n]
            torch.nn.utils.clip_grad_norm_(value_params, max_norm=0.5)

            self.value_optimizer.step()
            self.value_scheduler.step()

        # Debugging prints
        if self.current_iteration % 10 == 0:
            print("Value Head Gradients:")
            for name, param in self.network.network.named_parameters():
                if 'value_head' in name and param.grad is not None:
                    print(f"  {name}: {param.grad.data.norm(2).item():.6f}")

        # Metrics / logging
        with torch.no_grad():
            pred_outcomes = (pred_values > 0).float()
            true_outcomes = (target_values > 0).float()
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

            # Track LR
            self.learning_rates['policy'].append(self.policy_optimizer.param_groups[0]['lr'])
            self.learning_rates['value'].append(self.value_optimizer.param_groups[0]['lr'])

        self.current_iteration += 1

        return (policy_loss_val, value_loss_val, value_accuracy, move_accuracies)

    def _monitor_value_head(self, pred_values: torch.Tensor,
                            target_values: torch.Tensor,
                            log_activations: bool = False) -> Dict:
        """Monitor value head performance and distributions."""
        with torch.no_grad():
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
        print("\nValue Head Analysis:")
        print("Pre-tanh Activations:")
        print(f"  Range: [{metrics['pre_tanh']['min']:.3f}, {metrics['pre_tanh']['max']:.3f}]")
        print(f"  Distribution: {metrics['pre_tanh']['mean']:.3f} ± {metrics['pre_tanh']['std']:.3f}")
        print(f"  Saturated: {metrics['pre_tanh']['saturated_pct']:.1f}%")

        print("\nPredictions:")
        print(f"  Confidence: {metrics['predictions']['mean_confidence']:.3f}")
        print(f"  High Confidence: {metrics['predictions']['high_confidence_pct']:.1f}%")
        print(f"  Distribution: {metrics['predictions']['mean']:.3f} ± {metrics['predictions']['std']:.3f}")

        print("\nAlignment with Targets:")
        print(f"  Sign Match: {metrics['alignment']['sign_match_pct']:.1f}%")
        print(f"  MSE: {metrics['alignment']['mse']:.3f}")
        print(f"  MAE: {metrics['alignment']['mae']:.3f}")

        if any(k.startswith('layer_') for k in metrics.keys()):
            print("\nLayer-wise Activations:")
            for name, layer_metrics in metrics.items():
                if name.startswith('layer_'):
                    print(f"  {name}:")
                    print(f"    Mean: {layer_metrics['mean']:.3f}")
                    print(f"    Std: {layer_metrics['std']:.3f}")
                    print(f"    Zeros: {layer_metrics['zeros_pct']:.1f}%")

    def train_epoch(self, batch_size: int, batches_per_epoch: int, ring_placement_weight: float = 1.0) -> Dict:
        """Train for one epoch and return comprehensive stats."""
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

        for batch in range(batches_per_epoch):
            phase_weights = {
                "RING_PLACEMENT": 0.5,  # half emphasis
                "MAIN_GAME": 2.0,  # double emphasis
                "RING_REMOVAL": 0.5  # half emphasis
            }
            p_loss, v_loss, v_acc, move_accs = self.train_step(
                batch_size,
                phase_weights=phase_weights
            )

            # Track value head metrics
            value_confidences.append(self._get_current_value_confidence())
            value_sign_matches.append(v_acc)

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

        # Add average value confidence
        stats_accum['value_confidence'] = np.mean(value_confidences) if value_confidences else 0.0

        # Track value head improvement
        current_value_accuracy = stats_accum['value_accuracy']
        if current_value_accuracy > self.best_value_accuracy:
            self.best_value_accuracy = current_value_accuracy
            self.value_accuracy_patience = 0
            self.logger.info(f"New best value accuracy: {current_value_accuracy:.2%}")
        else:
            self.value_accuracy_patience += 1
            # If accuracy hasn't improved for several epochs, adjust learning rate
            if self.value_accuracy_patience >= 3:
                new_lr = self.value_optimizer.param_groups[0]['lr'] * 0.7
                self.logger.warning(f"Value accuracy not improving for {self.value_accuracy_patience} epochs. "
                                    f"Reducing value LR: {current_value_lr:.2e} -> {new_lr:.2e}")
                for param_group in self.value_optimizer.param_groups:
                    param_group['lr'] = new_lr
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
            f"conf={stats_accum['value_confidence']:.3f}, "  # Added confidence reporting
            f"lr={metrics.learning_rates['value']:.2e}\n"
            f"Moves:  acc={metrics.move_accuracies['top_1_accuracy']:.2%}, "
            f"top3={metrics.move_accuracies['top_3_accuracy']:.2%}\n"
            f"Grad norm: {metrics.gradient_norm:.2e}, Loss improvement: {metrics.loss_improvement:.2%}\n"
            f"{'=' * 50}"
        )

        return vars(metrics)

    # Add this helper method to the YinshTrainer class
    def _get_current_value_confidence(self) -> float:
        """Helper method to get the current average value head confidence."""
        if not hasattr(self, 'value_head_activations') or not self.value_head_activations:
            return 0.0

        # Get the absolute value of predictions (confidence)
        pred_values = self.value_head_activations.get('predictions', {}).get('mean_confidence', 0.0)
        return float(pred_values)

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

    def add_game_experience(self, states: list, policies: list, outcome, discount_factor: float = 1.0):
        """
        Delegating method for backward compatibility.

        **IMPORTANT:** Outcome must be provided as a tuple of final scores,
        e.g. (final_white_score, final_black_score) such as (3,1) or (3,2),
        so that the normalized margin is computed correctly.

        Args:
            states:   List of game state numpy arrays.
            policies: List of move probability distributions.
            outcome:  Either an integer (1 for white win, -1 for black win, 0 for draw)
                      or a tuple (final_white_score, final_black_score).
            discount_factor: Discount factor applied per move (default 1.0, meaning no discount).
        """
        if not (isinstance(outcome, (list, tuple)) and len(outcome) == 2):
            raise ValueError("Outcome must be a tuple of final scores, e.g. (3,1) or (3,2)")

        final_white_score, final_black_score = outcome
        self.experience.add_game_experience(
            states,
            policies,
            final_white_score,
            final_black_score,
            discount_factor
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