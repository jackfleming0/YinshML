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
from collections import deque
import random

from ..network.wrapper import NetworkWrapper

# Setup logger
logger = logging.getLogger(__name__)

class GameExperience:
    """Stores game states and outcomes for training."""

    def __init__(self, max_size: int = 100000):
        self.states = deque(maxlen=max_size)
        self.move_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)

    def add_game(self, states: List[np.ndarray],
                 move_probs: List[np.ndarray],
                 winner: int):
        """
        Add a completed game to the experience buffer.

        Args:
            states: List of game states
            move_probs: List of move probability distributions
            winner: 1 for white win, -1 for black win, 0 for draw
        """
        game_length = len(states)
        for idx, (state, probs) in enumerate(zip(states, move_probs)):
            # Calculate discounted value based on winner and move number
            value = winner * (0.99 ** (game_length - idx))

            self.states.append(state)
            self.move_probs.append(probs)
            self.values.append(value)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of experiences."""
        indices = np.random.choice(len(self.states), batch_size)

        states = torch.stack([torch.from_numpy(self.states[i]).float() for i in indices])  # Ensure float32
        probs = torch.stack([torch.from_numpy(self.move_probs[i]).float() for i in indices])  # Ensure float32
        values = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)

        return states, probs, values.unsqueeze(1)


class YinshTrainer:
    """Handles the training of the YINSH neural network."""

    def __init__(self, network: NetworkWrapper, device: Optional[str] = None,
                 l2_reg: float = 0.0):
        """
        Initialize the trainer.

        Args:
            network: NetworkWrapper instance
            device: Device to train on ('cuda', 'mps', or 'cpu')
            l2_reg: L2 regularization coefficient
        """
        # Device setup remains the same
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

        # Keep the same optimizer settings for now
        self.optimizer = optim.Adam(
            self.network.network.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        self.experience = GameExperience()

        # Replace ReduceLROnPlateau with CosineAnnealingLR
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,  # Will adjust based on iterations * epochs
            eta_min=1e-5
        )

        self.l2_reg = l2_reg

        # Setup logging
        self.logger = logging.getLogger("YinshTrainer")
        self.logger.setLevel(logging.INFO)

        # Enhanced metrics tracking
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.learning_rates = []
        self.value_accuracies = []  # Track value prediction accuracy
        self.move_accuracies = []  # Track move prediction accuracy

    def _smooth_policy_targets(self, targets: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        n_classes = targets.shape[1]
        uniform = torch.ones_like(targets) / n_classes
        return (1 - epsilon) * targets + epsilon * uniform

    def train_step(self, batch_size: int) -> Tuple[float, float, float, Dict]:
        """Training step with additional metrics."""
        if len(self.experience.states) < batch_size:
            return 0.0, 0.0, 0.0, {'top_1_accuracy': 0.0, 'top_3_accuracy': 0.0, 'top_5_accuracy': 0.0}

        self.network.network.train()
        states, target_probs, target_values = self.experience.sample_batch(batch_size)

        # Move to device
        states = states.to(self.device)
        target_probs = target_probs.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass
        pred_logits, pred_values = self.network.network(states)

        # Add comprehensive value head monitoring
        value_metrics = self._monitor_value_head(
            pred_values,
            target_values,
            log_activations=(self.iteration % 10 == 0)  # Log full activations every 10 iterations
        )
        self._log_value_head_metrics(value_metrics)

        # Add pre-tanh activation monitoring
        with torch.no_grad():
            value_head_layers = [module for name, module in self.network.network.value_head.named_modules()
                                 if isinstance(module, (nn.Linear, nn.ReLU))]
            activations = []
            for layer in value_head_layers:
                activations.append(layer)

        # Add value confidence tracking with more detailed metrics
        value_confidence = torch.abs(pred_values)
        high_confidence = (value_confidence > 0.8).float().mean()
        print(f"\nValue Prediction Analysis:")
        print(f"  Mean confidence: {value_confidence.mean():.3f}")
        print(f"  High confidence predictions (>0.8): {high_confidence:.1%}")
        print(f"  Value distribution: {pred_values.mean():.3f} ± {pred_values.std():.3f}")
        print(f"  Pre-tanh activation range: [{pred_values.min():.3f}, {pred_values.max():.3f}]")

        # Calculate value loss using Huber loss instead of MSE
        value_loss = F.smooth_l1_loss(pred_values, target_values, beta=0.5)

        # Rest of the code remains the same until the combined loss section

        # Modified loss combination with increased value loss weight
        total_loss = policy_loss + 2.0 * value_loss

        if self.l2_reg > 0:
            l2_loss = 0
            for param in self.network.network.parameters():
                l2_loss += torch.norm(param)
            total_loss = total_loss + self.l2_reg * l2_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Add separate gradient clipping for value head
        value_params = [p for n, p in self.network.network.named_parameters() if 'value_head' in n]
        torch.nn.utils.clip_grad_norm_(value_params, max_norm=0.5)

        # Keep general gradient clipping as well
        torch.nn.utils.clip_grad_norm_(self.network.network.parameters(), max_norm=1.0)

        # Add gradient norm monitoring for value head
        with torch.no_grad():
            value_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in value_params]))
            print(f"  Value head gradient norm: {value_grad_norm:.3f}")

        self.optimizer.step()
        self.scheduler.step()

        return (
            policy_loss.item(),
            value_loss.item(),
            value_accuracy,
            move_accuracies
        )

    def _monitor_value_head(self, pred_values: torch.Tensor,
                            target_values: torch.Tensor,
                            log_activations: bool = False) -> Dict:
        """Monitor value head performance and distributions.

        Args:
            pred_values: Predicted values from the network
            target_values: Target values
            log_activations: Whether to log full activation distributions

        Returns:
            Dict of monitoring metrics
        """
        with torch.no_grad():
            metrics = {}

            # 1. Pre-tanh activation monitoring
            pre_tanh = pred_values  # These are now pre-tanh values
            metrics['pre_tanh'] = {
                'mean': float(pre_tanh.mean()),
                'std': float(pre_tanh.std()),
                'min': float(pre_tanh.min()),
                'max': float(pre_tanh.max()),
                'saturated_pct': float((torch.abs(pre_tanh) > 0.99).float().mean() * 100)
            }

            # 2. Layer-wise activation tracking
            if log_activations:
                activations = self.network.network.value_head_activations
                for name, activation in activations.items():
                    metrics[f'layer_{name}'] = {
                        'mean': float(activation.mean()),
                        'std': float(activation.std()),
                        'zeros_pct': float((activation == 0).float().mean() * 100)
                    }

            # 3. Value prediction analysis
            value_confidence = torch.abs(pred_values)
            high_confidence = (value_confidence > 0.8).float().mean()
            metrics['predictions'] = {
                'mean_confidence': float(value_confidence.mean()),
                'high_confidence_pct': float(high_confidence * 100),
                'mean': float(pred_values.mean()),
                'std': float(pred_values.std())
            }

            # 4. Target analysis
            metrics['targets'] = {
                'mean': float(target_values.mean()),
                'std': float(target_values.std()),
                'positive_pct': float((target_values > 0).float().mean() * 100)
            }

            # 5. Prediction-target alignment
            pred_signs = torch.sign(pred_values)
            target_signs = torch.sign(target_values)
            metrics['alignment'] = {
                'sign_match_pct': float((pred_signs == target_signs).float().mean() * 100),
                'mse': float(F.mse_loss(pred_values, target_values)),
                'mae': float(F.l1_loss(pred_values, target_values))
            }

            return metrics

    def _log_value_head_metrics(self, metrics: Dict):
        """Log value head metrics in a readable format."""
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

    def train_epoch(self, batch_size: int, batches_per_epoch: int) -> Dict:
        """Train for one epoch and return comprehensive stats."""
        policy_loss_sum = 0
        value_loss_sum = 0
        value_acc_sum = 0
        move_accuracies_list = []
        policy_entropy_sum = 0

        for _ in range(batches_per_epoch):
            p_loss, v_loss, v_acc, move_accs = self.train_step(batch_size)
            policy_loss_sum += p_loss
            value_loss_sum += v_loss
            value_acc_sum += v_acc
            move_accuracies_list.append(move_accs)

            # Calculate policy entropy from last batch if available
            if hasattr(self, 'last_policy_entropy'):
                policy_entropy_sum += self.last_policy_entropy

        # Calculate averages
        stats = {
            'policy_loss': policy_loss_sum / batches_per_epoch,
            'value_loss': value_loss_sum / batches_per_epoch,
            'value_accuracy': value_acc_sum / batches_per_epoch,
            'policy_entropy': policy_entropy_sum / batches_per_epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        # Average move accuracies across batches
        if move_accuracies_list:
            avg_move_accuracies = {
                'top_1_accuracy': np.mean([x['top_1_accuracy'] for x in move_accuracies_list]),
                'top_3_accuracy': np.mean([x['top_3_accuracy'] for x in move_accuracies_list]),
                'top_5_accuracy': np.mean([x['top_5_accuracy'] for x in move_accuracies_list])
            }
            stats['move_accuracies'] = avg_move_accuracies

        # Store in instance variables
        self.policy_losses.append(stats['policy_loss'])
        self.value_losses.append(stats['value_loss'])
        self.total_losses.append(stats['policy_loss'] + stats['value_loss'])
        self.learning_rates.append(stats['learning_rate'])

        self.logger.info(
            f"Epoch complete - "
            f"Policy Loss: {stats['policy_loss']:.4f}, "
            f"Value Loss: {stats['value_loss']:.4f}, "
            f"Value Acc: {stats['value_accuracy']:.2%}, "
            f"Move Acc: {stats['move_accuracies']['top_1_accuracy']:.2%}, "
            f"LR: {stats['learning_rate']:.2e}"
        )

        return stats

    def save_checkpoint(self, path: str, epoch: int):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.total_losses = checkpoint['total_losses']
        self.logger.info(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']

    def add_game_experience(self, states: List[np.ndarray],
                            policies: List[np.ndarray],
                            outcome: int):
        """Add game experience with pure win/loss values."""
        # Simple addition of experiences with pure outcome values
        #print(f"Received outcome value: {outcome}")  # Debug
        for state, policy in zip(states, policies):
            # No discounting or noise - pure win/loss signal
            self.experience.states.append(state)
            self.experience.move_probs.append(policy)
            self.experience.values.append(outcome)  # Pure -1 or 1

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

    def _calculate_accuracies(self, pred_values: torch.Tensor, target_values: torch.Tensor,
                              pred_logits: torch.Tensor, target_probs: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate value and move prediction accuracies.

        Args:
            pred_values: Predicted values [-1, 1]
            target_values: True values [-1, 1]
            pred_logits: Raw move prediction logits
            target_probs: True move probabilities

        Returns:
            Tuple of (value_accuracy, move_accuracy)
        """
        with torch.no_grad():
            # Value accuracy: Check if we correctly predict win/loss
            # Convert from [-1,1] to binary predictions
            pred_outcomes = (pred_values > 0).float()
            true_outcomes = (target_values > 0).float()
            value_accuracy = (pred_outcomes == true_outcomes).float().mean().item()

            # Move accuracy: Check if highest probability move matches
            # Note: This is a strict metric - only counts exact matches
            pred_moves = torch.argmax(pred_logits, dim=1)
            true_moves = torch.argmax(target_probs, dim=1)
            move_accuracy = (pred_moves == true_moves).float().mean().item()

            # Could also add top-k accuracy for moves if desired
            k = 5  # Consider top 5 moves
            _, pred_top_k = torch.topk(pred_logits, k, dim=1)
            _, true_top_k = torch.topk(target_probs, k, dim=1)

            # Check if true best move is in predicted top k
            top_k_accuracy = torch.any(
                pred_top_k == true_moves.unsqueeze(1),
                dim=1
            ).float().mean().item()

            return {
                'value_accuracy': value_accuracy,
                'move_accuracy': move_accuracy,
                'top_k_accuracy': top_k_accuracy
            }