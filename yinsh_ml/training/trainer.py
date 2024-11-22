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

    def train_step(self, batch_size: int) -> Tuple[float, float]:
        """Training step with proper binary outcomes and loss calculations."""
        if len(self.experience.states) < batch_size:
            return 0.0, 0.0

        self.network.network.train()
        states, target_probs, target_values = self.experience.sample_batch(batch_size)

        # Move to device
        states = states.to(self.device)
        target_probs = target_probs.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass
        pred_logits, pred_values = self.network.network(states)

        # Ensure predictions are in [-1, 1] (though network should do this with tanh)
        pred_values = torch.tanh(pred_values)

        # After forward pass but before loss calculation
        accuracies = self._calculate_accuracies(pred_values, target_values,
                                                pred_logits, target_probs)

        # Update tracking metrics
        self.value_accuracies.append(accuracies['value_accuracy'])
        self.move_accuracies.append(accuracies['move_accuracy'])

        # Calculate value loss using binary cross entropy
        # Scale from [-1,1] to [0,1] for BCE
        scaled_pred_values = (pred_values + 1) / 2
        scaled_target_values = (target_values + 1) / 2
        value_loss = F.binary_cross_entropy(
            scaled_pred_values,
            scaled_target_values,
            reduction='mean'
        )

        # Calculate policy loss with label smoothing
        temperature = 1.0  # Could make configurable
        scaled_logits = pred_logits / temperature
        log_probs = F.log_softmax(scaled_logits, dim=1)

        # Apply label smoothing to policy targets
        n_classes = target_probs.shape[1]
        epsilon = 0.1
        uniform = torch.ones_like(target_probs) / n_classes
        smoothed_targets = (1 - epsilon) * target_probs + epsilon * uniform
        policy_loss = -(smoothed_targets * log_probs).sum(dim=1).mean()

        # Combined loss with L2 regularization
        total_loss = policy_loss + value_loss

        # Add L2 regularization if specified (keep as is)
        if self.l2_reg > 0:
            l2_loss = 0
            for param in self.network.network.parameters():
                l2_loss += torch.norm(param)
            total_loss = total_loss + self.l2_reg * l2_loss

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update learning rate (changed from step(total_loss) for CosineAnnealingLR)
        self.scheduler.step()

        return policy_loss.item(), value_loss.item()

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