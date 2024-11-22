"""Training implementation for YINSH ML model."""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
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
        """
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        self.network = network  # Store the wrapper
        self.network.network = self.network.network.to(self.device)  # Move the actual network to device

        self.optimizer = optim.Adam(
            self.network.network.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        self.experience = GameExperience()

        # Add learning rate scheduler. adjusting to try not to create such a steep curve.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=10,
            verbose=True,
            min_lr=1e-6  # Add minimum learning rate
        )

        self.l2_reg = l2_reg

        # Setup logging
        self.logger = logging.getLogger("YinshTrainer")
        self.logger.setLevel(logging.INFO)

        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        self.learning_rates = []

    def train_step(self, batch_size: int) -> Tuple[float, float]:
        """Improved training step with better loss calculations."""
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

        # Calculate policy loss using KL divergence
        log_pred_probs = F.log_softmax(pred_logits, dim=1)
        policy_loss = -(target_probs * log_pred_probs).sum(dim=1).mean()

        # Calculate value loss with added L1 regularization
        mse_loss = F.mse_loss(pred_values, target_values)
        l1_loss = torch.abs(pred_values).mean()  # L1 regularization
        value_loss = mse_loss + 0.01 * l1_loss  # Small L1 coefficient

        # Combined loss with L2 regularization
        total_loss = policy_loss + value_loss

        # Add L2 regularization if specified
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

        # Update learning rate
        self.scheduler.step(total_loss)

        return policy_loss.item(), value_loss.item()

    def train_epoch(self, batch_size: int, batches_per_epoch: int):
        """Train for one epoch."""
        policy_loss_sum = 0
        value_loss_sum = 0

        for _ in range(batches_per_epoch):
            p_loss, v_loss = self.train_step(batch_size)
            policy_loss_sum += p_loss
            value_loss_sum += v_loss

        avg_policy_loss = policy_loss_sum / batches_per_epoch
        avg_value_loss = value_loss_sum / batches_per_epoch

        # Track current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.total_losses.append(avg_policy_loss + avg_value_loss)

        self.logger.info(f"Epoch complete - Policy Loss: {avg_policy_loss:.4f}, "
                         f"Value Loss: {avg_value_loss:.4f}, "
                         f"Learning Rate: {current_lr:.2e}")

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
        """Modified experience addition with improved value targets."""
        game_length = len(states)

        # Calculate discounted values with better decay
        for idx, (state, policy) in enumerate(zip(states, policies)):
            # Use a slower decay rate for longer-term planning
            discount = 0.95
            steps_from_end = game_length - idx
            value = outcome * (discount ** steps_from_end)

            # Add some noise to prevent overfitting
            value_noise = np.random.normal(0, 0.05)
            value = np.clip(value + value_noise, -1, 1)

            self.experience.states.append(state)
            self.experience.move_probs.append(policy)
            self.experience.values.append(value)

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