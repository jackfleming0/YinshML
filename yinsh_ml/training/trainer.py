"""Training implementation for YINSH ML model."""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
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

        states = torch.stack([torch.from_numpy(self.states[i]) for i in indices])
        probs = torch.stack([torch.from_numpy(self.move_probs[i]) for i in indices])
        values = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)

        return states, probs, values.unsqueeze(1)


class YinshTrainer:
    """Handles the training of the YINSH neural network."""

    def __init__(self, network: NetworkWrapper, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the trainer.

        Args:
            network: NetworkWrapper instance
            device: Device to train on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.network = network  # Store the wrapper
        self.network.network = self.network.network.to(self.device)  # Move the actual network to device

        self.optimizer = optim.Adam(
            self.network.network.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        self.experience = GameExperience()

        # Setup logging
        self.logger = logging.getLogger("YinshTrainer")
        self.logger.setLevel(logging.INFO)

        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []

    def train_step(self, batch_size: int) -> Tuple[float, float]:
        """
        Perform one training step on a batch of data.

        Returns:
            policy_loss, value_loss
        """
        if len(self.experience.states) < batch_size:
            return 0.0, 0.0

        self.network.network.train()
        states, target_probs, target_values = self.experience.sample_batch(batch_size)

        # Move to device
        states = states.to(self.device)
        target_probs = target_probs.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass
        pred_probs, pred_values = self.network.network(states)

        # Calculate losses
        policy_loss = -torch.mean(torch.sum(target_probs * F.log_softmax(pred_probs, dim=1), dim=1))
        value_loss = nn.MSELoss()(pred_values, target_values)

        # Combined loss
        total_loss = policy_loss + value_loss

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

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

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.total_losses.append(avg_policy_loss + avg_value_loss)

        self.logger.info(f"Epoch complete - Policy Loss: {avg_policy_loss:.4f}, "
                         f"Value Loss: {avg_value_loss:.4f}")

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
        """Add a completed game to the experience buffer."""
        self.experience.add_game(states, policies, outcome)

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