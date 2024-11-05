"""Neural network model for YINSH."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..utils.encoding import StateEncoder


class ResBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class YinshNetwork(nn.Module):
    """
    Neural network for YINSH game prediction.
    Architecture similar to AlphaZero:
    - Input convolutional layers
    - Residual blocks
    - Policy head (move prediction)
    - Value head (position evaluation)
    """

    def __init__(self, num_channels: int = 128, num_blocks: int = 10):
        super().__init__()

        # Fixed output size
        self.total_moves = 7395

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(6, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_blocks)
        ])

        # Policy head (outputs move probabilities)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.total_moves)
            # No activation here - we want raw logits
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout to prevent overconfidence
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        # Initialize weights with different scaling
        self._initialize_weights()



    def _initialize_weights(self):
        """Initialize weights with careful scaling."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Use a smaller initialization scale
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _init_weights(self, module):
        """Initialize weights using He initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 6, 11, 11)
               Channels:
                - 0: White rings
                - 1: Black rings
                - 2: White markers
                - 3: Black markers
                - 4: Valid moves mask
                - 5: Game phase

        Returns:
            Tuple of:
            - move_probabilities: Shape (batch_size, 14641)
            - value: Shape (batch_size, 1)
        """
        # Initial convolution
        x = self.conv_block(x)

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = self.policy_head(x)

        # Value head
        value = self.value_head(x)

        return policy, value

    def predict(self, x: torch.Tensor, move_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a prediction with optional move masking.

        Args:
            x: Input tensor
            move_mask: Optional mask for valid moves

        Returns:
            Tuple of (move_probabilities, value)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            policy, value = self(x)

            # Apply move mask if provided
            if move_mask is not None:
                policy = policy * move_mask

            # Convert to probabilities
            policy = F.softmax(policy, dim=1)

            return policy, value