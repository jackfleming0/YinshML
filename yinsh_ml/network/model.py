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

        self.value_head_activations = {}
        # Modified value head with better scaling and normalization
        # self.value_head = nn.Sequential(
        #     # Initial convolution with batch norm
        #     nn.Conv2d(num_channels, 32, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #
        #     # First dense layer with careful initialization
        #     nn.Linear(32 * 11 * 11, 256),
        #     nn.BatchNorm1d(256),  # Changed from LayerNorm to BatchNorm
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #
        #     # Second dense layer
        #     nn.Linear(256, 64),
        #     nn.BatchNorm1d(64),  # Changed from LayerNorm to BatchNorm
        #     nn.ReLU(),
        #
        #     # Final prediction layer with careful initialization and normalization
        #     nn.Linear(64, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1),
        #     nn.BatchNorm1d(1),  # Add BatchNorm before tanh
        #     nn.Tanh()
        # )
        self.value_head = nn.Sequential(
            # Kept same: Initial feature extraction
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),

            # Changed: Reduced width from 256->128 to prevent overconfidence
            # and force the network to be more selective about features
            nn.Linear(32 * 11 * 11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Increased dropout from 0.1->0.2 to reduce overfitting
            nn.Dropout(0.2),

            # Simplified: Removed intermediate 64-unit layer
            # Direct path to final 32 units helps clearer value signals
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Added second dropout layer for more regularization
            nn.Dropout(0.2),

            # Simplified final layers: Removed extra BatchNorm
            # Direct path from 32->1 with immediate Tanh
            # This should reduce "wavering" in value predictions
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # Register hooks for activation tracking
        for name, module in self.value_head.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LayerNorm)):
                module.register_forward_hook(self._make_activation_hook(name))

        # Initialize weights with different scaling
        self._initialize_weights()
        self._initialize_value_head()  # Add our new specific initialization for value head

    def _initialize_value_head(self):
        """Careful initialization of value head weights."""
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                # Smaller initialization scale for final layer
                if m.out_features == 1:
                    nn.init.uniform_(m.weight, -0.01, 0.01)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_activation_hook(self, name: str):
        def hook(module, input, output):
            self.value_head_activations[name] = output
        return hook

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