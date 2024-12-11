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


class SpatialAttention(nn.Module):
    """Spatial attention module for YINSH board."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, 1)
        self.conv2 = nn.Conv2d(channels // 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, 11, 11)
        Returns:
            Attention weights of shape (batch_size, 1, 11, 11)
        """
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return attention


class AttentionBlock(nn.Module):
    """Combines spatial attention with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = SpatialAttention(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Compute attention weights
        attention = self.attention(x)

        # Apply attention and residual connection
        out = F.relu(self.bn1(self.conv1(x * attention)))
        out = self.bn2(self.conv2(out))
        out = out * attention  # Re-weight after processing

        return F.relu(out + identity)

class YinshNetwork(nn.Module):
    """
    Neural network for YINSH game prediction.
    Architecture similar to AlphaZero:
    - Input convolutional layers
    - Residual blocks
    - Policy head (move prediction)
    - Value head (position evaluation)
    """

    def __init__(self, num_channels: int = 128, num_blocks: int = 8):  # Reduced blocks for attention
        super().__init__()

        # Fixed output size
        self.total_moves = 7395

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(6, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Alternate between residual and attention blocks
        self.main_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i % 2 == 0:
                self.main_blocks.append(ResBlock(num_channels))
            else:
                self.main_blocks.append(AttentionBlock(num_channels))

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

        # Value head with attention
        self.value_attention = SpatialAttention(num_channels)
        self.value_head = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),

            # Reduced width architecture
            nn.Linear(32 * 11 * 11, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Tanh()
        )

        # Register hooks for activation tracking
        for name, module in self.value_head.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LayerNorm)):
                module.register_forward_hook(self._make_activation_hook(name))

        # Initialize weights
        self._initialize_weights()
        self._initialize_value_head()

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
            - move_probabilities: Shape (batch_size, total_moves)
            - value: Shape (batch_size, 1)
        """
        # Initial convolution
        x = self.conv_block(x)

        # Process through main blocks
        attn_maps = []
        for block in self.main_blocks:
            x = block(x)
            if isinstance(block, AttentionBlock):
                attn_maps.append(block.attention(x))

        # Policy head
        policy = self.policy_head(x)

        # Value head with attention
        value_attn = self.value_attention(x)
        value_features = x * value_attn
        value = self.value_head(value_features)

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