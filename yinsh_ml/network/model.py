"""Neural network model for YINSH."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from ..utils.encoding import StateEncoder
from ..game.moves import Move

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

        # Simplified Value head without spatial attention:
        # Flatten the shared trunk features, then Linear(num_channels*11*11, 128) -> ReLU -> Linear(128, 1) -> Tanh
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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

        # Simplified value head: directly use the shared trunk features
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

    def select_move(self, move_probs: torch.Tensor, valid_moves: List[Move],
                    temperature: float = 1.0) -> Move:
        """
        Select a move using the policy probabilities with better numerical stability.

        Args:
            move_probs: Probability distribution over all moves
            valid_moves: List of valid moves
            temperature: Temperature for probability scaling

        Returns:
            Selected move
        """
        # Ensure move_probs is 1D
        if len(move_probs.shape) > 1:
            move_probs = move_probs.squeeze()

        # Create tensor of valid move probabilities
        valid_probs = torch.zeros(len(valid_moves), device=self.device)
        valid_moves_indices = []

        for i, move in enumerate(valid_moves):
            try:
                idx = self.state_encoder.move_to_index(move)
                valid_moves_indices.append(idx)
                if idx < len(move_probs):  # Bounds check
                    valid_probs[i] = move_probs[idx].item()
            except Exception as e:
                self.logger.warning(f"Error processing move {move}: {e}")
                continue

        # Better handling of edge cases
        if len(valid_moves_indices) == 0:
            self.logger.warning("No valid moves could be processed")
            return random.choice(valid_moves)

        if valid_probs.max() < 1e-8 or torch.isnan(valid_probs.sum()):
            self.logger.warning("Very low or invalid probabilities detected")
            return random.choice(valid_moves)

        # Apply temperature and normalization with better numerical stability
        if temperature > 0:
            # Use log space for numerical stability
            log_probs = torch.log(valid_probs + 1e-10)
            scaled_log_probs = log_probs / temperature
            valid_probs = F.softmax(scaled_log_probs, dim=0)
        else:
            # Temperature 0 means greedy selection
            max_idx = torch.argmax(valid_probs).item()
            return valid_moves[max_idx]

        try:
            # Ensure probs sum to 1
            valid_probs = valid_probs / valid_probs.sum()

            # Move to CPU for multinomial sampling
            valid_probs = valid_probs.cpu()
            selected_idx = torch.multinomial(valid_probs, 1).item()
            return valid_moves[selected_idx]
        except Exception as e:
            self.logger.error(f"Error in move selection: {e}")
            self.logger.debug(f"Probabilities: {valid_probs}")
            # Fallback to random selection
            return random.choice(valid_moves)