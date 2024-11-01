"""Network wrapper for YINSH ML model."""

import torch
import coremltools as ct
import logging
from typing import Optional, Tuple, NamedTuple
import os
from collections import namedtuple

from .model import YinshNetwork
from ..utils.encoding import StateEncoder  # Add this line
import torch.nn.functional as F

# Define output type for traced model
ModelOutput = namedtuple('ModelOutput', ['policy', 'value'])

class NetworkWrapper:
    """Wrapper class for the YINSH neural network model."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the network wrapper.

        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = YinshNetwork().to(self.device)

        # Setup logging
        self.logger = logging.getLogger("NetworkWrapper")
        self.logger.setLevel(logging.DEBUG)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        self.network.eval()  # Set to evaluation mode by default

        # Initialize StateEncoder
        self.state_encoder = StateEncoder()

    def predict(self, state_tensor: torch.Tensor,
                move_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a prediction for the given state.

        Args:
            state_tensor: Game state tensor of shape (6, 11, 11)
            move_mask: Optional mask for valid moves

        Returns:
            Tuple of (move_probabilities, value)
        """
        self.network.eval()
        with torch.no_grad():
            # Move tensors to correct device
            state_tensor = state_tensor.to(self.device)
            if move_mask is not None:
                move_mask = move_mask.to(self.device)

            # Add batch dimension if needed
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)

            # Get network predictions
            move_logits, value = self.network(state_tensor)

            # Apply move mask if provided
            if move_mask is not None:
                move_logits = move_logits * move_mask

            # Convert to probabilities
            move_probabilities = F.softmax(move_logits, dim=1)

            return move_probabilities.squeeze(), value.squeeze()

    def save_model(self, path: str):
        """Save model weights to file."""
        try:
            torch.save(self.network.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str):
        """Load model weights from file."""
        try:
            self.network.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")

    def export_to_coreml(self, path: str):
        """Export model to CoreML format."""
        try:
            # Set model to eval mode
            self.network.eval()

            # Create example input
            example_input = torch.randn(1, 6, 11, 11)

            # Create a traced model with named outputs using NamedTuple
            class TracedModel(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, game_state):
                    policy, value = self.model(game_state)
                    return ModelOutput(policy=policy, value=value)

            traced_model = TracedModel(self.network)

            # Trace the model with strict=False to allow NamedTuple output
            traced_script = torch.jit.trace(
                traced_model,
                example_input,
                strict=False
            )

            # Convert to CoreML
            mlmodel = ct.convert(
                traced_script,
                convert_to="mlprogram",
                inputs=[ct.TensorType(name="game_state", shape=(1, 6, 11, 11))],
                compute_units=ct.ComputeUnit.CPU_AND_GPU
            )

            # Save the model
            mlmodel.save(path)
            self.logger.info(f"Model exported to CoreML format at {path}")

        except Exception as e:
            self.logger.error(f"Error exporting to CoreML: {str(e)}")
            raise