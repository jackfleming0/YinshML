"""Network wrapper for YINSH ML model."""

import random
import torch
import coremltools as ct
import logging
from typing import Optional, Tuple, NamedTuple, List
import os
from collections import namedtuple
from ..game.moves import Move, MoveType

from .model import YinshNetwork
from ..utils.encoding import StateEncoder  # Add this line
import torch.nn.functional as F

# Define output type for traced model
ModelOutput = namedtuple('ModelOutput', ['policy', 'value'])

logging.getLogger('NetworkWrapper').setLevel(logging.ERROR)

class NetworkWrapper:
    """Wrapper class for the YINSH neural network model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the network wrapper.

        Args:
            model_path: Optional path to load a pre-trained model
            device: Device to use ('cuda', 'mps', or 'cpu')
        """
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        self.network = YinshNetwork().to(self.device)

        # Setup logging
        self.logger = logging.getLogger("NetworkWrapper")
        self.logger.setLevel(logging.ERROR)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        self.network.eval()  # Set to evaluation mode by default

        # Initialize StateEncoder
        self.state_encoder = StateEncoder()

    def predict(self, state_tensor: torch.Tensor,
                move_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a prediction for the given state.

        Args:
            state_tensor: Game state tensor of shape (6, 11, 11)
            move_mask: Optional mask for valid moves (True for valid moves)
            temperature: Temperature for move probability scaling

        Returns:
            Tuple of (move_probabilities, value)
        """
        self.network.eval()
        with torch.no_grad():
            # Get network predictions
            move_logits, value = self.network(state_tensor)

            # Ensure value predictions are in [-1, 1]
            value = torch.tanh(value)

            # Apply move mask using proper masking
            if move_mask is not None:
                # Use masked_fill for proper handling of invalid moves
                move_logits = move_logits.masked_fill(~move_mask, float('-inf'))

            # Apply temperature scaling to logits
            if temperature != 0:
                scaled_logits = move_logits / temperature
            else:
                # If temperature is 0, just take argmax
                scaled_logits = move_logits

            # Convert to probabilities
            move_probabilities = F.softmax(scaled_logits, dim=1)

            return move_probabilities, value

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

    def save_model(self, path: str):
        """Save model weights to file."""
        try:
            torch.save(self.network.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    # def load_model(self, path: str):
    #     try:
    #         state_dict = torch.load(path, map_location=self.device)
    #         self.network.load_state_dict(state_dict, strict=False)
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load model: {e}")

    def load_model(self, path: str):
        """Load model with architecture adaptation."""
        try:
            # Load checkpoint
            state_dict = torch.load(path, map_location=self.device)

            # Filter out incompatible layers, keeping only matching shapes
            compatible_state_dict = {}
            model_state = self.network.state_dict()

            for key, param in state_dict.items():
                if key in model_state and param.shape == model_state[key].shape:
                    compatible_state_dict[key] = param

            # Load compatible weights
            self.network.load_state_dict(compatible_state_dict, strict=False)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def export_to_coreml(self, path: str):
        """Export model to CoreML format."""
        try:
            # Set model to eval mode and move to CPU
            self.network.eval().to('cpu')

            # Create example input on CPU
            example_input = torch.randn(1, 6, 11, 11).to('cpu')

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

            # Save the CoreML model
            mlmodel.save(path)
            self.logger.info(f"Model exported to CoreML format at {path}")

            # Move model back to MPS
            self.network.to('mps')

        except Exception as e:
            self.logger.error(f"Error exporting to CoreML: {str(e)}")
            raise