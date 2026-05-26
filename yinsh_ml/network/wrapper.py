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
# --- Memory Pool Imports ---
from ..memory import TensorPool, TensorPoolConfig

# Define output type for traced model
ModelOutput = namedtuple('ModelOutput', ['policy', 'value'])

logging.getLogger('NetworkWrapper').setLevel(logging.DEBUG)

class NetworkWrapper:
    """Wrapper class for the YINSH neural network model."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 tensor_pool: Optional[TensorPool] = None,
                 value_mode: str = 'classification', num_value_classes: int = 7,
                 use_enhanced_encoding: bool = False,
                 num_channels: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 value_head_type: Optional[str] = None):
        """
        Initialize the network wrapper.

        Args:
            model_path: Optional path to load a pre-trained model
            device: Device to use ('cuda', 'mps', or 'cpu')
            tensor_pool: Optional TensorPool for memory management
            value_mode: 'classification' (AlphaZero-style) or 'regression' (legacy MSE)
            num_value_classes: Number of discrete outcome classes for classification mode
            use_enhanced_encoding: If True, use 15-channel enhanced encoding. If False, use basic 6-channel.
            num_channels: ResNet channel count. If None and model_path given, auto-detect from state_dict.
            num_blocks: Number of residual/attention blocks. If None and model_path given, auto-detect.
            value_head_type: 'spatial' (legacy ~4M params) or 'gap' (Branch D.1 ~17K params).
                If None and model_path given, auto-detect by inspecting `value_head.0.weight`'s
                kernel size (3 → spatial, 1 → gap). If None and no model_path, defaults to
                'spatial'. To DELIBERATELY swap heads during warm-start (e.g. load a spatial
                checkpoint into a gap wrapper for Branch D.1), pass 'gap' explicitly — the
                shape filter in `load_model` will load the trunk + policy and silently drop
                the spatial-head keys, leaving the GAP head freshly initialized.
        """
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )

        # Store encoding configuration
        self.use_enhanced_encoding = use_enhanced_encoding
        input_channels = 15 if use_enhanced_encoding else 6

        # Auto-detect capacity (and encoding) from checkpoint when not
        # explicitly given. Lets supervised pretrains with custom
        # --num-channels / --num-blocks / --use-enhanced-encoding load without
        # callers needing to know the dims in advance.
        if model_path and os.path.exists(model_path):
            sd = torch.load(model_path, map_location='cpu', weights_only=True)
            # The entry layer is nn.Sequential(conv, bn, relu) named conv_block;
            # the first Conv2d's weight is at conv_block.0.weight. Shape is
            # (num_channels, input_channels, 3, 3).
            w = sd.get('conv_block.0.weight')
            if num_channels is None and w is not None:
                num_channels = int(w.shape[0])
            if w is not None:
                # input_channels lives in dim 1; override use_enhanced_encoding
                # to match what the checkpoint actually has, so subsequent
                # encoder construction is consistent.
                ckpt_input_ch = int(w.shape[1])
                if ckpt_input_ch == 15:
                    self.use_enhanced_encoding = True
                    input_channels = 15
                elif ckpt_input_ch == 6:
                    self.use_enhanced_encoding = False
                    input_channels = 6
            if num_blocks is None:
                # Count main_blocks.{i}.* prefixes
                block_ids = set()
                for k in sd.keys():
                    if k.startswith('main_blocks.'):
                        try:
                            block_ids.add(int(k.split('.')[1]))
                        except (ValueError, IndexError):
                            continue
                if block_ids:
                    num_blocks = len(block_ids)
            # Auto-detect value head type. Spatial head's first layer is
            # Conv2d(c, 64, kernel=3) → weight shape (64, c, 3, 3). Both GAP
            # variants use Conv2d(c, 64, kernel=1) → weight shape (64, c, 1, 1).
            # To discriminate GAP v1 vs v2, count Linear modules in the head:
            #   spatial: 3 Linears (7744→512, 512→256, 256→out)
            #   gap (v1): 1 Linear (64→out)
            #   gap_v2:   2 Linears (64→hidden=80, hidden→out)
            # Detection by counting `value_head.N.weight` keys with shape == 2
            # (which marks a Linear's weight tensor).
            # Only override if the caller didn't specify a type — explicit
            # 'gap' / 'gap_v2' on a spatial checkpoint is the warm-start path.
            if value_head_type is None:
                vh0 = sd.get('value_head.0.weight')
                if vh0 is not None and vh0.dim() == 4:
                    if vh0.shape[-1] == 3:
                        value_head_type = 'spatial'
                    else:
                        # kernel=1 → GAP variant. Count Linears to distinguish.
                        n_linears = sum(
                            1 for k, v in sd.items()
                            if k.startswith('value_head.') and k.endswith('.weight')
                            and v.dim() == 2
                        )
                        value_head_type = 'gap_v2' if n_linears >= 2 else 'gap'
            # Auto-detect value_mode + num_value_classes from the LAST Linear
            # weight in the value head:
            #   classification → out_features == num_value_classes (e.g. 7)
            #   regression     → out_features == 1
            # Only override if the caller didn't specify the mode explicitly
            # — explicit kwarg wins (parallel to value_head_type's policy).
            # The default value_mode kwarg is 'classification', so we detect
            # 'regression' from the checkpoint OR keep the default. To
            # distinguish "caller passed classification" from "caller didn't
            # care", we check the LAST linear's shape and switch only when
            # it clearly indicates regression. (Symmetric to value_head_type's
            # `is None` check, but value_mode has a non-None default so we
            # use the checkpoint as the ground truth when it disagrees.)
            value_head_linear_weights = [
                (int(k.split('.')[1]), v) for k, v in sd.items()
                if k.startswith('value_head.') and k.endswith('.weight') and v.dim() == 2
            ]
            if value_head_linear_weights:
                value_head_linear_weights.sort(key=lambda kv: kv[0])
                last_linear_weight = value_head_linear_weights[-1][1]
                ckpt_out_dim = int(last_linear_weight.shape[0])
                if ckpt_out_dim == 1:
                    # Regression checkpoint. Override the caller's
                    # classification default if it wasn't deliberately set.
                    if value_mode == 'classification':
                        value_mode = 'regression'
                elif ckpt_out_dim > 1:
                    # Classification checkpoint with N=ckpt_out_dim classes.
                    if value_mode == 'classification':
                        num_value_classes = ckpt_out_dim
        if num_channels is None:
            num_channels = 256
        if num_blocks is None:
            num_blocks = 12
        if value_head_type is None:
            value_head_type = 'spatial'

        self.network = YinshNetwork(
            num_channels=num_channels,
            num_blocks=num_blocks,
            value_mode=value_mode,
            num_value_classes=num_value_classes,
            input_channels=input_channels,
            value_head_type=value_head_type,
        ).to(self.device)

        # Setup logging
        self.logger = logging.getLogger("NetworkWrapper")
        self.logger.setLevel(logging.ERROR)

        # Memory Pool Management
        if tensor_pool is not None:
            self.tensor_pool = tensor_pool
            self._pool_enabled = True
        else:
            # Create a default tensor pool for this device
            pool_config = TensorPoolConfig(
                initial_size=20,  # Start with moderate pool
                enable_statistics=False,  # Keep overhead low
                enable_adaptive_sizing=True,  # Enable adaptive sizing
                enable_tensor_reshaping=True,  # Enable tensor reshaping
                auto_device_selection=True   # Enable device-specific pooling
            )
            self.tensor_pool = TensorPool(pool_config)
            self._pool_enabled = True

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        self.network.eval()  # Set to evaluation mode by default

        # Initialize appropriate encoder based on configuration. Use
        # self.use_enhanced_encoding so the auto-detect path above (which
        # may have flipped this flag based on the checkpoint state_dict)
        # actually takes effect. Reading the local arg here would leave
        # the encoder mismatched with the network and produce 0-move games.
        if self.use_enhanced_encoding:
            from ..utils.enhanced_encoding import EnhancedStateEncoder
            self.state_encoder = EnhancedStateEncoder()
            self.logger.info("Using enhanced 15-channel encoding")
        else:
            self.state_encoder = StateEncoder()
            self.logger.info("Using basic 6-channel encoding")

        # Mirror value_head_type onto self so callers (e.g. supervisor's
        # network-recreation path) can preserve it without poking into
        # `self.network.value_head_type`.
        self.value_head_type = self.network.value_head_type
        vh_params = sum(p.numel() for p in self.network.value_head.parameters())
        self.logger.info(f"Using {self.value_head_type} value head ({vh_params:,d} params)")

        # Difficulty presets can be attached by runner for CoreML metadata
        self.difficulty_presets = None

        # Cache common tensor shapes for pooling - use appropriate channel count
        self._input_shape = (input_channels, 11, 11)  # YINSH state shape
        self._policy_size = self.state_encoder.total_moves

    def _acquire_input_tensor(self, batch_size: int = 1) -> torch.Tensor:
        """Acquire an input tensor from the pool."""
        if self._pool_enabled:
            try:
                shape = (batch_size, *self._input_shape)
                return self.tensor_pool.get(shape=shape, device=self.device, dtype=torch.float32)
            except Exception as e:
                self.logger.warning(f"Failed to acquire input tensor from pool: {e}")
                return torch.zeros(batch_size, *self._input_shape, device=self.device, dtype=torch.float32)
        else:
            return torch.zeros(batch_size, *self._input_shape, device=self.device, dtype=torch.float32)

    def _acquire_output_tensors(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Acquire output tensors (policy and value) from the pool."""
        if self._pool_enabled:
            try:
                policy_tensor = self.tensor_pool.get(
                    shape=(batch_size, self._policy_size), 
                    device=self.device, 
                    dtype=torch.float32
                )
                value_tensor = self.tensor_pool.get(
                    shape=(batch_size, 1), 
                    device=self.device, 
                    dtype=torch.float32
                )
                return policy_tensor, value_tensor
            except Exception as e:
                self.logger.warning(f"Failed to acquire output tensors from pool: {e}")
                # Fallback to creating new tensors
                policy_tensor = torch.zeros(batch_size, self._policy_size, device=self.device, dtype=torch.float32)
                value_tensor = torch.zeros(batch_size, 1, device=self.device, dtype=torch.float32)
                return policy_tensor, value_tensor
        else:
            policy_tensor = torch.zeros(batch_size, self._policy_size, device=self.device, dtype=torch.float32)
            value_tensor = torch.zeros(batch_size, 1, device=self.device, dtype=torch.float32)
            return policy_tensor, value_tensor

    def _release_tensor(self, tensor: torch.Tensor) -> None:
        """Release a tensor back to the pool."""
        if self._pool_enabled and tensor is not None:
            try:
                self.tensor_pool.release(tensor)
            except Exception as e:
                self.logger.warning(f"Failed to release tensor to pool: {e}")

    def _release_tensors(self, *tensors: torch.Tensor) -> None:
        """Release multiple tensors back to the pool."""
        for tensor in tensors:
            self._release_tensor(tensor)

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

            # PHASE 1.5 FIX: Value is already tanh'd by the network (model.py:139)
            # Don't apply tanh again - causes over-compression (tanh(tanh(x)))
            # value = torch.tanh(value)  # REMOVED

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
        """Save model weights to file.

        Sanity-checks BatchNorm running stats before writing. The state
        dict must include `running_mean` / `running_var` for every BN
        layer; if any are missing, something upstream (e.g. a memory-
        cleanup pass) deregistered the buffer, and the resulting file
        will produce wildly wrong inference on reload — this was the
        cloud_run_v1 50/0/0 failure mode (see CLOUD_RUN_V1_POSTMORTEM).
        Refuse to write a corrupted checkpoint rather than silently
        poison the next run that loads it.
        """
        try:
            sd = self.network.state_dict()

            # Expected: every nn.BatchNorm{1,2}d module contributes a
            # running_mean and running_var. Count BN modules and compare.
            import torch.nn as nn
            bn_count = sum(
                1 for m in self.network.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            )
            saved_running_mean = sum(1 for k in sd if k.endswith('.running_mean'))
            saved_running_var = sum(1 for k in sd if k.endswith('.running_var'))
            if bn_count > 0 and (saved_running_mean < bn_count or saved_running_var < bn_count):
                raise RuntimeError(
                    f"refusing to save checkpoint with missing BN running stats: "
                    f"network has {bn_count} BN modules but state_dict has only "
                    f"{saved_running_mean} running_mean / {saved_running_var} "
                    f"running_var keys. The cloud_run_v1 50/0/0 bug. Don't save."
                )

            torch.save(sd, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    # def load_model(self, path: str):
    #     try:
    #         state_dict = torch.load(path, map_location=self.device)
    #         self.network.load_state_dict(state_dict, strict=False)
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to load model: {e}")

    def load_model(self, path: str):
        """Load model with architecture adaptation."""
        try:
            state_dict = torch.load(path, map_location=self.device)

            # Hard-fail early on encoder mismatch: the input conv's in_channels
            # is a load-bearing invariant. Silent filtering would leave it randomly
            # initialized and corrupt inference.
            input_conv_key = next(
                (k for k in state_dict if k.endswith('conv_block.0.weight')), None
            )
            if input_conv_key is not None:
                ckpt_in = state_dict[input_conv_key].shape[1]
                expected_in = 15 if self.use_enhanced_encoding else 6
                if ckpt_in != expected_in:
                    raise RuntimeError(
                        f"Encoder channel mismatch: checkpoint has {ckpt_in} input channels "
                        f"but wrapper was constructed with use_enhanced_encoding="
                        f"{self.use_enhanced_encoding} (expects {expected_in}). "
                        f"Re-instantiate NetworkWrapper with the matching flag."
                    )

            # Hard-fail on value-head mode mismatch. The state dict shape may be
            # compatible enough to silently load (or get filtered out below), but
            # the value-head semantics differ: classification outputs softmax over
            # outcome classes, regression outputs a tanh'd scalar. Loading across
            # modes corrupts inference and the trainer's loss surface.
            ckpt_has_outcome_values = any(
                k.endswith('outcome_values') for k in state_dict
            )
            ckpt_mode = 'classification' if ckpt_has_outcome_values else 'regression'
            if ckpt_mode != self.network.value_mode:
                raise RuntimeError(
                    f"Value-head mode mismatch: checkpoint is {ckpt_mode} but wrapper "
                    f"was constructed with value_mode='{self.network.value_mode}'. "
                    f"Re-instantiate NetworkWrapper with the matching value_mode."
                )
            if ckpt_mode == 'classification':
                outcome_key = next(
                    k for k in state_dict if k.endswith('outcome_values')
                )
                ckpt_classes = state_dict[outcome_key].shape[0]
                if ckpt_classes != self.network.num_value_classes:
                    raise RuntimeError(
                        f"num_value_classes mismatch: checkpoint has {ckpt_classes} "
                        f"classes but wrapper was constructed with "
                        f"num_value_classes={self.network.num_value_classes}. "
                        f"Re-instantiate NetworkWrapper with the matching value."
                    )

            # Hard-fail on policy-head size mismatch. The final policy Linear's
            # out_features encodes the move-index layout: different sizes mean
            # different move→index meaning, and the silent shape-filter below
            # would randomly-initialize the policy output, playing garbage.
            #
            # Known past sizes (policy-head out_features):
            #   7395 — pre-move-encoder-rework legacy layout
            #   8390 — post-rework, collision-free ring-movement, 1080-slot
            #          REMOVE_MARKERS sequence hash (had 17 collisions out of
            #          123 valid 5-in-a-row lines and an incorrect inverse
            #          that fabricated an illegal diagonal)
            #   7433 — current layout: collision-free REMOVE_MARKERS
            #          (123 slots, one per valid hex 5-line). Changing the
            #          REMOVE_MARKERS sub-layout is a BREAKING change for any
            #          saved checkpoint — even if total_moves were identical,
            #          the slot ↔ line mapping differs, so MCTS / policy
            #          logits would silently route to wrong moves. This guard
            #          fires on total-size change; any intra-size layout swap
            #          must add a layout-version tag to the checkpoint.
            policy_out_key = next(
                (k for k in state_dict if k.endswith('policy_head.7.weight')), None
            )
            if policy_out_key is not None:
                ckpt_policy_size = state_dict[policy_out_key].shape[0]
                expected_policy_size = self.network.total_moves
                if ckpt_policy_size != expected_policy_size:
                    raise RuntimeError(
                        f"Policy-head size mismatch: checkpoint has "
                        f"{ckpt_policy_size} output slots but the current encoder "
                        f"emits {expected_policy_size}. Known past sizes: 7395 "
                        f"(pre-rework), 8390 (collision-prone REMOVE_MARKERS), "
                        f"7433 (current, collision-free 123-slot REMOVE_MARKERS). "
                        f"Retrain from scratch or migrate the policy head "
                        f"explicitly — silently filtering this layer would "
                        f"randomly re-initialize the policy output."
                    )

            compatible_state_dict = {}
            model_state = self.network.state_dict()
            for key, param in state_dict.items():
                if key in model_state and param.shape == model_state[key].shape:
                    compatible_state_dict[key] = param

            self.network.load_state_dict(compatible_state_dict, strict=False)

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def export_to_coreml(self, path: str):
        """Export model to CoreML format."""
        try:
            # Set model to eval mode and move to CPU
            self.network.eval().to('cpu')

            # Create example input on CPU
            example_input = torch.randn(1, self._input_shape[0], 11, 11).to('cpu')

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
            # Attach user metadata for difficulty presets if available
            try:
                if self.difficulty_presets:
                    # coremltools models have user_defined_metadata
                    md = mlmodel.user_defined_metadata or {}
                    # Keep it compact; store as JSON string
                    import json
                    md.update({
                        'yinsh_difficulty_presets': json.dumps(self.difficulty_presets)
                    })
                    mlmodel.user_defined_metadata = md
            except Exception as _:
                # Ignore metadata failures
                pass

            mlmodel.save(path)
            self.logger.info(f"Model exported to CoreML format at {path}")

            # Move model back to its original device (CUDA / MPS / CPU).
            # Previously hardcoded to 'mps' which crashed on CUDA boxes.
            self.network.to(self.device)

        except Exception as e:
            self.logger.error(f"Error exporting to CoreML: {str(e)}")
            raise

    def predict_from_state(self, game_state,
                          move_mask: Optional[torch.Tensor] = None,
                          temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make a prediction directly from a GameState using tensor pooling.

        Args:
            game_state: GameState object to encode and predict
            move_mask: Optional mask for valid moves (True for valid moves)
            temperature: Temperature for move probability scaling

        Returns:
            Tuple of (move_probabilities, value)
        """
        # Acquire input tensor from pool
        input_tensor = self._acquire_input_tensor(batch_size=1)

        try:
            # Encode the state into the pooled tensor
            state_array = self.state_encoder.encode_state(game_state)
            input_tensor[0] = torch.from_numpy(state_array).float()

            # Use the regular predict method
            result = self.predict(input_tensor, move_mask, temperature)

            return result

        finally:
            # Always release the input tensor
            self._release_tensor(input_tensor)

    def predict_batch(self, game_states: List, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for multiple game states in a single forward pass.

        This is the key optimization for batched MCTS - instead of evaluating
        each leaf node individually, we collect many leaves and evaluate them
        all at once, dramatically improving throughput.

        Args:
            game_states: List of GameState objects to encode and predict
            temperature: Temperature for move probability scaling

        Returns:
            Tuple of (batch_move_probabilities, batch_values) where:
                - batch_move_probabilities: tensor of shape (batch_size, num_moves)
                - batch_values: tensor of shape (batch_size, 1)
        """
        if not game_states:
            raise ValueError("Cannot predict on empty batch of game states")

        batch_size = len(game_states)

        # Acquire batch input tensor from pool
        batch_tensor = self._acquire_input_tensor(batch_size=batch_size)

        try:
            # Encode all states into the batch tensor
            for i, game_state in enumerate(game_states):
                state_array = self.state_encoder.encode_state(game_state)
                batch_tensor[i] = torch.from_numpy(state_array).float()

            # Single batched forward pass through the network
            self.network.eval()
            with torch.no_grad():
                policy_logits, values = self.network(batch_tensor)

                # Value is already in [-1, 1] for both modes:
                # - classification: softmax * outcome_values (model.py:319)
                # - regression: final Tanh layer (model.py:191)
                # Re-applying tanh here over-compresses (matches the predict() fix).

                # Apply temperature scaling to policy logits
                if temperature != 1.0:
                    policy_logits = policy_logits / temperature

                # Return raw policy logits and values
                # Note: Caller should apply softmax and masking as needed
                return policy_logits, values

        finally:
            # Always release the batch tensor
            self._release_tensor(batch_tensor)

    def cleanup(self) -> None:
        """
        Explicitly release all resources held by this wrapper.

        Call this method when you're done using the NetworkWrapper to ensure
        all tensor pool memory is released immediately. This is particularly
        important in tournament scenarios where many models are loaded/unloaded.
        """
        import gc

        # Clear tensor pool if we own it
        if self._pool_enabled and hasattr(self, 'tensor_pool') and self.tensor_pool is not None:
            try:
                cleared = self.tensor_pool.clear_all()
                if cleared > 0:
                    self.logger.debug(f"NetworkWrapper cleanup: cleared {cleared} tensors")
            except Exception as e:
                self.logger.warning(f"Error during tensor pool cleanup: {e}")

        # CRITICAL: Synchronize MPS before moving model to ensure all GPU ops complete
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except Exception:
                pass

        # Move model to CPU to free GPU memory
        if self.network is not None:
            try:
                self.network.cpu()
            except Exception as e:
                self.logger.warning(f"Error moving network to CPU: {e}")

        # Clear any stored tensors in the model
        if self.network is not None:
            try:
                if hasattr(self.network, 'value_head_activations'):
                    self.network.value_head_activations.clear()
                if hasattr(self.network, '_value_logits'):
                    self.network._value_logits = None
            except Exception:
                pass

        # Force garbage collection
        gc.collect()

        # Clear device cache with synchronization
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:
                pass
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Second GC pass after GPU cleanup
        gc.collect()

