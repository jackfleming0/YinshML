"""Capture value-head activations for SAE training (Track B §8).

The value head is `nn.Sequential` defined at `network/model.py:139-157`. The
penultimate activation — the 256-dim ReLU output at index 13 (line 153) — is
the right hook point: it's the last representation before the small Linear
that maps to `num_value_classes` logits, so it carries everything the value
head's final classifier sees.

The model already auto-registers forward hooks on every Linear/ReLU/LayerNorm
in the value_head (model.py:213-215) into `self.value_head_activations[name]`,
keyed by the module's string name in the Sequential. We lean on that — no
new hooks needed. We just read key '13' after each forward pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from ..network.wrapper import NetworkWrapper

# Index of the penultimate ReLU in `YinshNetwork.value_head` (Sequential).
# See class header docstring for the per-element mapping.
PENULTIMATE_KEY = '13'
PENULTIMATE_DIM = 256


class ValueHeadActivationCapture:
    """Run encoded positions through a network checkpoint, recording the
    256-dim penultimate activation per position alongside the encoded state.

    Usage:
        cap = ValueHeadActivationCapture(network)
        for batch in batches_of_encoded_states:
            cap.capture(batch)  # accumulates internally
        cap.save(path)  # writes activations.npy + states.npy + meta.json
    """

    def __init__(self, network: NetworkWrapper, device: Optional[str] = None):
        self.network = network
        self.network.network.eval()
        self.device = torch.device(device) if device else next(network.network.parameters()).device
        self.network.network.to(self.device)

        self._activations: List[np.ndarray] = []
        self._states: List[np.ndarray] = []
        self._verified = False

    def _verify_hook_present(self) -> None:
        """First-call sanity check: confirm the model exposes the activation
        dict and the penultimate key after a forward pass. If the model
        architecture drifts (e.g., value_head depth changes) and the index
        is no longer the penultimate ReLU, we want a loud error here, not a
        silently-wrong SAE training run downstream."""
        if self._verified:
            return
        if not hasattr(self.network.network, 'value_head_activations'):
            raise RuntimeError(
                "Model does not expose `value_head_activations` dict — "
                "expected forward hooks at network/model.py:213-215. "
                "If the model architecture changed, update activation_capture."
            )
        # Smoke-test forward pass with a single zero-input.
        with torch.no_grad():
            sample = torch.zeros(1, *self._infer_input_shape(), device=self.device)
            _ = self.network.network(sample)
        acts = self.network.network.value_head_activations
        if PENULTIMATE_KEY not in acts:
            raise RuntimeError(
                f"Penultimate activation key {PENULTIMATE_KEY!r} not in "
                f"value_head_activations (got keys: {sorted(acts.keys())}). "
                f"Either the value_head structure changed or the hook wiring did."
            )
        observed_dim = acts[PENULTIMATE_KEY].shape[-1]
        if observed_dim != PENULTIMATE_DIM:
            raise RuntimeError(
                f"Expected penultimate dim {PENULTIMATE_DIM} but observed "
                f"{observed_dim}. Update PENULTIMATE_DIM to match."
            )
        self._verified = True

    def _infer_input_shape(self) -> tuple:
        """Pull the expected input channels from the wrapper's encoder.
        Always returns (C, 11, 11) — board geometry is fixed."""
        return (self.network.network.input_channels, 11, 11)

    def capture(self, encoded_states: np.ndarray) -> np.ndarray:
        """Run a batch of encoded states through the network. Returns the
        captured activations (also accumulated for later save).

        Args:
            encoded_states: (N, C, 11, 11) numpy array of encoded states.

        Returns:
            (N, 256) numpy array of penultimate activations.
        """
        self._verify_hook_present()
        if encoded_states.ndim != 4:
            raise ValueError(
                f"Expected (N, C, 11, 11) input; got shape {encoded_states.shape}."
            )

        states_tensor = torch.from_numpy(encoded_states).float().to(self.device)
        with torch.no_grad():
            _ = self.network.network(states_tensor)
        acts_tensor = self.network.network.value_head_activations[PENULTIMATE_KEY]
        acts_np = acts_tensor.detach().cpu().numpy().astype(np.float32)
        # Clear hook dict to avoid the same memory-leak the trainer guards
        # against in the main path (trainer.py:835-836).
        self.network.network.value_head_activations.clear()

        self._activations.append(acts_np)
        self._states.append(encoded_states.astype(np.float32))
        return acts_np

    def capture_iter(self, batches: Iterable[np.ndarray]) -> None:
        """Convenience wrapper to drive `capture` over an iterator of batches."""
        for batch in batches:
            self.capture(batch)

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate all captured batches and return (activations, states).
        Idempotent — safe to call multiple times."""
        if not self._activations:
            raise RuntimeError("No activations captured yet. Call capture() first.")
        acts = np.concatenate(self._activations, axis=0)
        states = np.concatenate(self._states, axis=0)
        return acts, states

    def save(self, output_dir: str | Path) -> dict:
        """Persist accumulated activations + states to disk. Returns a small
        metadata dict the SAE trainer can read."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        acts, states = self.stack()
        np.save(output_dir / 'activations.npy', acts)
        np.save(output_dir / 'states.npy', states)
        meta = {
            'num_positions': int(acts.shape[0]),
            'activation_dim': int(acts.shape[1]),
            'state_shape': list(states.shape[1:]),
            'hook_key': PENULTIMATE_KEY,
            'value_head_module_index': PENULTIMATE_KEY,
        }
        import json
        with open(output_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        return meta

    def reset(self) -> None:
        """Clear accumulated batches without resetting the verified flag."""
        self._activations.clear()
        self._states.clear()
