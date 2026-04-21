"""Exponential moving average of model weights.

The tournament gate uses whichever checkpoint the supervisor saves at the
end of an iteration. Because that checkpoint is just the live training
weights after the last optimizer step, single-iteration loss dips leak
straight into gate decisions. An EMA shadow copy averages out that
step-to-step noise: tournament plays the smoothed weights while the live
net keeps training on the unsmoothed ones.

Tracks both `nn.Parameter` tensors and buffers (BatchNorm's running_mean /
running_var matter for inference stability; they're floats so they get
averaged the same way). Integer buffers (`num_batches_tracked`) are copied
through verbatim.

Usage:
    ema = EMAShadow(network, decay=0.999)
    ...
    optimizer.step()
    ema.update()
    ...
    # At checkpoint time:
    ema.swap_into(network)
    network_save_model('ckpt_ema.pt')
    ema.restore(network)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class EMAShadow:
    """Maintains a decayed moving average of a module's state_dict."""

    def __init__(self, module: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.module = module
        self.shadow: Dict[str, torch.Tensor] = {
            name: t.detach().clone() for name, t in module.state_dict().items()
        }
        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self) -> None:
        """Pull one step's worth of live weights into the shadow.

        shadow ← decay · shadow + (1 − decay) · live for float tensors;
        non-float tensors (e.g. BN `num_batches_tracked`) are just copied.
        """
        for name, live in self.module.state_dict().items():
            s = self.shadow.get(name)
            if s is None:
                continue
            if live.is_floating_point():
                s.mul_(self.decay).add_(live.detach(), alpha=1.0 - self.decay)
            else:
                s.copy_(live)

    @torch.no_grad()
    def swap_into(self, module: Optional[nn.Module] = None) -> None:
        """Copy EMA weights onto the module, stashing the live weights.

        Only one swap can be in flight at a time; call `restore` before
        swapping again.

        Tolerant of the module having a subset of the shadow's keys — the
        supervisor's between-iteration cleanup nulls out BatchNorm running
        stats (see `supervisor.py::train_iteration`), which drops those
        keys from `module.state_dict()`. We load only the intersection and
        restore only what we saved, so the swap is robust to that.
        """
        if self._backup is not None:
            raise RuntimeError("EMAShadow: swap_into called twice without restore")
        m = module if module is not None else self.module
        live_sd = m.state_dict()
        self._backup = {
            name: t.detach().clone() for name, t in live_sd.items()
        }
        to_load = {k: v for k, v in self.shadow.items() if k in live_sd}
        m.load_state_dict(to_load, strict=False)

    @torch.no_grad()
    def restore(self, module: Optional[nn.Module] = None) -> None:
        """Reinstate the live weights stashed by the last `swap_into`.

        `strict=False` in case the module's key set has drifted between
        swap and restore (e.g. buffers got re-registered in between).
        """
        if self._backup is None:
            return
        m = module if module is not None else self.module
        m.load_state_dict(self._backup, strict=False)
        self._backup = None

    def state_dict(self) -> dict:
        return {
            'decay': self.decay,
            'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state['decay'])
        incoming = state['shadow']
        for name, src in incoming.items():
            dst = self.shadow.get(name)
            if dst is None:
                continue
            dst.copy_(src.to(dst.device, dtype=dst.dtype))
