"""Regression test: `_reset_network_objects` must rebind the EMA shadow's
`module` reference to the new nn.Module.

Pre-fix bug: when `_reset_network_objects()` rebuilt the wrapper every
3 iterations, it updated `self.network`, `self.self_play.network`, and
`self.trainer.network` — but left `self.trainer.ema.module` pointing at
the now-orphaned OLD nn.Module. EMAShadow.update() then iterates a frozen
state_dict on every train_step, and EMAShadow.swap_into() at checkpoint
save time loads those frozen values into the live network, which froze
BN running stats.

Symptom in the 2026-05-07 cloud_run_v1 rerun: `num_batches_tracked`
stuck at 2084 from iter 5 onward, despite continued training. EMA
checkpoints carried stale BN; live checkpoints failed the BN-count save
guard intermittently. See overnight_watch_log.md for the full trace.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.ema import EMAShadow


def test_reset_network_objects_rebinds_ema_module():
    """_reset_network_objects must rebind ema.module to the new nn.Module."""
    # Lazy import — TrainingSupervisor pulls in heavy deps at module load.
    from yinsh_ml.training.supervisor import TrainingSupervisor

    # Build a minimal supervisor stand-in. We're testing one method, so
    # we mock everything except the bits that method touches.
    nw_old = NetworkWrapper(device="cpu")
    ema = EMAShadow(nw_old.network, decay=0.9)

    stub = MagicMock()
    stub.network = nw_old
    stub.self_play = MagicMock()
    stub.trainer = MagicMock()
    stub.trainer.ema = ema
    stub.trainer.network = nw_old
    stub.tensor_pool = None  # NetworkWrapper will spin up a default
    import logging
    stub.logger = logging.getLogger("test.MiniSupervisor")

    # Pre-condition: ema.module is the OLD module instance.
    assert ema.module is nw_old.network

    # Bind the real method and run it.
    TrainingSupervisor._reset_network_objects(stub)
    TrainingSupervisor._reinitialize_optimizers(stub)  # called inside the real one

    # Post-condition: stub.network is a new wrapper, and ema.module
    # follows it. If this assertion fails, the BN-freeze regression has
    # come back.
    assert stub.network is not nw_old, (
        "_reset_network_objects didn't actually rebuild the wrapper"
    )
    assert ema.module is stub.network.network, (
        f"ema.module is still pointing at the OLD nn.Module after reset. "
        f"This is the cloud_run_v1 nbt-frozen-at-2084 regression. "
        f"Expected: ema.module is stub.network.network ({id(stub.network.network)}); "
        f"got: ema.module is {id(ema.module)} (likely the dead OLD module)."
    )


def test_reset_network_objects_handles_no_ema():
    """If trainer.ema is None (EMA disabled), reset should still succeed."""
    from yinsh_ml.training.supervisor import TrainingSupervisor

    nw_old = NetworkWrapper(device="cpu")

    stub = MagicMock()
    stub.network = nw_old
    stub.self_play = MagicMock()
    stub.trainer = MagicMock()
    stub.trainer.ema = None  # No EMA shadow at all
    stub.trainer.network = nw_old
    stub.tensor_pool = None
    import logging
    stub.logger = logging.getLogger("test.MiniSupervisor")

    # Should not raise (the rebinding code is gated on `ema is not None`).
    TrainingSupervisor._reset_network_objects(stub)
    TrainingSupervisor._reinitialize_optimizers(stub)
