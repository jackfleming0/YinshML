"""Regression test: NetworkWrapper.save_model refuses to write a checkpoint
that's missing BatchNorm running stats.

If something upstream (a future memory-cleanup pass, a buggy refactor,
etc.) deregisters BN buffers, the save guard catches it before the
corrupted file lands on disk and poisons every downstream load. See
CLOUD_RUN_V1_POSTMORTEM.md for why this matters.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from yinsh_ml.network.wrapper import NetworkWrapper


def _null_all_buffers(network: torch.nn.Module) -> None:
    """Reproduce the pre-fix bug: nullify every module buffer."""
    for module in network.modules():
        if hasattr(module, "_buffers"):
            for key in list(module._buffers.keys()):
                buf = module._buffers[key]
                if buf is not None and torch.is_tensor(buf):
                    module._buffers[key] = None


def test_save_model_refuses_to_write_without_bn_running_stats():
    """save_model raises rather than ship a checkpoint that BN-fails on reload."""
    nw = NetworkWrapper(device="cpu")
    _null_all_buffers(nw.network)  # simulate the cloud_run_v1 bug

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "should_not_land.pt")
        with pytest.raises(RuntimeError, match="missing BN running stats"):
            nw.save_model(path)

        # File must not exist — refusal happens before torch.save runs.
        assert not os.path.exists(path), (
            "save_model wrote a corrupted file before raising; the guard must "
            "fail-fast before disk I/O."
        )


def test_save_model_writes_when_bn_stats_present():
    """A normally-initialised network saves cleanly."""
    nw = NetworkWrapper(device="cpu")
    # Populate BN running stats.
    nw.network.train()
    x = torch.randn(2, 6, 11, 11)
    for _ in range(2):
        nw.network(x)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "good.pt")
        nw.save_model(path)
        assert os.path.exists(path)
        sd = torch.load(path, map_location="cpu")
        running_means = sum(1 for k in sd if k.endswith(".running_mean"))
        assert running_means > 0, "saved file should include BN running_mean keys"
