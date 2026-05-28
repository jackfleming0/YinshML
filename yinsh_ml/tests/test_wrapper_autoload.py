"""Regression test: NetworkWrapper constructor auto-detects encoding and
architecture from the checkpoint at __init__ time, so callers don't need
to know `use_enhanced_encoding` or `num_channels` in advance.

This is the contract that lets scripts/eval_vs_frozen_anchor.py,
scripts/cross_era_tournament.py, etc. load cross-encoding checkpoints
without prior knowledge. The bug it prevents (D.2 mid-run): bare
`NetworkWrapper(device=...)` constructed a default 6-channel network,
then `.load_model(path)` of a 15-channel checkpoint hard-failed.

The fix is to pass `model_path` to `__init__` — this engages auto-detect
of input_channels, num_channels, num_blocks, and value_head_type from
the state_dict before constructing the network.
"""

from __future__ import annotations

import os
import tempfile

import torch

from yinsh_ml.network.wrapper import NetworkWrapper


def _populate_bn_stats(nw: NetworkWrapper, channels: int) -> None:
    """Forward a tiny batch so BN running stats exist (save guard requires this)."""
    nw.network.train()
    x = torch.randn(2, channels, 11, 11)
    for _ in range(2):
        nw.network(x)
    nw.network.eval()


def test_constructor_autoload_round_trips_15ch():
    """A 15-ch wrapper saves, then a NEW wrapper constructed with only
    model_path+device auto-detects encoding=enhanced and reproduces the
    exact same forward output."""
    nw_save = NetworkWrapper(device="cpu", use_enhanced_encoding=True)
    assert nw_save.use_enhanced_encoding is True
    _populate_bn_stats(nw_save, channels=15)

    x = torch.randn(1, 15, 11, 11)
    with torch.no_grad():
        p0, v0 = nw_save.network(x)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "15ch.pt")
        nw_save.save_model(path)

        # Construct fresh — NO use_enhanced_encoding flag. Auto-detect must
        # flip it on based on the checkpoint's first conv weight shape.
        nw_load = NetworkWrapper(model_path=path, device="cpu")
        assert nw_load.use_enhanced_encoding is True, (
            "constructor should have flipped use_enhanced_encoding from the "
            "15-channel checkpoint"
        )

        with torch.no_grad():
            p1, v1 = nw_load.network(x)

        assert torch.allclose(p0, p1, atol=1e-5), "policy logits diverged after autoload"
        assert torch.allclose(v0, v1, atol=1e-5), "value diverged after autoload"


def test_constructor_autoload_round_trips_6ch():
    """6-ch path is the default and should still work — guards against the
    autoload logic accidentally forcing enhanced encoding."""
    nw_save = NetworkWrapper(device="cpu", use_enhanced_encoding=False)
    assert nw_save.use_enhanced_encoding is False
    _populate_bn_stats(nw_save, channels=6)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "6ch.pt")
        nw_save.save_model(path)

        nw_load = NetworkWrapper(model_path=path, device="cpu")
        assert nw_load.use_enhanced_encoding is False, (
            "constructor should have left use_enhanced_encoding=False for the "
            "6-channel checkpoint"
        )


def test_constructor_autoload_detects_regression_value_mode():
    """A regression-mode 15-ch checkpoint round-trips through the constructor
    auto-detect path without the caller passing value_mode.

    This is the A4 phase 1.5 wiring: regression-trained pretrains saved via
    `--value-mode regression` must load cleanly into self-play (run_training.py
    constructs the wrapper without passing value_mode). Without this
    auto-detect, the wrapper would build a classification head (7 outputs)
    and load_model would hard-fail on the 1-output checkpoint.
    """
    nw_save = NetworkWrapper(device="cpu", use_enhanced_encoding=True, value_mode="regression")
    assert nw_save.network.value_mode == "regression"
    _populate_bn_stats(nw_save, channels=15)

    x = torch.randn(1, 15, 11, 11)
    with torch.no_grad():
        _, v0 = nw_save.network(x)
    assert v0.shape == (1,), "regression mode value output should be (B,) shape"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "15ch_regression.pt")
        nw_save.save_model(path)

        # Construct fresh — NO value_mode kwarg. Auto-detect must flip it
        # to 'regression' based on the checkpoint's last value-head Linear's
        # out_features == 1.
        nw_load = NetworkWrapper(model_path=path, device="cpu")
        assert nw_load.network.value_mode == "regression", (
            "constructor should have auto-detected regression mode from the "
            "checkpoint's last value-head layer output dim (== 1). Got "
            f"value_mode={nw_load.network.value_mode}"
        )

        with torch.no_grad():
            _, v1 = nw_load.network(x)
        assert torch.allclose(v0, v1, atol=1e-5), "regression value diverged after autoload"


def test_constructor_autoload_keeps_classification_default():
    """Classification-mode round-trip: no false-positive switch to regression.
    The default kwarg is value_mode='classification'; a 7-class checkpoint
    must keep it.
    """
    nw_save = NetworkWrapper(device="cpu", use_enhanced_encoding=True)  # default mode
    assert nw_save.network.value_mode == "classification"
    _populate_bn_stats(nw_save, channels=15)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "15ch_classification.pt")
        nw_save.save_model(path)

        nw_load = NetworkWrapper(model_path=path, device="cpu")
        assert nw_load.network.value_mode == "classification", (
            "constructor must keep classification when checkpoint's last "
            "value-head layer has > 1 output dim. Got "
            f"value_mode={nw_load.network.value_mode}"
        )


def test_bare_construct_then_load_15ch_into_6ch_wrapper_hard_fails():
    """The unsafe pattern (bare construct, then load mismatched checkpoint)
    must hard-fail with a clear error. This is the guard that surfaced the
    D.2 mid-run crash — we want it to STAY a hard fail so future callers
    can't silently misconfigure the encoder."""
    nw_save = NetworkWrapper(device="cpu", use_enhanced_encoding=True)
    _populate_bn_stats(nw_save, channels=15)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "15ch.pt")
        nw_save.save_model(path)

        # Bare construct (6-ch default), then try to load 15-ch checkpoint.
        nw_bad = NetworkWrapper(device="cpu")  # implicit use_enhanced_encoding=False
        assert nw_bad.use_enhanced_encoding is False
        try:
            nw_bad.load_model(path)
        except (RuntimeError, ValueError) as e:
            # Expected — the wrapper's hard-fail guard fires.
            assert "channel" in str(e).lower() or "enhanced" in str(e).lower() or "6" in str(e) or "15" in str(e), (
                f"hard-fail message should mention the channel mismatch; got: {e}"
            )
            return
        raise AssertionError(
            "load_model(15-ch ckpt) into a 6-ch wrapper should have raised, "
            "but it silently loaded. The hard-fail guard is gone — investigate."
        )
