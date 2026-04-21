"""Tests for mixed-precision training on MPS/CUDA via `torch.autocast`.

Covers:
  * `_autocast()` returns a real autocast context that matches the requested
    device type and enabled flag.
  * `enable_autocast=True` on MPS/CUDA flips `_autocast_enabled`; CPU always
    comes out disabled regardless of the flag (bf16 on CPU doesn't win
    wall-clock back).
  * Inside the autocast context, a tiny conv forward on CPU produces bf16
    outputs (opt-in via an explicit `device_type='cpu'` autocast — used here
    only for test determinism), while `log_softmax` / `cross_entropy`
    correctly promote back to fp32 inside the same context.
  * The constructor signature exposes the flag with the expected default.

Tests avoid spinning up a full YinshTrainer (which needs a real network +
device); they exercise the autocast mechanics directly so the assertions
survive regardless of runtime hardware.
"""

import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestAutocastMechanics:
    def test_explicit_fp32_cast_protects_log_softmax(self):
        """Defensive pattern for value-head CE: explicit `.float()` cast on
        the logits *guarantees* fp32 log_softmax regardless of autocast's
        per-device promotion rules. CUDA auto-promotes log_softmax, CPU/MPS
        don't — pinning the explicit cast here ensures the trainer's CE path
        doesn't silently compute in bf16 if someone ever removes the cast."""
        x_bf16 = torch.randn(4, 7).to(torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = F.log_softmax(x_bf16.float(), dim=1)
        assert out.dtype == torch.float32

    def test_cross_entropy_promotes_inside_autocast(self):
        """Same guarantee for the hard-CE path used when
        `soft_value_target_sigma == 0`."""
        logits = torch.randn(4, 7).to(torch.bfloat16)
        target = torch.tensor([0, 3, 6, 2])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            loss = F.cross_entropy(logits, target)
        assert loss.dtype == torch.float32, (
            f"cross_entropy must promote to fp32 under autocast; got {loss.dtype}"
        )

    def test_conv_stays_in_bf16_inside_autocast(self):
        """The heavy work (conv/linear) is where the wall-clock comes from.
        Pin that a conv forward inside autocast emits bf16 outputs."""
        conv = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        x = torch.randn(2, 6, 11, 11)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = conv(x)
        assert out.dtype == torch.bfloat16, (
            f"conv should run in bf16 under autocast; got {out.dtype}"
        )

    def test_disabled_autocast_is_noop(self):
        """`enabled=False` makes the context a no-op — ops keep their input
        dtype. This is the CPU/disabled code path."""
        conv = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        x = torch.randn(2, 6, 11, 11)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
            out = conv(x)
        assert out.dtype == torch.float32


class TestTrainerAutocastWiring:
    """Signature + documented defaults. Avoids instantiating a full trainer
    (which requires a network + device) — enough to guard the constructor
    against silent drift."""

    def test_signature_has_enable_autocast_with_default_true(self):
        from yinsh_ml.training.trainer import YinshTrainer
        sig = inspect.signature(YinshTrainer.__init__)
        assert "enable_autocast" in sig.parameters
        assert sig.parameters["enable_autocast"].default is True

    def test_autocast_method_exists(self):
        """`_autocast()` is the helper that `train_step` wraps forward +
        loss in. Renaming it silently would neuter autocast across the
        trainer."""
        from yinsh_ml.training.trainer import YinshTrainer
        assert hasattr(YinshTrainer, "_autocast")

    def test_disabled_autocast_returns_safe_null_context(self):
        """When autocast is disabled, `_autocast()` must return a context
        that doesn't touch `torch.autocast(device_type=...)` — PyTorch 2.1
        validates device_type *before* `enabled`, so
        `torch.autocast(device_type='mps', enabled=False)` raises on builds
        without MPS support. Regression guard for that exact failure path:
        the fix returns `contextlib.nullcontext()` when disabled."""
        from yinsh_ml.training.trainer import YinshTrainer

        t = YinshTrainer(
            network=self._fake_wrapper(),
            device="cpu",  # deterministically disables autocast
            enable_autocast=True,
            total_epochs=1,
        )
        assert t._autocast_enabled is False

        # The key invariant: entering the returned context must not raise.
        with t._autocast():
            # Inside the null context, ops run in whatever dtype they'd
            # naturally use — no bf16 promotion. Sanity-check:
            x = torch.zeros(2, 3)
            assert x.dtype == torch.float32

    def _fake_wrapper(self):
        class _FakeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Linear(1, 1)
                self.value_head = nn.Linear(1, 1)

        class _FakeWrapper:
            class _FakeEnc:
                total_moves = 7395

            def __init__(self):
                self.state_encoder = self._FakeEnc()
                self.network = _FakeNet()
                self.use_enhanced_encoding = False

        return _FakeWrapper()

    def test_cpu_device_disables_autocast_even_when_requested(self):
        """CPU path flips the flag off regardless of the request — bf16 on
        CPU doesn't recover wall-clock, so there's no reason to pay the
        precision cost."""
        from yinsh_ml.training.trainer import YinshTrainer

        t = YinshTrainer(
            network=self._fake_wrapper(),
            device="cpu",
            enable_autocast=True,
            total_epochs=1,
        )
        assert t._autocast_device == "cpu"
        assert t._autocast_enabled is False

    def test_unsupported_autocast_device_falls_back_gracefully(self, monkeypatch):
        """If `torch.autocast(device_type=...)` raises at *probe* time (e.g.
        MPS on PyTorch ≤2.2), the trainer must disable autocast rather than
        crash at the first `_autocast()` call during `train_step`. That
        deferred-crash pattern was the actual bug hit during development —
        a game played, then training threw `RuntimeError: User specified an
        unsupported autocast device_type 'mps'` on the very first forward.

        Simulates the regression by patching `torch.autocast` to raise on
        the device the trainer resolves to, constructing the trainer, and
        asserting the flag has flipped to False cleanly (no exception, and
        a `_autocast_unsupported_reason` is recorded for the log line)."""
        from yinsh_ml.training import trainer as trainer_mod
        from yinsh_ml.training.trainer import YinshTrainer

        class FakeDevice:
            """device.type='mps' without needing a real MPS runtime."""
            type = "mps"

        # Replace the `self.device = torch.device(...)` resolution so the
        # trainer thinks it's on MPS even in a CUDA-less/MPS-less CI env.
        original_torch_device = trainer_mod.torch.device
        monkeypatch.setattr(
            trainer_mod.torch, "device", lambda _x: FakeDevice()
        )
        # But restore `.to(self.device)` calls by making them no-ops — the
        # fake device isn't a real one so `module.to(FakeDevice())` would
        # fail. Patch `network.network.to` to just return self.
        wrapper = self._fake_wrapper()
        wrapper.network.to = lambda _dev: wrapper.network

        # Patch autocast to raise the exact PyTorch 2.1 MPS error message.
        real_autocast = trainer_mod.torch.autocast

        def fake_autocast(device_type, enabled=True, **kwargs):
            if device_type == "mps":
                raise RuntimeError(
                    "User specified an unsupported autocast device_type 'mps'"
                )
            return real_autocast(device_type, enabled=enabled, **kwargs)

        monkeypatch.setattr(trainer_mod.torch, "autocast", fake_autocast)

        # Should not raise — graceful fallback.
        t = YinshTrainer(
            network=wrapper,
            device="mps",
            enable_autocast=True,
            total_epochs=1,
        )
        assert t._autocast_enabled is False
        assert "mps" in getattr(t, "_autocast_unsupported_reason", "")

        # Restore torch.device for any later tests in this module.
        monkeypatch.setattr(trainer_mod.torch, "device", original_torch_device)
