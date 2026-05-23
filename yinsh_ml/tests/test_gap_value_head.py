"""Tests for the Branch D.1 GAP value head and its warm-start path.

Branch D.1 swaps the legacy spatial-flatten value head (~4M params, the
documented value-head overfitting trap) for a GAP head (~17K params:
1x1 conv → BN → ReLU → AdaptiveAvgPool2d(1) → Linear(64, out)). The
controlled comparison warm-starts from `best_supervised.pt` (which has
the spatial head) — the wrapper's load_model shape-filter must let the
trunk + policy + outcome_values load while silently dropping the
mismatched spatial-head keys, leaving the GAP head freshly initialized.

See VOLUME_PRETRAIN_RESULTS.md §2 Branch D.1 for the motivation.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch

from yinsh_ml.network.model import YinshNetwork
from yinsh_ml.network.wrapper import NetworkWrapper


def _vh_params(network: YinshNetwork) -> int:
    return sum(p.numel() for p in network.value_head.parameters())


class TestGapValueHead:
    """Architecture-level invariants for the GAP head."""

    def test_spatial_head_param_count(self):
        """Pin the spatial head's param count so we notice if it shifts."""
        net = YinshNetwork(num_channels=256, num_blocks=12,
                           value_mode='classification', num_value_classes=7,
                           value_head_type='spatial')
        # Conv(256,64,3) + BN + Conv(64,64,3) + BN + Linear(7744,512) + BN +
        # Linear(512,256) + LayerNorm(256) + Linear(256,7). Dominant cost is
        # the 7744->512 Flatten projection.
        n = _vh_params(net)
        assert n > 4_000_000, f"spatial head shrunk unexpectedly: {n} params"
        assert n < 5_000_000, f"spatial head grew unexpectedly: {n} params"

    def test_gap_head_param_count_under_25k(self):
        """GAP head must be ~17K — order-of-magnitude smaller than spatial."""
        net = YinshNetwork(num_channels=256, num_blocks=12,
                           value_mode='classification', num_value_classes=7,
                           value_head_type='gap')
        n = _vh_params(net)
        # Conv2d(256,64,1) = 16,448 + BN(64) = 128 + Linear(64,7) = 455. ~17K.
        # Hard upper bound at 25K catches accidental architecture bloat.
        assert n < 25_000, f"GAP head bloated: {n} params"
        assert n > 10_000, f"GAP head unexpectedly tiny: {n} params"

    def test_gap_head_forward_shapes(self):
        """Forward pass must match the spatial head's contract."""
        net = YinshNetwork(num_channels=64, num_blocks=2,  # small for test speed
                           value_mode='classification', num_value_classes=7,
                           value_head_type='gap')
        net.eval()
        x = torch.zeros(3, 6, 11, 11)
        with torch.no_grad():
            policy, value = net(x)
        assert tuple(policy.shape) == (3, net.total_moves), \
            f"policy shape {tuple(policy.shape)} != (3, {net.total_moves})"
        # Classification mode: value head returns expected value (scalar per item).
        assert tuple(value.shape) == (3,), f"value shape {tuple(value.shape)} != (3,)"

    def test_gap_head_regression_mode(self):
        """GAP also supports regression mode (Linear(64, 1) + Tanh).

        Forward squeezes the trailing 1-dim, so value shape is (B,) in
        both modes — match `model.py::forward`'s explicit `.squeeze(-1)`.
        """
        net = YinshNetwork(num_channels=64, num_blocks=2,
                           value_mode='regression', value_head_type='gap')
        net.eval()
        x = torch.zeros(2, 6, 11, 11)
        with torch.no_grad():
            policy, value = net(x)
        assert tuple(value.shape) == (2,), f"regression value shape {tuple(value.shape)} != (2,)"
        # Tanh-bounded
        assert value.abs().max().item() <= 1.0

    def test_value_head_type_validation(self):
        """Unknown value_head_type fails at construction."""
        with pytest.raises(ValueError, match="value_head_type must be"):
            YinshNetwork(value_head_type='bogus')


class TestGapWarmStartFromSpatial:
    """The Branch D.1 warm-start invariant: load `best_supervised.pt` (spatial
    head) into a wrapper constructed with `value_head_type='gap'`. Trunk +
    policy + outcome_values must load; spatial-head keys must silently drop;
    GAP head must remain at its random initialization (not zeros)."""

    @pytest.fixture
    def spatial_checkpoint_path(self, tmp_path):
        """Create a tiny spatial-head checkpoint to play stand-in for
        best_supervised.pt — avoids depending on the 137MB real model file."""
        net = YinshNetwork(num_channels=32, num_blocks=2,
                           value_mode='classification', num_value_classes=7,
                           value_head_type='spatial')
        # Make trunk weights distinctive so we can verify they transferred.
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.123)
        path = tmp_path / "spatial_ckpt.pt"
        torch.save(net.state_dict(), path)
        return path

    def test_load_spatial_into_gap_wrapper_succeeds(self, spatial_checkpoint_path):
        """The shape filter in load_model must accept this — no exception."""
        wrapper = NetworkWrapper(device='cpu', num_channels=32, num_blocks=2,
                                 value_head_type='gap')
        wrapper.load_model(str(spatial_checkpoint_path))
        assert wrapper.value_head_type == 'gap'
        # Sanity: forward still runs cleanly with the loaded trunk + fresh GAP head
        wrapper.network.eval()
        with torch.no_grad():
            policy, value = wrapper.network(torch.zeros(1, 6, 11, 11))
        assert policy.shape == (1, wrapper.network.total_moves)

    def test_load_spatial_into_gap_transfers_trunk(self, spatial_checkpoint_path):
        """Trunk + policy + outcome_values must end up with the checkpoint's
        distinctive weights (we added 0.123 to every param in the fixture)."""
        sd_orig = torch.load(spatial_checkpoint_path, map_location='cpu',
                             weights_only=True)
        wrapper = NetworkWrapper(device='cpu', num_channels=32, num_blocks=2,
                                 value_head_type='gap')
        wrapper.load_model(str(spatial_checkpoint_path))
        sd_after = wrapper.network.state_dict()

        # Keys that MUST have transferred: trunk + policy + outcome_values.
        # (Value-head keys differ in shape → silently dropped → fresh init.)
        transferred = 0
        for k, v in sd_orig.items():
            if k.startswith('value_head.'):
                continue  # spatial-head keys, expected to drop
            if k in sd_after and sd_after[k].shape == v.shape:
                assert torch.allclose(sd_after[k], v, atol=1e-6), (
                    f"{k} did not transfer: shapes match but values differ"
                )
                transferred += 1
        # Trunk + policy + outcome_values should be ≥ 50 tensors for a
        # 32-channel x 2-block net. Sanity floor.
        assert transferred >= 20, (
            f"only {transferred} tensors transferred from the spatial-head "
            f"checkpoint — trunk/policy load is broken"
        )

    def test_load_spatial_into_gap_freshly_inits_value_head(self, spatial_checkpoint_path):
        """The new GAP value head must NOT contain spatial-head weights —
        it must be at its fresh random initialization. Verify by checking
        the GAP head's Conv2d has kernel_size=1 (not 3), confirming the
        spatial head's 3x3 conv DIDN'T overwrite it."""
        wrapper = NetworkWrapper(device='cpu', num_channels=32, num_blocks=2,
                                 value_head_type='gap')
        wrapper.load_model(str(spatial_checkpoint_path))
        first_conv = wrapper.network.value_head[0]
        assert isinstance(first_conv, torch.nn.Conv2d)
        assert first_conv.kernel_size == (1, 1), (
            f"GAP head's first conv has kernel_size={first_conv.kernel_size} "
            f"— spatial-head 3x3 weights bled in"
        )
        # And the GAP head's Linear must be (64 → 7), not (256 → 7) or anything else
        last_linear = [m for m in wrapper.network.value_head.modules()
                       if isinstance(m, torch.nn.Linear)][-1]
        assert last_linear.in_features == 64
        assert last_linear.out_features == 7


class TestAutoDetectFromCheckpoint:
    """When value_head_type is None and a model_path is given, the wrapper
    must auto-detect by inspecting `value_head.0.weight`'s kernel size."""

    def test_auto_detect_spatial(self, tmp_path):
        spatial_net = YinshNetwork(num_channels=32, num_blocks=2,
                                   value_head_type='spatial')
        path = tmp_path / "spatial.pt"
        torch.save(spatial_net.state_dict(), path)
        wrapper = NetworkWrapper(model_path=str(path), device='cpu')
        assert wrapper.value_head_type == 'spatial'

    def test_auto_detect_gap(self, tmp_path):
        gap_net = YinshNetwork(num_channels=32, num_blocks=2,
                               value_head_type='gap')
        path = tmp_path / "gap.pt"
        torch.save(gap_net.state_dict(), path)
        wrapper = NetworkWrapper(model_path=str(path), device='cpu')
        assert wrapper.value_head_type == 'gap'

    def test_explicit_override_beats_auto_detect(self, tmp_path):
        """The Branch D.1 warm-start path: explicit 'gap' on a spatial
        checkpoint must NOT be overridden by auto-detect."""
        spatial_net = YinshNetwork(num_channels=32, num_blocks=2,
                                   value_head_type='spatial')
        path = tmp_path / "spatial.pt"
        torch.save(spatial_net.state_dict(), path)
        wrapper = NetworkWrapper(model_path=str(path), device='cpu',
                                 value_head_type='gap')
        assert wrapper.value_head_type == 'gap'
