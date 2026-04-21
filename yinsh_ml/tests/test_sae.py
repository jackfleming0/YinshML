"""Tests for the value-head SAE probe (Track B §8).

Covers:
  * SAE round-trip math: encode → decode reconstructs reasonably; sparsity
    pressure works; output shapes correct.
  * Activation capture verifies the value-head hook structure is what we
    expect (will fail loudly if the model architecture drifts).
  * Feature analysis: top-K positions per feature, dead-feature counts,
    confident-error mining.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.interpretability.activation_capture import (
    PENULTIMATE_DIM,
    PENULTIMATE_KEY,
    ValueHeadActivationCapture,
)
from yinsh_ml.interpretability.feature_analysis import (
    FeatureAnalyzer,
    save_report,
)
from yinsh_ml.interpretability.sae import (
    SAEConfig,
    SparseAutoencoder,
    load_sae,
    save_sae,
    train_sae,
)


# ---------------------------------------------------------------------------
# SAE math
# ---------------------------------------------------------------------------


class TestSAEShapes:
    def test_forward_shapes(self):
        cfg = SAEConfig(input_dim=16, expansion=4)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(8, 16)
        x_hat, z = sae(x)
        assert x_hat.shape == (8, 16)
        assert z.shape == (8, cfg.num_features)

    def test_num_features_derived(self):
        cfg = SAEConfig(input_dim=256, expansion=8)
        assert cfg.num_features == 2048

    def test_encoder_relu_keeps_z_nonneg(self):
        cfg = SAEConfig(input_dim=8, expansion=2)
        sae = SparseAutoencoder(cfg)
        x = torch.randn(4, 8) * 10  # large magnitudes
        z = sae.encode(x)
        assert (z >= 0).all()


class TestSAETraining:
    """Tiny synthetic dataset where we know the right answer: the SAE should
    learn to reconstruct a low-dimensional input mixture."""

    def _synthetic_data(self, n_samples=512, input_dim=16, n_components=8, seed=0):
        rng = np.random.default_rng(seed)
        # Mix from `n_components` orthogonal directions — should be perfectly
        # reconstructible by an SAE with num_features ≥ n_components.
        directions = rng.normal(size=(n_components, input_dim)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
        coefficients = np.abs(rng.normal(size=(n_samples, n_components)).astype(np.float32))
        data = coefficients @ directions
        return data

    def test_reconstruction_loss_decreases(self):
        cfg = SAEConfig(input_dim=16, expansion=4, epochs=20, l1_coefficient=1e-4)
        sae = SparseAutoencoder(cfg)
        data = self._synthetic_data(input_dim=16)
        stats = train_sae(sae, data, verbose=False)
        # Reconstruction loss at the end should be strictly less than at the start.
        assert stats.recon_loss[-1] < stats.recon_loss[0]
        # And meaningfully lower — pin a generous threshold so we catch
        # a fully-broken trainer but not transient noise.
        assert stats.recon_loss[-1] < 0.5 * stats.recon_loss[0]

    def test_sparsity_increases_with_higher_l1(self):
        """Stronger L1 coefficient → fewer active features per sample.
        Pins the L1 mechanism so a sign flip or coefficient drop wouldn't
        silently disable sparsity pressure."""
        data = self._synthetic_data(n_samples=512, input_dim=16)

        cfg_low = SAEConfig(input_dim=16, expansion=4, epochs=20, l1_coefficient=1e-5, seed=42)
        cfg_high = SAEConfig(input_dim=16, expansion=4, epochs=20, l1_coefficient=1e-1, seed=42)

        sae_low = SparseAutoencoder(cfg_low)
        sae_high = SparseAutoencoder(cfg_high)
        stats_low = train_sae(sae_low, data, verbose=False)
        stats_high = train_sae(sae_high, data, verbose=False)
        assert stats_high.sparsity[-1] < stats_low.sparsity[-1]

    def test_save_load_round_trip(self, tmp_path):
        cfg = SAEConfig(input_dim=8, expansion=4, epochs=2)
        sae = SparseAutoencoder(cfg)
        data = self._synthetic_data(n_samples=64, input_dim=8)
        stats = train_sae(sae, data, verbose=False)
        save_sae(sae, stats, tmp_path)

        loaded = load_sae(tmp_path / 'sae.pt')
        # Forward pass should produce identical output.
        x = torch.randn(4, 8)
        with torch.no_grad():
            ref_x_hat, ref_z = sae(x)
            new_x_hat, new_z = loaded(x)
        assert torch.allclose(ref_x_hat, new_x_hat, atol=1e-6)
        assert torch.allclose(ref_z, new_z, atol=1e-6)


# ---------------------------------------------------------------------------
# Activation capture: hook key / dim assertion
# ---------------------------------------------------------------------------


class TestActivationCaptureModelContract:
    """The capture module hard-asserts the value-head's penultimate ReLU is
    at module index 13 with dim 256. If the YinshNetwork value_head is ever
    restructured, these tests will catch it before we accidentally train an
    SAE on the wrong layer."""

    def test_real_yinsh_network_has_expected_hook_key(self):
        """Smoke-instantiate a YinshNetwork and confirm the expected key is
        in `value_head_activations` after a forward pass."""
        from yinsh_ml.network.model import YinshNetwork
        net = YinshNetwork(input_channels=6, num_channels=64, num_blocks=2)
        net.eval()
        with torch.no_grad():
            _ = net(torch.zeros(1, 6, 11, 11))
        assert PENULTIMATE_KEY in net.value_head_activations
        assert net.value_head_activations[PENULTIMATE_KEY].shape == (1, PENULTIMATE_DIM)

    def test_capture_round_trip(self, tmp_path):
        """Construct a real (untrained) network, capture activations on a
        synthetic batch, save+verify metadata."""
        from yinsh_ml.network.model import YinshNetwork

        # Build a fake NetworkWrapper-ish object with the bits the capture needs.
        class _Wrapper:
            def __init__(self, net):
                self.network = net

        net = YinshNetwork(input_channels=6, num_channels=64, num_blocks=2)
        wrapper = _Wrapper(net)
        cap = ValueHeadActivationCapture(wrapper, device='cpu')

        batch = np.random.RandomState(0).randn(5, 6, 11, 11).astype(np.float32)
        acts = cap.capture(batch)
        assert acts.shape == (5, PENULTIMATE_DIM)

        out_dir = tmp_path / 'cap'
        meta = cap.save(out_dir)
        assert meta['num_positions'] == 5
        assert meta['activation_dim'] == PENULTIMATE_DIM
        assert (out_dir / 'activations.npy').exists()
        assert (out_dir / 'states.npy').exists()
        assert (out_dir / 'meta.json').exists()


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------


class TestFeatureAnalysis:
    def _trained_sae(self, n_samples=256, input_dim=8, expansion=4, epochs=15):
        cfg = SAEConfig(input_dim=input_dim, expansion=expansion, epochs=epochs,
                        l1_coefficient=1e-3)
        sae = SparseAutoencoder(cfg)
        rng = np.random.default_rng(0)
        data = np.abs(rng.normal(size=(n_samples, input_dim))).astype(np.float32)
        train_sae(sae, data, verbose=False)
        return sae, data

    def test_per_feature_summary_shape_and_invariants(self):
        sae, data = self._trained_sae()
        analyzer = FeatureAnalyzer(sae)
        report = analyzer.per_feature_summary(data, top_k=5)
        assert report.num_positions == data.shape[0]
        assert report.num_features == sae.cfg.num_features
        assert len(report.features) == sae.cfg.num_features

        # Per-feature: firing_rate ∈ [0, 1]; top_k_indices length ≤ k; sorted.
        for fs in report.features:
            assert 0.0 <= fs.firing_rate <= 1.0
            assert len(fs.top_k_indices) <= 5
            # Activations descending order.
            assert fs.top_k_activations == sorted(fs.top_k_activations, reverse=True)

    def test_dead_feature_counts(self):
        """An SAE trained briefly with strong L1 will have many dead features
        — pin that the report counts them and the firing-rate matches."""
        cfg = SAEConfig(input_dim=8, expansion=8, epochs=5, l1_coefficient=1e-1, seed=1)
        sae = SparseAutoencoder(cfg)
        rng = np.random.default_rng(1)
        data = np.abs(rng.normal(size=(128, 8))).astype(np.float32)
        train_sae(sae, data, verbose=False)
        analyzer = FeatureAnalyzer(sae)
        report = analyzer.per_feature_summary(data, top_k=3)
        # Dead-feature count derived from the report should match what
        # we observe by counting features with firing_rate == 0.
        manual_dead = sum(1 for f in report.features if f.firing_rate == 0.0)
        assert report.dead_feature_count == manual_dead

    def test_save_report(self, tmp_path):
        sae, data = self._trained_sae()
        analyzer = FeatureAnalyzer(sae)
        report = analyzer.per_feature_summary(data, top_k=3)
        path = tmp_path / 'report.json'
        save_report(report, path)
        assert path.exists()
        # Round-trippable JSON.
        import json
        with open(path) as f:
            payload = json.load(f)
        assert payload['num_features'] == sae.cfg.num_features
        assert len(payload['features']) == sae.cfg.num_features


class TestConfidentErrors:
    def test_finds_confidently_wrong_predictions(self):
        # 5 positions: 2 confidently wrong, 1 confidently right, 1 uncertain,
        # 1 confident on a draw (excluded).
        v_pred = np.array([0.9, -0.85, 0.95, 0.1, 0.8], dtype=np.float32)
        actual = np.array([-1.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float32)
        idx = FeatureAnalyzer.find_confident_errors(
            v_pred, actual, confidence_threshold=0.7
        )
        # Position 0: pred +0.9, truth -1.0 → confidently wrong → in
        # Position 1: pred -0.85, truth +1.0 → confidently wrong → in
        # Position 2: pred +0.95, truth +1.0 → confidently right → out
        # Position 3: pred +0.1 → not confident → out
        # Position 4: truth == 0 → excluded → out
        assert sorted(idx.tolist()) == [0, 1]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            FeatureAnalyzer.find_confident_errors(
                np.zeros(3), np.zeros(4)
            )
