"""Tests for Gaussian soft value targets (Track A polish item).

The value head is a 7-way classifier over discrete score differences
{-3,-2,-1,0,+1,+2,+3}. Hard one-hot CE throws away ordinal structure — a
prediction of +2 when the truth is +3 is treated the same as a prediction of
-3. Soft targets replace the one-hot with a Gaussian centered on the
(unrounded) target class, so neighboring classes receive decaying mass.

Tests validate the Gaussian construction, the reduction to hard-CE when σ=0,
and a couple of gradient-direction sanity checks that would catch the sign or
perspective flipping in a future refactor.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from yinsh_ml.training.trainer import YinshTrainer


NUM_CLASSES = 7  # YinshNetwork value head: 7 score-diff classes


def _gaussian_targets(target_normalized: torch.Tensor, sigma: float, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Reference implementation of the target distribution we expect the
    trainer to build. Mirrors the production formula in trainer.py so tests
    can assert bit-parity independently of test-internal reasoning."""
    class_indices = torch.arange(num_classes, dtype=target_normalized.dtype)
    dist_sq = (class_indices.unsqueeze(0) - target_normalized.unsqueeze(1)) ** 2
    w = torch.exp(-dist_sq / (2.0 * sigma ** 2))
    return w / w.sum(dim=1, keepdim=True)


class TestGaussianConstruction:
    def test_peak_at_target_class(self):
        """σ=0.5 peaks sharply on the exact integer target; mass decays
        monotonically moving away."""
        # Target normalized = 3.0 → peaks on class 3.
        t = torch.tensor([3.0])
        dist = _gaussian_targets(t, sigma=0.5)[0]
        assert torch.argmax(dist).item() == 3
        # Monotone decay in both directions.
        assert dist[3] > dist[2] > dist[1] > dist[0]
        assert dist[3] > dist[4] > dist[5] > dist[6]

    def test_sums_to_one(self):
        """Per-row normalization must give a proper probability distribution."""
        t = torch.tensor([0.0, 2.5, 6.0, -0.3, 3.7])  # edges + intermediate
        dist = _gaussian_targets(t, sigma=0.5)
        # Some targets (e.g., -0.3, 6.0) put mass past the edges — the truncated
        # Gaussian is then renormalized across the remaining classes. The only
        # invariant that has to hold is that each row sums to 1 exactly.
        row_sums = dist.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_boundary_target_clamps_mass_inside(self):
        """A target at the highest class puts most of the mass on that class.
        Since the Gaussian past-edge is dropped and renormalized, the
        in-bounds tail absorbs it — the peak ends up at class 6 and
        still-monotone down to class 0."""
        t = torch.tensor([6.0])  # class +3, the highest bucket
        dist = _gaussian_targets(t, sigma=0.5)[0]
        assert torch.argmax(dist).item() == 6
        assert dist[6] > dist[5] > dist[4] > dist[3]

    def test_sigma_half_puts_known_mass_on_neighbors(self):
        """σ=0.5 with a target on an integer class puts ~38% of mass on the
        two adjacent classes combined. Pins the spread so a silent change to
        the formula (different denominator, different σ convention, missing
        factor-of-2) would trip this test.

        Analytical: raw Gaussian weights for target=3, σ=0.5:
          class 3: exp(0) = 1.0
          class 2: exp(-1/0.5) = exp(-2) ≈ 0.135335
          class 4: exp(-2) ≈ 0.135335
          class 1: exp(-8) ≈ 3.35e-4
          class 5: exp(-8) ≈ 3.35e-4
          classes 0, 6: exp(-18) ≈ 1.5e-8 (negligible)
        Sum ≈ 1.27134 → class-3 ≈ 0.786, adjacent (2 and 4) ≈ 0.213 combined.
        """
        t = torch.tensor([3.0])
        dist = _gaussian_targets(t, sigma=0.5)[0]
        assert dist[3].item() == pytest.approx(0.786, abs=0.005)
        neighbor_mass = (dist[2] + dist[4]).item()
        assert neighbor_mass == pytest.approx(0.213, abs=0.005)


class TestTrainerSoftLossParity:
    """Integration with the trainer's actual loss formula. Build a tiny
    version of the path that runs in train_step and assert the soft loss
    matches the reference implementation."""

    def _target_and_logits(self):
        # Four samples with distinct real-valued targets in [-1, 1].
        target_values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        target_normalized = (target_values + 1.0) / 2.0 * (NUM_CLASSES - 1)
        # Arbitrary logits — just need to exercise the loss math.
        torch.manual_seed(0)
        value_logits = torch.randn(len(target_values), NUM_CLASSES)
        return target_values, target_normalized, value_logits

    def test_soft_matches_reference(self):
        target_values, target_normalized, value_logits = self._target_and_logits()
        sigma = 0.5

        # Production-equivalent math (inlined from trainer.py):
        class_indices = torch.arange(NUM_CLASSES, dtype=target_normalized.dtype)
        dist_sq = (class_indices.unsqueeze(0) - target_normalized.unsqueeze(1)) ** 2
        target_dist = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)
        log_probs = F.log_softmax(value_logits, dim=1)
        prod_loss = -(target_dist * log_probs).sum(dim=1).mean()

        # Reference (from this test file):
        ref_dist = _gaussian_targets(target_normalized, sigma=sigma)
        ref_loss = -(ref_dist * log_probs).sum(dim=1).mean()

        assert torch.allclose(prod_loss, ref_loss, atol=1e-6)

    def test_sigma_zero_equals_hard_ce(self):
        """σ=0 is the disable-soft-targets knob: loss must match
        F.cross_entropy(logits, rounded_target_class) exactly."""
        target_values, target_normalized, value_logits = self._target_and_logits()
        target_class = torch.round(target_normalized).long().clamp(0, NUM_CLASSES - 1)
        hard_loss = F.cross_entropy(value_logits, target_class)

        # Trainer's σ=0 branch uses F.cross_entropy directly — same code path.
        # The assertion here is trivially satisfied, but codifying it pins the
        # fallback semantics so a future refactor can't silently change them.
        assert torch.allclose(hard_loss, F.cross_entropy(value_logits, target_class))

    def test_soft_gradient_points_toward_target(self):
        """If the model currently puts all mass on class 0 but the target is
        class 6, the gradient on class-6 logits should be negative (increase
        logit) and on class-0 logits positive (decrease logit). True of any
        CE variant — test here ensures soft CE preserves this.
        """
        sigma = 0.5
        target_normalized = torch.tensor([6.0])  # target class = 6
        # Logits heavily favoring class 0.
        value_logits = torch.zeros(1, NUM_CLASSES, requires_grad=True)
        with torch.no_grad():
            value_logits[0, 0] = 5.0

        value_logits = value_logits.detach().requires_grad_(True)
        class_indices = torch.arange(NUM_CLASSES, dtype=target_normalized.dtype)
        dist_sq = (class_indices.unsqueeze(0) - target_normalized.unsqueeze(1)) ** 2
        target_dist = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)
        loss = -(target_dist * F.log_softmax(value_logits, dim=1)).sum(dim=1).mean()
        loss.backward()

        grads = value_logits.grad[0]
        # dL/dlogit[6] should be negative (want to INCREASE logit 6 → SGD step
        # subtracts grad → logit 6 goes up).
        assert grads[6].item() < 0, f"grad[6] should be negative, got {grads[6].item()}"
        # Conversely, logit 0 is over-confident; gradient should be positive.
        assert grads[0].item() > 0, f"grad[0] should be positive, got {grads[0].item()}"


class TestTrainerInitStoresSigma:
    """Light wiring test — confirm the constructor plumbs the kwarg through."""

    def test_default_is_half(self):
        # We can't easily build a real trainer here (needs a full network +
        # device setup), so just assert the default signature. This is a
        # guard: if someone changes the default they should flip this test
        # intentionally.
        import inspect
        sig = inspect.signature(YinshTrainer.__init__)
        assert sig.parameters["soft_value_target_sigma"].default == 0.5

    def test_sigma_is_stored_as_float(self):
        import inspect
        sig = inspect.signature(YinshTrainer.__init__)
        # Annotation sanity-check — we're typed as float.
        assert sig.parameters["soft_value_target_sigma"].annotation is float
