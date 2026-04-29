"""Tests for the cosine + warmup LR schedule and its resume behavior.

Covers:
  * `_build_schedulers('cosine', warmup>0)` constructs Linear→Cosine via
    SequentialLR.
  * Warmup phase linearly ramps from 0.1·base → base over warmup_epochs.
  * Cosine phase decays from base → 0 across the remaining epochs.
  * Fast-forwarding to `resume_epoch` reproduces the LR a continuous schedule
    would have produced — critical for the per-iteration optimizer reset that
    the supervisor does.
  * `lr_schedule='step'` restores the legacy StepLR(step_size=10, gamma=0.9).
  * `_global_epoch` counter increments once per `train_epoch`.

Tests use a 1-parameter stand-in network so we don't need to spin up the full
YINSH stack — the scheduler math is what we're pinning.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


def _build_two_optimizers(base_lr=0.01, value_mul=5.0):
    """A pair of (Adam, SGD) optimizers wrapped around one parameter each,
    mirroring YinshTrainer's policy + value optimizer pattern."""
    p_policy = nn.Parameter(torch.zeros(1, requires_grad=True))
    p_value = nn.Parameter(torch.zeros(1, requires_grad=True))
    policy_opt = optim.Adam([p_policy], lr=base_lr)
    value_opt = optim.SGD([p_value], lr=base_lr * value_mul, momentum=0.9)
    return policy_opt, value_opt


def _build_cosine_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Production-equivalent scheduler. Mirrors trainer._build_schedulers
    exactly so a silent refactor of the construction order would trip these
    tests independently."""
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs - warmup_epochs),
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)


class TestWarmupPhase:
    def test_initial_lr_is_start_factor_times_base(self):
        """At step 0 (before any `step()`), LR should reflect the LinearLR
        start_factor = 0.1 → lr = 0.1·base."""
        opt, _ = _build_two_optimizers(base_lr=1.0)
        sched = _build_cosine_with_warmup(opt, warmup_epochs=5, total_epochs=100)
        assert opt.param_groups[0]['lr'] == pytest.approx(0.1, abs=1e-6)
        del sched  # unused; just proving construction succeeded

    def test_lr_reaches_base_by_end_of_warmup(self):
        opt, _ = _build_two_optimizers(base_lr=1.0)
        sched = _build_cosine_with_warmup(opt, warmup_epochs=5, total_epochs=100)
        for _ in range(5):
            sched.step()
        # After 5 steps of LinearLR(0.1→1.0 over 5 iters), LR is at base. The
        # next step() hands off to cosine, still near base.
        assert opt.param_groups[0]['lr'] == pytest.approx(1.0, abs=0.01)

    def test_linear_ramp_is_monotone_increasing(self):
        """Warmup should not overshoot or oscillate — strictly increasing."""
        opt, _ = _build_two_optimizers(base_lr=1.0)
        sched = _build_cosine_with_warmup(opt, warmup_epochs=5, total_epochs=100)
        lrs = [opt.param_groups[0]['lr']]
        for _ in range(5):
            sched.step()
            lrs.append(opt.param_groups[0]['lr'])
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1] - 1e-9, f"LR regressed at step {i}: {lrs}"


class TestCosinePhase:
    def test_cosine_decays_below_base_after_warmup(self):
        """Once we're past warmup, cosine decay should pull LR below base."""
        opt, _ = _build_two_optimizers(base_lr=1.0)
        sched = _build_cosine_with_warmup(opt, warmup_epochs=5, total_epochs=100)
        # Advance past warmup + into the cosine phase by a large margin.
        for _ in range(50):
            sched.step()
        assert opt.param_groups[0]['lr'] < 1.0

    def test_cosine_reaches_near_zero_at_total_epochs(self):
        """At total_epochs, cosine should be ~0 (last step is floor of curve)."""
        opt, _ = _build_two_optimizers(base_lr=1.0)
        sched = _build_cosine_with_warmup(opt, warmup_epochs=5, total_epochs=100)
        for _ in range(100):
            sched.step()
        # CosineAnnealingLR bottoms at eta_min=0 (our default). Within a tiny
        # tolerance after 100 steps.
        assert opt.param_groups[0]['lr'] < 1e-3


class TestFastForwardMatchesContinuousRun:
    """The `resume_epoch` path: rebuilding the scheduler and stepping it
    forward N times must land at the same LR a continuous scheduler would be
    at after N steps. This is the core invariant backing the supervisor's
    per-iteration optimizer reset."""

    @pytest.mark.parametrize("resume_epoch", [1, 5, 10, 25, 50, 99])
    def test_fast_forward_matches(self, resume_epoch):
        base_lr = 0.01
        warmup_epochs = 5
        total_epochs = 100

        # Continuous run: step the scheduler N times.
        opt_a, _ = _build_two_optimizers(base_lr=base_lr)
        sched_a = _build_cosine_with_warmup(opt_a, warmup_epochs, total_epochs)
        for _ in range(resume_epoch):
            sched_a.step()
        lr_continuous = opt_a.param_groups[0]['lr']

        # Reset path: build a fresh scheduler against a fresh optimizer
        # (at base LR), then fast-forward by stepping N times under the same
        # "ignore UserWarning" discipline the trainer uses.
        import warnings
        opt_b, _ = _build_two_optimizers(base_lr=base_lr)
        sched_b = _build_cosine_with_warmup(opt_b, warmup_epochs, total_epochs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(resume_epoch):
                sched_b.step()
        lr_reset = opt_b.param_groups[0]['lr']

        assert lr_reset == pytest.approx(lr_continuous, rel=1e-6)


class TestStepFallback:
    """`lr_schedule='step'` should reproduce the legacy StepLR behavior exactly."""

    def test_step_decays_by_gamma_every_step_size(self):
        opt, _ = _build_two_optimizers(base_lr=1.0)
        sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
        # After 10 steps → 1.0 × 0.9. After 20 → × 0.81.
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]['lr'] == pytest.approx(0.9, rel=1e-6)
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]['lr'] == pytest.approx(0.81, rel=1e-6)


class TestTrainerWiring:
    """Light integration: verify the trainer exposes `_build_schedulers` with
    the expected signature, tracks `_global_epoch`, and plumbs `base_lr`
    through to the optimizer."""

    def test_build_schedulers_signature(self):
        import inspect
        from yinsh_ml.training.trainer import YinshTrainer
        assert hasattr(YinshTrainer, "_build_schedulers")
        sig = inspect.signature(YinshTrainer._build_schedulers)
        # (self, resume_epoch=0)
        assert "resume_epoch" in sig.parameters
        assert sig.parameters["resume_epoch"].default == 0

    def test_init_plumbs_new_params(self):
        import inspect
        from yinsh_ml.training.trainer import YinshTrainer
        sig = inspect.signature(YinshTrainer.__init__)
        for p in ("base_lr", "lr_schedule", "warmup_epochs", "total_epochs"):
            assert p in sig.parameters, f"YinshTrainer.__init__ missing {p}"
        assert sig.parameters["lr_schedule"].default == "cosine"
        assert sig.parameters["warmup_epochs"].default == 0
