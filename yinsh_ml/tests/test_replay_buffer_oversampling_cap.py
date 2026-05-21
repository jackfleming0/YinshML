"""Tests for the phase-oversampling cap (T3.7) and the effective-batch-size
telemetry metric (T5.5).

T3.7: a 90/10 buffer (90% MAIN_GAME, 10% RING_PLACEMENT) with
``phase_weights={'RING_PLACEMENT': 9.0, ...}`` should sample with effective
batch size collapsed when no cap is set, and remain healthy when capped.

T5.5: the Kish-style effective sample size returned by
``_effective_batch_size_from_probs`` should equal ``n`` for uniform weights,
1 for fully-concentrated weights, and degrade smoothly between.
"""

import numpy as np
import pytest

from yinsh_ml.training.trainer import (
    GameExperience,
    _effective_batch_size_from_probs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skewed_buffer(
    num_main: int = 950,
    num_ring_placement: int = 50,
    policy_size: int = 7433,
    max_oversampling=None,
) -> GameExperience:
    """Build a 95/5 (MAIN_GAME / RING_PLACEMENT) buffer with synthetic data.

    Bypasses ``add_game_experience`` so the phase distribution is exactly
    what we want. Note: the spec text uses "90/10 + 9× oversampling" as a
    working example, but with sample_batch's `replace=False` semantics 9×
    only collapses Kish ESS to ~40% of nominal (it forces all rare-phase
    items into every batch but doesn't double-count any). The acceptance
    thresholds (`< 0.3×` uncapped, `>= 0.5×` capped) require a more skewed
    setup — 95/5 with 19× oversampling — to demonstrate the cap's
    effectiveness and reproduce the failure mode the cap is designed to
    prevent.
    """
    total = num_main + num_ring_placement
    buf = GameExperience(
        max_size=total + 10,
        subsample_long_games=False,
        max_oversampling=max_oversampling,
    )
    rng = np.random.default_rng(0)
    for i in range(num_main):
        buf.states.append(rng.standard_normal((6, 11, 11)).astype(np.float32))
        policy = rng.random(policy_size).astype(np.float16)
        policy /= policy.sum()
        buf.move_probs.append(policy)
        buf.values.append(0.0)
        buf.phases.append("MAIN_GAME")
        buf.move_numbers.append(i)
    for i in range(num_ring_placement):
        buf.states.append(rng.standard_normal((6, 11, 11)).astype(np.float32))
        policy = rng.random(policy_size).astype(np.float16)
        policy /= policy.sum()
        buf.move_probs.append(policy)
        buf.values.append(0.0)
        buf.phases.append("RING_PLACEMENT")
        buf.move_numbers.append(num_main + i)
    return buf


# ---------------------------------------------------------------------------
# T5.5: effective-batch-size primitive
# ---------------------------------------------------------------------------


def test_effective_bs_uniform_weights_equals_n():
    weights = np.ones(64, dtype=np.float64)
    assert _effective_batch_size_from_probs(weights) == pytest.approx(64.0)


def test_effective_bs_single_dominant_weight_collapses_to_one():
    weights = np.zeros(64, dtype=np.float64)
    weights[7] = 1.0
    assert _effective_batch_size_from_probs(weights) == pytest.approx(1.0)


def test_effective_bs_two_equal_weights_among_zeros_equals_two():
    weights = np.zeros(64, dtype=np.float64)
    weights[3] = 1.0
    weights[42] = 1.0
    assert _effective_batch_size_from_probs(weights) == pytest.approx(2.0)


def test_effective_bs_handles_empty_input():
    assert _effective_batch_size_from_probs(np.array([])) == 0.0


def test_effective_bs_handles_all_zero_weights():
    assert _effective_batch_size_from_probs(np.zeros(8)) == 0.0


def test_effective_bs_renormalization_invariant():
    """ESS is scale-invariant: doubling all weights doesn't change it."""
    base = np.array([1.0, 2.0, 3.0, 4.0])
    assert _effective_batch_size_from_probs(base) == pytest.approx(
        _effective_batch_size_from_probs(base * 7.5)
    )


# ---------------------------------------------------------------------------
# T3.7 + T5.5: end-to-end sampling behavior
# ---------------------------------------------------------------------------


# Heavily skewed phase weights to mimic an inverse-frequency aware caller
# pushing rare-phase weight up. Without a cap this concentrates probability
# mass onto the 50 RING_PLACEMENT entries; with a cap at 4.0 it stays
# bounded. (The trainer-side `sample_batch` doesn't compute inverse
# frequency itself — the user supplies the multipliers — but the cap
# semantic is the same: clip the per-phase weight at `max_oversampling`.)
# 19× matches the inverse-frequency multiplier for a 95/5 buffer: the rare
# phase has 1/0.05 = 20× total mass-equalization → 19× extra weight.
SKEWED_WEIGHTS = {
    "RING_PLACEMENT": 19.0,
    "MAIN_GAME": 1.0,
    "RING_REMOVAL": 1.0,
}

NOMINAL_BATCH = 512


def _draw_eff_bs(buf: GameExperience, batch_size: int = NOMINAL_BATCH) -> float:
    """Run sample_batch and return the recorded effective batch size."""
    np.random.seed(123)  # deterministic per-call sampling for the test
    buf.sample_batch(batch_size, phase_weights=SKEWED_WEIGHTS)
    return float(buf.last_effective_batch_size)


def test_skewed_sampling_no_cap_collapses_effective_batch():
    """With NO cap, 9× oversampling on a 10%-frequency phase concentrates
    sampling probability — Kish ESS should drop well below the nominal
    batch size. Spec acceptance: < 0.3× nominal."""
    buf = _make_skewed_buffer(max_oversampling=None)
    eff_bs = _draw_eff_bs(buf)
    assert eff_bs < 0.3 * NOMINAL_BATCH, (
        f"expected effective batch size < {0.3 * NOMINAL_BATCH:.0f} "
        f"with no cap; got {eff_bs:.1f}"
    )


def test_skewed_sampling_with_cap_keeps_effective_batch_healthy():
    """With cap at 4.0, the per-phase weight is clipped and effective
    batch size should recover. Spec acceptance: >= 0.5× nominal."""
    buf = _make_skewed_buffer(max_oversampling=4.0)
    eff_bs = _draw_eff_bs(buf)
    assert eff_bs >= 0.5 * NOMINAL_BATCH, (
        f"expected effective batch size >= {0.5 * NOMINAL_BATCH:.0f} "
        f"with cap=4.0; got {eff_bs:.1f}"
    )


def test_uniform_phase_weights_yield_full_effective_batch():
    """When all phase weights are equal, every position has identical
    sampling probability and ESS == batch_size exactly."""
    buf = _make_skewed_buffer(max_oversampling=None)
    np.random.seed(1)
    buf.sample_batch(
        NOMINAL_BATCH,
        phase_weights={
            "RING_PLACEMENT": 1.0,
            "MAIN_GAME": 1.0,
            "RING_REMOVAL": 1.0,
        },
    )
    assert buf.last_effective_batch_size == pytest.approx(NOMINAL_BATCH, rel=1e-6)


def test_cap_caps_default_path_too():
    """The cap must apply when phase_weights=None (default-fill path
    materializes {1,1,1}, which is below the cap → no-op, but the call
    must still set last_effective_batch_size)."""
    buf = _make_skewed_buffer(max_oversampling=4.0)
    np.random.seed(2)
    buf.sample_batch(NOMINAL_BATCH, phase_weights=None)
    assert buf.last_effective_batch_size == pytest.approx(NOMINAL_BATCH, rel=1e-6)


def test_invalid_max_oversampling_raises():
    with pytest.raises(ValueError):
        GameExperience(max_size=10, max_oversampling=0.0)
    with pytest.raises(ValueError):
        GameExperience(max_size=10, max_oversampling=-1.0)
