"""Tests for the heuristic weight re-fitting core (numpy-only)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from yinsh_ml.heuristics import weight_fitting as wf
from yinsh_ml.heuristics.weight_manager import WeightManager


def test_phase_bucketing():
    assert wf.phase_of_move_count(5, 15, 35) == "early"
    assert wf.phase_of_move_count(15, 15, 35) == "early"
    assert wf.phase_of_move_count(20, 15, 35) == "mid"
    assert wf.phase_of_move_count(40, 15, 35) == "late"


def test_fit_logistic_recovers_signs_on_separable_data():
    rng = np.random.default_rng(0)
    n = 800
    # feature 0 positively drives the outcome, feature 1 negatively, feature 2 noise.
    X = rng.normal(size=(n, 3))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y = (1.0 / (1.0 + np.exp(-logits)) > rng.uniform(size=n)).astype(float)
    coefs = wf.fit_logistic(X, y, l2=0.1, iters=800, lr=0.5)
    assert coefs[0] > 0.5          # positive driver
    assert coefs[1] < -0.3         # negative driver recovered as negative
    assert abs(coefs[2]) < abs(coefs[0])  # noise feature smaller


def test_clamp_drops_negative_and_caps_high():
    coefs = np.array([-3.0, 0.5, 100.0])
    w = wf.clamp_and_scale(coefs, scale=2.0)
    assert w[0] == 0.0            # negative -> dropped
    assert w[1] == pytest.approx(1.0)
    assert w[2] == wf.WEIGHT_MAX  # capped at 50


def test_fit_weights_from_samples_is_weightmanager_loadable(tmp_path):
    rng = np.random.default_rng(1)
    feats = wf.PRODUCTION_FEATURES
    samples = []
    for phase in wf.PHASES:
        for _ in range(200):
            fd = {f: float(rng.normal()) for f in feats}
            # potential_runs_count positively predicts winning in this synthetic world
            label = int(fd["potential_runs_count"] + 0.3 * rng.normal() > 0)
            samples.append((phase, fd, label))

    weights = wf.fit_weights_from_samples(samples, method="logreg", scale=10.0)

    # every phase + every feature present
    assert set(weights) == set(wf.PHASES)
    for phase in wf.PHASES:
        assert set(weights[phase]) == set(feats)
        assert all(wf.WEIGHT_MIN <= v <= wf.WEIGHT_MAX for v in weights[phase].values())
    # the predictive feature earns a positive weight
    assert weights["mid"]["potential_runs_count"] > 0

    # round-trips through WeightManager (validates structure + constraints)
    path = tmp_path / "w.json"
    with open(path, "w") as fh:
        json.dump(weights, fh)
    loaded = WeightManager().load_from_file(str(path))
    assert loaded == weights


def test_undersampled_phase_uses_fallback():
    feats = wf.PRODUCTION_FEATURES
    fallback = {p: {f: 1.0 for f in feats} for p in wf.PHASES}
    # only 3 samples for 'early' -> below threshold -> fallback
    samples = [("early", {f: 0.0 for f in feats}, 1) for _ in range(3)]
    weights = wf.fit_weights_from_samples(
        samples, min_samples_per_phase=50, fallback=fallback
    )
    assert weights["early"] == fallback["early"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
