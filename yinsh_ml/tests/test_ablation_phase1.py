"""Tests for the Phase 1 ablation harness core (stats + arm construction)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))
import ablation_phase1 as ab  # noqa: E402

from yinsh_ml.heuristics.feature_registry import PRODUCTION_FEATURES


def _base():
    return {p: {f: 1.0 for f in PRODUCTION_FEATURES} for p in ("early", "mid", "late")}


def test_with_feature_adds_one_feature_to_every_phase_only():
    base = _base()
    out = ab.with_feature(base, "defensive_disruption", 7.0)
    for phase in ("early", "mid", "late"):
        assert out[phase]["defensive_disruption"] == 7.0
        # the 6 production weights are untouched (one-variable change)
        for f in PRODUCTION_FEATURES:
            assert out[phase][f] == base[phase][f]
    # base not mutated
    assert "defensive_disruption" not in base["early"]


def test_parse_arms_expands_weight_grid():
    arms = ab.parse_arms(["defensive_disruption:4,8", "ring_mobility_differential:2"])
    assert ("defensive_disruption", 4.0) in arms
    assert ("defensive_disruption", 8.0) in arms
    assert ("ring_mobility_differential", 2.0) in arms
    assert len(arms) == 3


def test_parse_arms_rejects_unknown_feature():
    with pytest.raises(SystemExit):
        ab.parse_arms(["not_a_feature:1"])


def test_wilson_ci_basic_properties():
    # symmetric 50% over many games -> interval brackets 0.5, fairly tight
    lo, hi = ab.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert hi - lo < 0.25
    # a strong result excludes 0.5
    lo2, hi2 = ab.wilson_ci(90, 100)
    assert lo2 > 0.5
    # no games -> maximally uncertain
    assert ab.wilson_ci(0, 0) == (0.0, 1.0)
    # more games -> tighter interval (more power)
    w_small = ab.wilson_ci(15, 20)
    w_big = ab.wilson_ci(150, 200)  # same 0.75 rate
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
