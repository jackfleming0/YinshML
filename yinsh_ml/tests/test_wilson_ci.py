"""Tests for the Wilson CI helpers + tournament per-pair CI computation.

Why these tests exist: at games_per_match=50, the 95% Wilson CI half-width on
a 50/50 split is ~14%, so a 49.5% rejection at the 55% gate is statistically
indistinguishable from the threshold. The promotion log was previously just
showing the lower bound; we added full (lower, upper) + SE + a "straddles
threshold" flag so noisy rejections are visible at-a-glance. These tests pin
the math and the per-pair aggregation.
"""

import math
import pytest

from yinsh_ml.utils.stats import wilson_bounds, standard_error


class TestWilsonBounds:
    def test_zero_total_returns_widest_interval(self):
        """No data → full [0, 1] CI. Lets straddle-checks correctly classify as inconclusive."""
        assert wilson_bounds(0, 0) == (0.0, 1.0)

    def test_zero_wins_lower_pinned_at_zero(self):
        """0 / 50 → lower bound = 0; upper bound = the rule of three-ish region."""
        lower, upper = wilson_bounds(0, 50)
        assert lower == 0.0
        assert 0.0 < upper < 0.10  # one-sided ~7.1%

    def test_all_wins_upper_pinned_at_one(self):
        """50 / 50 → upper bound = 1; lower bound is non-trivial."""
        lower, upper = wilson_bounds(50, 50)
        assert upper == 1.0
        assert 0.90 < lower < 1.0

    def test_50_50_split_is_centered(self):
        """25/50: mean ~0.5, half-width matches 1.96 * sqrt(0.25/50) ≈ 0.139."""
        lower, upper = wilson_bounds(25, 50)
        center = (lower + upper) / 2
        half_width = (upper - lower) / 2
        assert center == pytest.approx(0.5, abs=0.01)
        # Allow some tolerance — Wilson CI half-width differs slightly from
        # naive √(p(1-p)/n) at small n.
        assert 0.13 < half_width < 0.15

    def test_larger_sample_narrows_ci(self):
        """50/100 should be narrower than 25/50."""
        _, upper_50 = wilson_bounds(25, 50)
        _, upper_100 = wilson_bounds(50, 100)
        # Both centered at 0.5; larger sample → tighter upper bound
        assert upper_100 < upper_50

    def test_explicit_z_value(self):
        """z=1.0 (~68% CI) is narrower than z=1.96 (95% CI) at the same data."""
        _, upper_95 = wilson_bounds(25, 50, z=1.96)
        _, upper_68 = wilson_bounds(25, 50, z=1.0)
        assert upper_68 < upper_95


class TestStandardError:
    def test_zero_total(self):
        assert standard_error(0, 0) == 0.0

    def test_50_50_at_n_50(self):
        # √(0.25/50) ≈ 0.0707
        assert standard_error(25, 50) == pytest.approx(math.sqrt(0.25 / 50), abs=1e-9)

    def test_extremes_are_zero(self):
        assert standard_error(0, 50) == 0.0
        assert standard_error(50, 50) == 0.0


class TestStraddlesThreshold:
    """Property the gate's CI-aware log line is supposed to surface."""

    def test_50_games_at_55_percent_straddles_055(self):
        """At our default games_per_match (was 50), 27/50 (54%) straddles the 55% gate."""
        lower, upper = wilson_bounds(27, 50)
        threshold = 0.55
        assert lower <= threshold <= upper, \
            f"Expected 27/50 to straddle 55% gate; got [{lower:.3f}, {upper:.3f}]"

    def test_100_games_at_55_percent_still_straddles(self):
        """At 100 games we narrow but a true 55% rate is still inside the CI for 55/100.
        This is a reminder that even after the bump, gate/CI overlap is the norm
        near threshold."""
        lower, upper = wilson_bounds(55, 100)
        assert lower <= 0.55 <= upper

    def test_200_games_at_clear_pass_does_not_straddle(self):
        """130/200 (65% win rate, ~10pp above gate) at 200 games clearly clears."""
        lower, _ = wilson_bounds(130, 200)
        assert lower > 0.55


class TestComputePairCIs:
    """ModelTournament._compute_pair_cis uses the helpers above. Exercise it on
    a stub _pair_results to confirm the structure ships through correctly."""

    def _make_stub(self, pair_results):
        """Build a minimal object with the attributes _compute_pair_cis reads."""
        from yinsh_ml.utils.tournament import ModelTournament
        stub = type('StubTournament', (), {})()
        stub._pair_results = pair_results
        stub._compute_pair_cis = ModelTournament._compute_pair_cis.__get__(stub)
        return stub

    def test_empty_pair_results(self):
        stub = self._make_stub({})
        assert stub._compute_pair_cis() == []

    def test_zero_total_pair_skipped(self):
        """If a pair somehow has only draws=0 and no decisives, skip it."""
        stub = self._make_stub({
            ('iter_1', 'iter_2'): {'wins_a': 0, 'wins_b': 0, 'draws': 0},
        })
        assert stub._compute_pair_cis() == []

    def test_single_pair_basic(self):
        stub = self._make_stub({
            ('iter_1', 'iter_2'): {'wins_a': 30, 'wins_b': 20, 'draws': 0},
        })
        result = stub._compute_pair_cis()
        assert len(result) == 1
        entry = result[0]
        assert entry['model_a'] == 'iter_1'
        assert entry['model_b'] == 'iter_2'
        assert entry['wins_a'] == 30
        assert entry['total'] == 50
        assert entry['win_rate_a'] == pytest.approx(0.6)
        assert 0.0 <= entry['ci_lower'] < entry['win_rate_a'] < entry['ci_upper'] <= 1.0
        assert entry['se'] == pytest.approx(math.sqrt(0.6 * 0.4 / 50), abs=1e-9)

    def test_draws_excluded_from_decisives(self):
        """Draws shouldn't bias the win-rate proportion. 30 wins / (30+20) decisives
        even with 10 draws still gives p = 0.6."""
        stub = self._make_stub({
            ('iter_1', 'iter_2'): {'wins_a': 30, 'wins_b': 20, 'draws': 10},
        })
        entry = stub._compute_pair_cis()[0]
        assert entry['win_rate_a'] == pytest.approx(0.6)
        assert entry['draws'] == 10
        assert entry['total'] == 60

    def test_pairs_sorted_by_iteration(self):
        stub = self._make_stub({
            ('iter_5', 'iter_7'): {'wins_a': 25, 'wins_b': 25, 'draws': 0},
            ('iter_1', 'iter_3'): {'wins_a': 25, 'wins_b': 25, 'draws': 0},
            ('iter_3', 'iter_5'): {'wins_a': 25, 'wins_b': 25, 'draws': 0},
        })
        result = stub._compute_pair_cis()
        assert [(e['model_a'], e['model_b']) for e in result] == [
            ('iter_1', 'iter_3'),
            ('iter_3', 'iter_5'),
            ('iter_5', 'iter_7'),
        ]
