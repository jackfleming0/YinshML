"""Tests for the Track A bake-off harness (`scripts/run_bakeoff.py`).

The script mostly wires `ModelTournament._play_match` + stats aggregation +
a Markdown renderer. Tests cover the aggregation math + report rendering
directly — no neural-net inference needed for the paths that matter.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_bakeoff():
    """Import `scripts/run_bakeoff.py` directly — it's a script not a module."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_bakeoff.py"
    spec = importlib.util.spec_from_file_location("run_bakeoff", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_bakeoff"] = module
    spec.loader.exec_module(module)
    return module


class TestEloDelta:
    """`_elo_delta` maps win rate → ELO using the standard 400·log10 formula.
    Clamps at ±800 to avoid log(0) blowups on boundary win rates."""

    def test_50_percent_is_zero_elo(self):
        m = _import_bakeoff()
        assert m._elo_delta(0.5) == pytest.approx(0.0, abs=0.01)

    def test_symmetric_around_fifty(self):
        m = _import_bakeoff()
        assert m._elo_delta(0.75) == pytest.approx(-m._elo_delta(0.25), abs=0.01)

    def test_known_elo_ranks(self):
        """Sanity values from the standard ELO table:
        - 75% → ~+191 ELO
        - 64% → ~+100 ELO
        - 55% → ~+35 ELO"""
        m = _import_bakeoff()
        assert m._elo_delta(0.75) == pytest.approx(190.85, abs=0.5)
        assert m._elo_delta(0.64) == pytest.approx(100.0, abs=1.5)
        assert m._elo_delta(0.55) == pytest.approx(34.89, abs=0.5)

    def test_boundary_win_rates_are_clamped(self):
        """Exact 0% or 100% would give ±∞; the eps-clamp keeps them finite."""
        m = _import_bakeoff()
        assert m._elo_delta(0.0) > -10000  # finite
        assert m._elo_delta(1.0) < 10000


class TestDeviceResolution:
    def test_explicit_passthrough(self):
        m = _import_bakeoff()
        assert m._resolve_device("cpu") == "cpu"
        assert m._resolve_device("cuda") == "cuda"
        assert m._resolve_device("mps") == "mps"

    def test_auto_picks_available_accelerator(self, monkeypatch):
        m = _import_bakeoff()
        import torch
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.backends.mps, "is_available", return_value=False):
            assert m._resolve_device("auto") == "cpu"
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.backends.mps, "is_available", return_value=True):
            assert m._resolve_device("auto") == "mps"
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert m._resolve_device("auto") == "cuda"


class TestRunBakeoffAggregation:
    """The aggregation path: two mocked MatchResults go in, the aggregate
    should correctly combine from challenger's perspective (summing
    challenger wins across both directions, handling draws, computing
    Wilson CI)."""

    def _mk_match_result(self, white_wins, black_wins, draws):
        """Stand-in that quacks like MatchResult."""
        mr = MagicMock()
        mr.white_wins = white_wins
        mr.black_wins = black_wins
        mr.draws = draws
        return mr

    def _run_with_mock_tournament(self, fwd_result, bwd_result, games=200):
        """Drive `run_bakeoff` with a mocked ModelTournament so we can
        control the played-game outcomes without spinning up networks."""
        m = _import_bakeoff()

        fake_tournament = MagicMock()
        fake_tournament._load_model = MagicMock(side_effect=[MagicMock(), MagicMock()])
        fake_tournament._play_match = MagicMock(side_effect=[fwd_result, bwd_result])

        with patch("run_bakeoff.ModelTournament", return_value=fake_tournament):
            return m.run_bakeoff(
                challenger_path=Path("/fake/challenger.pt"),
                baseline_path=Path("/fake/baseline.pt"),
                games=games,
                eval_seed=42,
                device="cpu",
                temperature=0.1,
                use_ema_for_eval=False,
            )

    def test_odd_games_rejected(self):
        """Games must split evenly per direction. Odd count raises early
        rather than silently dropping a game."""
        m = _import_bakeoff()
        with pytest.raises(ValueError, match="must be even"):
            m.run_bakeoff(
                challenger_path=Path("/fake"),
                baseline_path=Path("/fake"),
                games=101,
                eval_seed=42,
                device="cpu",
                temperature=0.1,
                use_ema_for_eval=False,
            )

    def test_perfect_challenger_sweep(self):
        """Challenger wins every game in both directions."""
        # 100 games per direction, challenger wins all as White in fwd, all
        # as Black in bwd.
        fwd = self._mk_match_result(white_wins=100, black_wins=0, draws=0)
        bwd = self._mk_match_result(white_wins=0, black_wins=100, draws=0)
        stats = self._run_with_mock_tournament(fwd, bwd, games=200)

        agg = stats["aggregate"]
        assert agg["challenger_wins"] == 200
        assert agg["baseline_wins"] == 0
        assert agg["draws"] == 0
        assert agg["decisive"] == 200
        assert agg["win_rate"] == pytest.approx(1.0)
        # ELO delta clamped at the eps boundary; should be very positive.
        assert agg["elo_delta"] > 500

    def test_perfect_baseline_sweep(self):
        fwd = self._mk_match_result(white_wins=0, black_wins=100, draws=0)
        bwd = self._mk_match_result(white_wins=100, black_wins=0, draws=0)
        stats = self._run_with_mock_tournament(fwd, bwd, games=200)

        agg = stats["aggregate"]
        assert agg["challenger_wins"] == 0
        assert agg["baseline_wins"] == 200
        assert agg["win_rate"] == pytest.approx(0.0)
        assert agg["elo_delta"] < -500

    def test_50_50_is_inconclusive(self):
        """Even split: win rate 0.5, ELO delta 0, Wilson CI straddles 50%."""
        fwd = self._mk_match_result(white_wins=50, black_wins=50, draws=0)
        bwd = self._mk_match_result(white_wins=50, black_wins=50, draws=0)
        stats = self._run_with_mock_tournament(fwd, bwd, games=200)

        agg = stats["aggregate"]
        assert agg["win_rate"] == pytest.approx(0.5)
        assert agg["elo_delta"] == pytest.approx(0.0, abs=0.01)
        assert agg["wilson_lower"] <= 0.5 <= agg["wilson_upper"]

    def test_draws_excluded_from_win_rate_denominator(self):
        """Wilson CI is over *decisive* games — draws are reported separately
        and don't inflate or deflate the proportion."""
        # 100 decisive (all challenger wins) + 100 draws. Win rate should be
        # 1.0 over decisive, not 100/200 = 0.5.
        fwd = self._mk_match_result(white_wins=50, black_wins=0, draws=50)
        bwd = self._mk_match_result(white_wins=0, black_wins=50, draws=50)
        stats = self._run_with_mock_tournament(fwd, bwd, games=200)

        agg = stats["aggregate"]
        assert agg["draws"] == 100
        assert agg["decisive"] == 100
        assert agg["win_rate"] == pytest.approx(1.0)

    def test_forward_and_reverse_reported_separately(self):
        """Per-direction breakdown preserves color-specific outcomes so
        White-advantage asymmetry is visible in the report."""
        # Challenger dominates as White, loses as Black — classic
        # color-advantage mirage that an aggregate could hide.
        fwd = self._mk_match_result(white_wins=80, black_wins=20, draws=0)
        bwd = self._mk_match_result(white_wins=80, black_wins=20, draws=0)
        # In fwd: challenger=white, white_wins=80 → challenger wins 80.
        # In bwd: challenger=black, black_wins=20 → challenger wins 20.
        stats = self._run_with_mock_tournament(fwd, bwd, games=200)

        assert stats["forward_direction"]["challenger_wins"] == 80
        assert stats["forward_direction"]["baseline_wins"] == 20
        assert stats["reverse_direction"]["challenger_wins"] == 20
        assert stats["reverse_direction"]["baseline_wins"] == 80

    def test_games_split_evenly_across_directions(self):
        """`games=200` must dispatch exactly 100 games per direction to
        `_play_match`."""
        m = _import_bakeoff()
        fake_tournament = MagicMock()
        fake_tournament._load_model = MagicMock(side_effect=[MagicMock(), MagicMock()])
        fake_tournament._play_match = MagicMock(side_effect=[
            self._mk_match_result(50, 50, 0),
            self._mk_match_result(50, 50, 0),
        ])

        with patch("run_bakeoff.ModelTournament", return_value=fake_tournament) as TM:
            m.run_bakeoff(
                challenger_path=Path("/fake"),
                baseline_path=Path("/fake"),
                games=200,
                eval_seed=42,
                device="cpu",
                temperature=0.1,
                use_ema_for_eval=False,
            )
            # games_per_match passed to ModelTournament ctor = 100.
            TM.assert_called_once()
            assert TM.call_args.kwargs["games_per_match"] == 100

    def test_eval_seed_propagates_to_tournament(self):
        """Deterministic eval seeding must reach the underlying tournament
        so reruns reproduce — the whole point of pinning a seed."""
        m = _import_bakeoff()
        fake_tournament = MagicMock()
        fake_tournament._load_model = MagicMock(side_effect=[MagicMock(), MagicMock()])
        fake_tournament._play_match = MagicMock(side_effect=[
            self._mk_match_result(50, 50, 0),
            self._mk_match_result(50, 50, 0),
        ])
        with patch("run_bakeoff.ModelTournament", return_value=fake_tournament) as TM:
            m.run_bakeoff(
                challenger_path=Path("/fake"),
                baseline_path=Path("/fake"),
                games=200,
                eval_seed=99999,
                device="cpu",
                temperature=0.1,
                use_ema_for_eval=False,
            )
            assert TM.call_args.kwargs["eval_seed"] == 99999


class TestRenderReport:
    """Markdown rendering. Verify structural invariants — top-of-file verdict,
    JSON dump round-trips, CI straddle classification matches the aggregate."""

    def _sample_stats(self, lb, ub, win_rate=0.6):
        return {
            "challenger_path": "/fake/c.pt",
            "baseline_path": "/fake/b.pt",
            "games": 200,
            "eval_seed": 42,
            "temperature": 0.1,
            "device": "cpu",
            "elapsed_seconds": 180.0,
            "aggregate": {
                "challenger_wins": 120, "baseline_wins": 80, "draws": 0,
                "decisive": 200, "win_rate": win_rate,
                "wilson_lower": lb, "wilson_upper": ub,
                "elo_delta": 70.0,
            },
            "forward_direction": {
                "description": "challenger plays White",
                "challenger_wins": 60, "baseline_wins": 40, "draws": 0,
                "wilson_ci": [0.50, 0.70],
            },
            "reverse_direction": {
                "description": "challenger plays Black",
                "challenger_wins": 60, "baseline_wins": 40, "draws": 0,
                "wilson_ci": [0.50, 0.70],
            },
        }

    def test_challenger_win_verdict(self):
        m = _import_bakeoff()
        # CI clears 0.5: 53% lower bound.
        report = m.render_report(self._sample_stats(lb=0.53, ub=0.67))
        assert "Challenger wins" in report
        assert "Inconclusive" not in report

    def test_inconclusive_verdict_when_ci_straddles_fifty(self):
        m = _import_bakeoff()
        # 50% is inside [0.48, 0.58].
        report = m.render_report(self._sample_stats(lb=0.48, ub=0.58))
        assert "Inconclusive" in report

    def test_baseline_win_verdict(self):
        m = _import_bakeoff()
        # CI fully below 0.5.
        report = m.render_report(self._sample_stats(lb=0.30, ub=0.45, win_rate=0.37))
        assert "Baseline wins" in report

    def test_json_dump_is_parseable(self):
        """The embedded JSON block must round-trip cleanly so downstream
        tools can parse it programmatically."""
        m = _import_bakeoff()
        stats = self._sample_stats(lb=0.53, ub=0.67)
        report = m.render_report(stats)
        start = report.index("```json") + len("```json\n")
        end = report.index("```", start)
        parsed = json.loads(report[start:end])
        assert parsed["aggregate"]["win_rate"] == pytest.approx(stats["aggregate"]["win_rate"])

    def test_report_includes_both_directions(self):
        m = _import_bakeoff()
        report = m.render_report(self._sample_stats(lb=0.53, ub=0.67))
        assert "Challenger plays White" in report
        assert "Challenger plays Black" in report
