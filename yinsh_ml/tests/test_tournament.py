from __future__ import annotations

import math
import os

os.environ.setdefault("NPY_DISABLE_MAC_OS_CHECK", "1")

import pytest

from yinsh_ml.agents import TournamentEvaluator, TournamentMetrics, HeuristicAgentConfig


class DummyAgent:
    def __init__(self, config: HeuristicAgentConfig):
        self.config = config
        self.last_search_stats = {"nodes_evaluated": 0}

    def select_move(self, state):
        raise RuntimeError("DummyAgent.select_move should not be called in these tests")


def test_run_tournament_aggregates_metrics(monkeypatch: pytest.MonkeyPatch):
    evaluator = TournamentEvaluator(
        heuristic_agent_factory=lambda config: DummyAgent(config),
        opponent_factory=lambda: object(),
    )

    sample_results = [
        ((1, 0, 0), 2.0, 0.8, 120, 50),
        ((0, 1, 0), 3.0, 0.9, 140, 60),
        ((0, 0, 1), 1.0, 0.5, 100, 40),
        ((1, 0, 0), 4.0, 1.0, 160, 70),
    ]

    calls = {"index": 0}

    def fake_play_game(*args, **kwargs):
        result = sample_results[calls["index"]]
        calls["index"] += 1
        return result

    monkeypatch.setattr(evaluator, "play_game", fake_play_game)

    metrics = evaluator.run_tournament(games=len(sample_results))

    assert metrics.total_games == 4
    assert metrics.wins == 2
    assert metrics.losses == 1
    assert metrics.draws == 1
    assert pytest.approx(metrics.win_rate, rel=1e-6) == 0.5
    assert pytest.approx(metrics.average_game_length, rel=1e-6) == 55.0
    assert metrics.std_game_length > 0.0
    assert pytest.approx(metrics.average_move_time, rel=1e-6) == (2.0 + 3.0 + 1.0 + 4.0) / 4
    assert metrics.max_move_time == 1.0
    assert metrics.nodes_per_second > 0.0


def test_tournament_smoke_runs_small_sample(monkeypatch: pytest.MonkeyPatch):
    evaluator = TournamentEvaluator(
        heuristic_agent_factory=lambda config: DummyAgent(config),
        heuristic_config=HeuristicAgentConfig(max_depth=1, time_limit_seconds=0.0),
        opponent_factory=lambda: object(),
    )

    def fake_play_game(*args, **kwargs):
        return ((1, 0, 0), 0.5, 0.2, 50, 30)

    monkeypatch.setattr(evaluator, "play_game", fake_play_game)

    metrics = evaluator.run_tournament(games=3)

    assert metrics.total_games == 3
    assert metrics.wins == 3
    assert metrics.losses == 0
    assert metrics.draws == 0
    assert metrics.win_rate == 1.0

