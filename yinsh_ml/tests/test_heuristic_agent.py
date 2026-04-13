from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import pytest

from yinsh_ml.agents import HeuristicAgent, HeuristicAgentConfig


class StubPlayer:
    WHITE = 1
    BLACK = -1

    def __init__(self, value: int):
        self.value = value

    @property
    def opponent(self) -> "StubPlayer":
        return StubPlayer(StubPlayer.BLACK if self.value == StubPlayer.WHITE else StubPlayer.WHITE)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StubPlayer) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True)
class StubMove:
    identifier: str

    def __str__(self) -> str:
        return self.identifier


class StubGameState:
    """Minimal search tree state matching the HeuristicAgent expectations."""

    def __init__(self, depth_remaining: int = 2, branching: int = 3, current_player: StubPlayer | None = None):
        self.depth_remaining = depth_remaining
        self.branching = branching
        self.current_player = current_player or StubPlayer(StubPlayer.WHITE)
        self.move_history: List[StubMove] = []

    def copy(self) -> "StubGameState":
        clone = StubGameState(self.depth_remaining, self.branching, StubPlayer(self.current_player.value))
        clone.move_history = list(self.move_history)
        return clone

    def get_valid_moves(self) -> List[StubMove]:
        if self.is_terminal():
            return []
        return [StubMove(f"{self.depth_remaining}_{idx}") for idx in range(self.branching)]

    def make_move(self, move: StubMove) -> bool:
        if move not in self.get_valid_moves():
            return False

        self.move_history.append(move)
        if self.depth_remaining > 0:
            self.depth_remaining -= 1
        self.current_player = self.current_player.opponent
        return True

    def is_terminal(self) -> bool:
        return self.depth_remaining == 0


class DummyHeuristic:
    """Minimal heuristic stub for deterministic testing."""

    def evaluate_position(self, game_state: StubGameState, player: StubPlayer) -> float:
        # Prefer positions with more moves made by the perspective player.
        multiplier = 1 if player.value == StubPlayer.WHITE else -1
        return float(len(game_state.move_history)) * multiplier


def _collect_valid_moves(state: StubGameState) -> Tuple[StubMove, ...]:
    """Helper to collect current legal moves."""
    return tuple(state.get_valid_moves())


def test_select_move_returns_valid_move():
    state = StubGameState(depth_remaining=2, branching=4)
    agent = HeuristicAgent(
        config=HeuristicAgentConfig(
            max_depth=1,
            time_limit_seconds=0.0,
            random_tiebreak=False,
        ),
        evaluator=DummyHeuristic(),
    )

    valid_moves = _collect_valid_moves(state)
    move = agent.select_move(state)

    assert move in valid_moves, "Agent should return one of the legal moves"
    assert agent.last_search_stats["depth_reached"] <= 1
    assert agent.last_search_stats["nodes_evaluated"] > 0
    metrics = agent.last_search_stats["depth_metrics"]
    assert metrics, "Depth metrics should be recorded"
    assert metrics[0]["completed"] is True


def test_depth_limit_respected():
    state = StubGameState(depth_remaining=3, branching=2)
    agent = HeuristicAgent(
        config=HeuristicAgentConfig(
            max_depth=2,
            use_iterative_deepening=False,
            time_limit_seconds=0.0,
            random_tiebreak=False,
        ),
        evaluator=DummyHeuristic(),
    )

    agent.select_move(state)
    assert agent.last_search_stats["depth_reached"] <= 2
    assert agent.last_search_stats["last_completed_depth"] <= 2
    assert not math.isnan(agent.last_search_stats["score"])
    assert agent.last_search_stats["depth_metrics"]


def test_timeout_fallback(monkeypatch: pytest.MonkeyPatch):
    state = StubGameState(depth_remaining=2, branching=2)
    agent = HeuristicAgent(
        config=HeuristicAgentConfig(
            max_depth=3,
            time_limit_seconds=0.0,
            random_tiebreak=False,
        ),
        evaluator=DummyHeuristic(),
    )

    def fake_iterative_deepening(*args, **kwargs):
        return None, float("nan"), 0, True, []

    monkeypatch.setattr(agent, "_iterative_deepening_search", fake_iterative_deepening)

    valid_moves = _collect_valid_moves(state)
    move = agent.select_move(state)

    assert move in valid_moves, "Fallback should choose a legal move"
    assert agent.last_search_stats["timed_out"] is True
    assert math.isnan(agent.last_search_stats["score"])
    assert agent.last_search_stats["depth_metrics"] == []


def test_timeout_respects_last_completed_depth(monkeypatch: pytest.MonkeyPatch):
    state = StubGameState(depth_remaining=3, branching=3)
    agent = HeuristicAgent(
        config=HeuristicAgentConfig(
            min_depth=1,
            max_depth=3,
            time_limit_seconds=1.0,
            time_buffer_seconds=0.5,
            random_tiebreak=False,
        ),
        evaluator=DummyHeuristic(),
    )

    def fake_check(start_time: float, allow_overrun: bool = False) -> bool:
        if allow_overrun:
            return False
        return agent._nodes_searched >= 6

    monkeypatch.setattr(agent, "_is_time_exceeded", fake_check)

    move = agent.select_move(state)

    assert move is not None
    assert agent.last_search_stats["last_completed_depth"] == 1
    assert agent.last_search_stats["timed_out"] is True
    metrics = agent.last_search_stats["depth_metrics"]
    assert metrics[-1]["completed"] is False


def test_debug_logging_emits(caplog: pytest.LogCaptureFixture):
    state = StubGameState(depth_remaining=1, branching=2)
    agent = HeuristicAgent(
        config=HeuristicAgentConfig(
            max_depth=1,
            time_limit_seconds=0.0,
            random_tiebreak=False,
            debug=True,
            slow_warning_threshold=0.0,
        ),
        evaluator=DummyHeuristic(),
    )

    with caplog.at_level(logging.DEBUG):
        agent.select_move(state)

    assert any("HeuristicAgent depth" in record.message for record in caplog.records)

