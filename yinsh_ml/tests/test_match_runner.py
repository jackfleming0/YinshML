"""Tests for the parallel match runner (aggregation + color alternation)."""

import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[2] / "scripts" / "experiments"
sys.path.insert(0, str(EXP))
import match_runner as mr  # noqa: E402
from yinsh_ml.game.constants import Player  # noqa: E402


def test_aggregation_and_color_alternation(monkeypatch):
    # Stub out agent construction and game play so the test is fast and
    # deterministic. play_game always returns WHITE as the winner.
    monkeypatch.setattr(mr.vw, "_make_agent", lambda *a, **k: object())
    monkeypatch.setattr(mr.vw, "play_game", lambda white, black: Player.WHITE)

    # Colors alternate by game index: even -> A is white (A wins),
    # odd -> A is black (white=B wins). 4 games => 2 A wins, 2 B wins.
    res = mr.run_ab_parallel("a.json", "b.json", games=4, depth=1, seed=0, workers=1)
    assert res["a_wins"] == 2
    assert res["b_wins"] == 2
    assert res["draws"] == 0
    assert res["a_win_rate"] == 0.5


def test_draw_handling(monkeypatch):
    monkeypatch.setattr(mr.vw, "_make_agent", lambda *a, **k: object())
    monkeypatch.setattr(mr.vw, "play_game", lambda white, black: None)
    res = mr.run_ab_parallel("a.json", "b.json", games=3, depth=1, seed=0, workers=1)
    assert res["draws"] == 3
    assert res["a_win_rate"] == 0.5  # draws count as half


@pytest.mark.slow
def test_real_parallel_smoke():
    base = str(EXP.parents[1] / "configs" / "heuristic_weights" / "baseline.json")
    res = mr.run_ab_parallel(base, base, games=2, depth=1, seed=1, workers=2)
    assert res["a_wins"] + res["b_wins"] + res["draws"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
