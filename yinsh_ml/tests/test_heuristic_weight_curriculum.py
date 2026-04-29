"""Tests for the heuristic-weight curriculum in TrainingSupervisor.

The curriculum dampens the iter-1/2 warm-start regression by linearly annealing
self-play's heuristic_weight over a fixed number of iterations. Two things to
verify: (a) the schedule function's shape at endpoints and midpoints; (b) the
per-iteration update propagates to every surface that MCTS reads from — the
SelfPlay instance attr, the main MCTS object, and the mcts_config dict that
worker processes pick up when they spawn their own per-game MCTS.
"""

import types
import pytest

from yinsh_ml.training.supervisor import TrainingSupervisor


class _StubMCTS:
    def __init__(self, hw):
        self.heuristic_weight = hw


class _StubSelfPlay:
    def __init__(self, hw):
        self.heuristic_weight = hw
        self.mcts = _StubMCTS(hw)
        self.mcts_config = {'heuristic_weight': hw}
        self.current_iteration = 0
        self.network = None


@pytest.fixture
def stub_supervisor():
    """Minimal supervisor surface: just the curriculum fields + a fake self_play."""
    sup = types.SimpleNamespace()
    sup._hw_start = 1.0
    sup._hw_end = 0.0
    sup._hw_anneal_iters = 10
    sup.self_play = _StubSelfPlay(hw=sup._hw_start)
    sup.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    sup._compute_heuristic_weight = TrainingSupervisor._compute_heuristic_weight.__get__(sup)
    sup._apply_heuristic_curriculum = TrainingSupervisor._apply_heuristic_curriculum.__get__(sup)
    return sup


class TestScheduleShape:
    @pytest.mark.parametrize("iteration,expected", [
        (0, 1.0),
        (1, 0.9),
        (5, 0.5),
        (7, 0.3),
        (10, 0.0),
        (15, 0.0),
        (100, 0.0),
    ])
    def test_linear_schedule(self, stub_supervisor, iteration, expected):
        assert stub_supervisor._compute_heuristic_weight(iteration) == pytest.approx(expected, abs=1e-6)

    def test_anneal_zero_snaps_to_end(self, stub_supervisor):
        stub_supervisor._hw_anneal_iters = 0
        assert stub_supervisor._compute_heuristic_weight(0) == pytest.approx(0.0)
        assert stub_supervisor._compute_heuristic_weight(5) == pytest.approx(0.0)

    def test_negative_iteration_clamps_to_start(self, stub_supervisor):
        assert stub_supervisor._compute_heuristic_weight(-1) == pytest.approx(stub_supervisor._hw_start)

    def test_start_equals_end_is_constant(self, stub_supervisor):
        stub_supervisor._hw_start = 0.3
        stub_supervisor._hw_end = 0.3
        for i in (0, 5, 100):
            assert stub_supervisor._compute_heuristic_weight(i) == pytest.approx(0.3)


class TestPropagation:
    """train_iteration updates three surfaces: self_play.heuristic_weight,
    self_play.mcts.heuristic_weight, and self_play.mcts_config['heuristic_weight'].
    Workers read from the last one to build fresh per-game MCTS instances, so all
    three must move together or the curriculum only hits the main-process MCTS."""

    def test_all_three_surfaces_update_together(self, stub_supervisor):
        for iteration in (0, 3, 7, 10, 20):
            hw = stub_supervisor._apply_heuristic_curriculum(iteration)
            assert stub_supervisor.self_play.heuristic_weight == hw
            assert stub_supervisor.self_play.mcts.heuristic_weight == hw
            assert stub_supervisor.self_play.mcts_config['heuristic_weight'] == hw

    def test_progression_matches_schedule(self, stub_supervisor):
        trajectory = [stub_supervisor._apply_heuristic_curriculum(i) for i in range(12)]
        assert trajectory[0] == pytest.approx(1.0)
        assert trajectory[5] == pytest.approx(0.5)
        assert trajectory[10] == pytest.approx(0.0)
        assert trajectory[11] == pytest.approx(0.0)  # held
