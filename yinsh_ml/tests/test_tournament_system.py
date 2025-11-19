"""Tests for enhanced tournament system."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from yinsh_ml.agents.tournament import (
    TournamentEvaluator,
    TournamentConfig,
    TournamentMetrics,
)


class DummyAgent:
    """Dummy agent for testing."""
    
    def __init__(self, win_probability=0.5):
        self.win_probability = win_probability
        self.last_search_stats = {"nodes_evaluated": 100}
    
    def select_move(self, game_state):
        from yinsh_ml.game.moves import MoveGenerator
        valid_moves = MoveGenerator.get_valid_moves(game_state.board, game_state)
        return valid_moves[0] if valid_moves else None


class DummyOpponent:
    """Dummy opponent for testing."""
    
    def select_move(self, game_state):
        from yinsh_ml.game.moves import MoveGenerator
        valid_moves = MoveGenerator.get_valid_moves(game_state.board, game_state)
        return valid_moves[0] if valid_moves else None


def test_tournament_config_defaults():
    """Test TournamentConfig default values."""
    config = TournamentConfig()
    assert config.num_games == 1000
    assert config.concurrent_workers == 4
    assert config.save_interval == 100
    assert config.format == "round-robin"


def test_tournament_metrics_serialization():
    """Test TournamentMetrics serialization."""
    metrics = TournamentMetrics(
        total_games=100,
        wins=60,
        losses=35,
        draws=5,
        win_rate=0.6,
        average_game_length=50.0,
        std_game_length=10.0,
        average_move_time=0.1,
        max_move_time=0.5,
        nodes_per_second=1000.0,
    )
    
    # Test to_dict
    data = metrics.to_dict()
    assert data["total_games"] == 100
    assert data["win_rate"] == 0.6
    
    # Test from_dict
    restored = TournamentMetrics.from_dict(data)
    assert restored.total_games == metrics.total_games
    assert restored.win_rate == metrics.win_rate


def test_tournament_persistence():
    """Test tournament result persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "results.json")
        
        evaluator = TournamentEvaluator()
        config = TournamentConfig(
            num_games=10,
            output_path=output_path,
            save_interval=5,
        )
        
        # Run small tournament
        metrics = evaluator.run_large_scale_tournament(config)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Load and verify
        with open(output_path, "r") as f:
            data = json.load(f)
        
        assert "metrics" in data
        assert "timestamp" in data
        assert data["metrics"]["total_games"] == 10


def test_tournament_resume():
    """Test tournament resumption from saved results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "results.json")
        
        evaluator = TournamentEvaluator()
        
        # Create initial results
        initial_metrics = TournamentMetrics(
            total_games=5,
            wins=3,
            losses=2,
            draws=0,
            win_rate=0.6,
            average_game_length=50.0,
            std_game_length=10.0,
            average_move_time=0.1,
            max_move_time=0.5,
            nodes_per_second=1000.0,
        )
        
        evaluator._save_results(output_path, initial_metrics)
        
        # Resume tournament
        config = TournamentConfig(
            num_games=10,
            output_path=output_path,
            resume=True,
        )
        
        final_metrics = evaluator.run_large_scale_tournament(config)
        
        # Should have 10 total games (5 initial + 5 new)
        assert final_metrics.total_games == 10


def test_tournament_basic_execution():
    """Test basic tournament execution."""
    evaluator = TournamentEvaluator()
    metrics = evaluator.run_tournament(games=10)
    
    assert metrics.total_games == 10
    assert metrics.wins + metrics.losses + metrics.draws == 10
    assert 0.0 <= metrics.win_rate <= 1.0
    assert metrics.average_game_length > 0


def test_tournament_metrics_properties():
    """Test TournamentMetrics computed properties."""
    metrics = TournamentMetrics(
        total_games=100,
        wins=60,
        losses=30,
        draws=10,
        win_rate=0.6,
        average_game_length=50.0,
        std_game_length=10.0,
        average_move_time=0.1,
        max_move_time=0.5,
        nodes_per_second=1000.0,
    )
    
    assert metrics.loss_rate == 0.3
    assert metrics.draw_rate == 0.1
    assert abs(metrics.win_rate + metrics.loss_rate + metrics.draw_rate - 1.0) < 0.001

