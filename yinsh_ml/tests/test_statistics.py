"""Tests for statistical analysis module."""

import pytest

from yinsh_ml.agents.statistics import (
    StatisticalAnalyzer,
    ConfidenceInterval,
    SignificanceTest,
    ELORating,
)
from yinsh_ml.agents.tournament import TournamentMetrics


def test_confidence_interval_calculation():
    """Test confidence interval calculation."""
    analyzer = StatisticalAnalyzer()
    
    # Test with 100 games, 60% win rate
    ci = analyzer.compute_confidence_interval(0.6, 100, 0.95)
    
    assert ci.lower < 0.6 < ci.upper
    assert ci.confidence == 0.95
    assert 0.0 <= ci.lower <= 1.0
    assert 0.0 <= ci.upper <= 1.0


def test_confidence_interval_edge_cases():
    """Test confidence interval edge cases."""
    analyzer = StatisticalAnalyzer()
    
    # Zero games
    ci = analyzer.compute_confidence_interval(0.5, 0, 0.95)
    assert ci.lower == 0.0
    assert ci.upper == 1.0
    
    # All wins
    ci = analyzer.compute_confidence_interval(1.0, 100, 0.95)
    assert ci.upper == 1.0
    assert ci.lower > 0.0
    
    # No wins
    ci = analyzer.compute_confidence_interval(0.0, 100, 0.95)
    assert ci.lower == 0.0
    assert ci.upper < 1.0


def test_significance_test():
    """Test statistical significance testing."""
    analyzer = StatisticalAnalyzer()
    
    baseline = TournamentMetrics(
        total_games=1000,
        wins=500,
        losses=450,
        draws=50,
        win_rate=0.5,
        average_game_length=50.0,
        std_game_length=10.0,
        average_move_time=0.1,
        max_move_time=0.5,
        nodes_per_second=1000.0,
    )
    
    test = TournamentMetrics(
        total_games=1000,
        wins=600,
        losses=350,
        draws=50,
        win_rate=0.6,
        average_game_length=50.0,
        std_game_length=10.0,
        average_move_time=0.1,
        max_move_time=0.5,
        nodes_per_second=1000.0,
    )
    
    result = analyzer.test_significance(baseline, test)
    
    assert result.test_name == "chi-square"
    assert 0.0 <= result.p_value <= 1.0
    assert isinstance(result.significant, bool)


def test_elo_rating_calculation():
    """Test ELO rating calculation."""
    analyzer = StatisticalAnalyzer()
    
    match_results = [
        {"agent1": "AgentA", "agent2": "AgentB", "winner": "agent1"},
        {"agent1": "AgentA", "agent2": "AgentB", "winner": "agent2"},
        {"agent1": "AgentA", "agent2": "AgentB", "winner": "agent1"},
        {"agent1": "AgentA", "agent2": "AgentC", "winner": "agent1"},
        {"agent1": "AgentB", "agent2": "AgentC", "winner": "agent2"},
    ]
    
    ratings = analyzer.calculate_elo_ratings(match_results)
    
    assert len(ratings) == 3
    assert all(isinstance(r, ELORating) for r in ratings)
    
    # Check that ratings are reasonable
    for rating in ratings:
        assert rating.games_played > 0
        assert 0.0 <= rating.win_rate <= 1.0


def test_performance_report_generation():
    """Test performance report generation."""
    analyzer = StatisticalAnalyzer()
    
    metrics_list = [
        TournamentMetrics(
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
        ),
        TournamentMetrics(
            total_games=100,
            wins=55,
            losses=40,
            draws=5,
            win_rate=0.55,
            average_game_length=50.0,
            std_game_length=10.0,
            average_move_time=0.1,
            max_move_time=0.5,
            nodes_per_second=1000.0,
        ),
    ]
    
    report = analyzer.generate_performance_report(metrics_list, ["Agent1", "Agent2"])
    
    assert isinstance(report, str)
    assert "Agent1" in report
    assert "Agent2" in report
    assert "Win Rate" in report or "win rate" in report.lower()


def test_significance_test_string_representation():
    """Test SignificanceTest string representation."""
    test = SignificanceTest(
        p_value=0.03,
        significant=True,
        test_name="chi-square",
        statistic=4.5,
        alpha=0.05,
    )
    
    str_repr = str(test)
    assert "chi-square" in str_repr
    assert "significant" in str_repr.lower()
    assert "0.03" in str_repr or "0.0300" in str_repr


def test_confidence_interval_string_representation():
    """Test ConfidenceInterval string representation."""
    ci = ConfidenceInterval(lower=0.5, upper=0.7, confidence=0.95)
    
    str_repr = str(ci)
    assert "0.5" in str_repr or "0.5000" in str_repr
    assert "0.7" in str_repr or "0.7000" in str_repr
    assert "95" in str_repr

