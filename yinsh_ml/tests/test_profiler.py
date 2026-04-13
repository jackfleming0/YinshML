"""Tests for performance profiler."""

import time

import pytest

from yinsh_ml.agents.profiler import (
    PerformanceProfiler,
    ProfileMetrics,
    RegressionAlert,
)
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig


def test_profile_metrics_properties():
    """Test ProfileMetrics computed properties."""
    metrics = ProfileMetrics(
        move_times=[0.1, 0.2, 0.3, 0.4, 0.5],
        memory_usage_mb=100.0,
        nodes_per_second=1000.0,
        game_quality_score=0.8,
        total_games=5,
    )
    
    assert metrics.avg_move_time == 0.3
    assert metrics.min_move_time == 0.1
    assert metrics.max_move_time == 0.5
    assert metrics.p95_move_time >= 0.4
    assert metrics.p99_move_time >= 0.4


def test_profile_metrics_empty():
    """Test ProfileMetrics with empty data."""
    metrics = ProfileMetrics()
    
    assert metrics.avg_move_time == 0.0
    assert metrics.min_move_time == 0.0
    assert metrics.max_move_time == 0.0
    assert metrics.p95_move_time == 0.0
    assert metrics.std_move_time == 0.0


def test_profiler_basic():
    """Test basic profiling functionality."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    agent = HeuristicAgent(
        config=HeuristicAgentConfig(
            max_depth=2,
            time_limit_seconds=0.1,
        )
    )
    
    # Profile with small number of games
    metrics = profiler.profile_agent(agent, num_games=2)
    
    assert metrics.total_games == 2
    assert len(metrics.move_times) > 0
    assert metrics.avg_move_time >= 0.0


def test_regression_detection():
    """Test performance regression detection."""
    profiler = PerformanceProfiler()
    
    baseline = ProfileMetrics(
        move_times=[0.1] * 100,
        memory_usage_mb=50.0,
        nodes_per_second=2000.0,
        game_quality_score=0.8,
        total_games=10,
    )
    
    # Current metrics with regression (slower moves)
    current = ProfileMetrics(
        move_times=[0.15] * 100,  # 50% slower
        memory_usage_mb=75.0,  # 50% more memory
        nodes_per_second=1500.0,  # 25% slower throughput
        game_quality_score=0.7,
        total_games=10,
    )
    
    alerts = profiler.detect_regressions(current, baseline)
    
    # Should detect regressions
    assert len(alerts) > 0
    assert any("avg_move_time" in alert.metric_name for alert in alerts)


def test_bottleneck_identification():
    """Test bottleneck identification."""
    profiler = PerformanceProfiler()
    
    # Metrics with high variance (potential bottleneck)
    metrics = ProfileMetrics(
        move_times=[0.1, 0.2, 0.3, 0.4, 1.0, 1.5],  # High variance
        memory_usage_mb=600.0,  # High memory
        nodes_per_second=500.0,  # Low throughput
        game_quality_score=0.5,
        total_games=6,
    )
    
    bottlenecks = profiler.identify_bottlenecks(metrics)
    
    # Should identify some bottlenecks
    assert isinstance(bottlenecks, list)


def test_performance_report_generation():
    """Test performance report generation."""
    profiler = PerformanceProfiler()
    
    metrics = ProfileMetrics(
        move_times=[0.1, 0.2, 0.3],
        memory_usage_mb=100.0,
        nodes_per_second=1000.0,
        game_quality_score=0.8,
        total_games=3,
    )
    
    report = profiler.generate_performance_report(metrics)
    
    assert isinstance(report, str)
    assert "Performance Profile Report" in report
    assert "Average Move Time" in report or "avg_move_time" in report.lower()


def test_regression_alert_string():
    """Test RegressionAlert string representation."""
    alert = RegressionAlert(
        metric_name="avg_move_time",
        baseline_value=0.1,
        current_value=0.15,
        change_percent=50.0,
        severity="high",
        threshold=20.0,
    )
    
    str_repr = str(alert)
    assert "avg_move_time" in str_repr
    assert "high" in str_repr.lower()
    assert "50" in str_repr or "50.0" in str_repr

