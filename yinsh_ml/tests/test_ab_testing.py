"""Tests for A/B testing infrastructure."""

import pytest

from yinsh_ml.agents.ab_testing import (
    ABTestRunner,
    ExperimentConfig,
    AgentVariant,
    ExperimentResults,
    MatchupResult,
)
from yinsh_ml.agents.heuristic_agent import HeuristicAgentConfig


def test_agent_variant_creation():
    """Test AgentVariant creation."""
    config = HeuristicAgentConfig(max_depth=3)
    variant = AgentVariant(
        name="test_variant",
        config=config,
        description="Test variant",
    )
    
    assert variant.name == "test_variant"
    assert variant.config == config
    assert variant.description == "Test variant"


def test_experiment_config_validation():
    """Test ExperimentConfig validation."""
    config1 = HeuristicAgentConfig(max_depth=2)
    config2 = HeuristicAgentConfig(max_depth=3)
    
    variant1 = AgentVariant("variant1", config1)
    variant2 = AgentVariant("variant2", config2)
    
    # Valid config
    exp_config = ExperimentConfig(
        name="test_experiment",
        variants=[variant1, variant2],
        num_games_per_matchup=10,
    )
    
    assert exp_config.name == "test_experiment"
    assert len(exp_config.variants) == 2
    
    # Invalid: not enough variants
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid",
            variants=[variant1],
        )
    
    # Invalid: duplicate names
    variant3 = AgentVariant("variant1", config2)  # Duplicate name
    with pytest.raises(ValueError):
        ExperimentConfig(
            name="invalid",
            variants=[variant1, variant3],
        )


def test_experiment_results_serialization():
    """Test ExperimentResults serialization."""
    config1 = HeuristicAgentConfig(max_depth=2)
    config2 = HeuristicAgentConfig(max_depth=3)
    
    variant1 = AgentVariant("variant1", config1)
    variant2 = AgentVariant("variant2", config2)
    
    exp_config = ExperimentConfig(
        name="test",
        variants=[variant1, variant2],
        num_games_per_matchup=10,
    )
    
    results = ExperimentResults(
        experiment_name="test",
        config=exp_config,
    )
    
    # Test serialization
    data = results.to_dict()
    assert data["experiment_name"] == "test"
    assert "config" in data
    assert "matchup_results" in data


def test_ab_test_runner_initialization():
    """Test ABTestRunner initialization."""
    runner = ABTestRunner()
    assert runner.statistical_analyzer is not None


def test_statistical_power_calculation():
    """Test statistical power calculation."""
    runner = ABTestRunner()
    
    # Calculate required sample size
    sample_size = runner.calculate_statistical_power(
        effect_size=0.05,  # 5% difference
        alpha=0.05,
        power=0.80,
    )
    
    assert sample_size > 0
    assert isinstance(sample_size, int)


def test_dashboard_generation():
    """Test dashboard generation."""
    runner = ABTestRunner()
    
    config1 = HeuristicAgentConfig(max_depth=2)
    config2 = HeuristicAgentConfig(max_depth=3)
    
    variant1 = AgentVariant("variant1", config1)
    variant2 = AgentVariant("variant2", config2)
    
    exp_config = ExperimentConfig(
        name="test_experiment",
        variants=[variant1, variant2],
        num_games_per_matchup=10,
    )
    
    results = ExperimentResults(
        experiment_name="test_experiment",
        config=exp_config,
    )
    
    dashboard = runner.generate_dashboard(results)
    
    assert isinstance(dashboard, str)
    assert "test_experiment" in dashboard
    assert "variant1" in dashboard
    assert "variant2" in dashboard


def test_analyze_results():
    """Test result analysis."""
    runner = ABTestRunner()
    
    config1 = HeuristicAgentConfig(max_depth=2)
    config2 = HeuristicAgentConfig(max_depth=3)
    
    variant1 = AgentVariant("variant1", config1)
    variant2 = AgentVariant("variant2", config2)
    
    exp_config = ExperimentConfig(
        name="test",
        variants=[variant1, variant2],
        num_games_per_matchup=10,
    )
    
    results = ExperimentResults(
        experiment_name="test",
        config=exp_config,
    )
    
    analysis = runner.analyze_results(results)
    
    assert "experiment_name" in analysis
    assert "total_matchups" in analysis
    assert "recommendations" in analysis

