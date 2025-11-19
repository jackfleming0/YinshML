"""CLI commands for configuration management.

This module provides command-line interface for managing heuristic
configurations, including viewing, setting, optimizing, and comparing
configurations.
"""

import json
import click
from pathlib import Path
from typing import Optional

from yinsh_ml.heuristics.config_manager import ConfigManager
from yinsh_ml.heuristics.optimizers import (
    GridSearchOptimizer,
    GeneticAlgorithmOptimizer,
)
from yinsh_ml.agents.ab_testing import ABTestRunner, ExperimentConfig, AgentVariant
from yinsh_ml.agents.heuristic_agent import HeuristicAgentConfig


@click.group()
def config():
    """Manage heuristic configuration (weights and phase boundaries)."""
    pass


@config.command()
@click.option('--file', '-f', 'config_file',
              help='Path to configuration file to display')
@click.option('--json', 'output_json', is_flag=True,
              help='Output in JSON format')
def show(config_file: Optional[str], output_json: bool):
    """Display current configuration."""
    manager = ConfigManager()
    
    # Load config if file specified
    if config_file:
        try:
            manager.load_config(config_file)
        except FileNotFoundError:
            click.echo(f"Error: Configuration file not found: {config_file}", err=True)
            return
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            return
    
    # Get current config
    current_config = manager.get_current_config()
    
    if output_json:
        click.echo(json.dumps(current_config, indent=2))
    else:
        click.echo("Current Configuration:")
        click.echo("=" * 50)
        
        # Display weights
        click.echo("\nWeights:")
        for phase in ["early", "mid", "late"]:
            click.echo(f"\n  {phase.upper()} Phase:")
            if phase in current_config["weights"]:
                for feature, value in current_config["weights"][phase].items():
                    click.echo(f"    {feature}: {value:.2f}")
        
        # Display phase config
        click.echo("\nPhase Boundaries:")
        phase_config = current_config.get("phase_config", {})
        click.echo(f"  Early max moves: {phase_config.get('early_max_moves', 'N/A')}")
        click.echo(f"  Mid max moves: {phase_config.get('mid_max_moves', 'N/A')}")
        click.echo(f"  Transition window: {phase_config.get('transition_window', 'N/A')}")


@config.group()
def set():
    """Set configuration values."""
    pass


@set.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--output', '-o', 'output_file',
              help='Path to save the loaded configuration')
def weights(filepath: str, output_file: Optional[str]):
    """Load weights from a JSON file."""
    manager = ConfigManager()
    
    try:
        # Load weights file (should contain weights structure)
        with open(filepath, 'r') as f:
            weights_data = json.load(f)
        
        # Set weights
        if "weights" in weights_data:
            manager.weight_manager.set_default_weights(weights_data["weights"])
        else:
            # Assume entire file is weights structure
            manager.weight_manager.set_default_weights(weights_data)
        
        # Save if output specified
        if output_file:
            manager.save_config(output_file)
            click.echo(f"Configuration saved to {output_file}")
        else:
            click.echo(f"Weights loaded from {filepath}")
            
    except Exception as e:
        click.echo(f"Error loading weights: {e}", err=True)
        return


@set.command()
@click.argument('key')
@click.argument('value', type=float)
@click.option('--file', '-f', 'config_file',
              help='Configuration file to update')
def phase(key: str, value: float, config_file: Optional[str]):
    """Update a phase boundary parameter.
    
    KEY: Parameter name (early_max_moves, mid_max_moves, transition_window)
    VALUE: New value for the parameter
    """
    manager = ConfigManager()
    
    # Load config if file specified
    if config_file:
        try:
            manager.load_config(config_file)
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            return
    
    # Validate key
    valid_keys = ["early_max_moves", "mid_max_moves", "transition_window"]
    if key not in valid_keys:
        click.echo(
            f"Error: Invalid key '{key}'. Must be one of: {', '.join(valid_keys)}",
            err=True
        )
        return
    
    try:
        # Update phase config
        manager.update_phase_config(**{key: int(value)})
        click.echo(f"Updated {key} to {value}")
        
        # Save if config file specified
        if config_file:
            manager.save_config(config_file)
            click.echo(f"Configuration saved to {config_file}")
    except Exception as e:
        click.echo(f"Error updating phase config: {e}", err=True)


@config.group()
def optimize():
    """Run optimization algorithms."""
    pass


@optimize.command()
@click.option('--param-grid', 'param_grid_file',
              type=click.Path(exists=True),
              help='JSON file defining parameter grid')
@click.option('--games', '-g', default=50,
              help='Number of games per evaluation')
@click.option('--seed', type=int,
              help='Random seed for reproducibility')
@click.option('--output', '-o',
              help='Path to save optimization results')
def grid_search(param_grid_file: Optional[str], games: int, seed: Optional[int], output: Optional[str]):
    """Run grid search optimization."""
    manager = ConfigManager()
    optimizer = GridSearchOptimizer(manager)
    
    # Load parameter grid
    if not param_grid_file:
        click.echo("Error: --param-grid required", err=True)
        return
    
    with open(param_grid_file, 'r') as f:
        param_grid = json.load(f)
    
    # Run optimization
    click.echo("Starting grid search optimization...")
    with click.progressbar(length=100, label="Optimizing") as bar:
        result = optimizer.optimize(
            param_grid=param_grid,
            evaluation_games=games,
            random_seed=seed,
        )
        bar.update(100)
    
    # Display results
    click.echo(f"\nOptimization complete!")
    click.echo(f"Best score: {result.best_score:.3f}")
    click.echo(f"Evaluations: {result.num_evaluations}")
    
    # Save results if specified
    if output:
        results_data = {
            "best_config": result.best_config,
            "best_score": result.best_score,
            "num_evaluations": result.num_evaluations,
            "optimization_method": result.optimization_method,
        }
        with open(output, 'w') as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"Results saved to {output}")


@optimize.command()
@click.option('--population', '-p', default=20,
              help='Population size')
@click.option('--generations', '-g', default=10,
              help='Number of generations')
@click.option('--mutation-rate', '-m', default=0.1,
              help='Mutation rate (0.0 to 1.0)')
@click.option('--games', default=50,
              help='Number of games per evaluation')
@click.option('--seed', type=int,
              help='Random seed for reproducibility')
@click.option('--output', '-o',
              help='Path to save optimization results')
def genetic(population: int, generations: int, mutation_rate: float,
            games: int, seed: Optional[int], output: Optional[str]):
    """Run genetic algorithm optimization."""
    manager = ConfigManager()
    optimizer = GeneticAlgorithmOptimizer(manager)
    
    # Run optimization
    click.echo("Starting genetic algorithm optimization...")
    with click.progressbar(length=generations, label="Evolving") as bar:
        result = optimizer.optimize(
            population_size=population,
            generations=generations,
            mutation_rate=mutation_rate,
            evaluation_games=games,
            random_seed=seed,
        )
        bar.update(generations)
    
    # Display results
    click.echo(f"\nOptimization complete!")
    click.echo(f"Best score: {result.best_score:.3f}")
    click.echo(f"Evaluations: {result.num_evaluations}")
    
    # Save results if specified
    if output:
        results_data = {
            "best_config": result.best_config,
            "best_score": result.best_score,
            "num_evaluations": result.num_evaluations,
            "optimization_method": result.optimization_method,
        }
        with open(output, 'w') as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"Results saved to {output}")


@config.command()
@click.argument('config1', type=click.Path(exists=True))
@click.argument('config2', type=click.Path(exists=True))
@click.option('--games', '-g', default=100,
              help='Number of games per matchup')
@click.option('--output', '-o',
              help='Path to save comparison results')
def compare(config1: str, config2: str, games: int, output: Optional[str]):
    """Compare two configurations using A/B testing."""
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    
    # Load configurations
    try:
        manager1.load_config(config1)
        manager2.load_config(config2)
    except Exception as e:
        click.echo(f"Error loading configurations: {e}", err=True)
        return
    
    # Create agent configs
    agent_config1 = HeuristicAgentConfig()
    agent_config2 = HeuristicAgentConfig()
    
    # Create variants
    variant1 = AgentVariant(
        name="Config1",
        config=agent_config1,
        description=f"Configuration from {config1}",
    )
    variant2 = AgentVariant(
        name="Config2",
        config=agent_config2,
        description=f"Configuration from {config2}",
    )
    
    # Create experiment config
    experiment_config = ExperimentConfig(
        name="Config Comparison",
        variants=[variant1, variant2],
        num_games_per_matchup=games,
    )
    
    # Run A/B test
    runner = ABTestRunner()
    click.echo("Running A/B test comparison...")
    results = runner.run_experiment(experiment_config)
    
    # Display results
    dashboard = runner.generate_dashboard(results)
    click.echo(dashboard)
    
    # Save results if specified
    if output:
        with open(output, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        click.echo(f"\nResults saved to {output}")

