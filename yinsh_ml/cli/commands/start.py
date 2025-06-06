"""
Start command for creating new experiments.
"""

import json
import os
import click
from typing import List, Optional

from ..utils import (
    parse_parameters, 
    handle_error, 
    get_experiment_tracker,
    colorize_status
)
from ..config import get_config


@click.command()
@click.argument('name')
@click.option('--description', '-d', 
              help='Description of the experiment')
@click.option('--tags', '-t', multiple=True,
              help='Tags for the experiment (can be used multiple times)')
@click.option('--config-file', '--config',
              help='Path to experiment configuration file')
@click.option('--parameter', '-p', multiple=True,
              help='Set parameter in key=value format (can be used multiple times)')
@click.pass_context
def start(ctx, name: str, description: Optional[str], tags: List[str], 
          config_file: Optional[str], parameter: List[str]):
    """
    Start a new experiment.
    
    Creates a new experiment with the given NAME and begins tracking.
    
    Examples:
        yinsh-track start "Model v2.1" --description "Testing new architecture"
        yinsh-track start "Hyperparameter Search" --tags ml tuning --parameter lr=0.001
    """
    config = get_config()
    verbose = config.get('verbose', False)
    use_color = config.get('color_output', True)
    
    try:
        # Parse parameters
        params = {}
        if parameter:
            params = parse_parameters(parameter)
        
        # Load configuration file if provided
        file_config = {}
        if config_file:
            if not os.path.exists(config_file):
                raise click.ClickException(f"Configuration file not found: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        file_config = json.load(f)
                    else:
                        # Try to parse as JSON anyway
                        file_config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                raise click.ClickException(f"Failed to load configuration file: {e}")
        
        # Merge parameters and file config
        experiment_config = {**file_config, **params}
        
        # Get experiment tracker
        tracker = get_experiment_tracker()
        
        # Create experiment
        if verbose:
            click.echo("Creating experiment...")
        
        experiment_id = tracker.create_experiment(
            name=name,
            description=description,
            tags=list(tags) if tags else None,
            config=experiment_config if experiment_config else None
        )
        
        # Display success message
        click.echo(f"✓ Experiment created successfully!")
        click.echo(f"  ID: {experiment_id}")
        click.echo(f"  Name: {name}")
        
        if description:
            click.echo(f"  Description: {description}")
        
        if tags:
            click.echo(f"  Tags: {', '.join(tags)}")
        
        if experiment_config:
            click.echo(f"  Parameters: {len(experiment_config)} configured")
            if verbose:
                for key, value in experiment_config.items():
                    click.echo(f"    {key}: {value}")
        
        # Start the experiment
        tracker.start_experiment(experiment_id)
        status_display = colorize_status("running", use_color)
        click.echo(f"  Status: {status_display}")
        
        if config_file:
            click.echo(f"  Config file: {config_file}")
        
        click.echo(f"\nUse 'yinsh-track list' to see all experiments")
        click.echo(f"Use 'yinsh-track show {experiment_id}' to view details (when implemented)")
        
    except Exception as e:
        handle_error(e, verbose)
        ctx.exit(1) 