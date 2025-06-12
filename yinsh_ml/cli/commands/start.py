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
    colorize_status,
    format_success_message,
    verbose_echo,
    debug_echo
)
from ..config import get_config


@click.command()
@click.argument('name')
@click.option('--description', '-d', 
              help='Description of the experiment')
@click.option('--tags', '-t', multiple=True,
              help='Tags for the experiment (can be used multiple times)')
@click.option('--config-file', '--config',
              help='Path to experiment configuration file (JSON format)')
@click.option('--parameter', '-p', multiple=True,
              help='Set parameter in key=value format (can be used multiple times)')
@click.pass_context
def start(ctx, name: str, description: Optional[str], tags: List[str], 
          config_file: Optional[str], parameter: List[str]):
    """
    Start a new experiment with the given NAME.
    
    Creates a new experiment entry and begins tracking. The experiment will
    automatically capture system information, git status, and environment details.
    
    \b
    Parameters can be provided in key=value format and will be automatically
    type-converted (strings, numbers, booleans). Configuration files should
    be in JSON format.
    
    \b
    Examples:
        # Basic experiment
        yinsh-track start "Model v2.1" --description "Testing new architecture"
        
        # With tags and parameters
        yinsh-track start "Hyperparameter Search" \\
            --tags ml --tags tuning \\
            --parameter lr=0.001 --parameter batch_size=32
        
        # Using configuration file
        yinsh-track start "Production Run" \\
            --config config/model.json \\
            --description "Final model training"
        
        # Complex parameter types
        yinsh-track start "A/B Test" \\
            --parameter model_type=transformer \\
            --parameter dropout=0.1 \\
            --parameter use_attention=true
    
    \b
    Tips:
        • Use descriptive names for easy identification
        • Add relevant tags for filtering and organization
        • Configuration files override individual parameters
        • Use 'yinsh-track list' to view created experiments
    """
    config = get_config()
    verbose = config.get('verbose', False)
    use_color = config.get('color_output', True)
    
    try:
        debug_echo(f"Starting experiment creation for: {name}")
        
        # Parse parameters
        params = {}
        if parameter:
            verbose_echo(f"Parsing {len(parameter)} parameters...")
            params = parse_parameters(parameter)
            debug_echo(f"Parsed parameters: {params}")
        
        # Load configuration file if provided
        file_config = {}
        if config_file:
            verbose_echo(f"Loading configuration from: {config_file}")
            if not os.path.exists(config_file):
                raise click.ClickException(f"Configuration file not found: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        file_config = json.load(f)
                    else:
                        # Try to parse as JSON anyway
                        file_config = json.load(f)
                debug_echo(f"Loaded {len(file_config)} configuration items")
            except (json.JSONDecodeError, IOError) as e:
                raise click.ClickException(f"Failed to load configuration file: {e}")
        
        # Merge parameters and file config
        experiment_config = {**file_config, **params}
        debug_echo(f"Final config: {experiment_config}")
        
        # Get experiment tracker
        verbose_echo("Initializing experiment tracker...")
        tracker = get_experiment_tracker()
        
        # Create experiment
        verbose_echo("Creating experiment entry...")
        experiment_id = tracker.create_experiment(
            name=name,
            description=description,
            tags=list(tags) if tags else None,
            config=experiment_config if experiment_config else None
        )
        
        # Start the experiment
        verbose_echo("Starting experiment tracking...")
        tracker.start_experiment(experiment_id)
        
        # Display success message
        details = {
            "ID": experiment_id,
            "Name": name,
            "Status": colorize_status("running", use_color)
        }
        
        if description:
            details["Description"] = description
        
        if tags:
            details["Tags"] = ', '.join(tags)
        
        if experiment_config:
            details["Parameters"] = f"{len(experiment_config)} configured"
        
        if config_file:
            details["Config file"] = config_file
        
        click.echo(format_success_message("Experiment created successfully!", details))
        
        # Show parameter details in verbose mode
        if verbose and experiment_config:
            click.echo(click.style("\nParameter Details:", fg='cyan'))
            for key, value in experiment_config.items():
                click.echo(f"    {key}: {value}")
        
        # Next steps
        click.echo(click.style("\n💡 Next steps:", fg='yellow'))
        click.echo(f"  • Use 'yinsh-track list' to see all experiments")
        click.echo(f"  • Use 'yinsh-track show {experiment_id}' to view details")
        click.echo(f"  • Start logging metrics with the tracking API")
        
    except Exception as e:
        handle_error(e, verbose, context="experiment creation")
        ctx.exit(1) 