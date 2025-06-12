"""
Main CLI entry point for YinshML experiment tracking.

This module provides the main command-line interface for the yinsh-track tool.
"""

import sys
import logging
import click
from typing import Optional

from .config import get_config, set_config, CLIConfig
from .commands import start, list_cmd, compare, reproduce, search, tensorboard, migrate
from .completion import generate_completion_script, install_completion


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging configuration."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@click.group(name='yinsh-track', invoke_without_command=True)
@click.option('--config', '-c', 
              help='Path to configuration file')
@click.option('--database', '--db', 
              help='Path to experiment database')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress non-essential output')
@click.option('--no-color', is_flag=True,
              help='Disable colored output')
@click.option('--install-completion', 
              type=click.Choice(['bash', 'zsh']),
              help='Generate shell completion script')
@click.version_option(version='0.1.0', prog_name='yinsh-track')
@click.pass_context
def cli(ctx, config: Optional[str], database: Optional[str], 
        verbose: bool, quiet: bool, no_color: bool, install_completion: Optional[str]):
    """
    YinshML Experiment Tracking CLI
    
    A command-line interface for managing machine learning experiments,
    including creation, tracking, comparison, and reproduction.
    
    Examples:
        yinsh-track start "New experiment" --description "Testing new model"
        yinsh-track list --status running
        yinsh-track compare 123 124 --metrics accuracy loss
        yinsh-track reproduce 123
    """
    # Handle completion installation first, before other processing
    if install_completion:
        script = generate_completion_script(install_completion)
        click.echo(script)
        return
    
    # If no command is provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return
    
    # Setup logging
    setup_logging(verbose, quiet)
    
    # Load configuration
    if config:
        cli_config = CLIConfig(config_file=config)
    else:
        cli_config = get_config()
    
    # Override config with command line options
    if database:
        cli_config.set('database_path', database)
    if verbose:
        cli_config.set('verbose', True)
    if quiet:
        cli_config.set('quiet', True)
    if no_color:
        cli_config.set('color_output', False)
    
    set_config(cli_config)
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['config'] = cli_config


# Register command groups
cli.add_command(start.start)
cli.add_command(list_cmd.list_experiments)
cli.add_command(compare.compare)
cli.add_command(reproduce.reproduce)
cli.add_command(search.search)
cli.add_command(tensorboard.tensorboard)
cli.add_command(migrate.migrate)


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    config_obj = ctx.obj['config']
    
    click.echo("Current configuration:")
    click.echo("=" * 50)
    
    for key, value in config_obj.to_dict().items():
        click.echo(f"{key:<25}: {value}")
    
    if config_obj._config_file:
        click.echo(f"\nLoaded from: {config_obj._config_file}")
    else:
        click.echo("\nUsing default configuration (no config file found)")


if __name__ == '__main__':
    cli() 