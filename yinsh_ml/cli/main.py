"""
Main CLI entry point for YinshML experiment tracking.

This module provides the main command-line interface for the yinsh-track tool.
"""

import sys
import logging
import click
from typing import Optional

from .config import get_config, set_config, CLIConfig
from .commands import start, list_cmd, compare, reproduce, search, tensorboard, migrate, launch
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
@click.option('--no-tensorboard-setup', is_flag=True,
              help='Skip automatic TensorBoard environment setup')
@click.option('--install-completion', 
              type=click.Choice(['bash', 'zsh']),
              help='Generate shell completion script')
@click.version_option(version='0.1.0', prog_name='yinsh-track')
@click.pass_context
def cli(ctx, config: Optional[str], database: Optional[str], 
        verbose: bool, quiet: bool, no_color: bool, no_tensorboard_setup: bool,
        install_completion: Optional[str]):
    """
    YinshML Experiment Tracking CLI
    
    A command-line interface for managing machine learning experiments,
    including creation, tracking, comparison, and reproduction.
    
    Automatically configures TensorBoard environment for seamless experiment tracking.
    
    Examples:
        yinsh-track start "New experiment" --description "Testing new model"
        yinsh-track list --status running
        yinsh-track compare 123 124 --metrics accuracy loss
        yinsh-track reproduce 123
        yinsh-track launch smoke --device mps  # Full training with TensorBoard
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
    
    # Set up TensorBoard environment automatically (unless disabled)
    if not no_tensorboard_setup:
        try:
            cli_config.setup_tensorboard_environment()
            if verbose:
                click.echo(click.style("✅ TensorBoard environment configured", fg="green"))
        except Exception as e:
            if verbose:
                click.echo(click.style(f"⚠️  TensorBoard setup warning: {e}", fg="yellow"))
    
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
cli.add_command(launch.launch)


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


@cli.command()
@click.pass_context
def env(ctx):
    """Show current TensorBoard and experiment tracking environment variables."""
    import os
    
    click.echo("TensorBoard Environment:")
    click.echo("=" * 50)
    
    tensorboard_vars = [
        'YINSH_TENSORBOARD_LOGGING',
        'YINSH_TENSORBOARD_LOG_DIR', 
        'YINSH_TENSORBOARD_PORT',
        'YINSH_TENSORBOARD_HOST'
    ]
    
    for var in tensorboard_vars:
        value = os.environ.get(var, '<not set>')
        status = "✅" if var in os.environ else "❌"
        click.echo(f"{status} {var:<30}: {value}")
    
    click.echo(f"\nTensorBoard logs directory: {os.environ.get('YINSH_TENSORBOARD_LOG_DIR', './logs')}")
    
    # Check if logs directory exists
    from pathlib import Path
    log_dir = Path(os.environ.get('YINSH_TENSORBOARD_LOG_DIR', './logs'))
    if log_dir.exists():
        click.echo(f"✅ Log directory exists: {log_dir.absolute()}")
        
        # Count experiment directories
        experiment_dirs = list(log_dir.glob('experiment_*'))
        click.echo(f"📊 Found {len(experiment_dirs)} experiment log directories")
        
        if experiment_dirs:
            click.echo("\nRecent experiments:")
            for exp_dir in sorted(experiment_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                click.echo(f"   • {exp_dir.name}")
    
    else:
        click.echo(f"❌ Log directory not found: {log_dir.absolute()}")


if __name__ == '__main__':
    cli() 