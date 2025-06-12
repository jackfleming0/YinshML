"""
TensorBoard management commands for YinshML experiment tracking.

This module provides CLI commands for starting, stopping, and managing
TensorBoard servers, as well as specialized Yinsh-specific visualization utilities.
"""

import sys
import os
import subprocess
import signal
import time
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging
import psutil
import click
from datetime import datetime

from ..config import get_config
from ..utils import handle_error, get_experiment_tracker, verbose_echo
from ...tracking import ExperimentTracker, TensorBoardLogger, YinshBoardVisualizer


logger = logging.getLogger(__name__)

# Global state for TensorBoard process management
TENSORBOARD_PID_FILE = Path.home() / '.yinsh_tensorboard.pid'
DEFAULT_TENSORBOARD_PORT = 6006


class TensorBoardManager:
    """Manages TensorBoard server processes and configuration."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.pid_file = TENSORBOARD_PID_FILE
        
    def is_running(self) -> Tuple[bool, Optional[int]]:
        """Check if TensorBoard server is running."""
        if not self.pid_file.exists():
            return False, None
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists and is TensorBoard
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                if 'tensorboard' in proc.name().lower() or any('tensorboard' in arg for arg in proc.cmdline()):
                    return True, pid
            
            # PID file exists but process is not running, clean up
            self.pid_file.unlink()
            return False, None
            
        except (ValueError, psutil.NoSuchProcess, PermissionError):
            # Clean up invalid PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False, None
    
    def start_server(self, logdir: str, port: int = DEFAULT_TENSORBOARD_PORT, 
                    bind_all: bool = False, reload_interval: int = 30) -> int:
        """Start TensorBoard server."""
        is_running, existing_pid = self.is_running()
        if is_running:
            raise click.ClickException(f"TensorBoard is already running with PID {existing_pid}")
        
        # Build command
        cmd = [
            'tensorboard',
            '--logdir', logdir,
            '--port', str(port),
            '--reload_interval', str(reload_interval),
            '--load_fast', 'false'  # Ensure all data is loaded
        ]
        
        if bind_all:
            cmd.extend(['--bind_all'])
        
        try:
            # Start TensorBoard as background process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if process started successfully
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                raise click.ClickException(f"TensorBoard failed to start: {stderr.decode()}")
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(proc.pid))
            
            return proc.pid
            
        except FileNotFoundError:
            raise click.ClickException("TensorBoard not found. Please install: pip install tensorboard")
        except Exception as e:
            raise click.ClickException(f"Failed to start TensorBoard: {e}")
    
    def stop_server(self, force: bool = False) -> bool:
        """Stop TensorBoard server."""
        is_running, pid = self.is_running()
        if not is_running:
            return False
        
        try:
            proc = psutil.Process(pid)
            
            if force:
                proc.kill()
            else:
                proc.terminate()
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=10)
                except psutil.TimeoutExpired:
                    proc.kill()
            
            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            return True
            
        except psutil.NoSuchProcess:
            # Process already gone, clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True
        except Exception as e:
            raise click.ClickException(f"Failed to stop TensorBoard: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get TensorBoard server status information."""
        is_running, pid = self.is_running()
        
        status = {
            'running': is_running,
            'pid': pid,
            'port': None,
            'logdir': None,
            'uptime': None,
            'memory_usage': None
        }
        
        if is_running and pid:
            try:
                proc = psutil.Process(pid)
                status['uptime'] = time.time() - proc.create_time()
                status['memory_usage'] = proc.memory_info().rss / 1024 / 1024  # MB
                
                # Try to extract port and logdir from command line
                cmdline = proc.cmdline()
                for i, arg in enumerate(cmdline):
                    if arg == '--port' and i + 1 < len(cmdline):
                        status['port'] = int(cmdline[i + 1])
                    elif arg == '--logdir' and i + 1 < len(cmdline):
                        status['logdir'] = cmdline[i + 1]
                        
            except psutil.NoSuchProcess:
                status['running'] = False
                status['pid'] = None
        
        return status


def get_experiment_logdir(experiment_id: int) -> Optional[Path]:
    """Get the TensorBoard log directory for an experiment."""
    try:
        tracker = ExperimentTracker()
        logger_instance = tracker._tensorboard_logger
        
        if logger_instance:
            return Path(logger_instance.get_log_dir(experiment_id))
        
        # Fallback: construct path from default structure
        base_log_dir = Path('logs')  # Default location
        return base_log_dir / f'experiment_{experiment_id}'
        
    except Exception as e:
        logger.warning(f"Could not determine log directory for experiment {experiment_id}: {e}")
        return None


def collect_experiment_logdirs(experiment_ids: List[int]) -> Dict[int, Path]:
    """Collect log directories for multiple experiments."""
    logdirs = {}
    
    for exp_id in experiment_ids:
        logdir = get_experiment_logdir(exp_id)
        if logdir and logdir.exists():
            logdirs[exp_id] = logdir
        else:
            click.echo(f"Warning: No log directory found for experiment {exp_id}", err=True)
    
    return logdirs


@click.group(name='tensorboard', invoke_without_command=True)
@click.pass_context
def tensorboard(ctx):
    """
    TensorBoard management and visualization utilities.
    
    Manage TensorBoard servers, compare experiments, and generate
    Yinsh-specific visualizations for training analysis.
    
    Examples:
        yinsh-track tensorboard start
        yinsh-track tensorboard compare 123 124 125
        yinsh-track tensorboard export 123 --format csv
        yinsh-track tensorboard visualize 123 --iteration 100
    """
    if ctx.invoked_subcommand is None:
        # Show status if no subcommand
        ctx.invoke(status)


@tensorboard.command()
@click.option('--port', '-p', default=DEFAULT_TENSORBOARD_PORT, type=int,
              help=f'Port to run TensorBoard on (default: {DEFAULT_TENSORBOARD_PORT})')
@click.option('--bind-all', is_flag=True,
              help='Bind to all interfaces (allows external access)')
@click.option('--reload-interval', default=30, type=int,
              help='How often to reload data (seconds)')
@click.option('--experiments', '-e', multiple=True, type=int,
              help='Specific experiment IDs to include')
@click.option('--all-experiments', is_flag=True,
              help='Include all experiments')
@click.option('--recent', type=int, metavar='N',
              help='Include N most recent experiments')
@click.pass_context
def start(ctx, port: int, bind_all: bool, reload_interval: int,
          experiments: List[int], all_experiments: bool, recent: Optional[int]):
    """Start TensorBoard server with experiment logs."""
    config = ctx.obj['config']
    manager = TensorBoardManager(config)
    
    # Determine which experiments to include
    experiment_ids = []
    
    if experiments:
        experiment_ids = list(experiments)
    elif all_experiments:
        try:
            tracker = get_experiment_tracker()
            experiments = tracker.query_experiments(status=None, limit=None)
            experiment_ids = [exp['id'] for exp in experiments]
        except Exception as e:
            handle_error(e, verbose=config.get('verbose', False), context="querying experiments")
            raise click.ClickException(f"Failed to query experiments: {e}")
    elif recent:
        try:
            tracker = get_experiment_tracker()
            experiments = tracker.query_experiments(status=None, limit=recent)
            experiment_ids = [exp['id'] for exp in experiments]
        except Exception as e:
            handle_error(e, verbose=config.get('verbose', False), context="querying recent experiments")
            raise click.ClickException(f"Failed to query recent experiments: {e}")
    else:
        # Default: use all experiments
        try:
            tracker = get_experiment_tracker()
            experiments = tracker.query_experiments(status=None, limit=None)
            experiment_ids = [exp['id'] for exp in experiments]
        except Exception:
            # If no database, try to find log directories
            logs_dir = Path('logs')
            if logs_dir.exists():
                for exp_dir in logs_dir.glob('experiment_*'):
                    try:
                        exp_id = int(exp_dir.name.split('_')[1])
                        experiment_ids.append(exp_id)
                    except (ValueError, IndexError):
                        continue
            
            if not experiment_ids:
                raise click.ClickException("No experiments found. Run some experiments first or specify --experiments")
    
    if not experiment_ids:
        raise click.ClickException("No experiments to visualize")
    
    # Collect log directories
    logdirs = collect_experiment_logdirs(experiment_ids)
    
    if not logdirs:
        raise click.ClickException("No valid experiment log directories found")
    
    # Create a unified log directory structure for TensorBoard
    temp_logdir = Path('.tensorboard_logs')
    temp_logdir.mkdir(exist_ok=True)
    
    # Create symlinks to experiment log directories
    for exp_id, logdir in logdirs.items():
        symlink_path = temp_logdir / f'experiment_{exp_id}'
        if symlink_path.exists():
            symlink_path.unlink()  # Remove existing symlink
        symlink_path.symlink_to(logdir.absolute())
    
    try:
        # Start TensorBoard
        pid = manager.start_server(
            logdir=str(temp_logdir),
            port=port,
            bind_all=bind_all,
            reload_interval=reload_interval
        )
        
        click.echo(f"✅ TensorBoard started successfully!")
        click.echo(f"   PID: {pid}")
        click.echo(f"   Port: {port}")
        click.echo(f"   Experiments: {', '.join(map(str, sorted(logdirs.keys())))}")
        click.echo(f"   URL: http://{'0.0.0.0' if bind_all else 'localhost'}:{port}")
        click.echo(f"\n💡 To stop TensorBoard: yinsh-track tensorboard stop")
        
    except click.ClickException:
        # Clean up temp directory on failure
        for symlink in temp_logdir.glob('*'):
            if symlink.is_symlink():
                symlink.unlink()
        temp_logdir.rmdir()
        raise


@tensorboard.command()
@click.option('--force', '-f', is_flag=True,
              help='Force kill TensorBoard process')
@click.pass_context
def stop(ctx, force: bool):
    """Stop running TensorBoard server."""
    config = ctx.obj['config']
    manager = TensorBoardManager(config)
    
    is_running, pid = manager.is_running()
    if not is_running:
        click.echo("TensorBoard is not running")
        return
    
    try:
        stopped = manager.stop_server(force=force)
        if stopped:
            click.echo(f"✅ TensorBoard stopped (PID: {pid})")
            
            # Clean up temp log directory
            temp_logdir = Path('.tensorboard_logs')
            if temp_logdir.exists():
                for symlink in temp_logdir.glob('*'):
                    if symlink.is_symlink():
                        symlink.unlink()
                if not any(temp_logdir.iterdir()):  # Directory is empty
                    temp_logdir.rmdir()
        else:
            click.echo("Failed to stop TensorBoard")
    except click.ClickException:
        raise


@tensorboard.command()
@click.pass_context
def status(ctx):
    """Show TensorBoard server status."""
    config = ctx.obj['config']
    manager = TensorBoardManager(config)
    
    status_info = manager.get_status()
    
    if status_info['running']:
        click.echo("📊 TensorBoard Status")
        click.echo("=" * 40)
        click.echo(f"Status:      🟢 Running")
        click.echo(f"PID:         {status_info['pid']}")
        click.echo(f"Port:        {status_info['port'] or 'Unknown'}")
        click.echo(f"Log Dir:     {status_info['logdir'] or 'Unknown'}")
        
        if status_info['uptime']:
            uptime_hours = status_info['uptime'] / 3600
            click.echo(f"Uptime:      {uptime_hours:.1f} hours")
            
        if status_info['memory_usage']:
            click.echo(f"Memory:      {status_info['memory_usage']:.1f} MB")
            
        if status_info['port']:
            click.echo(f"URL:         http://localhost:{status_info['port']}")
    else:
        click.echo("📊 TensorBoard Status")
        click.echo("=" * 40)
        click.echo("Status:      🔴 Not running")


@tensorboard.command()
@click.argument('experiment_ids', nargs=-1, type=int, required=True)
@click.option('--port', '-p', default=DEFAULT_TENSORBOARD_PORT, type=int,
              help=f'Port to run TensorBoard on (default: {DEFAULT_TENSORBOARD_PORT})')
@click.option('--bind-all', is_flag=True,
              help='Bind to all interfaces')
@click.pass_context
def compare(ctx, experiment_ids: List[int], port: int, bind_all: bool):
    """Start TensorBoard to compare specific experiments."""
    if not experiment_ids:
        raise click.ClickException("At least one experiment ID is required")
    
    config = ctx.obj['config']
    
    # Validate experiment IDs
    try:
        tracker = get_experiment_tracker()
        # Simple validation - check if experiments exist
        for exp_id in experiment_ids:
            exp = tracker.get_experiment(exp_id)
            if not exp:
                raise click.ClickException(f"Experiment {exp_id} not found")
    except Exception as e:
        handle_error(e, verbose=config.get('verbose', False), context="validating experiment IDs")
        raise click.ClickException("One or more experiment IDs are invalid")
    
    # Start TensorBoard with specific experiments
    ctx.invoke(start, port=port, bind_all=bind_all, experiments=experiment_ids,
               all_experiments=False, recent=None)


@tensorboard.command()
@click.argument('experiment_id', type=int)
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['csv', 'json']), default='csv',
              help='Output format')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.option('--metrics', '-m', multiple=True,
              help='Specific metrics to export')
@click.pass_context
def export(ctx, experiment_id: int, output_format: str, output: Optional[str], 
           metrics: List[str]):
    """Export TensorBoard data for external analysis."""
    config = ctx.obj['config']
    
    # Get experiment log directory
    logdir = get_experiment_logdir(experiment_id)
    if not logdir or not logdir.exists():
        raise click.ClickException(f"No log directory found for experiment {experiment_id}")
    
    # For now, we'll export the experiment metadata and basic metrics
    # In a full implementation, you'd parse TensorBoard event files
    try:
        tracker = get_experiment_tracker()
        experiment = tracker.get_experiment(experiment_id)
        
        if not experiment:
            raise click.ClickException(f"Experiment {experiment_id} not found in database")
        
        # Prepare data for export
        export_data = {
            'experiment_id': experiment_id,
            'name': experiment.get('name', ''),
            'description': experiment.get('description', ''),
            'status': experiment.get('status', ''),
            'created_at': experiment.get('created_at', ''),
            'updated_at': experiment.get('updated_at', ''),
            'hyperparameters': experiment.get('hyperparameters', {}),
            'tags': experiment.get('tags', [])
        }
        
        # Determine output file
        if not output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = f'experiment_{experiment_id}_{timestamp}.{output_format}'
        
        # Export data
        if output_format == 'json':
            with open(output, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:  # csv
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['key', 'value'])
                for key, value in export_data.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    writer.writerow([key, value])
        
        click.echo(f"✅ Experiment data exported to: {output}")
        
    except Exception as e:
        raise click.ClickException(f"Failed to export experiment data: {e}")


@tensorboard.command()
@click.option('--clean-old', is_flag=True,
              help='Remove log files older than 30 days')
@click.option('--dry-run', is_flag=True,
              help='Show what would be cleaned without actually deleting')
@click.pass_context
def clean(ctx, clean_old: bool, dry_run: bool):
    """Clean up TensorBoard log files and temporary data."""
    cleaned_count = 0
    cleaned_size = 0
    
    # Clean up temporary TensorBoard directory
    temp_logdir = Path('.tensorboard_logs')
    if temp_logdir.exists():
        if not dry_run:
            for symlink in temp_logdir.glob('*'):
                if symlink.is_symlink():
                    symlink.unlink()
                    cleaned_count += 1
            if not any(temp_logdir.iterdir()):
                temp_logdir.rmdir()
        else:
            for symlink in temp_logdir.glob('*'):
                if symlink.is_symlink():
                    cleaned_count += 1
        
        click.echo(f"{'Would clean' if dry_run else 'Cleaned'} {cleaned_count} temporary symlinks")
    
    if clean_old:
        # Clean old log files (implementation would scan logs directory)
        logs_dir = Path('logs')
        if logs_dir.exists():
            cutoff_time = time.time() - (30 * 24 * 60 * 60)  # 30 days ago
            
            for log_file in logs_dir.rglob('*.tfevents.*'):
                if log_file.stat().st_mtime < cutoff_time:
                    file_size = log_file.stat().st_size
                    if not dry_run:
                        log_file.unlink()
                    cleaned_count += 1
                    cleaned_size += file_size
            
            if cleaned_count > 0:
                size_mb = cleaned_size / 1024 / 1024
                click.echo(f"{'Would clean' if dry_run else 'Cleaned'} {cleaned_count} old log files ({size_mb:.1f} MB)")
            else:
                click.echo("No old log files found to clean")
    
    if cleaned_count == 0:
        click.echo("Nothing to clean")


@tensorboard.command()
@click.pass_context
def config(ctx):
    """Show TensorBoard configuration and validate installation."""
    config = ctx.obj['config']
    
    click.echo("🔧 TensorBoard Configuration")
    click.echo("=" * 40)
    
    # Check TensorBoard installation
    try:
        result = subprocess.run(['tensorboard', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            click.echo(f"TensorBoard:     ✅ {version}")
        else:
            click.echo(f"TensorBoard:     ❌ Error: {result.stderr.strip()}")
    except FileNotFoundError:
        click.echo("TensorBoard:     ❌ Not installed")
        click.echo("                 Install with: pip install tensorboard")
    
    # Show log directory
    logs_dir = Path('logs')
    if logs_dir.exists():
        log_count = len(list(logs_dir.glob('experiment_*')))
        click.echo(f"Log Directory:   ✅ {logs_dir.absolute()} ({log_count} experiments)")
    else:
        click.echo(f"Log Directory:   ⚠️  {logs_dir.absolute()} (not found)")
    
    # Show database
    db_path = config.get('database_path', 'experiments.db')
    if db_path and Path(db_path).exists():
        click.echo(f"Database:        ✅ {db_path}")
    elif db_path:
        click.echo(f"Database:        ⚠️  {db_path} (not found)")
    else:
        click.echo(f"Database:        ⚠️  Not configured")
    
    # Check if TensorBoard is running
    manager = TensorBoardManager(config)
    is_running, pid = manager.is_running()
    if is_running:
        click.echo(f"Server Status:   🟢 Running (PID: {pid})")
    else:
        click.echo(f"Server Status:   🔴 Not running")


# Register all commands with the main CLI
def register_commands(cli):
    """Register TensorBoard commands with the main CLI."""
    cli.add_command(tensorboard)