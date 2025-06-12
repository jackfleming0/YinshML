from __future__ import annotations

"""Launch command – one-shot orchestrator for running a YinshML training experiment
with optional live monitoring (TensorBoard & running-experiment list).

Usage example:

    yinsh-track launch smoke --device mps

The command will:
1. Ensure PYTHONPATH points to the project root (so imports work regardless of CWD).
2. Set up TensorBoard environment variables automatically.
3. Start TensorBoard (unless --no-tensorboard).
4. Optionally start a background thread that periodically prints the list of running
   experiments (unless --no-monitor).
5. Invoke `experiments/runner.py` with the supplied config/device/debug flags.
6. Cleanly shut down spawned background processes on completion or Ctrl-C.
"""

from pathlib import Path
import os
import sys
import subprocess
import threading
import time
import signal
import click

from ..config import get_config

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOGDIR = _REPO_ROOT / "logs"


def _ensure_pythonpath() -> None:
    """Add the repo root to PYTHONPATH if it is not already present."""
    repo_str = str(_REPO_ROOT)
    current = os.environ.get("PYTHONPATH", "")
    if repo_str not in current.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_str, current]) if current else repo_str


def _setup_tensorboard_environment() -> None:
    """Set up TensorBoard environment variables automatically."""
    config = get_config()
    config.setup_tensorboard_environment()
    
    # Ensure log directory exists
    log_dir = Path(os.environ.get('YINSH_TENSORBOARD_LOG_DIR', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(click.style("🔧 TensorBoard environment configured:", fg="cyan"))
    click.echo(f"   Logging: {os.environ.get('YINSH_TENSORBOARD_LOGGING', 'false')}")
    click.echo(f"   Log Dir: {os.environ.get('YINSH_TENSORBOARD_LOG_DIR', './logs')}")
    click.echo(f"   Port: {os.environ.get('YINSH_TENSORBOARD_PORT', '6006')}")


def _start_tensorboard(logdir: Path) -> subprocess.Popen:
    """Launch TensorBoard serving the given log directory.

    Returns the subprocess.Popen object so the caller can terminate it later.
    """
    # Get port from environment or config
    port = os.environ.get('YINSH_TENSORBOARD_PORT', '6006')
    host = os.environ.get('YINSH_TENSORBOARD_HOST', '0.0.0.0')
    
    proc = subprocess.Popen([
        sys.executable,
        "-m",
        "tensorboard.main",  # avoids PATH issues
        "--logdir",
        str(logdir),
        "--port",
        str(port),
        "--host",
        str(host),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Give TensorBoard a moment to start
    time.sleep(2)
    
    click.echo(click.style(f"📊 TensorBoard started – http://{host}:{port} (PID {proc.pid})", fg="green"))
    
    # Check if TensorBoard started successfully
    if proc.poll() is not None:
        # Process already terminated, there was an error
        stdout, stderr = proc.communicate()
        error_msg = stderr.decode() if stderr else "Unknown error"
        click.echo(click.style(f"❌ TensorBoard failed to start: {error_msg}", fg="red"))
        return None
    
    return proc


def _monitor_experiments(stop_event: threading.Event, interval: int = 30) -> None:
    """Periodically prints the list of running experiments using the YinshML CLI."""
    while not stop_event.is_set():
        try:
            subprocess.run([
                sys.executable,
                "-m",
                "yinsh_ml.cli.main",
                "list",
                "--status",
                "running",
            ])
        except Exception as e:
            click.echo(click.style(f"Experiment monitor error: {e}", fg="red"))
        stop_event.wait(interval)


# ---------------------------------------------------------------------------
# Click command definition
# ---------------------------------------------------------------------------

@click.command(name="launch")
@click.argument("config")
@click.option("--device", default="mps", show_default=True, help="Device to use: cuda | mps | cpu")
@click.option("--debug", is_flag=True, help="Enable debug logging for the training run")
@click.option("--no-tensorboard", is_flag=True, help="Do NOT start TensorBoard automatically")
@click.option("--no-monitor", is_flag=True, help="Do NOT show the live running-experiment list")
@click.option("--tensorboard-port", type=int, help="Override TensorBoard port (default: 6006)")
@click.option("--tensorboard-logdir", help="Override TensorBoard log directory (default: ./logs)")
@click.pass_context
def launch(ctx: click.Context, config: str, device: str, debug: bool, no_tensorboard: bool, 
           no_monitor: bool, tensorboard_port: int, tensorboard_logdir: str):
    """Run a full YinshML training experiment with one command.

    CONFIG is the name of the experiment configuration (e.g. "smoke").
    
    This command automatically:
    • Sets up TensorBoard environment variables for experiment tracking
    • Starts TensorBoard web interface (unless --no-tensorboard)
    • Configures proper Python path for the training
    • Runs the training experiment with the specified configuration
    • Monitors running experiments (unless --no-monitor)
    
    Examples:
        # Basic training run with TensorBoard
        yinsh-track launch smoke --device mps
        
        # Training without TensorBoard interface
        yinsh-track launch smoke --device cuda --no-tensorboard
        
        # Custom TensorBoard port and log directory
        yinsh-track launch smoke --tensorboard-port 8006 --tensorboard-logdir ./my_logs
    """
    _ensure_pythonpath()
    
    # Override TensorBoard settings if provided via command line
    cli_config = get_config()
    if tensorboard_port:
        cli_config.set('tensorboard_port', tensorboard_port)
    if tensorboard_logdir:
        cli_config.set('tensorboard_log_dir', tensorboard_logdir)
    
    # Set up TensorBoard environment automatically
    _setup_tensorboard_environment()
    
    # Use the configured log directory
    logdir = Path(os.environ.get('YINSH_TENSORBOARD_LOG_DIR', './logs'))

    # ---------------------------------------------------------------------
    # 1. Optionally spin up TensorBoard and experiment monitor
    # ---------------------------------------------------------------------
    tb_proc: subprocess.Popen | None = None
    monitor_thread: threading.Thread | None = None
    stop_event = threading.Event()

    try:
        if not no_tensorboard:
            tb_proc = _start_tensorboard(logdir)
            if tb_proc is None:
                click.echo(click.style("⚠️  Continuing without TensorBoard due to startup failure", fg="yellow"))

        if not no_monitor:
            monitor_thread = threading.Thread(target=_monitor_experiments, args=(stop_event,), daemon=True)
            monitor_thread.start()

        # -----------------------------------------------------------------
        # 2. Build training command and execute
        # -----------------------------------------------------------------
        train_cmd = [
            sys.executable,
            str(_REPO_ROOT / "experiments" / "runner.py"),
            "--config",
            config,
            "--device",
            device,
        ]
        if debug:
            train_cmd.append("--debug")

        click.echo(click.style(f"🚀 Starting training: {' '.join(train_cmd)}", fg="cyan"))
        click.echo(click.style("📈 Training metrics will appear in TensorBoard automatically", fg="green"))
        
        exit_code = subprocess.call(train_cmd)
        if exit_code != 0:
            click.echo(click.style(f"Training process exited with status {exit_code}", fg="red"))
        else:
            click.echo(click.style("✅ Training finished successfully", fg="green"))
            if tb_proc and tb_proc.poll() is None:
                port = os.environ.get('YINSH_TENSORBOARD_PORT', '6006')
                host = os.environ.get('YINSH_TENSORBOARD_HOST', '0.0.0.0')
                click.echo(click.style(f"📊 TensorBoard still running at http://{host}:{port}", fg="cyan"))

    finally:
        # -----------------------------------------------------------------
        # 3. Cleanup background helpers
        # -----------------------------------------------------------------
        stop_event.set()
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.join(timeout=1)

        if tb_proc and tb_proc.poll() is None:  # Still running
            click.echo("Shutting down TensorBoard…")
            # Politely ask it to terminate
            try:
                if os.name == "nt":
                    tb_proc.terminate()
                else:
                    tb_proc.send_signal(signal.SIGINT)
                    # Give it a moment before force-killing
                    time.sleep(1)
                    if tb_proc.poll() is None:
                        tb_proc.kill()
            except Exception:
                tb_proc.kill()

# End of launch command module 