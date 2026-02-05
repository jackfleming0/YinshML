#!/usr/bin/env python3
"""
Live Training Monitor - Real-time terminal dashboard for YinshML training.

Displays:
- Iteration progress with ETA
- Loss trends (policy and value)
- ELO progression with promotion/rejection stats
- Memory usage and tensor counts
- Alert conditions (loss plateau, ELO regression, etc.)

Usage:
    python scripts/monitor_training_live.py experiments/abc123/
    python scripts/monitor_training_live.py --experiment abc123
    python scripts/monitor_training_live.py --latest
"""

import argparse
import curses
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import signal


@dataclass
class IterationData:
    """Parsed iteration metrics."""
    iteration: int
    policy_loss: float
    value_loss: float
    elo: float
    promoted: bool
    game_time: float
    train_time: float
    buffer_size: int
    memory_mb: float
    tensor_count: int
    timestamp: str


class TrainingMonitor:
    """Live training monitor with curses UI."""

    def __init__(self, experiment_dir: Path, refresh_rate: float = 2.0):
        self.experiment_dir = experiment_dir
        self.refresh_rate = refresh_rate
        self.iterations: List[IterationData] = []
        self.alerts: List[str] = []
        self.config: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self._running = True

        # Load initial data
        self._load_config()
        self._load_iterations()

    def _load_config(self):
        """Load experiment configuration."""
        config_path = self.experiment_dir / "metrics.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    self.config = data.get('hyperparameters', {})
            except (json.JSONDecodeError, KeyError):
                pass

    def _load_iterations(self):
        """Load all iteration data."""
        self.iterations = []

        # Try loading from metrics.json first
        metrics_path = self.experiment_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    for m in data.get('metrics', []):
                        if m.get('name') == 'loss/policy':
                            # Build iteration from aggregated metrics
                            pass
            except (json.JSONDecodeError, KeyError):
                pass

        # Also scan iteration directories
        for iter_dir in sorted(self.experiment_dir.glob("iteration_*")):
            metrics_file = iter_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        iteration = IterationData(
                            iteration=int(iter_dir.name.split('_')[1]),
                            policy_loss=data.get('policy_loss', 0),
                            value_loss=data.get('value_loss', 0),
                            elo=data.get('tournament_rating', 1500),
                            promoted=data.get('promoted', False),
                            game_time=data.get('game_time', 0),
                            train_time=data.get('train_time', 0),
                            buffer_size=data.get('buffer_size', 0),
                            memory_mb=data.get('memory_mb', 0),
                            tensor_count=data.get('tensor_count', 0),
                            timestamp=data.get('timestamp', '')
                        )
                        self.iterations.append(iteration)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

        # Sort by iteration number
        self.iterations.sort(key=lambda x: x.iteration)

        # Set start time from first iteration
        if self.iterations and self.start_time is None:
            try:
                self.start_time = datetime.fromisoformat(self.iterations[0].timestamp)
            except:
                self.start_time = datetime.now()

    def _check_alerts(self):
        """Check for alert conditions."""
        self.alerts = []

        if len(self.iterations) < 2:
            return

        # Check loss plateau (last 3 iterations)
        if len(self.iterations) >= 3:
            recent_policy = [it.policy_loss for it in self.iterations[-3:]]
            recent_value = [it.value_loss for it in self.iterations[-3:]]

            policy_change = abs(recent_policy[-1] - recent_policy[0]) / max(recent_policy[0], 0.001)
            value_change = abs(recent_value[-1] - recent_value[0]) / max(recent_value[0], 0.001)

            if policy_change < 0.01:
                self.alerts.append("Policy loss plateau (<1% change in 3 iterations)")
            if value_change < 0.01:
                self.alerts.append("Value loss plateau (<1% change in 3 iterations)")

        # Check ELO regression
        if len(self.iterations) >= 3:
            recent_elo = [it.elo for it in self.iterations[-3:]]
            if all(recent_elo[i] > recent_elo[i+1] for i in range(len(recent_elo)-1)):
                self.alerts.append("ELO declining for 3+ iterations")

        # Check consecutive rejections
        recent_promoted = [it.promoted for it in self.iterations[-5:]]
        if len(recent_promoted) >= 5 and not any(recent_promoted):
            self.alerts.append("5 consecutive model rejections")

        # Check memory growth
        if len(self.iterations) >= 2:
            memory_growth = self.iterations[-1].memory_mb - self.iterations[0].memory_mb
            if memory_growth > 500:
                self.alerts.append(f"Memory grew by {memory_growth:.0f}MB")

        # Check tensor count stability
        if len(self.iterations) >= 2:
            tensor_diff = self.iterations[-1].tensor_count - self.iterations[-2].tensor_count
            if tensor_diff > 100:
                self.alerts.append(f"Tensor count increased by {tensor_diff}")

    def _draw_ui(self, stdscr):
        """Draw the terminal UI."""
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

        GREEN = curses.color_pair(1)
        YELLOW = curses.color_pair(2)
        RED = curses.color_pair(3)
        CYAN = curses.color_pair(4)
        BOLD = curses.A_BOLD

        row = 0

        # Header
        title = f" YinshML Training Monitor - {self.experiment_dir.name} "
        stdscr.addstr(row, 0, "=" * width, BOLD)
        row += 1
        stdscr.addstr(row, (width - len(title)) // 2, title, BOLD | CYAN)
        row += 1
        stdscr.addstr(row, 0, "=" * width, BOLD)
        row += 2

        # Get config info
        total_iterations = self.config.get('iterations', 10)
        current_iteration = len(self.iterations)

        # Progress bar
        progress = current_iteration / total_iterations if total_iterations > 0 else 0
        bar_width = width - 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        stdscr.addstr(row, 2, f"Iteration: {current_iteration}/{total_iterations}  [{bar}] {progress*100:.0f}%")
        row += 1

        # Elapsed and ETA
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            elapsed_str = str(elapsed).split('.')[0]

            if current_iteration > 0:
                avg_time = elapsed.total_seconds() / current_iteration
                remaining = (total_iterations - current_iteration) * avg_time
                eta_str = str(timedelta(seconds=int(remaining)))
            else:
                eta_str = "calculating..."

            stdscr.addstr(row, 2, f"Elapsed: {elapsed_str} | ETA: {eta_str}")
        row += 2

        # Separator
        stdscr.addstr(row, 0, "-" * width)
        row += 1

        # Current iteration progress (if running)
        current_iter_dir = self.experiment_dir / f"iteration_{current_iteration}"
        if current_iter_dir.exists():
            stdscr.addstr(row, 2, "Current Iteration:", BOLD)
            row += 1
            # Could add more detailed progress here
            stdscr.addstr(row, 4, "In progress...")
            row += 1
        row += 1

        # Loss trends
        if self.iterations:
            stdscr.addstr(row, 2, "Loss (last 5 iterations):", BOLD)
            row += 1

            # Policy loss
            policy_losses = [f"{it.policy_loss:.2f}" for it in self.iterations[-5:]]
            policy_str = " → ".join(policy_losses)
            trend = "↓" if len(self.iterations) >= 2 and self.iterations[-1].policy_loss < self.iterations[-2].policy_loss else "↑"
            color = GREEN if trend == "↓" else RED
            stdscr.addstr(row, 4, f"Policy: {policy_str} {trend}", color)
            row += 1

            # Value loss
            value_losses = [f"{it.value_loss:.2f}" for it in self.iterations[-5:]]
            value_str = " → ".join(value_losses)
            trend = "↓" if len(self.iterations) >= 2 and self.iterations[-1].value_loss < self.iterations[-2].value_loss else "↑"
            color = GREEN if trend == "↓" else RED
            stdscr.addstr(row, 4, f"Value:  {value_str} {trend}", color)
            row += 2

        # ELO progression
        if self.iterations:
            stdscr.addstr(row, 2, "ELO Progression:", BOLD)
            row += 1

            elo_values = [f"{int(it.elo)}" for it in self.iterations[-5:]]
            elo_str = " → ".join(elo_values)
            stdscr.addstr(row, 4, elo_str)
            row += 1

            # Best ELO and promotion stats
            best_elo = max(it.elo for it in self.iterations)
            best_iter = next(it.iteration for it in self.iterations if it.elo == best_elo)
            promoted = sum(1 for it in self.iterations if it.promoted)
            rejected = len(self.iterations) - promoted

            stdscr.addstr(row, 4, f"Best: {int(best_elo)} (iter {best_iter}) | ", CYAN)
            stdscr.addstr(row, 35, f"Promoted: {promoted}", GREEN)
            stdscr.addstr(row, 50, f" | Rejected: {rejected}", RED)
            row += 2

        # Separator
        stdscr.addstr(row, 0, "-" * width)
        row += 1

        # Alerts
        stdscr.addstr(row, 2, "Alerts:", BOLD)
        row += 1

        if self.alerts:
            for alert in self.alerts[:3]:  # Show max 3 alerts
                stdscr.addstr(row, 4, f"⚠️  {alert}", YELLOW)
                row += 1
        else:
            stdscr.addstr(row, 4, "None", GREEN)
            row += 1
        row += 1

        # Memory stats
        if self.iterations:
            latest = self.iterations[-1]
            stdscr.addstr(row, 2, f"Memory: {latest.memory_mb:.1f} MB | ", BOLD)
            stdscr.addstr(row, 25, f"Tensors: {latest.tensor_count}")
            if latest.buffer_size > 0:
                stdscr.addstr(row, 45, f" | Buffer: {latest.buffer_size:,}")
            row += 2

        # Footer
        stdscr.addstr(height - 2, 0, "-" * width)
        stdscr.addstr(height - 1, 2, "Press 'q' to quit | 'r' to refresh | Auto-refresh every {:.0f}s".format(self.refresh_rate))

        stdscr.refresh()

    def _handle_input(self, stdscr) -> bool:
        """Handle keyboard input. Returns False to quit."""
        stdscr.timeout(int(self.refresh_rate * 1000))
        try:
            key = stdscr.getch()
            if key == ord('q'):
                return False
            elif key == ord('r'):
                self._load_iterations()
                self._check_alerts()
        except:
            pass
        return True

    def run(self):
        """Run the monitor."""
        def main(stdscr):
            curses.curs_set(0)  # Hide cursor

            while self._running:
                self._load_iterations()
                self._check_alerts()
                self._draw_ui(stdscr)

                if not self._handle_input(stdscr):
                    break

        try:
            curses.wrapper(main)
        except KeyboardInterrupt:
            pass

    def stop(self):
        """Stop the monitor."""
        self._running = False


class SimpleMonitor:
    """Simple non-curses monitor for environments without curses support."""

    def __init__(self, experiment_dir: Path, refresh_rate: float = 5.0):
        self.experiment_dir = experiment_dir
        self.refresh_rate = refresh_rate
        self._running = True

    def run(self):
        """Run simple text-based monitoring."""
        print(f"\nMonitoring experiment: {self.experiment_dir}")
        print("Press Ctrl+C to stop\n")

        while self._running:
            self._print_status()
            try:
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                break

    def _print_status(self):
        """Print current status."""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 60)
        print(f" YinshML Training Monitor - {self.experiment_dir.name}")
        print("=" * 60)
        print()

        # Find iteration directories
        iter_dirs = sorted(self.experiment_dir.glob("iteration_*"))

        if not iter_dirs:
            print("Waiting for first iteration...")
            return

        print(f"Completed iterations: {len(iter_dirs)}")
        print()

        # Show last few iterations
        print("Recent iterations:")
        print("-" * 40)

        for iter_dir in iter_dirs[-5:]:
            metrics_file = iter_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        iter_num = iter_dir.name.split('_')[1]
                        elo = data.get('tournament_rating', 1500)
                        policy = data.get('policy_loss', 0)
                        value = data.get('value_loss', 0)
                        print(f"  Iter {iter_num}: ELO={elo:.0f}, Policy={policy:.3f}, Value={value:.3f}")
                except:
                    pass

        print()
        print(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def find_latest_experiment(experiments_dir: Path) -> Optional[Path]:
    """Find the most recently modified experiment directory."""
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not experiment_dirs:
        return None
    return max(experiment_dirs, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Live training monitor for YinshML')
    parser.add_argument('experiment_dir', nargs='?', help='Experiment directory to monitor')
    parser.add_argument('--experiment', '-e', help='Experiment ID to monitor')
    parser.add_argument('--latest', '-l', action='store_true', help='Monitor latest experiment')
    parser.add_argument('--refresh', '-r', type=float, default=2.0, help='Refresh rate in seconds')
    parser.add_argument('--simple', '-s', action='store_true', help='Use simple text mode (no curses)')

    args = parser.parse_args()

    # Determine experiment directory
    experiments_base = Path("experiments")

    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    elif args.experiment:
        experiment_dir = experiments_base / args.experiment
    elif args.latest:
        experiment_dir = find_latest_experiment(experiments_base)
        if not experiment_dir:
            print("No experiments found in experiments/")
            sys.exit(1)
    else:
        # Default to latest
        experiment_dir = find_latest_experiment(experiments_base)
        if not experiment_dir:
            print("No experiments found. Specify an experiment directory.")
            parser.print_help()
            sys.exit(1)

    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    print(f"Monitoring: {experiment_dir}")

    # Run monitor
    if args.simple or not sys.stdout.isatty():
        monitor = SimpleMonitor(experiment_dir, args.refresh)
    else:
        try:
            monitor = TrainingMonitor(experiment_dir, args.refresh)
        except:
            # Fall back to simple mode if curses fails
            monitor = SimpleMonitor(experiment_dir, args.refresh)

    # Handle signals
    def signal_handler(sig, frame):
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor.run()


if __name__ == '__main__':
    main()
