#!/usr/bin/env python3
"""
Experiment Stability Analysis Script

Analyzes training stability metrics for completed experiments.
Run after experiments complete to generate stability report.

Usage:
    python scripts/analyze_experiment_stability.py <log_file_or_experiment_id>
    python scripts/analyze_experiment_stability.py experiments/logs/mcts_depth_batch.log
    python scripts/analyze_experiment_stability.py --all  # Analyze all completed experiments
"""

import argparse
import re
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class StabilityMetrics:
    """Stability metrics for an experiment."""
    experiment_name: str
    iterations: List[int]
    elos: List[int]
    best_elo: int
    best_iter: int
    final_elo: int
    std_dev: float
    regressions: int  # Number of iterations where ELO dropped
    peak_final_diff: int  # Best ELO - Final ELO
    trend_slope: float  # Linear regression slope
    held_peak: bool  # Whether final ELO == best ELO

    @property
    def stability_score(self) -> float:
        """
        Composite stability score (lower = more stable).
        Combines std_dev, regressions, and peak retention.
        """
        # Normalize components
        std_penalty = self.std_dev / 10  # ~10-40 range -> 1-4
        regression_penalty = self.regressions  # 0-9 range
        peak_loss_penalty = self.peak_final_diff / 20  # 0-40 range -> 0-2

        return std_penalty + regression_penalty + peak_loss_penalty

    @property
    def quality_score(self) -> float:
        """
        Combined quality score (higher = better).
        Balances performance with stability.
        """
        # Performance component (normalized around 1500 baseline)
        perf_score = (self.best_elo - 1400) / 100  # 1500 -> 1.0, 1600 -> 2.0

        # Stability bonus (inverted stability_score)
        stability_bonus = max(0, 2 - self.stability_score / 5)

        # Trend bonus
        trend_bonus = 0.5 if self.trend_slope > 0 else 0

        # Peak retention bonus
        retention_bonus = 0.5 if self.held_peak else 0

        return perf_score + stability_bonus + trend_bonus + retention_bonus


def parse_log_file(log_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse a log file and extract iteration ELOs per experiment."""
    experiments = {}
    current_experiment = None

    with open(log_path, 'r') as f:
        for line in f:
            # Detect experiment start
            if 'Starting experiment:' in line or 'Experiment ' in line and 'running' in line:
                match = re.search(r'name[=:][\s]*(\S+)', line)
                if match:
                    current_experiment = match.group(1)
                    experiments[current_experiment] = []

            # Parse iteration completion
            match = re.search(r'Iteration (\d+) complete: ELO=(\d+)', line)
            if match and current_experiment:
                iter_num = int(match.group(1))
                elo = int(match.group(2))
                experiments[current_experiment].append((iter_num, elo))

            # Detect experiment name from config loading
            match = re.search(r"Running experiment: (\S+)", line)
            if match:
                current_experiment = match.group(1)
                if current_experiment not in experiments:
                    experiments[current_experiment] = []

    return experiments


def compute_stability_metrics(name: str, data: List[Tuple[int, int]]) -> StabilityMetrics:
    """Compute stability metrics for an experiment."""
    if not data:
        raise ValueError(f"No data for experiment {name}")

    iterations = [d[0] for d in data]
    elos = [d[1] for d in data]

    best_elo = max(elos)
    best_iter = elos.index(best_elo)
    final_elo = elos[-1]
    std_dev = float(np.std(elos))

    # Count regressions
    regressions = sum(1 for i in range(1, len(elos)) if elos[i] < elos[i-1])

    # Peak to final difference
    peak_final_diff = best_elo - final_elo

    # Trend (linear regression slope)
    x = np.arange(len(elos))
    trend_slope = float(np.polyfit(x, elos, 1)[0])

    # Did it hold peak?
    held_peak = (final_elo == best_elo)

    return StabilityMetrics(
        experiment_name=name,
        iterations=iterations,
        elos=elos,
        best_elo=best_elo,
        best_iter=best_iter,
        final_elo=final_elo,
        std_dev=std_dev,
        regressions=regressions,
        peak_final_diff=peak_final_diff,
        trend_slope=trend_slope,
        held_peak=held_peak
    )


def print_stability_report(metrics_list: List[StabilityMetrics], title: str = "Stability Report"):
    """Print a formatted stability report."""
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)

    # Header
    print(f"{'Experiment':<30} {'Best':>6} {'Final':>6} {'Std':>7} {'Regr':>5} "
          f"{'Peak-Δ':>7} {'Trend':>7} {'Held':>5} {'Quality':>8}")
    print("-" * 100)

    # Sort by quality score
    sorted_metrics = sorted(metrics_list, key=lambda m: m.quality_score, reverse=True)

    for m in sorted_metrics:
        held_str = "✅" if m.held_peak else "❌"
        trend_str = f"+{m.trend_slope:.1f}" if m.trend_slope > 0 else f"{m.trend_slope:.1f}"
        stable_marker = "⭐" if m.std_dev < 25 else "  "

        print(f"{stable_marker}{m.experiment_name:<28} {m.best_elo:>6} {m.final_elo:>6} "
              f"{m.std_dev:>7.1f} {m.regressions:>5} {m.peak_final_diff:>7} "
              f"{trend_str:>7} {held_str:>5} {m.quality_score:>8.2f}")

    print("-" * 100)

    # Summary statistics
    best_quality = sorted_metrics[0]
    most_stable = min(metrics_list, key=lambda m: m.std_dev)
    highest_elo = max(metrics_list, key=lambda m: m.best_elo)

    print(f"\n📊 Summary:")
    print(f"   Best Quality Score: {best_quality.experiment_name} ({best_quality.quality_score:.2f})")
    print(f"   Most Stable:        {most_stable.experiment_name} (std={most_stable.std_dev:.1f})")
    print(f"   Highest ELO:        {highest_elo.experiment_name} ({highest_elo.best_elo})")

    # Recommendations
    print(f"\n💡 Recommendations:")
    if best_quality.experiment_name == highest_elo.experiment_name:
        print(f"   ✅ {best_quality.experiment_name} achieves both best quality and highest ELO")
    else:
        print(f"   ⚖️  Trade-off: {highest_elo.experiment_name} has highest ELO but "
              f"{best_quality.experiment_name} has better overall quality")

    stable_high_performers = [m for m in metrics_list if m.std_dev < 25 and m.best_elo > 1520]
    if stable_high_performers:
        print(f"   🎯 Stable high performers: {', '.join(m.experiment_name for m in stable_high_performers)}")

    print("=" * 100)


def generate_markdown_report(metrics_list: List[StabilityMetrics], output_path: Path):
    """Generate a markdown stability report."""
    sorted_metrics = sorted(metrics_list, key=lambda m: m.quality_score, reverse=True)

    with open(output_path, 'w') as f:
        f.write("# Experiment Stability Analysis\n\n")
        f.write(f"Generated from experiment logs.\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Rank | Experiment | Best ELO | Final ELO | Std Dev | Regressions | Held Peak | Quality Score |\n")
        f.write("|------|------------|----------|-----------|---------|-------------|-----------|---------------|\n")

        for i, m in enumerate(sorted_metrics, 1):
            held = "✅" if m.held_peak else "❌"
            stable = "⭐" if m.std_dev < 25 else ""
            f.write(f"| {i} | {stable}{m.experiment_name} | {m.best_elo} | {m.final_elo} | "
                    f"{m.std_dev:.1f} | {m.regressions} | {held} | {m.quality_score:.2f} |\n")

        f.write("\n## Metrics Explained\n\n")
        f.write("- **Std Dev**: Standard deviation of ELO across iterations (lower = more stable)\n")
        f.write("- **Regressions**: Number of iterations where ELO dropped from previous\n")
        f.write("- **Held Peak**: Whether final ELO equals best ELO (no regression from peak)\n")
        f.write("- **Quality Score**: Composite metric balancing performance and stability\n")
        f.write("\n⭐ = Stable experiment (std < 25)\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment stability")
    parser.add_argument("input", nargs="?", help="Log file path or --all for all experiments")
    parser.add_argument("--all", action="store_true", help="Analyze all completed experiments")
    parser.add_argument("--output", "-o", help="Output markdown file path")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.all or args.input == "--all":
        # Find all log files
        log_dir = project_root / "experiments" / "logs"
        log_files = list(log_dir.glob("*.log"))
        print(f"Found {len(log_files)} log files")
    else:
        log_files = [Path(args.input)]

    all_metrics = []

    for log_file in log_files:
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping")
            continue

        experiments = parse_log_file(log_file)

        for name, data in experiments.items():
            if len(data) >= 5:  # Only analyze experiments with enough data
                try:
                    metrics = compute_stability_metrics(name, data)
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"Warning: Could not compute metrics for {name}: {e}")

    if all_metrics:
        print_stability_report(all_metrics, f"Stability Analysis ({len(all_metrics)} experiments)")

        if args.output:
            output_path = Path(args.output)
            generate_markdown_report(all_metrics, output_path)
            print(f"\nMarkdown report saved to: {output_path}")
    else:
        print("No experiment data found to analyze")


if __name__ == "__main__":
    main()
