#!/usr/bin/env python3
"""Generate comprehensive validation report from tournament results.

This script analyzes tournament results and generates a markdown report
with performance metrics, statistical analysis, and success criteria verification.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.agents.tournament import TournamentMetrics
from yinsh_ml.agents.statistics import StatisticalAnalyzer, ConfidenceInterval, SignificanceTest
from yinsh_ml.heuristics.benchmark_evaluator import benchmark_evaluator, generate_test_positions
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.game.constants import Player

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tournament_results(file_path: Path) -> Optional[TournamentMetrics]:
    """Load tournament results from JSON file.
    
    Args:
        file_path: Path to JSON results file
        
    Returns:
        TournamentMetrics if file exists and is valid, None otherwise
    """
    if not file_path.exists():
        logger.warning(f"Results file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if "metrics" in data:
            return TournamentMetrics.from_dict(data["metrics"])
        else:
            logger.error(f"Invalid results file format: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load results from {file_path}: {e}")
        return None


def benchmark_heuristic_evaluation() -> Dict[str, float]:
    """Benchmark heuristic evaluation performance.
    
    Returns:
        Dictionary with benchmark results (times in ms)
    """
    evaluator = YinshHeuristics()
    positions = generate_test_positions(1000)
    results = benchmark_evaluator(evaluator, positions, Player.WHITE, warmup_iterations=100)
    return results


def verify_success_criteria(metrics: TournamentMetrics, eval_benchmark: Dict[str, float]) -> Dict[str, Any]:
    """Verify success criteria against tournament metrics.
    
    Args:
        metrics: Tournament metrics to verify
        eval_benchmark: Benchmark results for heuristic evaluation
        
    Returns:
        Dictionary with criteria verification results
    """
    # Note: "evaluation time" refers to heuristic evaluation (evaluate_position),
    # not move selection time (which includes search)
    avg_eval_time_ms = eval_benchmark.get('avg_time_ms', 0.0)
    max_eval_time_ms = eval_benchmark.get('max_time_ms', 0.0)
    
    criteria = {
        "win_rate_60_percent": {
            "requirement": "Win rate >= 60%",
            "threshold": 0.60,
            "actual": metrics.win_rate,
            "passed": metrics.win_rate >= 0.60,
        },
        "eval_time_1ms": {
            "requirement": "Heuristic evaluation time < 1ms",
            "threshold": 1.0,  # 1ms
            "actual": avg_eval_time_ms,
            "passed": avg_eval_time_ms < 1.0,
        },
        "max_eval_time_10ms": {
            "requirement": "Max heuristic evaluation time < 10ms (safety check)",
            "threshold": 10.0,  # 10ms
            "actual": max_eval_time_ms,
            "passed": max_eval_time_ms < 10.0,
        },
    }
    
    return criteria


def generate_report(
    metrics_random: Optional[TournamentMetrics],
    metrics_baseline: Optional[TournamentMetrics],
    output_path: Path,
) -> None:
    """Generate comprehensive validation report.
    
    Args:
        metrics_random: Metrics from heuristic vs random tournament
        metrics_baseline: Metrics from heuristic vs baseline tournament
        output_path: Path to save report
    """
    analyzer = StatisticalAnalyzer()
    
    # Benchmark heuristic evaluation performance
    logger.info("Benchmarking heuristic evaluation performance...")
    eval_benchmark = benchmark_heuristic_evaluation()
    
    lines = []
    lines.append("# Final Validation Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    
    # Overall status
    all_passed = True
    if metrics_random:
        criteria_random = verify_success_criteria(metrics_random, eval_benchmark)
        all_passed = all_passed and all(c["passed"] for c in criteria_random.values())
    
    status_icon = "✅ PASS" if all_passed else "❌ FAIL"
    lines.append(f"**Overall Status:** {status_icon}")
    lines.append("")
    
    # Success criteria summary
    lines.append("### Success Criteria")
    lines.append("")
    lines.append("| Criterion | Status | Requirement | Actual |")
    lines.append("|-----------|--------|-------------|-------|")
    
    if metrics_random:
        criteria_random = verify_success_criteria(metrics_random, eval_benchmark)
        for name, criterion in criteria_random.items():
            status = "✅ PASS" if criterion["passed"] else "❌ FAIL"
            if "time" in name or "eval_time" in name:
                # Evaluation time is already in ms
                actual_str = f"{criterion['actual']:.3f} ms"
                threshold_str = f"{criterion['threshold']:.0f} ms"
            else:
                actual_str = f"{criterion['actual']:.1%}"
                threshold_str = f"{criterion['threshold']:.0%}"
            lines.append(
                f"| {criterion['requirement']} | {status} | {threshold_str} | {actual_str} |"
            )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Tournament Results
    lines.append("## Tournament Results")
    lines.append("")
    
    if metrics_random:
        lines.append("### Heuristic Agent vs Random Policy")
        lines.append("")
        lines.append(f"- **Total Games:** {metrics_random.total_games}")
        lines.append(f"- **Wins:** {metrics_random.wins}")
        lines.append(f"- **Losses:** {metrics_random.losses}")
        lines.append(f"- **Draws:** {metrics_random.draws}")
        lines.append(f"- **Win Rate:** {metrics_random.win_rate:.3f} ({metrics_random.win_rate:.1%})")
        lines.append("")
        
        # Confidence interval
        ci = analyzer.compute_confidence_interval(
            metrics_random.win_rate,
            metrics_random.total_games,
            confidence=0.95,
        )
        lines.append(f"- **95% Confidence Interval:** {ci}")
        lines.append("")
        
        # Performance metrics
        lines.append("#### Performance Metrics")
        lines.append("")
        lines.append(f"- **Average Move Time:** {metrics_random.average_move_time*1000:.3f} ms")
        lines.append(f"- **Max Move Time:** {metrics_random.max_move_time*1000:.3f} ms")
        lines.append(f"- **Average Game Length:** {metrics_random.average_game_length:.1f} turns")
        lines.append(f"- **Std Dev Game Length:** {metrics_random.std_game_length:.1f} turns")
        lines.append(f"- **Nodes Per Second:** {metrics_random.nodes_per_second:.0f}")
        lines.append("")
    
    if metrics_baseline:
        lines.append("### Heuristic Agent vs Baseline Policy")
        lines.append("")
        lines.append(f"- **Total Games:** {metrics_baseline.total_games}")
        lines.append(f"- **Wins:** {metrics_baseline.wins}")
        lines.append(f"- **Losses:** {metrics_baseline.losses}")
        lines.append(f"- **Draws:** {metrics_baseline.draws}")
        lines.append(f"- **Win Rate:** {metrics_baseline.win_rate:.3f} ({metrics_baseline.win_rate:.1%})")
        lines.append("")
        
        # Confidence interval
        ci = analyzer.compute_confidence_interval(
            metrics_baseline.win_rate,
            metrics_baseline.total_games,
            confidence=0.95,
        )
        lines.append(f"- **95% Confidence Interval:** {ci}")
        lines.append("")
        
        # Performance metrics
        lines.append("#### Performance Metrics")
        lines.append("")
        lines.append(f"- **Average Move Time:** {metrics_baseline.average_move_time*1000:.3f} ms")
        lines.append(f"- **Max Move Time:** {metrics_baseline.max_move_time*1000:.3f} ms")
        lines.append(f"- **Average Game Length:** {metrics_baseline.average_game_length:.1f} turns")
        lines.append(f"- **Std Dev Game Length:** {metrics_baseline.std_game_length:.1f} turns")
        lines.append(f"- **Nodes Per Second:** {metrics_baseline.nodes_per_second:.0f}")
        lines.append("")
        
        # Statistical comparison
        if metrics_random:
            lines.append("### Statistical Comparison")
            lines.append("")
            lines.append("Comparing Heuristic vs Random vs Heuristic vs Baseline:")
            lines.append("")
            
            test = analyzer.test_significance(metrics_random, metrics_baseline)
            lines.append(f"- **Test:** {test.test_name}")
            lines.append(f"- **Statistic:** {test.statistic:.4f}")
            lines.append(f"- **p-value:** {test.p_value:.4f}")
            lines.append(f"- **Significant:** {'Yes' if test.significant else 'No'} (α=0.05)")
            lines.append("")
    
    # Detailed Analysis
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Analysis")
    lines.append("")
    
    if metrics_random:
        lines.append("### Win Rate Analysis")
        lines.append("")
        ci = analyzer.compute_confidence_interval(
            metrics_random.win_rate,
            metrics_random.total_games,
            confidence=0.95,
        )
        lines.append(f"The heuristic agent achieved a win rate of **{metrics_random.win_rate:.1%}** "
                     f"against random opponents.")
        lines.append("")
        lines.append(f"With 95% confidence, the true win rate lies between "
                     f"**{ci.lower:.1%}** and **{ci.upper:.1%}**.")
        lines.append("")
        
        if metrics_random.win_rate >= 0.60:
            lines.append("✅ **SUCCESS:** Win rate exceeds the 60% target.")
        else:
            lines.append("❌ **FAILURE:** Win rate does not meet the 60% target.")
        lines.append("")
    
    if metrics_random:
        lines.append("### Performance Analysis")
        lines.append("")
        lines.append("#### Heuristic Evaluation Performance")
        lines.append("")
        lines.append("The heuristic evaluator (`YinshHeuristics.evaluate_position`) performance:")
        lines.append("")
        avg_eval_ms = eval_benchmark.get('avg_time_ms', 0.0)
        max_eval_ms = eval_benchmark.get('max_time_ms', 0.0)
        median_eval_ms = eval_benchmark.get('median_time_ms', 0.0)
        p95_eval_ms = eval_benchmark.get('p95_time_ms', 0.0)
        
        lines.append(f"- **Average evaluation time**: **{avg_eval_ms:.3f} ms**")
        if avg_eval_ms < 1.0:
            lines.append("  ✅ **SUCCESS:** Average heuristic evaluation time is below 1ms target.")
        else:
            lines.append("  ❌ **FAILURE:** Average heuristic evaluation time exceeds 1ms target.")
        lines.append("")
        lines.append(f"- **Maximum evaluation time**: **{max_eval_ms:.3f} ms**")
        if max_eval_ms < 10.0:
            lines.append("  ✅ Maximum evaluation time is within acceptable bounds (<10ms).")
        else:
            lines.append("  ⚠️  Maximum evaluation time exceeds 10ms threshold.")
        lines.append("")
        lines.append(f"- **Median evaluation time**: **{median_eval_ms:.3f} ms**")
        lines.append(f"- **95th percentile**: **{p95_eval_ms:.3f} ms**")
        lines.append(f"- **Throughput**: **{eval_benchmark.get('evaluations_per_second', 0):.0f}** evaluations/second")
        lines.append("")
        
        lines.append("#### Move Selection Performance")
        lines.append("")
        lines.append("Note: Move selection time includes full search (negamax with depth 2), not just evaluation:")
        lines.append("")
        lines.append(f"- **Average move selection time**: **{metrics_random.average_move_time*1000:.3f} ms**")
        lines.append(f"- **Maximum move selection time**: **{metrics_random.max_move_time*1000:.3f} ms**")
        lines.append("")
        lines.append(f"- **Average game length**: **{metrics_random.average_game_length:.1f}** turns "
                     f"(σ={metrics_random.std_game_length:.1f})")
        lines.append("")
        lines.append(f"- **Search throughput**: **{metrics_random.nodes_per_second:.0f}** nodes/second")
        lines.append("")
    
    # Conclusion
    lines.append("---")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    
    if all_passed:
        lines.append("✅ **All success criteria have been met.**")
        lines.append("")
        lines.append("The heuristic agent is ready for integration into the AlphaZero training pipeline.")
    else:
        lines.append("❌ **Some success criteria were not met.**")
        lines.append("")
        lines.append("Please review the criteria above and address any failures before proceeding.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by generate_validation_report.py*")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Validation report saved to: {output_path}")


def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate validation report from tournament results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="yinsh_ml/docs/validation_results",
        help="Directory containing tournament results (default: yinsh_ml/docs/validation_results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="yinsh_ml/docs/validation_results/final_validation_report.md",
        help="Output path for report (default: yinsh_ml/docs/validation_results/final_validation_report.md)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    logger.info("Loading tournament results...")
    
    # Load results
    metrics_random = load_tournament_results(
        results_dir / "heuristic_vs_random.json"
    )
    metrics_baseline = load_tournament_results(
        results_dir / "heuristic_vs_baseline.json"
    )
    
    if not metrics_random and not metrics_baseline:
        logger.error("No tournament results found. Please run run_final_validation.py first.")
        return 1
    
    if not metrics_random:
        logger.warning("Random tournament results not found. Report will be incomplete.")
    
    logger.info("Generating validation report...")
    generate_report(metrics_random, metrics_baseline, output_path)
    
    logger.info("Report generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

