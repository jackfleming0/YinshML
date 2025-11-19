#!/usr/bin/env python3
"""Run comprehensive tournament validation for heuristic agent.

This script executes large-scale tournaments comparing the heuristic agent
against random and baseline opponents to validate performance metrics.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.agents.tournament import TournamentEvaluator, TournamentConfig, TournamentMetrics
from yinsh_ml.agents.heuristic_agent import HeuristicAgentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_heuristic_vs_random(num_games: int = 1000, output_dir: Path = None) -> TournamentMetrics:
    """Run tournament: Heuristic Agent vs Random Policy.
    
    Args:
        num_games: Number of games to play
        output_dir: Directory to save results
        
    Returns:
        Tournament metrics
    """
    logger.info(f"Starting heuristic vs random tournament ({num_games} games)")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "heuristic_vs_random.json")
    else:
        output_path = None
    
    # Use faster config for tournament validation
    # Default config (depth 3 + iterative deepening + forced sequences) is too slow
    config = HeuristicAgentConfig(
        max_depth=2,  # Reduced from 3 for speed
        time_limit_seconds=0.5,  # Reduced from 1.0 for speed
        max_branching_factor=20,  # Reduced from 24 for speed
        use_iterative_deepening=False,  # Disable iterative deepening for speed
    )
    evaluator = TournamentEvaluator(
        heuristic_agent_factory=None,  # Use default
        heuristic_config=config,
        opponent_factory=None,  # Use default random
    )
    
    tournament_config = TournamentConfig(
        num_games=num_games,
        concurrent_workers=4,
        save_interval=100,
        output_path=output_path,
        max_turns_per_game=200,
        resume=False,
    )
    
    metrics = evaluator.run_large_scale_tournament(
        config=tournament_config,
        opponent_seed=1234,
    )
    
    logger.info(f"Heuristic vs Random Results:")
    logger.info(f"  Win Rate: {metrics.win_rate:.3f}")
    logger.info(f"  Wins: {metrics.wins}, Losses: {metrics.losses}, Draws: {metrics.draws}")
    logger.info(f"  Average Move Time: {metrics.average_move_time*1000:.3f} ms")
    logger.info(f"  Max Move Time: {metrics.max_move_time*1000:.3f} ms")
    logger.info(f"  Average Game Length: {metrics.average_game_length:.1f} turns")
    
    return metrics


def run_heuristic_vs_baseline(num_games: int = 1000, output_dir: Path = None) -> TournamentMetrics:
    """Run tournament: Heuristic Agent vs Baseline Policy.
    
    For baseline, we use RandomMovePolicy with rule_based_probability=0.0 (pure random)
    as a simple baseline. This is essentially the same as random but allows for
    future extension to more sophisticated baselines.
    
    Args:
        num_games: Number of games to play
        output_dir: Directory to save results
        
    Returns:
        Tournament metrics
    """
    logger.info(f"Starting heuristic vs baseline tournament ({num_games} games)")
    logger.info("Note: Baseline is currently pure random (same as random opponent)")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "heuristic_vs_baseline.json")
    else:
        output_path = None
    
    # Use faster config for tournament validation (same as random tournament)
    config = HeuristicAgentConfig(
        max_depth=2,  # Reduced from 3 for speed
        time_limit_seconds=0.5,  # Reduced from 1.0 for speed
        max_branching_factor=20,  # Reduced from 24 for speed
        use_iterative_deepening=False,  # Disable iterative deepening for speed
    )
    evaluator = TournamentEvaluator(
        heuristic_agent_factory=None,
        heuristic_config=config,
        opponent_factory=None,  # Use default random as baseline
    )
    
    tournament_config = TournamentConfig(
        num_games=num_games,
        concurrent_workers=4,
        save_interval=100,
        output_path=output_path,
        max_turns_per_game=200,
        resume=False,
    )
    
    metrics = evaluator.run_large_scale_tournament(
        config=tournament_config,
        opponent_seed=5678,  # Different seed from random tournament
    )
    
    logger.info(f"Heuristic vs Baseline Results:")
    logger.info(f"  Win Rate: {metrics.win_rate:.3f}")
    logger.info(f"  Wins: {metrics.wins}, Losses: {metrics.losses}, Draws: {metrics.draws}")
    logger.info(f"  Average Move Time: {metrics.average_move_time*1000:.3f} ms")
    logger.info(f"  Max Move Time: {metrics.max_move_time*1000:.3f} ms")
    logger.info(f"  Average Game Length: {metrics.average_game_length:.1f} turns")
    
    return metrics


def main():
    """Main entry point for tournament validation."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tournament validation for heuristic agent"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games per tournament (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="yinsh_ml/docs/validation_results",
        help="Directory to save tournament results (default: yinsh_ml/docs/validation_results)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline tournament (only run vs random)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Final Tournament Validation")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Games per tournament: {args.num_games}")
    logger.info("")
    
    # Run heuristic vs random
    logger.info("Tournament 1: Heuristic Agent vs Random Policy")
    logger.info("-" * 80)
    metrics_random = run_heuristic_vs_random(args.num_games, output_dir)
    logger.info("")
    
    # Run heuristic vs baseline (if not skipped)
    if not args.skip_baseline:
        logger.info("Tournament 2: Heuristic Agent vs Baseline Policy")
        logger.info("-" * 80)
        metrics_baseline = run_heuristic_vs_baseline(args.num_games, output_dir)
        logger.info("")
    else:
        logger.info("Skipping baseline tournament (--skip-baseline flag set)")
        metrics_baseline = None
    
    # Summary
    logger.info("=" * 80)
    logger.info("Tournament Validation Complete")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    logger.info("Next step: Run generate_validation_report.py to analyze results")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

