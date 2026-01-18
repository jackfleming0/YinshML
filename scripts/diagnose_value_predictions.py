"""
Diagnostic script to analyze value prediction usage in MCTS.

This script tests whether value predictions are actually helping MCTS make better decisions.
It analyzes:
1. Distribution of value predictions across game positions
2. Correlation between value predictions and move selection
3. Impact of different value_weight settings on play strength
4. Whether values effectively discriminate between moves
"""

import argparse
import logging
import numpy as np
import torch
from pathlib import Path
import sys
import json
from collections import defaultdict
from typing import List, Dict, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player
from yinsh_ml.search.mcts import MCTS, MCTSConfig, EvaluationMode
from yinsh_ml.utils.encoding import StateEncoder


class ValueDiagnostics:
    """Collect and analyze value prediction diagnostics."""

    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize diagnostics with a trained model."""
        # Handle 'auto' device selection
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.network = NetworkWrapper(model_path=model_path, device=device)
        self.encoder = StateEncoder()
        self.logger = logging.getLogger("ValueDiagnostics")

        # Collected data
        self.value_predictions = []
        self.game_outcomes = []
        self.position_values = {}  # position_hash -> value

    def collect_value_predictions(self, num_games: int = 50) -> Dict:
        """
        Collect value predictions from self-play games.

        Returns dictionary with statistics about value distributions.
        """
        self.logger.info(f"Collecting value predictions from {num_games} games...")

        # Create MCTS for self-play
        config = MCTSConfig(
            num_simulations=50,  # Lighter budget for diagnostics
            evaluation_mode=EvaluationMode.PURE_NEURAL,
            heuristic_weight=0.0
        )
        mcts = MCTS(self.network, config=config)

        all_values = []
        game_count = 0

        for game_idx in range(num_games):
            state = GameState()
            game_values = []

            move_count = 0
            while not state.is_terminal() and move_count < 200:
                # Get value prediction for current position
                state_tensor = self.encoder.encode_state(state)
                state_tensor = torch.from_numpy(state_tensor).float().unsqueeze(0)
                state_tensor = state_tensor.to(self.network.device)

                with torch.no_grad():
                    _, value = self.network.predict(state_tensor)
                    value = value.item()

                game_values.append(value)
                all_values.append(value)

                # Make move using MCTS
                try:
                    policy = mcts.search(state, move_count)
                    moves = state.get_valid_moves()  # FIXED: use get_valid_moves()
                    if not moves:
                        break

                    # Sample move from policy
                    move_probs = []
                    for move in moves:
                        idx = self.encoder.move_to_index(move)
                        if 0 <= idx < len(policy):
                            move_probs.append(policy[idx])
                        else:
                            move_probs.append(0.0)

                    # Normalize
                    total = sum(move_probs)
                    if total > 0:
                        move_probs = [p / total for p in move_probs]
                        move_idx = np.random.choice(len(moves), p=move_probs)
                        move = moves[move_idx]
                    else:
                        move = moves[0]

                    state.make_move(move)
                    move_count += 1
                except Exception as e:
                    self.logger.warning(f"Error during move selection: {e}")
                    break

            # Record game outcome
            if state.is_terminal():
                winner = state.get_winner()
                if winner == Player.WHITE:
                    outcome = 1.0
                elif winner == Player.BLACK:
                    outcome = -1.0
                else:
                    outcome = 0.0
                self.game_outcomes.append(outcome)

            game_count += 1
            if game_count % 10 == 0:
                self.logger.info(f"Completed {game_count}/{num_games} games")

        self.value_predictions = all_values

        # Compute statistics
        stats = {
            'count': len(all_values),
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'median': float(np.median(all_values)),
            'percentiles': {
                '10': float(np.percentile(all_values, 10)),
                '25': float(np.percentile(all_values, 25)),
                '75': float(np.percentile(all_values, 75)),
                '90': float(np.percentile(all_values, 90)),
            },
            'high_confidence_pct': float(100 * np.mean(np.abs(all_values) > 0.7)),
            'low_confidence_pct': float(100 * np.mean(np.abs(all_values) < 0.3)),
        }

        self.logger.info("Value prediction statistics:")
        self.logger.info(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        self.logger.info(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        self.logger.info(f"  High confidence (|v| > 0.7): {stats['high_confidence_pct']:.1f}%")
        self.logger.info(f"  Low confidence (|v| < 0.3): {stats['low_confidence_pct']:.1f}%")

        return stats

    def analyze_value_discrimination(self, num_positions: int = 100) -> Dict:
        """
        Analyze whether values discriminate between moves at a position.

        For random positions, evaluate all legal moves and check if
        value predictions vary significantly.
        """
        self.logger.info(f"Analyzing value discrimination across {num_positions} positions...")

        discriminations = []

        for pos_idx in range(num_positions):
            # Generate random position
            state = GameState()
            for _ in range(np.random.randint(5, 20)):  # Random number of moves
                moves = state.get_valid_moves()
                if not moves or state.is_terminal():
                    break
                move = moves[np.random.randint(len(moves))]
                state.make_move(move)

            if state.is_terminal():
                continue

            # Evaluate all legal next positions
            moves = state.get_valid_moves()
            if len(moves) < 2:
                continue

            move_values = []
            for move in moves[:10]:  # Limit to first 10 moves for speed
                next_state = state.copy()
                next_state.make_move(move)

                state_tensor = self.encoder.encode_state(next_state)
                state_tensor = torch.from_numpy(state_tensor).float().unsqueeze(0)
                state_tensor = state_tensor.to(self.network.device)

                with torch.no_grad():
                    _, value = self.network.predict(state_tensor)
                    move_values.append(value.item())

            # Measure discrimination: std dev of values
            if len(move_values) >= 2:
                discrimination = float(np.std(move_values))
                discriminations.append(discrimination)

            if (pos_idx + 1) % 20 == 0:
                self.logger.info(f"Analyzed {pos_idx + 1}/{num_positions} positions")

        stats = {
            'count': len(discriminations),
            'mean_std': float(np.mean(discriminations)),
            'median_std': float(np.median(discriminations)),
            'max_std': float(np.max(discriminations)) if discriminations else 0,
            'weak_discrimination_pct': float(100 * np.mean(np.array(discriminations) < 0.05)),
        }

        self.logger.info("Value discrimination statistics:")
        self.logger.info(f"  Mean std dev between moves: {stats['mean_std']:.3f}")
        self.logger.info(f"  Weak discrimination (<0.05 std): {stats['weak_discrimination_pct']:.1f}%")

        return stats

    def test_value_weight_impact(self, num_games: int = 50) -> Dict:
        """
        Test impact of different value_weight settings on play strength.

        Plays games with different value weights and measures outcomes.
        """
        self.logger.info("Testing different value_weight settings...")

        results = {}
        weight_settings = [0.0, 0.5, 1.0, 2.0]

        for weight in weight_settings:
            self.logger.info(f"Testing value_weight={weight}...")

            config = MCTSConfig(
                num_simulations=50,
                evaluation_mode=EvaluationMode.PURE_NEURAL,
                value_weight=weight,
                heuristic_weight=0.0
            )
            mcts = MCTS(self.network, config=config)

            outcomes = []
            game_lengths = []

            for game_idx in range(num_games):
                state = GameState()
                move_count = 0

                while not state.is_terminal() and move_count < 200:
                    try:
                        policy = mcts.search(state, move_count)
                        moves = state.get_valid_moves()
                        if not moves:
                            break

                        # Choose best move
                        move_probs = []
                        for move in moves:
                            idx = self.encoder.move_to_index(move)
                            if 0 <= idx < len(policy):
                                move_probs.append(policy[idx])
                            else:
                                move_probs.append(0.0)

                        best_idx = np.argmax(move_probs)
                        move = moves[best_idx]
                        state.make_move(move)
                        move_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error during move: {e}")
                        break

                game_lengths.append(move_count)

                if state.is_terminal():
                    winner = state.get_winner()
                    if winner == Player.WHITE:
                        outcomes.append(1.0)
                    elif winner == Player.BLACK:
                        outcomes.append(-1.0)
                    else:
                        outcomes.append(0.0)

                if (game_idx + 1) % 10 == 0:
                    self.logger.info(f"  Completed {game_idx + 1}/{num_games} games")

            results[f"weight_{weight}"] = {
                'avg_outcome': float(np.mean(outcomes)) if outcomes else 0,
                'avg_game_length': float(np.mean(game_lengths)),
                'win_rate_white': float(np.mean([o > 0 for o in outcomes])),
                'win_rate_black': float(np.mean([o < 0 for o in outcomes])),
            }

            self.logger.info(f"  Results: avg_outcome={results[f'weight_{weight}']['avg_outcome']:.3f}, "
                           f"avg_length={results[f'weight_{weight}']['avg_game_length']:.1f}")

        return results

    def generate_report(self, output_path: str):
        """Generate comprehensive diagnostic report."""
        self.logger.info(f"Generating diagnostic report to {output_path}...")

        # Collect all diagnostics
        self.logger.info("=" * 80)
        self.logger.info("PRIORITY 1 DIAGNOSTICS: Value Prediction Analysis")
        self.logger.info("=" * 80)

        value_stats = self.collect_value_predictions(num_games=50)
        discrimination_stats = self.analyze_value_discrimination(num_positions=100)
        # Note: value_weight test disabled for speed - takes ~30 min
        # weight_impact = self.test_value_weight_impact(num_games=50)

        report = {
            'value_statistics': value_stats,
            'discrimination_analysis': discrimination_stats,
            # 'value_weight_impact': weight_impact,
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w') as f:
            f.write("# Value Prediction Diagnostics Report\n\n")
            f.write("## Executive Summary\n\n")

            f.write("### Value Prediction Distribution\n\n")
            f.write(f"- **Range**: [{value_stats['min']:.3f}, {value_stats['max']:.3f}]\n")
            f.write(f"- **Mean**: {value_stats['mean']:.3f} ± {value_stats['std']:.3f}\n")
            f.write(f"- **High Confidence** (|v| > 0.7): {value_stats['high_confidence_pct']:.1f}%\n")
            f.write(f"- **Low Confidence** (|v| < 0.3): {value_stats['low_confidence_pct']:.1f}%\n")
            f.write(f"- **Median**: {value_stats['median']:.3f}\n\n")

            f.write("### Value Discrimination Between Moves\n\n")
            f.write(f"- **Mean Std Dev**: {discrimination_stats['mean_std']:.3f}\n")
            f.write(f"- **Weak Discrimination** (<0.05 std): {discrimination_stats['weak_discrimination_pct']:.1f}%\n\n")

            f.write("## Interpretation\n\n")

            # Analyze and provide interpretation
            if value_stats['high_confidence_pct'] < 5.0:
                f.write("⚠️ **CRITICAL**: Very few high-confidence predictions (<5%).\n")
                f.write("The network rarely makes strong predictions, which limits MCTS guidance.\n\n")

            if value_stats['std'] < 0.15:
                f.write("⚠️ **CRITICAL**: Very low standard deviation (<0.15).\n")
                f.write("Predictions are clustered around the mean, providing weak discrimination.\n\n")

            if discrimination_stats['weak_discrimination_pct'] > 50:
                f.write("⚠️ **CRITICAL**: Weak discrimination between moves (>50%).\n")
                f.write("Values don't vary much between different moves at the same position.\n")
                f.write("This means MCTS can't effectively use values to prune bad branches.\n\n")

            f.write("## Recommendations\n\n")

            if value_stats['std'] < 0.15:
                f.write("1. **Modify loss function** to encourage confident predictions\n")
                f.write("   - Add confidence penalty term\n")
                f.write("   - Reward predictions near -1 or +1\n")
                f.write("   - Penalize predictions near 0\n\n")

            if discrimination_stats['weak_discrimination_pct'] > 50:
                f.write("2. **Improve training data quality**\n")
                f.write("   - Bootstrap from stronger baseline (heuristic MCTS)\n")
                f.write("   - Ensure diverse, informative positions\n")
                f.write("   - Check if value targets are too clustered\n\n")

            f.write("3. **Consider architectural changes**\n")
            f.write("   - Increase network capacity (20 blocks instead of 12)\n")
            f.write("   - Add value head normalization layers\n\n")

        self.logger.info(f"Report saved to {output_path} and {md_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description='Diagnose value prediction usage in MCTS')
    parser.add_argument('model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='value_diagnostics.json',
                       help='Output path for diagnostic report')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, mps, cpu, or auto)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run diagnostics
    diagnostics = ValueDiagnostics(args.model_path, device=args.device)
    report = diagnostics.generate_report(args.output)

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print(f"\nReport saved to: {args.output}")
    print(f"Markdown summary: {args.output.replace('.json', '.md')}")
    print("\nKey findings:")
    print(f"  - Value range: [{report['value_statistics']['min']:.3f}, {report['value_statistics']['max']:.3f}]")
    print(f"  - High confidence predictions: {report['value_statistics']['high_confidence_pct']:.1f}%")
    print(f"  - Mean discrimination: {report['discrimination_analysis']['mean_std']:.3f}")
    print(f"  - Weak discrimination: {report['discrimination_analysis']['weak_discrimination_pct']:.1f}%")


if __name__ == '__main__':
    main()
