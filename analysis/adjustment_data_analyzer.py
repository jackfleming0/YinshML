import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import argparse


class AdjustmentDataAnalyzer:
    def __init__(self, metrics_dir: Path, checkpoints_dir: Path):
        # Get project root (parent of analysis directory)
        project_root = Path(__file__).parent.parent

        # Build absolute paths
        self.metrics_dir = project_root / metrics_dir
        self.checkpoints_dir = project_root / checkpoints_dir

        print(f"Project root: {project_root}")
        print(f"Full metrics path: {self.metrics_dir}")

        if not self.metrics_dir.exists():
            raise ValueError(f"Metrics directory not found: {self.metrics_dir}")

    def analyze_value_head_performance(self) -> Dict:
        metrics = self._load_metrics()
        analysis = {
            'phase_accuracy': defaultdict(list),
            'phase_confidence': defaultdict(list),
            'value_flips': defaultdict(list),
            'confidence_accuracy_correlation': defaultdict(list),
            'phase_performance': defaultdict(lambda: {'confidence': [], 'accuracy': [], 'calibration_error': []})
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                # Use phases directly from the data structure
                for phase, values in game['phase_values'].items():
                    if not values:
                        continue

                    flips = sum(1 for i in range(len(values) - 1)
                                if np.sign(values[i]) != np.sign(values[i + 1]))
                    analysis['value_flips'][phase].append(flips)

                    final_value = values[-1]
                    correct_prediction = (final_value > 0) == (game['outcome'] > 0)
                    analysis['phase_accuracy'][phase].append(float(correct_prediction))

                    # Map confidence values to corresponding phase
                    move_offset = 0 if phase == 'placement' else (
                        10 if phase == 'main_game' else len(game['phase_values']['placement']) +
                                                        len(game['phase_values']['main_game']))

                    confidence_values = []
                    for i, value in enumerate(values):
                        move_num = move_offset + i
                        if move_num < len(game['temperature_data']['move_stats']):
                            confidence_values.append(
                                game['temperature_data']['move_stats'][move_num]['top_prob'])

                    # Calculate and store confidence-accuracy correlation for each game/phase
                    if len(confidence_values) >= 2:
                        correlation, _ = pearsonr(confidence_values,
                                                  [float(correct_prediction)] * len(confidence_values))
                        analysis['confidence_accuracy_correlation'][phase].append(correlation)

                    # Store accuracy and confidence values for calibration
                    for i, value in enumerate(values):
                        move_num = move_offset + i
                        if move_num < len(game['temperature_data']['move_stats']):
                            analysis['phase_performance'][phase]['confidence'].append(
                                game['temperature_data']['move_stats'][move_num]['top_prob'])
                            analysis['phase_performance'][phase]['accuracy'].append(float(correct_prediction))
                            analysis['phase_performance'][phase]['calibration_error'].append(
                                abs(float(correct_prediction) - game['temperature_data']['move_stats'][move_num][
                                    'top_prob']))
        return analysis

    def plot_value_head_analysis(self, analysis: Dict):
        """Generate comprehensive value head performance visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Confidence-Accuracy Correlation Over Time
        confidence_accuracy_correlation = analysis.get('confidence_accuracy_correlation', {})
        for phase, correlations in confidence_accuracy_correlation.items():
            axes[0, 0].plot(correlations, label=phase)
        axes[0, 0].set_title('Confidence-Accuracy Correlation')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].legend()

        # Plot 2: Value Prediction Flips
        for phase, flips in analysis.get('value_flips', {}).items():
            axes[0, 1].plot(flips, label=phase)
        axes[0, 1].set_title('Value Prediction Flips per Game')
        axes[0, 1].legend()

        # Plot 3: Calibration Error
        for phase, data in analysis.get('phase_performance', {}).items():
            axes[1, 0].plot(data.get('calibration_error', []), label=phase)
        axes[1, 0].set_title('Calibration Error by Phase')
        axes[1, 0].set_ylabel('|Accuracy - Confidence|')
        axes[1, 0].legend()

        # Plot 4: Accuracy vs Confidence Scatter
        for phase, data in analysis.get('phase_performance', {}).items():
            axes[1, 1].scatter(data.get('confidence', []), data.get('accuracy', []),
                               label=phase, alpha=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Perfect calibration line
        axes[1, 1].set_title('Accuracy vs Confidence')
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def _load_metrics(self) -> List[Dict]:
        """Load and validate metrics files."""
        metrics = []
        for file in sorted(self.metrics_dir.glob("iteration_*.json")):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if 'metrics' in data and 'games' in data['metrics']:
                        metrics.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not metrics:
            print("No valid metrics files found")
            return []

        print(f"Loaded {len(metrics)} metric files")
        return metrics

    def analyze_move_selection_patterns(self) -> Dict:
        metrics = self._load_metrics()
        analysis = {
            'move_temps': defaultdict(list),
            'move_entropies': defaultdict(list),
            'move_times': defaultdict(list),
            'branching_factors': defaultdict(list),
            'move_selection': defaultdict(lambda: defaultdict(int)),
            'timing': defaultdict(list)
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                for move_stat in game['temperature_data']['move_stats']:
                    phase = self._determine_phase(move_stat['move_number'])

                    analysis['move_temps'][phase].append(move_stat['temperature'])
                    analysis['move_entropies'][phase].append(move_stat['entropy'])
                    analysis['move_times'][phase].append(move_stat['move_time'])

                    # Check if valid_moves exists, default branching factor to 0 if not
                    if 'valid_moves' in move_stat:
                        analysis['branching_factors'][phase].append(len(move_stat['valid_moves']))
                    else:
                        analysis['branching_factors'][phase].append(0)

                    analysis['timing'][phase].append(move_stat['move_time'])

                    move_type = move_stat.get('move_type', 'Unknown')
                    analysis['move_selection'][phase][move_type] += 1

        return analysis

    def _calculate_move_selection_summary(self, analysis: Dict) -> Dict:
        """Calculate summary statistics for move selection patterns."""
        summary = {}

        for phase in analysis['branching_factors']:
            bf_data = analysis['branching_factors'][phase]
            temp_data = analysis['move_temps'][phase]
            time_data = analysis['timing'][phase]

            summary[phase] = {
                'avg_branching_factor': int(np.mean(bf_data)),
                'max_branching_factor': int(np.max(bf_data)),
                'avg_temperature': float(np.mean(temp_data)),
                'avg_move_time': float(np.mean(time_data)),
                'move_type_distribution': dict(analysis['move_selection'][phase])
            }

        return summary

    def plot_move_selection_analysis(self, analysis: Dict, summary: Dict):
        """Visualize move selection patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Branching Factor Distribution by Phase
        for phase in analysis['branching_factors']:
            sns.kdeplot(data=analysis['branching_factors'][phase],
                        label=phase, ax=axes[0, 0])
        axes[0, 0].set_title('Branching Factor Distribution')
        axes[0, 0].set_xlabel('Number of Valid Moves')
        axes[0, 0].legend()

        # Plot 2: Temperature Evolution
        for phase in analysis['move_temps']:
            axes[0, 1].plot(analysis['move_temps'][phase],
                            label=phase, alpha=0.7)
        axes[0, 1].set_title('Temperature Evolution')
        axes[0, 1].set_xlabel('Move Number')
        axes[0, 1].legend()

        # Plot 3: Move Type Distribution
        phase_data = []
        for phase, moves in analysis['move_selection'].items():
            for move_type, count in moves.items():
                phase_data.append({
                    'Phase': phase,
                    'Move Type': move_type,
                    'Count': count
                })
        df = pd.DataFrame(phase_data)
        if not df.empty:
            sns.barplot(data=df, x='Phase', y='Count', hue='Move Type', ax=axes[1, 0])
            axes[1, 0].set_title('Move Type Distribution by Phase')

        # Plot 4: Move Timing Analysis
        for phase in analysis['timing']:
            sns.kdeplot(data=analysis['timing'][phase],
                        label=phase, ax=axes[1, 1])
        axes[1, 1].set_title('Move Time Distribution')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def get_move_selection_recommendations(self, analysis: Dict, summary: Dict) -> List[str]:
        """Generate recommendations based on move selection patterns."""
        recommendations = []

        # Analyze branching factor patterns
        for phase, stats in summary.items():
            bf = stats['avg_branching_factor']
            if bf > 50:
                recommendations.append(
                    f"High branching factor in {phase} ({bf:.1f}). "
                    f"Consider increasing temperature or exploration parameters."
                )

        # Check for move type imbalances
        for phase, stats in summary.items():
            move_dist = stats['move_type_distribution']
            total_moves = sum(move_dist.values())
            for move_type, count in move_dist.items():
                if count / total_moves > 0.8:
                    recommendations.append(
                        f"Possible overuse of {move_type} in {phase} ({count / total_moves:.1%}). "
                        f"Consider adjusting exploration parameters."
                    )

        # Temperature recommendations
        for phase, stats in summary.items():
            if stats['avg_temperature'] < 0.3:
                recommendations.append(
                    f"Low average temperature in {phase} ({stats['avg_temperature']:.2f}). "
                    f"Consider increasing exploration."
                )

        return recommendations

    def analyze_game_structure(self) -> Dict:
        metrics = self._load_metrics()

        analysis = {
            'game_lengths': [],
            'win_conditions': {'markers': 0, 'rings': 0},
            'color_balance': {'white_wins': 0, 'black_wins': 0},
            'ring_removal_timing': []
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                # Game length
                analysis['game_lengths'].append(game['length'])

                # Win condition (inferring from final phase values)
                if 'ring_removal' in game['phase_values']:
                    analysis['win_conditions']['rings'] += 1

                    # Calculate timing of ring removals
                    if game['phase_values'].get('ring_removal') is not None:
                        analysis['ring_removal_timing'].append(
                            len(game['phase_values']['placement']) + len(game['phase_values']['main_game']))
                else:
                    analysis['win_conditions']['markers'] += 1

                # Color balance from outcome
                if game['outcome'] == 1:
                    analysis['color_balance']['white_wins'] += 1
                elif game['outcome'] == -1:
                    analysis['color_balance']['black_wins'] += 1

        return analysis, self._calculate_game_structure_summary(analysis)

    def _calculate_game_structure_summary(self, analysis: Dict) -> Dict:
        """Calculate summary statistics with error handling."""
        total_games = len(analysis.get('game_lengths', []))
        print(f"Found {total_games} games to analyze")

        if total_games == 0:
            return {'avg_game_length': 0, 'std_game_length': 0}

        try:
            summary = {
                'avg_game_length': float(np.mean(analysis['game_lengths'])),
                'std_game_length': float(np.std(analysis['game_lengths'])),
                'win_condition_distribution': {
                    k: v / total_games for k, v in analysis['win_conditions'].items()
                },
                'color_balance': {
                    'white_win_rate': analysis['color_balance']['white_wins'] / total_games,
                    'black_win_rate': analysis['color_balance']['black_wins'] / total_games
                }
            }

            # Optional statistics
            if analysis.get('ring_removal_timing'):
                summary['ring_removal_timing'] = {
                    'mean': float(np.mean(analysis['ring_removal_timing'])),
                    'std': float(np.std(analysis['ring_removal_timing']))
                }

            return summary
        except Exception as e:
            print(f"Error calculating summary: {e}")
            return {'avg_game_length': 0, 'std_game_length': 0}

    def plot_game_structure_analysis(self, analysis: Dict, summary: Dict):
        """Visualize game structure patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Game Length Distribution
        sns.histplot(data=analysis['game_lengths'], ax=axes[0, 0])
        axes[0, 0].axvline(summary['avg_game_length'], color='r', linestyle='--',
                           label=f'Mean: {summary["avg_game_length"]:.1f}')
        axes[0, 0].set_title('Game Length Distribution')
        axes[0, 0].legend()

        # Plot 2: Win Conditions
        win_cond_data = pd.DataFrame([
            {'condition': k, 'percentage': v * 100}
            for k, v in summary['win_condition_distribution'].items()
        ])
        sns.barplot(data=win_cond_data, x='condition', y='percentage', ax=axes[0, 1])
        axes[0, 1].set_title('Win Condition Distribution')
        axes[0, 1].set_ylabel('Percentage of Games')

        # Plot 3: Color Balance Over Time
        color_balance = analysis['color_balance']
        total_wins = color_balance['white_wins'] + color_balance['black_wins']

        if total_wins > 0:
            white_win_ratio = color_balance['white_wins'] / total_wins
            axes[1, 0].axhline(white_win_ratio, color='b', label=f"White Win Ratio: {white_win_ratio:.2f}")
            axes[1, 0].axhline(0.5, color='r', linestyle='--', label='Perfect Balance')
            axes[1, 0].set_title('Color Balance')
            axes[1, 0].set_ylabel('White Win Rate')
            axes[1, 0].legend()
        else:
            axes[1, 0].set_title('Color Balance (No Wins Yet)')

        # Plot 4: Ring Removal Timing
        if 'ring_removal_timing' in analysis and summary.get('ring_removal_timing'):
            sns.histplot(data=analysis['ring_removal_timing'], ax=axes[1, 1])
            axes[1, 1].axvline(summary['ring_removal_timing']['mean'], color='r',
                               linestyle='--', label='Mean Timing')
            axes[1, 1].set_title('Ring Removal Timing Distribution')
            axes[1, 1].set_xlabel('Move Number')
            axes[1, 1].legend()
        else:
            axes[1, 1].set_title('Ring Removal Timing Distribution (No Data)')

        plt.tight_layout()
        return fig

    def _determine_phase(self, move_number: int) -> str:
        """Determine phase of the game based on move number."""
        if move_number <= 10:
            return 'placement'
        elif move_number <= 100:  # assuming max of 90 main game moves
            return 'main_game'
        else:
            return 'ring_removal'


def main():
    parser = argparse.ArgumentParser(description='Analyze YINSH training data for adjustments')
    parser.add_argument('--metrics_dir', type=Path, required=True,
                        help='Path to metrics directory')
    parser.add_argument('--checkpoints_dir', type=Path, required=True,
                        help='Path to checkpoints directory')
    parser.add_argument('--output_dir', type=Path, default=Path('analysis_output'),
                        help='Directory to save analysis results')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = AdjustmentDataAnalyzer(args.metrics_dir, args.checkpoints_dir)

    # Run all analyses
    print("\nAnalyzing Value Head Performance...")
    value_analysis = analyzer.analyze_value_head_performance()
    value_fig = analyzer.plot_value_head_analysis(value_analysis)
    value_fig.savefig(args.output_dir / 'value_head_analysis.png')

    print("\nAnalyzing Move Selection Patterns...")
    move_analysis = analyzer.analyze_move_selection_patterns()
    move_summary = analyzer._calculate_move_selection_summary(move_analysis)
    move_fig = analyzer.plot_move_selection_analysis(move_analysis, move_summary)
    move_fig.savefig(args.output_dir / 'move_selection_analysis.png')

    print("\nAnalyzing Game Structure...")
    game_analysis, game_summary = analyzer.analyze_game_structure()
    game_fig = analyzer.plot_game_structure_analysis(game_analysis, game_summary)
    game_fig.savefig(args.output_dir / 'game_structure_analysis.png')

    # Generate and save recommendations
    recommendations = analyzer.get_move_selection_recommendations(move_analysis, move_summary)

    # Save all summaries and recommendations
    with open(args.output_dir / 'analysis_summary.txt', 'w') as f:
        f.write("=== YINSH Training Analysis Summary ===\n\n")

        f.write("Value Head Performance:\n")
        f.write(json.dumps(value_analysis, indent=2))
        f.write("\n\n")

        f.write("Move Selection Patterns:\n")
        f.write(json.dumps(move_summary, indent=2))
        f.write("\n\n")

        f.write("Game Structure:\n")
        f.write(json.dumps(game_summary, indent=2))
        f.write("\n\n")

        f.write("Recommendations:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()