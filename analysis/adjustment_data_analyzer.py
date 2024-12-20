import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


class AdjustmentDataAnalyzer:
    def __init__(self, metrics_dir: Path, checkpoints_dir: Path):
        self.metrics_dir = metrics_dir if isinstance(metrics_dir, Path) else Path(metrics_dir)
        self.checkpoints_dir = checkpoints_dir if isinstance(checkpoints_dir, Path) else Path(checkpoints_dir)

        print(f"Metrics directory: {self.metrics_dir}")
        print(f"Metrics directory exists: {self.metrics_dir.exists()}")
        if self.metrics_dir.exists():
            print("Contents:", list(self.metrics_dir.glob("*")))

    def analyze_value_head_performance(self) -> Dict:
        metrics = self._load_metrics()
        analysis = {
            'phase_accuracy': defaultdict(list),
            'phase_confidence': defaultdict(list),
            'value_flips': defaultdict(list)
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                for phase, values in game['phase_values'].items():
                    # Track value prediction flips
                    flips = sum(1 for i in range(len(values) - 1)
                                if np.sign(values[i]) != np.sign(values[i + 1]))
                    analysis['value_flips'][phase].append(flips)

                    # Track accuracy (compare with game outcome)
                    final_value = values[-1]
                    correct_prediction = (final_value > 0) == (game['outcome'] > 0)
                    analysis['phase_accuracy'][phase].append(float(correct_prediction))

                # Track confidence from temperature data
                for move_stat in game['temperature_data']['move_stats']:
                    phase = self._determine_phase(move_stat['move_number'])
                    analysis['phase_confidence'][phase].append(move_stat['top_prob'])

        return analysis

    def plot_value_head_analysis(self, analysis: Dict):
        """Generate comprehensive value head performance visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Confidence-Accuracy Correlation Over Time
        for phase, correlations in analysis['confidence_accuracy_correlation'].items():
            axes[0, 0].plot(correlations, label=phase)
        axes[0, 0].set_title('Confidence-Accuracy Correlation')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].legend()

        # Plot 2: Value Prediction Flips
        for phase, flips in analysis['value_flips'].items():
            axes[0, 1].plot(flips, label=phase)
        axes[0, 1].set_title('Value Prediction Flips per Game')
        axes[0, 1].legend()

        # Plot 3: Calibration Error
        for phase, data in analysis['phase_performance'].items():
            axes[1, 0].plot(data['calibration_error'], label=phase)
        axes[1, 0].set_title('Calibration Error by Phase')
        axes[1, 0].set_ylabel('|Accuracy - Confidence|')
        axes[1, 0].legend()

        # Plot 4: Accuracy vs Confidence Scatter
        for phase, data in analysis['phase_performance'].items():
            axes[1, 1].scatter(data['confidence'], data['accuracy'],
                               label=phase, alpha=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Perfect calibration line
        axes[1, 1].set_title('Accuracy vs Confidence')
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

    def _load_metrics(self) -> List[Dict]:
        metrics = []

        # Handle nested metrics directory
        full_metrics_path = self.metrics_dir / "metrics"
        if full_metrics_path.exists():
            search_path = full_metrics_path
        else:
            search_path = self.metrics_dir

        print(f"Searching in: {search_path}")
        print(f"Contents: {list(search_path.glob('*'))}")

        for file in sorted(search_path.glob("iteration_*.json")):
            try:
                with open(file) as f:
                    metrics.append(json.load(f))
            except Exception as e:
                print(f"Error loading {file}: {e}")

        return metrics

        print(f"Loaded {len(metrics)} files")
        return metrics

    def analyze_move_selection_patterns(self) -> Dict:
        metrics = self._load_metrics()
        analysis = {
            'move_temps': defaultdict(list),
            'move_entropies': defaultdict(list),
            'move_times': defaultdict(list)
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                for move_stat in game['temperature_data']['move_stats']:
                    phase = self._determine_phase(move_stat['move_number'])

                    analysis['move_temps'][phase].append(move_stat['temperature'])
                    analysis['move_entropies'][phase].append(move_stat['entropy'])
                    analysis['move_times'][phase].append(move_stat['move_time'])

        return analysis

    def _calculate_move_selection_summary(self, analysis: Dict) -> Dict:
        """Calculate summary statistics for move selection patterns."""
        summary = {}

        for phase in analysis['branching_factors']:
            bf_data = analysis['branching_factors'][phase]
            temp_data = analysis['move_temperatures'][phase]
            time_data = analysis['timing'][phase]

            summary[phase] = {
                'avg_branching_factor': np.mean(bf_data),
                'max_branching_factor': np.max(bf_data),
                'avg_temperature': np.mean(temp_data),
                'avg_move_time': np.mean(time_data),
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
        for phase in analysis['move_temperatures']:
            axes[0, 1].plot(analysis['move_temperatures'][phase],
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
            'color_balance': {'white_wins': 0, 'black_wins': 0}
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                # Game length
                analysis['game_lengths'].append(game['length'])

                # Win condition (inferring from final phase values)
                if game['phase_values'].get('ring_removal'):
                    analysis['win_conditions']['rings'] += 1
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
        color_data = pd.DataFrame(analysis['color_balance'])
        color_data['ratio'] = color_data['white_wins'] / (color_data['white_wins'] + color_data['black_wins'])
        axes[1, 0].plot(color_data['ratio'])
        axes[1, 0].axhline(0.5, color='r', linestyle='--', label='Perfect Balance')
        axes[1, 0].set_title('Color Balance')
        axes[1, 0].set_ylabel('White Win Rate')
        axes[1, 0].legend()

        # Plot 4: Ring Removal Timing
        sns.histplot(data=analysis['ring_removal_timing'], ax=axes[1, 1])
        axes[1, 1].axvline(summary['ring_removal_timing']['mean'], color='r',
                           linestyle='--', label='Mean Timing')
        axes[1, 1].set_title('Ring Removal Timing Distribution')
        axes[1, 1].set_xlabel('Move Number')
        axes[1, 1].legend()

        plt.tight_layout()
        return fig

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
    move_analysis, move_summary = analyzer.analyze_move_selection_patterns()
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