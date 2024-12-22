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
import yaml


class AdjustmentDataAnalyzer:
    def __init__(self, metrics_dir: Path, checkpoints_dir: Path):
        # Get project root (parent of analysis directory)
        project_root = Path(__file__).parent.parent

        # Build absolute paths using project root
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
                for phase, values in game['phase_values'].items():
                    if not values:
                        continue

                    flips = sum(1 for i in range(len(values) - 1)
                                if np.sign(values[i]) != np.sign(values[i + 1]))
                    analysis['value_flips'][phase].append(flips)

                    final_value = values[-1]
                    correct_prediction = (final_value > 0) == (game['outcome'] > 0)
                    analysis['phase_accuracy'][phase].append(float(correct_prediction))

                    move_offset = 0 if phase == 'placement' else (
                        10 if phase == 'main_game' else len(game['phase_values']['placement']) + len(
                            game['phase_values']['main_game']))

                    confidence_values = []
                    for i, value in enumerate(values):
                        move_num = move_offset + i
                        if move_num < len(game['temperature_data']['move_stats']):
                            confidence_values.append(
                                game['temperature_data']['move_stats'][move_num]['top_prob'])

                    if len(confidence_values) >= 2:
                        correlation, _ = pearsonr(confidence_values,
                                                  [float(correct_prediction)] * len(confidence_values))
                        analysis['confidence_accuracy_correlation'][phase].append(correlation)

                    for i, value in enumerate(values):
                        move_num = move_offset + i
                        if move_num < len(game['temperature_data']['move_stats']):
                            analysis['phase_performance'][phase]['confidence'].append(
                                game['temperature_data']['move_stats'][move_num]['top_prob'])
                            analysis['phase_performance'][phase]['accuracy'].append(float(correct_prediction))
                            analysis['phase_performance'][phase]['calibration_error'].append(
                                abs(float(correct_prediction) -
                                    game['temperature_data']['move_stats'][move_num]['top_prob']))
        return analysis

    def plot_value_head_analysis(self, analysis: Dict):
        """Generate comprehensive value head performance visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        confidence_accuracy_correlation = analysis.get('confidence_accuracy_correlation', {})
        for phase, correlations in confidence_accuracy_correlation.items():
            if correlations:
                axes[0, 0].plot(correlations, label=phase)
        axes[0, 0].set_title('Confidence-Accuracy Correlation')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].legend()

        global_min_flip = float('inf')
        global_max_flip = float('-inf')
        for phase, flips in analysis.get('value_flips', {}).items():
            if flips:
                global_min_flip = min(global_min_flip, min(flips))
                global_max_flip = max(global_max_flip, max(flips))

        for phase, flips in analysis.get('value_flips', {}).items():
            if flips:
                axes[0, 1].plot(flips, label=phase)
                axes[0, 1].set_xlim([global_min_flip, global_max_flip])  # Set x-axis limits (ADD THIS LINE)
        axes[0, 1].set_title('Value Prediction Flips per Game')
        axes[0, 1].legend()

        for phase, data in analysis.get('phase_performance', {}).items():
            if data.get('calibration_error'):
                axes[1, 0].plot(data.get('calibration_error', []), label=phase)
        axes[1, 0].set_title('Calibration Error by Phase')
        axes[1, 0].set_ylabel('|Accuracy - Confidence|')
        axes[1, 0].legend()

        for phase, data in analysis.get('phase_performance', {}).items():
            if data.get('confidence') and data.get('accuracy'):
                axes[1, 1].scatter(data.get('confidence', []), data.get('accuracy', []),
                                   label=phase, alpha=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
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
                'avg_branching_factor': int(np.mean(bf_data)) if bf_data else 0,
                'max_branching_factor': int(np.max(bf_data)) if bf_data else 0,
                'avg_temperature': float(np.mean(temp_data)) if temp_data else 0,
                'avg_move_time': float(np.mean(time_data)) if time_data else 0,
                'move_type_distribution': dict(analysis['move_selection'][phase])
            }

        return summary

    def plot_move_selection_analysis(self, analysis: Dict, summary: Dict):
        """Visualize move selection patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        for phase in analysis['branching_factors']:
            if analysis['branching_factors'][phase] and np.var(analysis['branching_factors'][phase]) > 0:
                sns.kdeplot(data=analysis['branching_factors'][phase],
                            label=phase, ax=axes[0, 0])
        axes[0, 0].set_title('Branching Factor Distribution')
        axes[0, 0].set_xlabel('Number of Valid Moves')
        if axes[0, 0].has_data():
            axes[0, 0].legend()

        for phase in analysis['move_temps']:
            if analysis['move_temps'][phase] and len(set(analysis['move_temps'][phase])) > 1:
                axes[0, 1].plot(analysis['move_temps'][phase],
                                label=phase, alpha=0.7)
        axes[0, 1].set_title('Temperature Evolution')
        axes[0, 1].set_xlabel('Move Number')
        axes[0, 1].legend()

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

        for phase in analysis['timing']:
            if analysis['timing'][phase]:
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

        for phase, stats in summary.items():
            bf = stats['avg_branching_factor']
            if bf > 50:
                recommendations.append(
                    f"High branching factor in {phase} ({bf:.1f}). "
                    f"Consider increasing temperature or exploration parameters."
                )

        for phase, stats in summary.items():
            move_dist = stats['move_type_distribution']
            total_moves = sum(move_dist.values())
            for move_type, count in move_dist.items():
                if total_moves > 0 and count / total_moves > 0.8:
                    recommendations.append(
                        f"Possible overuse of {move_type} in {phase} ({count / total_moves:.1%}). "
                        f"Consider adjusting exploration parameters."
                    )

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
            'win_conditions': {'markers': 0, 'rings': 0},  # This is where I made the mistake
            'color_balance': {'white_wins': 0, 'black_wins': 0},
            'ring_removal_timing': [],
            'scores': []  # Track scores when rows of 5 are formed
        }

        for iteration_data in metrics:
            for game in iteration_data['metrics']['games']:
                analysis['game_lengths'].append(game['length'])

                # Win condition (rings removed)
                if 'ring_removal' in game['phase_values']:
                    analysis['win_conditions']['rings'] += 1

                    # Timing of first ring removal
                    if game['phase_values'].get('ring_removal') is not None:
                        analysis['ring_removal_timing'].append(
                            len(game['phase_values']['placement']) + len(game['phase_values']['main_game']))

                # Color balance (who removed the third ring)
                if game['outcome'] == 1:
                    analysis['color_balance']['white_wins'] += 1
                elif game['outcome'] == -1:
                    analysis['color_balance']['black_wins'] += 1

                # Track scores
                if 'score' in game:
                    analysis['scores'].append(game['score'])

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
                    'white_win_rate': analysis['color_balance']['white_wins'] / total_games if analysis['color_balance'][
                        'white_wins'] else 0,
                    'black_win_rate': analysis['color_balance']['black_wins'] / total_games if analysis['color_balance'][
                        'black_wins'] else 0
                },
                'avg_score': float(np.mean(analysis['scores'])) if analysis['scores'] else 0,  # Add average score
                'std_score': float(np.std(analysis['scores'])) if analysis['scores'] else 0  # Add score standard deviation
            }

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
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))  # Adjusted for an additional row

        # Plot 1: Game Length Distribution
        if analysis['game_lengths']:
            sns.histplot(data=analysis['game_lengths'], ax=axes[0, 0])
            axes[0, 0].axvline(summary['avg_game_length'], color='r', linestyle='--',
                               label=f'Mean: {summary["avg_game_length"]:.1f}')
            axes[0, 0].set_title('Game Length Distribution')
            axes[0, 0].legend()

        # Plot 2: Win Conditions (Rings Removed)
        win_cond_data = pd.DataFrame([
            {'condition': k, 'percentage': v * 100}  # Only rings can be a win condition
            for k, v in summary['win_condition_distribution'].items()
        ])
        if not win_cond_data.empty:
            sns.barplot(data=win_cond_data, x='condition', y='percentage', ax=axes[0, 1])
            axes[0, 1].set_title('Win Condition Distribution')
            axes[0, 1].set_ylabel('Percentage of Games')
            axes[0, 1].set_ylim([0, 100])  # Set y-axis limits from 0 to 100

            # Add the percentage text above the bars
            for p in axes[0, 1].patches:
                axes[0, 1].annotate(format(p.get_height(), '.1f'),
                                    (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='center',
                                    xytext=(0, 9),
                                    textcoords='offset points')

        # Plot 3: Color Balance
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

        # Plot 5: Score Distribution (NEW)
        if analysis['scores']:
            sns.histplot(data=analysis['scores'], ax=axes[2, 0])
            axes[2, 0].axvline(summary['avg_score'], color='r', linestyle='--',
                               label=f'Mean Score: {summary["avg_score"]:.1f}')
            axes[2, 0].set_title('Score Distribution')
            axes[2, 0].set_xlabel('Score')
            axes[2, 0].legend()

        # Remove the empty subplot (axes[2, 1])
        fig.delaxes(axes[2][1])

        plt.tight_layout()
        return fig

    def _determine_phase(self, move_number: int) -> str:
        """Determine phase of the game based on move number."""
        if move_number <= 10:
            return 'placement'
        elif move_number <= 100:
            return 'main_game'
        else:
            return 'ring_removal'

    def get_analysis_summary(self) -> Dict:
        """
        Returns a dictionary of key metrics for model ranking and comparison.
        """
        value_summary = self.analyze_value_head_performance()
        move_summary = self._calculate_move_selection_summary(self.analyze_move_selection_patterns())
        _, game_summary = self.analyze_game_structure()

        summary = {
            'avg_game_length': game_summary.get('avg_game_length', 0),
            'std_game_length': game_summary.get('std_game_length', 0),
            'white_win_rate': game_summary.get('color_balance', {}).get('white_win_rate', 0),
            'rings_win_rate': game_summary.get('win_condition_distribution', {}).get('rings', 0),
            # Now only tracks "rings"
            'avg_confidence_accuracy_correlation': np.mean(
                [np.nanmean(val) for val in value_summary.get('confidence_accuracy_correlation', {}).values() if
                 val]),
            'avg_calibration_error': np.mean(
                [np.nanmean(val) for phase_data in value_summary.get('phase_performance', {}).values() for val in
                 phase_data.get('calibration_error', []) if val]),
            'avg_value_flips': np.mean(
                [np.nanmean(val) for val in value_summary.get('value_flips', {}).values() if val]),
            'avg_placement_branching_factor': move_summary.get('placement', {}).get('avg_branching_factor', 0),
            'avg_main_game_branching_factor': move_summary.get('main_game', {}).get('avg_branching_factor', 0),
            'avg_ring_removal_branching_factor': move_summary.get('ring_removal', {}).get('avg_branching_factor', 0),
            'avg_placement_temperature': move_summary.get('placement', {}).get('avg_temperature', 0),
            'avg_main_game_temperature': move_summary.get('main_game', {}).get('avg_temperature', 0),
            'avg_ring_removal_temperature': move_summary.get('ring_removal', {}).get('avg_temperature', 0),
            'avg_placement_move_time': move_summary.get('placement', {}).get('avg_move_time', 0),
            'avg_main_game_move_time': move_summary.get('main_game', {}).get('avg_move_time', 0),
            'avg_ring_removal_move_time': move_summary.get('ring_removal', {}).get('avg_move_time', 0),
            'avg_score': game_summary.get('avg_score', 0),  # Add average score
            'std_score': game_summary.get('std_score', 0)  # Add score standard deviation
        }
        return summary

    def suggest_hyperparameter_adjustments(self, analysis_summary: Dict) -> List[str]:
        """
        Generates hyperparameter adjustment suggestions based on analysis results.

        Args:
            analysis_summary: A dictionary of key metrics, typically the output of
                              `get_analysis_summary`.

        Returns:
            A list of strings, where each string is a specific hyperparameter adjustment
            recommendation.
        """
        recommendations = []

        if analysis_summary['avg_confidence_accuracy_correlation'] < 0.2:
            recommendations.append(
                "Value head calibration is poor. Consider increasing value head network depth/width, "
                "adding a calibration loss term, or using more diverse training data."
            )

        if analysis_summary['avg_calibration_error'] > 0.3:
            recommendations.append(
                f"High average calibration error ({analysis_summary['avg_calibration_error']:.2f}). "
                "Consider a calibration-specific loss or more rigorous training of the value head."
            )

        if analysis_summary['avg_value_flips'] > 5:
            recommendations.append(
                f"High average value flips ({analysis_summary['avg_value_flips']:.2f}). "
                "Consider increasing training iterations or adjusting the learning rate schedule."
            )

        if analysis_summary['avg_placement_temperature'] < 0.5:
            recommendations.append(
                f"Low average temperature in placement phase ({analysis_summary['avg_placement_temperature']:.2f}). "
                "Consider increasing the initial temperature to encourage more exploration."
            )

        if analysis_summary['avg_main_game_temperature'] < 0.3:
            recommendations.append(
                f"Low average temperature in main game phase ({analysis_summary['avg_main_game_temperature']:.2f}). "
                "Consider a slower temperature annealing schedule."
            )

        if analysis_summary['avg_placement_branching_factor'] < 10:
            recommendations.append(
                f"Low average branching factor in placement phase ({analysis_summary['avg_placement_branching_factor']:.1f}). "
                "Investigate if the model is exploring enough valid moves. Consider increasing the temperature."
            )

        if analysis_summary['avg_main_game_branching_factor'] > 50:
            recommendations.append(
                f"High average branching factor in main game phase ({analysis_summary['avg_main_game_branching_factor']:.1f}). "
                "Consider increasing search depth or using a more selective move evaluation strategy."
            )

        if analysis_summary['avg_main_game_move_time'] > 60:
            recommendations.append(
                f"Long average move time in main game phase ({analysis_summary['avg_main_game_move_time']:.1f}s). "
                "Consider optimizing the move selection algorithm or adjusting search depth."
            )

        if analysis_summary['avg_game_length'] > 120:
            recommendations.append(
                f"Games are longer than average ({analysis_summary['avg_game_length']:.1f} moves). This could be due to excessive exploration or inefficient play. Consider tuning temperature and search depth."
            )

        if analysis_summary['avg_game_length'] < 60:
            recommendations.append(
                f"Games are shorter than average ({analysis_summary['avg_game_length']:.1f} moves). This could indicate premature convergence or a lack of exploration. Consider increasing the initial temperature or training for more iterations."
            )

        if analysis_summary['avg_score'] < 1:  # Add a threshold that makes sense for your game
            recommendations.append(
                f"Low average score ({analysis_summary['avg_score']:.2f}). "
                "Consider adding rewards for forming lines of 3 or 4 to encourage scoring opportunities."
            )

        return recommendations


def rank_models(analyzers: List[AdjustmentDataAnalyzer],
                criteria: Dict[str, Tuple[str, float]]) -> List[Tuple[AdjustmentDataAnalyzer, float]]:
    """
    Ranks models based on specified criteria and assigns a composite score.

    Args:
        analyzers: A list of AdjustmentDataAnalyzer instances, each representing a model.
        criteria: A dictionary where keys are metric names (e.g., 'avg_game_length',
                  'white_win_rate', 'calibration_error'), and values are tuples of
                  (direction, weight).
                  - direction: 'higher' if higher is better, 'lower' if lower is better.
                  - weight: A float representing the importance of the metric in the ranking.

    Returns:
        A list of tuples, where each tuple contains an analyzer and its composite score,
        sorted by score in descending order (best model first).
    """
    model_scores = []
    for analyzer in analyzers:
        score = 0
        summary = analyzer.get_analysis_summary()
        for metric, (direction, weight) in criteria.items():
            value = summary.get(metric, None)
            if value is not None:
                if direction == 'higher':
                    score += value * weight
                elif direction == 'lower':
                    score -= value * weight
        model_scores.append((analyzer, score))

    return sorted(model_scores, key=lambda x: x[1], reverse=True)


def plot_comparison(summaries: List[Dict], metric_names: List[str], output_dir: Path):
    """
    Generates comparison plots for specified metrics across multiple models.

    Args:
        summaries: A list of dictionaries, where each dictionary is the output of
                   `get_analysis_summary` for a model.
        metric_names: A list of metric names to plot.
        output_dir: The directory to save the plots.
    """
    for metric_name in metric_names:
        data = []
        model_names = []
        for i, summary in enumerate(summaries):
            if metric_name in summary:
                model_name = f"Model {i + 1}"
                model_names.append(model_name)
                data.append({'Model': model_name, 'Value': summary[metric_name]})

        if data:
            df = pd.DataFrame(data)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Model', y='Value', data=df)
            plt.title(f'Comparison of {metric_name}')
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f'{metric_name}_comparison.png')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze YINSH training data for adjustments')
    parser.add_argument('--config', type=Path, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--metrics_dirs', type=Path, nargs='+', required=True,
                        help='Paths to metrics directories for each model')
    parser.add_argument('--checkpoints_dirs', type=Path, nargs='+', required=True,
                        help='Paths to checkpoints directories for each model')
    parser.add_argument('--output_dir', type=Path, default=Path('analysis_output'),
                        help='Directory to save analysis results')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzers = []
    for metrics_dir, checkpoints_dir in zip(args.metrics_dirs, args.checkpoints_dirs):
        analyzer = AdjustmentDataAnalyzer(metrics_dir, checkpoints_dir)
        analyzers.append(analyzer)

    all_model_summaries = []
    for i, analyzer in enumerate(analyzers):
        print(f"\nAnalyzing Model {i + 1}...")

        value_analysis = analyzer.analyze_value_head_performance()
        move_analysis = analyzer.analyze_move_selection_patterns()
        move_summary = analyzer._calculate_move_selection_summary(move_analysis)
        game_analysis, game_summary = analyzer.analyze_game_structure()

        analysis_summary = analyzer.get_analysis_summary()
        all_model_summaries.append(analysis_summary)

        value_fig = analyzer.plot_value_head_analysis(value_analysis)
        value_fig.savefig(args.output_dir / f'model_{i + 1}_value_head_analysis.png')

        move_fig = analyzer.plot_move_selection_analysis(move_analysis, move_summary)
        move_fig.savefig(args.output_dir / f'model_{i + 1}_move_selection_analysis.png')

        game_fig = analyzer.plot_game_structure_analysis(game_analysis, game_summary)
        game_fig.savefig(args.output_dir / f'model_{i + 1}_game_structure_analysis.png')

        recommendations = analyzer.suggest_hyperparameter_adjustments(analysis_summary)
        with open(args.output_dir / f'model_{i + 1}_recommendations.txt', 'w') as f:
            f.write("=== Hyperparameter Adjustment Recommendations ===\n")
            for j, rec in enumerate(recommendations, 1):
                f.write(f"{j}. {rec}\n")

        with open(args.output_dir / f'model_{i + 1}_summary.txt', 'w') as f:
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

    ranking_criteria = config['ranking_criteria']
    ranked_models = rank_models(analyzers, ranking_criteria)

    metric_names_to_compare = config['comparison_metrics']
    plot_comparison(all_model_summaries, metric_names_to_compare, args.output_dir)

    print("\nModel Rankings:")
    with open(args.output_dir / 'model_rankings.txt', 'w') as f:
        f.write("=== Model Rankings ===\n")
        for i, (analyzer, score) in enumerate(ranked_models):
            f.write(f"Rank {i + 1}: Model (Score: {score:.2f})\n")
            print(f"Rank {i + 1}: Model (Score: {score:.2f})")

    print(f"\nAnalysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()