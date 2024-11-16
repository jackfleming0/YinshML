"""Metrics tracking and analysis for YINSH training experiments."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from experiments.config import RESULTS_SUBDIRS


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    policy_losses: List[float]
    value_losses: List[float]
    elo_changes: List[float]
    game_lengths: List[float]
    timestamps: List[float]
    move_entropies: Optional[List[float]] = None
    win_rates: Optional[List[float]] = None
    search_times: Optional[List[float]] = None


class MetricsTracker:
    def __init__(self):
        self.logger = logging.getLogger("MetricsTracker")

        # Cache for loaded metrics
        self._metrics_cache: Dict[str, Dict[str, ExperimentMetrics]] = {}

    def load_experiment_results(self, experiment_type: str) -> Dict[str, ExperimentMetrics]:
        """Load all results for a given experiment type."""
        if experiment_type in self._metrics_cache:
            return self._metrics_cache[experiment_type]

        results_dir = RESULTS_SUBDIRS[experiment_type]
        metrics_by_config = {}

        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)

                config_name = result_file.stem
                metrics = ExperimentMetrics(**data["metrics"])
                metrics_by_config[config_name] = metrics

            except Exception as e:
                self.logger.error(f"Error loading {result_file}: {e}")
                continue

        self._metrics_cache[experiment_type] = metrics_by_config
        return metrics_by_config

    def analyze_learning_dynamics(self, experiment_type: str) -> pd.DataFrame:
        """Analyze learning dynamics across configurations."""
        metrics_by_config = self.load_experiment_results(experiment_type)

        analysis = []
        for config_name, metrics in metrics_by_config.items():
            # Calculate key metrics
            policy_trend = self._calculate_trend(metrics.policy_losses)
            value_trend = self._calculate_trend(metrics.value_losses)
            elo_final = metrics.elo_changes[-1] if metrics.elo_changes else 0

            # Calculate stability metrics
            policy_stability = self._calculate_stability(metrics.policy_losses)
            value_stability = self._calculate_stability(metrics.value_losses)

            # Calculate convergence metrics
            policy_converged = self._check_convergence(metrics.policy_losses)
            value_converged = self._check_convergence(metrics.value_losses)

            analysis.append({
                'config': config_name,
                'policy_loss_trend': policy_trend,
                'value_loss_trend': value_trend,
                'final_elo': elo_final,
                'policy_stability': policy_stability,
                'value_stability': value_stability,
                'policy_converged': policy_converged,
                'value_converged': value_converged,
                'training_time': sum(metrics.timestamps)
            })

        return pd.DataFrame(analysis)

    def compare_configurations(self, experiment_type: str) -> Tuple[str, Dict]:
        """Compare configurations and identify the best one."""
        analysis_df = self.analyze_learning_dynamics(experiment_type)

        # Calculate scores
        scores = {}
        for config in analysis_df['config']:
            row = analysis_df[analysis_df['config'] == config].iloc[0]
            score = (
                    -1.0 * row['policy_loss_trend']  # Lower is better
                    - 1.0 * row['value_loss_trend']  # Lower is better
                    + 2.0 * row['final_elo']  # Higher is better
                    + 1.0 * row['policy_stability']  # Higher is better
                    + 1.0 * row['value_stability']  # Higher is better
                    - 0.5 * row['training_time']  # Lower is better
            )
            scores[config] = score

        # Get best configuration
        best_config = max(scores.items(), key=lambda x: x[1])[0]

        # Calculate relative improvement if baseline exists
        baseline_score = scores.get('baseline', None)
        relative_improvement = {}
        if baseline_score is not None:
            relative_improvement = {
                config: (score - baseline_score) / abs(baseline_score)
                for config, score in scores.items()
                if config != 'baseline' and baseline_score != 0
            }
        else:
            # If no baseline, compare to mean score
            mean_score = np.mean(list(scores.values()))
            relative_improvement = {
                config: (score - mean_score) / abs(mean_score)
                for config, score in scores.items()
                if mean_score != 0
            }

        comparison = {
            'scores': scores,
            'relative_improvement': relative_improvement,
            'training_times': {
                config: float(analysis_df[analysis_df['config'] == config]['training_time'].iloc[0])
                for config in scores.keys()
                if not analysis_df[analysis_df['config'] == config]['training_time'].empty
            }
        }

        return best_config, comparison

    def generate_report(self, experiment_type: str, output_dir: Path) -> None:
        """Generate comprehensive analysis report."""
        metrics_by_config = self.load_experiment_results(experiment_type)
        analysis_df = self.analyze_learning_dynamics(experiment_type)
        best_config, comparison = self.compare_configurations(experiment_type)

        # Create report directory
        report_dir = output_dir / experiment_type
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        self._plot_learning_curves(metrics_by_config, report_dir)
        self._plot_elo_progression(metrics_by_config, report_dir)
        if experiment_type == "temperature":
            self._plot_entropy_analysis(metrics_by_config, report_dir)

        # Write analysis report
        report = {
            'summary': {
                'best_configuration': best_config,
                'relative_improvements': comparison['relative_improvement'],
                'training_times': comparison['training_times']
            },
            'detailed_analysis': analysis_df.to_dict(orient='records'),
            'statistical_tests': self._run_statistical_tests(metrics_by_config)
        }

        with open(report_dir / 'analysis.json', 'w') as f:
            json.dump(report, f, indent=2)

    def get_recommendations(self, experiment_type: str) -> Dict:
        """Get actionable recommendations based on analysis."""
        try:
            analysis_df = self.analyze_learning_dynamics(experiment_type)
            if analysis_df.empty:
                return {
                    'best_configuration': None,
                    'suggestions': ['No experiment data available']
                }

            best_config, comparison = self.compare_configurations(experiment_type)

            recommendations = {
                'best_configuration': best_config,
                'improvement_potential': max(comparison['relative_improvement'].values())
                if comparison['relative_improvement'] else 0.0,
                'suggestions': []
            }

            # Analyze based on experiment type
            if experiment_type == "learning_rate":
                policy_trends = analysis_df['policy_loss_trend']
                elo_stability = analysis_df['final_elo']
                policy_stability = analysis_df['policy_stability']

                # Check for problematic increasing trends
                if any(trend > 0 for trend in policy_trends):
                    recommendations['suggestions'].append(
                        "Some configurations show increasing losses - consider lowering their learning rates"
                    )

                # Add recommendations based on policy stability
                low_stability_configs = analysis_df[analysis_df['policy_stability'] < 0.8]['config'].tolist()
                if low_stability_configs:
                    recommendations['suggestions'].append(
                        f"Configurations {', '.join(low_stability_configs)} show training instability - "
                        "consider adding warmup or reducing learning rate"
                    )

                # Add recommendations based on ELO performance
                best_elo_config = analysis_df.loc[analysis_df['final_elo'].idxmax(), 'config']
                if best_elo_config != best_config:
                    recommendations['suggestions'].append(
                        f"Consider hybrid of {best_config} (best overall) and {best_elo_config} "
                        "(best ELO) configurations"
                    )

                # Add convergence recommendations
                non_converged = analysis_df[~analysis_df['policy_converged']]['config'].tolist()
                if non_converged:
                    recommendations['suggestions'].append(
                        f"Configurations {', '.join(non_converged)} haven't converged - "
                        "consider longer training"
                    )

            elif experiment_type == "mcts":
                # Add MCTS-specific recommendations
                best_elo = analysis_df['final_elo'].max()
                if best_elo < 50:  # Arbitrary threshold
                    recommendations['suggestions'].append(
                        "MCTS improvements show limited ELO gains - consider adjusting c_puct or "
                        "number of simulations"
                    )

            elif experiment_type == "temperature":
                # Add temperature-specific recommendations
                if 'move_entropies' in analysis_df.columns:
                    low_entropy_configs = analysis_df[
                        analysis_df['move_entropies'].apply(lambda x: np.mean(x) < 0.5)
                    ]['config'].tolist()
                    if low_entropy_configs:
                        recommendations['suggestions'].append(
                            f"Configurations {', '.join(low_entropy_configs)} show low move diversity - "
                            "consider higher temperature or slower annealing"
                        )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {
                'best_configuration': None,
                'suggestions': ['Error analyzing experiment data']
            }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in metrics using linear regression."""
        if not values:
            return 0.0
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        return slope * r_value ** 2  # Weighted by fit quality

    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score based on moving average."""
        if len(values) < 3:
            return 0.0

        window_size = min(5, len(values))
        moving_avg = pd.Series(values).rolling(window=window_size).mean()
        moving_std = pd.Series(values).rolling(window=window_size).std()

        # Return average ratio of std to mean (coefficient of variation)
        return 1.0 - np.mean(moving_std[window_size:] / moving_avg[window_size:])

    def _check_convergence(self, values: List[float],
                           window: int = 5, threshold: float = 0.01) -> bool:
        """Check if metrics have converged."""
        if len(values) < window * 2:
            return False

        recent_values = values[-window:]
        recent_std = np.std(recent_values)
        recent_mean = np.mean(recent_values)

        return recent_std / (recent_mean + 1e-8) < threshold

    def _run_statistical_tests(self, metrics_by_config: Dict) -> Dict:
        """Run statistical tests comparing configurations."""
        baseline_metrics = metrics_by_config.get('baseline')
        if not baseline_metrics:
            return {}

        tests = {}
        for config, metrics in metrics_by_config.items():
            if config == 'baseline':
                continue

            # Compare ELO progression
            elo_ttest = stats.ttest_ind(
                baseline_metrics.elo_changes,
                metrics.elo_changes
            )

            # Compare policy loss trends
            policy_ttest = stats.ttest_ind(
                baseline_metrics.policy_losses,
                metrics.policy_losses
            )

            tests[config] = {
                'elo_p_value': float(elo_ttest.pvalue),
                'policy_p_value': float(policy_ttest.pvalue)
            }

        return tests

    def _plot_learning_curves(self, metrics_by_config: Dict, output_dir: Path) -> None:
        """Plot learning curves for all configurations."""
        plt.figure(figsize=(12, 6))

        for config, metrics in metrics_by_config.items():
            plt.plot(metrics.policy_losses, label=f"{config} - Policy")
            plt.plot(metrics.value_losses, label=f"{config} - Value", linestyle='--')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Learning Curves by Configuration')
        plt.legend()
        plt.grid(True)

        plt.savefig(output_dir / 'learning_curves.png')
        plt.close()

    def _plot_elo_progression(self, metrics_by_config: Dict, output_dir: Path) -> None:
        """Plot ELO progression for all configurations."""
        plt.figure(figsize=(12, 6))

        for config, metrics in metrics_by_config.items():
            plt.plot(metrics.elo_changes, label=config)

        plt.xlabel('Iteration')
        plt.ylabel('ELO Change')
        plt.title('ELO Progression by Configuration')
        plt.legend()
        plt.grid(True)

        plt.savefig(output_dir / 'elo_progression.png')
        plt.close()

    def _plot_entropy_analysis(self, metrics_by_config: Dict, output_dir: Path) -> None:
        """Plot move entropy analysis for temperature experiments."""
        plt.figure(figsize=(12, 6))

        for config, metrics in metrics_by_config.items():
            if metrics.move_entropies:
                plt.plot(metrics.move_entropies, label=config)

        plt.xlabel('Iteration')
        plt.ylabel('Move Entropy')
        plt.title('Move Diversity by Configuration')
        plt.legend()
        plt.grid(True)

        plt.savefig(output_dir / 'entropy_analysis.png')
        plt.close()


def main():
    """CLI for metrics analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze YINSH training experiments')
    parser.add_argument('--type', required=True,
                        choices=['learning_rate', 'mcts', 'temperature'],
                        help='Type of experiment to analyze')
    parser.add_argument('--output', type=Path, default=Path('analysis'),
                        help='Output directory for analysis')

    args = parser.parse_args()

    tracker = MetricsTracker()
    tracker.generate_report(args.type, args.output)
    recommendations = tracker.get_recommendations(args.type)

    print("\nRecommendations:")
    print(f"Best configuration: {recommendations['best_configuration']}")
    print(f"Potential improvement: {recommendations['improvement_potential']:.1%}")
    for suggestion in recommendations['suggestions']:
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()