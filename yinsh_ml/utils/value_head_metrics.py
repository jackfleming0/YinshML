import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict
import matplotlib.pyplot as plt
from ..game.game_state import GameState, GamePhase
from ..game.constants import Player
from ..game.moves import Move

class ValueHeadMetrics:
    """Enhanced monitoring for value head analysis."""

    def __init__(self):
        self.phase_metrics = {
            'placement': defaultdict(list),
            'main_game': defaultdict(list),
            'ring_removal': defaultdict(list)
        }
        self.position_cache = {}  # For tracking repeated positions
        self.confidence_curves = []  # Track confidence over move number
        self.critical_positions = []  # Store important position evaluations

    def record_evaluation(self, state: GameState, value_pred: float,
                          policy_probs: np.ndarray, chosen_move: Optional[Move],
                          temperature: float, actual_outcome: Optional[float] = None):
        """Record comprehensive evaluation data."""
        phase = str(state.phase)
        board_str = str(state.board)  # Use str(state.board) directly

        # Track value prediction consistency
        if board_str in self.position_cache:
            prev_value = self.position_cache[board_str]['value']
            self.phase_metrics[phase]['value_consistency'].append(
                abs(prev_value - value_pred)
            )

        self.position_cache[board_str] = {
            'value': value_pred,
            'move_count': len(state.move_history)  # Assuming you want to track this
        }

        # Track value influence on move selection
        top_policy_value = np.max(policy_probs)
        value_influence = 1.0 - temperature  # Higher at lower temperatures

        metrics = {
            'value_pred': value_pred,
            'temperature': temperature,
            'top_policy_prob': top_policy_value,
            'value_influence': value_influence,
            'move_count': len(state.move_history)
        }

        # Add phase-specific metrics
        if phase == 'main_game':
            metrics.update(self._analyze_main_game(state))
        elif phase == 'ring_removal':
            metrics.update(self._analyze_ring_removal(state))

        # Store metrics
        for key, value in metrics.items():
            self.phase_metrics[phase][key].append(value)

        # Track critical positions
        if self._is_critical_position(state, value_pred, actual_outcome):
            self.critical_positions.append({
                'board_state': str(state.board),
                'value_pred': value_pred,
                'actual_outcome': actual_outcome,
                'phase': phase,
                'move_count': len(state.move_history)
            })

    def _analyze_main_game(self, state: GameState) -> Dict:
        """Analyze main game phase specifics."""
        return {
            'ring_mobility': self._calculate_ring_mobility(state),
            'marker_chains': self._count_marker_chains(state),
            'territory_control': self._analyze_territory(state)
        }

    def _analyze_ring_removal(self, state: GameState) -> Dict:
        """Analyze ring removal phase specifics."""
        return {
            'rings_remaining': {
                'white': self._count_rings(state, Player.WHITE),
                'black': self._count_rings(state, Player.BLACK)
            },
            'position_tension': self._calculate_position_tension(state)
        }

    def _is_critical_position(self, state: GameState,
                              value_pred: float,
                              actual_outcome: Optional[float]) -> bool:
        """Identify critical positions for analysis."""
        if actual_outcome is None:
            return False

        # Large prediction error
        if abs(value_pred - actual_outcome) > 0.5:
            return True

        # Near end of game
        if len(state.move_history) > 40:
            return True

        # During ring removal
        if state.phase == GamePhase.RING_REMOVAL:
            return True

        return False

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            'phase_summaries': {},
            'value_evolution': self._analyze_value_evolution(),
            'critical_positions': self._analyze_critical_positions(),
            'consistency_metrics': self._analyze_consistency()
        }

        for phase, metrics in self.phase_metrics.items():
            report['phase_summaries'][phase] = {
                'avg_value_pred': np.mean(metrics['value_pred']),
                'value_std': np.std(metrics['value_pred']),
                'avg_influence': np.mean(metrics['value_influence']),
                'consistency': np.mean(metrics.get('value_consistency', [1.0]))
            }

        return report

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """
        Generate diagnostic plots for value head evaluations.

        Args:
            save_path: Optional path to save the plots
        """
        if not self.evaluations:
            print("No evaluation data available to plot.")
            return

        # Extract relevant data
        phases = [eval['phase'] for eval in self.evaluations]
        value_predictions = np.array([eval['value_prediction'] for eval in self.evaluations])
        actual_outcomes = np.array([eval['actual_outcome'] for eval in self.evaluations])

        # Plot 1: Histogram of Value Predictions
        plt.figure(figsize=(10, 5))
        plt.hist(value_predictions, bins=20, alpha=0.7, label='Predictions')
        plt.title('Distribution of Value Predictions')
        plt.xlabel('Predicted Value')
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(f"{save_path}_value_predictions_hist.png")
        plt.show()

        # Plot 2: Scatter Plot of Predictions vs. Actual Outcomes
        plt.figure(figsize=(10, 5))
        plt.scatter(value_predictions, actual_outcomes, alpha=0.5)
        plt.title('Value Predictions vs. Actual Outcomes')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Outcome')
        plt.plot([-1, 1], [-1, 1], color='red', linestyle='--')  # Diagonal line
        if save_path:
            plt.savefig(f"{save_path}_predictions_vs_outcomes.png")
        plt.show()

        # Plot 3: Value Prediction Distribution by Game Phase
        unique_phases = sorted(set(phases))
        plt.figure(figsize=(10, 5))
        for phase in unique_phases:
            phase_values = [v for p, v in zip(phases, value_predictions) if p == phase]
            plt.hist(phase_values, bins=20, alpha=0.7, label=phase)
        plt.title('Value Prediction Distribution by Game Phase')
        plt.xlabel('Predicted Value')
        plt.ylabel('Frequency')
        plt.legend()
        if save_path:
            plt.savefig(f"{save_path}_value_predictions_by_phase.png")
        plt.show()

    def _calculate_average_value_prediction(self) -> float:
        """Calculate the average value prediction."""
        if not self.evaluations:
            return 0.0
        return float(np.mean([eval['value_prediction'] for eval in self.evaluations]))

    def _calculate_std_value_prediction(self) -> float:
        """Calculate the standard deviation of value predictions."""
        if not self.evaluations:
            return 0.0
        return float(np.std([eval['value_prediction'] for eval in self.evaluations]))

    def _calculate_prediction_confidence(self) -> float:
        """Calculate the average confidence of value predictions."""
        if not self.evaluations:
            return 0.0
        return float(np.mean([abs(eval['value_prediction']) for eval in self.evaluations]))

    def _get_phase_distribution(self) -> Dict[str, int]:
        """Get the distribution of game phases in evaluations."""
        distribution = {}
        for eval in self.evaluations:
            phase = eval['phase']
            distribution[phase] = distribution.get(phase, 0) + 1
        return distribution

    def _calculate_sign_match_percentage(self) -> float:
        """Calculate the percentage of evaluations where prediction sign matches outcome sign."""
        if not self.evaluations:
            return 0.0
        correct_sign_matches = sum(
            1 for eval in self.evaluations
            if np.sign(eval['value_prediction']) == np.sign(eval['actual_outcome'])
        )
        return (correct_sign_matches / len(self.evaluations)) * 100

    def save(self, file_path: str):
        """Save the evaluation data to a file."""
        with open(file_path, 'w') as f:
            json.dump(self.evaluations, f, indent=2)

    def load(self, file_path: str):
        """Load evaluation data from a file."""
        with open(file_path, 'r') as f:
            self.evaluations = json.load(f)