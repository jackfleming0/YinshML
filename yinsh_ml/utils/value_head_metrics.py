import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from ..game.game_state import GameState, GamePhase
from ..game.constants import Player, PieceType
from ..game.moves import Move
import json

class ValueHeadMetrics:
    """Enhanced monitoring for value head analysis."""

    def __init__(self):
        self.phase_metrics = {
            'placement': defaultdict(list),
            'main_game': defaultdict(list),
            'ring_removal': defaultdict(list)
        }
        self.position_cache = {}  # For tracking repeated positions and their value predictions
        self.move_values = []  # Track value predictions over the course of each game

    def _get_phase_name(self, phase: GamePhase) -> str:
        """Helper function to get phase name."""
        if phase == GamePhase.RING_PLACEMENT:
            return 'placement'
        elif phase == GamePhase.MAIN_GAME:
            return 'main_game'
        elif phase == GamePhase.ROW_COMPLETION:
            return 'main_game'  # Assuming ROW_COMPLETION is similar to MAIN_GAME
        elif phase == GamePhase.RING_REMOVAL:
            return 'ring_removal'
        else:
            return 'unknown'

    def record_evaluation(self, state: GameState, value_pred: float,
                          policy_probs: np.ndarray, chosen_move: Optional[Move],
                          temperature: float, actual_outcome: Optional[float] = None):
        """Record comprehensive evaluation data."""
        phase_name = self._get_phase_name(state.phase)
        board_hash = str(state.board)

        # Track value prediction consistency
        if board_hash in self.position_cache:
            prev_value = self.position_cache[board_hash]['value']
            self.phase_metrics[phase_name]['value_consistency'].append(
                abs(prev_value - value_pred)
            )

        self.position_cache[board_hash] = {
            'value': value_pred,
            'move_count': len(state.move_history)
        }

        # Record value prediction and move number for evolution tracking
        self.move_values.append((len(state.move_history), value_pred))

        metrics = {
            'value_pred': value_pred,
            'temperature': temperature,
            'top_policy_prob': np.max(policy_probs) if policy_probs is not None else 0,
            'ring_mobility': self._calculate_ring_mobility(state),
            'move_count': len(state.move_history)
        }

        # Store metrics
        for key, value in metrics.items():
            self.phase_metrics[phase_name][key].append(value)

    def _calculate_ring_mobility(self, state: GameState) -> float:
        """Calculate average mobility of rings."""
        total_mobility = 0
        total_rings = 0
        for player in Player:
            # Use get_rings_positions instead of get_pieces_positions
            rings = state.board.get_rings_positions(player)
            total_rings += len(rings)
            for ring_pos in rings:
                total_mobility += len(state.get_ring_valid_moves(ring_pos))  # Use get_ring_valid_moves from GameState
        return total_mobility / (total_rings or 1)

    def _analyze_value_evolution(self) -> List[Tuple[int, float]]:
        """Return the evolution of value predictions over move numbers."""
        return self.move_values

    def _analyze_consistency(self) -> Dict[str, float]:
        """Analyze consistency of value predictions for repeated positions."""
        consistency_metrics = {}
        for phase in self.phase_metrics:
            if 'value_consistency' in self.phase_metrics[phase]:
                consistency_metrics[phase] = np.mean(self.phase_metrics[phase]['value_consistency'])
        return consistency_metrics

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            'phase_summaries': {},
            'value_evolution': self._analyze_value_evolution(),
            'consistency_metrics': self._analyze_consistency()
        }

        for phase, metrics in self.phase_metrics.items():
            report['phase_summaries'][phase] = {
                'avg_value_pred': np.mean(metrics['value_pred']),
                'value_std': np.std(metrics['value_pred']),
                'avg_ring_mobility': np.mean(metrics.get('ring_mobility', [0.0])),
                'consistency': np.mean(metrics.get('value_consistency', [1.0]))
            }

        return report

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """
        Generate diagnostic plots for value head evaluations.

        Args:
            save_path: Optional path to save the plots
        """
        if not self.move_values:
            print("No evaluation data available to plot.")
            return

        # Extract relevant data for plotting
        move_numbers, value_preds = zip(*self.move_values)

        # Plot 1: Histogram of Value Predictions
        plt.figure(figsize=(10, 5))
        plt.hist(value_preds, bins=20, alpha=0.7, label='Predictions')
        plt.title('Distribution of Value Predictions')
        plt.xlabel('Predicted Value')
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(f"{save_path}_value_predictions_hist.png")
        plt.close()  # Close the figure after saving

        # Plot 2: Value Prediction Evolution
        plt.figure(figsize=(10, 5))
        plt.plot(move_numbers, value_preds)
        plt.title('Value Prediction Evolution Over Game')
        plt.xlabel('Move Number')
        plt.ylabel('Predicted Value')
        plt.ylim(-1, 1)  # Assuming value predictions are in the range [-1, 1]
        if save_path:
            plt.savefig(f"{save_path}_value_evolution.png")
        plt.close()  # Close the figure after saving

    def save(self, file_path: str):
        """Save the evaluation data to a file."""

        def convert_to_serializable(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        data_to_save = {
            'phase_metrics': self.phase_metrics,
            'position_cache': self.position_cache,
            'move_values': self.move_values,
        }
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=convert_to_serializable)

    def load(self, file_path: str):
        """Load evaluation data from a file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.phase_metrics = data['phase_metrics']
            self.position_cache = data['position_cache']
            self.move_values = data['move_values']
            self.critical_positions = data['critical_positions']