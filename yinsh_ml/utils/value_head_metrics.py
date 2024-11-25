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
                          policy_probs: np.ndarray, chosen_move: Move,
                          temperature: float, actual_outcome: Optional[float] = None):
        """Record comprehensive evaluation data."""
        phase = str(state.phase)
        board_hash = self._hash_board_state(state)

        # Track value prediction consistency
        if board_hash in self.position_cache:
            prev_value = self.position_cache[board_hash]['value']
            self.phase_metrics[phase]['value_consistency'].append(
                abs(prev_value - value_pred)
            )

        self.position_cache[board_hash] = {
            'value': value_pred,
            'move_count': len(state.move_history)
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
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Value prediction distribution by phase
        self._plot_value_distributions(axes[0, 0])

        # Value influence over game progression
        self._plot_value_influence(axes[0, 1])

        # Consistency metrics
        self._plot_consistency_metrics(axes[1, 0])

        # Phase-specific metrics
        self._plot_phase_metrics(axes[1, 1])

        if save_path:
            plt.savefig(save_path)
        plt.close()