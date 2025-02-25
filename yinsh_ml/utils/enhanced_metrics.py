from pathlib import Path
import json
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class PhaseMetrics:
    """Detailed metrics for each game phase"""
    value_predictions: List[float] = field(default_factory=list)
    actual_outcomes: List[float] = field(default_factory=list)
    move_times: List[float] = field(default_factory=list)
    position_values: Dict[str, float] = field(default_factory=dict)
    decision_confidence: List[float] = field(default_factory=list)


class EnhancedMetricsCollector:
    """Collects detailed training metrics by phase"""

    def __init__(self):
        self.phase_metrics = {
            'placement': PhaseMetrics(),
            'main_game': PhaseMetrics(),
            'ring_removal': PhaseMetrics(),
            'game_over': PhaseMetrics()
        }
        self.critical_positions = defaultdict(list)
        self.current_summary = {}

    def add_state_metrics(self,
                          phase: str,
                          board_state: str,
                          value_pred: float,
                          actual_outcome: float,
                          move_time: float,
                          confidence: float):
        """Record metrics for a single game state"""
        # Add phase mapping
        phase_map = {
            'RING_PLACEMENT': 'placement',
            'MAIN_GAME': 'main_game',
            'RING_REMOVAL': 'ring_removal'
        }

        # Convert phase enum to string and map to our phase keys
        phase_str = str(phase).split('.')[-1]  # Gets "RING_PLACEMENT" from "GamePhase.RING_PLACEMENT"
        phase_key = phase_map.get(phase_str, 'main_game')  # Default to main_game if unknown

        metrics = self.phase_metrics[phase_key]
        metrics.value_predictions.append(float(value_pred))
        metrics.actual_outcomes.append(float(actual_outcome))
        metrics.move_times.append(float(move_time))
        metrics.decision_confidence.append(float(confidence))

        # Track value prediction consistency
        if board_state in metrics.position_values:
            prev_value = metrics.position_values[board_state]
            value_change = abs(prev_value - value_pred)
            if value_change > 0.3:  # Significant change threshold
                self.critical_positions[phase].append({
                    'board_state': board_state,
                    'prev_value': float(prev_value),
                    'new_value': float(value_pred),
                    'change': float(value_change)
                })
        metrics.position_values[board_state] = value_pred

    def get_metrics_summary(self) -> Dict:
        """Generate summary statistics for all collected metrics"""
        summary = {'phases': {}}

        for phase, metrics in self.phase_metrics.items():
            if not metrics.value_predictions:
                continue

            # Compute value head metrics
            value_accuracy = np.mean([
                (pred > 0) == (actual > 0)
                for pred, actual in zip(metrics.value_predictions, metrics.actual_outcomes)
            ])

            value_calibration = np.mean([
                abs(pred - actual)
                for pred, actual in zip(metrics.value_predictions, metrics.actual_outcomes)
            ])

            confidence_correlation = np.corrcoef(
                metrics.decision_confidence,
                np.abs([pred - actual for pred, actual in
                        zip(metrics.value_predictions, metrics.actual_outcomes)])
            )[0, 1]

            summary['phases'][phase] = {
                'value_accuracy': float(value_accuracy),
                'value_calibration': float(value_calibration),
                'avg_confidence': float(np.mean(metrics.decision_confidence)),
                'confidence_correlation': float(confidence_correlation),
                'avg_move_time': float(np.mean(metrics.move_times)),
                'num_critical_positions': len(self.critical_positions[phase])
            }

        # Add overall statistics
        all_predictions = []
        all_outcomes = []
        for metrics in self.phase_metrics.values():
            all_predictions.extend(metrics.value_predictions)
            all_outcomes.extend(metrics.actual_outcomes)

        if all_predictions:
            summary['overall'] = {
                'total_positions': len(all_predictions),
                'avg_value_accuracy': float(np.mean([
                    (pred > 0) == (actual > 0)
                    for pred, actual in zip(all_predictions, all_outcomes)
                ])),
                'total_critical_positions': sum(
                    len(positions) for positions in self.critical_positions.values()
                )
            }

        self.current_summary = summary
        return summary

    def get_serializable_data(self) -> Dict:
        """Get all metrics in a JSON-serializable format"""
        return {
            'summary': self.get_metrics_summary(),
            'phase_metrics': {
                phase: {
                    'value_predictions': [float(x) for x in metrics.value_predictions],
                    'actual_outcomes': [float(x) for x in metrics.actual_outcomes],
                    'move_times': [float(x) for x in metrics.move_times],
                    'decision_confidence': [float(x) for x in metrics.decision_confidence]
                }
                for phase, metrics in self.phase_metrics.items()
            },
            'critical_positions': {
                phase: [
                    {k: float(v) if isinstance(v, (int, float)) else v
                     for k, v in pos.items()}
                    for pos in positions
                ]
                for phase, positions in self.critical_positions.items()
            }
        }