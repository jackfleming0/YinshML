import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from .enhanced_metrics import EnhancedMetricsCollector

@dataclass
class GameMetrics:
    length: int
    outcome: int
    duration: float
    avg_move_time: float
    phase_values: Dict[str, List[float]]  # Value predictions by game phase
    final_confidence: float
    temperature_data: Dict

@dataclass
class EpochMetrics:
    policy_loss: float
    value_loss: float
    value_accuracy: float
    move_accuracies: Dict[str, float]
    learning_rates: Dict[str, float]
    gradient_norm: float
    loss_improvement: float  # Relative to previous epoch

class MetricsLogger:
    def __init__(self, save_dir: Path, debug: bool = False):
        self.enhanced_metrics = EnhancedMetricsCollector()
        self.save_dir = Path(save_dir)
        self.metrics_dir = self.save_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("MetricsLogger")
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)

        # Enhanced storage
        self.current_iteration = None
        self.current_metrics = self._init_metrics_storage()

        # Running statistics
        self.game_length_history = []
        self.value_accuracy_history = defaultdict(list)
        self.training_curves = defaultdict(list)

    def _init_metrics_storage(self) -> Dict:
        """Initialize metrics storage with all necessary fields."""
        return {
            'games': [],
            'training': [],
            'tournament': None,
            'summary_stats': {
                'game_lengths': {
                    'mean': 0.0,
                    'std': 0.0,
                    'distribution': None
                },
                'value_head': {
                    'accuracy_by_phase': {},
                    'confidence_trend': [],
                    'correlation_with_outcome': 0.0
                },
                'learning_dynamics': {
                    'policy_loss_trend': [],
                    'value_loss_trend': [],
                    'gradient_norms': [],
                    'plateau_detected': False
                }
            }
        }

    # def record_game_history(self, game_history: List[Dict]):
    #     """Record the history of a single game for later analysis."""
    #     for entry in game_history:
    #         self.enhanced_metrics.record_evaluation(
    #             state=entry['state'],
    #             value_pred=entry['value_pred'],
    #             policy_probs=entry['move_probs'],
    #             chosen_move=entry['move'],
    #             temperature=entry['temperature'],
    #             actual_outcome=entry['outcome']
    #         )

    def start_iteration(self, iteration: int):
        """Start tracking a new iteration."""
        self.current_iteration = iteration
        self.current_metrics = self._init_metrics_storage()
        self.logger.info(f"Starting iteration {iteration}")

    def log_game(self, metrics: GameMetrics):
        """Log metrics from a completed game with enhanced tracking."""
        if self.current_iteration is None:
            raise ValueError("Must call start_iteration first")

        # Store basic game metrics
        self.current_metrics['games'].append(vars(metrics))

        # Update running statistics
        self.game_length_history.append(metrics.length)

        # Track value head performance by phase
        for phase, values in metrics.phase_values.items():
            self.value_accuracy_history[phase].append(
                self._compute_value_accuracy(values, metrics.outcome)
            )

    def log_training(self, metrics: EpochMetrics):
        """Log training metrics with enhanced analysis."""
        if self.current_iteration is None:
            raise ValueError("Must call start_iteration first")

        # Store epoch metrics
        self.current_metrics['training'].append(vars(metrics))

        # Update training curves
        self.training_curves['policy_loss'].append(metrics.policy_loss)
        self.training_curves['value_loss'].append(metrics.value_loss)
        self.training_curves['gradient_norm'].append(metrics.gradient_norm)

        # Check for plateau
        if self._detect_plateau():
            self.current_metrics['summary_stats']['learning_dynamics']['plateau_detected'] = True

    def _compute_value_accuracy(self, predictions: List[float], actual_outcome: int) -> float:
        """Compute accuracy of value head predictions."""
        predicted_outcomes = [1 if v > 0 else -1 for v in predictions]
        return sum(p == actual_outcome for p in predicted_outcomes) / len(predictions)

    def _detect_plateau(self, window_size: int = 5) -> bool:
        """Detect if training has plateaued."""
        if len(self.training_curves['policy_loss']) < window_size:
            return False

        recent_loss = self.training_curves['policy_loss'][-window_size:]
        loss_std = np.std(recent_loss)

        # Add checks for zero values
        if not recent_loss or recent_loss[0] == 0:
            return False

        loss_improvement = (recent_loss[0] - recent_loss[-1]) / (recent_loss[0] + 1e-8)  # Add small epsilon

        return loss_std < 0.01 and loss_improvement < 0.001

    def summarize_iteration(self) -> Dict:
        """Generate comprehensive iteration summary."""
        games = self.current_metrics['games']

        # Game length analysis
        lengths = [g['length'] for g in games]
        self.current_metrics['summary_stats']['game_lengths'] = {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'distribution': np.histogram(lengths, bins=10)
        }

        # Value head analysis
        for phase, accuracies in self.value_accuracy_history.items():
            self.current_metrics['summary_stats']['value_head']['accuracy_by_phase'][phase] = {
                'mean': np.mean(accuracies),
                'trend': self._compute_trend(accuracies)
            }

        # Learning dynamics
        self.current_metrics['summary_stats']['learning_dynamics'].update({
            'policy_loss_trend': self._compute_trend(self.training_curves['policy_loss']),
            'value_loss_trend': self._compute_trend(self.training_curves['value_loss']),
            'gradient_norms': self.training_curves['gradient_norm']
        })

        return self.current_metrics['summary_stats']

    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend in metric using linear regression."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]

    def save_iteration(self):
        """Save all metrics with enhanced summary stats."""
        if self.current_iteration is None:
            self.logger.warning("Trying to save iteration but no iteration started")
            return

        self.summarize_iteration()

        # Add debug output for metrics structure
        print("\nMetrics structure before conversion:")
        print(json.dumps(self._debug_structure(self.current_metrics), indent=2))

        # Combine standard and enhanced metrics
        output_file = self.metrics_dir / f"iteration_{self.current_iteration}.json"
        metrics = {
            'iteration': self.current_iteration,
            'timestamp': datetime.now().isoformat(),
            'metrics': self._convert_to_serializable(self.current_metrics),
            'enhanced_metrics': self.enhanced_metrics.get_serializable_data()
        }

        self.logger.info(f"Saving metrics to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def _debug_structure(self, obj):
        """Return structure of object for debugging."""
        if isinstance(obj, dict):
            return {k: f"{type(v).__name__}" for k, v in obj.items()}
        if isinstance(obj, list):
            return [f"{type(item).__name__}" for item in obj]
        return f"{type(obj).__name__}"

    def _convert_to_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        # Handle numpy scalar types first
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        if isinstance(obj, (bool, str, int, float)):
            return obj
        if obj is None:
            return None

        # Convert anything else to string
        return str(obj)

    def plot_current_metrics(self) -> None:
        """Generate plots for current iteration metrics."""
        if self.current_iteration is None:
            self.logger.warning("Trying to plot metrics but no iteration started")
            return

        self.logger.info("Generating plots...")  # Debug log

        if not self.current_metrics['games']:
            self.logger.warning("No games data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Game length distribution
        lengths = [g['length'] for g in self.current_metrics['games']]
        axes[0, 0].hist(lengths, bins=20, alpha=0.7, label='Game Lengths')
        axes[0, 0].set_title('Game Length Distribution')
        axes[0, 0].set_xlabel('Number of Moves')
        axes[0, 0].set_ylabel('Frequency')

        # Value accuracy by phase
        phases = list(self.value_accuracy_history.keys())
        accuracies = [np.mean(self.value_accuracy_history[p]) for p in phases]
        axes[0, 1].bar(phases, accuracies, alpha=0.7)
        axes[0, 1].set_title('Value Head Accuracy by Phase')
        axes[0, 1].set_xlabel('Game Phase')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim([0, 1])  # Set y-axis limits to 0-1 for accuracy

        # Training curves
        if self.training_curves['policy_loss']:
            epochs = range(len(self.training_curves['policy_loss']))

            # Primary axis for Policy Loss
            ax1 = axes[1, 0]
            ax1.set_title('Training Losses')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Policy Loss', color='blue')
            p1 = ax1.plot(epochs, self.training_curves['policy_loss'], label='Policy Loss', color='blue')

            # Create twin axis for Value Loss
            ax2 = ax1.twinx()
            ax2.set_ylabel('Value Loss', color='orange')
            p2 = ax2.plot(epochs, self.training_curves['value_loss'], label='Value Loss', color='orange')

            # Combine legends
            lines = p1 + p2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc=0)
        else:
            axes[1, 0].text(
                0.5, 0.5, 'No Data',
                horizontalalignment='center',
                verticalalignment='center'
            )

        # Gradient norms
        if self.training_curves['gradient_norm']:
            axes[1, 1].plot(epochs, self.training_curves['gradient_norm'])
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Norm Value')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

        # Save and close figure
        plot_path = self.metrics_dir / f"iteration_{self.current_iteration}_plots.png"
        self.logger.info(f"Saving plots to {plot_path}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()