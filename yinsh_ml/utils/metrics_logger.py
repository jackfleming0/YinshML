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

        # Optional experiment tracker handle for forwarding log_event /
        # log_scalar calls. Set via set_experiment_tracker(). Not required —
        # events still land in the JSON sidecar when this is None.
        self._experiment_tracker = None
        self._experiment_id: Optional[int] = None

    def _init_metrics_storage(self) -> Dict:
        """Initialize metrics storage with all necessary fields."""
        return {
            'games': [],
            'training': [],
            'tournament': None,
            # Generic time-series scalars logged via `log_scalar()`. Keyed by
            # metric name, each entry is a list of {step, value, ...} dicts.
            # Used by W1b telemetry safeguards (B1 effective_child_visits,
            # B2 value_outcome_correlation, B3 policy_target_entropy_mean)
            # and W1e collapse-alert routing.
            'scalars': defaultdict(list),
            # Discrete events/alerts logged via `log_event()` (W1e). Flat
            # list of dicts so dashboards can render an event timeline.
            'events': [],
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

    def log_scalar(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Log a scalar metric value (W1b telemetry + W1e collapse routing).

        Storage is dict-keyed by metric name in
        ``self.current_metrics['scalars'][name]`` so each metric is its own
        time series. Non-finite values are stored as ``None`` (JSON survives).
        Non-numeric values are dropped with a warning.

        Args:
            name: Metric path, e.g. ``"mcts/effective_child_visits"``.
            value: Scalar value (coerced to float).
            step: Optional global step (preferred when training-step semantics
                matter, e.g. inside the inner train loop).
            iteration: Optional iteration index (W1e callers from eval/anchor
                paths use this). Falls back to ``self.current_iteration``.

        ``step`` and ``iteration`` are functionally equivalent; ``step`` wins
        when both are supplied, then ``iteration``, then current_iteration.
        Forwarded to a connected experiment tracker if one is set.
        """
        # Lazy-init so callers can use this without start_iteration() first
        # (focused unit tests, eval-only invocations).
        if 'scalars' not in self.current_metrics:
            self.current_metrics['scalars'] = defaultdict(list)
        try:
            v = float(value)
        except (TypeError, ValueError):
            self.logger.warning(f"log_scalar({name}): non-numeric value {value!r}, dropping")
            return
        if not np.isfinite(v):
            self.logger.warning(f"log_scalar({name}): non-finite value {v}, storing as None")
            stored: Optional[float] = None
        else:
            stored = v
        resolved_step = step if step is not None else iteration
        if resolved_step is None:
            resolved_step = self.current_iteration
        self.current_metrics['scalars'][name].append({'step': resolved_step, 'value': stored})
        self.logger.debug(f"scalar/{name} @ {resolved_step}: {stored}")

        # Forward to experiment tracker (W1e). No-op if none attached.
        if (
            self._experiment_tracker is not None
            and self._experiment_id is not None
            and stored is not None
        ):
            try:
                self._experiment_tracker.log_metric(
                    self._experiment_id, name, stored, resolved_step,
                )
            except Exception as e:
                self.logger.warning(f"Failed to forward scalar '{name}' to tracker: {e}")

    # ---- B2: value-outcome correlation ------------------------------------
    # The intent: per evaluation pass, collect (root_value_at_position_start,
    # terminal_outcome_in_position_pov) pairs across the eval games, then
    # Pearson r → log as ``eval/value_outcome_correlation``.
    #
    # Why it lives here and not in `enhanced_metrics.py`: that collector
    # already computes a CONFIDENCE-vs-error correlation per game phase; the
    # B2 safeguard needs a fundamentally different aggregation (across all
    # eval positions, not per-phase), so a separate buffer keeps the two
    # signals independent.
    #
    # See `compute_and_log_value_outcome_correlation()` for the aggregation
    # entry point. `log_eval_value_pair()` is the per-position append; both
    # are no-ops if the caller never wired them up.

    def log_eval_value_pair(self, root_value: float, terminal_outcome: float) -> None:
        """Append a (root_value, terminal_outcome) pair for B2 correlation.

        Both arguments are expected in the same POV — typically the side
        whose move it was at position start. Non-finite values are skipped
        with a debug log (don't poison Pearson with NaN).

        TODO(W1e): wire the actual call sites in tournament eval and/or the
        per-iteration validation pass. Until then this method is callable
        but uncalled; the unit test exercises it directly so the math is
        proven before the plumbing lands. The integration point is
        ``ModelTournament.run_anchor_eval`` (after each game completes,
        you have access to ``terminal_outcome`` from the game runner and
        the MCTS's ``last_root_value`` from the candidate's first move) —
        but doing so cleanly requires touching the eval loop that W1e is
        also restructuring, so we coordinate via the W1e workstream.
        """
        if not hasattr(self, '_eval_value_pairs'):
            self._eval_value_pairs: List[Tuple[float, float]] = []
        try:
            rv = float(root_value)
            to = float(terminal_outcome)
        except (TypeError, ValueError):
            self.logger.debug(f"log_eval_value_pair: non-numeric inputs {(root_value, terminal_outcome)}, skipping")
            return
        if not (np.isfinite(rv) and np.isfinite(to)):
            self.logger.debug(f"log_eval_value_pair: non-finite inputs {(rv, to)}, skipping")
            return
        self._eval_value_pairs.append((rv, to))

    def compute_and_log_value_outcome_correlation(
        self, step: Optional[int] = None, clear: bool = True
    ) -> Optional[float]:
        """Pearson r over the buffered eval pairs; logged as
        ``eval/value_outcome_correlation``.

        Returns the computed r (None if too few points). When ``clear`` is
        True (default), the buffer is reset so each eval pass produces one
        scalar. Set ``clear=False`` if you want a running correlation across
        a multi-pass eval.

        Edge cases:
          * Fewer than 2 pairs → returns None, logs nothing.
          * Zero variance on either side (degenerate eval where every game
            had the same outcome) → returns None, logs nothing, debug-warn.
        """
        pairs = getattr(self, '_eval_value_pairs', [])
        if len(pairs) < 2:
            self.logger.debug(
                f"value_outcome_correlation: only {len(pairs)} pair(s) buffered, skipping"
            )
            if clear:
                self._eval_value_pairs = []
            return None
        arr = np.asarray(pairs, dtype=np.float64)
        x, y = arr[:, 0], arr[:, 1]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            self.logger.debug(
                "value_outcome_correlation: zero variance on one side, skipping"
            )
            if clear:
                self._eval_value_pairs = []
            return None
        # np.corrcoef returns a 2x2 matrix; we want the off-diagonal Pearson r.
        r = float(np.corrcoef(x, y)[0, 1])
        self.log_scalar('eval/value_outcome_correlation', r, step=step)
        if clear:
            self._eval_value_pairs = []
        return r

    def log_event(self,
                  event_name: str,
                  severity: str = 'info',
                  iteration: Optional[int] = None,
                  details: Optional[Dict] = None) -> None:
        """Log a discrete event (alert / warning / milestone) into the metrics
        stream.

        Events are stored in ``current_metrics['events']`` so they persist into
        ``iteration_<N>.json`` and can be picked up by dashboards. Also forwarded
        to a connected experiment tracker (when set via
        :meth:`set_experiment_tracker`) as a counter metric
        ``event/<event_name>`` so it shows up as a scalar series.

        Args:
            event_name: Stable identifier (e.g. ``'deterministic_collapse_alert'``).
                Used as a metric key in the experiment tracker so keep it
                lowercase, snake_case, and namespaced.
            severity: One of ``info | warning | error``. Passed to the Python
                logger so operators see the alert in the run log.
            iteration: Training iteration where the event fired. Defaults to
                the current iteration if one has been started.
            details: Arbitrary JSON-serializable payload for context (e.g.
                ``{'sides': ['white', 'black']}``).
        """
        it = iteration if iteration is not None else self.current_iteration
        entry = {
            'name': event_name,
            'severity': severity,
            'iteration': it,
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
        }

        # Always log to the Python logger so the event appears in the run log
        # even when no iteration has been started (e.g. eval-only invocations).
        log_msg = f"[event:{event_name}] iter={it} details={details}"
        if severity == 'error':
            self.logger.error(log_msg)
        elif severity == 'warning':
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

        # Buffer the event for the next save_iteration() if we have an active
        # iteration. If not, fall back to a sidecar events file so the alert
        # isn't silently dropped.
        if self.current_iteration is not None:
            self.current_metrics.setdefault('events', []).append(entry)
        else:
            try:
                events_file = self.metrics_dir / 'events.jsonl'
                with open(events_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')
            except Exception as e:
                self.logger.warning(f"Failed to write event to sidecar: {e}")

        # Forward to experiment tracker as a counter so it shows up alongside
        # other metrics in the experiment DB / dashboards. We log the count
        # of times this event has fired this iteration (always 1 per call,
        # so plotting "sum by iteration" gives you the count).
        if self._experiment_tracker is not None and self._experiment_id is not None:
            try:
                self._experiment_tracker.log_metric(
                    self._experiment_id,
                    f'event/{event_name}',
                    1.0,
                    it,
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to forward event '{event_name}' to tracker: {e}"
                )

    def set_experiment_tracker(self, tracker, experiment_id: Optional[int]) -> None:
        """Attach an experiment tracker so log_event / log_scalar forward to it.

        Optional — when not set, events and scalars only land in the JSON
        sidecar / iteration file. The supervisor wires this up after creating
        the tracker so collapse alerts surface in dashboards.
        """
        self._experiment_tracker = tracker
        self._experiment_id = experiment_id

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