"""
Unified metrics aggregation that broadcasts to multiple backends.

Provides a single interface for logging metrics that get distributed to:
- JSON files (for easy inspection)
- SQLite database (for querying)
- TensorBoard (for visualization)
- Console (for real-time monitoring)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """A single metric entry."""
    name: str
    value: float
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def log_metric(self, entry: MetricEntry):
        """Log a single metric."""
        pass

    @abstractmethod
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters at experiment start."""
        pass

    @abstractmethod
    def flush(self):
        """Flush any buffered data."""
        pass

    @abstractmethod
    def close(self):
        """Close the backend."""
        pass


class JSONMetricsBackend(MetricsBackend):
    """Writes metrics to a JSON file."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics: List[Dict[str, Any]] = []
        self.hyperparameters: Dict[str, Any] = {}
        self._load_existing()

    def _load_existing(self):
        """Load existing metrics if file exists."""
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
                    self.metrics = data.get('metrics', [])
                    self.hyperparameters = data.get('hyperparameters', {})
            except (json.JSONDecodeError, KeyError):
                pass

    def log_metric(self, entry: MetricEntry):
        self.metrics.append({
            'name': entry.name,
            'value': entry.value,
            'step': entry.step,
            'timestamp': entry.timestamp,
            'tags': entry.tags
        })

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        self.hyperparameters = hparams

    def flush(self):
        with open(self.output_path, 'w') as f:
            json.dump({
                'hyperparameters': self.hyperparameters,
                'metrics': self.metrics,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    def close(self):
        self.flush()


class SQLiteMetricsBackend(MetricsBackend):
    """Writes metrics to SQLite via ExperimentDB."""

    def __init__(self, db_path: str, experiment_id: str):
        from .experiment_db import ExperimentDB, IterationMetrics
        self.db = ExperimentDB(db_path)
        self.experiment_id = experiment_id
        self._current_iteration_metrics: Dict[str, float] = {}
        self._current_iteration: int = 0

    def log_metric(self, entry: MetricEntry):
        # Collect metrics for the current iteration
        self._current_iteration = entry.step
        self._current_iteration_metrics[entry.name] = entry.value

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        self.db.update_experiment(
            self.experiment_id,
            config_json=json.dumps(hparams)
        )

    def log_iteration_summary(self, iteration: int, summary: Dict[str, Any]):
        """Log complete iteration summary."""
        from .experiment_db import IterationMetrics

        metrics = IterationMetrics(
            iteration=iteration,
            selfplay_time=summary.get('training', {}).get('game_time', 0),
            selfplay_games=summary.get('training_games', {}).get('pseudo_wins_white', 0) +
                          summary.get('training_games', {}).get('pseudo_wins_black', 0) +
                          summary.get('training_games', {}).get('pseudo_draws', 0),
            avg_game_length=summary.get('training_games', {}).get('avg_game_length', 0),
            training_time=summary.get('training', {}).get('training_time', 0),
            policy_loss=summary.get('training', {}).get('policy_loss', 0),
            value_loss=summary.get('training', {}).get('value_loss', 0),
            tournament_elo=summary.get('evaluation', {}).get('tournament', {}).get('rating', 1500),
            tournament_win_rate=summary.get('evaluation', {}).get('tournament', {}).get('win_rate', 0),
            promoted=summary.get('model_selection', {}).get('best_iteration', -1) == iteration,
            reverted=summary.get('model_selection', {}).get('reverted_to_best', False),
            active_iteration=summary.get('model_selection', {}).get('active_network_iteration', iteration)
        )

        self.db.log_iteration(self.experiment_id, metrics)

    def flush(self):
        pass  # SQLite commits are immediate

    def close(self):
        pass


class ConsoleMetricsBackend(MetricsBackend):
    """Prints metrics to console."""

    def __init__(self, verbosity: str = 'info'):
        self.verbosity = verbosity
        self._last_step = -1

    def log_metric(self, entry: MetricEntry):
        if self.verbosity == 'debug':
            print(f"[{entry.step}] {entry.name}: {entry.value:.4f}")
        elif self.verbosity == 'info' and entry.step != self._last_step:
            # Print summary only on step change
            self._last_step = entry.step

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        if self.verbosity in ('debug', 'info'):
            print("\n" + "="*60)
            print("EXPERIMENT HYPERPARAMETERS")
            print("="*60)
            self._print_dict(hparams, indent=0)
            print("="*60 + "\n")

    def _print_dict(self, d: Dict, indent: int = 0):
        for k, v in d.items():
            prefix = "  " * indent
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                self._print_dict(v, indent + 1)
            else:
                print(f"{prefix}{k}: {v}")

    def flush(self):
        pass

    def close(self):
        pass


class TensorBoardMetricsBackend(MetricsBackend):
    """Writes metrics to TensorBoard."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        self._init_writer()

    def _init_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("TensorBoard not available, metrics will not be logged")

    def log_metric(self, entry: MetricEntry):
        if self.writer:
            self.writer.add_scalar(entry.name, entry.value, entry.step)

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        if self.writer:
            # Flatten nested dict for TensorBoard
            flat_hparams = self._flatten_dict(hparams)
            # TensorBoard expects metric_dict for hparams
            self.writer.add_hparams(flat_hparams, {'dummy': 0})

    def _flatten_dict(self, d: Dict, prefix: str = '') -> Dict[str, Any]:
        items = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, key))
            elif isinstance(v, (int, float, str, bool)):
                items[key] = v
        return items

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


class MetricsAggregator:
    """
    Central metrics aggregator that broadcasts to all backends.

    Usage:
        with MetricsAggregator(experiment_id, config) as metrics:
            metrics.log('loss/policy', 0.5, step=0)
            metrics.log('loss/value', 0.3, step=0)
            metrics.log_iteration_summary(0, result_dict)
    """

    def __init__(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        output_dir: str = "experiments",
        enable_tensorboard: bool = True,
        enable_json: bool = True,
        enable_sqlite: bool = True,
        enable_console: bool = True,
        verbosity: str = 'info'
    ):
        self.experiment_id = experiment_id
        self.config = config
        self.output_dir = Path(output_dir)
        self.backends: List[MetricsBackend] = []

        # Initialize backends
        if enable_json:
            json_path = self.output_dir / experiment_id / "metrics.json"
            self.backends.append(JSONMetricsBackend(str(json_path)))

        if enable_sqlite:
            db_path = self.output_dir / "experiments.db"
            self.sqlite_backend = SQLiteMetricsBackend(str(db_path), experiment_id)
            self.backends.append(self.sqlite_backend)
        else:
            self.sqlite_backend = None

        if enable_tensorboard:
            tb_dir = self.output_dir / experiment_id / "tensorboard"
            self.backends.append(TensorBoardMetricsBackend(str(tb_dir)))

        if enable_console:
            self.backends.append(ConsoleMetricsBackend(verbosity))

        self._start_time = time.time()
        logger.info(f"MetricsAggregator initialized with {len(self.backends)} backends")

    def __enter__(self):
        # Log hyperparameters at start
        self.log_hyperparameters(self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def log(
        self,
        metric_name: str,
        value: float,
        step: int,
        tags: Optional[Dict[str, str]] = None
    ):
        """Log a single metric to all backends."""
        entry = MetricEntry(
            name=metric_name,
            value=value,
            step=step,
            tags=tags or {}
        )

        for backend in self.backends:
            try:
                backend.log_metric(entry)
            except Exception as e:
                logger.warning(f"Failed to log metric to {type(backend).__name__}: {e}")

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters to all backends."""
        for backend in self.backends:
            try:
                backend.log_hyperparameters(hparams)
            except Exception as e:
                logger.warning(f"Failed to log hparams to {type(backend).__name__}: {e}")

    def log_iteration_summary(self, iteration: int, summary: Dict[str, Any]):
        """
        Log complete iteration summary.

        This is the main entry point for logging training iteration results.
        """
        # Log to SQLite with full structure
        if self.sqlite_backend:
            self.sqlite_backend.log_iteration_summary(iteration, summary)

        # Log individual metrics to other backends
        training = summary.get('training', {})
        evaluation = summary.get('evaluation', {})
        tournament = evaluation.get('tournament', {})
        model_selection = summary.get('model_selection', {})

        # Training metrics
        self.log('loss/policy', training.get('policy_loss', 0), iteration)
        self.log('loss/value', training.get('value_loss', 0), iteration)
        self.log('time/selfplay', training.get('game_time', 0), iteration)
        self.log('time/training', training.get('training_time', 0), iteration)

        # Tournament metrics
        self.log('tournament/elo', tournament.get('rating', 1500), iteration)
        self.log('tournament/win_rate', tournament.get('win_rate', 0), iteration)

        # Model selection
        self.log('model/best_iteration', model_selection.get('best_iteration', 0), iteration)
        self.log('model/promoted', 1 if model_selection.get('best_iteration') == iteration else 0, iteration)

        # Flush all backends
        self.flush()

    def log_alert(self, alert_type: str, message: str, step: int):
        """Log an alert condition."""
        self.log(f'alert/{alert_type}', 1, step, tags={'message': message})
        logger.warning(f"[ALERT] {alert_type}: {message}")

    def flush(self):
        """Flush all backends."""
        for backend in self.backends:
            try:
                backend.flush()
            except Exception as e:
                logger.warning(f"Failed to flush {type(backend).__name__}: {e}")

    def close(self):
        """Close all backends."""
        elapsed = time.time() - self._start_time
        logger.info(f"MetricsAggregator closing after {elapsed:.1f}s")

        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                logger.warning(f"Failed to close {type(backend).__name__}: {e}")

    def get_elapsed_time(self) -> float:
        """Get elapsed time since aggregator was created."""
        return time.time() - self._start_time
