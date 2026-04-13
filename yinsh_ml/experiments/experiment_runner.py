"""
Experiment runner with full observability integration.

Orchestrates training experiments with:
- Automatic configuration loading
- Metrics aggregation
- Early stopping
- Git commit tracking
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .experiment_config import ExperimentConfig, load_config, validate_config
from .experiment_db import ExperimentDB, ExperimentRecord
from .metrics_aggregator import MetricsAggregator

logger = logging.getLogger(__name__)


def get_git_info() -> Dict[str, str]:
    """Get current git commit and branch."""
    info = {'commit': '', 'branch': ''}
    try:
        info['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]

        info['branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


class EarlyStopChecker:
    """Monitors training for early stopping conditions."""

    def __init__(
        self,
        enabled: bool = True,
        elo_threshold: float = 1400,
        patience: int = 5,
        loss_divergence_threshold: float = 10.0,
        value_collapse_threshold: float = 0.01,
        peak_detection_enabled: bool = True,
        peak_patience: int = 3,
        peak_regression_threshold: float = 30.0
    ):
        self.enabled = enabled
        self.elo_threshold = elo_threshold
        self.patience = patience
        self.loss_divergence_threshold = loss_divergence_threshold
        self.value_collapse_threshold = value_collapse_threshold

        # Peak detection settings
        self.peak_detection_enabled = peak_detection_enabled
        self.peak_patience = peak_patience
        self.peak_regression_threshold = peak_regression_threshold

        self._consecutive_rejections = 0
        self._consecutive_elo_drops = 0
        self._consecutive_below_peak = 0
        self._last_elo = 1500.0
        self._peak_elo = 1500.0
        self._peak_iteration = 0
        self._elo_history: list = []

    def check(self, iteration_result: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if training should stop early.

        Returns:
            (should_stop, reason) tuple
        """
        if not self.enabled:
            return False, ""

        # Extract metrics
        tournament = iteration_result.get('evaluation', {}).get('tournament', {})
        elo = tournament.get('rating', 1500)
        training = iteration_result.get('training', {})
        policy_loss = training.get('policy_loss', 0)
        value_loss = training.get('value_loss', 0)
        model_selection = iteration_result.get('model_selection', {})
        promoted = model_selection.get('best_iteration') == iteration_result.get('iteration')

        self._elo_history.append(elo)
        iteration = iteration_result.get('iteration', len(self._elo_history) - 1)

        # Update peak tracking
        if elo > self._peak_elo:
            self._peak_elo = elo
            self._peak_iteration = iteration
            self._consecutive_below_peak = 0
        elif self.peak_detection_enabled and (self._peak_elo - elo) >= self.peak_regression_threshold:
            self._consecutive_below_peak += 1
        else:
            # Still below peak but within threshold - don't increment
            pass

        # Check 0: Peak regression detection (most important for stability)
        if self.peak_detection_enabled and self._consecutive_below_peak >= self.peak_patience:
            return True, (
                f"Peak regression detected: ELO {elo:.0f} is {self._peak_elo - elo:.0f} points below "
                f"peak {self._peak_elo:.0f} (iter {self._peak_iteration}) for {self._consecutive_below_peak} iterations"
            )

        # Check 1: ELO below threshold for too long
        if elo < self.elo_threshold:
            self._consecutive_elo_drops += 1
            if self._consecutive_elo_drops >= self.patience:
                return True, f"ELO ({elo:.0f}) below threshold ({self.elo_threshold}) for {self.patience} iterations"
        else:
            self._consecutive_elo_drops = 0

        # Check 2: Too many consecutive rejections
        if not promoted:
            self._consecutive_rejections += 1
            if self._consecutive_rejections >= self.patience:
                return True, f"Model rejected for {self.patience} consecutive iterations"
        else:
            self._consecutive_rejections = 0

        # Check 3: Loss divergence
        if policy_loss > self.loss_divergence_threshold or value_loss > self.loss_divergence_threshold:
            return True, f"Loss divergence detected (policy={policy_loss:.2f}, value={value_loss:.2f})"

        # Check 4: NaN/Inf loss
        if any(not (isinstance(v, (int, float)) and abs(v) < float('inf'))
               for v in [policy_loss, value_loss]):
            return True, "NaN or Inf loss detected"

        self._last_elo = elo
        return False, ""

    def get_status(self) -> Dict[str, Any]:
        """Get current early stop status."""
        return {
            'consecutive_rejections': self._consecutive_rejections,
            'consecutive_elo_drops': self._consecutive_elo_drops,
            'consecutive_below_peak': self._consecutive_below_peak,
            'last_elo': self._last_elo,
            'peak_elo': self._peak_elo,
            'peak_iteration': self._peak_iteration,
            'elo_history': self._elo_history[-10:]  # Last 10
        }


class ExperimentRunner:
    """
    Runs training experiments with full observability.

    Usage:
        runner = ExperimentRunner(config)
        result = runner.run()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "experiments",
        resume_from: Optional[str] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.resume_from = resume_from

        # Initialize database
        self.db = ExperimentDB(str(self.output_dir / "experiments.db"))

        # Initialize early stop checker
        self.early_stop = EarlyStopChecker(
            enabled=config.early_stop_enabled,
            elo_threshold=config.early_stop_elo_threshold,
            patience=config.early_stop_patience,
            peak_detection_enabled=config.peak_detection_enabled,
            peak_patience=config.peak_patience,
            peak_regression_threshold=config.peak_regression_threshold
        )

        # Will be set during run()
        self.experiment_id: Optional[str] = None
        self.metrics: Optional[MetricsAggregator] = None
        self.supervisor = None

    def _create_experiment_record(self) -> ExperimentRecord:
        """Create experiment record for database."""
        git_info = get_git_info()

        return ExperimentRecord(
            name=self.config.name,
            description=self.config.description,
            git_commit=git_info['commit'],
            git_branch=git_info['branch'],
            config_json=json.dumps(self.config.to_dict()),
            status="pending",
            total_iterations=self.config.iterations
        )

    def _create_supervisor(self):
        """Create TrainingSupervisor from config."""
        from ..network.wrapper import NetworkWrapper
        from ..training.supervisor import TrainingSupervisor

        # Determine device
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # Create network
        network = NetworkWrapper(device=device)

        # Create save directory
        save_dir = self.output_dir / self.experiment_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create supervisor with config
        mode_settings = self.config.to_mode_settings()

        supervisor = TrainingSupervisor(
            network=network,
            save_dir=str(save_dir),
            device=device,
            tournament_games=self.config.tournament.games_per_match,
            mcts_simulations=self.config.mcts.early_simulations,
            mode_settings=mode_settings
        )

        return supervisor

    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.

        Returns:
            Dict with experiment results
        """
        start_time = time.time()

        # Validate config
        validate_config(self.config)

        # Create or resume experiment
        if self.resume_from:
            record = self.db.get_experiment(self.resume_from)
            if not record:
                raise ValueError(f"Cannot resume: experiment {self.resume_from} not found")
            self.experiment_id = record.experiment_id
            start_iteration = record.current_iteration
            logger.info(f"Resuming experiment {self.experiment_id} from iteration {start_iteration}")
        else:
            record = self._create_experiment_record()
            self.experiment_id = self.db.create_experiment(record)
            start_iteration = 0
            logger.info(f"Created new experiment: {self.experiment_id} ({self.config.name})")

        # Update status to running
        self.db.update_experiment(self.experiment_id, status="running")

        # Initialize metrics aggregator
        self.metrics = MetricsAggregator(
            experiment_id=self.experiment_id,
            config=self.config.to_dict(),
            output_dir=str(self.output_dir),
            enable_tensorboard=self.config.tensorboard_enabled,
            verbosity=self.config.verbosity
        )

        # Create supervisor
        self.supervisor = self._create_supervisor()

        # Set experiment tracker on supervisor
        self.supervisor.set_experiment_tracker(self.db, self.experiment_id)

        results = {
            'experiment_id': self.experiment_id,
            'iterations': [],
            'early_stopped': False,
            'early_stop_reason': '',
            'final_status': 'completed'
        }

        try:
            for iteration in range(start_iteration, self.config.iterations):
                logger.info(f"\n{'='*60}")
                logger.info(f"ITERATION {iteration}/{self.config.iterations - 1}")
                logger.info(f"{'='*60}\n")

                # Run training iteration
                iteration_result = self.supervisor.train_iteration(
                    num_games=self.config.training.games_per_iteration,
                    epochs=self.config.training.epochs_per_iteration
                )

                # Log to metrics aggregator
                self.metrics.log_iteration_summary(iteration, iteration_result)
                results['iterations'].append(iteration_result)

                # Check for early stopping
                should_stop, reason = self.early_stop.check(iteration_result)
                if should_stop:
                    logger.warning(f"Early stopping triggered: {reason}")
                    results['early_stopped'] = True
                    results['early_stop_reason'] = reason
                    self.metrics.log_alert('early_stop', reason, iteration)
                    break

                # Log progress
                elo = iteration_result.get('evaluation', {}).get('tournament', {}).get('rating', 1500)
                best_iter = iteration_result.get('model_selection', {}).get('best_iteration', 0)
                logger.info(f"Iteration {iteration} complete: ELO={elo:.0f}, Best={best_iter}")

            results['final_status'] = 'completed'

        except KeyboardInterrupt:
            logger.warning("Experiment interrupted by user")
            results['final_status'] = 'cancelled'
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            results['final_status'] = 'failed'
            raise
        finally:
            # Update final status
            elapsed = time.time() - start_time
            self.db.update_experiment(
                self.experiment_id,
                status=results['final_status'],
                total_runtime_seconds=elapsed
            )

            # Close metrics
            if self.metrics:
                self.metrics.close()

            # Cleanup supervisor
            if self.supervisor:
                self.supervisor.cleanup_memory_pools()

            logger.info(f"\nExperiment {self.experiment_id} finished: {results['final_status']}")
            logger.info(f"Total runtime: {elapsed/3600:.1f} hours")

        return results


def run_experiment(config_path: str, output_dir: str = "experiments", resume: Optional[str] = None) -> Dict[str, Any]:
    """
    Run an experiment from a config file.

    Args:
        config_path: Path to YAML config file
        output_dir: Directory for experiment outputs
        resume: Optional experiment ID to resume

    Returns:
        Experiment results dict
    """
    config = load_config(config_path)
    runner = ExperimentRunner(config, output_dir, resume)
    return runner.run()


def run_experiment_from_config(config: ExperimentConfig, output_dir: str = "experiments") -> Dict[str, Any]:
    """
    Run an experiment from a config object.

    Args:
        config: ExperimentConfig object
        output_dir: Directory for experiment outputs

    Returns:
        Experiment results dict
    """
    runner = ExperimentRunner(config, output_dir)
    return runner.run()


# CLI entry point
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run YinshML training experiment')
    parser.add_argument('config', help='Path to experiment config YAML')
    parser.add_argument('--output-dir', default='experiments', help='Output directory')
    parser.add_argument('--resume', help='Experiment ID to resume')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run experiment
    result = run_experiment(args.config, args.output_dir, args.resume)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"ID: {result['experiment_id']}")
    print(f"Status: {result['final_status']}")
    print(f"Iterations: {len(result['iterations'])}")
    if result['early_stopped']:
        print(f"Early stopped: {result['early_stop_reason']}")
    print("="*60)
