# experiments/runner.py

import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Optional
import torch
import numpy as np
from collections import defaultdict
from dataclasses import asdict # Needed for _supervisor_kwargs_from_config

# Import YINSH specific modules
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.supervisor import TrainingSupervisor # Now the core driver
from yinsh_ml.utils.metrics_logger import MetricsLogger # Still useful potentially, or handled by Supervisor
# Removed imports for Trainer, SelfPlay, Tournament, ValueHeadMetrics, MCTSMetrics as Runner delegates to Supervisor

# Import experiment tracking
try:
    from yinsh_ml.tracking.experiment_tracker import ExperimentTracker
    from yinsh_ml.tracking.config_serializer import ConfigurationSerializer
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

from experiments.config import (
    get_experiment_config,
    RESULTS_DIR, # Keep RESULTS_DIR
    # RESULTS_SUBDIRS, # REMOVED
    CombinedConfig # Now the only config type
)

# Setup logger at module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperimentRunner")


def _supervisor_kwargs_from_config(cfg: CombinedConfig) -> dict:
    """
    Convert a CombinedConfig object into a dictionary of keyword arguments
    suitable for initializing TrainingSupervisor.

    Args:
        cfg: The CombinedConfig object for the experiment.

    Returns:
        A dictionary containing parameters for TrainingSupervisor.
    """
    # Convert the dataclass to a dictionary
    config_dict = asdict(cfg)

    # --- Prepare the **mode_settings dictionary ---
    # Start with all config parameters
    mode_settings = config_dict.copy()

    # --- Prepare the final kwargs for TrainingSupervisor ---
    supervisor_kwargs = {}

    # 1. Map the config's 'num_simulations' to the supervisor's 'mcts_simulations' argument
    if 'num_simulations' in config_dict:
        supervisor_kwargs['mcts_simulations'] = config_dict['num_simulations']

    # 2. FIXED: Don't remove num_simulations from mode_settings
    #    This ensures both keys exist downstream

    # 3. Add the remaining parameters as **mode_settings
    supervisor_kwargs['mode_settings'] = mode_settings

    # Add debugging information
    logger.debug(f"Supervisor: mcts_simulations = {supervisor_kwargs.get('mcts_simulations', 'NOT SET')}")
    logger.debug(f"mode_settings: num_simulations = {mode_settings.get('num_simulations', 'NOT SET')}")

    return supervisor_kwargs


class ExperimentRunner:
    def __init__(self, device: str = 'cuda', debug: bool = False, enable_tracking: bool = True):
        self.device = device
        self.enable_tracking = enable_tracking and TRACKING_AVAILABLE
        self.logger = logging.getLogger("ExperimentRunner") # Use module logger

        # Set logging levels
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)
        # Configure other loggers if needed (e.g., supervisor, trainer)
        logging.getLogger("TrainingSupervisor").setLevel(level)
        logging.getLogger("SelfPlay").setLevel(level)
        logging.getLogger("YinshTrainer").setLevel(level)
        logging.getLogger("ModelTournament").setLevel(level)
        logging.getLogger("MCTS").setLevel(logging.DEBUG) # Keep MCTS debug for now

        # --- Initialize Experiment Tracker ---
        self.experiment_tracker = None
        self.config_serializer = None
        if self.enable_tracking:
            try:
                self.experiment_tracker = ExperimentTracker.get_instance()
                self.config_serializer = ConfigurationSerializer()
                self.logger.info("ExperimentTracker and ConfigurationSerializer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ExperimentTracker: {e}. Tracking disabled.")
                self.enable_tracking = False
        elif not TRACKING_AVAILABLE:
            self.logger.info("Experiment tracking not available (import failed). Running without tracking.")
        else:
            self.logger.info("Experiment tracking disabled by configuration.")

        # --- Simplified Directory Structure ---
        # Checkpoints are now saved within the results directory for simpler organization
        self.results_dir = RESULTS_DIR # Base results directory from config.py
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Removed baseline model logic - rely on Supervisor's promotion gate
        # Removed self_play attribute - managed by Supervisor
        # Removed metrics_logger and value_head_metrics - managed by Supervisor/Trainer
        # Removed tournament_manager - managed by Supervisor

        self.logger.info(f"ExperimentRunner initialized on device: {self.device}")
        self.logger.info(f"Results will be saved under: {self.results_dir.resolve()}")


    def run_experiment(self, config_name: str) -> Dict:
        """
        Run a single experiment using the unified CombinedConfig.

        Args:
            config_name: The name of the configuration in COMBINED_EXPERIMENTS.

        Returns:
            A dictionary containing aggregated metrics over the experiment run.
            Returns an empty dict if the experiment fails.
        """
        config = get_experiment_config(config_name)
        if config is None:
            self.logger.error(f"Failed to load configuration: {config_name}")
            return {} # Return empty dict on failure

        self.logger.info(f"Starting experiment: {config_name}")
        self.logger.info(f"Configuration details: {config}") # Log the config being used

        # --- Create experiment tracking record ---
        experiment_id = None
        if self.enable_tracking and self.experiment_tracker and self.config_serializer:
            try:
                # Capture comprehensive configuration using ConfigurationSerializer
                user_config = asdict(config)
                
                # Collect configuration objects from various components
                config_objects = {
                    'experiment_config': config,
                    'device_config': {'device': self.device, 'debug': self.logger.level == logging.DEBUG},
                }
                
                # Add capture manager configuration if available
                try:
                    from yinsh_ml.tracking.capture import CaptureManager
                    capture_manager = CaptureManager()
                    config_objects['capture_manager'] = capture_manager.get_configuration()
                except Exception as e:
                    self.logger.debug(f"Could not capture CaptureManager config: {e}")
                
                # Create comprehensive configuration snapshot
                comprehensive_config = self.config_serializer.capture_comprehensive_config(
                    user_config=user_config,
                    config_objects=config_objects
                )
                
                experiment_id = self.experiment_tracker.create_experiment(
                    name=f"yinsh_experiment_{config_name}",
                    description=f"YINSH ML training experiment using configuration: {config_name}",
                    tags=[config_name, "yinsh", "training"],
                    config=comprehensive_config
                )
                self.logger.info(f"Created experiment tracking record with ID: {experiment_id}")
                
                # Start the experiment
                self.experiment_tracker.start_experiment(experiment_id)
                
            except Exception as e:
                self.logger.warning(f"Failed to create experiment tracking record: {e}. Continuing without tracking.")
                experiment_id = None

        # --- Experiment-specific directory ---
        # All results, checkpoints, logs for this run go here
        experiment_run_dir = self.results_dir / config_name
        experiment_run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Experiment output directory: {experiment_run_dir.resolve()}")

        # --- Save experiment ID to file for checkpoint linking ---
        if experiment_id:
            try:
                experiment_id_file = experiment_run_dir / "experiment_id.txt"
                with open(experiment_id_file, 'w') as f:
                    f.write(str(experiment_id))
                self.logger.info(f"Saved experiment ID {experiment_id} to {experiment_id_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save experiment ID file: {e}")

        # Log file specific to this run
        run_log_file = experiment_run_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(run_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        # Add handler to root logger to capture everything
        logging.getLogger().addHandler(file_handler)
        self.logger.info(f"Logging detailed run output to: {run_log_file}")

        try:
            # --- Use the consolidated run method ---
            metrics = self._run_supervised_experiment(config, config_name, experiment_run_dir, experiment_id)

            # --- Save final aggregated metrics ---
            results_file = experiment_run_dir / "final_summary_metrics.json"
            # Ensure metrics are JSON serializable (convert numpy types if needed)
            serializable_metrics = {k: [float(vi) if isinstance(vi, (np.floating, np.integer)) else vi for vi in v]
                                    for k, v in metrics.items()}
            result_data = {
                "config": asdict(config), # Save the config used
                "final_metrics": serializable_metrics,
                "completed_timestamp": time.time(),
                "experiment_id": experiment_id  # Include experiment ID in results
            }

            with open(results_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            self.logger.info(f"Final aggregated metrics saved to {results_file}")

            # --- Complete experiment tracking ---
            if experiment_id and self.enable_tracking and self.experiment_tracker:
                try:
                    # Log final summary metrics
                    for metric_name, values in metrics.items():
                        if values and len(values) > 0:
                            # Log the final value of each metric
                            final_value = values[-1] if isinstance(values[-1], (int, float)) else None
                            if final_value is not None:
                                self.experiment_tracker.log_metric(
                                    experiment_id, 
                                    f"final_{metric_name}", 
                                    float(final_value)
                                )
                    
                    # Complete the experiment
                    self.experiment_tracker.complete_experiment(experiment_id)
                    self.logger.info(f"Completed experiment tracking record {experiment_id}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to complete experiment tracking: {e}")

            # Remove the run-specific file handler after the run
            logging.getLogger().removeHandler(file_handler)

            return metrics

        except Exception as e:
            self.logger.exception(f"Experiment '{config_name}' failed spectacularly!") # Log traceback
            
            # --- Mark experiment as failed in tracking ---
            if experiment_id and self.enable_tracking and self.experiment_tracker:
                try:
                    self.experiment_tracker.fail_experiment(experiment_id, str(e))
                    self.logger.info(f"Marked experiment {experiment_id} as failed in tracking")
                except Exception as track_err:
                    self.logger.warning(f"Failed to mark experiment as failed in tracking: {track_err}")
            
            # Remove the run-specific file handler on failure too
            logging.getLogger().removeHandler(file_handler)
            return {} # Return empty dict on failure


    def _run_supervised_experiment(self,
                                   cfg: CombinedConfig,
                                   cfg_name: str,
                                   run_dir: Path,
                                   experiment_id: Optional[int] = None) -> Dict[str, List]:
        """
        Generic driver: build a TrainingSupervisor from cfg and let it
        execute the canonical gate-aware loop.

        Args:
            cfg: The CombinedConfig object.
            cfg_name: The name of the configuration.
            run_dir: The specific directory for this experiment run's outputs.
            experiment_id: Optional experiment tracking ID to pass to supervisor.

        Returns:
            Dictionary containing lists of metrics collected per iteration.
        """
        # --- Initialize Network ---
        # Creates a new network for each experiment run
        network = NetworkWrapper(device=self.device)
        self.logger.info("Initialized new network for the experiment.")

        # --- Prepare Supervisor Arguments ---
        # Use the helper function to generate kwargs, including **mode_settings
        supervisor_kwargs = _supervisor_kwargs_from_config(cfg)

        # --- Initialize Supervisor ---
        # save_dir for supervisor is the run_dir where checkpoints, etc., will live
        supervisor = TrainingSupervisor(
            network=network,
            save_dir=str(run_dir), # Pass the specific run directory
            device=self.device,
            tournament_games=20, # Example: Set tournament games, or pull from config if added
            **supervisor_kwargs   # Pass base mcts_simulations and **mode_settings
        )

        # --- Pass experiment tracking info to supervisor if available ---
        if experiment_id and hasattr(supervisor, 'set_experiment_tracker'):
            try:
                supervisor.set_experiment_tracker(self.experiment_tracker, experiment_id)
                self.logger.info(f"Linked supervisor to experiment tracking ID {experiment_id}")
            except Exception as e:
                self.logger.warning(f"Failed to link supervisor to experiment tracking: {e}")

        # --- Metrics Accumulation ---
        # Stores lists of metrics, one entry per iteration
        collected_metrics: Dict[str, List] = defaultdict(list)

        # --- Main Training Loop (delegated to Supervisor) ---
        total_start_time = time.time()
        for it in range(cfg.num_iterations):
            iter_start_time = time.time()
            self.logger.info(f"--- Starting Iteration {it + 1}/{cfg.num_iterations} for {cfg_name} ---")

            # Supervisor handles self-play, training, evaluation, model saving/loading
            # Note: games_per_iteration and epochs_per_iteration are now read from cfg
            # inside the supervisor's _run_supervised_experiment uses them via the mode_settings
            summary = supervisor.train_iteration(
                num_games=cfg.games_per_iteration, # Pass explicitly or ensure it's in mode_settings
                epochs=cfg.epochs_per_iteration    # Pass explicitly or ensure it's in mode_settings
            )

            # --- Log metrics to experiment tracker ---
            if experiment_id and self.enable_tracking and self.experiment_tracker and summary:
                try:
                    iteration = it + 1  # 1-based iteration numbering
                    
                    # Log training metrics
                    if 'training' in summary:
                        training = summary['training']
                        if 'policy_loss' in training and training['policy_loss'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'policy_loss', float(training['policy_loss']), iteration)
                        if 'value_loss' in training and training['value_loss'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'value_loss', float(training['value_loss']), iteration)
                        if 'training_time' in training and training['training_time'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'training_time', float(training['training_time']), iteration)
                    
                    # Log evaluation metrics
                    if 'evaluation' in summary and 'tournament' in summary['evaluation']:
                        eval_data = summary['evaluation']['tournament']
                        if 'rating' in eval_data and eval_data['rating'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'tournament_rating', float(eval_data['rating']), iteration)
                        if 'win_rate' in eval_data and eval_data['win_rate'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'tournament_win_rate', float(eval_data['win_rate']), iteration)
                    
                    # Log game metrics
                    if 'training_games' in summary:
                        game_data = summary['training_games']
                        if 'avg_game_length' in game_data and game_data['avg_game_length'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'avg_game_length', float(game_data['avg_game_length']), iteration)
                        if 'win_rate' in game_data and game_data['win_rate'] is not None:
                            self.experiment_tracker.log_metric(experiment_id, 'training_win_rate', float(game_data['win_rate']), iteration)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to log metrics to experiment tracker for iteration {it + 1}: {e}")

            # --- Collect Metrics from Iteration Summary ---
            # Adapt this based on the actual structure of the 'summary' dict returned by train_iteration
            if summary:
                if 'training' in summary:
                    collected_metrics['policy_loss'].append(summary['training'].get('policy_loss', None))
                    collected_metrics['value_loss'].append(summary['training'].get('value_loss', None))
                    collected_metrics['training_time'].append(summary['training'].get('training_time', None))
                if 'evaluation' in summary and 'tournament' in summary['evaluation']:
                     eval_data = summary['evaluation']['tournament']
                     collected_metrics['tournament_rating'].append(eval_data.get('rating', None))
                     collected_metrics['tournament_win_rate'].append(eval_data.get('win_rate', None))
                if 'training_games' in summary:
                     game_data = summary['training_games']
                     collected_metrics['avg_game_length'].append(game_data.get('avg_game_length', None))
                     collected_metrics['training_win_rate'].append(game_data.get('win_rate', None)) # Win rate in self-play
                # Add other metrics you want to track per iteration

            iter_time = time.time() - iter_start_time
            collected_metrics['iteration_time'].append(iter_time)
            
            # --- Log iteration time to experiment tracker ---
            if experiment_id and self.enable_tracking and self.experiment_tracker:
                try:
                    self.experiment_tracker.log_metric(experiment_id, 'iteration_time', float(iter_time), it + 1)
                except Exception as e:
                    self.logger.warning(f"Failed to log iteration time to experiment tracker: {e}")
            
            self.logger.info(f"--- Iteration {it + 1} completed in {iter_time:.2f}s ---")
            # Log key metrics for the iteration
            p_loss = collected_metrics['policy_loss'][-1] if collected_metrics['policy_loss'] else 'N/A'
            v_loss = collected_metrics['value_loss'][-1] if collected_metrics['value_loss'] else 'N/A'
            elo = collected_metrics['tournament_rating'][-1] if collected_metrics['tournament_rating'] else 'N/A'
            self.logger.info(f"Iter {it+1} Metrics: Policy Loss={p_loss:.4f}, Value Loss={v_loss:.4f}, Elo={elo:.1f}")


        total_time = time.time() - total_start_time
        self.logger.info(f"Experiment '{cfg_name}' completed {cfg.num_iterations} iterations in {total_time:.2f}s.")

        # --- Log total experiment time ---
        if experiment_id and self.enable_tracking and self.experiment_tracker:
            try:
                self.experiment_tracker.log_metric(experiment_id, 'total_experiment_time', float(total_time))
            except Exception as e:
                self.logger.warning(f"Failed to log total experiment time: {e}")

        # --- Append final trainer losses (might be slightly different from last iteration avg) ---
        # Note: Accessing supervisor.trainer assumes it exists and holds final loss lists
        if hasattr(supervisor, 'trainer'):
             collected_metrics['final_policy_losses'] = supervisor.trainer.policy_losses
             collected_metrics['final_value_losses'] = supervisor.trainer.value_losses

        return dict(collected_metrics) # Convert back to regular dict


    # --- Removed Methods ---
    # _init_baseline_model (Rely on tournament history)
    # _save_checkpoint (Handled by Supervisor)
    # _load_checkpoint (Handled by Supervisor)
    # _run_learning_rate_experiment (Consolidated)
    # _run_mcts_experiment (Consolidated)
    # _run_temperature_experiment (Consolidated)
    # _run_combined_experiment (Consolidated - Renamed to _run_supervised_experiment)
    # _evaluate_against_baseline (Rely on tournament)
    # get_terminal_state (Helper moved or unused)
    # _save_results (Integrated into run_experiment)
    # _should_stop_early (Can be added back to Supervisor if needed)
    # _calculate_win_rate (Done within Supervisor or SelfPlay)
    # _calculate_move_entropy (Done within Supervisor or SelfPlay)
    # _calculate_move_accuracy (Done within Trainer/Supervisor)
    # _track_metrics (Integrated into iteration loop)
    # _calculate_average_search_time (Potentially add to SelfPlay/Supervisor if needed)
    # _win_rate_to_elo (Potentially add to Tournament/Supervisor if needed)
    # _play_evaluation_game (Rely on Tournament)


# --- Main execution block (remains largely the same, points to new runner structure) ---
def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run YINSH training experiments')
    # Simplified arguments - type is no longer needed
    parser.add_argument('--config', required=True,
                        help='Name of the configuration section in config.py (e.g., "smoke", "value_head_config")')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
                        choices=['cuda', 'mps', 'cpu'],
                        help='Device to run on (default: best available)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--no-tracking', action='store_true', help='Disable experiment tracking')

    args = parser.parse_args()

    # --- Initialize Runner ---
    runner = ExperimentRunner(device=args.device, debug=args.debug, enable_tracking=not args.no_tracking)

    # --- Run the specified experiment ---
    # No need for experiment_type anymore
    runner.run_experiment(config_name=args.config)


if __name__ == "__main__":
    # Consider setting spawn method if using multiprocessing heavily, especially on macOS/Windows
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
    main()