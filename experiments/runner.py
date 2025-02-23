"""Experiment runner for YINSH training analysis."""

import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Optional
import torch

from yinsh_ml.utils.mcts_metrics import MCTSMetrics
from yinsh_ml.game.moves import Move
import numpy as np

# Import YINSH specific modules
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player
from yinsh_ml.training.trainer import YinshTrainer
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import SelfPlay
from yinsh_ml.utils.tournament import ModelTournament
from yinsh_ml.utils.metrics_logger import MetricsLogger
from yinsh_ml.utils.value_head_metrics import ValueHeadMetrics


from experiments.config import (
    get_experiment_config,
    RESULTS_DIR,
    RESULTS_SUBDIRS,
    LearningRateConfig,
    MCTSConfig,
    TemperatureConfig,
    CombinedConfig
)


class ExperimentRunner:
    def __init__(self, device: str = 'cuda', debug: bool = False):
        self.device = device
        self.logger = logging.getLogger("ExperimentRunner")

        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            save_dir=Path("results"),
            debug=debug
        )

        # Initialize value head metrics
        self.value_head_metrics = ValueHeadMetrics()

        # Set logging levels
        loggers = [
            logging.getLogger("ExperimentRunner"),
            logging.getLogger("SelfPlay"),
            logging.getLogger("YinshTrainer"),
            logging.getLogger("ModelTournament")
        ]

        level = logging.DEBUG if debug else logging.DEBUG
        for logger in loggers:
            logger.setLevel(level)

        # Add checkpoint directory
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = RESULTS_DIR # Now it exists


        # Initialize baseline model
        self.baseline_model = self._init_baseline_model()

        # Initialize self_play attribute here
        self.self_play = None


    def _init_baseline_model(self) -> NetworkWrapper:
        """Initialize baseline model for comparisons."""
        try:
            model = NetworkWrapper(device=self.device)
            # Comment out model loading for testing new architecture
            # model.load_model("models/training_dev2/checkpoint_iteration_10.pt")
            self.logger.info("Initialized fresh baseline model for testing")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize baseline model: {e}")
            raise

    def run_experiment(self, experiment_type: str, config_name: str) -> Dict:
        """Run a single experiment with specified configuration."""
        config = get_experiment_config(experiment_type, config_name)
        if config is None:
            raise ValueError(f"Invalid experiment configuration: {experiment_type}/{config_name}")

        self.logger.info(f"Starting experiment: {experiment_type}/{config_name}")

        # Create experiment-specific metrics directory
        metrics_dir = Path("results") / experiment_type / config_name / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics logger with experiment-specific path
        self.metrics_logger = MetricsLogger(
            save_dir=metrics_dir,
            debug=False
        )

        # Initialize tournament manager with correct path
        self.tournament_manager = ModelTournament(
            training_dir=self.checkpoint_dir / experiment_type / config_name,
            device=self.device,
            games_per_match=10,
            temperature=0.1
        )

        # Create experiment-specific checkpoint directory
        experiment_dir = self.checkpoint_dir / experiment_type / config_name
        experiment_dir.mkdir(parents=True, exist_ok=True)


        try:
            if experiment_type == "learning_rate":
                metrics = self._run_learning_rate_experiment(config, config_name)
            elif experiment_type == "mcts":
                metrics = self._run_mcts_experiment(config, config_name)
            elif experiment_type == "temperature":
                metrics = self._run_temperature_experiment(config, config_name)
            elif experiment_type == "combined":  # Add this case
                metrics = self._run_combined_experiment(config, config_name)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")

            # Save results
            results_dir = RESULTS_SUBDIRS[experiment_type]
            results_file = results_dir / f"{config_name}.json"
            result_data = {
                "config": vars(config),
                "metrics": metrics,
                "timestamp": time.time()
            }

            with open(results_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            self.logger.info(f"Results saved to {results_file}")

            # Save MCTS metrics if they exist
            # Store the SelfPlay instance during learning rate experiments
            if hasattr(self, '_current_selfplay') and hasattr(self._current_selfplay.mcts, 'metrics'):
                mcts_metrics_path = results_dir / f"{config_name}_mcts_metrics.json"
                self._current_selfplay.mcts.metrics.save(str(mcts_metrics_path))
                self.logger.info(f"MCTS metrics saved to {mcts_metrics_path}")

            return metrics

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

    def _save_checkpoint(self, network: NetworkWrapper, experiment_type: str,
                        config_name: str, iteration: int):
        """Save model checkpoint."""
        checkpoint_path = (self.checkpoint_dir / experiment_type / config_name /
                         f"checkpoint_iteration_{iteration}.pt")
        network.save_model(str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, experiment_type: str, config_name: str,
                        iteration: int) -> Optional[NetworkWrapper]:
        """Load model from checkpoint."""
        checkpoint_path = (self.checkpoint_dir / experiment_type / config_name /
                         f"checkpoint_iteration_{iteration}.pt")
        if checkpoint_path.exists():
            network = NetworkWrapper(device=self.device)
            network.load_model(str(checkpoint_path))
            return network
        return None

    def _run_learning_rate_experiment(self, config: LearningRateConfig, config_name: str) -> Dict:
        """Run learning rate experiment."""
        metrics = {
            "policy_losses": [],
            "value_losses": [],
            "elo_changes": [],
            "tournament_elo": [],
            "game_lengths": [],
            "timestamps": [],
            "value_accuracies": [],
            "move_accuracies": [],
            "policy_entropy": []
        }

        print(f"\nStarting learning rate experiment with config: {config}")

        # Initialize model and trainer
        print("Initializing model and trainer...")

        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network, device=self.device)

        # Set learning rate configuration
        trainer.policy_optimizer.param_groups[0]['lr'] = config.lr
        trainer.value_optimizer.param_groups[0]['lr'] = config.lr * 0.1  # Value head uses lower lr

        # Set weight decay
        trainer.policy_optimizer.param_groups[0]['weight_decay'] = config.weight_decay
        trainer.value_optimizer.param_groups[0][
            'weight_decay'] = config.weight_decay * 10  # Higher regularization for value head

        for iteration in range(config.num_iterations):
            # Tell metrics logger we're starting a new iteration
            self.metrics_logger.start_iteration(iteration)
            start_time = time.time()
            print(f"\nIteration {iteration + 1}/{config.num_iterations}")

            # Generate self-play games
            print(f"Generating {config.games_per_iteration} self-play games...")

            self._current_selfplay = SelfPlay(
                network=network,
            #    num_workers=self._get_optimal_workers(), # 4 on local, will use more cores on cloud compute
                num_simulations=100
            )
            self._current_selfplay.current_iteration = iteration

            games = self._current_selfplay.generate_games(num_games=config.games_per_iteration)

            # Rest of the method remains the same...
            # Add games to trainer's experience
            print("Adding games to trainer's experience...")

            for states, policies, outcome in games:
                trainer.add_game_experience(states, policies, outcome)

            # Train on games
            print(f"Training for {config.epochs_per_iteration} epochs...")

            for _ in range(config.epochs_per_iteration):
                actual_batches = 10 if config.batches_per_epoch > 10 else config.batches_per_epoch
                epoch_stats = trainer.train_epoch(
                    batch_size=config.batch_size,
                    batches_per_epoch=actual_batches
                )

            print("Training completed")

            # Get the losses from trainer's stored metrics
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            value_loss = trainer.value_losses[-1] if trainer.value_losses else 0

            # Evaluate against baseline
            print("Evaluating against baseline...")
            elo_change = self._evaluate_against_baseline(network, quick_eval=True)

            # Save checkpoint for this iteration
            self._save_checkpoint(network, "learning_rate", config_name, iteration)

            # Run tournament evaluation if we have previous iterations
            tournament_elo = 0.0  # Default if no tournament run
            if iteration > 0:
                print("Running tournament evaluation...")
                self.tournament_manager.run_tournament(
                    experiment_type="learning_rate",
                    config_name=config_name,
                    current_iteration=iteration
                )
                tournament_stats = self.tournament_manager.get_model_performance(f"iteration_{iteration}")
                tournament_elo = tournament_stats['current_rating']
                print(f"Tournament ELO: {tournament_elo:+.1f}")


            # Record metrics
            metrics["policy_losses"].append(float(epoch_stats['policy_loss']))
            metrics["value_losses"].append(float(epoch_stats['value_loss']))
            metrics["value_accuracies"].append(float(epoch_stats['value_accuracy']))
            metrics["move_accuracies"].append(epoch_stats['move_accuracies'])
            metrics["policy_entropy"].append(float(epoch_stats.get('policy_entropy', 0.0)))
            metrics["elo_changes"].append(float(elo_change))
            metrics["game_lengths"].append(float(np.mean([len(g[0]) for g in games])))
            metrics["timestamps"].append(time.time() - start_time)

            # Log progress
            self.logger.info(
                f"Iteration {iteration + 1}/{config.num_iterations}: "
                f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                f"ELO Change: {elo_change:+.1f}"
            )

        # After training completes
        # Save value head metrics
        value_metrics_path = RESULTS_SUBDIRS[experiment_type] / f"{config_name}_value_metrics.json"
        value_metrics_plots = RESULTS_SUBDIRS[experiment_type] / f"{config_name}_value_diagnostics.png"

        trainer.value_metrics.plot_diagnostics(save_path=str(value_metrics_plots))

        with open(value_metrics_path, 'w') as f:
            json.dump(trainer.value_metrics.generate_report(), f, indent=2)

        self.logger.info(f"Value head metrics saved to {value_metrics_path}")
        self.logger.info(f"Value head diagnostics plotted to {value_metrics_plots}")


        return metrics

    def _run_mcts_experiment(self, config: MCTSConfig, config_name: str) -> Dict:
        """Run MCTS simulation experiment."""
        metrics = {
            "policy_losses": [],
            "value_losses": [],
            "elo_changes": [],
            "tournament_elo": [],
            "game_lengths": [],
            "timestamps": [],
            "value_accuracies": [],
            "move_accuracies": [],
            "policy_entropy": []
        }
        # Initialize model and trainer
        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network, device=self.device)

        for iteration in range(config.num_iterations):
            # Tell metrics logger we're starting a new iteration
            self.metrics_logger.start_iteration(iteration)
            start_time = time.time()

            # Generate games with specified MCTS depth
            self_play = SelfPlay(
                network=network,
            #     num_workers=4,
                num_simulations=config.num_simulations,
                initial_temp=1.0,  # Use your default values
                final_temp=0.2,
                annealing_steps=30
            )
            self_play.current_iteration = iteration

            games = self_play.generate_games(num_games=config.games_per_iteration)

            # Add games to trainer's experience
            for states, policies, outcome in games:
                trainer.add_game_experience(states, policies, outcome)

            # Train for specified epochs
            for _ in range(config.epochs_per_iteration):
                actual_batches = 10 if config.batches_per_epoch > 10 else config.batches_per_epoch
                epoch_stats = trainer.train_epoch(
                    batch_size=config.batch_size,
                    batches_per_epoch=actual_batches
                )

            # Get the latest losses
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            value_loss = trainer.value_losses[-1] if trainer.value_losses else 0

            # Evaluate against baseline
            elo_change = self._evaluate_against_baseline(network)

            # Save checkpoint for this iteration
            self._save_checkpoint(network, "mcts", config_name, iteration)

            # Run tournament evaluation if we have previous iterations
            tournament_elo = 0.0  # Default if no tournament run
            if iteration > 0:
                print("Running tournament evaluation...")
                self.tournament_manager.run_tournament(
                    experiment_type="mcts",
                    config_name=config_name,
                    current_iteration=iteration
                )
                tournament_stats = self.tournament_manager.get_model_performance(f"iteration_{iteration}")
                tournament_elo = tournament_stats['current_rating']
                print(f"Tournament ELO: {tournament_elo:+.1f}")


            # Record metrics
            metrics["policy_losses"].append(float(epoch_stats['policy_loss']))
            metrics["value_losses"].append(float(epoch_stats['value_loss']))
            metrics["value_accuracies"].append(float(epoch_stats['value_accuracy']))
            metrics["move_accuracies"].append(epoch_stats['move_accuracies'])
            metrics["policy_entropy"].append(float(epoch_stats.get('policy_entropy', 0.0)))
            metrics["elo_changes"].append(float(elo_change))
            metrics["game_lengths"].append(float(np.mean([len(g[0]) for g in games])))
            metrics["timestamps"].append(time.time() - start_time)

            self.logger.info(
                f"Iteration {iteration + 1}/{config.num_iterations}: "
                f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                f"ELO Change: {elo_change:+.1f}"
            )

        return metrics

    def _run_temperature_experiment(self, config: TemperatureConfig, config_name: str) -> Dict:
        """Run temperature annealing experiment."""
        metrics = {
            "policy_losses": [],
            "value_losses": [],
            "elo_changes": [],
            "tournament_elo": [],
            "game_lengths": [],
            "timestamps": [],
            "value_accuracies": [],
            "move_accuracies": [],
            "policy_entropy": [],
            "move_entropies": []
        }

        # Initialize model and trainer
        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network, device=self.device)

        for iteration in range(config.num_iterations):
            # Tell metrics logger we're starting a new iteration
            self.metrics_logger.start_iteration(iteration)
            start_time = time.time()

            # Generate self-play games with temperature configuration
            self_play = SelfPlay(
                network=network,
            #     num_workers=4,
                num_simulations=config.mcts_simulations,
                initial_temp=config.initial_temp,
                final_temp=config.final_temp,
                annealing_steps=config.annealing_steps
            )
            self_play.current_iteration = iteration


            games = self_play.generate_games(num_games=config.games_per_iteration)

            # Add games to trainer's experience
            for states, policies, outcome in games:
                trainer.add_game_experience(states, policies, outcome)

            # Train for specified epochs
            for _ in range(config.epochs_per_iteration):
                actual_batches = 10 if config.batches_per_epoch > 10 else config.batches_per_epoch
                epoch_stats = trainer.train_epoch(
                    batch_size=config.batch_size,
                    batches_per_epoch=actual_batches
                )

            # Get the latest losses
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            value_loss = trainer.value_losses[-1] if trainer.value_losses else 0

            # Calculate average move entropy for this iteration
            move_entropy = np.mean([
                -np.sum(p * np.log(p + 1e-8))
                for game in games
                for p in game[1]  # game[1] contains policy vectors
            ])

            # Evaluate against baseline
            elo_change = self._evaluate_against_baseline(network)

            # Save checkpoint for this iteration
            self._save_checkpoint(network, "temperature", config_name, iteration)

            # Run tournament evaluation if we have previous iterations
            tournament_elo = 0.0  # Default if no tournament run
            if iteration > 0:
                print("Running tournament evaluation...")
                self.tournament_manager.run_tournament(
                    experiment_type="temperature",
                    config_name=config_name,
                    current_iteration=iteration
                )
                tournament_stats = self.tournament_manager.get_model_performance(f"iteration_{iteration}")
                tournament_elo = tournament_stats['current_rating']
                print(f"Tournament ELO: {tournament_elo:+.1f}")


            # Record metrics
            metrics["policy_losses"].append(float(epoch_stats['policy_loss']))
            metrics["value_losses"].append(float(epoch_stats['value_loss']))
            metrics["value_accuracies"].append(float(epoch_stats['value_accuracy']))
            metrics["move_accuracies"].append(epoch_stats['move_accuracies'])
            metrics["policy_entropy"].append(float(epoch_stats.get('policy_entropy', 0.0)))
            metrics["elo_changes"].append(float(elo_change))
            metrics["game_lengths"].append(float(np.mean([len(g[0]) for g in games])))
            metrics["timestamps"].append(time.time() - start_time)

            # Log progress
            self.logger.info(
                f"Iteration {iteration + 1}/{config.num_iterations}: "
                f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                f"ELO Change: {elo_change:+.1f}, Move Entropy: {move_entropy:.3f}"
            )

            # Early stopping check
            if self._should_stop_early(metrics):
                self.logger.info("Early stopping triggered")
                break

        return metrics

    def _run_combined_experiment(self, config: CombinedConfig, config_name: str) -> Dict:
        """Run combined experiment using multiple parameter types."""
        metrics = {
            "policy_losses": [],
            "value_losses": [],
            "elo_changes": [],
            "tournament_elo": [],
            "game_lengths": [],
            "timestamps": [],
            "value_accuracies": [],
            "move_accuracies": [],
            "policy_entropy": []
        }

        experiment_dir = self.checkpoint_dir / "combined_experiments" / config_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        replay_buffer_file = experiment_dir / "replay_buffer.pkl"

        print(f"Starting combined experiment with config: {config}")
        total_start_time = time.time()

        # Initialize model and trainer
        print("Initializing model and trainer...")
        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network,
                               device=self.device,
                               l2_reg=0.0,
                               metrics_logger=self.metrics_logger,
                               value_head_lr_factor=config.value_head_lr_factor,  # Pass the factor
                               value_loss_weights=config.value_loss_weights,  # Pass the weights
                               replay_buffer_path=str(replay_buffer_file))

        # Set learning rate configuration for both optimizers
        trainer.policy_optimizer.param_groups[0]['lr'] = config.lr
        trainer.value_optimizer.param_groups[0][
            'lr'] = config.lr * config.value_head_lr_factor  # Value head uses higher lr

        # Set weight decay for both optimizers
        trainer.policy_optimizer.param_groups[0]['weight_decay'] = config.weight_decay
        trainer.value_optimizer.param_groups[0][
            'weight_decay'] = config.weight_decay * 10  # Higher regularization for value head

        for iteration in range(config.num_iterations):
            # Create experiment-specific checkpoint directory
            iteration_dir = self.checkpoint_dir / config_name / f"iteration_{iteration}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            # Create experiment-specific metrics directory
            metrics_dir = self.results_dir / "combined" / config_name / f"iteration_{iteration}" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            # Tell metrics logger we're starting a new iteration
            self.metrics_logger.start_iteration(iteration)
            iter_start_time = time.time()
            print(f"\nIteration {iteration + 1}/{config.num_iterations}")

            # Generate self-play games with MCTS parameters
            print(f"Generating {config.games_per_iteration} self-play games...")
            game_start_time = time.time()
            self_play = SelfPlay(
                network=network,
                metrics_logger=self.metrics_logger,
                num_simulations=config.num_simulations,
                initial_temp=config.initial_temp,
                final_temp=config.final_temp,
                c_puct=config.c_puct,
                max_depth=config.max_depth
            )
            self_play.current_iteration = iteration

            games = self_play.generate_games(num_games=config.games_per_iteration)
            print(f"Games generated in {time.time() - game_start_time:.2f} seconds")

            # Save Value Head Metrics and plots
            value_metrics_path = metrics_dir / "value_metrics.json"
            self.value_head_metrics.save(str(value_metrics_path))
            self.logger.info(f"Value head metrics saved to {value_metrics_path}")
            value_metrics_plots = metrics_dir / "value_head_diagnostics.png"
            self.value_head_metrics.plot_diagnostics(save_path=str(value_metrics_plots))

            # Save MCTS Metrics if they exist
            if hasattr(self_play.mcts, 'metrics') and self_play.mcts.metrics:
                mcts_metrics_path = iteration_dir / "mcts_metrics.json"
                self_play.mcts.metrics.save(str(mcts_metrics_path))
                self.logger.info(f"MCTS metrics saved to {mcts_metrics_path}")

            # Add games to trainer's experience
            print("Adding games to trainer's experience...")
            exp_start_time = time.time()
            for states, policies, outcome, game_history in games:
                # Debug: print lengths of states and game_history
                print(
                    f"[DEBUG] Adding game experience: len(states)={len(states)}, len(game_history)={len(game_history)}")

                # Check if game_history is available and has a 'state' field in its last entry
                if game_history and isinstance(game_history, list) and len(game_history) > 0 and 'state' in \
                        game_history[-1]:
                    candidate = game_history[-1]['state']
                    # If candidate is already a GameState, use it directly; otherwise, decode it.
                    if isinstance(candidate, GameState):
                        final_state = candidate
                        print(
                            f"[DEBUG] Using GameState from game_history: W={final_state.white_score}, B={final_state.black_score}, Terminal={final_state.is_terminal()}")
                    else:
                        final_state = trainer.state_encoder.decode_state(candidate)
                        print(
                            f"[DEBUG] Decoded terminal state from game_history: W={final_state.white_score}, B={final_state.black_score}, Terminal={final_state.is_terminal()}")
                else:
                    final_state = self.get_terminal_state(states, trainer.state_encoder)
                    print(
                        f"[DEBUG] Using scanned terminal state: W={final_state.white_score}, B={final_state.black_score}, Terminal={final_state.is_terminal()}")

                final_white_score = final_state.white_score
                final_black_score = final_state.black_score
                trainer.add_game_experience(states, policies, (final_white_score, final_black_score))

            print(f"Experience added in {time.time() - exp_start_time:.2f} seconds")

            # Train on games
            print(f"Training for {config.epochs_per_iteration} epochs...")
            train_start_time = time.time()
            for _ in range(config.epochs_per_iteration):
                actual_batches = 10 if config.batches_per_epoch > 10 else config.batches_per_epoch
                ring_weight = 1.0 + iteration * 0.0125  # Schedule: ring placement weight increases as iterations progress.
                epoch_stats = trainer.train_epoch(
                    batch_size=config.batch_size,
                    batches_per_epoch=actual_batches,
                    ring_placement_weight=ring_weight
                )
            print(f"Training completed in {time.time() - train_start_time:.2f} seconds")

            # Get the latest losses
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            value_loss = trainer.value_losses[-1] if trainer.value_losses else 0

            # Evaluate against baseline
            print("Evaluating against baseline...")
            eval_start_time = time.time()
            elo_change = self._evaluate_against_baseline(network, quick_eval=True)
            print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

            # Save checkpoint for this iteration
            self._save_checkpoint(network, "combined", config_name, iteration)

            # Save replay buffer to disk
            trainer.experience.save_buffer(str(replay_buffer_file))

            # Run tournament evaluation if applicable
            tournament_elo = 0.0  # Default if no tournament run
            if iteration > 0:
                print("Running tournament evaluation...")
                self.tournament_manager.run_tournament(current_iteration=iteration)
                tournament_stats = self.tournament_manager.get_model_performance(f"iteration_{iteration}")
                tournament_elo = tournament_stats['current_rating']
                print(f"Tournament ELO: {tournament_elo:+.1f}")

            # Record metrics
            metrics["policy_losses"].append(float(epoch_stats['policy_loss']))
            metrics["value_losses"].append(float(epoch_stats['value_loss']))
            metrics["value_accuracies"].append(float(epoch_stats['value_accuracy']))
            metrics["move_accuracies"].append(epoch_stats['move_accuracies'])
            metrics["policy_entropy"].append(float(epoch_stats.get('policy_entropy', 0.0)))
            metrics["elo_changes"].append(float(elo_change))
            metrics["game_lengths"].append(float(np.mean([len(g[0]) for g in games])))
            metrics["timestamps"].append(time.time() - iter_start_time)

            # Save and plot metrics for this iteration
            self.metrics_logger.summarize_iteration()
            self.metrics_logger.plot_current_metrics()
            self.metrics_logger.save_iteration()

            print(f"Iteration completed in {time.time() - iter_start_time:.2f} seconds")
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, ELO Change: {elo_change:+.1f}")

        total_time = time.time() - total_start_time
        print(f"\nExperiment completed in {total_time / 60:.2f} minutes")
        return metrics

    def _evaluate_against_baseline(self, network: NetworkWrapper,
                                   num_games: int = 20,
                                   quick_eval: bool = False) -> float:
        """Evaluate network against baseline model."""
        if quick_eval:
            num_games = 2  # Reduce games for quick testing

        wins = 0
        total_games = num_games * 2  # Play as both colors

        for game_idx in range(total_games):
            test_is_white = game_idx % 2 == 0
            white_model = network if test_is_white else self.baseline_model
            black_model = self.baseline_model if test_is_white else network

            winner = self._play_evaluation_game(white_model, black_model)

            if winner is not None:  # Should always be true for YINSH
                if (test_is_white and winner == 1) or (not test_is_white and winner == -1):
                    wins += 1

        # Simple win rate calculation - no draws possible
        win_rate = wins / total_games
        return self._win_rate_to_elo(win_rate)

    def get_terminal_state(self, states, state_encoder):
        """Return the first terminal state found when scanning states in reverse.
           If none are terminal, return the last state.
        """
        for s in reversed(states):
            decoded = state_encoder.decode_state(s)
            if decoded.is_terminal():
                return decoded
        return state_encoder.decode_state(states[-1])

    def _save_results(self, experiment_type: str, config_name: str,
                      config: object, metrics: Dict) -> None:
        """Save experiment results."""
        results_dir = RESULTS_SUBDIRS[experiment_type]

        result_data = {
            "config": vars(config),
            "metrics": metrics,
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        }

        output_file = results_dir / f"{config_name}.json"
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)

        self.logger.info(f"Results saved to {output_file}")

    def _should_stop_early(self, metrics: Dict) -> bool:
        """Check if experiment should be stopped early."""
        if len(metrics["policy_losses"]) < 3:
            return False

        # Check if losses are consistently increasing
        recent_policy = metrics["policy_losses"][-3:]
        recent_value = metrics["value_losses"][-3:]

        policy_increasing = all(x < y for x, y in zip(recent_policy, recent_policy[1:]))
        value_increasing = all(x < y for x, y in zip(recent_value, recent_value[1:]))

        return policy_increasing and value_increasing

    def _calculate_win_rate(self, games: List) -> float:
        """Calculate win rate from games."""
        wins = sum(1 for game in games if game[2] == 1)  # Assuming game[2] is outcome
        return wins / len(games)

    def _calculate_move_entropy(self, games: List) -> float:
        """Calculate average move entropy."""
        entropies = []
        for game in games:
            for policy in game[1]:  # game[1] should be move policies
                entropy = -np.sum(policy * np.log(policy + 1e-8))
                entropies.append(entropy)
        return np.mean(entropies)

    def _calculate_move_accuracy(self, pred_moves: np.ndarray,
                                 actual_moves: List[Move],
                                 game_state: GameState) -> Dict[str, float]:
        """
        Calculate move prediction accuracy metrics.

        Args:
            pred_moves: Network's move probability distribution
            actual_moves: List of moves actually made in the game
            game_state: Current game state for move validation

        Returns:
            Dictionary containing various accuracy metrics
        """
        try:
            metrics = {}

            # Convert actual moves to indices for comparison
            actual_indices = [
                self.network.state_encoder.move_to_index(move)
                for move in actual_moves
            ]

            # Get top-k predictions for different k values
            for k in [1, 3, 5]:
                top_k_indices = np.argpartition(pred_moves, -k)[-k:]
                # Calculate how often actual move is in top k predictions
                accuracy = sum(
                    any(actual_idx in top_k_indices for actual_idx in actual_indices)
                ) / len(actual_moves)
                metrics[f'top_{k}_accuracy'] = accuracy

            # Calculate average predicted probability of chosen moves
            move_confidences = [
                pred_moves[idx] for idx in actual_indices
                if idx < len(pred_moves)  # Bounds check
            ]
            if move_confidences:
                metrics['mean_move_confidence'] = float(np.mean(move_confidences))
                metrics['min_move_confidence'] = float(np.min(move_confidences))
                metrics['max_move_confidence'] = float(np.max(move_confidences))

            # Policy entropy to measure exploration
            valid_probs = pred_moves[pred_moves > 0]  # Avoid log(0)
            if len(valid_probs) > 0:
                entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
                metrics['policy_entropy'] = float(entropy)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating move accuracy: {e}")
            return {
                'top_1_accuracy': 0.0,
                'top_3_accuracy': 0.0,
                'top_5_accuracy': 0.0,
                'mean_move_confidence': 0.0,
                'policy_entropy': 0.0
            }

    def _track_metrics(self, metrics: Dict):
        """
        Track and log enhanced training metrics.

        Args:
            metrics: Dictionary of metrics to track
        """
        try:
            # Create a metrics summary
            summary = {
                'timestamp': time.time(),
                'iteration': len(self.tracked_metrics) + 1
            }

            # Track basic metrics
            if 'policy_loss' in metrics:
                summary['policy_loss'] = float(metrics['policy_loss'])
            if 'value_loss' in metrics:
                summary['value_loss'] = float(metrics['value_loss'])

            # Track accuracy metrics
            if 'value_accuracy' in metrics:
                summary['value_accuracy'] = float(metrics['value_accuracy'])
            if 'move_accuracies' in metrics:
                for k, v in metrics['move_accuracies'].items():
                    summary[f'move_{k}'] = float(v)

            # Track ELO if available
            if 'elo_change' in metrics:
                summary['elo_change'] = float(metrics['elo_change'])

            # Track learning parameters
            current_lr = self.optimizer.param_groups[0]['lr']
            summary['learning_rate'] = float(current_lr)

            # Log important metrics
            self.logger.info(
                f"Iteration {summary['iteration']} - "
                f"Policy Loss: {summary.get('policy_loss', 'N/A'):.4f}, "
                f"Value Loss: {summary.get('value_loss', 'N/A'):.4f}, "
                f"Value Acc: {summary.get('value_accuracy', 'N/A'):.2%}, "
                f"Top-1 Move Acc: {summary.get('move_top_1_accuracy', 'N/A'):.2%}, "
                f"LR: {current_lr:.2e}"
            )

            # Store metrics
            self.tracked_metrics.append(summary)

            # Optionally save to file
            if hasattr(self, 'metrics_file'):
                with open(self.metrics_file, 'a') as f:
                    json.dump(summary, f)
                    f.write('\n')

        except Exception as e:
            self.logger.error(f"Error tracking metrics: {e}")

    def _calculate_average_search_time(self, games: List) -> float:
        """Calculate average MCTS search time."""
        try:
            # Extract search times from game data
            # Assuming each game tuple contains:
            # - game[0]: state sequence
            # - game[1]: policy sequence
            # - game[2]: outcome
            # - game[3]: metadata dictionary with timing information
            total_time = 0
            total_moves = 0

            for game in games:
                if len(game) >= 4 and isinstance(game[3], dict):
                    metadata = game[3]
                    if 'search_times' in metadata:
                        search_times = metadata['search_times']
                        total_time += sum(search_times)
                        total_moves += len(search_times)

            if total_moves == 0:
                return 0.0

            # Return average time in seconds
            return total_time / total_moves

        except Exception as e:
            self.logger.error(f"Error calculating search times: {e}")
            return 0.0

    def _win_rate_to_elo(self, win_rate: float) -> float:
        """Convert win rate to ELO difference."""
        # Add stricter bounds and handling
        if win_rate == 0:
            return -400  # Cap minimum
        if win_rate == 1:
            return 400  # Cap maximum
        win_rate = max(0.001, min(0.999, win_rate))
        return -400 * np.log10(1 / win_rate - 1)

    def _play_evaluation_game(self,
                              white_model: NetworkWrapper,
                              black_model: NetworkWrapper) -> Optional[int]:
        """Play a single evaluation game between two models."""
        game_state = GameState()
        move_count = 0
        max_moves = 500  # Prevent infinite games

        try:
            while not game_state.is_terminal() and move_count < max_moves:
                # Get current model
                current_model = white_model if game_state.current_player == Player.WHITE else black_model

                # Get valid moves
                valid_moves = game_state.get_valid_moves()
                if not valid_moves:
                    break

                # Get model's move choice with lower temperature for evaluation
                state_tensor = current_model.state_encoder.encode_state(game_state)
                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
                move_probs, _ = current_model.predict(state_tensor)

                # Use lower temperature (0.1) for evaluation games
                selected_move = current_model.select_move(
                    move_probs=move_probs,
                    valid_moves=valid_moves,
                    temperature=0.1  # Lower temperature for more deterministic evaluation
                )

                # Make move
                success = game_state.make_move(selected_move)
                if not success:
                    self.logger.error(
                        f"Invalid move by {'White' if game_state.current_player == Player.WHITE else 'Black'}"
                    )
                    break

                # Only increment move count for main game moves
                if game_state.phase == GamePhase.MAIN_GAME:
                    move_count += 1

            # Get winner
            if move_count >= max_moves:
                return None  # Draw if game too long

            winner = game_state.get_winner()
            if winner == Player.WHITE:
                return 1
            elif winner == Player.BLACK:
                return -1
            return None

        except Exception as e:
            self.logger.error(f"Error in evaluation game: {e}")
            return None


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run YINSH training experiments')
    parser.add_argument('--type', required=True,
                        choices=['learning_rate', 'mcts', 'temperature'],
                        help='Type of experiment to run')
    parser.add_argument('--config', required=True,
                        help='Name of configuration to use')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')

    args = parser.parse_args()

    runner = ExperimentRunner(device=args.device)
    runner.run_experiment(args.type, args.config)


if __name__ == "__main__":
    main()