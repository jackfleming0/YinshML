"""Experiment runner for YINSH training analysis."""

import json
import time
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


# Import YINSH specific modules
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player
from yinsh_ml.training.trainer import YinshTrainer
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import SelfPlay

from experiments.config import (
    get_experiment_config,
    RESULTS_SUBDIRS,
    LearningRateConfig,
    MCTSConfig,
    TemperatureConfig
)


class ExperimentRunner:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.logger = logging.getLogger("ExperimentRunner")

        # Initialize baseline model
        self.baseline_model = self._init_baseline_model()

    def _init_baseline_model(self) -> NetworkWrapper:
        """Initialize baseline model for comparisons."""
        try:
            model = NetworkWrapper(device=self.device)
            # Load your current best model weights
            model.load_model("models/training_dev2/checkpoint_iteration_10.pt")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load baseline model: {e}")
            raise

    def run_experiment(self, experiment_type: str, config_name: str) -> Dict:
        """Run a single experiment with specified configuration."""
        config = get_experiment_config(experiment_type, config_name)
        if config is None:
            raise ValueError(f"Invalid experiment configuration: {experiment_type}/{config_name}")

        self.logger.info(f"Starting experiment: {experiment_type}/{config_name}")

        try:
            if experiment_type == "learning_rate":
                metrics = self._run_learning_rate_experiment(config)
            elif experiment_type == "mcts":
                metrics = self._run_mcts_experiment(config)
            elif experiment_type == "temperature":
                metrics = self._run_temperature_experiment(config)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")

            # Save results
            self._save_results(experiment_type, config_name, config, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

    def _run_learning_rate_experiment(self, config: LearningRateConfig) -> Dict:
        """Run learning rate experiment."""
        metrics = {
            "policy_losses": [],
            "value_losses": [],
            "elo_changes": [],
            "game_lengths": [],
            "timestamps": []
        }

        # Initialize model and trainer
        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network, device=self.device)

        # Set learning rate configuration
        trainer.optimizer.param_groups[0]['lr'] = config.lr
        trainer.optimizer.param_groups[0]['weight_decay'] = config.weight_decay

        for iteration in range(config.num_iterations):
            start_time = time.time()

            # Generate self-play games
            self_play = SelfPlay(
                network=network,
                num_workers=4,
                num_simulations=100
            )

            games = self_play.generate_games(num_games=config.games_per_iteration)

            # Add games to trainer's experience
            for states, policies, outcome in games:
                trainer.add_game_experience(states, policies, outcome)

            # Train on games
            for _ in range(config.epochs_per_iteration):
                # Just call train_epoch without trying to capture returns
                trainer.train_epoch(
                    batch_size=config.batch_size,
                    batches_per_epoch=100
                )

            # Get the losses from trainer's stored metrics
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            value_loss = trainer.value_losses[-1] if trainer.value_losses else 0

            # Evaluate against baseline
            elo_change = self._evaluate_against_baseline(network)

            # Record metrics
            metrics["policy_losses"].append(float(policy_loss))
            metrics["value_losses"].append(float(value_loss))
            metrics["elo_changes"].append(float(elo_change))
            metrics["game_lengths"].append(float(np.mean([len(g[0]) for g in games])))
            metrics["timestamps"].append(time.time() - start_time)

            # Log progress
            self.logger.info(
                f"Iteration {iteration + 1}/{config.num_iterations}: "
                f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                f"ELO Change: {elo_change:+.1f}"
            )

            # Early stopping check
            if self._should_stop_early(metrics):
                self.logger.info("Early stopping triggered")
                break

        return metrics

    def _run_mcts_experiment(self, config: MCTSConfig) -> Dict:
        """Run MCTS simulation experiment."""
        metrics = {
            "policy_losses": [],
            "value_losses": [],
            "elo_changes": [],
            "game_lengths": [],
            "timestamps": []
        }

        # Initialize model and trainer
        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network, device=self.device)

        for iteration in range(config.num_iterations):
            start_time = time.time()

            # Generate games with specified MCTS depth
            self_play = SelfPlay(
                network=network,
                num_workers=4,
                num_simulations=config.num_simulations,
                initial_temp=1.0,  # Use your default values
                final_temp=0.2,
                annealing_steps=30
            )

            games = self_play.generate_games(num_games=config.games_per_iteration)

            # Add games to trainer's experience
            for states, policies, outcome in games:
                trainer.add_game_experience(states, policies, outcome)

            # Train for specified epochs
            for _ in range(config.epochs_per_iteration):
                trainer.train_epoch(
                    batch_size=256,  # Use default or config value
                    batches_per_epoch=100
                )

            # Get the latest losses
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            value_loss = trainer.value_losses[-1] if trainer.value_losses else 0

            # Evaluate against baseline
            elo_change = self._evaluate_against_baseline(network)

            # Record metrics
            metrics["policy_losses"].append(float(policy_loss))
            metrics["value_losses"].append(float(value_loss))
            metrics["elo_changes"].append(float(elo_change))
            metrics["game_lengths"].append(float(np.mean([len(g[0]) for g in games])))
            metrics["timestamps"].append(time.time() - start_time)

            self.logger.info(
                f"Iteration {iteration + 1}/{config.num_iterations}: "
                f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                f"ELO Change: {elo_change:+.1f}"
            )

        return metrics

    def _run_temperature_experiment(self, config: TemperatureConfig) -> Dict:
        """Run temperature annealing experiment."""
        metrics = {
            "move_entropies": [],
            "win_rates": [],
            "game_lengths": [],
            "elo_changes": [],
            "timestamps": []
        }

        # Initialize model
        network = NetworkWrapper(device=self.device)
        trainer = YinshTrainer(network, device=self.device)

        for iteration in range(config.num_iterations):
            start_time = time.time()

            # Configure self-play with temperature schedule
            self_play = SelfPlay(
                network=network,
                num_workers=4,
                num_simulations=config.mcts_simulations,
                initial_temp=config.initial_temp,
                final_temp=config.final_temp,
                annealing_steps=config.annealing_steps
            )

            games = self_play.generate_games(num_games=config.games_per_iteration)

            # Train on games
            for _ in range(config.epochs_per_iteration):
                trainer.train_epoch(games=games)

            # Evaluate against baseline
            elo_change = self._evaluate_against_baseline(network)

            # Calculate metrics
            move_entropy = self._calculate_move_entropy(games)
            win_rate = self._calculate_win_rate(games)
            avg_game_length = np.mean([len(g[0]) for g in games])

            # Record metrics
            metrics["move_entropies"].append(float(move_entropy))
            metrics["win_rates"].append(float(win_rate))
            metrics["game_lengths"].append(float(avg_game_length))
            metrics["elo_changes"].append(float(elo_change))
            metrics["timestamps"].append(time.time() - start_time)

            # Log progress
            self.logger.info(
                f"Iteration {iteration + 1}/{config.num_iterations}: "
                f"Move Entropy: {move_entropy:.3f}, Win Rate: {win_rate:.2%}, "
                f"Game Length: {avg_game_length:.1f}, ELO Change: {elo_change:+.1f}"
            )

        return metrics

    def _evaluate_against_baseline(self, network: NetworkWrapper, num_games: int = 20) -> float:
        """Evaluate network against baseline model."""
        wins = 0
        total_games = num_games * 2  # Play as both colors

        for game_idx in range(total_games):
            # Alternate colors
            test_is_white = game_idx % 2 == 0
            white_model = network if test_is_white else self.baseline_model
            black_model = self.baseline_model if test_is_white else network

            winner = self._play_evaluation_game(white_model, black_model)

            if winner is not None:
                if (test_is_white and winner == 1) or (not test_is_white and winner == -1):
                    wins += 1

        win_rate = wins / total_games
        return self._win_rate_to_elo(win_rate)

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