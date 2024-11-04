"""Training supervisor for YINSH ML model."""

import logging
from pathlib import Path
import time
from typing import Optional, List, Tuple
import numpy as np
import os
import json

from ..network.wrapper import NetworkWrapper
from .self_play import SelfPlay
from .trainer import YinshTrainer
from ..utils.visualization import TrainingVisualizer
from ..utils.encoding import StateEncoder
from ..game.constants import Player, PieceType  # Added these imports
from ..game.game_state import GameState
from ..utils.metrics_manager import TrainingMetrics

class TrainingSupervisor:
    def __init__(self,
                 network: NetworkWrapper,
                 save_dir: str,
                 num_workers: int = 4,
                 mcts_simulations: int = 100,
                 mode: str = 'dev',  # Add mode parameter with default
                 device='cpu'
                 ):
        """
        Initialize the training supervisor.

        Args:
            network: NetworkWrapper instance
            save_dir: Directory to save models and logs
            num_workers: Number of parallel workers for self-play
            mcts_simulations: Number of MCTS simulations per move
        """
        self.network = network
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.mode = mode  # Store mode


        self.num_workers = num_workers
        self.self_play = SelfPlay(
            network=network,
            num_simulations=mcts_simulations,
            num_workers=num_workers
        )
        self.trainer = YinshTrainer(network, device=device)
        self.visualizer = TrainingVisualizer()
        self.state_encoder = StateEncoder()
        self.metrics = TrainingMetrics()

        # Setup logging
        self.logger = logging.getLogger("TrainingSupervisor")
        self.logger.setLevel(logging.INFO)

    def train_iteration(self, num_games: int = 100, epochs: int = 10):
        """Perform one training iteration."""
        iteration_dir = self.save_dir / f"iteration_{int(time.time())}"
        iteration_dir.mkdir(exist_ok=True)

        # Generate self-play games
        self.logger.info(f"Generating {num_games} self-play games...")
        start_time = time.time()
        games = self.self_play.generate_games(num_games=num_games)
        game_time = time.time() - start_time
        if len(games) > 0:
            self.logger.info(f"Generated {len(games)} games in {game_time:.1f}s "
                             f"({game_time / len(games):.1f}s per game)")
        else:
            self.logger.error("No games were generated.")
            raise ZeroDivisionError("No games were generated.")

        # Calculate game metrics
        game_lengths = [len(game[0]) for game in games]
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

        ring_mobilities = []
        for game in games:
            final_state = game[0][-1]
            mobility = self._calculate_ring_mobility(final_state)
            ring_mobilities.append(mobility)
        avg_ring_mobility = sum(ring_mobilities) / len(ring_mobilities) if ring_mobilities else 0

        # Calculate win/draw rates
        outcomes = [game[2] for game in games]
        white_wins = sum(1 for o in outcomes if o == 1)
        black_wins = sum(1 for o in outcomes if o == -1)
        draws = len(outcomes) - white_wins - black_wins
        win_rate = (white_wins + black_wins) / len(outcomes)
        draw_rate = draws / len(outcomes)

        # Log current metrics
        self.logger.info(f"Average game length: {avg_game_length:.1f} moves")
        self.logger.info(f"Average ring mobility: {avg_ring_mobility:.1f} moves per ring")
        self.logger.info(f"Game outcomes - White wins: {white_wins}, Black wins: {black_wins}, "
                         f"Draws: {draws}")
        self.logger.info(f"Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}")

        # Save games
        games_path = iteration_dir / "games.npy"
        self.self_play.export_games(games, str(games_path))

        # Add games to training experience
        for states, policies, outcome in games:
            self.trainer.add_game_experience(states, policies, outcome)

        # Train network
        self.logger.info("Training network...")
        training_start = time.time()
        for epoch in range(epochs):
            self.trainer.train_epoch(batch_size=256, batches_per_epoch=100)
        training_time = time.time() - training_start
        self.logger.info(f"Training completed in {training_time:.1f}s")

        # Get training losses
        policy_loss = np.mean(self.trainer.policy_losses) if self.trainer.policy_losses else 0
        value_loss = np.mean(self.trainer.value_losses) if self.trainer.value_losses else 0

        # Update metrics
        self.metrics.add_iteration_metrics(
            avg_game_length=avg_game_length,
            avg_ring_mobility=avg_ring_mobility,
            win_rate=win_rate,
            draw_rate=draw_rate,
            policy_loss=policy_loss,
            value_loss=value_loss
        )

        # Save metrics
        self._save_metrics(iteration_dir)

        # Check stability if in dev mode
        if self.mode == 'dev':
            stability_checks = self.metrics.assess_stability()
            if all(stability_checks.values()):
                self.logger.info("Training appears stable - consider moving to full mode")
            else:
                self.logger.info("Training not yet stable for full mode")

        # Generate visualizations using the metrics object's properties
        self.visualizer.plot_training_history(
            {
                'policy_losses': self.metrics.policy_losses,
                'value_losses': self.metrics.value_losses,
                'win_rates': self.metrics.win_rates,
                'draw_rates': self.metrics.draw_rates
            },
            save_path=str(iteration_dir / "training_history.png")
        )

        # Return metrics for this iteration
        return {
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'game_time': game_time,
            'training_time': training_time,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }

    def load_iteration(self, iteration_dir: str) -> Optional[int]:
        """Load a previous training iteration."""
        iteration_dir = Path(iteration_dir)

        # Load metrics
        metrics_path = iteration_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)

        # Find latest checkpoint
        checkpoints = list(iteration_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return self.trainer.load_checkpoint(str(latest_checkpoint))

    def _save_metrics(self, iteration_dir: Path) -> None:
        """Save training metrics to file."""
        # Get temperature metrics summary
        temp_summary = self.self_play.temp_metrics.get_summary() if hasattr(self.self_play, 'temp_metrics') else {}

        metrics_dict = {
            'iteration': len(self.metrics.game_lengths),
            'game_lengths': float(np.mean(self.metrics.game_lengths[-1])) if self.metrics.game_lengths else 0,
            'ring_mobility': float(np.mean(self.metrics.ring_mobility[-1])) if self.metrics.ring_mobility else 0,
            'win_rates': float(np.mean(self.metrics.win_rates[-1])) if self.metrics.win_rates else 0,
            'draw_rates': float(np.mean(self.metrics.draw_rates[-1])) if self.metrics.draw_rates else 0,
            'policy_losses': float(np.mean(self.metrics.policy_losses[-1])) if self.metrics.policy_losses else 0,
            'value_losses': float(np.mean(self.metrics.value_losses[-1])) if self.metrics.value_losses else 0,

            # Temperature annealing metrics
            'avg_temperature': float(np.mean([t for _, t in self.self_play.temp_metrics.move_temps[-100:]])) if hasattr(
                self.self_play, 'temp_metrics') else 0.0,
            'early_game_entropy': float(np.mean(temp_summary.get('early_game', {}).get('avg_entropy', 0.0))),
            'late_game_entropy': float(np.mean(temp_summary.get('late_game', {}).get('avg_entropy', 0.0))),
            'move_selection_confidence': float(
                np.mean(temp_summary.get('late_game', {}).get('avg_selected_prob', 0.0))),

            'timestamp': time.time()
        }

        metrics_path = iteration_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        self.logger.info(f"Saved metrics to {metrics_path}")

    def _calculate_ring_mobility(self, state_tensor: np.ndarray) -> float:
        """Calculate average number of valid moves available per ring."""
        game_state = self.state_encoder.decode_state(state_tensor)

        total_moves = 0
        num_rings = 0

        # For each player's rings
        for player in [Player.WHITE, Player.BLACK]:
            ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
            ring_positions = game_state.board.get_pieces_positions(ring_type)

            for pos in ring_positions:
                valid_moves = game_state.board.valid_move_positions(pos)
                total_moves += len(valid_moves)
                num_rings += 1

        return total_moves / num_rings if num_rings > 0 else 0

    def evaluate_model(self, num_games: int = 100) -> Tuple[float, float]:
        """Evaluate current model strength."""
        self.logger.info(f"Evaluating model over {num_games} games...")
        start_time = time.time()
        games = self.self_play.generate_games(num_games)
        eval_time = time.time() - start_time

        outcomes = [game[2] for game in games]
        white_wins = sum(1 for o in outcomes if o == 1)
        black_wins = sum(1 for o in outcomes if o == -1)
        draws = len(outcomes) - white_wins - black_wins

        win_rate = (white_wins + black_wins) / len(outcomes) if len(outcomes) > 0 else 0
        draw_rate = draws / len(outcomes) if len(outcomes) > 0 else 0

        self.logger.info(f"Evaluation complete in {eval_time:.1f}s")
        self.logger.info(f"Results - White wins: {white_wins}, Black wins: {black_wins}, "
                        f"Draws: {draws}")
        self.logger.info(f"Win Rate: {win_rate:.2f}, Draw Rate: {draw_rate:.2f}")

        return win_rate, draw_rate