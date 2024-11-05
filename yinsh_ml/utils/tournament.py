"""Tournament system for evaluating YINSH models within a training run."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from datetime import datetime
import json

from ..network.wrapper import NetworkWrapper
from ..game.game_state import GameState
from ..game.constants import Player
from .elo_manager import EloTracker, MatchResult


class ModelTournament:
    """Manages tournaments between YINSH models from the current training run."""

    def __init__(self,
                 training_dir: Path,
                 device: str = 'cpu',
                 games_per_match: int = 10,
                 temperature: float = 0.1):
        """
        Initialize tournament manager.

        Args:
            training_dir: Directory containing the current training run
            device: Device to run models on
            games_per_match: Number of games to play per match
            temperature: Temperature for move selection
        """
        self.training_dir = Path(training_dir)
        self.device = device
        self.games_per_match = games_per_match
        self.temperature = temperature

        # Initialize ELO tracker
        self.elo_tracker = EloTracker(self.training_dir)

        # Setup logging
        self.logger = logging.getLogger("ModelTournament")

        # Tournament tracking
        self.current_tournament_id = None
        self.tournament_history_file = self.training_dir / "tournament_history.json"

        # Load tournament history if exists
        self.tournament_history = self._load_tournament_history()

    def _load_tournament_history(self) -> Dict:
        """Load tournament history from file."""
        if self.tournament_history_file.exists():
            try:
                with open(self.tournament_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading tournament history: {e}")
        return {}

    def _save_tournament_history(self):
        """Save tournament history to file."""
        try:
            with open(self.tournament_history_file, 'w') as f:
                json.dump(self.tournament_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving tournament history: {e}")

    def _load_model(self, checkpoint_path: Path) -> NetworkWrapper:
        """Load a model from checkpoint."""
        model = NetworkWrapper(device=self.device)
        model.load_model(str(checkpoint_path))
        return model

    def _play_match(self, white_model: NetworkWrapper, black_model: NetworkWrapper,
                    white_id: str, black_id: str) -> MatchResult:
        """Play a match between two models."""
        white_wins = 0
        black_wins = 0
        draws = 0
        total_moves = 0

        for game_num in range(self.games_per_match):
            self.logger.info(f"Playing game {game_num + 1}/{self.games_per_match}: "
                             f"{white_id} (White) vs {black_id} (Black)")

            game_state = GameState()
            move_count = 0

            while not game_state.is_terminal() and move_count < 500:
                current_model = white_model if game_state.current_player == Player.WHITE else black_model

                # Get valid moves
                valid_moves = game_state.get_valid_moves()
                if not valid_moves:
                    break

                # Get model's move choice
                state_tensor = current_model.state_encoder.encode_state(game_state)
                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
                move_probs, _ = current_model.predict(state_tensor)

                # Select move
                selected_move = current_model.select_move(move_probs, valid_moves, self.temperature)

                # Make move
                success = game_state.make_move(selected_move)
                if not success:
                    self.logger.error(
                        f"Invalid move by {'White' if game_state.current_player == Player.WHITE else 'Black'}")
                    break

                move_count += 1

            # Record game result
            winner = game_state.get_winner()
            if winner == Player.WHITE:
                white_wins += 1
            elif winner == Player.BLACK:
                black_wins += 1
            else:
                draws += 1

            total_moves += move_count

        # Create match result
        match_result = MatchResult(
            white_model=white_id,
            black_model=black_id,
            white_wins=white_wins,
            black_wins=black_wins,
            draws=draws,
            avg_game_length=total_moves / self.games_per_match
        )

        self.logger.info(f"Match complete: White: {white_wins}, Black: {black_wins}, Draws: {draws}")
        return match_result

    def run_tournament(self, current_iteration: int):
        """
        Run a tournament for the current iteration.

        Will play the current model against all previous models from this training run.
        """
        tournament_id = f"tournament_{current_iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_tournament_id = tournament_id

        # Get all available models up to current iteration
        model_paths = []
        for i in range(current_iteration + 1):
            checkpoint_path = self.training_dir / f"checkpoint_iteration_{i}.pt"
            if checkpoint_path.exists():
                model_paths.append((i, checkpoint_path))

        if len(model_paths) < 2:
            self.logger.info("Not enough models for tournament yet")
            return

        tournament_results = []
        current_model = self._load_model(model_paths[-1][1])
        current_id = f"iteration_{current_iteration}"

        # Play against each previous model
        for prev_iter, prev_path in model_paths[:-1]:
            prev_id = f"iteration_{prev_iter}"
            prev_model = self._load_model(prev_path)

            # Play match with current model as White
            white_result = self._play_match(current_model, prev_model, current_id, prev_id)
            tournament_results.append(white_result)
            self.elo_tracker.update_ratings(white_result)

            # Play match with current model as Black
            black_result = self._play_match(prev_model, current_model, prev_id, current_id)
            tournament_results.append(black_result)
            self.elo_tracker.update_ratings(black_result)

        # Save tournament results
        self.tournament_history[tournament_id] = {
            'iteration': current_iteration,
            'timestamp': datetime.now().isoformat(),
            'results': [vars(result) for result in tournament_results]
        }
        self._save_tournament_history()

        # Log summary
        self.logger.info(f"\nTournament {tournament_id} complete")
        self.logger.info("Current ratings:")
        for model_id, rating in sorted(self.elo_tracker.ratings.items()):
            self.logger.info(f"{model_id}: {rating:.1f}")

    def get_latest_tournament_summary(self) -> Dict:
        """Get summary of the most recent tournament."""
        if not self.current_tournament_id:
            return None
        return self.tournament_history.get(self.current_tournament_id)

    def get_model_performance(self, model_id: str) -> Dict:
        """Get performance statistics for a specific model."""
        matches = self.elo_tracker.get_match_history(model_id)
        total_games = sum(m.total_games() for m in matches)
        wins = sum(m.white_wins if m.white_model == model_id else m.black_wins for m in matches)
        draws = sum(m.draws for m in matches)

        return {
            'total_games': total_games,
            'wins': wins,
            'draws': draws,
            'losses': total_games - wins - draws,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'current_rating': self.elo_tracker.get_rating(model_id)
        }