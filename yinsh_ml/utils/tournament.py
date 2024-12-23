import json
import logging
from pathlib import Path
from typing import Dict, Optional
import torch

from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.elo_manager import EloTracker
from yinsh_ml.utils.encoding import StateEncoder

class ModelTournament:
    """Manages a tournament between multiple models."""

    def __init__(self, training_dir: Path, device: str, games_per_match: int = 10, temperature: float = 0.1):
        self.training_dir = training_dir
        self.games_per_match = games_per_match
        self.temperature = temperature
        self.device = device
        self.logger = logging.getLogger("ModelTournament")
        # self.elo_tracker = EloTracker() remove this line
        self.tournament_history = {}  # Store results
        self.results_dir = Path("tournament_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _get_elo_tracker(self, experiment_dir: Path, current_iteration: int):
        """Loads or initializes EloTracker with appropriate save directory."""
        save_dir = experiment_dir / "elo_tracker" / f"iteration_{current_iteration}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return EloTracker(save_dir=save_dir)

    def _load_model(self, checkpoint_path: Path) -> NetworkWrapper:
        """Loads a model from the specified checkpoint."""
        try:
            model = NetworkWrapper(device=self.device)
            model.load_model(str(checkpoint_path))
            self.logger.info(f"Loaded model from {checkpoint_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model from {checkpoint_path}: {e}")
            raise

    def _play_match(self, white_model: NetworkWrapper, black_model: NetworkWrapper,
                    white_id: str, black_id: str, state_encoder: StateEncoder) -> float:
        """Plays a match between two models and returns the result for the white player."""
        self.logger.info(f"Playing game: {white_id} (White) vs {black_id} (Black)")

        # set current player for each model
        white_model.current_player = Player.WHITE
        black_model.current_player = Player.BLACK

        results = []
        for game_num in range(1, self.games_per_match + 1):
            print(f"Playing game {game_num}/{self.games_per_match}: {white_id} (White) vs {black_id} (Black)")  # Added print
            result = self._play_evaluation_game(white_model, black_model, state_encoder)
            print(f"Game {game_num} result: {result}")  # Added print
            if result is None:
                # Handle invalid game result
                self.logger.error(f"Invalid game result encountered in game {game_num}. Skipping.")
                continue

            results.append(result)

        # Calculate average result for the match (assuming 1 for win, -1 for loss, 0 for draw)
        avg_result = sum(results) / len(results) if results else 0
        return avg_result

    def _play_evaluation_game(self, white_model: NetworkWrapper, black_model: NetworkWrapper,
                               state_encoder: StateEncoder) -> Optional[int]:
        """Plays a single game between two models."""
        game_state = GameState()
        move_count = 0
        max_moves = 500

        while not game_state.is_terminal() and move_count < max_moves:
            current_model = white_model if game_state.current_player == Player.WHITE else black_model
            valid_moves = game_state.get_valid_moves()
            if not valid_moves:
                break

            state_tensor = state_encoder.encode_state(game_state)
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(current_model.device)

            with torch.no_grad():
                move_probs, _ = current_model.predict(state_tensor)

            selected_move = current_model.select_move(
                move_probs=move_probs,
                valid_moves=valid_moves,
                temperature=self.temperature
            )

            if not game_state.make_move(selected_move):
                self.logger.error(
                    f"Invalid move made by {'White' if game_state.current_player == Player.WHITE else 'Black'}"
                )
                return None

            if game_state.phase == GamePhase.MAIN_GAME:
                move_count += 1

        winner = game_state.get_winner()
        return 1 if winner == Player.WHITE else (-1 if winner == Player.BLACK else 0)

    def _update_results(self, model_a_id: str, model_b_id: str, white_result: float, black_result: float):
        """Updates the results dictionary with the outcome of the match."""
        if "models" not in self.tournament_history:
            self.tournament_history["models"] = {}

        self._update_model_entry(model_a_id)
        self._update_model_entry(model_b_id)

        self.tournament_history["models"][model_a_id]["match_results"][model_b_id] = white_result
        self.tournament_history["models"][model_b_id]["match_results"][model_a_id] = black_result

    def _update_model_entry(self, model_id: str):
        """Initializes or updates a model entry in the results dictionary."""
        if model_id not in self.tournament_history["models"]:
            self.tournament_history["models"][model_id] = {
                "match_results": {},
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0
            }

    def _save_tournament_history(self):
        """Saves the tournament history to a JSON file."""
        file_path = self.results_dir / "tournament_history.json"
        with open(file_path, "w") as f:
            json.dump(self.tournament_history, f, indent=4)
        self.logger.info(f"Tournament history saved to {file_path}")

    def get_model_performance(self, model_id: str) -> Dict:
        """Retrieves the performance of a specified model."""
        if "models" not in self.tournament_history or model_id not in self.tournament_history["models"]:
            self.logger.warning(f"No data found for model {model_id}.")
            return {}

        model_data = self.tournament_history["models"][model_id]
        total_games = model_data["total_games"]
        wins = model_data["wins"]
        losses = model_data["losses"]
        draws = model_data["draws"]

        win_rate = wins / total_games if total_games > 0 else 0
        loss_rate = losses / total_games if total_games > 0 else 0
        draw_rate = draws / total_games if total_games > 0 else 0

        return {
            "model_id": model_id,
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate
        }

    def _load_models(self, experiment_type: str, config_name: str, current_iteration: int) -> Dict[str, NetworkWrapper]:
        """Loads models for the tournament."""
        models = {}
        for i in range(current_iteration + 1):
            checkpoint_path = self.training_dir / experiment_type / config_name / f"checkpoint_iteration_{i}.pt"
            if checkpoint_path.exists():
                model_id = f"iteration_{i}"
                models[model_id] = self._load_model(checkpoint_path)
        return models

    def run_tournament(self, experiment_type: str = "combined", config_name: str = "", current_iteration: int = 0, state_encoder = None):
        """Runs the tournament with multiple models."""
        self.logger.info(f"\nStarting tournament for iteration {current_iteration}")

        # Load models for the tournament
        models = self._load_models(experiment_type, config_name, current_iteration)

        if len(models) < 2:
            self.logger.warning("Not enough models for a tournament. Skipping.")
            return

        experiment_dir = self.training_dir / experiment_type / config_name
        elo_tracker = self._get_elo_tracker(experiment_dir, current_iteration)

        # Play the matches and update Elo ratings after each match
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a_id = list(models.keys())[i]
                model_b_id = list(models.keys())[j]
                self.logger.info(f"\nMatching {model_a_id} vs {model_b_id}")

                white_result = self._play_match(models[model_a_id], models[model_b_id],
                                                model_a_id, model_b_id, state_encoder)
                black_result = self._play_match(models[model_b_id], models[model_a_id],
                                                model_b_id, model_a_id, state_encoder)

                # Update results based on the outcome of the match
                self._update_results(model_a_id, model_b_id, white_result, black_result)

                # Update Elo ratings after each match
                elo_tracker.update_ratings(model_a_id, model_b_id, white_result, black_result)

        # Save Elo ratings after the tournament
        elo_tracker._save_ratings()
        self._save_tournament_history()