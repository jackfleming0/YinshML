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

import math

q = math.log(10) / 400  # constant used in Glicko formulas


class GlickoPlayer:
    def __init__(self, rating=1500.0, rd=350.0):
        self.rating = rating
        self.rd = rd  # Rating Deviation

class GlickoTracker:
    """
    Robust Glicko-1 tracker.

    Records match results during a rating period and then updates each player's
    rating and rating deviation (RD) using the full set of matches.
    """
    def __init__(self, training_dir, initial_rating=1500.0, initial_rd=350.0, K_factor=0.00001):
        self.training_dir = training_dir
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.K_factor = K_factor  # New dampening factor
        self.players = {}  # model_id -> GlickoPlayer
        self.match_history = []  # list of match records for the rating period

    def add_model(self, model_id):
        """Ensure a model is in the tracker."""
        if model_id not in self.players:
            self.players[model_id] = GlickoPlayer(self.initial_rating, self.initial_rd)

    def g(self, rd):
        """Scaling function of opponent's RD."""
        return 1 / math.sqrt(1 + 3 * (q ** 2) * (rd ** 2) / (math.pi ** 2))

    def E(self, rating, opp_rating, opp_rd):
        """Expected score for a player against an opponent."""
        return 1 / (1 + 10 ** (-self.g(opp_rd) * (rating - opp_rating) / 400.0))

    def record_match(self, white_model, black_model, white_wins, black_wins, draws):
        """
        Record a match result.

        For an aggregated match (e.g. 50 games), total_games = white_wins+black_wins+draws.
        For white, S = (white_wins + 0.5 * draws) / total_games; similarly for black.
        """
        total_games = white_wins + black_wins + draws
        if total_games == 0:
            return
        white_score = (white_wins + 0.5 * draws) / total_games
        black_score = (black_wins + 0.5 * draws) / total_games
        self.match_history.append({
            'white_model': white_model,
            'black_model': black_model,
            'white_score': white_score,
            'black_score': black_score
        })
        # Ensure both players are in the tracker.
        self.add_model(white_model)
        self.add_model(black_model)

    def update_ratings(self):
        """
        Update all players’ ratings and RDs using Glicko-1 formulas over the entire rating period.

        For each player, gather all matches played in this period and then:
          1. Compute the variance v.
          2. Compute the rating change delta.
          3. Update the rating and RD.
        The final rating update is scaled by self.K_factor to prevent over-large changes.
        """
        # Organize results per player:
        results = {}  # player_id -> list of (opp_rating, opp_rd, score)
        for match in self.match_history:
            white = match['white_model']
            black = match['black_model']
            white_score = match['white_score']
            black_score = match['black_score']
            if white not in results:
                results[white] = []
            if black not in results:
                results[black] = []
            # For white: opponent is black.
            opp_black = self.players[black]
            results[white].append((opp_black.rating, opp_black.rd, white_score))
            # For black: opponent is white.
            opp_white = self.players[white]
            results[black].append((opp_white.rating, opp_white.rd, black_score))

        # Update each player's rating and RD based on all matches:
        for player_id, matches in results.items():
            player = self.players[player_id]
            if not matches:
                continue

            # Compute variance v:
            sum_term = 0.0
            for opp_rating, opp_rd, score in matches:
                E_val = self.E(player.rating, opp_rating, opp_rd)
                sum_term += (self.g(opp_rd) ** 2) * E_val * (1 - E_val)
            v = 1 / (q ** 2 * sum_term) if sum_term != 0 else float('inf')

            # Compute delta:
            delta = 0.0
            for opp_rating, opp_rd, score in matches:
                E_val = self.E(player.rating, opp_rating, opp_rd)
                delta += self.g(opp_rd) * (score - E_val)
            delta *= v * q

            # Compute the update denominator:
            rating_inv = 1 / (player.rd ** 2) + 1 / v

            # Apply dampening via K_factor:
            new_rating = player.rating + self.K_factor * (delta / rating_inv)
            new_rd = math.sqrt(1 / rating_inv)

            player.rating = new_rating
            player.rd = new_rd

        # Clear match history after processing the rating period.
        self.match_history = []

    def get_rating(self, model_id):
        """Return the current rating of the specified model."""
        if model_id in self.players:
            return self.players[model_id].rating
        return self.initial_rating

    def get_rd(self, model_id):
        """Return the current RD of the specified model."""
        if model_id in self.players:
            return self.players[model_id].rd
        return self.initial_rd

    def get_match_history(self, model_id):
        """Return all recorded match results for a given model (as stored during the rating period)."""
        # This returns the matches recorded before update_ratings() is called.
        return [m for m in self.match_history if m['white_model'] == model_id or m['black_model'] == model_id]

class ModelTournament:
    """Manages tournaments between YINSH models from the current training run."""

    def __init__(self,
                 training_dir: Path,
                 device: str = 'cpu',
                 games_per_match: int = 15,
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
        #self.elo_tracker = EloTracker(self.training_dir)

        #initialize glicko tracker
        self.glicko_tracker = GlickoTracker(self.training_dir)

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

    def _play_match(self,
                    white_model: NetworkWrapper,
                    black_model: NetworkWrapper,
                    white_id: str,
                    black_id: str) -> MatchResult:
        """Play a match (several games) between two models."""
        white_wins = 0
        black_wins = 0
        draws = 0
        total_moves = 0

        for game_num in range(self.games_per_match):
            self.logger.debug(f"Playing game {game_num + 1}/{self.games_per_match}: "
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
                selected_move = current_model.select_move(
                    move_probs, valid_moves, self.temperature
                )

                # Make move
                success = game_state.make_move(selected_move)
                if not success:
                    self.logger.error(
                        f"Invalid move by {game_state.current_player.name}"
                    )
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

        match_result = MatchResult(
            white_model=white_id,
            black_model=black_id,
            white_wins=white_wins,
            black_wins=black_wins,
            draws=draws,
            avg_game_length=total_moves / self.games_per_match if self.games_per_match > 0 else 0
        )

        self.logger.debug(f"Match complete: White: {white_wins}, Black: {black_wins}, Draws: {draws}")
        return match_result

    def run_full_round_robin_tournament(self, current_iteration: int):
        """
        Run a full round-robin tournament among all iterations from 0..current_iteration.
        This version uses a robust Glicko-1 rating system:
          - It resets all model ratings to an initial value (1500) and RD (350) by reinitializing
            the GlickoTracker.
          - It records every match outcome using record_match.
          - After all matches, it updates the ratings in a batch via update_ratings.
        The final summary includes each model's updated Glicko rating and RD.
        """
        # Create a unique tournament ID
        tournament_id = f"full_round_robin_{current_iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_tournament_id = tournament_id

        # Gather all model checkpoints up to current_iteration
        model_paths = []
        for i in range(current_iteration + 1):
            ckpt = self.training_dir / f"checkpoint_iteration_{i}.pt"
            if ckpt.exists():
                model_paths.append((i, ckpt))

        if len(model_paths) < 2:
            self.logger.info("Skipping tournament - need at least 2 models.")
            return

        # Load all models into memory
        models = {}
        for iter_num, path in model_paths:
            model_id = f"iteration_{iter_num}"
            models[model_id] = self._load_model(path)

        # Reset Glicko ratings for all models by initializing a new tracker
        self.logger.info("Resetting Glicko ratings for all models...")
        self.glicko_tracker = GlickoTracker(self.training_dir, initial_rating=1500.0, initial_rd=350.0)
        for model_id in models.keys():
            self.glicko_tracker.add_model(model_id)

        # Keep track of individual match results
        round_robin_results = []

        # Round-robin among all pairs
        model_ids = sorted(models.keys(), key=lambda x: int(x.split('_')[-1]))
        self.logger.info(f"Starting round-robin among {len(model_ids)} models.")

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                id_i = model_ids[i]
                id_j = model_ids[j]
                model_i = models[id_i]
                model_j = models[id_j]

                self.logger.info(f"Match: {id_i} vs {id_j} (i as White, j as Black)")
                result_white = self._play_match(
                    white_model=model_i,
                    black_model=model_j,
                    white_id=id_i,
                    black_id=id_j
                )
                # Record the match result for the white side
                self.glicko_tracker.record_match(id_i, id_j,
                                                 white_wins=result_white.white_wins,
                                                 black_wins=result_white.black_wins,
                                                 draws=result_white.draws)
                round_robin_results.append(result_white)

                self.logger.info(f"Reverse Match: {id_j} vs {id_i} (j as White, i as Black)")
                result_black = self._play_match(
                    white_model=model_j,
                    black_model=model_i,
                    white_id=id_j,
                    black_id=id_i
                )
                # Record the match result for the reverse match
                self.glicko_tracker.record_match(id_j, id_i,
                                                 white_wins=result_black.white_wins,
                                                 black_wins=result_black.black_wins,
                                                 draws=result_black.draws)
                round_robin_results.append(result_black)

        # After recording all matches, update the Glicko ratings in a batch
        self.glicko_tracker.update_ratings()

        # Build summary stats for each model and include Glicko ratings and RD
        summary_stats = self._aggregate_round_robin_stats(round_robin_results)
        for model_id in summary_stats.keys():
            summary_stats[model_id]['glicko_rating'] = self.glicko_tracker.get_rating(model_id)
            summary_stats[model_id]['rd'] = self.glicko_tracker.get_rd(model_id)

        # Save tournament results
        self.tournament_history[tournament_id] = {
            'iteration': current_iteration,
            'timestamp': datetime.now().isoformat(),
            'round_robin_results': [vars(r) for r in round_robin_results],
            'stats': summary_stats,
        }
        self._save_tournament_history()

        # Log final summary
        self.logger.info(f"\n{'=' * 20} Round-Robin Summary {'=' * 20}")
        for model_id in sorted(summary_stats.keys(), key=lambda m: int(m.split('_')[-1])):
            st = summary_stats[model_id]
            self.logger.info(
                f"{model_id} | Glicko Rating: {st['glicko_rating']:.1f} (RD: {st['rd']:.1f}) | "
                f"W-L-D: {st['wins']}-{st['losses']}-{st['draws']} "
                f"({st['win_rate'] * 100:.1f}% win) | "
                f"WhiteWinRate: {st['white_win_rate'] * 100:.1f}% | "
                f"BlackWinRate: {st['black_win_rate'] * 100:.1f}%"
            )
        self.logger.info("=" * 60)

    def _aggregate_round_robin_stats(self, match_results: List[MatchResult]) -> Dict[str, Dict]:
        """
        Aggregate overall and color-specific stats for each model:
          - total wins, draws, losses
          - color-based stats
          - final Glicko rating (retrieved via get_rating)
          - overall and color-specific win rates
        """
        stats = {}
        for mr in match_results:
            if mr.white_model not in stats:
                stats[mr.white_model] = {
                    'wins': 0, 'losses': 0, 'draws': 0,
                    'white_wins': 0, 'white_losses': 0, 'white_draws': 0,
                    'black_wins': 0, 'black_losses': 0, 'black_draws': 0,
                    'games': 0, 'white_games': 0, 'black_games': 0
                }
            if mr.black_model not in stats:
                stats[mr.black_model] = {
                    'wins': 0, 'losses': 0, 'draws': 0,
                    'white_wins': 0, 'white_losses': 0, 'white_draws': 0,
                    'black_wins': 0, 'black_losses': 0, 'black_draws': 0,
                    'games': 0, 'white_games': 0, 'black_games': 0
                }

            total_g = mr.total_games()
            # White side
            w = mr.white_wins
            d = mr.draws
            l = mr.black_wins  # from White's perspective
            stats[mr.white_model]['wins'] += w
            stats[mr.white_model]['draws'] += d
            stats[mr.white_model]['losses'] += l
            stats[mr.white_model]['white_wins'] += w
            stats[mr.white_model]['white_draws'] += d
            stats[mr.white_model]['white_losses'] += l
            stats[mr.white_model]['games'] += total_g
            stats[mr.white_model]['white_games'] += total_g

            # Black side
            w_b = mr.black_wins
            d_b = mr.draws
            l_b = mr.white_wins  # from Black's perspective
            stats[mr.black_model]['wins'] += w_b
            stats[mr.black_model]['draws'] += d_b
            stats[mr.black_model]['losses'] += l_b
            stats[mr.black_model]['black_wins'] += w_b
            stats[mr.black_model]['black_draws'] += d_b
            stats[mr.black_model]['black_losses'] += l_b
            stats[mr.black_model]['games'] += total_g
            stats[mr.black_model]['black_games'] += total_g

        # Calculate win rates and attach final Glicko ratings
        for model_id, data in stats.items():
            g = data['games']
            w = data['wins']
            data['win_rate'] = w / g if g > 0 else 0.0

            wg = data['white_games']
            data['white_win_rate'] = data['white_wins'] / wg if wg > 0 else 0.0

            bg = data['black_games']
            data['black_win_rate'] = data['black_wins'] / bg if bg > 0 else 0.0

            # Use the get_rating method instead of accessing a non-existent 'ratings' attribute
            data['elo'] = self.glicko_tracker.get_rating(model_id)

        return stats

    def get_latest_tournament_summary(self) -> Dict:
        """Get summary of the most recent tournament."""
        if not self.current_tournament_id:
            return None
        return self.tournament_history.get(self.current_tournament_id)

    # def run_tournament(self, current_iteration: int):
    #     """Run tournament for current iteration against previous models."""
    #     tournament_id = f"tournament_{current_iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    #     self.current_tournament_id = tournament_id
    #
    #     # Silently get model paths
    #     model_paths = [
    #         (i, self.training_dir / f"checkpoint_iteration_{i}.pt")
    #         for i in range(current_iteration + 1)
    #         if (self.training_dir / f"checkpoint_iteration_{i}.pt").exists()
    #     ]
    #
    #     if len(model_paths) < 2:
    #         self.logger.info("Skipping tournament - need at least 2 models")
    #         return
    #
    #     # Single log at start
    #     self.logger.info(f"\nStarting tournament for iteration {current_iteration}")
    #     self.logger.info(f"Playing against {len(model_paths) - 1} previous models")
    #
    #     tournament_results = []
    #     current_model = self._load_model(model_paths[-1][1])
    #     current_id = f"iteration_{current_iteration}"
    #
    #     for prev_iter, prev_path in model_paths[:-1]:
    #         prev_id = f"iteration_{prev_iter}"
    #         prev_model = self._load_model(prev_path)
    #
    #         # Log start of match
    #         self.logger.info(f"\nMatching iteration {current_iteration} vs {prev_iter}")
    #
    #         # Play both colors
    #         white_result = self._play_match(current_model, prev_model, current_id, prev_id)
    #         black_result = self._play_match(prev_model, current_model, prev_id, current_id)
    #
    #         tournament_results.extend([white_result, black_result])
    #         self.elo_tracker.update_ratings(white_result)
    #         self.elo_tracker.update_ratings(black_result)
    #
    #         # Log match result
    #         total_wins = white_result.white_wins + black_result.black_wins
    #         total_games = white_result.total_games() + black_result.total_games()
    #         self.logger.info(f"Win rate vs iter {prev_iter}: {total_wins / total_games:.1%}")
    #
    #     # Save results silently
    #     self.tournament_history[tournament_id] = {
    #         'iteration': current_iteration,
    #         'timestamp': datetime.now().isoformat(),
    #         'results': [vars(result) for result in tournament_results]
    #     }
    #     self._save_tournament_history()
    #
    #     # Single summary at end
    #     self.logger.info(f"\n{'=' * 20} Tournament Summary {'=' * 20}")
    #     self.logger.info(f"Current model: iteration {current_iteration}")
    #     self.logger.info("\nELO Ratings:")
    #     for model_id, rating in sorted(self.elo_tracker.ratings.items()):
    #         prefix = "→" if model_id == current_id else " "
    #         self.logger.info(f"{prefix} {model_id}: {rating:.1f}")
    #     self.logger.info("=" * 50)

    def get_model_performance(self, model_id: str) -> Dict:
        """Get performance statistics for a specific model from the EloTracker's match history."""
        matches = self.glicko_tracker.get_match_history(model_id)
        total_games = sum(m.total_games() for m in matches)
        wins = sum(
            (m.white_wins if m.white_model == model_id else m.black_wins)
            for m in matches
        )
        draws = sum(m.draws for m in matches)

        return {
            'total_games': total_games,
            'wins': wins,
            'draws': draws,
            'losses': total_games - wins - draws,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'current_rating': self.glicko_tracker.get_rating(model_id)
        }