"""ELO rating system for tracking YINSH model performance."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MatchResult:
    """Stores the result of a match between two models."""
    white_model: str  # model identifier (e.g., "iteration_10")
    black_model: str
    white_wins: int
    black_wins: int
    draws: int
    timestamp: str = None
    avg_game_length: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def total_games(self) -> int:
        return self.white_wins + self.black_wins + self.draws

    def white_score(self) -> float:
        """Calculate score for white (1 for win, 0.5 for draw, 0 for loss)."""
        total = self.total_games()
        return (self.white_wins + 0.5 * self.draws) / total if total > 0 else 0

    def black_score(self) -> float:
        """Calculate score for black."""
        return 1 - self.white_score()


class EloTracker:
    """Manages ELO ratings for YINSH models."""

    def __init__(self, save_dir: Path, k_factor: float = 20, initial_rating: float = 1500):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[Tuple[str, float]]] = {}
        self.match_history: List[MatchResult] = []

        # Setup logging
        self.logger = logging.getLogger("EloTracker")
        self.logger.setLevel(logging.INFO)

    def _load_data(self):
        """Load existing ratings and match history."""
        ratings_file = self.save_dir / "elo_ratings.json"
        history_file = self.save_dir / "match_history.json"

        if ratings_file.exists():
            try:
                with open(ratings_file, 'r') as f:
                    data = json.load(f)
                    self.ratings = data['current_ratings']
                    self.rating_history = {k: [(t, r) for t, r in v]
                                           for k, v in data['rating_history'].items()}
                self.logger.info(f"Loaded ratings for {len(self.ratings)} models")
            except Exception as e:
                self.logger.error(f"Error loading ratings: {e}")

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    matches = json.load(f)
                    self.match_history = [MatchResult(**m) for m in matches]
                self.logger.info(f"Loaded {len(self.match_history)} historical matches")
            except Exception as e:
                self.logger.error(f"Error loading match history: {e}")

    def _save_data(self):
        """Save current ratings and match history."""
        # Save ratings
        ratings_data = {
            'current_ratings': self.ratings,
            'rating_history': {k: [(t, r) for t, r in v]
                               for k, v in self.rating_history.items()},
            'last_updated': datetime.now().isoformat()
        }

        with open(self.save_dir / "elo_ratings.json", 'w') as f:
            json.dump(ratings_data, f, indent=2)

        # Save match history
        matches_data = [vars(m) for m in self.match_history]
        with open(self.save_dir / "match_history.json", 'w') as f:
            json.dump(matches_data, f, indent=2)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _update_single_game(self, white_id: str, black_id: str, white_won: bool):
        """Update ratings for a single game."""
        # Get or initialize ratings
        white_rating = self.ratings.get(white_id, self.initial_rating)
        black_rating = self.ratings.get(black_id, self.initial_rating)

        # Calculate expected scores
        white_expected = self._expected_score(white_rating, black_rating)
        black_expected = 1 - white_expected

        # Actual scores (1 for win, 0 for loss)
        white_actual = 1.0 if white_won else 0.0
        black_actual = 1.0 - white_actual

        # Update ratings
        white_new = white_rating + self.k_factor * (white_actual - white_expected)
        black_new = black_rating + self.k_factor * (black_actual - black_expected)

        # Store new ratings
        self.ratings[white_id] = white_new
        self.ratings[black_id] = black_new

        return white_new, black_new

    def _update_rating_history(self, model_id: str, rating: float, timestamp: str):
        """Add a new rating to a model's history."""
        if model_id not in self.rating_history:
            self.rating_history[model_id] = []
        self.rating_history[model_id].append((timestamp, rating))

    def get_rating(self, model_id: str) -> float:
        """Get current rating for a model, initializing if necessary."""
        if model_id not in self.ratings:
            self.ratings[model_id] = self.initial_rating
            self._update_rating_history(model_id, self.initial_rating)
        return self.ratings[model_id]

    def update_ratings(self, match_result: MatchResult):
        """Update ratings based on match result, processing each game individually."""
        white_id = match_result.white_model
        black_id = match_result.black_model

        # Process white wins
        for _ in range(match_result.white_wins):
            self._update_single_game(white_id, black_id, white_won=True)

        # Process black wins
        for _ in range(match_result.black_wins):
            self._update_single_game(white_id, black_id, white_won=False)

        # Record final ratings in history
        final_white = self.ratings[white_id]
        final_black = self.ratings[black_id]

        timestamp = datetime.now().isoformat()
        self._update_rating_history(white_id, final_white, timestamp)
        self._update_rating_history(black_id, final_black, timestamp)

        # Add match to history with final ratings
        self.match_history.append(MatchResult(
            white_model=white_id,
            black_model=black_id,
            white_wins=match_result.white_wins,
            black_wins=match_result.black_wins,
            draws=0,  # Yinsh has no draws
            timestamp=timestamp,
            avg_game_length=match_result.avg_game_length
        ))

        # Log the overall rating changes
        self.logger.info(f"Updated ratings after match:")
        self.logger.info(f"  {white_id}: {final_white:.1f}")
        self.logger.info(f"  {black_id}: {final_black:.1f}")

        # Save updated data
        self._save_data()

    def get_rating_history(self, model_id: str) -> List[Tuple[str, float]]:
        """Get rating history for a model."""
        return self.rating_history.get(model_id, [])

    def get_match_history(self, model_id: Optional[str] = None) -> List[MatchResult]:
        """Get match history, optionally filtered for a specific model."""
        if model_id is None:
            return self.match_history
        return [m for m in self.match_history
                if m.white_model == model_id or m.black_model == model_id]

    def plot_rating_history(self, save_path: Optional[str] = None):
        """Plot rating history for all models."""
        plt.figure(figsize=(12, 8))

        for model_id, history in self.rating_history.items():
            timestamps = [datetime.fromisoformat(t) for t, _ in history]
            ratings = [r for _, r in history]
            plt.plot(timestamps, ratings, marker='o', label=f"Model {model_id}")

        plt.title("Model Rating History")
        plt.xlabel("Time")
        plt.ylabel("ELO Rating")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def generate_summary(self) -> Dict:
        """Generate summary statistics of ratings and matches."""
        summary = {
            'total_models': len(self.ratings),
            'total_matches': len(self.match_history),
            'current_ratings': dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)),
            'avg_rating': np.mean(list(self.ratings.values())),
            'rating_std': np.std(list(self.ratings.values())),
            'latest_matches': self.match_history[-5:] if self.match_history else []
        }
        return summary