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

    def __init__(self, save_dir: Path, k_factor: float = 32, initial_rating: float = 1500):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.k_factor = k_factor
        self.initial_rating = initial_rating

        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[Tuple[str, float]]] = {}  # model_id -> [(timestamp, rating)]
        self.match_history: List[MatchResult] = []

        self.logger = logging.getLogger("EloTracker")

        # Try to load existing data
        self._load_data()

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

    def _update_rating_history(self, model_id: str, rating: float):
        """Add a new rating to a model's history."""
        if model_id not in self.rating_history:
            self.rating_history[model_id] = []
        self.rating_history[model_id].append((datetime.now().isoformat(), rating))

    def get_rating(self, model_id: str) -> float:
        """Get current rating for a model, initializing if necessary."""
        if model_id not in self.ratings:
            self.ratings[model_id] = self.initial_rating
            self._update_rating_history(model_id, self.initial_rating)
        return self.ratings[model_id]

    def update_ratings(self, match_result: MatchResult):
        """Update ratings based on a match result."""
        # Get or initialize ratings
        white_rating = self.get_rating(match_result.white_model)
        black_rating = self.get_rating(match_result.black_model)

        # Calculate actual scores
        total_games = match_result.total_games()
        if total_games == 0:
            return

        white_score = match_result.white_score()
        black_score = 1 - white_score

        # Calculate expected scores
        white_expected = self._expected_score(white_rating, black_rating)
        black_expected = 1 - white_expected

        # Update ratings
        white_new = white_rating + self.k_factor * (white_score - white_expected)
        black_new = black_rating + self.k_factor * (black_score - black_expected)

        # Store new ratings
        self.ratings[match_result.white_model] = white_new
        self.ratings[match_result.black_model] = black_new

        # Update history
        self._update_rating_history(match_result.white_model, white_new)
        self._update_rating_history(match_result.black_model, black_new)

        # Add match to history
        self.match_history.append(match_result)

        # Save updated data
        self._save_data()

        self.logger.info(f"Updated ratings after match:")
        self.logger.info(f"  {match_result.white_model}: {white_rating:.1f} -> {white_new:.1f}")
        self.logger.info(f"  {match_result.black_model}: {black_rating:.1f} -> {black_new:.1f}")

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