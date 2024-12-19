"""Multi-model tournament management for YINSH."""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import torch
from collections import defaultdict
import argparse
from yinsh_ml.network.wrapper import NetworkWrapper

@dataclass
class ModelInfo:
    """Information about a specific model."""
    config_name: str      # Parent folder name (e.g., 'value_head_config2')
    iteration: int        # Model iteration number
    file_path: Path      # Full path to .pt file
    tournament_id: str    # Used in tournament: 'value_head_config2_iter_10'

class GlobalTournamentManager:
    def __init__(self, tournament_dir: Path):
        self.tournament_dir = tournament_dir
        self.tournament_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.ratings_file = self.tournament_dir / "global_elo_ratings.json"
        self.history_file = self.tournament_dir / "match_history.json"
        self.results_dir = self.tournament_dir / "tournament_results"
        self.results_dir.mkdir(exist_ok=True)

        # Load or initialize ratings
        self.ratings = self._load_ratings()
        self.match_history = self._load_match_history()

    def _load_ratings(self) -> Dict:
        """Load or initialize global ELO ratings."""
        if self.ratings_file.exists():
            with open(self.ratings_file) as f:
                return json.load(f)
        return {
            "ratings": {},
            "last_updated": datetime.now().isoformat(),
            "total_games": 0
        }

    def _load_match_history(self) -> List:
        """Load or initialize match history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def scan_models(self, checkpoints_root: Path) -> List[ModelInfo]:
        """Scan directories for model checkpoints."""
        models = []
        print(f"\nScanning for models in: {checkpoints_root}")

        if not checkpoints_root.exists():
            print(f"Error: Checkpoints directory not found!")
            return models

        # First check for "combined" subdirectory
        combined_dir = checkpoints_root / "combined"
        if combined_dir.exists():
            print(f"Found 'combined' directory, scanning configurations:")
            root_to_scan = combined_dir
        else:
            root_to_scan = checkpoints_root

        # List all configuration directories
        config_dirs = [d for d in root_to_scan.iterdir() if d.is_dir()]
        print(f"Found {len(config_dirs)} configuration directories:")
        for d in config_dirs:
            print(f"  - {d.name}")

        for config_dir in config_dirs:
            print(f"\nScanning config: {config_dir.name}")
            config_name = config_dir.name

            # Look for checkpoint files
            checkpoint_files = list(config_dir.glob("checkpoint_iteration_*.pt"))
            if checkpoint_files:
                print(f"  Found {len(checkpoint_files)} checkpoint files:")
                for cf in sorted(checkpoint_files):
                    print(f"    - {cf.name}")
                    try:
                        iteration = int(cf.stem.split('_')[-1])
                        tournament_id = f"{config_name}_iter_{iteration}"
                        models.append(ModelInfo(
                            config_name=config_name,
                            iteration=iteration,
                            file_path=cf,
                            tournament_id=tournament_id
                        ))
                    except ValueError as e:
                        print(f"      Error parsing iteration: {e}")
            else:
                print("  No checkpoint files found directly in directory")

        print(f"\nTotal models found: {len(models)}")
        if models:
            print("\nConfigurations found:")
            for config in sorted({m.config_name for m in models}):
                config_models = [m for m in models if m.config_name == config]
                print(f"  {config}: {len(config_models)} models")
                print(f"    Iterations: {', '.join(str(m.iteration) for m in sorted(config_models, key=lambda x: x.iteration))}")

        return sorted(models, key=lambda m: (m.config_name, m.iteration))

    def play_match(self, white_model: ModelInfo, black_model: ModelInfo,
                   num_games: int = 10) -> dict:
        """Play a match between two models."""
        from yinsh_ml.network.wrapper import NetworkWrapper
        from yinsh_ml.game.game_state import GameState
        from yinsh_ml.game.constants import Player

        # Load models with error handling
        try:
            # Check compatibility first
            if not self._check_network_compatibility(white_model, black_model):
                print(f"Match skipped due to incompatible architectures")
                return {
                    'white_wins': 0,
                    'black_wins': 0,
                    'total_games': 0,
                    'avg_game_length': 0,
                    'game_lengths': [],
                    'error': 'Incompatible architectures',
                    'skipped': True
                }

            # Then load models if compatible
            white = NetworkWrapper(model_path=str(white_model.file_path))
            black = NetworkWrapper(model_path=str(black_model.file_path))

        except Exception as e:
            print(f"Model loading failed: {white_model.tournament_id} vs {black_model.tournament_id}")
            print(f"Error: {e}")
            return {
                'white_wins': 0,
                'black_wins': 0,
                'total_games': 0,
                'avg_game_length': 0,
                'game_lengths': [],
                'error': str(e),
                'skipped': True
            }

        results = {
            'white_wins': 0,
            'black_wins': 0,
            'completed_games': 0,  # Add this
            'total_games': num_games,
            'avg_game_length': 0,
            'game_lengths': [],
            'skipped': False
        }

        total_moves = 0
        for game_num in range(num_games):
            game_state = GameState()
            move_count = 0
            max_moves = 500  # Prevent infinite games

            while not game_state.is_terminal() and move_count < max_moves:
                current_model = white if game_state.current_player == Player.WHITE else black
                valid_moves = game_state.get_valid_moves()

                if not valid_moves:
                    break

                # Get model's move choice
                state_tensor = current_model.state_encoder.encode_state(game_state)
                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(current_model.device)
                move_probs, _ = current_model.predict(state_tensor)

                # Use low temperature for evaluation
                selected_move = current_model.select_move(
                    move_probs=move_probs,
                    valid_moves=valid_moves,
                    temperature=0.1
                )

                success = game_state.make_move(selected_move)
                if not success:
                    print(f"Warning: Invalid move in game {game_num}")
                    break

                move_count += 1

            total_moves += move_count
            results['game_lengths'].append(move_count)

            # Record winner
            winner = game_state.get_winner()
            if winner == Player.WHITE:
                results['white_wins'] += 1
                results['completed_games'] += 1  # Add this

            elif winner == Player.BLACK:
                results['black_wins'] += 1
                results['completed_games'] += 1  # Add this

            # Calculate average using completed games:
            if results['completed_games'] > 0:
                results['avg_game_length'] = total_moves / results['completed_games']

        if total_moves > 0:  # Only calculate average if games were played
            results['avg_game_length'] = total_moves / num_games

        return results

    def update_elo(self, white_id: str, black_id: str, white_score: float, k_factor: float = 32.0,
                   total_games: int = 0):
        white_rating = self.ratings["ratings"].get(white_id, 1500)
        black_rating = self.ratings["ratings"].get(black_id, 1500)

        # Calculate expected score
        expected_white = 1 / (1 + 10 ** ((black_rating - white_rating) / 400))

        # Calculate change (using actual game results)
        white_actual = white_score / total_games if total_games > 0 else 0.5

        # Adjust K factor based on number of games and decisiveness
        games_factor = min(1.0, total_games / 20)
        margin_factor = abs(white_actual - expected_white)
        adjusted_k = k_factor * games_factor * margin_factor

        # Calculate rating changes
        rating_change = adjusted_k * (white_actual - expected_white)

        new_white = white_rating + rating_change
        new_black = black_rating - rating_change

        self.ratings["ratings"][white_id] = new_white
        self.ratings["ratings"][black_id] = new_black

        return new_white, new_black

    def _check_network_compatibility(self, white_model: ModelInfo, black_model: ModelInfo) -> bool:
        """Check if models should be allowed to play based on their configs."""
        # Map model IDs to their experiment types
        white_config = white_model.config_name
        black_config = black_model.config_name

        # Define compatible groups
        residual_configs = {'smoke', 'short_baseline', 'value_head_config', 'week_run'}
        attention_configs = {'attention_config', 'value_head_config2'}

        # Only allow matches within same architecture group
        if white_config in residual_configs and black_config in residual_configs:
            return True
        if white_config in attention_configs and black_config in attention_configs:
            return True

        return False

    def save_ratings(self):
        """Save current ratings to file."""
        try:
            with open(self.ratings_file, 'w') as f:
                json.dump(self.ratings, f, indent=2)
        except Exception as e:
            print(f"Error saving ratings: {e}")


    def run_tournament(self, sampled_models: List[ModelInfo], games_per_match: int = 10):
        """Run full tournament with sampled models."""
        print("\nStarting Tournament:")
        pairings = self.generate_pairings(sampled_models)
        total_matches = len(pairings)
        skipped_matches = []

        for idx, (model1, model2) in enumerate(pairings, 1):
            print(f"\nMatch {idx}/{total_matches}:")
            print(f"{model1.tournament_id} vs {model2.tournament_id}")

            # Play match
            results = self.play_match(model1, model2, games_per_match)

            # Handle skipped matches
            if results.get('skipped', False):
                skipped_matches.append((model1.tournament_id, model2.tournament_id))
                print("Match skipped due to incompatibility")
                continue

            total_games = results['white_wins'] + results['black_wins']
            if total_games == 0:
                print("No valid games played, skipping rating update")
                continue

            white_score = results['white_wins'] / total_games

            # Update ELO ratings
            white_new, black_new = self.update_elo(
                model1.tournament_id,
                model2.tournament_id,
                white_score,
                total_games=results['completed_games']  # Pass in completed games
            )

            # Record match in history
            match_record = {
                'timestamp': datetime.now().isoformat(),
                'white': {
                    'id': model1.tournament_id,
                    'config': model1.config_name,
                    'start_rating': self.ratings["ratings"][model1.tournament_id],
                    'end_rating': white_new
                },
                'black': {
                    'id': model2.tournament_id,
                    'config': model2.config_name,
                    'start_rating': self.ratings["ratings"][model2.tournament_id],
                    'end_rating': black_new
                },
                'results': {
                    'white_wins': results['white_wins'],
                    'black_wins': results['black_wins'],
                    'avg_game_length': results['avg_game_length']
                }
            }
            self.match_history.append(match_record)

            # Save after each match
            self.save_ratings()
            with open(self.history_file, 'w') as f:
                json.dump(self.match_history, f, indent=2)

            start_white = self.ratings["ratings"][model1.tournament_id]
            start_black = self.ratings["ratings"][model2.tournament_id]

            print(f"Results: White wins: {results['white_wins']}, Black wins: {results['black_wins']}")
            # print(f"White ELO: {white_new:.1f} ({white_new - self.ratings['ratings'][model1.tournament_id]:+.1f})")
            # print(f"Black ELO: {black_new:.1f} ({black_new - self.ratings['ratings'][model2.tournament_id]:+.1f})")
            print(f"White ELO: {white_new:.1f} ({white_new - start_white:+.1f})")
            print(f"Black ELO: {black_new:.1f} ({black_new - start_black:+.1f})")

        # Print summary of skipped matches
        if skipped_matches:
            print("\nSkipped Matches Summary:")
            print(f"Total skipped matches: {len(skipped_matches)}")
            for white_id, black_id in skipped_matches[:5]:
                print(f"  {white_id} vs {black_id}")
            if len(skipped_matches) > 5:
                print(f"  ... and {len(skipped_matches) - 5} more")

    def sample_models_for_tournament(self, models: List[ModelInfo],
                                   target_games_per_config: int =
                                   50) -> List[ModelInfo]:
        """Sample models to get reasonable tournament size with good coverage."""
        sampled = []
        by_config = defaultdict(list)

        # Sort models into configs
        for model in models:
            by_config[model.config_name].append(model)

        # Calculate base sampling rate
        num_configs = len(by_config)
        base_models_per_config = 4  # Minimum models to sample

        print(f"\nSampling Strategy:")
        print(f"Target games per config: {target_games_per_config}")

        for config_name, config_models in by_config.items():
            config_models = sorted(config_models, key=lambda m: m.iteration)
            print(f"\n{config_name}:")
            print(f"  Total models: {len(config_models)}")

            # Calculate number of models to sample based on config size
            if len(config_models) <= base_models_per_config:
                # For small configs, take all models
                models_to_sample = len(config_models)
            else:
                # For larger configs, scale up sampling
                models_to_sample = max(
                    base_models_per_config,
                    min(20, len(config_models) // 5)  # Cap at 20 models, but take at least 1/5th
                )

            selected = []
            # Always include first and last
            selected.append(config_models[0])
            if len(config_models) > 1:
                selected.append(config_models[-1])

            # Sample remaining models evenly
            if len(config_models) > 2:
                remaining_to_sample = models_to_sample - len(selected)
                if remaining_to_sample > 0:
                    step = max(1, (len(config_models) - 2) // remaining_to_sample)
                    indices = range(step, len(config_models) - 1, step)
                    indices = list(indices)[:remaining_to_sample]
                    for idx in indices:
                        selected.append(config_models[idx])

            print(f"  Selected {len(selected)} models: " +
                  f"{', '.join(str(m.iteration) for m in sorted(selected, key=lambda m: m.iteration))}")
            sampled.extend(selected)

        # Calculate expected games
        configs = list(by_config.keys())
        total_matches = 0
        config_matches = defaultdict(int)
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                config1_models = len([m for m in sampled if m.config_name == configs[i]])
                config2_models = len([m for m in sampled if m.config_name == configs[j]])
                matches = config1_models * config2_models * 2  # *2 for both colors
                total_matches += matches
                config_matches[configs[i]] += matches
                config_matches[configs[j]] += matches

        print(f"\nTournament Summary:")
        print(f"Total models selected: {len(sampled)}")
        print(f"Total matches to play: {total_matches}")
        print("\nMatches per configuration:")
        for config, matches in config_matches.items():
            print(f"  {config}: {matches} matches ({matches * 2} games)")

        return sorted(sampled, key=lambda m: (m.config_name, m.iteration))

    def sample_models_for_tournament(self, models: List[ModelInfo],
                                     target_games_per_config: int = 50) -> List[ModelInfo]:
        """Sample models to get reasonable tournament size with good coverage."""
        sampled = []
        by_config = defaultdict(list)

        # Sort models into configs
        for model in models:
            by_config[model.config_name].append(model)

        # Calculate base sampling rate
        num_configs = len(by_config)
        base_models_per_config = 4  # Minimum models to sample

        print(f"\nSampling Strategy:")
        print(f"Target games per config: {target_games_per_config}")

        for config_name, config_models in by_config.items():
            config_models = sorted(config_models, key=lambda m: m.iteration)
            print(f"\n{config_name}:")
            print(f"  Total models: {len(config_models)}")

            # Calculate number of models to sample based on config size
            if len(config_models) <= base_models_per_config:
                # For small configs, take all models
                models_to_sample = len(config_models)
            else:
                # For larger configs, scale up sampling
                models_to_sample = max(
                    base_models_per_config,
                    min(20, len(config_models) // 5)  # Cap at 20 models, but take at least 1/5th
                )

            selected = []
            # Always include first and last
            selected.append(config_models[0])
            if len(config_models) > 1:
                selected.append(config_models[-1])

            # Sample remaining models evenly
            if len(config_models) > 2:
                remaining_to_sample = models_to_sample - len(selected)
                if remaining_to_sample > 0:
                    step = max(1, (len(config_models) - 2) // remaining_to_sample)
                    indices = range(step, len(config_models) - 1, step)
                    indices = list(indices)[:remaining_to_sample]
                    for idx in indices:
                        selected.append(config_models[idx])

            print(f"  Selected {len(selected)} models: " +
                  f"{', '.join(str(m.iteration) for m in sorted(selected, key=lambda m: m.iteration))}")
            sampled.extend(selected)

        # Calculate expected games
        configs = list(by_config.keys())
        total_matches = 0
        config_matches = defaultdict(int)
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                config1_models = len([m for m in sampled if m.config_name == configs[i]])
                config2_models = len([m for m in sampled if m.config_name == configs[j]])
                matches = config1_models * config2_models * 2  # *2 for both colors
                total_matches += matches
                config_matches[configs[i]] += matches
                config_matches[configs[j]] += matches

        print(f"\nTournament Summary:")
        print(f"Total models selected: {len(sampled)}")
        print(f"Total matches to play: {total_matches}")
        print("\nMatches per configuration:")
        for config, matches in config_matches.items():
            print(f"  {config}: {matches} matches ({matches * 2} games)")

        return sorted(sampled, key=lambda m: (m.config_name, m.iteration))

    def generate_pairings(self, models: List[ModelInfo]) -> List[Tuple[ModelInfo, ModelInfo]]:
        """Generate tournament pairings between different configurations."""
        pairings = []
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                if model1.config_name != model2.config_name:
                    pairings.append((model1, model2))
        return pairings


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run global tournament between YINSH models')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games per match')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Target games per configuration')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be done, without playing matches')
    args = parser.parse_args()

    # Get project root (parent of analysis directory)
    project_root = Path(__file__).parent.parent

    # Set up paths relative to project root
    tournament_dir = project_root / "tournaments" / "global"
    checkpoints_root = project_root / "checkpoints"

    print(f"Project root: {project_root}")
    print(f"Looking for checkpoints in: {checkpoints_root}")
    print(f"Tournament data will be stored in: {tournament_dir}")

    manager = GlobalTournamentManager(tournament_dir)

    # Scan for all models
    print("\nScanning for models...")
    all_models = manager.scan_models(checkpoints_root)

    # Sample subset for tournament
    print("\nSelecting models for tournament...")
    tournament_models = manager.sample_models_for_tournament(
        all_models,
        target_games_per_config=args.sample_size
    )

    if args.dry_run:
        # Just show what would be done
        pairings = manager.generate_pairings(tournament_models)
        print(f"\nWould play {len(pairings)} matches ({len(pairings) * args.games} total games)")
        print("First few matches would be:")
        for m1, m2 in pairings[:5]:
            print(f"  {m1.tournament_id} vs {m2.tournament_id}")
        return

    # Run tournament
    print("\nStarting tournament...")
    manager.run_tournament(tournament_models, games_per_match=args.games)

    # Print final ratings by configuration
    print("\nFinal Ratings by Configuration:")
    ratings = manager.ratings["ratings"]
    by_config = defaultdict(list)
    for model_id, rating in ratings.items():
        config = model_id.split('_iter_')[0]
        by_config[config].append(rating)

    for config, config_ratings in sorted(by_config.items()):
        avg_rating = sum(config_ratings) / len(config_ratings)
        print(f"\n{config}:")
        print(f"  Average: {avg_rating:.1f}")
        print(f"  Best: {max(config_ratings):.1f}")
        print(f"  Worst: {min(config_ratings):.1f}")
        print(f"  Models rated: {len(config_ratings)}")


if __name__ == "__main__":
    main()
