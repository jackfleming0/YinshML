#!/usr/bin/env python3
"""Global Tournament Manager for YINSH with multi-folder support."""
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
import argparse
import torch

# Ensure the project root is in sys.path so that yinsh_ml package is found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom modules
from yinsh_ml.utils.elo_manager import EloTracker, MatchResult
from yinsh_ml.network.wrapper import NetworkWrapper

@dataclass
class ModelInfo:
    """Information about a specific model."""
    config_name: str      # e.g., 'value_head_config2'
    iteration: int        # iteration number
    file_path: Path       # full path to .pt file
    tournament_id: str    # e.g., 'value_head_config2_iter_10'

class GlobalTournamentManager:
    def __init__(self, tournament_dir: Path):
        self.tournament_dir = tournament_dir
        self.tournament_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.tournament_dir / "match_history.json"
        self.results_dir = self.tournament_dir / "tournament_results"
        self.results_dir.mkdir(exist_ok=True)
        # Initialize EloTracker (its data will be saved under tournament_dir/elo_data)
        elo_save_dir = self.tournament_dir / "elo_data"
        self.elo_tracker = EloTracker(save_dir=elo_save_dir)
        self.match_history = self._load_match_history()

    def _load_match_history(self):
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def scan_models(self, checkpoints_root: Path) -> list:
        models = []
        print(f"\nScanning for models in: {checkpoints_root}")
        if not checkpoints_root.exists():
            print("Error: Checkpoints directory not found!")
            return models

        # If a 'combined' subdirectory exists, use it; otherwise, use the folder directly
        combined_dir = checkpoints_root / "combined"
        root_to_scan = combined_dir if combined_dir.exists() else checkpoints_root

        config_dirs = [d for d in root_to_scan.iterdir() if d.is_dir()]
        if config_dirs:
            print(f"Found {len(config_dirs)} configuration directories:")
            for d in config_dirs:
                print(f"  - {d.name}")
            for config_dir in config_dirs:
                config_name = config_dir.name
                checkpoint_files = list(config_dir.glob("checkpoint_iteration_*.pt"))
                for cf in sorted(checkpoint_files):
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
                        print(f"Error parsing iteration from {cf.name}: {e}")
        else:
            # No subdirectories; scan for checkpoint files directly
            checkpoint_files = list(root_to_scan.glob("checkpoint_iteration_*.pt"))
            print(f"Found {len(checkpoint_files)} checkpoint files.")
            for cf in sorted(checkpoint_files):
                try:
                    iteration = int(cf.stem.split('_')[-1])
                    # Use the folder name as the config name
                    tournament_id = f"{checkpoints_root.name}_iter_{iteration}"
                    models.append(ModelInfo(
                        config_name=checkpoints_root.name,
                        iteration=iteration,
                        file_path=cf,
                        tournament_id=tournament_id
                    ))
                except ValueError as e:
                    print(f"Error parsing iteration from {cf.name}: {e}")
        print(f"\nTotal models found: {len(models)}")
        if models:
            configs = sorted({m.config_name for m in models})
            print("\nConfigurations found:")
            for config in configs:
                config_models = [m for m in models if m.config_name == config]
                iterations = ', '.join(str(m.iteration) for m in sorted(config_models, key=lambda m: m.iteration))
                print(f"  {config}: {len(config_models)} models (iterations: {iterations})")
        return sorted(models, key=lambda m: (m.config_name, m.iteration))

    def sample_models_for_tournament(self, models: list, target_games_per_config: int = 50) -> list:
        """Sample models to get reasonable tournament size with good coverage."""
        sampled = []
        by_config = defaultdict(list)
        for model in models:
            by_config[model.config_name].append(model)
        base_models_per_config = 4  # Minimum models to sample
        print(f"\nSampling Strategy:")
        print(f"Target games per config: {target_games_per_config}")
        for config_name, config_models in by_config.items():
            config_models = sorted(config_models, key=lambda m: m.iteration)
            print(f"\n{config_name}:")
            print(f"  Total models: {len(config_models)}")
            if len(config_models) <= base_models_per_config:
                models_to_sample = len(config_models)
            else:
                models_to_sample = max(base_models_per_config, min(20, len(config_models) // 5))
            selected = []
            # Always include first and last models
            selected.append(config_models[0])
            if len(config_models) > 1:
                selected.append(config_models[-1])
            # Evenly sample the remaining models
            if len(config_models) > 2:
                remaining_to_sample = models_to_sample - len(selected)
                if remaining_to_sample > 0:
                    step = max(1, (len(config_models) - 2) // remaining_to_sample)
                    indices = list(range(step, len(config_models) - 1, step))[:remaining_to_sample]
                    for idx in indices:
                        selected.append(config_models[idx])
            print(f"  Selected {len(selected)} models: " +
                  f"{', '.join(str(m.iteration) for m in sorted(selected, key=lambda m: m.iteration))}")
            sampled.extend(selected)
        # Print tournament summary (optional)
        configs = list(by_config.keys())
        total_matches = 0
        config_matches = defaultdict(int)
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                config1_models = len([m for m in sampled if m.config_name == configs[i]])
                config2_models = len([m for m in sampled if m.config_name == configs[j]])
                matches = config1_models * config2_models * 2  # *2 for playing both colors
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

    def reset_match_history(self):
        self.match_history = []
        with open(self.history_file, 'w') as f:
            json.dump(self.match_history, f, indent=2)

    def play_match(self, white_model: ModelInfo, black_model: ModelInfo, num_games: int = 10) -> dict:
        """Play a match between two models."""
        from yinsh_ml.game.game_state import GameState
        from yinsh_ml.game.constants import Player

        try:
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
            'completed_games': 0,
            'total_games': num_games,
            'avg_game_length': 0,
            'game_lengths': [],
            'skipped': False
        }
        total_moves = 0
        max_moves = 500  # Prevent infinite games

        for game_num in range(num_games):
            game_state = GameState()
            move_count = 0
            while not game_state.is_terminal() and move_count < max_moves:
                current_model = white if game_state.current_player == Player.WHITE else black
                valid_moves = game_state.get_valid_moves()
                if not valid_moves:
                    break
                state_tensor = current_model.state_encoder.encode_state(game_state)
                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(current_model.device)
                move_probs, _ = current_model.predict(state_tensor)
                selected_move = current_model.select_move(
                    move_probs=move_probs,
                    valid_moves=valid_moves,
                    temperature=0.1
                )
                if not game_state.make_move(selected_move):
                    print(f"Warning: Invalid move in game {game_num}")
                    break
                move_count += 1
            total_moves += move_count
            results['game_lengths'].append(move_count)
            winner = game_state.get_winner()
            if winner == Player.WHITE:
                results['white_wins'] += 1
                results['completed_games'] += 1
            elif winner == Player.BLACK:
                results['black_wins'] += 1
                results['completed_games'] += 1
            if results['completed_games'] > 0:
                results['avg_game_length'] = total_moves / results['completed_games']
        if total_moves > 0:
            results['avg_game_length'] = total_moves / num_games
        return results

    def generate_pairings(self, models: list) -> list:
        pairings = []
        for i, model1 in enumerate(models):
            for model2 in models[i + 1:]:
                pairings.append((model1, model2))
        return pairings

    def run_tournament(self, sampled_models: list, games_per_match: int = 10):
        print("\nStarting Tournament:")
        self.reset_match_history()
        pairings = self.generate_pairings(sampled_models)
        total_matches = len(pairings)
        skipped_matches = []

        for idx, (model1, model2) in enumerate(pairings, 1):
            print(f"\nMatch {idx}/{total_matches}:")
            print(f"{model1.tournament_id} vs {model2.tournament_id}")
            series_start = time.time()
            results = self.play_match(model1, model2, games_per_match)
            if results.get('skipped', False):
                skipped_matches.append((model1.tournament_id, model2.tournament_id))
                print("Match skipped due to an error or incompatibility")
                continue

            match_result = MatchResult(
                white_model=model1.tournament_id,
                black_model=model2.tournament_id,
                white_wins=results['white_wins'],
                black_wins=results['black_wins'],
                draws=0,
                avg_game_length=results['avg_game_length']
            )
            self.elo_tracker.update_ratings(match_result)
            white_rating = self.elo_tracker.get_rating(model1.tournament_id)
            black_rating = self.elo_tracker.get_rating(model2.tournament_id)
            elapsed = time.time() - series_start
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            print(f"Results: White wins: {results['white_wins']}, Black wins: {results['black_wins']}")
            print(f"White ELO: {white_rating:.1f}")
            print(f"Black ELO: {black_rating:.1f}")
            print(f"Series took {minutes} minutes, {seconds:02d} seconds")

        if skipped_matches:
            print("\nSkipped Matches Summary:")
            print(f"Total skipped matches: {len(skipped_matches)}")
            for white_id, black_id in skipped_matches[:5]:
                print(f"  {white_id} vs {black_id}")
            if len(skipped_matches) > 5:
                print(f"  ... and {len(skipped_matches) - 5} more")

        print("\nFinal Ratings by Configuration:")
        ratings = self.elo_tracker.ratings
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

def main():
    parser = argparse.ArgumentParser(description='Run global tournament between YINSH models')
    parser.add_argument('--games', type=int, default=10, help='Number of games per match')
    parser.add_argument('--sample-size', type=int, default=50, help='Target games per configuration')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be done, without playing matches')
    parser.add_argument('--checkpoint-folders', type=str, nargs='+', default=["combined/feb26_testing"],
                        help='List of relative paths under "checkpoints" to use for the tournament')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    tournament_dir = project_root / "tournaments" / "global"

    # Scan each specified folder and combine all models
    all_models = []
    for folder in args.checkpoint_folders:
        folder_path = project_root / "checkpoints" / folder
        # Create a temporary manager instance to scan models in each folder
        temp_manager = GlobalTournamentManager(tournament_dir)
        models = temp_manager.scan_models(folder_path)
        all_models.extend(models)

    # Initialize the main manager and sample models from the combined list
    manager = GlobalTournamentManager(tournament_dir)
    tournament_models = manager.sample_models_for_tournament(all_models, target_games_per_config=args.sample_size)

    if args.dry_run:
        pairings = manager.generate_pairings(tournament_models)
        print(f"\nWould play {len(pairings)} matches ({len(pairings) * args.games} total games)")
        print("First few matches would be:")
        for m1, m2 in pairings[:5]:
            print(f"  {m1.tournament_id} vs {m2.tournament_id}")
        return

    print("\nStarting tournament...")
    manager.run_tournament(tournament_models, games_per_match=args.games)

if __name__ == "__main__":
    main()