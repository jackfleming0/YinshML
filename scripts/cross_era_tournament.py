#!/usr/bin/env python3
"""
Cross-Era Tournament: Higher-power evaluation to detect long-term improvement.

Compares early checkpoints (iter 3, 9) against later checkpoints (iter 24, 27)
with more games per match to reduce noise.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.encoding import StateEncoder
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MatchResult:
    white_model: str
    black_model: str
    white_wins: int
    black_wins: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.white_wins + self.black_wins + self.draws

    @property
    def white_win_rate(self) -> float:
        return self.white_wins / self.total_games if self.total_games > 0 else 0.0


def load_model(checkpoint_path: str, device: str) -> NetworkWrapper:
    """Load a model from checkpoint."""
    network = NetworkWrapper(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        network.network.load_state_dict(checkpoint['model_state_dict'])
    else:
        network.network.load_state_dict(checkpoint)
    network.network.eval()
    return network


def play_game(white_model: NetworkWrapper, black_model: NetworkWrapper,
              max_moves: int = 300) -> int:
    """
    Play a single game between two models.
    Returns: 1 if white wins, -1 if black wins, 0 if draw
    """
    state = GameState()
    encoder = StateEncoder()
    move_count = 0

    while not state.is_terminal() and move_count < max_moves:
        current_model = white_model if state.current_player.value == 1 else black_model

        # Get valid moves
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            break

        # Get policy from network
        encoded = encoder.encode_state(state)
        encoded_tensor = torch.FloatTensor(encoded).unsqueeze(0).to(current_model.device)

        with torch.no_grad():
            policy, _ = current_model.network(encoded_tensor)
            policy = policy.squeeze().cpu().numpy()

        # Select best valid move
        valid_indices = [encoder.move_to_index(m) for m in valid_moves]
        valid_probs = policy[valid_indices]
        best_idx = np.argmax(valid_probs)
        selected_move = valid_moves[best_idx]

        state.make_move(selected_move)
        move_count += 1

    # Determine winner
    if state.is_terminal():
        score_diff = state.white_score - state.black_score
        if score_diff > 0:
            return 1  # White wins
        elif score_diff < 0:
            return -1  # Black wins
    return 0  # Draw


def run_match(model_a: NetworkWrapper, model_b: NetworkWrapper,
              name_a: str, name_b: str, games_per_side: int = 50) -> Tuple[MatchResult, MatchResult]:
    """
    Run a match between two models, playing both colors.
    Returns results for (A as white, A as black).
    """
    # A as White, B as Black
    a_white_wins = 0
    b_black_wins = 0
    draws_1 = 0

    for i in range(games_per_side):
        result = play_game(model_a, model_b)
        if result == 1:
            a_white_wins += 1
        elif result == -1:
            b_black_wins += 1
        else:
            draws_1 += 1
        if (i + 1) % 10 == 0:
            print(f"  {name_a} (W) vs {name_b} (B): {i+1}/{games_per_side} games")

    result_1 = MatchResult(name_a, name_b, a_white_wins, b_black_wins, draws_1)

    # B as White, A as Black
    b_white_wins = 0
    a_black_wins = 0
    draws_2 = 0

    for i in range(games_per_side):
        result = play_game(model_b, model_a)
        if result == 1:
            b_white_wins += 1
        elif result == -1:
            a_black_wins += 1
        else:
            draws_2 += 1
        if (i + 1) % 10 == 0:
            print(f"  {name_b} (W) vs {name_a} (B): {i+1}/{games_per_side} games")

    result_2 = MatchResult(name_b, name_a, b_white_wins, a_black_wins, draws_2)

    return result_1, result_2


def main():
    parser = argparse.ArgumentParser(description='Cross-era tournament evaluation')
    parser.add_argument('--run-dir', type=str,
                        default='runs/20260212_091343',
                        help='Training run directory')
    parser.add_argument('--games-per-side', type=int, default=50,
                        help='Games per side per match (total = 2x this)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (mps, cuda, cpu)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('CrossEraTournament')

    run_dir = Path(args.run_dir)

    # Define checkpoints to compare
    checkpoints = {
        'iter_3': run_dir / 'iteration_3' / 'checkpoint_iteration_3.pt',
        'iter_9': run_dir / 'iteration_9' / 'checkpoint_iteration_9.pt',
        'iter_12': run_dir / 'iteration_12' / 'checkpoint_iteration_12.pt',
        'iter_24': run_dir / 'iteration_24' / 'checkpoint_iteration_24.pt',
        'iter_27': run_dir / 'iteration_27' / 'checkpoint_iteration_27.pt',
    }

    # Verify checkpoints exist
    available = {}
    for name, path in checkpoints.items():
        if path.exists():
            available[name] = path
            logger.info(f"Found checkpoint: {name}")
        else:
            logger.warning(f"Missing checkpoint: {name} at {path}")

    if len(available) < 2:
        logger.error("Need at least 2 checkpoints for tournament")
        return

    # Load models
    logger.info(f"Loading models on {args.device}...")
    models = {}
    for name, path in available.items():
        logger.info(f"Loading {name}...")
        models[name] = load_model(str(path), args.device)

    # Define matchups: early vs late
    early = ['iter_3', 'iter_9']
    late = ['iter_24', 'iter_27']

    # Filter to available models
    early = [m for m in early if m in models]
    late = [m for m in late if m in models]

    logger.info(f"Early models: {early}")
    logger.info(f"Late models: {late}")

    # Run cross-era matches
    results = []

    print("\n" + "="*60)
    print("CROSS-ERA TOURNAMENT")
    print(f"Games per matchup: {args.games_per_side * 2} ({args.games_per_side} per side)")
    print("="*60 + "\n")

    for early_model in early:
        for late_model in late:
            print(f"\n--- {early_model} vs {late_model} ---")
            r1, r2 = run_match(
                models[early_model], models[late_model],
                early_model, late_model,
                games_per_side=args.games_per_side
            )
            results.append(r1)
            results.append(r2)

    # Also run late vs late to check consistency
    if len(late) >= 2:
        print(f"\n--- {late[0]} vs {late[1]} (late vs late) ---")
        r1, r2 = run_match(
            models[late[0]], models[late[1]],
            late[0], late[1],
            games_per_side=args.games_per_side
        )
        results.append(r1)
        results.append(r2)

    # Aggregate results by model
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    model_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})

    for r in results:
        model_stats[r.white_model]['wins'] += r.white_wins
        model_stats[r.white_model]['losses'] += r.black_wins
        model_stats[r.white_model]['draws'] += r.draws

        model_stats[r.black_model]['wins'] += r.black_wins
        model_stats[r.black_model]['losses'] += r.white_wins
        model_stats[r.black_model]['draws'] += r.draws

    print(f"\n{'Model':<12} {'Wins':>6} {'Losses':>6} {'Draws':>6} {'Win%':>8}")
    print("-" * 44)

    for model in sorted(model_stats.keys(), key=lambda m: int(m.split('_')[1])):
        stats = model_stats[model]
        total = stats['wins'] + stats['losses'] + stats['draws']
        win_rate = stats['wins'] / total * 100 if total > 0 else 0
        print(f"{model:<12} {stats['wins']:>6} {stats['losses']:>6} {stats['draws']:>6} {win_rate:>7.1f}%")

    # Head-to-head summary
    print("\n" + "="*60)
    print("HEAD-TO-HEAD (Early vs Late)")
    print("="*60)

    early_wins = 0
    late_wins = 0
    draws = 0

    for r in results:
        is_early_white = r.white_model in early
        is_late_white = r.white_model in late
        is_early_black = r.black_model in early
        is_late_black = r.black_model in late

        if is_early_white and is_late_black:
            early_wins += r.white_wins
            late_wins += r.black_wins
            draws += r.draws
        elif is_late_white and is_early_black:
            late_wins += r.white_wins
            early_wins += r.black_wins
            draws += r.draws

    total = early_wins + late_wins + draws
    if total > 0:
        print(f"\nEarly models (iter 3, 9): {early_wins} wins ({early_wins/total*100:.1f}%)")
        print(f"Late models (iter 24, 27): {late_wins} wins ({late_wins/total*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total*100:.1f}%)")

        if late_wins > early_wins:
            improvement = (late_wins - early_wins) / total * 100
            print(f"\n✅ Late models show +{improvement:.1f}% improvement over early models")
        elif early_wins > late_wins:
            regression = (early_wins - late_wins) / total * 100
            print(f"\n⚠️ Early models outperform late models by +{regression:.1f}%")
        else:
            print(f"\n⚖️ No significant difference between early and late models")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
