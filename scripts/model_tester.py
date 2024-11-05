"""Script for testing trained YINSH models."""

import sys
from pathlib import Path
import torch
import argparse
import time
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, Position, PieceType, is_valid_position
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.encoding import StateEncoder

def analyze_game_lengths(game_results, args):
    """Analyze and visualize game lengths by outcome."""
    # Collect game lengths by outcome
    white_lengths = []
    black_lengths = []
    draw_lengths = []
    all_lengths = []

    for length, winner in game_results:
        all_lengths.append(length)
        if winner == Player.WHITE:
            white_lengths.append(length)
        elif winner == Player.BLACK:
            black_lengths.append(length)
        else:
            draw_lengths.append(length)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # First subplot: Histogram
    bins = np.linspace(min(all_lengths), max(all_lengths), 20)
    ax1.hist([white_lengths, black_lengths],
             label=[f'White Wins (avg: {np.mean(white_lengths):.1f})',
                   f'Black Wins (avg: {np.mean(black_lengths):.1f})'],
             bins=bins, alpha=0.7)
    ax1.axvline(np.mean(white_lengths), color='blue', linestyle='--', alpha=0.5)
    ax1.axvline(np.mean(black_lengths), color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Game Length (moves)')
    ax1.set_ylabel('Number of Games')
    title = f'Game Lengths Distribution\n'
    title += f'White (it.{args.model1.split("_")[-1].split(".")[0]}) '
    title += f'vs Black (it.{args.model2.split("_")[-1].split(".")[0] if args.model2 else "same"})'
    ax1.set_title(title)
    ax1.legend()

    # Add statistical annotations
    stats_text = 'Performance Metrics:\n'
    if white_lengths:
        white_mean = np.mean(white_lengths)
        white_std = np.std(white_lengths)
        stats_text += f'White Wins ({len(white_lengths)}): {white_mean:.1f} ± {white_std:.1f} moves\n'
        stats_text += f'  Fastest Win: {min(white_lengths)} moves\n'
        stats_text += f'  Slowest Win: {max(white_lengths)} moves\n'
    if black_lengths:
        black_mean = np.mean(black_lengths)
        black_std = np.std(black_lengths)
        stats_text += f'\nBlack Wins ({len(black_lengths)}): {black_mean:.1f} ± {black_std:.1f} moves\n'
        stats_text += f'  Fastest Win: {min(black_lengths)} moves\n'
        stats_text += f'  Slowest Win: {max(black_lengths)} moves'

    ax1.text(0.98, 0.98, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Second subplot: Game length over time with moving average
    game_numbers = range(1, len(all_lengths) + 1)
    colors = ['blue' if w == Player.WHITE else 'red' for w in [r[1] for r in game_results]]
    ax2.scatter(game_numbers, all_lengths, c=colors, alpha=0.6)
    ax2.plot(game_numbers, all_lengths, 'k-', alpha=0.3)

    # Add moving average
    window = 5  # 5-game moving average
    if len(all_lengths) >= window:
        moving_avg = np.convolve(all_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(all_lengths) + 1), moving_avg, 'g-',
                label=f'{window}-game moving average', alpha=0.8)

    # Add trend line
    z = np.polyfit(game_numbers, all_lengths, 1)
    p = np.poly1d(z)
    ax2.plot(game_numbers, p(game_numbers), "r--", alpha=0.8,
             label=f'Trend (slope: {z[0]:.1f} moves/game)')

    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Game Length (moves)')
    ax2.set_title('Game Lengths Over Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()

class ModelPlayer:
    """AI player using a trained model."""

    def __init__(self, model_path: str, device: str = 'cpu', temperature: float = 0.5):
        self.network = NetworkWrapper(model_path=model_path, device=device)
        self.encoder = StateEncoder()
        self.temperature = temperature

    def choose_move(self, game_state: GameState) -> Move:
        """Choose a move using the trained model."""
        # Encode current state
        state_tensor = self.encoder.encode_state(game_state)
        state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)

        # Get move probabilities and value from model
        move_probs, value = self.network.predict(state_tensor)

        # Get raw logits before softmax for analysis
        raw_probs = move_probs.clone()

        # Get valid moves
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")

        # Filter for valid moves
        valid_indices = [self.encoder.move_to_index(move) for move in valid_moves]
        valid_probs = move_probs[valid_indices]

        # Print diagnostics
        print(f"\nDiagnostics:")
        print(f"Raw value output: {value.item():.3f}")
        print(f"Temperature: {self.temperature}")
        print(f"Number of valid moves: {len(valid_moves)}")
        print(f"Raw logit range: {raw_probs.min().item():.2f} to {raw_probs.max().item():.2f}")

        # Apply temperature
        if self.temperature != 0:
            valid_probs = torch.pow(valid_probs, 1.0 / self.temperature)

        # Normalize probabilities
        valid_probs = valid_probs / valid_probs.sum()

        # Print top 3 considered moves and position evaluation
        print(f"\nModel evaluation: {value.item():.3f} (positive favors White)")
        print("Top 3 considered moves:")
        sorted_moves = sorted(zip(valid_moves, valid_probs),
                            key=lambda x: x[1], reverse=True)[:3]
        for move, prob in sorted_moves:
            print(f"  {move}: {prob:.3%}")

        # Select move
        selected_idx = torch.multinomial(valid_probs, 1).item()
        selected_move = valid_moves[selected_idx]

        return selected_move

def visualize_board(game_state: GameState):
    """Simple visualization of the board state."""
    plt.clf()

    # Create grid
    for i in range(11):
        for j in range(11):
            pos = Position(chr(ord('A') + j), i + 1)
            if is_valid_position(pos):
                plt.plot(j, 10-i, '.', color='gray', markersize=10)

    # Draw pieces
    for pos, piece in game_state.board.pieces.items():
        x = ord(pos.column) - ord('A')
        y = 10 - (pos.row - 1)

        if piece == PieceType.WHITE_RING:
            plt.plot(x, y, 'o', color='white', markersize=20, markeredgecolor='black')
        elif piece == PieceType.BLACK_RING:
            plt.plot(x, y, 'o', color='black', markersize=20)
        elif piece == PieceType.WHITE_MARKER:
            plt.plot(x, y, 's', color='white', markersize=15, markeredgecolor='black')
        elif piece == PieceType.BLACK_MARKER:
            plt.plot(x, y, 's', color='black', markersize=15)

    plt.grid(True)
    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.5, 10.5)
    plt.xticks(range(11), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
    plt.yticks(range(11), range(11, 0, -1))
    plt.title(f'Phase: {game_state.phase}\nCurrent Player: {game_state.current_player}')
    plt.draw()
    plt.pause(0.01)  # Small pause to update display

def play_game(player1, player2, delay: float = 0.5) -> Tuple[Optional[Player], int]:
    """Play a game between two players."""
    game_state = GameState()
    move_count = 0
    max_moves = 500

    print("\nStarting new game")
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 10))

    try:
        while move_count < max_moves and game_state.phase != GamePhase.GAME_OVER:
            current_player = player1 if game_state.current_player == Player.WHITE else player2

            print(f"\nMove {move_count + 1}")
            print(f"Current phase: {game_state.phase}")
            print(f"Current player: {game_state.current_player}")

            # Show current state
            visualize_board(game_state)
            print(game_state.board)

            # Get and make move
            try:
                move = current_player.choose_move(game_state)
                print(f"Making move: {move}")
                success = game_state.make_move(move)

                if not success:
                    print(f"Move failed: {move}")
                    break

                # Only increment move count for main game moves
                if game_state.phase == GamePhase.MAIN_GAME:
                    move_count += 1

                print(f"Score - White: {game_state.white_score}, Black: {game_state.black_score}")

                if delay > 0:
                    plt.pause(delay)  # Pause between moves

            except Exception as e:
                print(f"Error making move: {e}")
                break

    except KeyboardInterrupt:
        print("\nGame interrupted by user")

    # Show final state
    visualize_board(game_state)
    plt.pause(2.0)  # Longer pause for final state

    print("\nGame Over!")
    print(f"Total moves: {move_count}")
    print(f"Final score - White: {game_state.white_score}, Black: {game_state.black_score}")

    plt.ioff()
    plt.close('all')

    return game_state.get_winner(), move_count

def main():
    parser = argparse.ArgumentParser(description='Test YINSH ML models')

    parser.add_argument('--model1', type=str, required=True,
                      help='Path to first model checkpoint')
    parser.add_argument('--model2', type=str,
                      help='Path to second model checkpoint (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda', 'mps'],
                      help='Device to run models on')
    parser.add_argument('--games', type=int, default=1,
                      help='Number of games to play')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Temperature for move selection (lower = more deterministic)')
    parser.add_argument('--delay', type=float, default=0.5,
                      help='Delay between moves in seconds')
    parser.add_argument('--show-raw', action='store_true',
                      help='Show raw network outputs before temperature scaling')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create players
    player1 = ModelPlayer(args.model1, args.device, args.temperature)
    player2 = ModelPlayer(args.model2 or args.model1, args.device, args.temperature)

    # Play games
    results = {'white_wins': 0, 'black_wins': 0, 'draws': 0}
    total_moves = 0
    game_results = []  # Store (length, winner) for each game

    for game_num in range(args.games):
        print(f"\nPlaying game {game_num + 1}/{args.games}")
        winner, moves = play_game(player1, player2, args.delay)
        total_moves += moves
        game_results.append((moves, winner))

        if winner == Player.WHITE:
            results['white_wins'] += 1
        elif winner == Player.BLACK:
            results['black_wins'] += 1
        else:
            results['draws'] += 1

    # Print results
    total_games = args.games
    print("\nFinal Results:")
    print(f"=== Series Summary ({total_games} games) ===")
    print(f"Model 1 (White, iteration {args.model1.split('_')[-1].split('.')[0]}): {results['white_wins']} wins ({results['white_wins']/total_games:.1%})")
    print(f"Model 2 (Black, iteration {args.model2.split('_')[-1].split('.')[0] if args.model2 else 'same'}): {results['black_wins']} wins ({results['black_wins']/total_games:.1%})")
    print(f"Draws: {results['draws']} ({results['draws']/total_games:.1%})")
    print(f"\nAverage moves per game: {total_moves / total_games:.1f}")

    # Print clear winner
    if results['white_wins'] > results['black_wins']:
        print("\nSeries Winner: WHITE (Later model)")
    elif results['black_wins'] > results['white_wins']:
        print("\nSeries Winner: BLACK (Earlier model)")
    else:
        print("\nSeries Result: TIED")

    # Analyze game lengths
    analyze_game_lengths(game_results, args)

if __name__ == "__main__":
    main()