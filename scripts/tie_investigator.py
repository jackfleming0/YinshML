"""Script for investigating tie conditions in YINSH model games."""

import sys
from pathlib import Path
import torch
import argparse
import time
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, Position, PieceType, is_valid_position
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.encoding import StateEncoder


class ModelPlayer:
    """AI player using a trained model."""

    def __init__(self, model_path: str, device: str = 'cpu', temperature: float = 0.5):
        self.network = NetworkWrapper(model_path=model_path, device=device)
        self.encoder = StateEncoder()
        self.temperature = temperature
        self.move_history = []

    def choose_move(self, game_state: GameState) -> Move:
        """Choose a move using the trained model."""
        # Encode current state
        state_tensor = self.encoder.encode_state(game_state)
        state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0)

        # Get valid moves
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")

        # Get move probabilities from model
        move_probs, value = self.network.predict(state_tensor)

        # Filter for valid moves
        valid_indices = [self.encoder.move_to_index(move) for move in valid_moves]
        valid_probs = move_probs[valid_indices]

        # Apply temperature
        if self.temperature != 0:
            valid_probs = torch.pow(valid_probs, 1.0 / self.temperature)

        # Normalize probabilities
        valid_probs = valid_probs / valid_probs.sum()

        # Select move
        selected_idx = torch.multinomial(valid_probs, 1).item()
        selected_move = valid_moves[selected_idx]

        # Store move in history
        self.move_history.append({
            'move': selected_move,
            'phase': game_state.phase,
            'evaluation': value.item(),
            'top_probs': list(zip(valid_moves[:3], valid_probs[:3].tolist()))
        })

        return selected_move


def visualize_board(game_state: GameState, save_path: Optional[str] = None):
    """Visualize the board state with option to save."""
    plt.clf()

    # Create grid
    for i in range(11):
        for j in range(11):
            pos = Position(chr(ord('A') + j), i + 1)
            if is_valid_position(pos):
                plt.plot(j, 10 - i, '.', color='gray', markersize=10)

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
    plt.title(f'Phase: {game_state.phase}\nCurrent Player: {game_state.current_player}\n' +
              f'Score - White: {game_state.white_score}, Black: {game_state.black_score}')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.draw()
        plt.pause(0.01)


def play_game(player1, player2, delay: float = 0.5, save_tie: bool = False) -> Tuple[Optional[Player], int, GameState]:
    """Play a game between two players. Returns winner, move count, and final state."""
    game_state = GameState()
    move_count = 0
    max_moves = 500
    moves_since_last_score = 0  # Track moves since last scoring

    print("\nStarting new game")
    plt.ion()
    plt.figure(figsize=(10, 10))

    # Clear move history
    player1.move_history.clear()
    player2.move_history.clear()

    try:
        while move_count < max_moves and game_state.phase != GamePhase.GAME_OVER:
            current_player = player1 if game_state.current_player == Player.WHITE else player2

            # Track scores before move
            prev_white_score = game_state.white_score
            prev_black_score = game_state.black_score

            # Show current state
            visualize_board(game_state)

            # Get and make move
            try:
                move = current_player.choose_move(game_state)
                success = game_state.make_move(move)

                if not success:
                    print(f"Move failed: {move}")
                    break

                # Check if any scoring occurred
                if (game_state.white_score > prev_white_score or
                        game_state.black_score > prev_black_score):
                    moves_since_last_score = 0
                else:
                    moves_since_last_score += 1

                # Only increment move count for main game moves
                if game_state.phase == GamePhase.MAIN_GAME:
                    move_count += 1

                if delay > 0:
                    plt.pause(delay)

                # Check for potential stalemate
                if moves_since_last_score > 100:  # Arbitrary threshold
                    print("\nPossible stalemate detected - no scoring in last 100 moves")
                    break

            except Exception as e:
                print(f"Error making move: {e}")
                break

    except KeyboardInterrupt:
        print("\nGame interrupted by user")

    # Show final state
    if save_tie and game_state.get_winner() is None:
        visualize_board(game_state, "tie_game_state.png")
    else:
        visualize_board(game_state)
    plt.pause(1.0)

    plt.ioff()
    plt.close('all')

    return game_state.get_winner(), move_count, game_state


def analyze_tie_game(game_state: GameState, player1: ModelPlayer, player2: ModelPlayer):
    """Analyze and print detailed information about a tie game."""
    print("\n=== Tie Game Analysis ===")

    # Print final board state
    print("\nFinal Board State:")
    print(game_state.board)

    # Print score information
    print(f"\nFinal Scores:")
    print(f"White: {game_state.white_score}")
    print(f"Black: {game_state.black_score}")

    # Print ring counts
    white_rings = len(game_state.board.get_pieces_positions(PieceType.WHITE_RING))
    black_rings = len(game_state.board.get_pieces_positions(PieceType.BLACK_RING))
    print(f"\nRings Remaining:")
    print(f"White Rings: {white_rings}")
    print(f"Black Rings: {black_rings}")

    # Print marker counts
    white_markers = len(game_state.board.get_pieces_positions(PieceType.WHITE_MARKER))
    black_markers = len(game_state.board.get_pieces_positions(PieceType.BLACK_MARKER))
    print(f"\nMarkers on Board:")
    print(f"White Markers: {white_markers}")
    print(f"Black Markers: {black_markers}")

    # Analyze last few moves
    print("\nLast 5 Moves for Each Player:")
    for player, name in [(player1, "White"), (player2, "Black")]:
        print(f"\n{name}'s last moves:")
        for move_data in player.move_history[-5:]:
            print(f"Phase: {move_data['phase']}")
            print(f"Move: {move_data['move']}")
            print(f"Position evaluation: {move_data['evaluation']:.3f}")
            print("Top move probabilities:")
            for move, prob in move_data['top_probs']:
                print(f"  {move}: {prob:.3f}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Investigate YINSH ties')
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to first model checkpoint')
    parser.add_argument('--model2', type=str,
                        help='Path to second model checkpoint (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run models on')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for move selection')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between moves in seconds')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TieInvestigator")

    # Create players
    player1 = ModelPlayer(args.model1, args.device, args.temperature)
    player2 = ModelPlayer(args.model2 or args.model1, args.device, args.temperature)

    # Search for a tie game
    games_played = 0
    tie_found = False

    logger.info("Starting tie investigation...")

    while not tie_found:
        games_played += 1
        logger.info(f"\nPlaying game {games_played}")

        winner, moves, final_state = play_game(player1, player2, args.delay)

        if winner is None:
            logger.info("\n!!! Tie game found !!!")
            tie_found = True

            # Analyze the tie game
            analyze_tie_game(final_state, player1, player2)

            # Save visualization
            plt.figure(figsize=(12, 12))
            visualize_board(final_state, "tie_game_state.png")
            logger.info("Board state saved to 'tie_game_state.png'")

            break
        else:
            logger.info(f"Game {games_played} completed - Winner: {winner}")

    logger.info(f"\nInvestigation complete after {games_played} games")


if __name__ == "__main__":
    main()