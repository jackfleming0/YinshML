"""Interactive YINSH game between human player and trained model."""

import sys
from pathlib import Path
import logging
from typing import Optional, Tuple
import torch
import argparse
import time

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

        # Print model's evaluation
        print(f"\nModel evaluation: {value:.2f}")
        print("Top 3 considered moves:")
        sorted_moves = sorted(zip(valid_moves, valid_probs),
                            key=lambda x: x[1], reverse=True)[:3]
        for move, prob in sorted_moves:
            print(f"  {move}: {prob:.1%}")

        return selected_move

class HumanYinshPlayer:
    def __init__(self):
        self.selected_ring_pos = None
        self.valid_moves = []

    def parse_position(self, pos_str: str) -> Optional[Position]:
        """Parse position string into Position object."""
        try:
            col = pos_str[0].upper()
            row = int(pos_str[1:])
            pos = Position(col, row)
            if not is_valid_position(pos):
                print(f"Invalid position: {pos_str}")
                return None
            return pos
        except (IndexError, ValueError):
            print(f"Invalid position format: {pos_str}")
            return None

    def choose_move(self, game_state: GameState) -> Optional[Move]:
        """Get move from human player."""
        while True:
            # Show available moves based on game phase
            print("\nAvailable moves:")
            if game_state.phase == GamePhase.RING_PLACEMENT:
                print("Place a ring (e.g., 'place E5')")
            elif game_state.phase == GamePhase.MAIN_GAME:
                if not self.selected_ring_pos:
                    print("Select a ring to move (e.g., 'select E5')")
                else:
                    print(f"Selected ring at {self.selected_ring_pos}")
                    print("Choose destination (e.g., 'move H5')")
                    if self.valid_moves:
                        print("Valid moves:", ", ".join(str(pos) for pos in self.valid_moves))
            elif game_state.phase == GamePhase.ROW_COMPLETION:
                print("Select 5 markers to remove (e.g., 'markers E5 E6 E7 E8 E9')")
            elif game_state.phase == GamePhase.RING_REMOVAL:
                print("Remove a ring (e.g., 'remove E5')")

            try:
                command = input("\nEnter move: ").strip().lower()
                parts = command.split()

                if not parts:
                    continue

                cmd = parts[0]

                if cmd == 'quit':
                    return None

                elif cmd == 'help':
                    self.print_help()
                    continue

                elif cmd == 'place' and len(parts) == 2:
                    pos = self.parse_position(parts[1])
                    if pos:
                        return Move(
                            type=MoveType.PLACE_RING,
                            player=game_state.current_player,
                            source=pos
                        )

                elif cmd == 'select' and len(parts) == 2:
                    pos = self.parse_position(parts[1])
                    if pos:
                        piece = game_state.board.get_piece(pos)
                        if not piece or not piece.is_ring():
                            print(f"No ring at position {pos}")
                            continue

                        ring_type = (PieceType.WHITE_RING
                                   if game_state.current_player == Player.WHITE
                                   else PieceType.BLACK_RING)
                        if piece != ring_type:
                            print(f"Can't select opponent's ring")
                            continue

                        # Store selected position and valid moves
                        self.selected_ring_pos = pos
                        self.valid_moves = game_state.board.valid_move_positions(pos)
                        continue

                elif cmd == 'move' and len(parts) == 2:
                    if not self.selected_ring_pos:
                        print("No ring selected! Use 'select <pos>' first")
                        continue

                    dest = self.parse_position(parts[1])
                    if dest:
                        if dest not in self.valid_moves:
                            print("Invalid destination! Must be one of:",
                                  ", ".join(str(p) for p in self.valid_moves))
                            continue

                        move = Move(
                            type=MoveType.MOVE_RING,
                            player=game_state.current_player,
                            source=self.selected_ring_pos,
                            destination=dest
                        )
                        self.selected_ring_pos = None
                        self.valid_moves = []
                        return move

                elif cmd == 'markers' and len(parts) == 6:
                    positions = [self.parse_position(pos) for pos in parts[1:]]
                    if not all(positions):
                        continue

                    return Move(
                        type=MoveType.REMOVE_MARKERS,
                        player=game_state.current_player,
                        markers=positions
                    )

                elif cmd == 'remove' and len(parts) == 2:
                    pos = self.parse_position(parts[1])
                    if pos:
                        return Move(
                            type=MoveType.REMOVE_RING,
                            player=game_state.current_player,
                            source=pos
                        )

                else:
                    print("Invalid command! Type 'help' for available commands.")

            except Exception as e:
                print(f"Error: {e}")
                continue

    def print_help(self):
        """Print available commands."""
        print("\nAvailable commands:")
        print("place <pos>        - Place a ring (e.g., 'place E5')")
        print("select <pos>       - Select a ring to move (e.g., 'select E5')")
        print("move <pos>         - Move selected ring to position (e.g., 'move H5')")
        print("markers <pos>...   - Select markers for removal (e.g., 'markers E5 E6 E7 E8 E9')")
        print("remove <pos>       - Remove a ring after completing a row (e.g., 'remove E5')")
        print("help              - Show this help message")
        print("quit              - Exit the game")

def play_game(model_player: ModelPlayer, human_player: HumanYinshPlayer,
              human_color: Player = Player.WHITE, think_time: float = 0.5) -> Optional[Player]:
    """Play a game between a human and the model."""
    game_state = GameState()
    move_count = 0

    print("\nStarting new game")
    print("You are playing as", "White" if human_color == Player.WHITE else "Black")

    try:
        while move_count < 500 and game_state.phase != GamePhase.GAME_OVER:
            print("\nCurrent game state:")
            print(f"Phase: {game_state.phase}")
            print(f"Current Player: {game_state.current_player}")
            print(f"Score - White: {game_state.white_score}, Black: {game_state.black_score}")
            print("\nBoard:")
            print(game_state.board)

            # Determine current player
            is_human_turn = game_state.current_player == human_color
            current_player = human_player if is_human_turn else model_player

            try:
                if is_human_turn:
                    print("\nYour turn!")
                else:
                    if think_time > 0:
                        print("\nModel is thinking...")
                        time.sleep(think_time)

                move = current_player.choose_move(game_state)
                if move is None:  # Human player quit
                    return None

                print(f"\nMaking move: {move}")
                success = game_state.make_move(move)

                if not success:
                    print(f"Invalid move: {move}")
                    if is_human_turn:
                        continue
                    else:
                        print("Model made invalid move! This should not happen.")
                        break

                # Only increment move count for main game moves
                if game_state.phase == GamePhase.MAIN_GAME:
                    move_count += 1

            except Exception as e:
                print(f"Error making move: {e}")
                if is_human_turn:
                    continue
                else:
                    break

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        return None

    # Print final game state
    print("\nGame Over!")
    print(f"Total moves: {move_count}")
    print(f"Final score - White: {game_state.white_score}, Black: {game_state.black_score}")
    print("\nFinal board state:")
    print(game_state.board)

    return game_state.get_winner()

def main():
    parser = argparse.ArgumentParser(description='Play YINSH against a trained model')

    parser.add_argument('--model', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda', 'mps'],
                      help='Device to run model on')
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Temperature for model move selection (higher = more random)')
    parser.add_argument('--color', type=str, default='white',
                      choices=['white', 'black'],
                      help='Color for human player')
    parser.add_argument('--think-time', type=float, default=0.5,
                      help='Artificial thinking time for model in seconds (0 for instant moves)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create players
    model_player = ModelPlayer(args.model, args.device, args.temperature)
    human_player = HumanYinshPlayer()

    # Play game
    human_color = Player.WHITE if args.color.lower() == 'white' else Player.BLACK
    winner = play_game(model_player, human_player, human_color, args.think_time)

    if winner is None:
        print("\nGame abandoned.")
    else:
        print("\nWinner:", "White" if winner == Player.WHITE else "Black")
        if winner == human_color:
            print("Congratulations! You won!")
        else:
            print("The model won this time!")

if __name__ == "__main__":
    main()