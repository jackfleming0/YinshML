import random
from pathlib import Path
import sys
import logging
from typing import List, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, Position, PieceType, is_valid_position
from yinsh_ml.game.moves import Move, MoveType


class ComputerPlayer:
    def __init__(self, player: Player):
        self.player = player

    def choose_ring_placement(self, game_state: GameState) -> Move:
        """Randomly choose a valid position to place a ring."""
        valid_positions = []
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if is_valid_position(pos) and game_state.board.get_piece(pos) is None:
                    valid_positions.append(pos)

        if not valid_positions:
            raise ValueError("No valid positions for ring placement")

        chosen_pos = random.choice(valid_positions)
        return Move(
            type=MoveType.PLACE_RING,
            player=self.player,
            source=chosen_pos
        )

    def choose_ring_movement(self, game_state: GameState) -> Move:
        """Randomly choose a ring to move and a valid destination."""
        # Find all rings belonging to this player
        ring_type = PieceType.WHITE_RING if self.player == Player.WHITE else PieceType.BLACK_RING
        ring_positions = []

        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if is_valid_position(pos):
                    piece = game_state.board.get_piece(pos)
                    if piece == ring_type:
                        ring_positions.append(pos)

        if not ring_positions:
            raise ValueError(f"No rings found for {self.player}")

        # Keep trying different rings until we find one with valid moves
        random.shuffle(ring_positions)
        for ring_pos in ring_positions:
            valid_moves = game_state.board.valid_move_positions(ring_pos)
            if valid_moves:
                destination = random.choice(valid_moves)
                return Move(
                    type=MoveType.MOVE_RING,
                    player=self.player,
                    source=ring_pos,
                    destination=destination
                )

        raise ValueError("No valid moves found for any ring")

    def choose_marker_removal(self, game_state: GameState) -> Move:
        """Choose markers to remove when a row is completed."""
        marker_type = PieceType.WHITE_MARKER if self.player == Player.WHITE else PieceType.BLACK_MARKER
        rows = game_state.board.find_marker_rows(marker_type)

        if not rows:
            raise ValueError("No valid marker rows found")

        # Take the first valid row of exactly 5 markers
        for row in rows:
            if len(row.positions) >= 5:
                return Move(
                    type=MoveType.REMOVE_MARKERS,
                    player=self.player,
                    markers=row.positions[:5]  # Take first 5 markers
                )

        raise ValueError("No valid marker sequences found")

    def choose_ring_removal(self, game_state: GameState) -> Move:
        """Choose a ring to remove after completing a row."""
        ring_type = PieceType.WHITE_RING if self.player == Player.WHITE else PieceType.BLACK_RING
        ring_positions = []

        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if is_valid_position(pos):
                    piece = game_state.board.get_piece(pos)
                    if piece == ring_type:
                        ring_positions.append(pos)

        if not ring_positions:
            raise ValueError("No rings available for removal")

        chosen_ring = random.choice(ring_positions)
        return Move(
            type=MoveType.REMOVE_RING,
            player=self.player,
            source=chosen_ring
        )


def play_computer_game(max_moves: int = 500):
    """Simulate a game between two computer players."""
    game_state = GameState()
    white_player = ComputerPlayer(Player.WHITE)
    black_player = ComputerPlayer(Player.BLACK)
    move_count = 0

    print("Starting new game between computer players")

    try:
        while move_count < max_moves and game_state.phase != GamePhase.GAME_OVER:
            current_player = white_player if game_state.current_player == Player.WHITE else black_player

            print(f"\nMove {move_count + 1}")
            print(f"Current phase: {game_state.phase}")
            print(f"Current player: {current_player.player}")

            try:
                # Choose and make move based on game phase
                if game_state.phase == GamePhase.RING_PLACEMENT:
                    move = current_player.choose_ring_placement(game_state)
                elif game_state.phase == GamePhase.MAIN_GAME:
                    move = current_player.choose_ring_movement(game_state)
                elif game_state.phase == GamePhase.ROW_COMPLETION:
                    move = current_player.choose_marker_removal(game_state)
                elif game_state.phase == GamePhase.RING_REMOVAL:
                    move = current_player.choose_ring_removal(game_state)
                else:
                    raise ValueError(f"Unexpected game phase: {game_state.phase}")

                print(f"Making move: {move}")
                success = game_state.make_move(move)

                if not success:
                    print(f"Move failed: {move}")
                    break

                # Only increment move count for main game moves
                if game_state.phase == GamePhase.MAIN_GAME:
                    move_count += 1

                # Print current score after each move
                print(f"Score - White: {game_state.white_score}, Black: {game_state.black_score}")

            except ValueError as e:
                print(f"Error making move: {e}")
                break

            # Periodically print board state (e.g., every 10 moves)
            if move_count % 10 == 0:
                print("\nCurrent board state:")
                print(game_state.board)

    except KeyboardInterrupt:
        print("\nGame interrupted by user")

    # Print final game state
    print("\nGame Over!")
    print(f"Total moves: {move_count}")
    print(f"Final score - White: {game_state.white_score}, Black: {game_state.black_score}")
    print("\nFinal board state:")
    print(game_state.board)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Start the game
    play_computer_game(500)  # Allow up to 500 moves total