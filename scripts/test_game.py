import sys
from pathlib import Path
import logging
from typing import Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, Position, PieceType, is_valid_position
from yinsh_ml.game.moves import Move, MoveType  # Add this line

class YinshTerminalTester:
    def __init__(self):
        self.game_state = GameState()
        self.print_help()
        self.print_game_state()
        self.selected_ring_pos = None  # Add this
        self.valid_moves = []  # Add this

    def print_help(self):
        """Print available commands."""
        print("\nAvailable commands:")
        print("place <pos>        - Place a ring (e.g., 'place E5')")
        print("select <pos>       - Select a ring to move (e.g., 'select E5')")
        print("move <pos>         - Move selected ring to position (e.g., 'move H5')")  # Updated
        print("markers <pos>...   - Select markers for removal (e.g., 'markers E5 E6 E7 E8 E9')")
        print("remove <pos>       - Remove a ring after completing a row (e.g., 'remove E5')")
        print("print             - Print current board state")
        print("test              - Set up test state with nearly-complete row")  # New command
        print("help              - Show this help message")
        print("quit              - Exit the game")

    def print_game_state(self):
        """Print current game state."""
        print("\nCurrent game state:")
        print(f"Phase: {self.game_state.phase}")
        print(f"Current Player: {self.game_state.current_player}")
        print(f"Score - White: {self.game_state.white_score}, Black: {self.game_state.black_score}")
        print(f"Rings Placed - White: {self.game_state.rings_placed[Player.WHITE]}, "
              f"Black: {self.game_state.rings_placed[Player.BLACK]}")
        print("\nBoard:")
        print(self.game_state.board)

    def handle_command(self, command: str) -> bool:
        """Handle user command. Returns False if should quit."""
        parts = command.lower().split()
        if not parts:
            return True

        cmd = parts[0]

        try:
            if cmd == 'quit':
                return False

            elif cmd == 'help':
                self.print_help()

            elif cmd == 'print':
                self.print_game_state()

            elif cmd == 'place' and len(parts) == 2:
                pos = self.parse_position(parts[1])
                if pos:
                    print(f"Attempting to place ring at {pos}")  # Debug
                    move = Move(
                        type=MoveType.PLACE_RING,
                        player=self.game_state.current_player,
                        source=pos
                    )
                    print(f"Created move: {move}")  # Debug
                    success = self.game_state.make_move(move)
                    if not success:
                        print("Invalid move - checking why:")  # Debug
                        print(f"Current phase: {self.game_state.phase}")
                        print(f"Is position valid? {is_valid_position(pos)}")
                        print(f"Is position empty? {self.game_state.board.get_piece(pos) is None}")
                        print(
                            f"Rings placed for {self.game_state.current_player}: {self.game_state.rings_placed[self.game_state.current_player]}")

            elif cmd == 'select' and len(parts) == 2:
                pos = self.parse_position(parts[1])
                if pos:
                    piece = self.game_state.board.get_piece(pos)
                    if not piece or not piece.is_ring():
                        print(f"No ring at position {pos}")
                        return True

                    if piece == PieceType.WHITE_RING and self.game_state.current_player != Player.WHITE:
                        print("Can't select white ring during black's turn")
                        return True
                    if piece == PieceType.BLACK_RING and self.game_state.current_player != Player.BLACK:
                        print("Can't select black ring during white's turn")
                        return True

                    # Store selected position
                    self.selected_ring_pos = pos

                    # Get and display valid moves
                    self.valid_moves = self.game_state.board.valid_move_positions(pos)
                    print(f"\nSelected ring at {pos}")
                    if self.valid_moves:
                        print("Valid moves:")
                        for valid_pos in self.valid_moves:
                            print(f"  {valid_pos}")
                    else:
                        print("No valid moves available for this ring")


            elif cmd == 'move' and len(parts) == 2:  # Changed to expect only destination

                if not self.selected_ring_pos:
                    print("No ring selected! Use 'select <pos>' first")

                    return True

                dest = self.parse_position(parts[1])

                if dest:
                    print(f"Moving from {self.selected_ring_pos} to {dest}")  # Debug

                    if dest not in self.valid_moves:
                        print(f"Invalid destination! Must be one of: {', '.join(str(p) for p in self.valid_moves)}")

                        return True

                    move = Move(

                        type=MoveType.MOVE_RING,

                        player=self.game_state.current_player,

                        source=self.selected_ring_pos,

                        destination=dest

                    )

                    success = self.game_state.make_move(move)

                    if not success:

                        print("Move failed!")

                    else:

                        # Clear selection after successful move

                        self.selected_ring_pos = None

                        self.valid_moves = []



            elif cmd == 'markers' and len(parts) == 6:
                positions = [self.parse_position(pos) for pos in parts[1:]]
                if not all(positions):
                    print("Invalid position format!")
                    return True

                print("\nAttempting to remove markers:")
                valid_pieces = True
                for pos in positions:
                    piece = self.game_state.board.get_piece(pos)
                    print(f"Position {pos}: {piece}")
                    if piece is None:
                        print(f"No piece at position {pos}")
                        valid_pieces = False
                        break
                    if not piece.is_marker():
                        print(f"Piece at position {pos} is not a marker")
                        valid_pieces = False
                        break
                    marker_type = (PieceType.WHITE_MARKER
                                   if self.game_state.current_player == Player.WHITE
                                   else PieceType.BLACK_MARKER)
                    if piece != marker_type:
                        print(f"Wrong marker color at {pos} (found {piece}, expected {marker_type})")
                        valid_pieces = False
                        break

                if not valid_pieces:
                    print("Invalid marker selection - wrong pieces!")
                    return True

                # Create and execute the move
                move = Move(
                    type=MoveType.REMOVE_MARKERS,
                    player=self.game_state.current_player,
                    markers=positions
                )

                print(f"\nCreated move: {move}")

                # First validate the sequence
                if not self.game_state.board.is_valid_marker_sequence(positions, self.game_state.current_player):
                    print("Invalid marker sequence - not in a row!")
                    return True

                # Execute the move
                success = self.game_state.make_move(move)
                if not success:
                    print("Failed to remove markers!")
                else:
                    print("Successfully removed markers!")

                return True

            elif cmd == 'test':
                print("Setting up test state...")
                self.setup_test_state()

            elif cmd == 'remove' and len(parts) == 2:
                pos = self.parse_position(parts[1])
                if pos:
                    success = self.game_state.make_move(
                        Move(type=MoveType.REMOVE_RING,
                             player=self.game_state.current_player,
                             source=pos)
                    )
                    if not success:
                        print("Invalid ring removal!")

            else:
                print("Invalid command! Type 'help' for available commands.")

        except Exception as e:
            print(f"Error executing command: {e}")

        # Print updated state after each command
        self.print_game_state()
        # If a ring is selected, remind player of valid moves
        if self.selected_ring_pos:
            print(f"\nCurrently selected: {self.selected_ring_pos}")
            print("Valid moves:", ', '.join(str(p) for p in self.valid_moves))

        return True


    def setup_test_state(self):
        """Set up a test state with rings placed and a nearly-complete row."""
        # Reset game state
        self.game_state = GameState()

        # Define ring positions for both players
        ring_positions = [
            ('E5', Player.WHITE),
            ('F6', Player.BLACK),
            ('E7', Player.WHITE),
            ('F8', Player.BLACK),
            ('E9', Player.WHITE),
            ('F10', Player.BLACK),
            ('G5', Player.WHITE),
            ('H5', Player.BLACK),
            ('C3', Player.WHITE),
            ('D4', Player.BLACK)
        ]

        # Place rings alternating between players
        for pos, player in ring_positions:
            self.game_state.current_player = player
            move = Move(
                type=MoveType.PLACE_RING,
                player=player,
                source=self.parse_position(pos)
            )
            success = self.game_state.make_move(move)
            print(f"Placing {player} ring at {pos}: {'Success' if success else 'Failed'}")  # Debug

        # Set up a specific position with a nearly-complete row
        # Place some markers to set up a potential row
        marker_positions = {
            'WHITE_MARKER': ['E1', 'E2', 'E3', 'E4'],
            'BLACK_MARKER': ['F2', 'F3', 'F4']
        }

        for pos in marker_positions['WHITE_MARKER']:
            self.game_state.board.place_piece(self.parse_position(pos), PieceType.WHITE_MARKER)
            print(f"Placed white marker at {pos}")  # Debug
        for pos in marker_positions['BLACK_MARKER']:
            self.game_state.board.place_piece(self.parse_position(pos), PieceType.BLACK_MARKER)
            print(f"Placed black marker at {pos}")  # Debug

        # Make sure it's White's turn and in MAIN_GAME phase
        self.game_state.current_player = Player.WHITE
        self.game_state.phase = GamePhase.MAIN_GAME
        print("Test state setup complete")  # Debug

    def print_game_state(self):
        """Print current game state."""
        print("\nCurrent game state:")
        print(f"Phase: {self.game_state.phase}")
        print(f"Current Player: {self.game_state.current_player}")
        print(f"Score - White: {self.game_state.white_score}, Black: {self.game_state.black_score}")
        print(f"Rings Placed - White: {self.game_state.rings_placed[Player.WHITE]}, "
              f"Black: {self.game_state.rings_placed[Player.BLACK]}")
        print("\nBoard:")
        print(self.game_state.board)

        # Add instructions for row completion
        if self.game_state.phase == GamePhase.ROW_COMPLETION:
            print("\nYou have completed a row! Use the 'markers' command to select 5 markers to remove.")
            print("Example: markers E1 E2 E3 E4 E5")
        print(self.game_state.board)

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



    def run(self):
        """Main game loop."""
        while True:
            try:
                command = input("\nEnter command (or 'help'): ").strip()
                if not self.handle_command(command):
                    break
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create and run game
    tester = YinshTerminalTester()
    tester.run()

if __name__ == "__main__":
    main()