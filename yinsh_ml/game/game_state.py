"""Game state and encoding for YINSH."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from .constants import (
    Player, Position, PieceType, RINGS_PER_PLAYER,
    is_valid_position  # Add this
)
from .board import Board
from .types import Move, MoveType, GamePhase
from .moves import MoveGenerator  # Add this import

import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StateEncoder:
    """Handles encoding and decoding of YINSH game states for ML model."""

    # Valid board positions map (based on the hexagonal board structure)
    VALID_POSITIONS = {
        'A': range(2, 6),  # A2-A5
        'B': range(1, 8),  # B1-B7
        'C': range(1, 9),  # C1-C8
        'D': range(1, 10),  # D1-D9
        'E': range(1, 11),  # E1-E10
        'F': range(2, 11),  # F2-F10
        'G': range(2, 12),  # G2-G11
        'H': range(3, 12),  # H3-H11
        'I': range(4, 12),  # I4-I11
        'J': range(5, 12),  # J5-J11
        'K': range(7, 11),  # K7-K10
    }

    def __init__(self):
        # Initialize position mappings
        self.position_to_index = {}
        self.index_to_position = {}
        self._initialize_position_mappings()

        # Channel indices
        self.CHANNELS = {
            'WHITE_RINGS': 0,
            'BLACK_RINGS': 1,
            'WHITE_MARKERS': 2,
            'BLACK_MARKERS': 3,
            'VALID_MOVES': 4,  # Added channel for valid moves
            'GAME_PHASE': 5  # Added channel for game phase
        }

    def _initialize_position_mappings(self):
        """Create bidirectional mappings between board positions and array indices."""
        idx = 0
        for col in 'ABCDEFGHIJK':
            for row in self.VALID_POSITIONS[col]:
                pos = f"{col}{row}"
                self.position_to_index[pos] = idx
                self.index_to_position[idx] = pos
                idx += 1

    def encode_state(self, game_state) -> np.ndarray:
        """Encode game state into ML input tensor."""
        state = np.zeros((6, 11, 11), dtype=np.float32)

        # Get board state
        board_state = game_state.board.to_numpy_array()
        state[0:4] = board_state

        # Add valid moves mask
        valid_moves = game_state.get_valid_moves()
        for move in valid_moves:
            if move.source:
                x, y = self._get_2d_coords(self.position_to_index[str(move.source)])
                state[4, x, y] = 1

        # Add game phase
        phase_value = game_state.phase.value / len(GamePhase)  # Normalize to [0,1]
        state[5] = np.full((11, 11), phase_value)

        return state

    def _get_2d_coords(self, index: int) -> Tuple[int, int]:
        """Convert linear index to 2D coordinates."""
        return index // 11, index % 11

    def decode_move(self, move_probabilities: np.ndarray) -> Tuple[str, str]:
        """Decode move probabilities to source/destination positions."""
        move_matrix = move_probabilities.reshape(121, 121)
        top_k = 5
        flat_indices = np.argsort(move_matrix.ravel())[-top_k:]

        for idx in reversed(flat_indices):
            source_idx = idx // 121
            dest_idx = idx % 121
            if source_idx in self.index_to_position and dest_idx in self.index_to_position:
                return (self.index_to_position[source_idx],
                        self.index_to_position[dest_idx])

        raise ValueError("No valid moves found in probability distribution")


@dataclass
class GameState:
    """Represents the complete state of a YINSH game."""

    board: Board
    current_player: Player
    phase: GamePhase
    white_score: int
    black_score: int
    rings_placed: Dict[Player, int]
    move_history: List[Move]

    def __init__(self):
        """Initialize a new game state."""
        self.board = Board()
        self.current_player = Player.WHITE
        self.phase = GamePhase.RING_PLACEMENT
        self.white_score = 0
        self.black_score = 0
        self.rings_placed = {Player.WHITE: 0, Player.BLACK: 0}
        self.move_history = []

    def copy_from(self, source: 'GameState') -> None:
        """Efficiently copy state from another GameState instance.
        
        This method is optimized for memory pool usage and is much faster
        than using copy.deepcopy() as it directly assigns field values
        and calls the board's copy_from method.
        
        Args:
            source: GameState instance to copy from
        """
        self.board.copy_from(source.board)
        self.current_player = source.current_player
        self.phase = source.phase
        self.white_score = source.white_score
        self.black_score = source.black_score
        
        # Copy dictionary efficiently
        self.rings_placed.clear()
        self.rings_placed.update(source.rings_placed)
        
        # Copy list efficiently
        self.move_history.clear()
        self.move_history.extend(source.move_history)
        
        # Copy any temporary attributes that might exist during row completion
        for attr in ['_move_maker', '_prev_player', '_last_regular_player']:
            if hasattr(source, attr):
                setattr(self, attr, getattr(source, attr))
            elif hasattr(self, attr):
                delattr(self, attr)

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        new_state = GameState()
        new_state.copy_from(self)
        return new_state

    def make_move(self, move: Move) -> bool:
        """Execute a move and update game state."""
        # logger.debug(f"\nAttempting to validate move: {move}")

        # First validate the move
       #  logger.debug(f"\nBeginning move validation...")  # Debug
        if not self.is_valid_move(move):
            logger.debug("Move validation failed")
            return False
        # logger.debug("Move validation passed!")  # Debug

        success = False
        before_phase = self.phase  # Store phase before move
        #logger.debug(f"\nProcessing move execution...")  # Debug

        if move.type == MoveType.PLACE_RING:
            # logger.debug("Processing ring placement")
            # logger.debug(f"Current game phase: {self.phase}")  # Debug
            # logger.debug(f"Current player: {self.current_player}")  # Debug
            # logger.debug(f"Rings placed: {self.rings_placed}")  # Debug
            success = self._handle_ring_placement(move)
            # logger.debug(f"Ring placement success: {success}")

        elif move.type == MoveType.MOVE_RING:
            success = self.board.move_ring(move.source, move.destination)

        elif move.type == MoveType.REMOVE_MARKERS:
            logger.debug("\nProcessing marker removal:")
            logger.debug(f"Phase: {self.phase}")
            logger.debug(f"Number of markers: {len(move.markers if move.markers else [])}")

            if self.phase != GamePhase.ROW_COMPLETION:
                logger.debug("Wrong phase for marker removal")
                return False

            if not move.markers or len(move.markers) != 5:
                logger.debug("Wrong number of markers")
                return False

            if self.board.is_valid_marker_sequence(move.markers, move.player):
                success = self._handle_marker_removal(move)
                logger.debug(f"Marker removal success: {success}")
            else:
                logger.debug("Invalid marker sequence")
                return False

        elif move.type == MoveType.REMOVE_RING:
            success = self._handle_ring_removal(move)

        if success:
            # Add move to history
            self.move_history.append(move)

            # Update game phase
            self._update_game_phase()

            # Player switching only needed for regular moves (placing/moving rings)
            # and only if we're not entering/in a row completion sequence
            if (move.type in {MoveType.PLACE_RING}
                    or (move.type == MoveType.MOVE_RING and self.phase == GamePhase.MAIN_GAME)):
#                logger.debug(f"Switching player from {self.current_player} to {self.current_player.opponent}")
                self._switch_player()

        return success

    def _switch_player(self):
        """Switch the current player."""
        # logger.debug(f"Switching player from {self.current_player} to {self.current_player.opponent}")  # Debug
        self.current_player = self.current_player.opponent

    def get_valid_moves(self) -> List['Move']:
        """Get all valid moves for current game state."""
        return MoveGenerator.get_valid_moves(self.board, self)

    def get_ring_valid_moves(self, position: Position) -> List[Move]:
        """Get valid moves for a ring at the given position."""
        valid_moves = self.get_valid_moves()  # Call the existing get_valid_moves
        ring_moves = [
            move for move in valid_moves
            if move.type == MoveType.MOVE_RING and move.source == position
        ]
        return ring_moves

    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid."""
        #logger.debug(f"\nValidating move: {move}")
    #    logger.debug(f"Current phase: {self.phase}")
        #logger.debug(f"Current player: {self.current_player}")
        #logger.debug(f"Move player: {move.player}")
        #logger.debug(f"Move type: {move.type}")  # Debug
        #logger.debug(f"Move source: {move.source}")  # Debug

        # Basic validation
        if move.player != self.current_player:
            logger.debug("Wrong player")
            return False

        if move.type == MoveType.PLACE_RING:
            #logger.debug("\nValidating PLACE_RING move:")  # Debug
            #logger.debug(f"1. Phase check: current={self.phase}, expected={GamePhase.RING_PLACEMENT}")  # Debug
            if self.phase != GamePhase.RING_PLACEMENT:
                logger.debug("Not in ring placement phase")
                return False

            #ogger.debug(f"2. Source position check: {move.source}")  # Debug
            if not move.source:
                logger.debug("No source position provided")
                return False
            if not is_valid_position(move.source):
                logger.debug("Invalid source position")
                return False

            #logger.debug(f"3. Empty position check")  # Debug
            current_piece = self.board.get_piece(move.source)
            #logger.debug(f"Current piece at position: {current_piece}")  # Debug
            if current_piece is not None:
                logger.debug("Position already occupied")
                return False

            #logger.debug(f"4. Ring count check: {self.rings_placed[move.player]}/{RINGS_PER_PLAYER}")  # Debug
            if self.rings_placed[move.player] >= RINGS_PER_PLAYER:
                logger.debug("All rings already placed")
                return False

            # logger.debug("All validation checks passed!")  # Debug
            return True

        elif move.type == MoveType.MOVE_RING:
            if self.phase != GamePhase.MAIN_GAME:
                logger.debug("Not in main game phase")
                return False

            # Check source has correct ring
            source_piece = self.board.get_piece(move.source)
            expected_ring = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
            if not source_piece or source_piece != expected_ring:
                logger.debug(f"Invalid source piece: {source_piece}")
                return False

            # Check destination is empty and path is valid
            if self.board.get_piece(move.destination) is not None:
                logger.debug("Destination is occupied")
                return False

            if not self._check_valid_move_path(move.source, move.destination):
                logger.debug("Invalid move path")
                return False

            return True

        elif move.type == MoveType.REMOVE_MARKERS:
            if self.phase != GamePhase.ROW_COMPLETION:
                logger.debug("Not in row completion phase")
                return False
            if not move.markers or len(move.markers) != 5:
                logger.debug("Invalid number of markers")
                return False
            return self.board.is_valid_marker_sequence(move.markers, move.player)

        elif move.type == MoveType.REMOVE_RING:
            if self.phase != GamePhase.RING_REMOVAL:
                logger.debug("Not in ring removal phase")
                return False
            ring_type = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
            return self.board.get_piece(move.source) == ring_type

        return False

    def get_winner(self) -> Optional['Player']:
        """Get the winner of the game, if any."""
        from .constants import Player
        if self.white_score >= 3:
            return Player.WHITE
        if self.black_score >= 3:
            return Player.BLACK
        return None

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.phase == GamePhase.GAME_OVER

    def _handle_ring_placement(self, move: Move) -> bool:
        """Handle ring placement during setup phase."""
        # logger.debug(f"\nHandling ring placement: {move}")  # Debug
        ring_type = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
        # logger.debug(f"Ring type to place: {ring_type}")  # Debug
        # logger.debug(f"Current board state before placement:")  # Debug
        logger.debug(self.board)

        if not move.source or not is_valid_position(move.source):
            logger.debug("Invalid position")  # Debug
            return False

        if self.board.get_piece(move.source) is not None:
            logger.debug("Position already occupied")  # Debug
            return False

        if self.rings_placed[move.player] >= RINGS_PER_PLAYER:
            logger.debug("All rings already placed")  # Debug
            return False

        # Place the ring
        place_success = self.board.place_piece(move.source, ring_type)
        # logger.debug(f"Place piece result: {place_success}")  # Debug
        if not place_success:
            logger.debug("Failed to place ring")  # Debug
            return False

        # Update ring count
        self.rings_placed[move.player] += 1
        #logger.debug(f"Successfully placed {ring_type} at {move.source}")  # Debug
        #logger.debug(f"Updated rings placed: {self.rings_placed}")  # Debug

        # Check if we need to transition to main game
        if all(count == RINGS_PER_PLAYER for count in self.rings_placed.values()):
            logger.debug("Transitioning to main game")  # Debug
            self.phase = GamePhase.MAIN_GAME

        # logger.debug(f"Final board state after placement:")  # Debug
        logger.debug(self.board)
        return True

    def _check_valid_move_path(self, source: Position, destination: Position) -> bool:
        """Check if there's a valid path for ring movement."""
        # Calculate direction vector
        col_diff = ord(destination.column) - ord(source.column)
        row_diff = destination.row - source.row

        if col_diff == 0 and row_diff == 0:
            return False

        # Normalize direction
        steps = max(abs(col_diff), abs(row_diff))
        dx = col_diff // steps if col_diff != 0 else 0
        dy = row_diff // steps if row_diff != 0 else 0

        # Check each position along the path
        current = source
        found_marker = False
        for _ in range(steps):
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)

            if not is_valid_position(next_pos):
                return False

            piece = self.board.get_piece(next_pos)
            if piece:
                if piece.is_ring():
                    return False  # Can't jump over rings
                else:
                    found_marker = True  # Can jump over markers
            elif found_marker:
                # Must stop at first empty space after marker
                return next_pos == destination

            current = next_pos

        return True

    def _handle_ring_movement(self, move: 'Move') -> bool:
        """Handle ring movement and marker placement."""
        return self.board.move_ring(move.source, move.destination)

    def _handle_marker_removal(self, move: Move) -> bool:
        """Handle marker removal after completing a row."""
        #logger.debug("\nHandling marker removal:")
        #logger.debug(f"Current player: {move.player}")
        # logger.debug(f"Current phase: {self.phase}")

        if len(move.markers) != 5:
            logger.debug(f"Wrong number of markers: {len(move.markers)}")
            return False

        # Remove the markers
        #logger.debug("Removing markers:")
        for pos in move.markers:
            old_piece = self.board.get_piece(pos)
            if old_piece is None:
                logger.debug(f"No piece at {pos}")
                return False
            if not old_piece.is_marker():
                logger.debug(f"Piece at {pos} is not a marker")
                return False

            self.board.remove_piece(pos)
            #logger.debug(f"Removed {old_piece} from {pos}")

        return True

    def _handle_ring_removal(self, move: Move) -> bool:
        """Handle ring removal after completing a row."""
        logger.debug("\nHandling ring removal:")
        logger.debug(f"Player: {move.player}")
        logger.debug(f"Position: {move.source}")

        ring_type = (PieceType.WHITE_RING if move.player == Player.WHITE
                     else PieceType.BLACK_RING)

        current_piece = self.board.get_piece(move.source)
        #logger.debug(f"Expected ring type: {ring_type}")
        #logger.debug(f"Found piece: {current_piece}")

        if current_piece != ring_type:
            logger.debug("Invalid ring type for removal")
            return False

        self.board.remove_piece(move.source)
        logger.debug(f"Removed ring at {move.source}")

        if move.player == Player.WHITE:
            self.white_score += 1
            #logger.debug(f"White score increased to {self.white_score}")
        else:
            self.black_score += 1
            #logger.debug(f"Black score increased to {self.black_score}")

        return True

    def _update_game_phase(self):
        """Update game phase based on current state."""
    #    logger.debug("\nUpdating game phase...")

        # Check for game end condition
        if self.white_score >= 3 or self.black_score >= 3:
            self.phase = GamePhase.GAME_OVER
            logger.debug(f"GAME OVER detected: White score = {self.white_score}, Black score = {self.black_score}")

            return

        # Handle ring placement phase
        if self.phase == GamePhase.RING_PLACEMENT:
            if all(self.rings_placed[p] == RINGS_PER_PLAYER for p in Player):
                self.phase = GamePhase.MAIN_GAME
                logger.debug("Transitioning from RING_PLACEMENT to MAIN_GAME")

            return

        # Find completed rows
        white_rows = self.board.find_marker_rows(PieceType.WHITE_MARKER)
        black_rows = self.board.find_marker_rows(PieceType.BLACK_MARKER)
        #logger.debug(f"Found {len(white_rows)} white rows and {len(black_rows)} black rows")

        completed_rows = white_rows + black_rows
        if completed_rows:
            logger.debug(f"Found completed rows: {completed_rows}")
            for row in completed_rows:
                logger.debug(f"Row color: {row.color}, positions: {row.positions}")

        # Track phase change for player management
        starting_phase = self.phase

        # State transition logic
        if self.phase == GamePhase.MAIN_GAME and completed_rows:
            logger.debug("Switching to ROW_COMPLETION phase")
            self.phase = GamePhase.ROW_COMPLETION
            # Store who made the original move that created the rows
            if not hasattr(self, '_move_maker'):
                self._move_maker = self.current_player
            # Set player who gets to remove markers
            self._set_row_completion_player(white_rows, black_rows)

        elif self.phase == GamePhase.ROW_COMPLETION:
            logger.debug("Switching to RING_REMOVAL phase")
            self.phase = GamePhase.RING_REMOVAL
            # Keep the same player for ring removal
            # No player change needed

        elif self.phase == GamePhase.RING_REMOVAL:
            if completed_rows:
                logger.debug("More rows found, switching back to ROW_COMPLETION")
                self.phase = GamePhase.ROW_COMPLETION
                # Determine who gets to remove the next row
                self._set_row_completion_player(white_rows, black_rows)
            else:
                prev_phase = self.phase
                logger.debug("No more rows, switching back to MAIN_GAME")
                self.phase = GamePhase.MAIN_GAME
                # Set current player to opponent of the original move maker
                if hasattr(self, '_move_maker'):
                    self.current_player = self._move_maker.opponent
                    logger.debug(f"Setting current player to opponent of original move maker ({self.current_player})")
                    delattr(self, '_move_maker')

                    # Clear any stored state for next sequence
                    for attr in ['_prev_player', '_last_regular_player']:
                        if hasattr(self, attr):
                            delattr(self, attr)
                return  # Important: return here to prevent additional player switches

    def _set_row_completion_player(self, white_rows: List['Row'], black_rows: List['Row']):
        """Set the player who gets to complete the next row.
        When both players have rows, the move maker gets priority."""
        move_maker = getattr(self, '_move_maker', None)

        # If we have a move maker and both colors have rows,
        # the move maker gets to go first
        if move_maker and white_rows and black_rows:
            logger.debug(f"Both colors have rows, giving priority to move maker ({move_maker})")
            self.current_player = move_maker
        # Otherwise fall back to original logic
        elif white_rows:
            logger.debug("White rows found, setting current player to WHITE")
            self.current_player = Player.WHITE
        elif black_rows:
            logger.debug("Black rows found, setting current player to BLACK")
            self.current_player = Player.BLACK

    def _switch_player(self):
        """Switch the current player."""
        self.current_player = self.current_player.opponent

    def to_numpy_array(self) -> np.ndarray:
        """Convert game state to numpy array for ML input."""
        # Get board state array (4 channels for pieces)
        state = self.board.to_numpy_array()

        # Add game phase channel
        phase_channel = np.full((11, 11), float(self.phase.value) / 4)

        # Add current player channel
        player_channel = np.full(
            (11, 11),
            1.0 if self.current_player == self.current_player.WHITE else -1.0
        )

        # Stack all channels
        return np.vstack([
            state,
            phase_channel[np.newaxis, :, :],
            player_channel[np.newaxis, :, :]
        ])

    def __str__(self) -> str:
        return (
            f"YINSH Game State:\n"
            f"Phase: {self.phase.name}\n"
            f"Current Player: {self.current_player.name}\n"
            f"Score - White: {self.white_score}, Black: {self.black_score}\n"
            f"Rings Placed - White: {self.rings_placed[self.current_player.WHITE]}, "
            f"Black: {self.rings_placed[self.current_player.BLACK]}\n"
            f"Board:\n{self.board}"
        )