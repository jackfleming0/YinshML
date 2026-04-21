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
#pease work
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
        # Row-completion bookkeeping. Always present (None when not active) so
        # pool-reuse paths that call __init__ don't inherit stale markers.
        self._move_maker: Optional[Player] = None
        self._prev_player: Optional[Player] = None
        self._last_regular_player: Optional[Player] = None

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
        
        # Copy row-completion bookkeeping. These are always present on a
        # properly-initialized GameState (None when not active), so we can
        # use plain attribute access instead of a hasattr/delattr dance.
        self._move_maker = getattr(source, '_move_maker', None)
        self._prev_player = getattr(source, '_prev_player', None)
        self._last_regular_player = getattr(source, '_last_regular_player', None)

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
            pass
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
            pass
            pass
            pass

            if self.phase != GamePhase.ROW_COMPLETION:
                pass
                return False

            if not move.markers or len(move.markers) != 5:
                pass
                return False

            if self.board.is_valid_marker_sequence(move.markers, move.player):
                success = self._handle_marker_removal(move)
                pass
            else:
                pass
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
            pass
            return False

        if move.type == MoveType.PLACE_RING:
            #logger.debug("\nValidating PLACE_RING move:")  # Debug
            #logger.debug(f"1. Phase check: current={self.phase}, expected={GamePhase.RING_PLACEMENT}")  # Debug
            if self.phase != GamePhase.RING_PLACEMENT:
                pass
                return False

            #ogger.debug(f"2. Source position check: {move.source}")  # Debug
            if not move.source:
                pass
                return False
            if not is_valid_position(move.source):
                pass
                return False

            #logger.debug(f"3. Empty position check")  # Debug
            current_piece = self.board.get_piece(move.source)
            #logger.debug(f"Current piece at position: {current_piece}")  # Debug
            if current_piece is not None:
                pass
                return False

            #logger.debug(f"4. Ring count check: {self.rings_placed[move.player]}/{RINGS_PER_PLAYER}")  # Debug
            if self.rings_placed[move.player] >= RINGS_PER_PLAYER:
                pass
                return False

            # logger.debug("All validation checks passed!")  # Debug
            return True

        elif move.type == MoveType.MOVE_RING:
            if self.phase != GamePhase.MAIN_GAME:
                pass
                return False

            # Check source has correct ring
            source_piece = self.board.get_piece(move.source)
            expected_ring = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
            if not source_piece or source_piece != expected_ring:
                pass
                return False

            # Check destination is empty
            if self.board.get_piece(move.destination) is not None:
                pass
                return False

            # Use board's valid_move_positions to check if destination is valid
            valid_destinations = self.board.valid_move_positions(move.source)
            if move.destination not in valid_destinations:
                pass
                pass
                return False

            return True

        elif move.type == MoveType.REMOVE_MARKERS:
            if self.phase != GamePhase.ROW_COMPLETION:
                pass
                return False
            if not move.markers or len(move.markers) != 5:
                pass
                return False
            return self.board.is_valid_marker_sequence(move.markers, move.player)

        elif move.type == MoveType.REMOVE_RING:
            if self.phase != GamePhase.RING_REMOVAL:
                pass
                return False
            ring_type = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
            return self.board.get_piece(move.source) == ring_type

        return False

    def get_winner(self) -> Optional['Player']:
        """Get the winner of the game, if any.

        Returns the winning player on a score-based terminal, or the opponent
        of the current player on a stalemate (current player has no legal
        moves in a non-GAME_OVER phase and therefore loses).
        """
        from .constants import Player
        if self.white_score >= 3:
            return Player.WHITE
        if self.black_score >= 3:
            return Player.BLACK
        if self.is_stalemate():
            return self.current_player.opponent
        return None

    def is_stalemate(self) -> bool:
        """Check if the current player has no legal moves in a non-GAME_OVER
        phase. Expensive (calls ``get_valid_moves``) — do not put on hot paths
        like MCTS node evaluation. Callers that terminate games (supervisor,
        tournament runner, get_winner) use this on demand; hot-path code uses
        ``is_terminal``."""
        if self.phase == GamePhase.GAME_OVER:
            return False
        return not self.get_valid_moves()

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (phase == GAME_OVER).

        **Hot path.** Kept O(1): MCTS and negamax call this per node.
        Stalemate detection lives in ``is_stalemate`` — it calls
        ``get_valid_moves``, which is expensive, and must not be invoked per
        simulation. Game-ending loops should either check ``is_stalemate``
        after each move (once per ply, not per simulation) or rely on the
        max-depth guard plus ``get_winner`` at game termination.
        """
        return self.phase == GamePhase.GAME_OVER

    def _handle_ring_placement(self, move: Move) -> bool:
        """Handle ring placement during setup phase."""
        # logger.debug(f"\nHandling ring placement: {move}")  # Debug
        ring_type = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
        # logger.debug(f"Ring type to place: {ring_type}")  # Debug
        # logger.debug(f"Current board state before placement:")  # Debug
        pass

        if not move.source or not is_valid_position(move.source):
            pass
            return False

        if self.board.get_piece(move.source) is not None:
            pass
            return False

        if self.rings_placed[move.player] >= RINGS_PER_PLAYER:
            pass
            return False

        # Place the ring
        place_success = self.board.place_piece(move.source, ring_type)
        # logger.debug(f"Place piece result: {place_success}")  # Debug
        if not place_success:
            pass
            return False

        # Update ring count
        self.rings_placed[move.player] += 1
        #logger.debug(f"Successfully placed {ring_type} at {move.source}")  # Debug
        #logger.debug(f"Updated rings placed: {self.rings_placed}")  # Debug

        # Check if we need to transition to main game
        if all(count == RINGS_PER_PLAYER for count in self.rings_placed.values()):
            pass
            self.phase = GamePhase.MAIN_GAME

        # logger.debug(f"Final board state after placement:")  # Debug
        pass
        return True



    def _handle_ring_movement(self, move: 'Move') -> bool:
        """Handle ring movement and marker placement."""
        return self.board.move_ring(move.source, move.destination)

    def _handle_marker_removal(self, move: Move) -> bool:
        """Handle marker removal after completing a row.

        Validates ALL markers before removing any, so that a malformed move
        (e.g. 4 valid markers + 1 empty square) leaves the board unchanged
        rather than half-applied.
        """
        if len(move.markers) != 5:
            pass
            return False

        expected_marker = (PieceType.WHITE_MARKER if move.player == Player.WHITE
                           else PieceType.BLACK_MARKER)

        # First pass: validate every target is a same-color marker.
        for pos in move.markers:
            old_piece = self.board.get_piece(pos)
            if old_piece is None:
                pass
                return False
            if not old_piece.is_marker():
                pass
                return False
            if old_piece != expected_marker:
                pass
                return False

        # Second pass: all validated, safe to mutate.
        for pos in move.markers:
            self.board.remove_piece(pos)

        return True

    def _handle_ring_removal(self, move: Move) -> bool:
        """Handle ring removal after completing a row."""
        pass
        pass
        pass

        ring_type = (PieceType.WHITE_RING if move.player == Player.WHITE
                     else PieceType.BLACK_RING)

        current_piece = self.board.get_piece(move.source)
        #logger.debug(f"Expected ring type: {ring_type}")
        #logger.debug(f"Found piece: {current_piece}")

        if current_piece != ring_type:
            pass
            return False

        self.board.remove_piece(move.source)
        pass

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
            pass
            # Clear any in-flight row-completion bookkeeping so a pool-reused
            # GameState doesn't inherit stale helpers from the previous game.
            self._move_maker = None
            self._prev_player = None
            self._last_regular_player = None
            return

        # Handle ring placement phase
        if self.phase == GamePhase.RING_PLACEMENT:
            if all(self.rings_placed[p] == RINGS_PER_PLAYER for p in Player):
                self.phase = GamePhase.MAIN_GAME
                pass

            return

        # Find completed rows
        white_rows = self.board.find_marker_rows(PieceType.WHITE_MARKER)
        black_rows = self.board.find_marker_rows(PieceType.BLACK_MARKER)
        #logger.debug(f"Found {len(white_rows)} white rows and {len(black_rows)} black rows")

        completed_rows = white_rows + black_rows
        if completed_rows:
            pass
            for row in completed_rows:
                pass

        # Track phase change for player management
        starting_phase = self.phase

        # State transition logic
        if self.phase == GamePhase.MAIN_GAME and completed_rows:
            pass
            self.phase = GamePhase.ROW_COMPLETION
            # Store who made the original move that created the rows
            if self._move_maker is None:
                self._move_maker = self.current_player
            # Set player who gets to remove markers
            self._set_row_completion_player(white_rows, black_rows)

        elif self.phase == GamePhase.ROW_COMPLETION:
            pass
            self.phase = GamePhase.RING_REMOVAL
            # Keep the same player for ring removal
            # No player change needed

        elif self.phase == GamePhase.RING_REMOVAL:
            if completed_rows:
                pass
                self.phase = GamePhase.ROW_COMPLETION
                # Determine who gets to remove the next row
                self._set_row_completion_player(white_rows, black_rows)
            else:
                prev_phase = self.phase
                pass
                self.phase = GamePhase.MAIN_GAME
                # Set current player to opponent of the original move maker
                if self._move_maker is not None:
                    self.current_player = self._move_maker.opponent
                    pass
                    # Clear row-completion bookkeeping for next sequence
                    self._move_maker = None
                    self._prev_player = None
                    self._last_regular_player = None
                return  # Important: return here to prevent additional player switches

    def _set_row_completion_player(self, white_rows: List['Row'], black_rows: List['Row']):
        """Set the player who gets to complete the next row.
        When both players have rows, the move maker gets priority."""
        move_maker = self._move_maker

        # If we have a move maker and both colors have rows,
        # the move maker gets to go first
        if move_maker and white_rows and black_rows:
            pass
            self.current_player = move_maker
        # Otherwise fall back to original logic
        elif white_rows:
            pass
            self.current_player = Player.WHITE
        elif black_rows:
            pass
            self.current_player = Player.BLACK

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