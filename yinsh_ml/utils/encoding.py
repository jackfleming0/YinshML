import numpy as np
from typing import Tuple, Dict, List
import logging

from ..game.constants import (
    Position,
    Player,
    PieceType,
    is_valid_position,
    RINGS_PER_PLAYER  # Added this import
)
from ..game.moves import Move, MoveType
from ..game.game_state import GameState, GamePhase

logger = logging.getLogger(__name__)

logging.getLogger('StateEncoder').setLevel(logging.DEBUG)


class StateEncoder:
    """
    Handles encoding and decoding of YINSH game states and moves for the neural network.
    """

    def __init__(self):
        # Initialize position mappings
        self.position_to_index = {}
        index = 0
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if is_valid_position(pos):
                    self.position_to_index[str(pos)] = index
                    index += 1

        self.index_to_position = {v: k for k, v in self.position_to_index.items()}
        self.num_positions = len(self.position_to_index)

        # Fixed total size that matches network expectation
        self.total_moves = 7395

        # Calculate ranges for all move types using similar base/range structure
        self.ring_place_base = 0
        self.ring_place_range = (self.ring_place_base, self.num_positions)

        # Ring movements get 80% of remaining space
        remaining_space = self.total_moves - self.num_positions
        self.move_ring_base = self.ring_place_range[1]
        self.move_ring_space = int(remaining_space * 0.8)
        self.move_ring_range = (self.move_ring_base, self.move_ring_base + self.move_ring_space)

        # Marker removals get 15% of remaining space
        self.remove_markers_base = self.move_ring_range[1]
        self.remove_markers_space = int(remaining_space * 0.15)
        self.remove_markers_range = (self.remove_markers_base, self.remove_markers_base + self.remove_markers_space)

        # Ring removals get the rest
        self.remove_ring_base = self.remove_markers_range[1]
        self.remove_ring_range = (self.remove_ring_base, self.total_moves)

        self.logger = logging.getLogger("StateEncoder")
        # self.logger.info(f"StateEncoder initialized with {self.total_moves} total moves:")
        # self.logger.info(f"  Valid positions: {self.num_positions}")
        # self.logger.info(f"  Ring placement: {self.ring_place_range}")
        # self.logger.info(f"  Ring movement: {self.move_ring_range} (space: {self.move_ring_space})")
        # self.logger.info(f"  Marker removal: {self.remove_markers_range} (space: {self.remove_markers_space})")
        # self.logger.info(f"  Ring removal: {self.remove_ring_range}")
        self.logger.setLevel(logging.DEBUG)


    def _initialize_position_to_index(self) -> Dict[str, int]:
        """Create a mapping from position strings to unique indices."""
        position_to_index = {}
        idx = 0
        for col in 'ABCDEFGHIJK':
            for row in range(1, 12):
                pos_str = f"{col}{row}"
                if is_valid_position(Position(col, row)):
                    position_to_index[pos_str] = idx
                    idx += 1
        return position_to_index

    def _initialize_move_mappings(self):
        """Initialize the mappings between moves and indices."""
        try:
            idx = 0
            # Add ring placement moves
            for col in 'ABCDEFGHIJK':
                for row in range(1, 12):
                    pos = Position(col, row)
                    if is_valid_position(pos):
                        move = Move(type=MoveType.PLACE_RING, player=None, source=pos)
                        key = self._move_to_key(move)
                        self.move_to_idx_map[key] = idx
                        self.idx_to_move_map[idx] = move
                        idx += 1

            # Add ring movement moves
            for src_col in 'ABCDEFGHIJK':
                for src_row in range(1, 12):
                    src_pos = Position(src_col, src_row)
                    if not is_valid_position(src_pos):
                        continue
                    for dst_col in 'ABCDEFGHIJK':
                        for dst_row in range(1, 12):
                            dst_pos = Position(dst_col, dst_row)
                            if is_valid_position(dst_pos) and src_pos != dst_pos:
                                move = Move(type=MoveType.MOVE_RING, player=None,
                                            source=src_pos, destination=dst_pos)
                                key = self._move_to_key(move)
                                self.move_to_idx_map[key] = idx
                                self.idx_to_move_map[idx] = move
                                idx += 1

            # Add ring removal moves (similar to ring placement positions)
            for col in 'ABCDEFGHIJK':
                for row in range(1, 12):
                    pos = Position(col, row)
                    if is_valid_position(pos):
                        move = Move(type=MoveType.REMOVE_RING, player=None, source=pos)
                        key = self._move_to_key(move)
                        self.move_to_idx_map[key] = idx
                        self.idx_to_move_map[idx] = move
                        idx += 1

        except Exception as e:
            self.logger.error(f"Error initializing move mappings: {str(e)}")
            raise
    def _generate_all_possible_moves(self) -> List[Move]:
        """
        Generate all possible moves in the game and return them as a list.

        Returns:
            List[Move]: A list of all possible moves.
        """
        all_moves = []

        # Generate all possible PLACE_RING moves
        for pos_str in self.position_to_index.keys():
            pos = Position.from_string(pos_str)
            move = Move(
                type=MoveType.PLACE_RING,
                player=None,  # Player is assigned during game play
                source=pos
            )
            all_moves.append(move)

        # Generate all possible MOVE_RING moves
        # This requires generating all valid ring movements
        # For this example, we'll assume a method `generate_all_move_ring_moves` exists
        all_moves.extend(self.generate_all_move_ring_moves())

        # Generate all possible REMOVE_RING moves
        for pos_str in self.position_to_index.keys():
            pos = Position.from_string(pos_str)
            move = Move(
                type=MoveType.REMOVE_RING,
                player=None,
                source=pos
            )
            all_moves.append(move)

        # Generate all possible REMOVE_MARKERS moves
        # Assuming that removing a row of markers can happen between any positions
        # For simplicity, we'll not include this move type in the indexing
        # Adjust accordingly if necessary

        # Ensure the total number of moves matches 2018
        all_moves = all_moves[:self.total_moves]  # Truncate or adjust as necessary

        return all_moves

    def generate_all_move_ring_moves(self) -> List[Move]:
        """
        Generate all possible MOVE_RING moves.

        Returns:
            List[Move]: A list of all possible MOVE_RING moves.
        """
        move_ring_moves = []
        # For each position on the board
        for source_pos_str in self.position_to_index.keys():
            source_pos = Position.from_string(source_pos_str)
            # For each possible destination
            for dest_pos_str in self.position_to_index.keys():
                dest_pos = Position.from_string(dest_pos_str)
                if source_pos != dest_pos:
                    move = Move(
                        type=MoveType.MOVE_RING,
                        player=None,
                        source=source_pos,
                        destination=dest_pos
                    )
                    move_ring_moves.append(move)
                    # You may need to check for validity based on game rules
                    # For this example, we include all combinations
        return move_ring_moves

    def _move_to_key(self, move: Move) -> str:
        """Convert a move to a unique string key."""
        try:
            move_type = move.type.name
            source = str(move.source) if move.source else 'None'
            dest = str(move.destination) if move.destination else 'None'
            markers = ','.join(str(m) for m in move.markers) if move.markers else 'None'
            return f"{move_type}:{source}:{dest}:{markers}"
        except Exception as e:
            self.logger.error(f"Error creating move key: {str(e)}")
            return "INVALID"

    def encode_state(self, game_state: GameState) -> np.ndarray:
        """Encode the game state into a numerical tensor."""
        state = np.zeros((6, 11, 11), dtype=np.float32)

        try:
            # Channel 0-1: Rings
            for pos in game_state.board.get_pieces_positions(PieceType.WHITE_RING):
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[0, row_idx, col_idx] = 1.0

            for pos in game_state.board.get_pieces_positions(PieceType.BLACK_RING):
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[1, row_idx, col_idx] = 1.0

            # Channel 2-3: Markers
            for pos in game_state.board.get_pieces_positions(PieceType.WHITE_MARKER):
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[2, row_idx, col_idx] = 1.0

            for pos in game_state.board.get_pieces_positions(PieceType.BLACK_MARKER):
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[3, row_idx, col_idx] = 1.0

            # Channel 4: Valid moves
            valid_moves = game_state.get_valid_moves()
            for move in valid_moves:
                if move.source:
                    col_idx = ord(move.source.column) - ord('A')
                    row_idx = move.source.row - 1
                    if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                        state[4, row_idx, col_idx] = 1.0

            # Channel 5: Game phase and current player
            phase_value = float(game_state.phase.value) / float(len(GamePhase) - 1)
            player_value = 1.0 if game_state.current_player == Player.WHITE else -1.0
            state[5] = phase_value * player_value

        except Exception as e:
            self.logger.error(f"Error encoding state: {str(e)}")
            raise

        return state

    def decode_move(self, move_probs: np.ndarray, valid_moves: List[Move]) -> Move:
        """
        Decode move probabilities into a valid move.

        Args:
            move_probs: Network output probabilities
            valid_moves: List of currently valid moves

        Returns:
            The selected move
        """
        if not valid_moves:
            raise ValueError("No valid moves available")

        # Get probabilities for valid moves only
        valid_indices = [self.move_to_index(move) for move in valid_moves]
        valid_probs = move_probs[valid_indices]

        # Normalize probabilities
        valid_probs = valid_probs / valid_probs.sum()

        # Select move based on probabilities
        selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
        return valid_moves[selected_idx]

    def _get_2d_coords(self, index: int) -> Tuple[int, int]:
        """Convert linear index to 2D coordinates."""
        return index // 11, index % 11

    def move_to_index(self, move: Move) -> int:
        """Convert a move to its index in the policy vector."""
        try:
            if move.type == MoveType.PLACE_RING:
                pos_str = str(move.source)
                base_idx = self.position_to_index.get(pos_str, -1)
                if base_idx == -1:
                    raise ValueError(f"Invalid position for ring placement: {pos_str}")
                return self.ring_place_base + base_idx

            elif move.type == MoveType.MOVE_RING:
                src_pos = str(move.source)
                dst_pos = str(move.destination)

                if src_pos not in self.position_to_index or dst_pos not in self.position_to_index:
                    raise ValueError(f"Invalid positions: {src_pos}->{dst_pos}")

                src_idx = self.position_to_index[src_pos]
                dst_idx = self.position_to_index[dst_pos]

                # Hash the move into our allocated space
                move_hash = ((src_idx * 31 + dst_idx) % self.move_ring_space)
                return self.move_ring_base + move_hash

            elif move.type == MoveType.REMOVE_MARKERS:
                if not move.markers or len(move.markers) != 5:
                    raise ValueError(f"Invalid marker removal: need exactly 5 markers")
                return self._compute_marker_sequence_hash(move.markers)

            elif move.type == MoveType.REMOVE_RING:
                pos_str = str(move.source)
                base_idx = self.position_to_index.get(pos_str, -1)
                if base_idx == -1:
                    raise ValueError(f"Invalid position for ring removal: {pos_str}")
                return self.remove_ring_base + base_idx

            else:
                raise ValueError(f"Unsupported move type: {move.type}")

        except Exception as e:
            self.logger.error(f"Error encoding move: {str(e)}")
            self.logger.error(f"Move details: type={move.type}, source={move.source}, dest={move.destination}")
            if move.type == MoveType.REMOVE_MARKERS and move.markers:
                self.logger.error(f"Markers: {[str(m) for m in move.markers]}")
            raise

    def index_to_move(self, index: int, player: Player) -> Move:
        """Convert an index back to a Move object."""
        try:
            if self.ring_place_base <= index < self.ring_place_range[1]:
                # Ring placement
                pos = list(self.position_to_index.keys())[index - self.ring_place_base]
                return Move(type=MoveType.PLACE_RING, player=player,
                            source=Position.from_string(pos))

            elif self.move_ring_base <= index < self.move_ring_range[1]:
                # Ring movement - reconstruct from hash
                relative_idx = index - self.move_ring_base
                src_idx = (relative_idx // 31) % self.num_positions
                dst_idx = relative_idx % self.num_positions

                src_pos = Position.from_string(list(self.position_to_index.keys())[src_idx])
                dst_pos = Position.from_string(list(self.position_to_index.keys())[dst_idx])

                return Move(type=MoveType.MOVE_RING, player=player,
                            source=src_pos, destination=dst_pos)

            elif self.remove_markers_base <= index < self.remove_markers_range[1]:
                # For marker removal, reconstruct a valid sequence
                relative_idx = index - self.remove_markers_base
                base_idx = relative_idx % self.num_positions
                base_pos = Position.from_string(list(self.position_to_index.keys())[base_idx])

                # Create a diagonal sequence of 5 markers
                markers = []
                current_pos = base_pos
                for i in range(5):
                    col = chr(ord(current_pos.column) + i)
                    row = current_pos.row + i
                    if 'A' <= col <= 'K' and 1 <= row <= 11:
                        markers.append(Position(col, row))

                if len(markers) == 5:
                    return Move(type=MoveType.REMOVE_MARKERS, player=player,
                                markers=markers)
                else:
                    raise ValueError(f"Could not reconstruct valid marker sequence")

            elif self.remove_ring_base <= index < self.remove_ring_range[1]:
                # Ring removal
                pos_idx = index - self.remove_ring_base
                pos = list(self.position_to_index.keys())[pos_idx]
                return Move(type=MoveType.REMOVE_RING, player=player,
                            source=Position.from_string(pos))

            else:
                raise ValueError(f"Index {index} out of range")

        except Exception as e:
            self.logger.error(f"Error decoding move index {index}: {str(e)}")
            raise

    def _compute_marker_sequence_hash(self, markers: List[Position]) -> int:
        """Compute a deterministic hash for a sequence of marker positions."""
        # Convert positions to tuples for consistent hashing
        position_tuples = [(m.column, m.row) for m in markers]
        sorted_positions = tuple(sorted(position_tuples))

        # Compute hash based on position tuples
        hash_val = 0
        for col, row in sorted_positions:
            pos_str = f"{col}{row}"
            pos_idx = self.position_to_index.get(pos_str, 0)
            hash_val = (hash_val * 31 + pos_idx) % self.remove_markers_space

        return self.remove_markers_base + hash_val

    def decode_state(self, state_tensor: np.ndarray) -> GameState:
        """
        Convert a state tensor back into a GameState object.

        Args:
            state_tensor: numpy array of shape (6, 11, 11) containing:
                - Channel 0: White rings
                - Channel 1: Black rings
                - Channel 2: White markers
                - Channel 3: Black markers
                - Channel 4: Valid moves mask
                - Channel 5: Game phase

        Returns:
            GameState object
        """
        logger.debug(f"Decoding state tensor of shape {state_tensor.shape}")
        game_state = GameState()

        # Convert channel data back into board pieces
        for row in range(11):
            for col in range(11):
                pos = Position(chr(ord('A') + col), row + 1)
                if not is_valid_position(pos):
                    continue

                # Check each piece type
                if state_tensor[0, row, col] > 0.5:
                    game_state.board.place_piece(pos, PieceType.WHITE_RING)
                elif state_tensor[1, row, col] > 0.5:
                    game_state.board.place_piece(pos, PieceType.BLACK_RING)
                elif state_tensor[2, row, col] > 0.5:
                    game_state.board.place_piece(pos, PieceType.WHITE_MARKER)
                elif state_tensor[3, row, col] > 0.5:
                    game_state.board.place_piece(pos, PieceType.BLACK_MARKER)

        # Determine game phase from channel 5
        phase_value = np.mean(state_tensor[5])  # Take average since it's broadcast
        num_phases = len(GamePhase)
        # Ensure phase_idx is within valid range (0 to num_phases-1)
        phase_idx = max(0, min(num_phases - 1, int(phase_value * num_phases)))
        logger.debug(f"Phase calculation: value={phase_value:.2f}, index={phase_idx}")
        game_state.phase = GamePhase(phase_idx)

        # Count rings to determine rings_placed
        white_rings = len(game_state.board.get_pieces_positions(PieceType.WHITE_RING))
        black_rings = len(game_state.board.get_pieces_positions(PieceType.BLACK_RING))
        game_state.rings_placed = {
            Player.WHITE: white_rings,
            Player.BLACK: black_rings
        }

        # Calculate scores based on missing rings
        game_state.white_score = max(0, RINGS_PER_PLAYER - white_rings)
        game_state.black_score = max(0, RINGS_PER_PLAYER - black_rings)

        # Determine current player (if white rings == black rings, it's white's turn)
        if white_rings <= black_rings:
            game_state.current_player = Player.WHITE
        else:
            game_state.current_player = Player.BLACK

        logger.debug(f"Decoded state - Phase: {game_state.phase.name}, "
                     f"White rings: {white_rings}, Black rings: {black_rings}, "
                     f"White score: {game_state.white_score}, Black score: {game_state.black_score}")
        return game_state