import numpy as np
from typing import Tuple, Dict, List, FrozenSet
import logging

from ..game.constants import (
    Position,
    Player,
    PieceType,
    is_valid_position,
    RINGS_PER_PLAYER,  # Added this import
    DIRECTIONS,
    MARKERS_FOR_ROW,
)
from ..game.moves import Move, MoveType
from ..game.game_state import GameState, GamePhase

logger = logging.getLogger(__name__)

logging.getLogger('StateEncoder').setLevel(logging.DEBUG)


def _enumerate_remove_markers_lines() -> List[Tuple[Position, ...]]:
    """Enumerate every valid 5-in-a-row line on the YINSH board.

    A line is 5 consecutive valid positions along one of the 3 hex axes
    (see game.constants.DIRECTIONS — the forward-only set of hex axes).
    Each axis has two opposite directions; scanning forward-only from
    every valid starting position produces each physical line exactly
    once (since any line has a unique "smaller" endpoint under the
    forward step).

    Ordering is deterministic: (axis_index, start_col_idx, start_row).
    """
    lines: List[Tuple[Position, ...]] = []
    for axis_idx, (dcol, drow) in enumerate(DIRECTIONS):
        # Iterate starts in a deterministic order (col, then row).
        for col_idx in range(11):
            col = chr(ord('A') + col_idx)
            for row in range(1, 12):
                start = Position(col, row)
                if not is_valid_position(start):
                    continue
                line: List[Position] = []
                ok = True
                for i in range(MARKERS_FOR_ROW):
                    c_ord = ord(col) + dcol * i
                    r = row + drow * i
                    if not (ord('A') <= c_ord <= ord('K')) or not (1 <= r <= 11):
                        ok = False
                        break
                    p = Position(chr(c_ord), r)
                    if not is_valid_position(p):
                        ok = False
                        break
                    line.append(p)
                if ok and len(line) == MARKERS_FOR_ROW:
                    lines.append(tuple(line))
    return lines


# Module-level precomputed mapping for REMOVE_MARKERS sub-layout.
# Any future change to _enumerate_remove_markers_lines() output changes the
# policy-head layout — this is a BREAKING change for any saved checkpoint
# trained under the previous layout. See NetworkWrapper.load_model for the
# fail-loudly guard that trips on policy-head size mismatch.
_REMOVE_MARKERS_LINES: Tuple[Tuple[Position, ...], ...] = tuple(
    _enumerate_remove_markers_lines()
)
_LINE_TO_INDEX: Dict[FrozenSet[Position], int] = {
    frozenset(line): idx for idx, line in enumerate(_REMOVE_MARKERS_LINES)
}
_REMOVE_MARKERS_COUNT: int = len(_REMOVE_MARKERS_LINES)


# Off-board cell reserved for the current-player sentinel on channel 5.
# A1 is off the hex board (column A only has rows 2..5), so writes to
# (row_idx=0, col_idx=0) never collide with real on-board state. This is
# how decode_state recovers side-to-move after encode_state's side
# normalization swapped channels 0↔1 and 2↔3.
_CURRENT_PLAYER_ROW = 0   # row_idx for "A1"
_CURRENT_PLAYER_COL = 0   # col_idx for "A1"
_CURRENT_PLAYER_WHITE_SENTINEL = 0.0
_CURRENT_PLAYER_BLACK_SENTINEL = 1.0
# Cell we read for the phase. Must be a VALID on-board cell that is NOT
# the current-player sentinel cell. F6 (row_idx=5, col_idx=5) is the
# board center and always on-board.
_PHASE_READ_ROW = 5
_PHASE_READ_COL = 5


class StateEncoder:
    """
    Handles encoding and decoding of YINSH game states and moves for the neural network.
    """

    # Channel layout for the basic 6-channel encoding. Named so that
    # downstream code (e.g., trainer's phase-aware sampling) can refer to
    # the phase channel by name instead of a magic index. Mirrors the
    # 15-channel encoder's `CH_GAME_PHASE = 12` convention; the two are
    # *not* the same number, which is exactly why magic-indexing breaks
    # silently on encoding swap (cf. trainer.py decode_phase bug,
    # 2026-05-26).
    NUM_CHANNELS = 6
    CH_CURRENT_RINGS = 0
    CH_OPPONENT_RINGS = 1
    CH_CURRENT_MARKERS = 2
    CH_OPPONENT_MARKERS = 3
    CH_VALID_MOVES = 4
    CH_GAME_PHASE = 5

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

        # Policy-head slot layout (total = 7433 slots):
        #
        #   placement:      [0,     85)    —  85 slots, one per valid position (collision-free)
        #   ring movement:  [85,    7225)  — 7140 slots, src_idx·84 + adjusted_dst_idx (collision-free)
        #   ring removal:   [7225,  7310)  —  85 slots (collision-free)
        #   marker removal: [7310,  7433)  — 123 slots, one per valid 5-in-a-row line (collision-free)
        #
        # The ring-movement range was the big source of bugs in the legacy layout:
        # `((src*31 + dst) % 5848)` with src/dst ∈ [0,84] only produced 2687 distinct
        # values out of 7140 pairs (62% collision rate) and left 3161 slots structurally
        # unreachable. Distinct moves could share the same policy slot in MCTS training
        # targets, so the network couldn't learn to tell them apart. The new encoding
        # `src·84 + adjusted_dst` (where adjusted_dst skips the src==dst diagonal) gives
        # exactly 85·84 = 7140 slots, every one reachable, every (src,dst) pair unique.
        #
        # The REMOVE_MARKERS range was similarly broken: the legacy 1080-slot sequence
        # hash `((hash*31 + pos_idx) % 1080)` produced 17 collisions out of 123 valid
        # 5-in-a-row lines (2-way fan-in), and the inverse `index_to_move` fabricated a
        # pseudo-diagonal sequence that was often illegal, causing "Could not reconstruct"
        # errors. The new encoding enumerates the 123 hex-axis 5-lines at module load,
        # maps each to a unique slot, and inverts cleanly back to the real line.
        self.total_moves = 85 + 7140 + 85 + _REMOVE_MARKERS_COUNT

        self.ring_place_base = 0
        self.ring_place_range = (self.ring_place_base, self.ring_place_base + self.num_positions)

        # Ring movement: direct (src_idx, dst_idx) pair index with the src==dst
        # diagonal skipped. Exactly num_positions · (num_positions - 1) slots.
        self.move_ring_base = self.ring_place_range[1]
        self.move_ring_space = self.num_positions * (self.num_positions - 1)
        self.move_ring_range = (self.move_ring_base, self.move_ring_base + self.move_ring_space)

        # Ring removal: one slot per valid position.
        self.remove_ring_base = self.move_ring_range[1]
        self.remove_ring_range = (self.remove_ring_base, self.remove_ring_base + self.num_positions)

        # Marker removal: one slot per valid 5-in-a-row line. Enumerated at module
        # import in _enumerate_remove_markers_lines(); every slot is reachable and
        # inverts back to the actual line positions (vs the legacy pseudo-diagonal).
        self.remove_markers_base = self.remove_ring_range[1]
        self.remove_markers_space = _REMOVE_MARKERS_COUNT
        self.remove_markers_range = (self.remove_markers_base, self.remove_markers_base + self.remove_markers_space)

        assert self.remove_markers_range[1] == self.total_moves, (
            f"encoder layout inconsistent: last range ends at "
            f"{self.remove_markers_range[1]}, total_moves={self.total_moves}"
        )

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
        """Encode the game state into a numerical tensor.

        Normalizes by side-to-move so channels 0/2 always represent
        the current player's rings/markers and 1/3 the opponent's.
        Channel 5 carries phase only (no player sign).
        """
        state = np.zeros((6, 11, 11), dtype=np.float32)

        try:
            # Side-normalized ring channels (0: current player, 1: opponent)
            is_white = (game_state.current_player == Player.WHITE)

            # Collect rings by absolute color
            white_ring_positions = game_state.board.get_pieces_positions(PieceType.WHITE_RING)
            black_ring_positions = game_state.board.get_pieces_positions(PieceType.BLACK_RING)

            # Map to channels based on side to move
            current_ring_positions = white_ring_positions if is_white else black_ring_positions
            opponent_ring_positions = black_ring_positions if is_white else white_ring_positions

            for pos in current_ring_positions:
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[0, row_idx, col_idx] = 1.0

            for pos in opponent_ring_positions:
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[1, row_idx, col_idx] = 1.0

            # Side-normalized marker channels (2: current player, 3: opponent)
            white_marker_positions = game_state.board.get_pieces_positions(PieceType.WHITE_MARKER)
            black_marker_positions = game_state.board.get_pieces_positions(PieceType.BLACK_MARKER)

            current_marker_positions = white_marker_positions if is_white else black_marker_positions
            opponent_marker_positions = black_marker_positions if is_white else white_marker_positions

            for pos in current_marker_positions:
                col_idx = ord(pos.column) - ord('A')
                row_idx = pos.row - 1
                if 0 <= col_idx < 11 and 0 <= row_idx < 11:
                    state[2, row_idx, col_idx] = 1.0

            for pos in opponent_marker_positions:
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

            # Channel 5: Game phase (normalized 0..1) + current-player sentinel.
            # Writing the phase uniformly first, THEN overriding the off-board
            # A1 cell (row_idx=0, col_idx=0) with a 0/1 sentinel that records
            # side-to-move. The decoder reads phase from an on-board cell
            # (F6 = row 5, col 5) and the player sentinel from A1. A1 is off
            # the hex board (col A only has rows 2..5), so this cell never
            # carries real game content, and the D2 augmentation's coord maps
            # only rewrite valid on-board cells — the sentinel survives every
            # transform.
            phase_value = float(game_state.phase.value) / float(len(GamePhase) - 1)
            state[5] = phase_value
            is_white = (game_state.current_player == Player.WHITE)
            state[5, _CURRENT_PLAYER_ROW, _CURRENT_PLAYER_COL] = (
                _CURRENT_PLAYER_WHITE_SENTINEL if is_white
                else _CURRENT_PLAYER_BLACK_SENTINEL
            )

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
                if src_idx == dst_idx:
                    raise ValueError(f"Invalid ring move: src == dst ({src_pos})")

                # Collision-free: src_idx·(num_positions-1) + adjusted_dst_idx,
                # where adjusted_dst skips the src==dst diagonal. Exactly 7140 slots
                # for 85 positions. See layout comment in __init__.
                adjusted_dst = dst_idx if dst_idx < src_idx else dst_idx - 1
                return self.move_ring_base + src_idx * (self.num_positions - 1) + adjusted_dst

            elif move.type == MoveType.REMOVE_MARKERS:
                if not move.markers or len(move.markers) != MARKERS_FOR_ROW:
                    raise ValueError(
                        f"Invalid marker removal: need exactly {MARKERS_FOR_ROW} markers"
                    )
                key = frozenset(move.markers)
                line_idx = _LINE_TO_INDEX.get(key)
                if line_idx is None:
                    raise ValueError(
                        f"REMOVE_MARKERS positions do not form a valid "
                        f"5-in-a-row hex-axis line: "
                        f"{[str(m) for m in move.markers]}"
                    )
                return self.remove_markers_base + line_idx

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
                # Ring movement — collision-free inversion. No search loop needed:
                # slot = src_idx · (num_positions-1) + adjusted_dst_idx, so
                # src_idx = slot // (num_positions-1), adjusted = slot % (num_positions-1),
                # and dst_idx = adjusted if adjusted < src_idx else adjusted + 1
                # (unskipping the diagonal).
                relative_idx = index - self.move_ring_base
                span = self.num_positions - 1
                src_idx = relative_idx // span
                adjusted_dst = relative_idx % span
                dst_idx = adjusted_dst if adjusted_dst < src_idx else adjusted_dst + 1
                src_pos = Position.from_string(list(self.position_to_index.keys())[src_idx])
                dst_pos = Position.from_string(list(self.position_to_index.keys())[dst_idx])
                return Move(type=MoveType.MOVE_RING, player=player,
                            source=src_pos, destination=dst_pos)

            elif self.remove_markers_base <= index < self.remove_markers_range[1]:
                # Reverse lookup into the precomputed line table — returns the
                # actual 5 line positions, not a fabricated diagonal.
                line_idx = index - self.remove_markers_base
                markers = _REMOVE_MARKERS_LINES[line_idx]
                return Move(type=MoveType.REMOVE_MARKERS, player=player,
                            markers=markers)

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
        """Deprecated — replaced by the collision-free line-table lookup.

        Retained as a thin shim because external callers may still import it.
        The only correct input is a valid 5-in-a-row hex line; anything else
        raises, matching the stricter behaviour of the new encoding.
        """
        if not markers or len(markers) != MARKERS_FOR_ROW:
            raise ValueError(
                f"Invalid marker sequence: need exactly {MARKERS_FOR_ROW} markers"
            )
        line_idx = _LINE_TO_INDEX.get(frozenset(markers))
        if line_idx is None:
            raise ValueError(
                "marker positions do not form a valid 5-in-a-row hex-axis line"
            )
        return self.remove_markers_base + line_idx

    def decode_state(self, state_tensor: np.ndarray) -> GameState:
        """
        Convert a side-normalized state tensor back into a GameState object.

        The state tensor is produced by encode_state, which normalizes by
        side-to-move so channels 0/2 are the CURRENT player's rings/markers
        and 1/3 are the opponent's — NOT (white, black) by absolute colour.
        Channel 5 carries a uniform phase value on every on-board cell and
        a dedicated current-player sentinel at the off-board cell A1 (see
        the encode_state comment on the sentinel slot).

        Args:
            state_tensor: numpy array of shape (6, 11, 11) containing:
                - Channel 0: Current player's rings
                - Channel 1: Opponent's rings
                - Channel 2: Current player's markers
                - Channel 3: Opponent's markers
                - Channel 4: Valid moves mask
                - Channel 5: Game phase (broadcast) + player sentinel at A1

        Returns:
            GameState object with current_player, phase, rings_placed and
            scores recovered from the tensor. Board colours are de-normalized
            back to absolute (white/black) using the recovered current_player.
        """
        pass
        game_state = GameState()

        # Recover side-to-move from the off-board sentinel cell (A1).
        # Before side-awareness, decode_state unconditionally labelled channel
        # 0 as WHITE, which silently swapped colours for every BLACK-to-move
        # state (~half of self-play samples) and poisoned augmentation, since
        # augmentation.py::_base_move_encoding calls decode_state on every
        # sample and enumerates valid_moves for the decoded board.
        sentinel = float(state_tensor[5, _CURRENT_PLAYER_ROW, _CURRENT_PLAYER_COL])
        is_white_to_move = sentinel < 0.5
        game_state.current_player = Player.WHITE if is_white_to_move else Player.BLACK

        # Channel-0/2 → current player; channel-1/3 → opponent. Route each to
        # the correct absolute colour.
        if is_white_to_move:
            current_ring_type, opponent_ring_type = PieceType.WHITE_RING, PieceType.BLACK_RING
            current_marker_type, opponent_marker_type = PieceType.WHITE_MARKER, PieceType.BLACK_MARKER
        else:
            current_ring_type, opponent_ring_type = PieceType.BLACK_RING, PieceType.WHITE_RING
            current_marker_type, opponent_marker_type = PieceType.BLACK_MARKER, PieceType.WHITE_MARKER

        for row in range(11):
            for col in range(11):
                pos = Position(chr(ord('A') + col), row + 1)
                if not is_valid_position(pos):
                    continue

                if state_tensor[0, row, col] > 0.5:
                    game_state.board.place_piece(pos, current_ring_type)
                elif state_tensor[1, row, col] > 0.5:
                    game_state.board.place_piece(pos, opponent_ring_type)
                elif state_tensor[2, row, col] > 0.5:
                    game_state.board.place_piece(pos, current_marker_type)
                elif state_tensor[3, row, col] > 0.5:
                    game_state.board.place_piece(pos, opponent_marker_type)

        # Read phase from a valid on-board cell (F6 is always on-board and is
        # not the sentinel cell). Channel 5 is otherwise uniform.
        phase_value = float(state_tensor[5, _PHASE_READ_ROW, _PHASE_READ_COL])
        num_phase_values = len(GamePhase)
        phase_idx = int(round(phase_value * (num_phase_values - 1)))
        phase_idx = max(0, min(num_phase_values - 1, phase_idx))
        pass
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

        pass
        return game_state

# ---------------------------------------------------------------------------
# Cross-encoder phase decoding
# ---------------------------------------------------------------------------

def phase_channel_index(num_channels: int) -> int:
    """Return the channel index that carries the GAME_PHASE signal for a
    given encoder. Single source of truth — keeps callers from baking
    magic numbers into their lookups (cf. the 2026-05-26 trainer bug where
    `state[5]` was read regardless of encoder, silently classifying every
    15-channel sample as RING_PLACEMENT).

    Supported: 6-channel basic encoder (CH_GAME_PHASE = 5), 15-channel
    enhanced encoder (CH_GAME_PHASE = 12). Adding a new encoder requires
    a new branch here AND in the encoder class itself.
    """
    if num_channels == StateEncoder.NUM_CHANNELS:
        return StateEncoder.CH_GAME_PHASE
    if num_channels == 15:
        # Lazy import — enhanced_encoding imports encoding at module load
        # time; the reverse import would be circular.
        from .enhanced_encoding import EnhancedStateEncoder
        return EnhancedStateEncoder.CH_GAME_PHASE
    raise ValueError(
        f"unknown encoder: {num_channels} channels (expected 6 or 15). "
        "If you added a new encoder, also update phase_channel_index() "
        "in yinsh_ml/utils/encoding.py."
    )


def decode_phase_from_state(state) -> str:
    """Decode the GAME_PHASE label from a state tensor.

    Works for both the 6-channel basic encoder and the 15-channel enhanced
    encoder by routing via `phase_channel_index`. Returns one of
    'RING_PLACEMENT', 'MAIN_GAME', 'RING_REMOVAL'.

    The phase channel encodes a uniform broadcast value in [0, 1]:
        0.0  → RING_PLACEMENT
        0.5  → MAIN_GAME
        1.0  → RING_REMOVAL
    (See encode_state / encode_state_for_phase logic in both encoders.)
    The classification thresholds here mirror the historical trainer
    implementation so loss-weight semantics stay compatible.
    """
    import numpy as np
    ch = phase_channel_index(state.shape[0])
    phase_channel = state[ch]
    avg = float(np.mean(np.abs(phase_channel)))
    if avg < 0.2:
        return "RING_PLACEMENT"
    if avg < 0.6:
        return "MAIN_GAME"
    return "RING_REMOVAL"
