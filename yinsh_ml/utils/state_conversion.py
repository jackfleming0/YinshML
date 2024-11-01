"""Utilities for converting between different state representations."""

import numpy as np
from typing import Dict, List, Tuple
import torch
from ..game.constants import Position, Player, PieceType, VALID_POSITIONS
from ..game.board import Board


class StateConverter:
    """Handles conversion between different state representations."""

    def __init__(self):
        # Initialize position mappings
        self.pos_to_index: Dict[str, int] = {}
        self.index_to_pos: Dict[int, str] = {}
        self._initialize_position_mappings()

        # Channel indices for the ML state tensor
        self.CHANNELS = {
            'WHITE_RINGS': 0,
            'BLACK_RINGS': 1,
            'WHITE_MARKERS': 2,
            'BLACK_MARKERS': 3,
            'VALID_MOVES': 4,
            'GAME_PHASE': 5
        }

    def _initialize_position_mappings(self):
        """Create mappings between board positions and tensor indices."""
        idx = 0
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = f"{col}{row}"
                if self._is_valid_position(pos):
                    self.pos_to_index[pos] = idx
                    self.index_to_pos[idx] = pos
                    idx += 1

    def _is_valid_position(self, pos_str: str) -> bool:
        """Check if a position string represents a valid board position."""
        col = pos_str[0]
        row = int(pos_str[1:])
        return col in VALID_POSITIONS and row in VALID_POSITIONS[col]

    def board_to_tensor(self, board: Board, game_phase: int,
                        valid_moves: List[Tuple[Position, Position]] = None) -> torch.Tensor:
        """
        Convert a board state to a tensor representation.

        Args:
            board: Current board state
            game_phase: Current game phase (encoded as int)
            valid_moves: Optional list of valid moves for move mask

        Returns:
            torch.Tensor of shape (6, 11, 11) containing:
                - Channel 0: White rings
                - Channel 1: Black rings
                - Channel 2: White markers
                - Channel 3: Black markers
                - Channel 4: Valid moves mask
                - Channel 5: Game phase encoding
        """
        # Initialize tensor
        state = np.zeros((6, 11, 11), dtype=np.float32)

        # Fill piece positions
        for pos, piece in board.pieces.items():
            row = pos.row - 1
            col = ord(pos.column) - ord('A')

            if piece == PieceType.WHITE_RING:
                state[self.CHANNELS['WHITE_RINGS'], row, col] = 1
            elif piece == PieceType.BLACK_RING:
                state[self.CHANNELS['BLACK_RINGS'], row, col] = 1
            elif piece == PieceType.WHITE_MARKER:
                state[self.CHANNELS['WHITE_MARKERS'], row, col] = 1
            elif piece == PieceType.BLACK_MARKER:
                state[self.CHANNELS['BLACK_MARKERS'], row, col] = 1

        # Add valid moves mask if provided
        if valid_moves:
            for start, end in valid_moves:
                start_row = start.row - 1
                start_col = ord(start.column) - ord('A')
                end_row = end.row - 1
                end_col = ord(end.column) - ord('A')

                state[self.CHANNELS['VALID_MOVES'], start_row, start_col] = 1
                state[self.CHANNELS['VALID_MOVES'], end_row, end_col] = 1

        # Add game phase encoding
        phase_channel = np.full((11, 11), game_phase / 4.0)  # Normalize to [0, 1]
        state[self.CHANNELS['GAME_PHASE']] = phase_channel

        return torch.FloatTensor(state)

    def tensor_to_board(self, tensor: torch.Tensor) -> Board:
        """
        Convert a tensor representation back to a Board object.

        Args:
            tensor: State tensor of shape (6, 11, 11)

        Returns:
            Board object representing the state
        """
        board = Board()
        tensor = tensor.cpu().numpy()

        # Process each position
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if not self._is_valid_position(str(pos)):
                    continue

                row_idx = row - 1
                col_idx = ord(col) - ord('A')

                # Check each channel for pieces
                if tensor[self.CHANNELS['WHITE_RINGS'], row_idx, col_idx] > 0.5:
                    board.place_piece(pos, PieceType.WHITE_RING)
                elif tensor[self.CHANNELS['BLACK_RINGS'], row_idx, col_idx] > 0.5:
                    board.place_piece(pos, PieceType.BLACK_RING)
                elif tensor[self.CHANNELS['WHITE_MARKERS'], row_idx, col_idx] > 0.5:
                    board.place_piece(pos, PieceType.WHITE_MARKER)
                elif tensor[self.CHANNELS['BLACK_MARKERS'], row_idx, col_idx] > 0.5:
                    board.place_piece(pos, PieceType.BLACK_MARKER)

        return board

    def move_to_index(self, start: Position, end: Position) -> int:
        """Convert a move (start, end positions) to a flat index."""
        if str(start) not in self.pos_to_index or str(end) not in self.pos_to_index:
            raise ValueError(f"Invalid positions: {start}, {end}")

        start_idx = self.pos_to_index[str(start)]
        end_idx = self.pos_to_index[str(end)]

        # Encode as single index in the range [0, 14641)
        return start_idx * 121 + end_idx

    def index_to_move(self, index: int) -> Tuple[Position, Position]:
        """Convert a flat index back to a move (start, end positions)."""
        if not 0 <= index < 14641:
            raise ValueError(f"Invalid move index: {index}")

        start_idx = index // 121
        end_idx = index % 121

        if start_idx not in self.index_to_pos or end_idx not in self.index_to_pos:
            raise ValueError(f"Invalid position indices: {start_idx}, {end_idx}")

        start_pos = Position.from_string(self.index_to_pos[start_idx])
        end_pos = Position.from_string(self.index_to_pos[end_idx])

        return start_pos, end_pos

    def augment_state(self, state: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate valid board state augmentations through rotations and reflections.

        Args:
            state: Original state tensor (6, 11, 11)

        Returns:
            List of augmented state tensors
        """
        augmentations = []

        # Original state
        augmentations.append(state)

        # Rotations (60°, 120°, 180°, 240°, 300°)
        for k in range(1, 6):
            rotated = self._rotate_hex_board(state, k)
            augmentations.append(rotated)

        # Reflections
        reflected = torch.flip(state, [2])  # Flip horizontally
        augmentations.append(reflected)

        # Rotations of reflection
        for k in range(1, 6):
            rotated = self._rotate_hex_board(reflected, k)
            augmentations.append(rotated)

        return augmentations

    def _rotate_hex_board(self, state: torch.Tensor, k: int) -> torch.Tensor:
        """
        Rotate hexagonal board state by k * 60 degrees counterclockwise.

        Args:
            state: State tensor to rotate
            k: Number of 60° rotations (1-5)

        Returns:
            Rotated state tensor
        """
        # Convert to numpy for easier manipulation
        state_np = state.numpy()
        rotated = np.zeros_like(state_np)

        # Process each channel separately
        for channel in range(state_np.shape[0]):
            channel_state = state_np[channel]

            # Apply rotation based on hexagonal geometry
            for col in "ABCDEFGHIJK":
                for row in range(1, 12):
                    if not self._is_valid_position(f"{col}{row}"):
                        continue

                    old_col = ord(col) - ord('A')
                    old_row = row - 1

                    # Convert to cube coordinates
                    x = old_col - 5
                    z = old_row - 5
                    y = -x - z

                    # Rotate in cube coordinates
                    for _ in range(k):
                        x, y, z = -z, -x, -y

                    # Convert back to axial coordinates
                    new_col = x + 5
                    new_row = z + 5

                    if 0 <= new_col < 11 and 0 <= new_row < 11:
                        rotated[channel, new_row, new_col] = channel_state[old_row, old_col]

        return torch.FloatTensor(rotated)