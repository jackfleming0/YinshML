"""Zobrist hashing utilities for Yinsh board states.

This module provides a complete Zobrist hashing implementation for Yinsh game states,
designed to support efficient transposition table lookups in search algorithms.

Zobrist hashing is a technique for creating hash values of board game positions
by XORing together precomputed random bitstrings for each piece at each position.
This allows for efficient incremental hash updates when the board state changes,
avoiding expensive full recomputation.

Key Features:
    - Deterministic hash generation with configurable seeds
    - Incremental hash updates for efficient move tracking
    - Cryptographically secure random bitstring generation
    - Comprehensive validation and testing suite

Example Usage:
    >>> from yinsh_ml.game.zobrist import ZobristHasher
    >>> from yinsh_ml.game.board import Board
    >>> from yinsh_ml.game.constants import Position, PieceType
    >>>
    >>> # Create a hasher (uses random seed by default)
    >>> hasher = ZobristHasher(seed="my-seed")
    >>>
    >>> # Hash a board state
    >>> board = Board()
    >>> board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    >>> hash_value = hasher.hash_board(board)
    >>>
    >>> # Incrementally update hash after a move
    >>> from yinsh_ml.game.constants import Player
    >>> new_hash = hasher.place_marker(Position.from_string("F6"), Player.WHITE, hash_value)

Integration with Transposition Tables:
    The hash values produced by this module are suitable for use as keys in
    transposition tables for game tree search algorithms. The deterministic
    nature ensures that identical board states always produce the same hash,
    while the cryptographic randomness minimizes collision probability.

Performance:
    - Full board hashing: O(n) where n is number of pieces
    - Incremental updates: O(1) per position change
    - Memory: ~425 KB for the full Zobrist table (85 positions × 5 piece types × 8 bytes)

See Also:
    - `yinsh_ml.game.board.Board`: Board representation
    - `yinsh_ml.game.game_state.GameState`: Complete game state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import hashlib
import secrets

import numpy as np

from .constants import PieceType, Position, Player, VALID_POSITIONS, is_valid_position

if TYPE_CHECKING:
    from .board import Board
    from .game_state import GameState

__all__ = ["ZobristInitializer", "ZobristTable", "ZobristHasher", "DEFAULT_PIECE_ORDER"]


DEFAULT_PIECE_ORDER: Tuple[PieceType, ...] = (
    PieceType.EMPTY,
    PieceType.WHITE_RING,
    PieceType.BLACK_RING,
    PieceType.WHITE_MARKER,
    PieceType.BLACK_MARKER,
)


def _normalize_seed(seed: Optional[object]) -> Optional[bytes]:
    if seed is None:
        return None
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, (int, float)):
        return str(seed).encode("utf-8")
    return str(seed).encode("utf-8")


@dataclass(frozen=True)
class ZobristTable:
    """Immutable view over a generated Zobrist table.

    This class provides read-only access to a precomputed Zobrist hash table.
    The table maps (position, piece_type) pairs to 64-bit random values.

    Attributes:
        positions: Tuple of all valid board positions in canonical order
        piece_types: Tuple of piece types (EMPTY, WHITE_RING, BLACK_RING, etc.)
        values: NumPy array of shape (num_positions, num_piece_types) containing
               64-bit hash values

    Example:
        >>> table = ZobristInitializer(seed="test").table
        >>> value = table.get(Position.from_string("E5"), PieceType.WHITE_RING)
        >>> isinstance(value, int)
        True
    """

    positions: Tuple[Position, ...]
    piece_types: Tuple[PieceType, ...]
    values: np.ndarray  # shape == (len(positions), len(piece_types))
    _position_to_index: Optional[Dict[Position, int]] = None
    _piece_to_index: Optional[Dict[PieceType, int]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_position_to_index", {pos: idx for idx, pos in enumerate(self.positions)})
        object.__setattr__(self, "_piece_to_index", {piece: idx for idx, piece in enumerate(self.piece_types)})

    def get(self, position: Position, piece_type: PieceType) -> int:
        """Return the 64-bit value for the given position/piece combination."""
        if position not in self._position_to_index or piece_type not in self._piece_to_index:
            raise KeyError(f"Unknown combination: {position} / {piece_type}")
        row = self._position_to_index[position]
        col = self._piece_to_index[piece_type]
        return int(self.values[row, col])


class ZobristInitializer:
    """Generates and stores random bitstrings for each position/piece state.

    This class creates the random bitstring table required for Zobrist hashing.
    It generates cryptographically secure random values for each combination of
    board position and piece type, ensuring uniqueness and uniform distribution.

    Args:
        seed: Optional seed for deterministic generation. Can be int, str, or bytes.
              If None, uses cryptographically secure random generation.
        piece_types: Sequence of piece types to generate values for.
                    Defaults to all 5 piece types (EMPTY, WHITE_RING, BLACK_RING,
                    WHITE_MARKER, BLACK_MARKER).

    Attributes:
        positions: Tuple of all valid board positions
        piece_types: Tuple of piece types used
        table: ZobristTable instance containing the generated values

    Example:
        >>> # Deterministic initialization
        >>> init1 = ZobristInitializer(seed="my-seed")
        >>> init2 = ZobristInitializer(seed="my-seed")
        >>> assert init1.table.values.tobytes() == init2.table.values.tobytes()
        >>>
        >>> # Random initialization
        >>> init3 = ZobristInitializer()  # No seed = random
        >>> assert init3.table.values.tobytes() != init1.table.values.tobytes()
    """

    def __init__(
        self,
        *,
        seed: Optional[object] = None,
        piece_types: Sequence[PieceType] = DEFAULT_PIECE_ORDER,
    ) -> None:
        self._seed = _normalize_seed(seed)
        self._piece_types = tuple(piece_types)
        self._positions = self._enumerate_positions()
        self._position_to_index = {pos: idx for idx, pos in enumerate(self._positions)}
        self._piece_to_index = {piece: idx for idx, piece in enumerate(self._piece_types)}
        self._table = self._generate_table()

    @staticmethod
    def _enumerate_positions() -> Tuple[Position, ...]:
        positions: List[Position] = []
        for column in sorted(VALID_POSITIONS.keys()):
            for row in sorted(VALID_POSITIONS[column]):
                positions.append(Position(column, row))
        return tuple(positions)

    def _generate_table(self) -> np.ndarray:
        num_positions = len(self._positions)
        num_pieces = len(self._piece_types)
        table = np.zeros((num_positions, num_pieces), dtype=np.uint64)
        used_values: set[int] = set()

        for pos_idx, position in enumerate(self._positions):
            for piece_idx, piece in enumerate(self._piece_types):
                counter = 0
                while True:
                    value = self._draw_value(position, piece, counter)
                    counter += 1
                    if value == 0 or value in used_values:
                        continue
                    used_values.add(value)
                    table[pos_idx, piece_idx] = value
                    break
        return table

    def _draw_value(self, position: Position, piece: PieceType, counter: int) -> int:
        if self._seed is not None:
            hasher = hashlib.blake2b(digest_size=16)
            hasher.update(self._seed)
            hasher.update(str(position).encode("utf-8"))
            hasher.update(piece.name.encode("utf-8"))
            hasher.update(counter.to_bytes(4, byteorder="little", signed=False))
            digest = hasher.digest()
            return int.from_bytes(digest[:8], byteorder="big", signed=False)
        return secrets.randbits(64)

    @property
    def positions(self) -> Tuple[Position, ...]:
        return self._positions

    @property
    def piece_types(self) -> Tuple[PieceType, ...]:
        return self._piece_types

    @property
    def table(self) -> ZobristTable:
        return ZobristTable(self._positions, self._piece_types, self._table.copy())

    def value(self, position: Position, piece: PieceType) -> int:
        try:
            row = self._position_to_index[position]
            col = self._piece_to_index[piece]
        except KeyError as exc:
            raise KeyError(f"Unknown key for Zobrist table: {exc}") from exc
        return int(self._table[row, col])


class ZobristHasher:
    """Computes Zobrist hashes for Yinsh board and game state objects.

    This is the main interface for computing and updating Zobrist hash values.
    It provides both full board hashing and efficient incremental update methods
    for tracking hash changes as moves are made.

    Args:
        table: Optional precomputed ZobristTable. If None, creates a new table.
        seed: Optional seed for table generation (only used if table is None).
              Cannot be used together with table parameter.

    Attributes:
        empty_board_hash: Precomputed hash value for an empty board

    Example:
        >>> hasher = ZobristHasher(seed="test")
        >>> board = Board()
        >>> board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
        >>>
        >>> # Full board hashing
        >>> hash1 = hasher.hash_board(board)
        >>>
        >>> # Incremental update
        >>> hash2 = hasher.place_marker(
        ...     Position.from_string("F6"), Player.WHITE, hash1
        ... )
        >>>
        >>> # Verify incremental matches full recomputation
        >>> board.place_piece(Position.from_string("F6"), PieceType.WHITE_MARKER)
        >>> hash3 = hasher.hash_board(board)
        >>> assert hash2 == hash3

    Thread Safety:
        This class is thread-safe for read operations (hashing). However, if
        using a shared ZobristTable instance, ensure it is not modified during
        concurrent access (though ZobristTable is immutable by design).
    """

    def __init__(
        self,
        table: Optional[ZobristTable] = None,
        *,
        seed: Optional[object] = None,
    ) -> None:
        if table is None:
            initializer = ZobristInitializer(seed=seed)
            table = initializer.table
        elif seed is not None:
            raise ValueError("Provide a table or a seed, not both.")

        self._table = table
        self._empty_hash = self._compute_empty_hash()

    def _compute_empty_hash(self) -> int:
        baseline = 0
        for position in self._table.positions:
            baseline ^= self._table.get(position, PieceType.EMPTY)
        return baseline

    @property
    def empty_board_hash(self) -> int:
        """Hash value for an empty board."""
        return self._empty_hash

    def hash_board(self, board: "Board") -> int:
        """Calculate the hash of a board by XORing the relevant values."""
        current = self._empty_hash
        for position in board.pieces:
            piece = board.pieces[position]
            if piece == PieceType.EMPTY:
                continue
            current ^= self._table.get(position, PieceType.EMPTY)
            current ^= self._table.get(position, piece)
        return current

    def hash_state(self, game_state: "GameState") -> int:
        """Hash a full GameState (currently only board contents)."""
        return self.hash_board(game_state.board)

    def toggle(self, position: Position, piece: PieceType, current_hash: int) -> int:
        """XOR the hash contribution of a specific position and piece."""
        return current_hash ^ self._table.get(position, piece)

    def update_position(
        self,
        position: Position,
        old_piece: PieceType,
        new_piece: PieceType,
        current_hash: int,
    ) -> int:
        """Swap old piece for new piece at a position."""
        current_hash = self.toggle(position, old_piece, current_hash)
        current_hash = self.toggle(position, new_piece, current_hash)
        return current_hash

    def place_ring(self, position: Position, player: Player, current_hash: int) -> int:
        """Place a ring at an empty position. Returns updated hash."""
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
        return self.update_position(position, PieceType.EMPTY, ring_type, current_hash)

    def place_marker(self, position: Position, player: Player, current_hash: int) -> int:
        """Place a marker at an empty position. Returns updated hash."""
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        return self.update_position(position, PieceType.EMPTY, marker_type, current_hash)

    def remove_piece(self, position: Position, piece_type: PieceType, current_hash: int) -> int:
        """Remove a piece from a position (replaces with EMPTY). Returns updated hash."""
        return self.update_position(position, piece_type, PieceType.EMPTY, current_hash)

    def _get_path_between(self, source: Position, destination: Position) -> List[Position]:
        """Get all positions between source and destination (excluding endpoints)."""
        col_diff = ord(destination.column) - ord(source.column)
        row_diff = destination.row - source.row
        if col_diff == 0 and row_diff == 0:
            return []

        steps = max(abs(col_diff), abs(row_diff))
        dx = col_diff // steps if col_diff != 0 else 0
        dy = row_diff // steps if row_diff != 0 else 0

        positions_between = []
        current = source
        for _ in range(steps - 1):
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)

            if not is_valid_position(next_pos):
                break

            positions_between.append(next_pos)
            current = next_pos

        return positions_between

    def move_ring(
        self,
        source: Position,
        destination: Position,
        player: Player,
        current_hash: int,
        board: Optional["Board"] = None,
    ) -> int:
        """
        Update hash for a ring move operation.

        This mirrors Board.move_ring behavior:
        1. Source position: ring -> marker (same color)
        2. Intermediate positions: markers get flipped (white <-> black)
        3. Destination position: empty -> ring (same color)

        Args:
            source: Source position of the ring
            destination: Destination position for the ring
            player: Player making the move (determines ring/marker color)
            current_hash: Current hash value
            board: Optional board to check intermediate positions for markers.
                   Must reflect the state BEFORE the move. If None, assumes no markers in path.

        Returns:
            Updated hash value
        """
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER

        # Collect all updates first (read board state before any changes)
        path = self._get_path_between(source, destination)
        updates: List[Tuple[Position, PieceType, PieceType]] = []

        # 1. Source: ring -> marker
        updates.append((source, ring_type, marker_type))

        # 2. Intermediate positions: flip markers if present
        if board is not None:
            for pos in path:
                piece = board.get_piece(pos)
                if piece and piece.is_marker():
                    # Flip the marker
                    flipped = (
                        PieceType.WHITE_MARKER if piece == PieceType.BLACK_MARKER
                        else PieceType.BLACK_MARKER
                    )
                    updates.append((pos, piece, flipped))

        # 3. Destination: empty -> ring
        updates.append((destination, PieceType.EMPTY, ring_type))

        # Apply all updates
        return self.batch_update(updates, current_hash)

    def batch_update(
        self,
        updates: List[Tuple[Position, PieceType, PieceType]],
        current_hash: int,
    ) -> int:
        """
        Apply multiple position updates atomically.

        Args:
            updates: List of (position, old_piece, new_piece) tuples
            current_hash: Current hash value

        Returns:
            Updated hash value after all updates
        """
        for position, old_piece, new_piece in updates:
            current_hash = self.update_position(position, old_piece, new_piece, current_hash)
        return current_hash

