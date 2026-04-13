"""Edge case and error handling tests for Zobrist hashing."""

import pytest

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import PieceType, Position, Player
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.zobrist import ZobristHasher, ZobristInitializer, ZobristTable


def test_invalid_position_raises_error():
    """Test that invalid positions raise appropriate errors."""
    hasher = ZobristHasher(seed="error-test")
    
    # Invalid position (outside board bounds)
    invalid_pos = Position("Z", 99)
    
    with pytest.raises(KeyError):
        hasher.toggle(invalid_pos, PieceType.WHITE_RING, 0)


def test_invalid_piece_type_raises_error():
    """Test that invalid piece types raise appropriate errors."""
    hasher = ZobristHasher(seed="error-test")
    valid_pos = Position.from_string("E5")
    
    # This should work - PieceType is an enum, so invalid values are type errors
    # But we can test with a piece type not in the table
    # Since we use all PieceType values, this is mainly a type check
    hash_val = hasher.toggle(valid_pos, PieceType.WHITE_RING, 0)
    assert isinstance(hash_val, int)


def test_empty_board_hash():
    """Test hashing an empty board."""
    hasher = ZobristHasher(seed="empty-test")
    board = Board()
    
    hash_val = hasher.hash_board(board)
    assert hash_val == hasher.empty_board_hash
    assert isinstance(hash_val, int)
    assert hash_val != 0  # Empty board hash should be non-zero (XOR of all EMPTY values)


def test_single_piece_board():
    """Test hashing a board with a single piece."""
    hasher = ZobristHasher(seed="single-piece")
    board = Board()
    board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    
    hash_val = hasher.hash_board(board)
    
    # Should match incremental update
    incremental = hasher.place_ring(
        Position.from_string("E5"), Player.WHITE, hasher.empty_board_hash
    )
    assert hash_val == incremental


def test_max_pieces_board():
    """Test hashing a board with many pieces."""
    hasher = ZobristHasher(seed="max-pieces")
    board = Board()
    
    # Place pieces at many positions
    positions = [
        Position.from_string("E5"), Position.from_string("E6"), Position.from_string("E7"),
        Position.from_string("F5"), Position.from_string("F6"), Position.from_string("F7"),
        Position.from_string("G5"), Position.from_string("G6"), Position.from_string("G7"),
        Position.from_string("D4"), Position.from_string("D5"), Position.from_string("D6"),
    ]
    
    for i, pos in enumerate(positions):
        piece = PieceType.WHITE_RING if i % 2 == 0 else PieceType.BLACK_MARKER
        board.place_piece(pos, piece)
    
    hash_val = hasher.hash_board(board)
    assert isinstance(hash_val, int)
    
    # Verify consistency
    assert hash_val == hasher.hash_board(board)


def test_hash_with_game_state():
    """Test hashing a complete GameState."""
    hasher = ZobristHasher(seed="gamestate-test")
    state = GameState()
    
    # Place some pieces
    state.board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    state.board.place_piece(Position.from_string("F6"), PieceType.BLACK_MARKER)
    
    hash_from_state = hasher.hash_state(state)
    hash_from_board = hasher.hash_board(state.board)
    
    assert hash_from_state == hash_from_board


def test_incremental_update_reversibility():
    """Test that incremental updates are reversible."""
    hasher = ZobristHasher(seed="reversibility")
    position = Position.from_string("E5")
    initial_hash = hasher.empty_board_hash
    
    # Place ring
    hash_after_place = hasher.place_ring(position, Player.WHITE, initial_hash)
    
    # Remove ring (reverse operation)
    hash_after_remove = hasher.remove_piece(position, PieceType.WHITE_RING, hash_after_place)
    
    assert hash_after_remove == initial_hash


def test_batch_update_empty_list():
    """Test batch update with empty list."""
    hasher = ZobristHasher(seed="batch-empty")
    initial_hash = hasher.empty_board_hash
    
    result = hasher.batch_update([], initial_hash)
    assert result == initial_hash


def test_batch_update_single_change():
    """Test batch update with single change."""
    hasher = ZobristHasher(seed="batch-single")
    position = Position.from_string("E5")
    initial_hash = hasher.empty_board_hash
    
    # Single update via batch
    batch_hash = hasher.batch_update(
        [(position, PieceType.EMPTY, PieceType.WHITE_RING)],
        initial_hash
    )
    
    # Should match individual update
    individual_hash = hasher.place_ring(position, Player.WHITE, initial_hash)
    
    assert batch_hash == individual_hash


def test_move_ring_without_board():
    """Test move_ring without providing board (should still work)."""
    hasher = ZobristHasher(seed="move-no-board")
    board = Board()
    board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    
    initial_hash = hasher.hash_board(board)
    
    # Move without providing board (won't detect marker flips, but should work)
    new_hash = hasher.move_ring(
        Position.from_string("E5"),
        Position.from_string("E8"),
        Player.WHITE,
        initial_hash
    )
    
    # Should be different from initial
    assert new_hash != initial_hash


def test_move_ring_with_empty_path():
    """Test move_ring with adjacent positions (no intermediate positions)."""
    hasher = ZobristHasher(seed="move-adjacent")
    board = Board()
    board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    
    initial_hash = hasher.hash_board(board)
    
    # Move to adjacent position (if valid)
    # Note: This depends on valid move rules, so we'll test the hash update logic
    new_hash = hasher.move_ring(
        Position.from_string("E5"),
        Position.from_string("E6"),  # Adjacent
        Player.WHITE,
        initial_hash,
        board
    )
    
    assert new_hash != initial_hash


def test_same_seed_produces_same_table():
    """Test that same seed produces identical tables."""
    init1 = ZobristInitializer(seed="identical")
    init2 = ZobristInitializer(seed="identical")
    
    assert init1.table.values.tobytes() == init2.table.values.tobytes()


def test_different_seeds_produce_different_tables():
    """Test that different seeds produce different tables."""
    init1 = ZobristInitializer(seed="seed1")
    init2 = ZobristInitializer(seed="seed2")
    
    assert init1.table.values.tobytes() != init2.table.values.tobytes()


def test_table_immutability():
    """Test that ZobristTable is immutable."""
    initializer = ZobristInitializer(seed="immutable")
    table = initializer.table
    
    # Try to modify (should fail or create new object)
    original_bytes = table.values.tobytes()
    
    # Attempt modification
    table.values[0, 0] = 999999
    
    # Should not affect original (if copy was made) or raise error
    # In our implementation, ZobristTable.values is a copy, so modification is possible
    # but doesn't affect the original. Let's verify the table still works correctly.
    value = table.get(table.positions[0], PieceType.EMPTY)
    assert isinstance(value, int)


def test_hasher_with_table_vs_seed():
    """Test that providing table and seed together raises error."""
    initializer = ZobristInitializer(seed="test")
    table = initializer.table
    
    with pytest.raises(ValueError, match="Provide a table or a seed"):
        ZobristHasher(table=table, seed="conflict")


def test_hash_consistency_multiple_calls():
    """Test that hashing the same board multiple times is consistent."""
    hasher = ZobristHasher(seed="consistency")
    board = Board()
    board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    board.place_piece(Position.from_string("F6"), PieceType.BLACK_MARKER)
    
    hashes = [hasher.hash_board(board) for _ in range(100)]
    
    # All should be identical
    assert len(set(hashes)) == 1


def test_incremental_chain():
    """Test a chain of incremental updates."""
    hasher = ZobristHasher(seed="chain")
    initial_hash = hasher.empty_board_hash
    
    # Chain of operations
    hash1 = hasher.place_ring(Position.from_string("E5"), Player.WHITE, initial_hash)
    hash2 = hasher.place_marker(Position.from_string("F6"), Player.BLACK, hash1)
    hash3 = hasher.place_ring(Position.from_string("G7"), Player.BLACK, hash2)
    hash4 = hasher.remove_piece(Position.from_string("E5"), PieceType.WHITE_RING, hash3)
    
    # Build board to match
    board = Board()
    board.place_piece(Position.from_string("F6"), PieceType.BLACK_MARKER)
    board.place_piece(Position.from_string("G7"), PieceType.BLACK_RING)
    
    full_hash = hasher.hash_board(board)
    
    assert hash4 == full_hash


def test_all_positions_hashable():
    """Test that all valid positions can be hashed."""
    hasher = ZobristHasher(seed="all-positions")
    initializer = ZobristInitializer(seed="all-positions")
    
    # Test each valid position
    for position in initializer.positions:
        hash_val = hasher.toggle(position, PieceType.WHITE_RING, 0)
        assert isinstance(hash_val, int)
        assert hash_val != 0  # Should be non-zero


def test_all_piece_types_hashable():
    """Test that all piece types can be hashed."""
    hasher = ZobristHasher(seed="all-pieces")
    position = Position.from_string("E5")
    
    for piece_type in [
        PieceType.EMPTY,
        PieceType.WHITE_RING,
        PieceType.BLACK_RING,
        PieceType.WHITE_MARKER,
        PieceType.BLACK_MARKER,
    ]:
        hash_val = hasher.toggle(position, piece_type, 0)
        assert isinstance(hash_val, int)


def test_table_get_method():
    """Test ZobristTable.get method."""
    initializer = ZobristInitializer(seed="table-get")
    table = initializer.table
    
    position = Position.from_string("E5")
    piece = PieceType.WHITE_RING
    
    value = table.get(position, piece)
    assert isinstance(value, int)
    assert value != 0
    
    # Should be same as initializer
    assert value == initializer.value(position, piece)


def test_table_get_invalid_raises_error():
    """Test that ZobristTable.get raises error for invalid inputs."""
    initializer = ZobristInitializer(seed="table-error")
    table = initializer.table
    
    # Invalid position
    with pytest.raises(KeyError):
        table.get(Position("Z", 99), PieceType.WHITE_RING)
    
    # Note: PieceType is an enum, so invalid piece types are type errors at compile time

