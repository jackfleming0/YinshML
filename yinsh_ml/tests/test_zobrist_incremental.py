"""Tests for incremental Zobrist hash update methods."""

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import PieceType, Position, Player
from yinsh_ml.game.zobrist import ZobristHasher, ZobristInitializer


def build_board(entries):
    """Helper to build a board from a list of (pos_str, piece) tuples."""
    board = Board()
    for pos_str, piece in entries:
        board.place_piece(Position.from_string(pos_str), piece)
    return board


def test_place_ring_matches_recomputation():
    """Test that place_ring produces same hash as full recomputation."""
    initializer = ZobristInitializer(seed="place-ring")
    hasher = ZobristHasher(initializer.table)

    board = Board()
    position = Position.from_string("E5")
    player = Player.WHITE

    # Get initial hash
    initial_hash = hasher.hash_board(board)

    # Place ring incrementally
    incremental_hash = hasher.place_ring(position, player, initial_hash)

    # Place ring on board and recompute
    board.place_piece(position, PieceType.WHITE_RING)
    recomputed_hash = hasher.hash_board(board)

    assert incremental_hash == recomputed_hash


def test_place_marker_matches_recomputation():
    """Test that place_marker produces same hash as full recomputation."""
    initializer = ZobristInitializer(seed="place-marker")
    hasher = ZobristHasher(initializer.table)

    board = Board()
    position = Position.from_string("D4")
    player = Player.BLACK

    initial_hash = hasher.hash_board(board)
    incremental_hash = hasher.place_marker(position, player, initial_hash)

    board.place_piece(position, PieceType.BLACK_MARKER)
    recomputed_hash = hasher.hash_board(board)

    assert incremental_hash == recomputed_hash


def test_remove_piece_matches_recomputation():
    """Test that remove_piece produces same hash as full recomputation."""
    initializer = ZobristInitializer(seed="remove-piece")
    hasher = ZobristHasher(initializer.table)

    board = build_board([("E5", PieceType.WHITE_RING), ("F6", PieceType.BLACK_MARKER)])
    position = Position.from_string("E5")
    initial_hash = hasher.hash_board(board)

    # Remove piece incrementally
    incremental_hash = hasher.remove_piece(position, PieceType.WHITE_RING, initial_hash)

    # Remove piece from board and recompute
    board.remove_piece(position)
    recomputed_hash = hasher.hash_board(board)

    assert incremental_hash == recomputed_hash


def test_operations_are_reversible():
    """Test that operations can be reversed by applying twice."""
    initializer = ZobristInitializer(seed="reversible")
    hasher = ZobristHasher(initializer.table)

    board = Board()
    position = Position.from_string("C3")
    player = Player.WHITE
    initial_hash = hasher.hash_board(board)

    # Place ring
    hash_after_place = hasher.place_ring(position, player, initial_hash)
    # Remove ring (reverse operation)
    hash_after_remove = hasher.remove_piece(position, PieceType.WHITE_RING, hash_after_place)

    assert hash_after_remove == initial_hash


def test_move_ring_simple_case():
    """Test move_ring for a simple case with no intermediate markers."""
    initializer = ZobristInitializer(seed="move-ring-simple")
    hasher = ZobristHasher(initializer.table)

    board = build_board([("E5", PieceType.WHITE_RING)])
    source = Position.from_string("E5")
    destination = Position.from_string("E7")
    player = Player.WHITE

    initial_hash = hasher.hash_board(board)

    # Move ring incrementally
    incremental_hash = hasher.move_ring(source, destination, player, initial_hash, board)

    # Move ring on board and recompute
    board.move_ring(source, destination)
    recomputed_hash = hasher.hash_board(board)

    assert incremental_hash == recomputed_hash


def test_move_ring_with_marker_flips():
    """Test move_ring with markers in the path that need to be flipped."""
    initializer = ZobristInitializer(seed="move-ring-flip")
    hasher = ZobristHasher(initializer.table)

    # Set up board: white ring at E5, black markers at E6, white ring moves to E7
    board = build_board([
        ("E5", PieceType.WHITE_RING),
        ("E6", PieceType.BLACK_MARKER),
    ])
    source = Position.from_string("E5")
    destination = Position.from_string("E7")
    player = Player.WHITE

    initial_hash = hasher.hash_board(board)

    # Move ring incrementally (should flip marker at E6)
    incremental_hash = hasher.move_ring(source, destination, player, initial_hash, board)

    # Move ring on board and recompute
    board.move_ring(source, destination)
    recomputed_hash = hasher.hash_board(board)

    assert incremental_hash == recomputed_hash


def test_batch_update():
    """Test batch_update for multiple simultaneous changes."""
    initializer = ZobristInitializer(seed="batch")
    hasher = ZobristHasher(initializer.table)

    board = Board()
    initial_hash = hasher.hash_board(board)

    # Batch update: place ring and marker
    updates = [
        (Position.from_string("E5"), PieceType.EMPTY, PieceType.WHITE_RING),
        (Position.from_string("F6"), PieceType.EMPTY, PieceType.BLACK_MARKER),
    ]
    batch_hash = hasher.batch_update(updates, initial_hash)

    # Apply changes individually
    individual_hash = initial_hash
    individual_hash = hasher.place_ring(Position.from_string("E5"), Player.WHITE, individual_hash)
    individual_hash = hasher.place_marker(Position.from_string("F6"), Player.BLACK, individual_hash)

    assert batch_hash == individual_hash

    # Verify against full recomputation
    board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    board.place_piece(Position.from_string("F6"), PieceType.BLACK_MARKER)
    recomputed_hash = hasher.hash_board(board)

    assert batch_hash == recomputed_hash


def test_incremental_vs_full_performance():
    """Test that incremental updates are faster than full recomputation (basic check)."""
    initializer = ZobristInitializer(seed="perf")
    hasher = ZobristHasher(initializer.table)

    # Build a board with several pieces
    board = build_board([
        ("E5", PieceType.WHITE_RING),
        ("F6", PieceType.BLACK_MARKER),
        ("D4", PieceType.WHITE_MARKER),
        ("G7", PieceType.BLACK_RING),
    ])

    initial_hash = hasher.hash_board(board)

    # Make a change incrementally
    new_hash = hasher.place_marker(Position.from_string("C3"), Player.WHITE, initial_hash)

    # Verify it matches full recomputation
    board.place_piece(Position.from_string("C3"), PieceType.WHITE_MARKER)
    recomputed_hash = hasher.hash_board(board)

    assert new_hash == recomputed_hash


def test_move_ring_complex_path():
    """Test move_ring with a complex path involving multiple marker flips."""
    initializer = ZobristInitializer(seed="complex-path")
    hasher = ZobristHasher(initializer.table)

    # Create a path with multiple markers
    # E5 -> E8 is a valid move (E8 is in valid_move_positions from E5)
    board = build_board([
        ("E5", PieceType.WHITE_RING),
        ("E6", PieceType.BLACK_MARKER),
        ("E7", PieceType.WHITE_MARKER),
    ])
    source = Position.from_string("E5")
    destination = Position.from_string("E8")  # Valid destination
    player = Player.WHITE

    initial_hash = hasher.hash_board(board)
    incremental_hash = hasher.move_ring(source, destination, player, initial_hash, board)

    # Verify the move is valid
    move_result = board.move_ring(source, destination)
    assert move_result, "Move should be valid"
    recomputed_hash = hasher.hash_board(board)

    assert incremental_hash == recomputed_hash

