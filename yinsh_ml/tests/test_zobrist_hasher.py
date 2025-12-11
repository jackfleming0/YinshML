from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import PieceType, Position
from yinsh_ml.game.zobrist import ZobristHasher, ZobristInitializer


def build_board(entries):
    board = Board()
    for pos_str, piece in entries:
        board.place_piece(Position.from_string(pos_str), piece)
    return board


def test_empty_board_hash_is_stable():
    hasher = ZobristHasher(seed="stable")
    board = Board()
    assert hasher.hash_board(board) == hasher.hash_board(board) == hasher.empty_board_hash


def test_identical_boards_match_different_boards_differ():
    initializer = ZobristInitializer(seed="deterministic")
    hasher = ZobristHasher(initializer.table)

    board_a = build_board([("E5", PieceType.WHITE_RING), ("F6", PieceType.BLACK_MARKER)])
    board_b = build_board([("E5", PieceType.WHITE_RING), ("F6", PieceType.BLACK_MARKER)])
    board_c = build_board([("E5", PieceType.WHITE_RING)])

    assert hasher.hash_board(board_a) == hasher.hash_board(board_b)
    assert hasher.hash_board(board_a) != hasher.hash_board(board_c)


def test_toggle_matches_recomputation():
    initializer = ZobristInitializer(seed="toggle")
    hasher = ZobristHasher(initializer.table)
    board = build_board([("C3", PieceType.WHITE_MARKER)])

    full_hash = hasher.hash_board(board)
    position = Position.from_string("C3")

    manual = hasher.empty_board_hash
    manual = hasher.update_position(position, PieceType.EMPTY, PieceType.WHITE_MARKER, manual)

    assert manual == full_hash


def test_hash_state_wraps_board_hash():
    initializer = ZobristInitializer(seed="state")
    hasher = ZobristHasher(initializer.table)

    from yinsh_ml.game.game_state import GameState  # Local import to avoid circularity

    state = GameState()
    state.board.place_piece(Position.from_string("D4"), PieceType.BLACK_RING)

    assert hasher.hash_board(state.board) == hasher.hash_state(state)

