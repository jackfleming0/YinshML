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


def test_hash_state_includes_side_to_move_and_phase():
    """hash_state must differ from bare hash_board because it folds in
    current_player and game phase."""
    initializer = ZobristInitializer(seed="state")
    hasher = ZobristHasher(initializer.table)

    from yinsh_ml.game.game_state import GameState  # Local import to avoid circularity
    from yinsh_ml.game.constants import Player

    state = GameState()
    state.board.place_piece(Position.from_string("D4"), PieceType.BLACK_RING)

    # Default state: WHITE to move, RING_PLACEMENT phase. Those still contribute
    # the phase key (side-to-move key is XORed only for BLACK), so hash_state
    # should NOT equal hash_board alone.
    assert state.current_player == Player.WHITE
    assert hasher.hash_state(state) != hasher.hash_board(state.board)


def test_same_board_different_player_different_hash():
    """Regression test: identical piece placement with a different player to
    move must not collide in the transposition table."""
    from yinsh_ml.game.game_state import GameState
    from yinsh_ml.game.constants import Player

    initializer = ZobristInitializer(seed="side-to-move")
    hasher = ZobristHasher(initializer.table)

    state_white = GameState()
    state_black = GameState()
    state_black.current_player = Player.BLACK

    # Same empty board, same phase, only side-to-move differs.
    assert hasher.hash_state(state_white) != hasher.hash_state(state_black)


def test_same_board_different_phase_different_hash():
    """A position's legal moves depend on phase, so hashes must differ."""
    from yinsh_ml.game.game_state import GameState
    from yinsh_ml.game.types import GamePhase

    initializer = ZobristInitializer(seed="phase")
    hasher = ZobristHasher(initializer.table)

    state_a = GameState()
    state_b = GameState()
    state_b.phase = GamePhase.RING_REMOVAL

    assert hasher.hash_state(state_a) != hasher.hash_state(state_b)


def test_hash_deterministic_across_fresh_hashers():
    """Same seed must produce identical hashes across independent hashers,
    including side-to-move and phase keys."""
    from yinsh_ml.game.game_state import GameState
    from yinsh_ml.game.constants import Player
    from yinsh_ml.game.types import GamePhase

    hasher_a = ZobristHasher(seed="determinism")
    hasher_b = ZobristHasher(seed="determinism")

    state = GameState()
    state.board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    state.current_player = Player.BLACK
    state.phase = GamePhase.MAIN_GAME

    assert hasher_a.hash_state(state) == hasher_b.hash_state(state)


def test_hash_state_roundtrip_with_copy():
    """A GameState copy must hash identically to its source."""
    from yinsh_ml.game.game_state import GameState
    from yinsh_ml.game.constants import Player

    initializer = ZobristInitializer(seed="roundtrip")
    hasher = ZobristHasher(initializer.table)

    state = GameState()
    state.board.place_piece(Position.from_string("E5"), PieceType.WHITE_RING)
    state.board.place_piece(Position.from_string("F6"), PieceType.BLACK_MARKER)
    state.current_player = Player.BLACK

    copy = state.copy()
    assert hasher.hash_state(state) == hasher.hash_state(copy)


def test_flip_side_to_move_is_involutive():
    hasher = ZobristHasher(seed="flip")
    h = 0xDEADBEEFCAFEBABE
    assert hasher.flip_side_to_move(hasher.flip_side_to_move(h)) == h


def test_toggle_side_to_move_white_is_noop():
    from yinsh_ml.game.constants import Player

    hasher = ZobristHasher(seed="toggle-white")
    h = 0x0123456789ABCDEF
    assert hasher.toggle_side_to_move(Player.WHITE, h) == h
    assert hasher.toggle_side_to_move(Player.BLACK, h) != h

