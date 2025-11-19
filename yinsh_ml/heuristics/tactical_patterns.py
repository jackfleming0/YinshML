"""Immediate tactical pattern recognition for YINSH game states.

This module detects immediate tactical opportunities like:
- Immediate ring removal opportunities (rows one move away from completion)
- Forced ring captures (opponent must complete a row)
- One-move tactical wins (completing a row leads to winning)
"""

from typing import Optional, List, Tuple
from ..game.game_state import GameState
from ..game.constants import (
    Player, PieceType, MARKERS_FOR_ROW, POINTS_TO_WIN,
    Position, is_valid_position
)
from ..game.types import GamePhase, MoveType
from ..game.board import Row


def detect_immediate_tactical_patterns(
    game_state: GameState,
    player: Player
) -> Optional[float]:
    """Detect immediate tactical patterns that should override heuristic evaluation.
    
    This function checks for:
    - Immediate ring removal opportunities (can complete a row in one move)
    - Forced ring captures (opponent must complete a row)
    - One-move tactical wins (completing a row leads to winning the game)
    
    Args:
        game_state: The current game state to check
        player: The player whose perspective to evaluate from
        
    Returns:
        A high-confidence tactical score if pattern detected:
        - Large positive value (e.g., 5000.0) if player has immediate win opportunity
        - Large negative value (e.g., -5000.0) if opponent has immediate win opportunity
        - Moderate positive value (e.g., 2000.0) if player can force ring removal
        - Moderate negative value (e.g., -2000.0) if opponent can force ring removal
        - None if no immediate tactical patterns detected
        
    Note:
        Scores are chosen to be larger than heuristic evaluations but smaller
        than terminal scores (10000.0) to maintain proper priority ordering.
    """
    # Only check during MAIN_GAME phase
    if game_state.phase != GamePhase.MAIN_GAME:
        return None
    
    # Check if rings are placed (game has started)
    from ..game.constants import RINGS_PER_PLAYER
    rings_placed = sum(game_state.rings_placed.values())
    if rings_placed < RINGS_PER_PLAYER * 2:
        return None
    
    # Check for immediate win opportunities (completing a row leads to score = 3)
    immediate_win_score = _check_immediate_win_opportunity(game_state, player)
    if immediate_win_score is not None:
        return immediate_win_score
    
    # Check for immediate ring removal opportunities
    ring_removal_score = _check_immediate_ring_removal(game_state, player)
    if ring_removal_score is not None:
        return ring_removal_score
    
    # No immediate tactical patterns detected
    return None


def _check_immediate_win_opportunity(
    game_state: GameState,
    player: Player
) -> Optional[float]:
    """Check if player can win immediately by completing a row.
    
    Returns:
        5000.0 if player can win immediately
        -5000.0 if opponent can win immediately
        None otherwise
    """
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Check if player can win immediately
    my_score = game_state.white_score if player == Player.WHITE else game_state.black_score
    if my_score == POINTS_TO_WIN - 1:  # One point away from winning
        if _can_complete_row_in_one_move(game_state, player, my_marker):
            return 5000.0
    
    # Check if opponent can win immediately
    opponent_score = game_state.black_score if player == Player.WHITE else game_state.white_score
    if opponent_score == POINTS_TO_WIN - 1:  # Opponent one point away from winning
        if _can_complete_row_in_one_move(game_state, player.opponent, opponent_marker):
            return -5000.0
    
    return None


def _check_immediate_ring_removal(
    game_state: GameState,
    player: Player
) -> Optional[float]:
    """Check if player or opponent can force ring removal immediately.
    
    Returns:
        2000.0 if player can force ring removal
        -2000.0 if opponent can force ring removal
        None otherwise
    """
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Check if player can force ring removal
    if _can_complete_row_in_one_move(game_state, player, my_marker):
        return 2000.0
    
    # Check if opponent can force ring removal
    if _can_complete_row_in_one_move(game_state, player.opponent, opponent_marker):
        return -2000.0
    
    return None


def _can_complete_row_in_one_move(
    game_state: GameState,
    player: Player,
    marker_type: PieceType
) -> bool:
    """Check if a player can complete a row of 5 markers in one move.
    
    This checks for rows that have exactly 4 markers and can be completed
    by placing a marker (via ring movement) in the missing position.
    
    Args:
        game_state: The current game state
        player: The player to check for
        marker_type: The marker type for the player
        
    Returns:
        True if player can complete a row in one move, False otherwise
    """
    # Find all rows of 4 markers (one away from completion)
    potential_rows = _find_near_complete_rows(game_state.board, marker_type, length=4)
    
    if not potential_rows:
        return False
    
    # Check if any of these rows can be completed with a valid move
    valid_moves = game_state.get_valid_moves()
    
    for row in potential_rows:
        # Find the missing position in the row
        missing_pos = _find_missing_position_in_row(game_state.board, row)
        if missing_pos is None:
            continue
        
        # Check if there's a valid move that would place a marker at missing_pos
        # A ring movement that ends at missing_pos would place a marker there
        for move in valid_moves:
            if move.type == MoveType.MOVE_RING and move.destination == missing_pos:
                # This move would complete the row
                return True
    
    return False


def _find_near_complete_rows(
    board: 'Board',
    marker_type: PieceType,
    length: int = 4
) -> List[Row]:
    """Find rows that are one marker away from completion.
    
    Args:
        board: The game board
        marker_type: The marker type to search for
        length: The length of near-complete rows to find (default: 4)
        
    Returns:
        List of Row objects that have exactly 'length' markers
    """
    # find_marker_rows only returns rows with 5+ markers
    # We need to manually find rows with exactly 4 markers
    from ..game.constants import DIRECTIONS
    
    near_complete_rows = []
    visited_sequences = set()
    
    # Get all positions with the marker type
    positions = [
        pos for pos, piece in board.pieces.items()
        if piece == marker_type
    ]
    
    # Check each position for potential rows
    for start_pos in positions:
        # Check in each direction
        for dx, dy in DIRECTIONS:
            # Try to build row in this direction
            row_positions = [start_pos]
            current = start_pos
            
            # Check up to 3 more positions in this direction (total 4)
            for i in range(length - 1):
                col_idx = ord(current.column) - ord('A')
                new_col = chr(ord('A') + col_idx + dx)
                new_row = current.row + dy
                next_pos = Position(new_col, new_row)
                
                if not is_valid_position(next_pos):
                    break
                
                piece = board.get_piece(next_pos)
                if piece != marker_type:
                    break
                
                row_positions.append(next_pos)
                current = next_pos
            
            # If we found exactly 'length' consecutive markers, we have a near-complete row
            # But we need to check if this is part of a longer row (already completed)
            if len(row_positions) == length:
                # Check if there's another marker in either direction (making it 5+)
                # Check forward (after last marker)
                last_pos = row_positions[-1]
                col_idx = ord(last_pos.column) - ord('A')
                next_col = chr(ord('A') + col_idx + dx)
                next_row = last_pos.row + dy
                next_pos = Position(next_col, next_row)
                
                if is_valid_position(next_pos):
                    piece = board.get_piece(next_pos)
                    if piece == marker_type:
                        continue  # This is part of a longer row, skip it
                
                # Check backward (before first marker)
                first_pos = row_positions[0]
                col_idx = ord(first_pos.column) - ord('A')
                prev_col = chr(ord('A') + col_idx - dx)
                prev_row = first_pos.row - dy
                prev_pos = Position(prev_col, prev_row)
                
                if is_valid_position(prev_pos):
                    piece = board.get_piece(prev_pos)
                    if piece == marker_type:
                        continue  # This is part of a longer row, skip it
                
                # Sort positions for consistent comparison
                sorted_positions = sorted(row_positions, key=lambda p: (p.column, p.row))
                pos_tuple = tuple(sorted_positions)
                
                if pos_tuple not in visited_sequences:
                    visited_sequences.add(pos_tuple)
                    # Create a Row object
                    row = Row(color=marker_type, positions=pos_tuple)
                    near_complete_rows.append(row)
    
    return near_complete_rows


def _find_missing_position_in_row(
    board: 'Board',
    row: Row
) -> Optional['Position']:
    """Find the position that would complete a near-complete row.
    
    Args:
        board: The game board
        row: A row with 4 markers that needs one more to complete
        
    Returns:
        The Position that would complete the row, or None if not found
    """
    if row.length != 4:
        return None
    
    positions = list(row.positions)
    if len(positions) < 4:
        return None
    
    # Determine the direction of the row
    # Check directions: (0, 1), (1, 0), (1, 1), (-1, 1)
    directions = [
        (0, 1),   # Vertical
        (1, 0),   # Horizontal
        (1, 1),   # Diagonal up-right
        (-1, 1),  # Diagonal up-left
    ]
    
    for direction in directions:
        # Try to find the missing position by checking both ends
        missing_pos = _check_direction_for_missing(board, positions, direction)
        if missing_pos is not None:
            return missing_pos
    
    return None


def _check_direction_for_missing(
    board: 'Board',
    positions: List['Position'],
    direction: Tuple[int, int]
) -> Optional['Position']:
    """Check if positions form a line in given direction and find missing position.
    
    Args:
        board: The game board
        positions: List of positions in the row
        direction: Direction tuple (dx, dy)
        
    Returns:
        Missing position if found, None otherwise
    """
    if len(positions) < 4:
        return None
    
    # Sort positions to find the line
    sorted_positions = sorted(positions, key=lambda p: (p.column, p.row))
    
    # Check if positions form a line in the given direction
    dx, dy = direction
    
    # Try both ends of the line
    # Check position before first marker
    first_pos = sorted_positions[0]
    col_idx = ord(first_pos.column) - ord('A')
    prev_col = chr(ord('A') + col_idx - dx)
    prev_row = first_pos.row - dy
    prev_pos = Position(prev_col, prev_row)
    
    if is_valid_position(prev_pos) and board.get_piece(prev_pos) is None:
        # Check if this position would complete the line
        if _positions_form_line(sorted_positions + [prev_pos], direction):
            return prev_pos
    
    # Check position after last marker
    last_pos = sorted_positions[-1]
    col_idx = ord(last_pos.column) - ord('A')
    next_col = chr(ord('A') + col_idx + dx)
    next_row = last_pos.row + dy
    next_pos = Position(next_col, next_row)
    
    if is_valid_position(next_pos) and board.get_piece(next_pos) is None:
        # Check if this position would complete the line
        if _positions_form_line(sorted_positions + [next_pos], direction):
            return next_pos
    
    return None


def _positions_form_line(
    positions: List['Position'],
    direction: Tuple[int, int]
) -> bool:
    """Check if positions form a consecutive line in the given direction.
    
    Args:
        positions: List of positions
        direction: Direction tuple (dx, dy)
        
    Returns:
        True if positions form a consecutive line, False otherwise
    """
    if len(positions) < 2:
        return False
    
    sorted_positions = sorted(positions, key=lambda p: (p.column, p.row))
    dx, dy = direction
    
    for i in range(len(sorted_positions) - 1):
        pos1 = sorted_positions[i]
        pos2 = sorted_positions[i + 1]
        
        col_idx1 = ord(pos1.column) - ord('A')
        col_idx2 = ord(pos2.column) - ord('A')
        
        expected_col_idx = col_idx1 + dx
        expected_row = pos1.row + dy
        
        if col_idx2 != expected_col_idx or pos2.row != expected_row:
            return False
    
    return True

