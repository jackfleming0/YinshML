"""Forced sequence analysis for YINSH game states.

This module analyzes forced move sequences that lead to definitive outcomes:
- Forced sequences leading to ring removal
- Unavoidable tactical combinations
- Sequences with only one legal response
- Multi-move forced wins
"""

from typing import Optional, List, Tuple
from ..game.game_state import GameState
from ..game.constants import Player, POINTS_TO_WIN, RINGS_PER_PLAYER
from ..game.types import GamePhase
from .terminal_detection import detect_terminal_position


# Maximum depth for forced sequence analysis (keep shallow for performance)
MAX_FORCED_SEQUENCE_DEPTH = 3


def detect_forced_sequences(
    game_state: GameState,
    player: Player
) -> Optional[float]:
    """Detect forced move sequences that lead to definitive outcomes.
    
    This function performs shallow lookahead to identify:
    - Forced sequences leading to ring removal
    - Multi-move forced wins
    - Sequences where opponent has only one legal response
    
    Args:
        game_state: The current game state to analyze
        player: The player whose perspective to evaluate from
        
    Returns:
        A high-confidence score if forced sequence detected:
        - Large positive value (e.g., 3000.0) if player has forced win sequence
        - Large negative value (e.g., -3000.0) if opponent has forced win sequence
        - Moderate positive value (e.g., 1500.0) if player has forced ring removal
        - Moderate negative value (e.g., -1500.0) if opponent has forced ring removal
        - None if no forced sequences detected
        
    Note:
        Uses shallow search (max depth 3) to maintain performance.
        Scores are between tactical patterns (2000-5000) and heuristic evaluations.
    """
    # Only check during MAIN_GAME phase
    if game_state.phase != GamePhase.MAIN_GAME:
        return None
    
    # Check if rings are placed (game has started)
    rings_placed = sum(game_state.rings_placed.values())
    if rings_placed < RINGS_PER_PLAYER * 2:
        return None
    
    # Perform shallow lookahead to detect forced sequences
    forced_score = _analyze_forced_sequences(game_state, player, depth=0)
    
    return forced_score


def _analyze_forced_sequences(
    game_state: GameState,
    player: Player,
    depth: int
) -> Optional[float]:
    """Recursively analyze forced sequences with depth limit.
    
    Args:
        game_state: Current game state
        player: Original player perspective
        depth: Current search depth
        
    Returns:
        Score if forced sequence found, None otherwise
    """
    # Limit depth for performance
    if depth >= MAX_FORCED_SEQUENCE_DEPTH:
        return None
    
    # Early exit for RING_PLACEMENT phase (forced sequences only relevant in MAIN_GAME)
    if game_state.phase != GamePhase.MAIN_GAME:
        return None
    
    # Check for terminal positions first
    terminal_score = detect_terminal_position(game_state, player)
    if terminal_score is not None:
        # Found a terminal position - this is a forced outcome
        # Scale score based on depth (closer = more valuable)
        depth_factor = 1.0 - (depth * 0.2)  # Reduce value by 20% per depth
        return terminal_score * depth_factor
    
    # Get valid moves for current player
    valid_moves = game_state.get_valid_moves()
    if not valid_moves:
        return None
    
    # Limit branching factor for performance (check only first N moves)
    MAX_MOVES_TO_CHECK = 10
    if len(valid_moves) > MAX_MOVES_TO_CHECK:
        valid_moves = valid_moves[:MAX_MOVES_TO_CHECK]
    
    current_player = game_state.current_player
    
    # Check if current player can force a win
    if current_player == player:
        # Check if any move leads to a forced win
        for move in valid_moves:
            test_state = game_state.copy()
            if test_state.make_move(move):
                # Check if this leads to a forced sequence
                result = _analyze_forced_sequences(test_state, player, depth + 1)
                if result is not None and result > 0:
                    # Found a forced sequence - check if it's forced (opponent has no good response)
                    if _is_sequence_forced(test_state, player, depth + 1):
                        # Scale based on depth
                        depth_factor = 1.0 - (depth * 0.15)
                        return result * depth_factor
    else:
        # Check if opponent can force a win (bad for us)
        for move in valid_moves:
            test_state = game_state.copy()
            if test_state.make_move(move):
                result = _analyze_forced_sequences(test_state, player, depth + 1)
                if result is not None and result < 0:
                    # Opponent has forced sequence
                    if _is_sequence_forced(test_state, player, depth + 1):
                        depth_factor = 1.0 - (depth * 0.15)
                        return result * depth_factor
    
    # Check for forced ring removal sequences (less critical than wins)
    ring_removal_score = _check_forced_ring_removal(game_state, player, depth)
    if ring_removal_score is not None:
        return ring_removal_score
    
    return None


def _is_sequence_forced(
    game_state: GameState,
    player: Player,
    depth: int
) -> bool:
    """Check if a sequence is truly forced (opponent has only bad options).
    
    Args:
        game_state: Current game state after our move
        player: Original player perspective
        depth: Current depth
        
    Returns:
        True if sequence is forced, False otherwise
    """
    if depth >= MAX_FORCED_SEQUENCE_DEPTH:
        return False
    
    # Get opponent's moves
    valid_moves = game_state.get_valid_moves()
    if not valid_moves:
        return True  # No moves = forced
    
    # If opponent has only one move, it's forced
    if len(valid_moves) == 1:
        return True
    
    # Check if all opponent moves lead to the same outcome
    # (simplified: if most moves lead to bad outcomes, consider it forced)
    opponent = player.opponent
    bad_outcomes = 0
    
    for move in valid_moves[:5]:  # Limit to first 5 moves for performance
        test_state = game_state.copy()
        if test_state.make_move(move):
            # Check if this leads to a terminal or bad position
            terminal_score = detect_terminal_position(test_state, player)
            if terminal_score is not None and terminal_score < 0:
                bad_outcomes += 1
    
    # If most moves lead to bad outcomes, consider it forced
    return bad_outcomes >= len(valid_moves) * 0.7


def _check_forced_ring_removal(
    game_state: GameState,
    player: Player,
    depth: int
) -> Optional[float]:
    """Check for forced sequences leading to ring removal.
    
    Args:
        game_state: Current game state
        player: Original player perspective
        depth: Current depth
        
    Returns:
        Score if forced ring removal found, None otherwise
    """
    if depth >= MAX_FORCED_SEQUENCE_DEPTH - 1:  # One less depth for ring removal
        return None
    
    current_player = game_state.current_player
    
    # Check if current player can force ring removal
    if current_player == player:
        # Check if any move leads to completing a row
        for move in game_state.get_valid_moves()[:3]:  # Limit moves for performance
            test_state = game_state.copy()
            if test_state.make_move(move):
                # Check if this move completes a row (leads to ring removal phase)
                if test_state.phase == GamePhase.ROW_COMPLETION:
                    # Check if opponent can prevent this
                    if _is_ring_removal_forced(test_state, player, depth + 1):
                        depth_factor = 1.0 - (depth * 0.2)
                        return 1500.0 * depth_factor
    else:
        # Check if opponent can force ring removal
        for move in game_state.get_valid_moves()[:3]:
            test_state = game_state.copy()
            if test_state.make_move(move):
                if test_state.phase == GamePhase.ROW_COMPLETION:
                    if _is_ring_removal_forced(test_state, player, depth + 1):
                        depth_factor = 1.0 - (depth * 0.2)
                        return -1500.0 * depth_factor
    
    return None


def _is_ring_removal_forced(
    game_state: GameState,
    player: Player,
    depth: int
) -> bool:
    """Check if ring removal is forced (cannot be prevented).
    
    Args:
        game_state: Game state in ROW_COMPLETION phase
        player: Original player perspective
        depth: Current depth
        
    Returns:
        True if ring removal is forced, False otherwise
    """
    # In ROW_COMPLETION phase, player must remove markers and then a ring
    # This is essentially forced - they have no choice but to complete the sequence
    # The only question is which markers/ring to remove, but the removal itself is forced
    
    # Check if we're in ring removal phase (markers already removed)
    if game_state.phase == GamePhase.RING_REMOVAL:
        return True
    
    # If in ROW_COMPLETION phase, check if there are multiple rows
    # (if multiple rows, player has choice, so not fully forced)
    if game_state.phase == GamePhase.ROW_COMPLETION:
        # Check if there's only one way to proceed
        valid_moves = game_state.get_valid_moves()
        # If very few options, consider it forced
        return len(valid_moves) <= 2
    
    return False

