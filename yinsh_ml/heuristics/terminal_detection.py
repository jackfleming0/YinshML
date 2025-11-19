"""Terminal position detection for YINSH game states.

This module provides fast detection of terminal positions and game over conditions
that should return definitive scores rather than heuristic estimates.
"""

from typing import Optional
from ..game.game_state import GameState
from ..game.constants import Player, POINTS_TO_WIN, RINGS_PER_PLAYER
from ..game.types import GamePhase


def detect_terminal_position(
    game_state: GameState,
    player: Player
) -> Optional[float]:
    """Detect if a position is terminal and return definitive score.
    
    This function checks for:
    - Game over conditions (winner determined)
    - Terminal board states (no legal moves)
    - Stalemate positions
    
    Args:
        game_state: The current game state to check
        player: The player whose perspective to evaluate from
        
    Returns:
        A definitive score if terminal position detected:
        - Large positive value (e.g., 10000.0) if player wins
        - Large negative value (e.g., -10000.0) if player loses
        - 0.0 if draw/stalemate
        - None if position is not terminal
        
    Note:
        Scores are chosen to be much larger than any heuristic evaluation
        to ensure they override heuristic scores.
    """
    # Check for game over condition (winner determined)
    winner = game_state.get_winner()
    if winner is not None:
        if winner == player:
            return 10000.0  # Player wins
        else:
            return -10000.0  # Player loses
    
    # Check if game state is marked as terminal
    if game_state.is_terminal():
        # If terminal but no winner, it's a draw
        return 0.0
    
    # Check for positions with no legal moves (stalemate)
    # This is a terminal condition even if not explicitly marked as GAME_OVER
    # Only check during MAIN_GAME phase - other phases may have no moves temporarily
    # Also ensure rings are placed, otherwise no moves is expected
    if game_state.phase == GamePhase.MAIN_GAME:
        # Check if rings are actually placed (game has started)
        rings_placed = sum(game_state.rings_placed.values())
        if rings_placed >= RINGS_PER_PLAYER * 2:  # Both players have placed rings
            valid_moves = game_state.get_valid_moves()
            if not valid_moves:
                # No legal moves available - this is a terminal position
                # In Yinsh, if a player has no moves, they lose
                # But we need to check whose turn it is
                if game_state.current_player == player:
                    # Current player has no moves - they lose
                    return -10000.0
                else:
                    # Opponent has no moves - current player wins
                    return 10000.0
    
    # Position is not terminal
    return None


def is_game_over(game_state: GameState) -> bool:
    """Check if the game is over.
    
    Args:
        game_state: The game state to check
        
    Returns:
        True if game is over, False otherwise
    """
    return game_state.is_terminal() or game_state.get_winner() is not None


def has_winner(game_state: GameState) -> bool:
    """Check if there is a winner.
    
    Args:
        game_state: The game state to check
        
    Returns:
        True if there is a winner, False otherwise
    """
    return game_state.get_winner() is not None

