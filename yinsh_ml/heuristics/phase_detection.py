"""Phase detection for YINSH game states.

This module provides functionality to classify game positions into different
phases (Early, Mid, Late) based on move count and game progression indicators.
Phase detection is used to apply phase-specific heuristic weights during
position evaluation.

The module also provides smooth transition weight calculation to avoid abrupt
strategy changes at phase boundaries.
"""

from enum import Enum
from typing import Tuple, Dict
import math
from ..game.game_state import GameState


class GamePhaseCategory(Enum):
    """Game phase categories for heuristic evaluation.
    
    These phases are based on move count and game progression:
    - EARLY: Opening phase (≤15 moves)
    - MID: Middle game (16-35 moves)
    - LATE: Endgame (36+ moves)
    """
    EARLY = "early"
    MID = "mid"
    LATE = "late"


def detect_phase(
    game_state: GameState,
    early_max: int = 15,
    mid_max: int = 35
) -> GamePhaseCategory:
    """Classify a game state into Early, Mid, or Late phase.
    
    Phase classification is primarily based on move count, with additional
    game progression indicators considered:
    - Move count: Primary indicator
    - Ring placement completion: Secondary indicator
    - Completed runs: Secondary indicator
    
    Args:
        game_state: The game state to classify
        early_max: Maximum move count for Early phase (default: 15)
        mid_max: Maximum move count for Mid phase (default: 35)
        
    Returns:
        GamePhaseCategory enum value indicating the detected phase
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If phase boundaries are invalid
        
    Example:
        >>> game_state = GameState()
        >>> # ... play some moves ...
        >>> phase = detect_phase(game_state)
        >>> print(f"Game phase: {phase}")
        GamePhaseCategory.EARLY
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if early_max <= 0 or mid_max <= 0:
        raise ValueError("Phase boundaries must be positive integers")
    
    if early_max >= mid_max:
        raise ValueError(
            f"'early_max' ({early_max}) must be less than 'mid_max' ({mid_max})"
        )
    
    # Get move count from move history
    move_count = len(game_state.move_history)
    
    # Primary classification based on move count
    if move_count <= early_max:
        return GamePhaseCategory.EARLY
    elif move_count <= mid_max:
        return GamePhaseCategory.MID
    else:
        return GamePhaseCategory.LATE


def get_move_count(game_state: GameState) -> int:
    """Get the total number of moves played in the game.
    
    Args:
        game_state: The game state to query
        
    Returns:
        Integer count of moves in move_history
        
    Raises:
        TypeError: If game_state is not a GameState instance
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    return len(game_state.move_history)


def get_game_progression_indicators(game_state: GameState) -> dict:
    """Get additional game progression indicators beyond move count.
    
    This function extracts secondary indicators that can help refine
    phase detection, such as ring placement completion and tactical
    developments.
    
    Args:
        game_state: The game state to analyze
        
    Returns:
        Dictionary containing progression indicators:
        - move_count: Total moves played
        - rings_placed_total: Total rings placed by both players
        - rings_placed_white: Rings placed by white
        - rings_placed_black: Rings placed by black
        - completed_runs_white: Completed runs by white
        - completed_runs_black: Completed runs by black
        
    Raises:
        TypeError: If game_state is not a GameState instance
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    from ..game.constants import Player
    
    rings_placed_white = game_state.rings_placed.get(Player.WHITE, 0)
    rings_placed_black = game_state.rings_placed.get(Player.BLACK, 0)
    rings_placed_total = rings_placed_white + rings_placed_black
    
    return {
        'move_count': len(game_state.move_history),
        'rings_placed_total': rings_placed_total,
        'rings_placed_white': rings_placed_white,
        'rings_placed_black': rings_placed_black,
        'completed_runs_white': game_state.white_score,
        'completed_runs_black': game_state.black_score,
    }


def calculate_transition_weights(
    move_count: int,
    early_max: int = 15,
    mid_max: int = 35,
    transition_window: int = 2,
    method: str = 'linear'
) -> Dict[str, float]:
    """Calculate phase transition weights for smooth phase boundaries.
    
    This function calculates weights for blending between phases to avoid
    abrupt strategy changes at phase boundaries. It returns weights for
    Early, Mid, and Late phases that sum to 1.0.
    
    Args:
        move_count: Current move count
        early_max: Maximum move count for Early phase (default: 15)
        mid_max: Maximum move count for Mid phase (default: 35)
        transition_window: Number of moves before/after boundary to transition
                     (default: 2)
        method: Interpolation method - 'linear' or 'sigmoid' (default: 'linear')
        
    Returns:
        Dictionary with keys 'early', 'mid', 'late' containing weight values
        that sum to 1.0
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> # At move 15 (Early/Mid boundary)
        >>> weights = calculate_transition_weights(15, early_max=15, mid_max=35)
        >>> print(weights)
        {'early': 0.5, 'mid': 0.5, 'late': 0.0}
    """
    if early_max <= 0 or mid_max <= 0:
        raise ValueError("Phase boundaries must be positive integers")
    
    if early_max >= mid_max:
        raise ValueError(
            f"'early_max' ({early_max}) must be less than 'mid_max' ({mid_max})"
        )
    
    if transition_window < 0:
        raise ValueError("transition_window must be non-negative")
    
    if method not in ('linear', 'sigmoid'):
        raise ValueError(f"method must be 'linear' or 'sigmoid', got '{method}'")
    
    # Initialize weights
    weights = {'early': 0.0, 'mid': 0.0, 'late': 0.0}
    
    # Early phase (before transition window)
    if move_count <= early_max - transition_window:
        weights['early'] = 1.0
        return weights
    
    # Late phase (after transition window)
    if move_count > mid_max + transition_window:
        weights['late'] = 1.0
        return weights
    
    # Transition from Early to Mid (around early_max boundary)
    early_mid_boundary = early_max
    early_mid_start = early_mid_boundary - transition_window
    early_mid_end = early_mid_boundary + transition_window
    
    if early_mid_start <= move_count <= early_mid_end:
        if method == 'linear':
            # Linear interpolation in transition window
            t = (move_count - early_mid_start) / (early_mid_end - early_mid_start)
        else:  # sigmoid
            # Sigmoid interpolation for smoother transition
            center = early_mid_boundary
            scale = transition_window / 2.0
            t = 1.0 / (1.0 + math.exp(-(move_count - center) / scale))
        
        weights['early'] = 1.0 - t
        weights['mid'] = t
        return weights
    
    # Mid phase (between transition windows)
    if move_count <= mid_max - transition_window:
        weights['mid'] = 1.0
        return weights
    
    # Transition from Mid to Late (around mid_max boundary)
    mid_late_boundary = mid_max
    mid_late_start = mid_late_boundary - transition_window
    mid_late_end = mid_late_boundary + transition_window
    
    if mid_late_start <= move_count <= mid_late_end:
        if method == 'linear':
            t = (move_count - mid_late_start) / (mid_late_end - mid_late_start)
        else:  # sigmoid
            center = mid_late_boundary
            scale = transition_window / 2.0
            t = 1.0 / (1.0 + math.exp(-(move_count - center) / scale))
        
        weights['mid'] = 1.0 - t
        weights['late'] = t
        return weights
    
    # Default (should not reach here, but for safety)
    return weights


def get_phase_weights(
    game_state: GameState,
    early_max: int = 15,
    mid_max: int = 35,
    transition_window: int = 2,
    method: str = 'linear'
) -> Dict[str, float]:
    """Get phase transition weights for a game state.
    
    Convenience function that combines move count extraction with
    transition weight calculation.
    
    Args:
        game_state: The game state to analyze
        early_max: Maximum move count for Early phase (default: 15)
        mid_max: Maximum move count for Mid phase (default: 35)
        transition_window: Transition window size (default: 2)
        method: Interpolation method - 'linear' or 'sigmoid' (default: 'linear')
        
    Returns:
        Dictionary with phase weights ('early', 'mid', 'late')
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If parameters are invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    move_count = len(game_state.move_history)
    return calculate_transition_weights(
        move_count, early_max, mid_max, transition_window, method
    )

