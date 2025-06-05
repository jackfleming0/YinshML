"""Memory management module for YINSH ML.

This module provides memory pooling and management utilities to optimize
memory usage and reduce garbage collection pressure during training.
"""

from .config import PoolConfig, GrowthPolicy
from .pool import MemoryPool, PoolStatistics, PooledObject, TrackedObject
from .game_state_pool import GameStatePool, GameStatePoolConfig, GameStatePoolStatistics
from .tensor_pool import TensorPool, TensorPoolConfig, TensorPoolStatistics, TensorKey
from .adaptive import (
    AdaptiveMetrics,
    TensorCompatibilityChecker,
    AdaptivePoolSizer,
    create_adaptive_metrics,
    create_adaptive_pool_sizer
)

def reset_game_state(state):
    """Reset a GameState instance to initial values for pool reuse."""
    from ..game import GameState, Board
    from ..game.constants import Player
    from ..game.types import GamePhase
    
    # Reset board
    state.board.pieces.clear()
    
    # Reset game state fields
    state.current_player = Player.WHITE
    state.phase = GamePhase.RING_PLACEMENT
    state.white_score = 0
    state.black_score = 0
    
    # Reset collections
    state.rings_placed.clear()
    state.rings_placed[Player.WHITE] = 0
    state.rings_placed[Player.BLACK] = 0
    
    state.move_history.clear()
    
    # Clean up any temporary attributes
    for attr in ['_move_maker', '_prev_player', '_last_regular_player']:
        if hasattr(state, attr):
            delattr(state, attr)

__all__ = [
    'MemoryPool',
    'PooledObject', 
    'PoolStatistics',
    'TrackedObject',
    'PoolConfig',
    'GrowthPolicy',
    'GameStatePool',
    'GameStatePoolConfig',
    'GameStatePoolStatistics',
    'create_game_state_pool',
    'create_game_state',
    'reset_game_state',
    'validate_reset_game_state',
    'TensorPool',
    'TensorPoolConfig',
    'TensorPoolStatistics',
    'TensorKey',
    'create_tensor',
    'reset_tensor',
    'validate_tensor_reset',
    'create_tensor_pool',
    'AdaptiveMetrics',
    'TensorCompatibilityChecker',
    'AdaptivePoolSizer',
    'create_adaptive_metrics',
    'create_adaptive_pool_sizer',
]

__version__ = '0.1.0' 