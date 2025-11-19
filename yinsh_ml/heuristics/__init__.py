"""Heuristics module for YINSH ML.

This module provides heuristic evaluation functions for Yinsh game positions,
including feature extraction and position evaluation based on statistical analysis
of game patterns.
"""

from .evaluator import YinshHeuristics
from .features import (
    extract_all_features,
    completed_runs_differential,
    potential_runs_count,
    connected_marker_chains,
    ring_positioning,
    ring_spread,
    board_control,
)
from .features_utils import calculate_differential
from .phase_detection import (
    detect_phase,
    GamePhaseCategory,
    get_move_count,
    get_game_progression_indicators,
    calculate_transition_weights,
    get_phase_weights,
)
from .phase_config import PhaseConfig, DEFAULT_PHASE_CONFIG
from .weight_manager import WeightManager
from .config_manager import ConfigManager
from .optimizers import (
    GridSearchOptimizer,
    GeneticAlgorithmOptimizer,
    OptimizationResult,
)

__all__ = [
    'YinshHeuristics',
    'extract_all_features',
    'completed_runs_differential',
    'potential_runs_count',
    'connected_marker_chains',
    'ring_positioning',
    'ring_spread',
    'board_control',
    'calculate_differential',
    'detect_phase',
    'GamePhaseCategory',
    'get_move_count',
    'get_game_progression_indicators',
    'calculate_transition_weights',
    'get_phase_weights',
    'PhaseConfig',
    'DEFAULT_PHASE_CONFIG',
    'WeightManager',
    'ConfigManager',
    'GridSearchOptimizer',
    'GeneticAlgorithmOptimizer',
    'OptimizationResult',
]

