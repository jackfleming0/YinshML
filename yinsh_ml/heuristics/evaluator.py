"""Heuristic evaluator for YINSH game positions.

This module provides the main YinshHeuristics class for evaluating game positions
using statistical analysis of game patterns and feature-based heuristics.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from ..game.game_state import GameState
from ..game.constants import Player
from ..game.types import GamePhase
from .phase_detection import detect_phase, get_phase_weights, GamePhaseCategory
from .phase_config import PhaseConfig, DEFAULT_PHASE_CONFIG
from .features import extract_all_features
from .weight_manager import WeightManager
from .terminal_detection import detect_terminal_position
from .tactical_patterns import detect_immediate_tactical_patterns
from .forced_sequences import detect_forced_sequences


class YinshHeuristics:
    """Heuristic evaluator for Yinsh game positions.
    
    This class evaluates Yinsh game positions using a combination of:
    - Feature extraction from board state
    - Phase-aware weighting based on game progression
    - Statistical patterns from game analysis
    
    The evaluation returns a differential score indicating the advantage
    for the current player (positive = good, negative = bad).
    
    Attributes:
        weights: Dictionary containing feature weights for different game phases
        phase_boundaries: Configuration for game phase detection
        
    Example:
        >>> heuristics = YinshHeuristics()
        >>> game_state = GameState()
        >>> score = heuristics.evaluate_position(game_state, Player.WHITE)
        >>> print(f"Position evaluation: {score}")
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, Any]] = None,
        phase_config: Optional[PhaseConfig] = None,
        weight_config_file: Optional[str] = None
    ):
        """Initialize the YinshHeuristics evaluator.
        
        Args:
            weights: Optional dictionary of feature weights for different phases.
                     If None, default weights will be used based on correlation
                     analysis from game data.
            phase_config: Optional PhaseConfig instance for phase detection settings.
                         If None, DEFAULT_PHASE_CONFIG will be used.
            weight_config_file: Optional path to weight configuration file.
                               If provided, weights will be loaded from this file.
                               If None and weights is None, default weights will be used.
                     
        Raises:
            ValueError: If weights structure is invalid.
            FileNotFoundError: If weight_config_file is specified but doesn't exist.
        """
        # Initialize weight manager
        self.weight_manager = WeightManager()
        
        # Load weights from file if specified
        if weight_config_file:
            self.weights = self.weight_manager.load_from_file(weight_config_file)
        elif weights is not None:
            # Validate weights if provided
            self._validate_weights(weights)
            self.weights = weights
            self.weight_manager.set_default_weights(weights)
        else:
            # Use default weights
            self.weights = self._get_default_weights()
            self.weight_manager.set_default_weights(self.weights)
        
        # Phase configuration
        self.phase_config = phase_config or DEFAULT_PHASE_CONFIG
        
        # Phase boundaries (for backward compatibility)
        self.phase_boundaries = {
            'early_max_moves': self.phase_config.early_max_moves,
            'mid_max_moves': self.phase_config.mid_max_moves,
        }
        
        # Validate phase boundaries
        self._validate_phase_boundaries()
        
        # Pre-compute phase config values for faster access
        self._early_max = self.phase_config.early_max_moves
        self._mid_max = self.phase_config.mid_max_moves
        self._transition_window = self.phase_config.transition_window
        self._interpolation_method = self.phase_config.interpolation_method
        
        # Pre-compute feature name list for efficient iteration
        self._feature_names = [
            'completed_runs_differential',
            'potential_runs_count',
            'connected_marker_chains',
            'ring_positioning',
            'ring_spread',
            'board_control',
        ]
        
        # Pre-extract weight dictionaries for faster access
        self._update_cached_weights()
        
        # Cache for phase transition weights (for batch evaluation optimization)
        self._phase_weights_cache: Dict[int, Dict[str, float]] = {}
        
        # Pre-compute weight matrix for vectorized operations
        self._build_weight_matrix()
    
    def evaluate_position(
        self,
        game_state: GameState,
        player: Player
    ) -> float:
        """Evaluate a game position and return a score.
        
        This method evaluates the current board position from the perspective
        of the specified player. The score is a differential value where:
        - Positive values indicate an advantage for the specified player
        - Negative values indicate a disadvantage
        - Zero indicates a roughly equal position
        
        The evaluation considers:
        1. Game phase (early/mid/late)
        2. Feature extraction (runs, chains, ring positioning, etc.)
        3. Phase-specific weighting of features
        4. Immediate win/loss detection
        
        Args:
            game_state: The current game state to evaluate
            player: The player whose perspective to evaluate from
            
        Returns:
            A float representing the position evaluation score. Higher values
            indicate better positions for the specified player.
            
        Raises:
            TypeError: If game_state is not a GameState instance
            ValueError: If player is invalid
        """
        # Validate inputs (optimized: use isinstance directly for speed)
        if not isinstance(game_state, GameState):
            raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
        if not isinstance(player, Player):
            raise ValueError(f"player must be a Player enum value, got {type(player)}")
        
        # 0. Check for terminal positions first (immediate win/loss detection)
        terminal_score = detect_terminal_position(game_state, player)
        if terminal_score is not None:
            return terminal_score
        
        # 0.5. Check for immediate tactical patterns (ring removal opportunities)
        tactical_score = detect_immediate_tactical_patterns(game_state, player)
        if tactical_score is not None:
            return tactical_score
        
        # 0.75. Check for forced sequences (multi-move forced outcomes)
        # Skip forced sequence detection during RING_PLACEMENT phase for performance
        # (forced sequences are only relevant in MAIN_GAME phase)
        if game_state.phase == GamePhase.MAIN_GAME:
            forced_score = detect_forced_sequences(game_state, player)
            if forced_score is not None:
                return forced_score
        
        # 1. Detect game phase (use cached config values)
        move_count = len(game_state.move_history)
        if move_count <= self._early_max:
            phase = GamePhaseCategory.EARLY
        elif move_count <= self._mid_max:
            phase = GamePhaseCategory.MID
        else:
            phase = GamePhaseCategory.LATE
        
        # Get phase transition weights for smooth blending (use cached config)
        phase_transition_weights = get_phase_weights(
            game_state,
            early_max=self._early_max,
            mid_max=self._mid_max,
            transition_window=self._transition_window,
            method=self._interpolation_method
        )
        
        # 2. Extract all features
        features = extract_all_features(game_state, player)
        
        # 3. Apply phase-specific weights and calculate score (optimized)
        score = self._apply_phase_weights_fast(features, phase_transition_weights)
        
        # 4. Return final evaluation score
        return score
    
    def evaluate_batch(
        self,
        game_states: List[GameState],
        players: List[Player]
    ) -> List[float]:
        """Evaluate multiple game positions in batch.
        
        This method efficiently evaluates multiple positions simultaneously,
        supporting parallel self-play and reducing overhead compared to
        individual position evaluations.
        
        Args:
            game_states: List of game states to evaluate
            players: List of players (one per game state) whose perspective to evaluate from
            
        Returns:
            List of float scores, one per position, in the same order as input.
            Higher values indicate better positions for the specified player.
            
        Raises:
            TypeError: If inputs are not lists or contain invalid types
            ValueError: If lists have mismatched lengths or are empty (empty returns empty list)
        """
        # Validate inputs
        if not isinstance(game_states, list):
            raise TypeError(f"game_states must be a list, got {type(game_states)}")
        if not isinstance(players, list):
            raise TypeError(f"players must be a list, got {type(players)}")
        
        # Handle empty lists
        if len(game_states) == 0:
            return []
        
        # Validate list lengths match
        if len(game_states) != len(players):
            raise ValueError(
                f"game_states and players must have same length, "
                f"got {len(game_states)} and {len(players)}"
            )
        
        # Validate each game state and player
        for i, game_state in enumerate(game_states):
            if not isinstance(game_state, GameState):
                raise TypeError(
                    f"game_states[{i}] must be a GameState instance, got {type(game_state)}"
                )
        
        for i, player in enumerate(players):
            if not isinstance(player, Player):
                raise TypeError(
                    f"players[{i}] must be a Player enum value, got {type(player)}"
                )
        
        # Optimized batch evaluation with vectorization and caching
        batch_size = len(game_states)
        
        # Fast path for single position (avoid batch overhead)
        if batch_size == 1:
            return [self.evaluate_position(game_states[0], players[0])]
        
        # Pre-allocate result array
        results = np.zeros(batch_size, dtype=np.float64)
        
        # Group positions by move count for phase caching
        # Process positions in batches by phase to reuse cached weights
        move_counts = [len(gs.move_history) for gs in game_states]
        
        # Extract features for all positions
        features_list = []
        terminal_mask = np.zeros(batch_size, dtype=bool)
        tactical_mask = np.zeros(batch_size, dtype=bool)
        forced_mask = np.zeros(batch_size, dtype=bool)
        
        for i, (game_state, player) in enumerate(zip(game_states, players)):
            # Check for terminal positions first
            terminal_score = detect_terminal_position(game_state, player)
            if terminal_score is not None:
                results[i] = terminal_score
                terminal_mask[i] = True
                continue
            
            # Check for tactical patterns
            tactical_score = detect_immediate_tactical_patterns(game_state, player)
            if tactical_score is not None:
                results[i] = tactical_score
                tactical_mask[i] = True
                continue
            
            # Check for forced sequences
            forced_score = detect_forced_sequences(game_state, player)
            if forced_score is not None:
                results[i] = forced_score
                forced_mask[i] = True
                continue
            
            # Extract features for non-terminal positions
            features = extract_all_features(game_state, player)
            features_list.append((i, features))
        
        # Process non-terminal positions with vectorized operations
        non_terminal_data = [(idx, feat) for idx, feat in features_list]
        
        if len(non_terminal_data) > 0:
            # Group by move count for phase caching
            move_count_groups: Dict[int, List[int]] = {}
            for idx, _ in non_terminal_data:
                move_count = move_counts[idx]
                if move_count not in move_count_groups:
                    move_count_groups[move_count] = []
                move_count_groups[move_count].append(idx)
            
            # Process each group
            for move_count, indices in move_count_groups.items():
                # Get or compute phase transition weights (cached)
                phase_weights = self._get_cached_phase_weights(move_count, game_states[indices[0]])
                
                # Extract features for this group
                group_features = [feat for idx, feat in non_terminal_data if idx in indices]
                
                # Convert to numpy array for vectorized operations
                feature_array = np.array([
                    [f.get(name, 0.0) for name in self._feature_names]
                    for f in group_features
                ], dtype=np.float64)
                
                # Apply weights vectorized
                # Blend weights based on phase transition
                w_early = phase_weights['early']
                w_mid = phase_weights['mid']
                w_late = phase_weights['late']
                
                # Get weight vectors
                early_weights_vec = np.array([self._early_weights.get(name, 0.0) for name in self._feature_names])
                mid_weights_vec = np.array([self._mid_weights.get(name, 0.0) for name in self._feature_names])
                late_weights_vec = np.array([self._late_weights.get(name, 0.0) for name in self._feature_names])
                
                # Blend weights
                blended_weights = (
                    early_weights_vec * w_early +
                    mid_weights_vec * w_mid +
                    late_weights_vec * w_late
                )
                
                # Vectorized dot product: feature_array @ blended_weights
                group_scores = np.dot(feature_array, blended_weights)
                
                # Store results
                for idx, score in zip(indices, group_scores):
                    results[idx] = score
        
        return results.tolist()
    
    def _validate_inputs(self, game_state: GameState, player: Player) -> None:
        """Validate input parameters for evaluation.
        
        Args:
            game_state: The game state to validate
            player: The player to validate
            
        Raises:
            TypeError: If game_state is not a GameState instance
            ValueError: If player is not a valid Player enum value
        """
        if not isinstance(game_state, GameState):
            raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
        
        if not isinstance(player, Player):
            raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    def _validate_weights(self, weights: Dict[str, Any]) -> None:
        """Validate weights dictionary structure.
        
        Args:
            weights: Dictionary of feature weights to validate
            
        Raises:
            ValueError: If weights structure is invalid
        """
        if not isinstance(weights, dict):
            raise ValueError(f"weights must be a dictionary, got {type(weights)}")
        
        # Check for required phase keys if weights contain phase-specific data
        if weights:
            # Validate that weight values are numeric
            for key, value in weights.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Weight value for '{key}' must be numeric, got {type(value)}"
                    )
    
    def _validate_phase_boundaries(self) -> None:
        """Validate phase boundary configuration.
        
        Raises:
            ValueError: If phase boundaries are invalid
        """
        early_max = self.phase_boundaries.get('early_max_moves')
        mid_max = self.phase_boundaries.get('mid_max_moves')
        
        if early_max is None or mid_max is None:
            raise ValueError("Phase boundaries must include 'early_max_moves' and 'mid_max_moves'")
        
        if not isinstance(early_max, int) or not isinstance(mid_max, int):
            raise ValueError("Phase boundaries must be integers")
        
        if early_max <= 0 or mid_max <= 0:
            raise ValueError("Phase boundaries must be positive integers")
        
        if early_max >= mid_max:
            raise ValueError(
                f"'early_max_moves' ({early_max}) must be less than "
                f"'mid_max_moves' ({mid_max})"
            )
    
    def _get_default_weights(self) -> Dict[str, Dict[str, float]]:
        """Get default feature weights based on correlation analysis.
        
        Weights are derived from statistical analysis of 100K+ games:
        - completed_runs_differential: r=0.116 (strongest correlation)
        - potential_runs_count: r=0.109
        - connected_marker_chains: r=0.071
        - Other features have lower but still meaningful correlations
        
        Phase-specific adjustments:
        - Early: Emphasize ring positioning and board control
        - Mid: Balance all features, emphasize runs
        - Late: Focus heavily on completed runs and tactical patterns
        
        Returns:
            Dictionary with phase keys ('early', 'mid', 'late') mapping to
            feature weight dictionaries
        """
        # Base weights derived from correlation analysis
        # Higher correlation = higher base weight
        base_weights = {
            'completed_runs_differential': 11.6,  # r=0.116 scaled
            'potential_runs_count': 10.9,          # r=0.109 scaled
            'connected_marker_chains': 7.1,        # r=0.071 scaled
            'ring_positioning': 5.0,                # Estimated
            'ring_spread': 4.0,                     # Estimated
            'board_control': 6.0,                   # Estimated
        }
        
        # Phase-specific adjustments
        # Early game: Emphasize ring positioning and board control
        early_weights = {
            'completed_runs_differential': base_weights['completed_runs_differential'] * 0.7,
            'potential_runs_count': base_weights['potential_runs_count'] * 0.8,
            'connected_marker_chains': base_weights['connected_marker_chains'] * 0.9,
            'ring_positioning': base_weights['ring_positioning'] * 1.3,  # More important early
            'ring_spread': base_weights['ring_spread'] * 1.2,            # More important early
            'board_control': base_weights['board_control'] * 1.1,       # More important early
        }
        
        # Mid game: Balance all features, slight emphasis on runs
        mid_weights = {
            'completed_runs_differential': base_weights['completed_runs_differential'] * 1.0,
            'potential_runs_count': base_weights['potential_runs_count'] * 1.1,  # Slight emphasis
            'connected_marker_chains': base_weights['connected_marker_chains'] * 1.0,
            'ring_positioning': base_weights['ring_positioning'] * 0.9,
            'ring_spread': base_weights['ring_spread'] * 0.9,
            'board_control': base_weights['board_control'] * 1.0,
        }
        
        # Late game: Focus heavily on completed runs and tactical patterns
        late_weights = {
            'completed_runs_differential': base_weights['completed_runs_differential'] * 1.5,  # Critical
            'potential_runs_count': base_weights['potential_runs_count'] * 1.3,               # Important
            'connected_marker_chains': base_weights['connected_marker_chains'] * 1.2,          # Tactical
            'ring_positioning': base_weights['ring_positioning'] * 0.6,                       # Less important
            'ring_spread': base_weights['ring_spread'] * 0.7,                                  # Less important
            'board_control': base_weights['board_control'] * 0.9,                              # Less important
        }
        
        return {
            'early': early_weights,
            'mid': mid_weights,
            'late': late_weights,
        }
    
    def _apply_phase_weights(
        self,
        features: Dict[str, float],
        phase: GamePhaseCategory,
        phase_transition_weights: Dict[str, float]
    ) -> float:
        """Apply phase-specific weights to features and calculate final score.
        
        This method applies weights to features using smooth phase transitions.
        When at phase boundaries, it blends weights from adjacent phases to
        avoid abrupt evaluation changes.
        
        Args:
            features: Dictionary of extracted feature values
            phase: Detected game phase category
            phase_transition_weights: Weights for blending between phases
            
        Returns:
            Final evaluation score combining all weighted features
        """
        # Get base weights for each phase
        early_weights = self.weights.get('early', {})
        mid_weights = self.weights.get('mid', {})
        late_weights = self.weights.get('late', {})
        
        # Calculate blended weights based on phase transitions
        blended_weights = {}
        for feature_name in features.keys():
            blended_weights[feature_name] = (
                early_weights.get(feature_name, 0.0) * phase_transition_weights['early'] +
                mid_weights.get(feature_name, 0.0) * phase_transition_weights['mid'] +
                late_weights.get(feature_name, 0.0) * phase_transition_weights['late']
            )
        
        # Apply weights to features and sum
        score = 0.0
        for feature_name, feature_value in features.items():
            weight = blended_weights.get(feature_name, 0.0)
            score += weight * feature_value
        
        return score
    
    def _apply_phase_weights_fast(
        self,
        features: Dict[str, float],
        phase_transition_weights: Dict[str, float]
    ) -> float:
        """Optimized version of phase weight application.
        
        This method minimizes dictionary lookups and uses pre-computed weight
        dictionaries for faster execution. It's designed for hot-path performance.
        
        Args:
            features: Dictionary of extracted feature values
            phase_transition_weights: Weights for blending between phases
            
        Returns:
            Final evaluation score combining all weighted features
        """
        # Extract transition weights once
        w_early = phase_transition_weights['early']
        w_mid = phase_transition_weights['mid']
        w_late = phase_transition_weights['late']
        
        # Pre-extract weight dictionaries (already done in __init__)
        early_weights = self._early_weights
        mid_weights = self._mid_weights
        late_weights = self._late_weights
        
        # Calculate score directly without intermediate dictionary
        # Iterate over known feature names for better cache locality
        score = 0.0
        for feature_name in self._feature_names:
            feature_value = features.get(feature_name, 0.0)
            if feature_value == 0.0:
                continue  # Skip zero features to save computation
            
            # Blend weights inline for better performance
            weight = (
                early_weights.get(feature_name, 0.0) * w_early +
                mid_weights.get(feature_name, 0.0) * w_mid +
                late_weights.get(feature_name, 0.0) * w_late
            )
            score += weight * feature_value
        
        return score
    
    def _get_cached_phase_weights(
        self,
        move_count: int,
        game_state: GameState
    ) -> Dict[str, float]:
        """Get phase transition weights, using cache if available.
        
        Args:
            move_count: Number of moves in the game state
            game_state: Game state (used if cache miss)
            
        Returns:
            Dictionary with 'early', 'mid', 'late' phase weights
        """
        if move_count not in self._phase_weights_cache:
            # Compute and cache phase transition weights
            phase_weights = get_phase_weights(
                game_state,
                early_max=self._early_max,
                mid_max=self._mid_max,
                transition_window=self._transition_window,
                method=self._interpolation_method
            )
            self._phase_weights_cache[move_count] = phase_weights
        
        return self._phase_weights_cache[move_count]
    
    def _build_weight_matrix(self) -> None:
        """Pre-compute weight matrices for vectorized operations.
        
        This method pre-computes numpy arrays for weight vectors to avoid
        repeated dictionary lookups during batch evaluation.
        """
        # Pre-compute weight vectors (done lazily in evaluate_batch for now)
        # This method is a placeholder for future optimizations
        pass
    
    def _update_cached_weights(self) -> None:
        """Update cached weight dictionaries after weight changes.
        
        This method should be called whenever weights are updated to ensure
        the pre-computed weight dictionaries are synchronized.
        """
        self._early_weights = self.weights.get('early', {})
        self._mid_weights = self.weights.get('mid', {})
        self._late_weights = self.weights.get('late', {})
    
    def load_weights_from_file(self, filepath: str) -> None:
        """Load weights from a configuration file and update evaluator.
        
        Args:
            filepath: Path to the weight configuration file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid weights
        """
        self.weights = self.weight_manager.load_from_file(filepath)
        self._update_cached_weights()
    
    def save_weights_to_file(
        self,
        filepath: str,
        create_backup: bool = True
    ) -> None:
        """Save current weights to a configuration file.
        
        Args:
            filepath: Path to save the weight configuration file
            create_backup: If True, creates a backup before saving
            
        Raises:
            ValueError: If weights are invalid
            IOError: If file cannot be written
        """
        self.weight_manager.set_default_weights(self.weights)
        self.weight_manager.save_to_file(filepath, create_backup=create_backup)
    
    def update_weight(
        self,
        phase: str,
        feature: str,
        value: float
    ) -> None:
        """Update a specific weight value at runtime.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            feature: Feature name (e.g., 'completed_runs_differential')
            value: New weight value
            
        Raises:
            ValueError: If phase, feature, or value is invalid
        """
        self.weight_manager.update_weights(phase, feature, value)
        self.weights = self.weight_manager.get_weights()
        self._update_cached_weights()
    
    def update_phase_weights(
        self,
        phase: str,
        weights: Dict[str, float]
    ) -> None:
        """Update all weights for a specific phase at runtime.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            weights: Dictionary mapping feature names to weight values
            
        Raises:
            ValueError: If phase or weights are invalid
        """
        self.weight_manager.update_phase_weights(phase, weights)
        self.weights = self.weight_manager.get_weights()
        self._update_cached_weights()
    
    def get_weight(self, phase: str, feature: str) -> Optional[float]:
        """Get a specific weight value.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            feature: Feature name
            
        Returns:
            Weight value or None if not set
        """
        return self.weight_manager.get_weight(phase, feature)
    
    def list_weight_backups(self) -> list:
        """List all available weight backup files.
        
        Returns:
            List of backup file paths
        """
        return self.weight_manager.list_backups()
    
    def restore_weights_from_backup(self, backup_path: str) -> None:
        """Restore weights from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Raises:
            FileNotFoundError: If backup file doesn't exist
            ValueError: If backup file is invalid
        """
        self.weights = self.weight_manager.restore_from_backup(backup_path)
        self._update_cached_weights()

