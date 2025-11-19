"""Move selection policies for self-play."""

import random
import logging
import math
import time
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..game.types import Move, MoveType, GamePhase
from ..game.constants import Player, Position
from ..game.board import Board
from ..game.game_state import GameState

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for move selection policies."""
    rule_based_probability: float = 0.1  # 10% chance of using rule-based moves
    random_seed: Optional[int] = None


class RandomMovePolicy:
    """Random move selection policy for self-play."""
    
    def __init__(self, config: PolicyConfig = None):
        """Initialize the random move policy.
        
        Args:
            config: Policy configuration
        """
        self.config = config or PolicyConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        logger.info(f"Initialized RandomMovePolicy with rule-based probability: {self.config.rule_based_probability}")
    
    def select_move(self, game_state: GameState) -> Move:
        """Select a move using random policy with optional rule-based moves.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move
            
        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Use rule-based move with configured probability
        if random.random() < self.config.rule_based_probability:
            move = self._select_rule_based_move(valid_moves, game_state)
            if move is not None:
                logger.debug(f"Selected rule-based move: {move}")
                return move
        
        # Fall back to random selection
        move = random.choice(valid_moves)
        logger.debug(f"Selected random move: {move}")
        return move
    
    def _select_rule_based_move(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Select a move using simple rule-based heuristics.
        
        Args:
            valid_moves: List of valid moves
            game_state: Current game state
            
        Returns:
            Rule-based move if found, None otherwise
        """
        if not valid_moves:
            return None
        
        # Rule-based strategies based on game phase
        if game_state.phase == GamePhase.RING_PLACEMENT:
            return self._rule_based_ring_placement(valid_moves, game_state)
        elif game_state.phase == GamePhase.MAIN_GAME:
            return self._rule_based_ring_movement(valid_moves, game_state)
        elif game_state.phase == GamePhase.ROW_COMPLETION:
            return self._rule_based_marker_removal(valid_moves, game_state)
        elif game_state.phase == GamePhase.RING_REMOVAL:
            return self._rule_based_ring_removal(valid_moves, game_state)
        
        return None
    
    def _rule_based_ring_placement(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based ring placement strategy.
        
        Prefer center positions and avoid edges early in the game.
        """
        # Filter moves by type
        ring_placement_moves = [m for m in valid_moves if m.type == MoveType.PLACE_RING]
        if not ring_placement_moves:
            return None
        
        # Prefer center positions (D4-E5-F6 area)
        center_moves = []
        edge_moves = []
        
        for move in ring_placement_moves:
            pos = move.source
            if pos is None:
                continue
                
            # Check if position is in center area
            if self._is_center_position(pos):
                center_moves.append(move)
            elif self._is_edge_position(pos):
                edge_moves.append(move)
        
        # Prefer center moves, fall back to non-edge moves, then any move
        if center_moves:
            return random.choice(center_moves)
        elif len(ring_placement_moves) > len(edge_moves):
            non_edge_moves = [m for m in ring_placement_moves if m not in edge_moves]
            if non_edge_moves:
                return random.choice(non_edge_moves)
        
        return random.choice(ring_placement_moves)
    
    def _rule_based_ring_movement(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based ring movement strategy.
        
        Prefer moves that create or block potential runs.
        """
        # Filter moves by type
        ring_moves = [m for m in valid_moves if m.type == MoveType.MOVE_RING]
        if not ring_moves:
            return None
        
        # Simple heuristic: prefer moves that advance toward opponent's side
        opponent = Player.BLACK if game_state.current_player == Player.WHITE else Player.WHITE
        
        scoring_moves = []
        for move in ring_moves:
            if move.destination is None:
                continue
            
            score = self._score_ring_move(move, game_state, opponent)
            scoring_moves.append((score, move))
        
        if scoring_moves:
            # Sort by score (higher is better) and pick from top moves
            scoring_moves.sort(key=lambda x: x[0], reverse=True)
            top_moves = [m for score, m in scoring_moves if score == scoring_moves[0][0]]
            return random.choice(top_moves)
        
        return random.choice(ring_moves)
    
    def _rule_based_marker_removal(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based marker removal strategy.
        
        Prefer removing opponent's markers when possible.
        """
        # Filter moves by type
        marker_moves = [m for m in valid_moves if m.type == MoveType.REMOVE_MARKERS]
        if not marker_moves:
            return None
        
        # Simple heuristic: prefer moves that remove opponent's markers
        opponent = Player.BLACK if game_state.current_player == Player.WHITE else Player.WHITE
        
        opponent_marker_moves = []
        for move in marker_moves:
            if move.markers is None:
                continue
            
            # Count opponent markers in this move
            opponent_count = sum(1 for pos in move.markers 
                              if game_state.board.get_piece(pos) == 
                              (PieceType.BLACK_MARKER if opponent == Player.BLACK else PieceType.WHITE_MARKER))
            
            if opponent_count > 0:
                opponent_marker_moves.append((opponent_count, move))
        
        if opponent_marker_moves:
            # Sort by opponent marker count and pick from top moves
            opponent_marker_moves.sort(key=lambda x: x[0], reverse=True)
            top_moves = [m for count, m in opponent_marker_moves 
                        if count == opponent_marker_moves[0][0]]
            return random.choice(top_moves)
        
        return random.choice(marker_moves)
    
    def _rule_based_ring_removal(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based ring removal strategy.
        
        Prefer removing rings that are less strategically important.
        """
        # Filter moves by type
        ring_removal_moves = [m for m in valid_moves if m.type == MoveType.REMOVE_RING]
        if not ring_removal_moves:
            return None
        
        # Simple heuristic: prefer removing rings from edge positions
        edge_moves = []
        center_moves = []
        
        for move in ring_removal_moves:
            if move.source is None:
                continue
            
            if self._is_edge_position(move.source):
                edge_moves.append(move)
            elif self._is_center_position(move.source):
                center_moves.append(move)
        
        # Prefer edge moves, fall back to center moves, then any move
        if edge_moves:
            return random.choice(edge_moves)
        elif center_moves:
            return random.choice(center_moves)
        
        return random.choice(ring_removal_moves)
    
    def _is_center_position(self, pos: Position) -> bool:
        """Check if position is in the center area of the board."""
        # Center area: D4-E5-F6 region
        center_positions = {
            'D4', 'D5', 'D6',
            'E4', 'E5', 'E6', 'E7',
            'F4', 'F5', 'F6', 'F7', 'F8'
        }
        return str(pos) in center_positions
    
    def _is_edge_position(self, pos: Position) -> bool:
        """Check if position is on the edge of the board."""
        # Edge positions are those with minimal neighbors
        edge_positions = {
            'A2', 'A3', 'A4', 'A5',
            'B1', 'B7',
            'C1', 'C8',
            'D1', 'D9',
            'E1', 'E10',
            'F2', 'F10',
            'G2', 'G11',
            'H3', 'H11',
            'I4', 'I11',
            'J5', 'J11',
            'K7', 'K8', 'K9', 'K10'
        }
        return str(pos) in edge_positions
    
    def _score_ring_move(self, move: Move, game_state: GameState, opponent: Player) -> int:
        """Score a ring move based on simple heuristics.
        
        Args:
            move: The move to score
            game_state: Current game state
            opponent: Opponent player
            
        Returns:
            Score for the move (higher is better)
        """
        if move.destination is None:
            return 0
        
        score = 0
        
        # Prefer moves toward opponent's side
        if opponent == Player.BLACK:
            # Black starts at bottom, so prefer moves toward higher rows
            if move.destination.row > 6:
                score += 2
        else:
            # White starts at top, so prefer moves toward lower rows
            if move.destination.row < 6:
                score += 2
        
        # Prefer center positions
        if self._is_center_position(move.destination):
            score += 1
        
        # Avoid edge positions
        if self._is_edge_position(move.destination):
            score -= 1
        
        return score


@dataclass
class HeuristicPolicyConfig:
    """Configuration for heuristic-based move selection policy."""
    search_depth: int = 3  # Depth for heuristic search
    randomness: float = 0.1  # Exploration randomness (0-1), epsilon-greedy
    time_limit: float = 1.0  # Per-move time budget in seconds
    temperature: float = 1.0  # Move selection temperature (higher = more exploration)
    random_seed: Optional[int] = None
    # Fast mode parameters
    use_fast_mode: bool = False  # Enable fast evaluation without search (<100ms per move)
    exploration_decay: float = 0.995  # Decay factor for epsilon/temperature over time
    epsilon_greedy: float = 0.0  # Epsilon-greedy exploration parameter (0-1)
    min_temperature: float = 0.1  # Lower bound for temperature decay


class HeuristicPolicy:
    """Heuristic-based move selection policy using HeuristicAgent."""
    
    def __init__(self, config: HeuristicPolicyConfig = None):
        """Initialize the heuristic policy.
        
        Args:
            config: Policy configuration
        """
        self.config = config or HeuristicPolicyConfig()
        
        # Import here to avoid circular dependencies
        from ..agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
        
        # Create HeuristicAgent with appropriate config
        agent_config = HeuristicAgentConfig(
            max_depth=self.config.search_depth,
            time_limit_seconds=self.config.time_limit,
            random_seed=self.config.random_seed
        )
        self.agent = HeuristicAgent(config=agent_config)
        
        self._rng = random.Random(self.config.random_seed)
        
        # Performance tracking
        self._move_times: List[float] = []
        self._max_move_time_history = 100  # Keep last 100 move times
        
        # Move and game tracking for decay
        self._move_count = 0  # Moves in current game
        self._game_count = 0  # Total games played
        
        # Cache heuristic evaluator reference for fast mode
        self._evaluator = self.agent._evaluator
        
        # Current temperature and epsilon (decay over time)
        self._current_temperature = self.config.temperature
        self._current_epsilon = self.config.epsilon_greedy
        
        logger.info(f"Initialized HeuristicPolicy with depth={self.config.search_depth}, "
                   f"randomness={self.config.randomness}, temperature={self.config.temperature}, "
                   f"fast_mode={self.config.use_fast_mode}")
    
    def select_move(self, game_state: GameState) -> Move:
        """Select a move using heuristic evaluation with exploration.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move
            
        Raises:
            ValueError: If no valid moves are available
        """
        start_time = time.perf_counter()
        
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Update decay values before move selection
        self._update_decay()
        
        # Fast mode: skip search, use direct heuristic evaluation
        if self.config.use_fast_mode:
            move = self._select_move_fast(game_state, valid_moves)
        else:
            # Original mode: use agent search
            # Epsilon-greedy exploration: random move with probability = randomness
            if self._rng.random() < self.config.randomness:
                move = self._rng.choice(valid_moves)
                logger.debug(f"Selected random move (exploration): {move}")
            else:
                # Get best move from heuristic agent
                best_move = self.agent.select_move(game_state)
                
                # Apply temperature-based selection if temperature > 0
                if self.config.temperature > 0 and len(valid_moves) > 1:
                    # Get heuristic scores for all moves for temperature sampling
                    scored_moves = self._score_all_moves(game_state, valid_moves, fast_evaluation=False)
                    move = self._sample_with_temperature(scored_moves)
                    logger.debug(f"Selected move with temperature sampling: {move}")
                else:
                    move = best_move
                    logger.debug(f"Selected heuristic move: {move}")
        
        # Track performance
        duration = time.perf_counter() - start_time
        self._track_performance(duration)
        
        # Increment move count for decay calculations
        self._move_count += 1
        
        return move
    
    def _select_move_fast(self, game_state: GameState, valid_moves: List[Move]) -> Move:
        """Fast move selection path without search (<100ms target).
        
        Args:
            game_state: Current game state
            valid_moves: List of valid moves
            
        Returns:
            Selected move
        """
        # Apply epsilon-greedy exploration if configured
        if self._current_epsilon > 0 and self._rng.random() < self._current_epsilon:
            move = self._rng.choice(valid_moves)
            logger.debug(f"Selected random move (epsilon-greedy): {move}")
            return move
        
        # Score all moves using fast evaluation
        scored_moves = self._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        
        # Apply temperature-based sampling with current (decayed) temperature
        move = self._sample_with_temperature(scored_moves, use_current_temp=True)
        logger.debug(f"Selected fast heuristic move: {move}")
        return move
    
    def _score_all_moves(self, game_state: GameState, valid_moves: List[Move], fast_evaluation: bool = False) -> List[tuple]:
        """Score all valid moves using heuristic evaluation.
        
        Args:
            game_state: Current game state
            valid_moves: List of valid moves
            fast_evaluation: If True, use direct evaluation without game copies (faster)
            
        Returns:
            List of (score, move) tuples, sorted by score (descending)
        """
        scored = []
        current_player = game_state.current_player
        
        if fast_evaluation:
            # Fast path: evaluate directly without full game copy overhead
            for move in valid_moves:
                # Create minimal copy only for move application
                state_copy = game_state.copy()
                if state_copy.make_move(move):
                    # Use cached evaluator directly
                    score = self._evaluator.evaluate_position(state_copy, current_player)
                    scored.append((score, move))
                else:
                    # Invalid move, assign low score
                    scored.append((-10000.0, move))
        else:
            # Original path: use agent's evaluation method
            for move in valid_moves:
                # Create a copy and evaluate the position after the move
                state_copy = game_state.copy()
                if state_copy.make_move(move):
                    # Use the agent's evaluator to get position score
                    score = self.agent._evaluate_position(state_copy, current_player)
                    scored.append((score, move))
                else:
                    # Invalid move, assign low score
                    scored.append((-10000.0, move))
        
        # Sort by score (descending) for consistent ranking
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    
    def _sample_with_temperature(self, scored_moves: List[tuple], use_current_temp: bool = False) -> Move:
        """Sample a move using temperature-based probability distribution.
        
        Args:
            scored_moves: List of (score, move) tuples (should be sorted by score)
            use_current_temp: If True, use decayed current temperature instead of config temperature
            
        Returns:
            Sampled move
        """
        if not scored_moves:
            raise ValueError("No moves to sample from")
        
        # Extract scores and normalize
        scores = np.array([score for score, _ in scored_moves])
        
        # Determine temperature to use
        if use_current_temp:
            temp = max(self.config.min_temperature, self._current_temperature)
        else:
            temp = self.config.temperature
        
        # Apply temperature: divide by temperature, then softmax
        if temp > 0:
            scores = scores / temp
        else:
            # Temperature = 0: deterministic, pick best (first in sorted list)
            return scored_moves[0][1]
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = exp_scores / exp_scores.sum()
        
        # Sample according to probabilities
        move_idx = self._rng.choices(range(len(scored_moves)), weights=probabilities)[0]
        return scored_moves[move_idx][1]
    
    def _update_decay(self):
        """Update temperature and epsilon based on decay schedule."""
        # Update temperature: decay per move
        if self.config.exploration_decay < 1.0:
            self._current_temperature = max(
                self.config.min_temperature,
                self.config.temperature * (self.config.exploration_decay ** self._move_count)
            )
        
        # Update epsilon: decay per game
        if self.config.epsilon_greedy > 0 and self.config.exploration_decay < 1.0:
            self._current_epsilon = max(
                0.0,
                self.config.epsilon_greedy * (self.config.exploration_decay ** self._game_count)
            )
    
    def _track_performance(self, duration: float):
        """Track move selection performance.
        
        Args:
            duration: Time taken for move selection in seconds
        """
        self._move_times.append(duration)
        
        # Keep only last N move times
        if len(self._move_times) > self._max_move_time_history:
            self._move_times = self._move_times[-self._max_move_time_history:]
        
        # Warn if move exceeds 100ms threshold (only in fast mode)
        if self.config.use_fast_mode and duration > 0.1:
            logger.warning(
                f"Move selection took {duration*1000:.2f}ms (exceeds 100ms threshold in fast mode)"
            )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for move selection.
        
        Returns:
            Dictionary with performance metrics:
            - avg_move_time: Average move selection time in seconds
            - max_move_time: Maximum move selection time in seconds
            - p95_move_time: 95th percentile move selection time in seconds
            - total_moves: Total number of moves tracked
        """
        if not self._move_times:
            return {
                "avg_move_time": 0.0,
                "max_move_time": 0.0,
                "p95_move_time": 0.0,
                "total_moves": 0
            }
        
        sorted_times = sorted(self._move_times)
        n = len(sorted_times)
        
        avg_time = sum(sorted_times) / n
        max_time = max(sorted_times)
        p95_idx = int(0.95 * n) if n > 0 else 0
        p95_time = sorted_times[p95_idx] if p95_idx < n else sorted_times[-1]
        
        return {
            "avg_move_time": avg_time,
            "max_move_time": max_time,
            "p95_move_time": p95_time,
            "total_moves": n
        }
    
    def reset_for_new_game(self):
        """Reset state for a new game (call at start of each game)."""
        self._move_count = 0
        self._game_count += 1
        
        # Update decay values based on new game count
        self._update_decay()
        
        logger.debug(f"Reset for new game (game #{self._game_count}, "
                    f"temp={self._current_temperature:.3f}, epsilon={self._current_epsilon:.3f})")


@dataclass
class MCTSPolicyConfig:
    """Configuration for MCTS-based move selection policy."""
    num_simulations: int = 100  # MCTS simulation budget
    evaluation_mode: str = "hybrid"  # pure_heuristic | hybrid | pure_neural
    heuristic_weight: float = 0.5  # Weight for heuristic in hybrid mode
    use_dirichlet: bool = True  # Add exploration noise
    random_seed: Optional[int] = None
    # Additional MCTS parameters
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    max_depth: int = 50


class MCTSPolicy:
    """MCTS-based move selection policy using Monte Carlo Tree Search."""
    
    def __init__(self, config: MCTSPolicyConfig = None, network=None):
        """Initialize the MCTS policy.
        
        Args:
            config: Policy configuration
            network: Optional neural network wrapper. If None, will use pure heuristic mode.
        """
        self.config = config or MCTSPolicyConfig()
        
        # Import here to avoid circular dependencies
        from ..search.mcts import MCTS, MCTSConfig, EvaluationMode
        
        # Determine evaluation mode
        if self.config.evaluation_mode == "pure_heuristic":
            eval_mode = EvaluationMode.PURE_HEURISTIC
        elif self.config.evaluation_mode == "pure_neural":
            eval_mode = EvaluationMode.PURE_NEURAL
        else:
            eval_mode = EvaluationMode.HYBRID
        
        # If no network provided and mode requires neural, fall back to heuristic
        if network is None and eval_mode == EvaluationMode.PURE_NEURAL:
            logger.warning("No network provided for pure_neural mode, falling back to pure_heuristic")
            eval_mode = EvaluationMode.PURE_HEURISTIC
        
        # Create MCTS config
        mcts_config = MCTSConfig(
            num_simulations=self.config.num_simulations,
            evaluation_mode=eval_mode,
            heuristic_weight=self.config.heuristic_weight,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha if self.config.use_dirichlet else 0.0,
            max_depth=self.config.max_depth
        )
        
        # Create a dummy network if needed (MCTS will use heuristic evaluation)
        if network is None:
            # Create a minimal network wrapper that will be ignored in pure heuristic mode
            from ..network.wrapper import NetworkWrapper
            network = NetworkWrapper()
        
        # Create MCTS instance
        self.mcts = MCTS(network=network, config=mcts_config)
        
        self._rng = random.Random(self.config.random_seed)
        self._move_number = 0
        
        logger.info(f"Initialized MCTSPolicy with mode={self.config.evaluation_mode}, "
                   f"simulations={self.config.num_simulations}")
    
    def select_move(self, game_state: GameState) -> Move:
        """Select a move using MCTS search.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move
            
        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Run MCTS search to get policy vector
        policy_vector = self.mcts.search(game_state, self._move_number)
        
        # Convert policy vector to move probabilities
        move_probs = self._policy_to_move_probs(policy_vector, valid_moves)
        
        # Sample move from distribution
        move = self._sample_move(valid_moves, move_probs)
        
        self._move_number += 1
        logger.debug(f"Selected MCTS move: {move}")
        return move
    
    def _policy_to_move_probs(self, policy_vector: np.ndarray, valid_moves: List[Move]) -> Dict[Move, float]:
        """Convert policy vector to move probabilities.
        
        Args:
            policy_vector: Policy vector from MCTS
            valid_moves: List of valid moves
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        from ..utils.encoding import StateEncoder
        encoder = StateEncoder()
        
        move_probs = {}
        total_prob = 0.0
        
        for move in valid_moves:
            # Get index for this move
            move_idx = encoder.move_to_index(move)
            if 0 <= move_idx < len(policy_vector):
                prob = float(policy_vector[move_idx])
                move_probs[move] = prob
                total_prob += prob
        
        # Normalize probabilities
        if total_prob > 0:
            for move in move_probs:
                move_probs[move] /= total_prob
        else:
            # Uniform distribution if no probabilities
            uniform_prob = 1.0 / len(valid_moves)
            for move in valid_moves:
                move_probs[move] = uniform_prob
        
        return move_probs
    
    def _sample_move(self, valid_moves: List[Move], move_probs: Dict[Move, float]) -> Move:
        """Sample a move from the probability distribution.
        
        Args:
            valid_moves: List of valid moves
            move_probs: Dictionary mapping moves to probabilities
            
        Returns:
            Sampled move
        """
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Extract moves and probabilities in order
        moves = []
        probs = []
        for move in valid_moves:
            moves.append(move)
            probs.append(move_probs.get(move, 0.0))
        
        # Sample according to probabilities
        move_idx = self._rng.choices(range(len(moves)), weights=probs)[0]
        return moves[move_idx]


@dataclass
class AdaptivePolicyConfig:
    """Configuration for adaptive policy that transitions between guidance systems."""
    initial_policy: str = "heuristic"  # Starting policy type: heuristic | mcts | neural
    target_policy: str = "neural"  # Target policy type: heuristic | mcts | neural
    transition_schedule: str = "linear"  # linear | exponential | step
    transition_steps: int = 10000  # Number of games for full transition
    checkpoint_interval: int = 1000  # Check neural network quality every N games
    # Policy-specific configs
    heuristic_config: Optional[HeuristicPolicyConfig] = None
    mcts_config: Optional[MCTSPolicyConfig] = None
    # Neural network quality threshold (0-1)
    min_neural_quality: float = 0.5  # Minimum quality to use neural network
    random_seed: Optional[int] = None


class AdaptivePolicy:
    """Adaptive policy that gradually transitions from heuristic to neural guidance."""
    
    def __init__(self, config: AdaptivePolicyConfig = None, network=None, 
                 training_progress_callback=None):
        """Initialize the adaptive policy.
        
        Args:
            config: Policy configuration
            network: Optional neural network wrapper
            training_progress_callback: Optional callback(game_count) -> float that returns
                training progress (0-1) or neural network quality metric
        """
        self.config = config or AdaptivePolicyConfig()
        self.network = network
        self.training_progress_callback = training_progress_callback
        
        # Track progress
        self._game_count = 0
        self._current_policy = None
        self._heuristic_policy = None
        self._mcts_policy = None
        self._neural_policy = None
        
        # Initialize policies
        self._initialize_policies()
        
        # Set initial policy
        self._update_policy()
        
        self._rng = random.Random(self.config.random_seed)
        
        logger.info(f"Initialized AdaptivePolicy: {self.config.initial_policy} -> "
                   f"{self.config.target_policy} ({self.config.transition_schedule})")
    
    def _initialize_policies(self):
        """Initialize all policy instances."""
        # Heuristic policy
        heuristic_config = self.config.heuristic_config or HeuristicPolicyConfig(
            random_seed=self.config.random_seed
        )
        self._heuristic_policy = HeuristicPolicy(config=heuristic_config)
        
        # MCTS policy
        mcts_config = self.config.mcts_config or MCTSPolicyConfig(
            random_seed=self.config.random_seed
        )
        self._mcts_policy = MCTSPolicy(config=mcts_config, network=self.network)
        
        # Neural policy (MCTS with pure neural mode)
        if self.network is not None:
            neural_mcts_config = MCTSPolicyConfig(
                evaluation_mode="pure_neural",
                random_seed=self.config.random_seed
            )
            self._neural_policy = MCTSPolicy(config=neural_mcts_config, network=self.network)
        else:
            logger.warning("No network provided, neural policy will not be available")
            self._neural_policy = None
    
    def _update_policy(self):
        """Update the current policy based on transition schedule."""
        # Get current progress (0-1)
        progress = self._get_transition_progress()
        
        # Determine which policy to use
        if self.config.target_policy == "neural" and self._neural_policy is None:
            # Fallback: can't use neural, use MCTS or heuristic
            if progress > 0.5:
                self._current_policy = self._mcts_policy
            else:
                self._current_policy = self._heuristic_policy
        elif progress < 0.5:
            # First half: use initial policy
            if self.config.initial_policy == "heuristic":
                self._current_policy = self._heuristic_policy
            elif self.config.initial_policy == "mcts":
                self._current_policy = self._mcts_policy
            else:
                self._current_policy = self._neural_policy or self._mcts_policy
        else:
            # Second half: transition to target policy
            if self.config.target_policy == "neural":
                # Check neural network quality
                if self._neural_policy is not None and self._check_neural_quality():
                    self._current_policy = self._neural_policy
                else:
                    # Fallback to MCTS hybrid
                    self._current_policy = self._mcts_policy
            elif self.config.target_policy == "mcts":
                self._current_policy = self._mcts_policy
            else:
                self._current_policy = self._heuristic_policy
    
    def _get_transition_progress(self) -> float:
        """Get current transition progress (0-1).
        
        Returns:
            Progress value between 0 and 1
        """
        if self.config.transition_steps <= 0:
            return 1.0
        
        # Get progress from callback if available
        if self.training_progress_callback is not None:
            callback_progress = self.training_progress_callback(self._game_count)
            if callback_progress is not None:
                return min(1.0, max(0.0, callback_progress))
        
        # Calculate progress based on game count
        raw_progress = self._game_count / self.config.transition_steps
        
        if self.config.transition_schedule == "linear":
            return min(1.0, raw_progress)
        elif self.config.transition_schedule == "exponential":
            # Exponential: faster transition early, slower later
            return min(1.0, 1.0 - math.exp(-2.0 * raw_progress))
        elif self.config.transition_schedule == "step":
            # Step: discrete transitions at intervals
            steps = max(1, self.config.transition_steps // 4)  # Ensure at least 1
            return min(1.0, (self._game_count // steps) / 4.0)
        else:
            return min(1.0, raw_progress)
    
    def _check_neural_quality(self) -> bool:
        """Check if neural network quality is sufficient.
        
        Returns:
            True if neural network should be used
        """
        if self.training_progress_callback is None:
            # No quality metric available, use based on game count
            return self._game_count >= self.config.checkpoint_interval
        
        # Check quality via callback
        quality = self.training_progress_callback(self._game_count)
        if quality is None:
            return False
        
        return quality >= self.config.min_neural_quality
    
    def select_move(self, game_state: GameState) -> Move:
        """Select a move using the current policy.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move
        """
        # Update policy periodically
        if self._game_count % self.config.checkpoint_interval == 0:
            self._update_policy()
        
        # Select move using current policy
        move = self._current_policy.select_move(game_state)
        
        # Increment game count (assuming one move per call, will be reset per game)
        return move
    
    def reset_game(self):
        """Reset state for a new game."""
        # Update policy at start of each game
        self._update_policy()
    
    def increment_game_count(self):
        """Increment the game counter (call after each completed game)."""
        self._game_count += 1
        self._update_policy()


class PolicyFactory:
    """Factory for creating move selection policies."""
    
    @staticmethod
    def create_random_policy(config: PolicyConfig = None) -> RandomMovePolicy:
        """Create a random move policy.
        
        Args:
            config: Policy configuration
            
        Returns:
            Random move policy instance
        """
        return RandomMovePolicy(config)
    
    @staticmethod
    def create_heuristic_policy(config: HeuristicPolicyConfig = None) -> HeuristicPolicy:
        """Create a heuristic move policy.
        
        Args:
            config: Heuristic policy configuration
            
        Returns:
            Heuristic policy instance
        """
        return HeuristicPolicy(config)
    
    @staticmethod
    def create_mcts_policy(config: MCTSPolicyConfig = None, network=None) -> MCTSPolicy:
        """Create an MCTS move policy.
        
        Args:
            config: MCTS policy configuration
            network: Optional neural network wrapper
            
        Returns:
            MCTS policy instance
        """
        return MCTSPolicy(config=config, network=network)
    
    @staticmethod
    def create_adaptive_policy(config: AdaptivePolicyConfig = None, network=None,
                              training_progress_callback=None) -> AdaptivePolicy:
        """Create an adaptive policy.
        
        Args:
            config: Adaptive policy configuration
            network: Optional neural network wrapper
            training_progress_callback: Optional callback for training progress
            
        Returns:
            Adaptive policy instance
        """
        return AdaptivePolicy(config=config, network=network,
                            training_progress_callback=training_progress_callback)
    
    @staticmethod
    def create_policy(policy_type: str, config: Any = None, network=None,
                     training_progress_callback=None) -> Any:
        """Create a policy by type.
        
        Args:
            policy_type: Type of policy to create (random | heuristic | mcts | adaptive)
            config: Policy configuration (type depends on policy_type)
            network: Optional neural network wrapper (for mcts/adaptive)
            training_progress_callback: Optional callback (for adaptive)
            
        Returns:
            Policy instance
            
        Raises:
            ValueError: If policy type is not supported
        """
        if policy_type == "random":
            return PolicyFactory.create_random_policy(config)
        elif policy_type == "heuristic":
            return PolicyFactory.create_heuristic_policy(config)
        elif policy_type == "mcts":
            return PolicyFactory.create_mcts_policy(config, network)
        elif policy_type == "adaptive":
            return PolicyFactory.create_adaptive_policy(config, network, training_progress_callback)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}. "
                           f"Supported types: random, heuristic, mcts, adaptive")
