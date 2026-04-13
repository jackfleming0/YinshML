"""Monte Carlo Tree Search with YinshHeuristics integration.

This module provides MCTS implementation that integrates the YinshHeuristics
evaluator for leaf node evaluation, supporting both pure heuristic and hybrid
evaluation modes with adaptive weight reduction.
"""

import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..game.game_state import GameState
from ..game.constants import Player
from ..utils.encoding import StateEncoder
from ..utils.mcts_metrics import MCTSMetrics
from ..memory.game_state_pool import GameStatePool
from ..network.wrapper import NetworkWrapper
from ..heuristics import YinshHeuristics
from .training_tracker import TrainingTracker


class EvaluationMode(Enum):
    """Evaluation mode for MCTS leaf nodes."""
    PURE_HEURISTIC = "pure_heuristic"
    PURE_NEURAL = "pure_neural"
    HYBRID = "hybrid"


@dataclass
class MCTSConfig:
    """Configuration for MCTS with heuristic integration."""
    
    # Standard MCTS parameters
    num_simulations: int = 100
    late_simulations: Optional[int] = None
    simulation_switch_ply: int = 20
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    value_weight: float = 1.0
    max_depth: int = 50
    
    # Temperature parameters
    initial_temp: float = 1.0
    final_temp: float = 0.1
    annealing_steps: int = 30
    temp_clamp_fraction: float = 0.8
    
    # Heuristic evaluation parameters
    evaluation_mode: EvaluationMode = EvaluationMode.HYBRID
    use_heuristic_evaluation: bool = True
    heuristic_weight: float = 0.3  # Weight for heuristic in hybrid mode
    neural_weight: float = 0.7  # Weight for neural network in hybrid mode
    
    # Adaptive weight reduction parameters
    auto_reduce_heuristic_weight: bool = True
    min_heuristic_weight: float = 0.1
    max_heuristic_weight: float = 0.5
    initial_heuristic_weight: float = 0.5
    
    # Heuristic score normalization
    heuristic_score_scale: float = 50.0  # Scale factor for tanh normalization
    
    # Phase-aware weighting
    use_phase_aware_weighting: bool = True


class MCTSNode:
    """MCTS node with heuristic evaluation caching."""
    
    def __init__(self, state: GameState, parent=None, prior_prob=0.0, c_puct=1.0, 
                 state_pool=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False
        self.c_puct = c_puct
        self._state_pool = state_pool
        self._owns_state = False
        
        # Cached evaluations
        self.heuristic_value = None
        self.neural_value = None
    
    def __del__(self):
        """Clean up GameState when node is destroyed."""
        if self._owns_state and self._state_pool is not None and self.state is not None:
            try:
                self._state_pool.return_game_state(self.state)
            except Exception as e:
                import logging
                logging.getLogger("MCTSNode").warning(f"Failed to release state: {e}")
    
    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, parent_visit_count: int) -> float:
        """Calculate Upper Confidence Bound score."""
        q_value = self.value()
        u_value = (self.c_puct * self.prior_prob *
                  np.sqrt(parent_visit_count) / (1 + self.visit_count))
        return q_value + u_value


class MCTS:
    """Monte Carlo Tree Search with YinshHeuristics integration."""
    
    def __init__(self,
                 network: NetworkWrapper,
                 config: Optional[MCTSConfig] = None,
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 game_state_pool: Optional[GameStatePool] = None,
                 training_tracker: Optional[TrainingTracker] = None):
        """
        Initialize MCTS with heuristic integration.
        
        Args:
            network: The neural network wrapper
            config: MCTS configuration (uses defaults if None)
            mcts_metrics: Optional metrics collector
            game_state_pool: Optional memory pool for game states
            training_tracker: Optional training progress tracker for adaptive weights
        """
        self.network = network
        self.config = config or MCTSConfig()
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("MCTS")
        
        # Memory pool management
        self.game_state_pool = game_state_pool
        self._pool_enabled = game_state_pool is not None
        
        # Metrics tracking
        self.metrics = mcts_metrics if mcts_metrics is not None else MCTSMetrics()
        
        # Initialize heuristic evaluator
        self.heuristic_evaluator = None
        if self.config.use_heuristic_evaluation:
            try:
                self.heuristic_evaluator = YinshHeuristics()
                self.logger.info("YinshHeuristics evaluator initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize heuristic evaluator: {e}")
                self.heuristic_evaluator = None
                # Fallback to pure neural if heuristic fails
                if self.config.evaluation_mode == EvaluationMode.PURE_HEURISTIC:
                    self.config.evaluation_mode = EvaluationMode.PURE_NEURAL
                    self.logger.warning("Falling back to pure neural evaluation mode")
        
        # Training tracker for adaptive weights
        self.training_tracker = training_tracker
        
        # Update heuristic weight based on training progress if adaptive mode enabled
        if self.config.auto_reduce_heuristic_weight and self.training_tracker is not None:
            self._update_heuristic_weight_from_tracker()
        
        self.logger.info(f"MCTS Initialized:")
        self.logger.info(f"  Evaluation Mode: {self.config.evaluation_mode.value}")
        self.logger.info(f"  Heuristic Weight: {self.config.heuristic_weight}")
        self.logger.info(f"  Use Heuristic: {self.config.use_heuristic_evaluation}")
    
    def _update_heuristic_weight_from_tracker(self):
        """Update heuristic weight based on training progress."""
        if self.training_tracker is None:
            return
        
        improvement_factor = self.training_tracker.get_improvement_factor()
        # Reduce heuristic weight as network improves
        new_weight = max(
            self.config.min_heuristic_weight,
            self.config.initial_heuristic_weight * (1 - improvement_factor)
        )
        new_weight = min(new_weight, self.config.max_heuristic_weight)
        
        if new_weight != self.config.heuristic_weight:
            self.logger.info(f"Updating heuristic weight: {self.config.heuristic_weight:.3f} -> {new_weight:.3f}")
            self.config.heuristic_weight = new_weight
            # Update neural weight to maintain sum = 1.0
            self.config.neural_weight = 1.0 - new_weight
    
    def _acquire_state_copy(self, original_state: GameState) -> GameState:
        """Get a GameState copy from pool or create new one."""
        if self._pool_enabled:
            try:
                pooled_state = self.game_state_pool.get()
                pooled_state.copy_from(original_state)
                return pooled_state
            except Exception as e:
                self.logger.warning(f"Failed to get state from pool: {e}, falling back to copy()")
                return copy.deepcopy(original_state)
        else:
            return copy.deepcopy(original_state)
    
    def _release_state(self, state: GameState) -> None:
        """Release a GameState back to the pool."""
        if self._pool_enabled and state is not None:
            try:
                self.game_state_pool.return_game_state(state)
            except Exception as e:
                self.logger.warning(f"Failed to return state to pool: {e}")
    
    def _create_child_node(self, state: GameState, parent, prior_prob: float) -> MCTSNode:
        """Create a child node with proper memory management."""
        if self._pool_enabled:
            child_state = self._acquire_state_copy(state)
            child_node = MCTSNode(
                child_state,
                parent=parent,
                prior_prob=prior_prob,
                c_puct=self.config.c_puct,
                state_pool=self.game_state_pool
            )
            child_node._owns_state = True
            return child_node
        else:
            return MCTSNode(
                state.copy(),
                parent=parent,
                prior_prob=prior_prob,
                c_puct=self.config.c_puct
            )
    
    def _get_simulation_budget(self, move_number: int) -> int:
        """Get simulation budget based on move number."""
        if move_number < self.config.simulation_switch_ply:
            return self.config.num_simulations
        else:
            return self.config.late_simulations or self.config.num_simulations
    
    def _select_action(self, node: MCTSNode) -> Optional[int]:
        """Select action using UCB."""
        if not node.children:
            return None
        
        parent_visit_count = node.visit_count
        best_action = None
        best_score = float('-inf')
        
        for action, child in node.children.items():
            score = child.get_ucb_score(parent_visit_count)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _evaluate_state(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Evaluate state using neural network."""
        state_tensor = self.state_encoder.encode_state(state)
        policy, value = self.network.predict(state_tensor)
        return policy, value
    
    def _get_terminal_value(self, state: GameState) -> Optional[float]:
        """Get terminal value if state is terminal."""
        if state.is_terminal():
            winner = state.get_winner()
            if winner == Player.WHITE:
                return 1.0
            elif winner == Player.BLACK:
                return -1.0
            else:
                return 0.0
        return None
    
    def _normalize_heuristic_score(self, score: float) -> float:
        """Normalize heuristic score to [-1, 1] range."""
        # Use tanh to normalize heuristic differential score
        normalized = np.tanh(score / self.config.heuristic_score_scale)
        return normalized
    
    def _evaluate_leaf_node(self, state: GameState, node: MCTSNode) -> float:
        """
        Evaluate a leaf node using heuristic and/or neural network.
        
        This is the core integration point for YinshHeuristics.
        """
        # Check terminal first
        terminal_value = self._get_terminal_value(state)
        if terminal_value is not None:
            return terminal_value
        
        # Get neural network evaluation
        _, neural_value = self._evaluate_state(state)
        node.neural_value = neural_value
        
        # Determine evaluation mode
        mode = self.config.evaluation_mode
        
        # If heuristic evaluator not available, fallback to neural
        if self.heuristic_evaluator is None:
            return neural_value
        
        # Get heuristic evaluation
        try:
            heuristic_score = self.heuristic_evaluator.evaluate_position(
                state, state.current_player
            )
            # Normalize heuristic score to [-1, 1] range
            heuristic_value = self._normalize_heuristic_score(heuristic_score)
            node.heuristic_value = heuristic_value
        except Exception as e:
            self.logger.warning(f"Heuristic evaluation failed: {e}, falling back to neural")
            return neural_value
        
        # Apply evaluation mode
        if mode == EvaluationMode.PURE_HEURISTIC:
            return heuristic_value
        elif mode == EvaluationMode.PURE_NEURAL:
            return neural_value
        elif mode == EvaluationMode.HYBRID:
            # Get phase-aware weights if enabled
            heuristic_weight = self.config.heuristic_weight
            neural_weight = self.config.neural_weight
            
            if self.config.use_phase_aware_weighting:
                move_count = len(state.move_history)
                # More heuristic in early game, more neural in late game
                if move_count < 20:
                    # Early game: favor heuristic more
                    phase_factor = 1.2
                elif move_count < 40:
                    # Mid game: balanced
                    phase_factor = 1.0
                else:
                    # Late game: favor neural more
                    phase_factor = 0.8
                
                heuristic_weight *= phase_factor
                neural_weight *= (2.0 - phase_factor)
                
                # Normalize weights
                total_weight = heuristic_weight + neural_weight
                heuristic_weight /= total_weight
                neural_weight /= total_weight
            
            # Combine evaluations
            combined_value = (heuristic_weight * heuristic_value + 
                            neural_weight * neural_value)
            return combined_value
        else:
            # Fallback to neural
            return neural_value
    
    def _mask_invalid_moves(self, policy: np.ndarray, valid_moves: List[int]) -> np.ndarray:
        """Mask invalid moves in policy vector."""
        masked_policy = np.zeros_like(policy)
        if valid_moves:
            valid_indices = [self.state_encoder.move_to_index(move) for move in valid_moves]
            valid_indices = [idx for idx in valid_indices if 0 <= idx < len(policy)]
            if valid_indices:
                masked_policy[valid_indices] = policy[valid_indices]
                masked_policy[valid_indices] /= masked_policy[valid_indices].sum()
        return masked_policy
    
    def _get_move_prob(self, policy: np.ndarray, move: int) -> float:
        """Get probability for a specific move from policy vector."""
        move_idx = self.state_encoder.move_to_index(move)
        if 0 <= move_idx < len(policy):
            return policy[move_idx]
        return 0.0
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Backpropagate value through search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
    
    def get_temperature(self, move_number: int) -> float:
        """Calculate temperature based on linear annealing schedule."""
        if self.config.annealing_steps <= 0:
            return self.config.final_temp
        
        clamp_moves = int(self.config.annealing_steps * self.config.temp_clamp_fraction)
        clamp_moves = max(1, clamp_moves)
        
        if move_number >= clamp_moves:
            return self.config.final_temp
        else:
            progress = move_number / clamp_moves
            return (self.config.initial_temp - 
                   (self.config.initial_temp - self.config.final_temp) * progress)
    
    def search(self, state: GameState, move_number: int) -> np.ndarray:
        """
        Perform MCTS search with heuristic integration.
        
        Args:
            state: Current game state
            move_number: Current move number for temperature calculation
            
        Returns:
            Policy vector over all possible moves
        """
        # Update heuristic weight if adaptive mode enabled
        if self.config.auto_reduce_heuristic_weight and self.training_tracker is not None:
            self._update_heuristic_weight_from_tracker()
        
        # Get simulation budget
        budget = self._get_simulation_budget(move_number)
        
        root = MCTSNode(state, c_puct=self.config.c_puct)
        move_probs = np.zeros(self.state_encoder.total_moves, dtype=np.float32)
        
        # Track states for cleanup
        simulation_states = []
        
        # Run simulations
        for sim in range(budget):
            node = root
            search_path = [node]
            current_state = self._acquire_state_copy(state)
            simulation_states.append(current_state)
            depth = 0
            action = None
            
            # 1. Selection
            while node.is_expanded and node.children:
                action = self._select_action(node)
                if action is None or action not in node.children:
                    self.logger.debug(f"Selection failed at depth {depth}")
                    break
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                depth += 1
                if depth >= self.config.max_depth:
                    break
            
            if action is None or (action not in node.children and node.is_expanded and node.children):
                self.logger.debug(f"Skipping simulation {sim+1} due to selection error")
                continue
            
            # 2. Expansion & Evaluation
            value = self._get_terminal_value(current_state)
            
            if value is None and depth < self.config.max_depth:
                # Expand node
                policy, _ = self._evaluate_state(current_state)
                
                # Check if state became terminal
                if current_state.is_terminal():
                    value = self._get_terminal_value(current_state)
                else:
                    # Evaluate leaf node using heuristic/neural/hybrid
                    value = self._evaluate_leaf_node(current_state, node)
                    
                    # Expand node
                    valid_moves = current_state.get_valid_moves()
                    if valid_moves:
                        node.is_expanded = True
                        policy = self._mask_invalid_moves(policy, valid_moves)
                        
                        for move in valid_moves:
                            node.children[move] = self._create_child_node(
                                current_state,
                                parent=node,
                                prior_prob=self._get_move_prob(policy, move)
                            )
                        
                        # Apply Dirichlet noise at root
                        if node is root and not hasattr(root, "dirichlet_applied"):
                            if self.config.dirichlet_alpha > 0 and len(valid_moves) > 0:
                                noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_moves))
                                epsilon_mix = 0.25
                                for i, move in enumerate(valid_moves):
                                    child_node = root.children[move]
                                    child_node.prior_prob = ((1 - epsilon_mix) * child_node.prior_prob + 
                                                           epsilon_mix * noise[i])
                                root.dirichlet_applied = True
            elif value is None and depth >= self.config.max_depth:
                # Max depth reached, evaluate leaf node
                value = self._evaluate_leaf_node(current_state, node)
            
            # 3. Backpropagation
            if value is None:
                self.logger.error("Value is None before backpropagation")
                value = 0.0
            
            self._backpropagate(search_path, value)
        
        # Clean up simulation states
        for state in simulation_states:
            self._release_state(state)
        
        # Calculate final move probabilities
        if not root.children:
            self.logger.debug("Root node has no children, returning uniform policy")
            valid_moves = state.get_valid_moves()
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for move in valid_moves:
                    move_idx = self.state_encoder.move_to_index(move)
                    if 0 <= move_idx < len(move_probs):
                        move_probs[move_idx] = prob
            return move_probs
        
        # Get visit counts and apply temperature
        valid_moves_at_root = list(root.children.keys())
        visit_counts = np.array([
            root.children[move].visit_count for move in valid_moves_at_root
        ], dtype=np.float32)
        
        if visit_counts.sum() == 0:
            self.logger.warning("Root children have zero visits, returning uniform policy")
            prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
            visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)
        else:
            temp = self.get_temperature(move_number)
            if temp == 0:
                visit_probs = np.zeros_like(visit_counts)
                visit_probs[np.argmax(visit_counts)] = 1.0
            else:
                visit_counts_temp = np.power(visit_counts, 1.0 / temp)
                total_visits_temp = visit_counts_temp.sum()
                if total_visits_temp > 1e-6:
                    visit_probs = visit_counts_temp / total_visits_temp
                else:
                    prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
                    visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)
        
        # Store probabilities in policy vector
        for move, prob in zip(valid_moves_at_root, visit_probs):
            move_idx = self.state_encoder.move_to_index(move)
            if 0 <= move_idx < len(move_probs):
                move_probs[move_idx] = prob

        # Store root value for use as training target (Fix #1)
        self.last_root_value = root.value() if root.visit_count > 0 else 0.0

        return move_probs

