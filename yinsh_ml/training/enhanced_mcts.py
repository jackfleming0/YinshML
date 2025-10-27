"""Enhanced MCTS implementation incorporating analysis findings.

This module extends the standard MCTS with insights from:
- Phase-based analysis for game phase awareness
- Heuristic evaluation function for faster simulations
- Linear analysis insights for improved UCB1 selection
- Phase-aware simulation budget allocation
"""

import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..game.game_state import GameState
from ..game.constants import Player
from ..utils.encoding import StateEncoder
from ..utils.mcts_metrics import MCTSMetrics
from ..memory.game_state_pool import GameStatePool
from ..network.wrapper import NetworkWrapper
from ..analysis.heuristic_evaluator import HeuristicEvaluator, EvaluationConfig
from ..analysis.phase_analyzer import PhaseAnalyzer, GamePhase, PhaseConfig


@dataclass
class EnhancedMCTSConfig:
    """Configuration for enhanced MCTS with analysis integration."""
    
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
    
    # Analysis integration parameters
    use_heuristic_evaluation: bool = True
    use_phase_aware_budget: bool = True
    use_enhanced_ucb: bool = True
    heuristic_weight: float = 0.3  # Weight for heuristic vs neural network evaluation
    phase_budget_multipliers: Dict[GamePhase, float] = None  # Will be set in __post_init__
    
    # Heuristic guidance parameters (Task 10)
    use_heuristic_guidance: bool = True
    heuristic_alpha: float = 0.3  # Weight for heuristic in UCB1 combination
    epsilon_greedy: float = 0.4  # Epsilon for greedy rollouts biased by heuristics
    use_heuristic_rollouts: bool = True  # Use heuristic evaluation for leaf nodes
    
    def __post_init__(self):
        """Initialize phase budget multipliers if not provided."""
        if self.phase_budget_multipliers is None:
            self.phase_budget_multipliers = {
                GamePhase.EARLY: 1.0,   # Standard budget for early game
                GamePhase.MID: 1.2,     # 20% more simulations for mid game complexity
                GamePhase.LATE: 0.8     # 20% fewer simulations for late game (more tactical)
            }


class EnhancedNode:
    """Enhanced MCTS node with phase awareness and analysis integration."""
    
    def __init__(self, state: GameState, parent=None, prior_prob=0.0, c_puct=1.0, 
                 state_pool=None, game_phase: GamePhase = None):
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
        self.game_phase = game_phase or self._determine_phase()
        
        # Enhanced metrics for analysis integration
        self.heuristic_value = None  # Cached heuristic evaluation
        self.neural_value = None     # Cached neural network evaluation
        self.last_evaluation_depth = 0  # Track evaluation depth for phase awareness
        
    def _determine_phase(self) -> GamePhase:
        """Determine game phase based on current state."""
        # Use turn number to determine phase
        turn_number = len(self.state.move_history) + 1
        phase_config = PhaseConfig()
        return phase_config.get_phase(turn_number)
    
    def __del__(self):
        """Clean up GameState when node is destroyed."""
        if self._owns_state and self._state_pool is not None and self.state is not None:
            try:
                self._state_pool.return_game_state(self.state)
            except Exception as e:
                import logging
                logging.getLogger("EnhancedNode").warning(f"Failed to release state: {e}")
    
    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_enhanced_ucb_score(self, parent_visit_count: int, config: EnhancedMCTSConfig, 
                              heuristic_evaluator: Optional[HeuristicEvaluator] = None) -> float:
        """Calculate enhanced UCB score with heuristic guidance and phase-aware adjustments."""
        q_value = self.value()
        
        # Base UCB calculation
        exploration_term = (self.c_puct * self.prior_prob * 
                           np.sqrt(parent_visit_count) / (1 + self.visit_count))
        
        # Phase-aware exploration adjustment
        if config.use_enhanced_ucb:
            phase_multiplier = self._get_phase_exploration_multiplier()
            exploration_term *= phase_multiplier
        
        # Heuristic-guided UCB1 combination (Task 10)
        if config.use_heuristic_guidance and heuristic_evaluator is not None:
            # Get heuristic evaluation for this node
            heuristic_value = self._get_heuristic_value(heuristic_evaluator)
            
            # Combine UCB1 with heuristic: combined_score = (1-alpha) * UCB1 + alpha * heuristic_eval
            ucb_score = q_value + exploration_term
            combined_score = ((1 - config.heuristic_alpha) * ucb_score + 
                            config.heuristic_alpha * heuristic_value)
            return combined_score
        
        return q_value + exploration_term
    
    def _get_phase_exploration_multiplier(self) -> float:
        """Get exploration multiplier based on game phase."""
        # Early game: more exploration (higher multiplier)
        # Mid game: balanced exploration
        # Late game: less exploration, more exploitation
        phase_multipliers = {
            GamePhase.EARLY: 1.2,
            GamePhase.MID: 1.0,
            GamePhase.LATE: 0.8
        }
        return phase_multipliers.get(self.game_phase, 1.0)
    
    def _get_heuristic_value(self, heuristic_evaluator: HeuristicEvaluator) -> float:
        """Get heuristic evaluation for this node."""
        if self.heuristic_value is None:
            self.heuristic_value = heuristic_evaluator.evaluate_position_fast(
                self.state, self.state.current_player
            )
        return self.heuristic_value
    
    def get_combined_evaluation(self, heuristic_evaluator: HeuristicEvaluator, 
                              neural_value: float, config: EnhancedMCTSConfig) -> float:
        """Get combined evaluation using both heuristic and neural network."""
        if not config.use_heuristic_evaluation:
            return neural_value
        
        # Use cached heuristic value if available
        if self.heuristic_value is None:
            self.heuristic_value = heuristic_evaluator.evaluate_position_fast(
                self.state, self.state.current_player
            )
        
        # Combine heuristic and neural evaluations
        heuristic_weight = config.heuristic_weight
        neural_weight = 1.0 - heuristic_weight
        
        # Phase-aware weighting: more heuristic in early game, more neural in late game
        if self.game_phase == GamePhase.EARLY:
            heuristic_weight *= 1.2
            neural_weight *= 0.8
        elif self.game_phase == GamePhase.LATE:
            heuristic_weight *= 0.8
            neural_weight *= 1.2
        
        # Normalize weights
        total_weight = heuristic_weight + neural_weight
        heuristic_weight /= total_weight
        neural_weight /= total_weight
        
        return heuristic_weight * self.heuristic_value + neural_weight * neural_value


class EnhancedMCTS:
    """Enhanced MCTS with analysis findings integration."""
    
    def __init__(self,
                 network: NetworkWrapper,
                 config: EnhancedMCTSConfig,
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 game_state_pool: Optional[GameStatePool] = None):
        """
        Initialize enhanced MCTS.
        
        Args:
            network: The neural network wrapper
            config: Enhanced MCTS configuration
            mcts_metrics: Optional metrics collector
            game_state_pool: Optional memory pool for game states
        """
        self.network = network
        self.config = config
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("EnhancedMCTS")
        
        # Memory pool management
        self.game_state_pool = game_state_pool
        self._pool_enabled = game_state_pool is not None
        
        # Metrics tracking
        self.metrics = mcts_metrics if mcts_metrics is not None else MCTSMetrics()
        self.current_iteration = 0
        
        # Initialize analysis components
        self.heuristic_evaluator = None
        if config.use_heuristic_evaluation:
            try:
                eval_config = EvaluationConfig(speed_target=2000)  # Optimize for MCTS speed
                self.heuristic_evaluator = HeuristicEvaluator(config=eval_config)
                self.logger.info("Heuristic evaluator initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize heuristic evaluator: {e}")
                self.heuristic_evaluator = None
        
        self.phase_analyzer = PhaseAnalyzer()
        
        self.logger.info(f"Enhanced MCTS Initialized:")
        self.logger.info(f"  Analysis Integration: Heuristic={config.use_heuristic_evaluation}, "
                        f"Phase-aware={config.use_phase_aware_budget}, Enhanced UCB={config.use_enhanced_ucb}")
        self.logger.info(f"  Heuristic Guidance: Enabled={config.use_heuristic_guidance}, "
                        f"Alpha={config.heuristic_alpha}, Epsilon={config.epsilon_greedy}")
        self.logger.info(f"  Phase Budget Multipliers: {config.phase_budget_multipliers}")
        self.logger.info(f"  Heuristic Weight: {config.heuristic_weight}")
    
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
    
    def _create_child_node(self, state: GameState, parent, prior_prob: float, 
                          game_phase: GamePhase = None) -> EnhancedNode:
        """Create a child node with proper memory management."""
        if self._pool_enabled:
            child_state = self._acquire_state_copy(state)
            child_node = EnhancedNode(
                child_state,
                parent=parent,
                prior_prob=prior_prob,
                c_puct=self.config.c_puct,
                state_pool=self.game_state_pool,
                game_phase=game_phase
            )
            child_node._owns_state = True
            return child_node
        else:
            return EnhancedNode(
                state.copy(),
                parent=parent,
                prior_prob=prior_prob,
                c_puct=self.config.c_puct,
                game_phase=game_phase
            )
    
    def _get_phase_aware_budget(self, move_number: int) -> int:
        """Get simulation budget based on game phase."""
        base_budget = (self.config.num_simulations if move_number < self.config.simulation_switch_ply 
                      else (self.config.late_simulations or self.config.num_simulations))
        
        if not self.config.use_phase_aware_budget:
            return base_budget
        
        # Determine current phase
        phase_config = PhaseConfig()
        current_phase = phase_config.get_phase(move_number)
        
        # Apply phase-specific multiplier
        multiplier = self.config.phase_budget_multipliers.get(current_phase, 1.0)
        return int(base_budget * multiplier)
    
    def _select_action(self, node: EnhancedNode) -> int:
        """Select action using enhanced UCB with heuristic guidance and phase awareness."""
        if not node.children:
            return None
        
        parent_visit_count = node.visit_count
        best_action = None
        best_score = float('-inf')
        
        for action, child in node.children.items():
            score = child.get_enhanced_ucb_score(parent_visit_count, self.config, self.heuristic_evaluator)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _evaluate_state(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Evaluate state using neural network."""
        state_tensor = self.state_encoder.encode_state(state)
        policy, value = self.network.predict(state_tensor)
        return policy, value
    
    def _get_value(self, state: GameState) -> Optional[float]:
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
    
    def _heuristic_guided_rollout(self, state: GameState, max_depth: int = 10) -> float:
        """Perform epsilon-greedy rollout biased by heuristics (Task 10)."""
        if self.heuristic_evaluator is None:
            return self._random_rollout(state, max_depth)
        
        current_state = state.copy()
        depth = 0
        
        while not current_state.is_terminal() and depth < max_depth:
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            # Epsilon-greedy selection: epsilon chance of random, (1-epsilon) chance of heuristic-guided
            if np.random.random() < self.config.epsilon_greedy:
                # Random move selection
                move = np.random.choice(valid_moves)
            else:
                # Heuristic-guided move selection
                move = self._select_heuristic_move(current_state, valid_moves)
            
            current_state.make_move(move)
            depth += 1
        
        # Evaluate final position with heuristic
        if current_state.is_terminal():
            winner = current_state.get_winner()
            if winner == Player.WHITE:
                return 1.0
            elif winner == Player.BLACK:
                return -1.0
            else:
                return 0.0
        else:
            # Use heuristic evaluation for non-terminal positions
            return self.heuristic_evaluator.evaluate_position_fast(
                current_state, current_state.current_player
            )
    
    def _select_heuristic_move(self, state: GameState, valid_moves: List[int]) -> int:
        """Select move based on heuristic evaluation of resulting positions."""
        if not valid_moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            # Evaluate position after this move
            test_state = state.copy()
            test_state.make_move(move)
            
            # Get heuristic evaluation (from opponent's perspective, so negate)
            heuristic_score = self.heuristic_evaluator.evaluate_position_fast(
                test_state, test_state.current_player
            )
            
            if heuristic_score > best_score:
                best_score = heuristic_score
                best_move = move
        
        return best_move or valid_moves[0]  # Fallback to first move
    
    def _random_rollout(self, state: GameState, max_depth: int = 10) -> float:
        """Perform random rollout as fallback."""
        current_state = state.copy()
        depth = 0
        
        while not current_state.is_terminal() and depth < max_depth:
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            move = np.random.choice(valid_moves)
            current_state.make_move(move)
            depth += 1
        
        # Return terminal value
        if current_state.is_terminal():
            winner = current_state.get_winner()
            if winner == Player.WHITE:
                return 1.0
            elif winner == Player.BLACK:
                return -1.0
            else:
                return 0.0
        else:
            return 0.0  # Draw for non-terminal
    
    def _backpropagate(self, search_path: List[EnhancedNode], value: float):
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
        Perform enhanced MCTS search with analysis integration.
        
        Args:
            state: Current game state
            move_number: Current move number for phase determination
            
        Returns:
            Policy vector over all possible moves
        """
        # Get phase-aware simulation budget
        budget = self._get_phase_aware_budget(move_number)
        
        # Determine current phase
        phase_config = PhaseConfig()
        current_phase = phase_config.get_phase(move_number)
        
        root = EnhancedNode(state, c_puct=self.config.c_puct, game_phase=current_phase)
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
                if action not in node.children:
                    self.logger.error(f"Selected action {action} not in node children")
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
            value = self._get_value(current_state)
            
            if value is None and depth < self.config.max_depth:
                policy, net_value = self._evaluate_state(current_state)
                
                # Use enhanced evaluation if heuristic evaluator is available
                if self.heuristic_evaluator is not None:
                    value = node.get_combined_evaluation(
                        self.heuristic_evaluator, net_value, self.config
                    )
                else:
                    value = net_value
                
                # Check if state became terminal
                if current_state.is_terminal():
                    value = self._get_value(current_state)
                else:
                    # Expand node
                    valid_moves = current_state.get_valid_moves()
                    if valid_moves:
                        node.is_expanded = True
                        policy = self._mask_invalid_moves(policy, valid_moves)
                        
                        for move in valid_moves:
                            node.children[move] = self._create_child_node(
                                current_state,
                                parent=node,
                                prior_prob=self._get_move_prob(policy, move),
                                game_phase=current_phase
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
                # Use heuristic-guided rollout for leaf nodes (Task 10)
                if self.config.use_heuristic_rollouts:
                    value = self._heuristic_guided_rollout(current_state, max_depth=5)
                else:
                    # Fallback to neural network evaluation
                    _, value = self._evaluate_state(current_state)
            
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
        
        return move_probs
