# training/self_play.py

import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor
import time
import tempfile
import os
from pathlib import Path
import concurrent.futures
import psutil
import platform
import random
import copy

# --- YINSH ML Imports ---
# Assuming these paths are correct relative to this file
from ..utils.mcts_metrics import MCTSMetrics
from ..utils.TemperatureMetrics import TemperatureMetrics # Keep if used
from ..utils.metrics_logger import MetricsLogger, GameMetrics # Keep if used
from ..utils.encoding import StateEncoder
from ..game.game_state import GameState, GamePhase # Added GamePhase
from ..game.constants import Player, PieceType # Added PieceType
from ..network.wrapper import NetworkWrapper
from ..game.moves import Move
# --- Memory Pool Imports ---
from ..memory import GameStatePool, TensorPool, GameStatePoolConfig, TensorPoolConfig
# --- Heuristic Evaluation Import ---
from ..heuristics.evaluator import YinshHeuristics

# Configure the logger at the module level
# Use getLogger to ensure consistency if already configured elsewhere
logger = logging.getLogger("SelfPlay")
# Set default level if not configured by runner/supervisor
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     logger.addHandler(logging.StreamHandler()) # Add console handler if none exist
     logger.setLevel(logging.INFO)


class Node:
    """Monte Carlo Tree Search node with virtual loss support for batched evaluation."""
    def __init__(self, state: GameState, parent=None, prior_prob=0.0, c_puct=1.0, state_pool=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False
        self.c_puct = c_puct # Store c_puct as instance variable
        self._state_pool = state_pool  # Reference to state pool for cleanup
        self._owns_state = False  # Track if this node owns the state for cleanup

        # Virtual loss mechanism for batched MCTS
        self.virtual_losses = 0  # Track in-flight evaluations

    def __del__(self):
        """Clean up GameState when node is destroyed."""
        if self._owns_state and self._state_pool is not None and self.state is not None:
            try:
                self._state_pool.return_game_state(self.state)
            except Exception as e:
                # Use basic logging since logger might not be available
                import logging
                logging.getLogger("Node").warning(f"Failed to release state in Node destructor: {e}")

    def value(self) -> float:
        """Get mean value of node, accounting for virtual losses."""
        adjusted_visits = self.visit_count + self.virtual_losses
        if adjusted_visits == 0:
            return 0.0
        return self.value_sum / adjusted_visits

    def clear_tree(self) -> None:
        """Recursively break circular references to allow garbage collection.

        MCTS trees have parent↔child circular references that can prevent
        Python's garbage collector from freeing memory promptly. Call this
        on the root node after search completes.
        """
        # First, recursively clear all children
        for child in self.children.values():
            child.clear_tree()

        # Return state to pool if this node owns it
        if self._owns_state and self._state_pool is not None and self.state is not None:
            try:
                self._state_pool.return_game_state(self.state)
            except:
                pass
            self._owns_state = False  # Prevent double-return in __del__

        # Break circular references
        self.children.clear()
        self.parent = None
        self.state = None
        self._state_pool = None  # Clear pool reference too

    def add_virtual_loss(self) -> None:
        """Mark this node as being evaluated (used in batched MCTS)."""
        self.virtual_losses += 1

    def remove_virtual_loss(self) -> None:
        """Remove virtual loss after evaluation completes."""
        self.virtual_losses -= 1
        if self.virtual_losses < 0:
            self.virtual_losses = 0  # Safety check

    def get_ucb_score(self, parent_visit_count: int) -> float:
        """Calculate Upper Confidence Bound score, accounting for virtual losses."""
        q_value = self.value()
        # Add small epsilon to prevent division by zero, account for virtual losses
        adjusted_visits = self.visit_count + self.virtual_losses
        u_value = (self.c_puct * self.prior_prob *
                  np.sqrt(parent_visit_count) / (1 + adjusted_visits))
        return q_value + u_value

class MCTS:
    """Monte-Carlo Tree-Search with temperature annealing and ply-adaptive rollouts."""

    def __init__(self,
                 network: NetworkWrapper,
                 # --- Evaluation Mode ---
                 evaluation_mode: str = "pure_neural",  # Options: "pure_neural", "pure_heuristic", "hybrid"
                 heuristic_evaluator: Optional[YinshHeuristics] = None,
                 heuristic_weight: float = 0.5,  # Weight for heuristic in hybrid mode (0.0 = pure neural, 1.0 = pure heuristic)
                 # --- MCTS Search Params ---
                 num_simulations: int = 100, # Base simulations (early game)
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 # *** Rename 'initial_value_weight' to 'value_weight' for clarity ***
                 value_weight: float = 1.0, # Weighting for value in UCB selection
                 max_depth: int = 500,
                 # --- Rollout Scheduling Params ---
                 late_simulations: Optional[int] = None, # Use Optional[int] for type hint clarity
                 simulation_switch_ply: int = 20,
                 # --- Temperature Params (passed down) ---
                 initial_temp: float = 1.0,
                 final_temp: float = 0.1,
                 annealing_steps: int = 30,
                 temp_clamp_fraction: float = 0.6,
                 # --- Optional Metrics ---
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 # --- Memory Pools ---
                 game_state_pool: Optional[GameStatePool] = None,
                ):
        """
        Initialize MCTS.

        Args:
            network: The neural network wrapper.
            evaluation_mode: Evaluation mode - "pure_neural", "pure_heuristic", or "hybrid".
            heuristic_evaluator: YinshHeuristics instance for heuristic evaluation (required for pure_heuristic/hybrid modes).
            heuristic_weight: Weight for heuristic in hybrid mode (0.0-1.0, where 0.0=pure neural, 1.0=pure heuristic).
            num_simulations: Base simulation budget (used before switch_ply).
            c_puct: Exploration constant for UCB.
            dirichlet_alpha: Alpha parameter for Dirichlet noise at the root.
            value_weight: Factor to scale the node's value in the UCB calculation.
            max_depth: Maximum depth for the search path during simulation.
            late_simulations: Simulation budget *after* simulation_switch_ply. If None, defaults to num_simulations.
            simulation_switch_ply: Ply number to switch simulation budget.
            initial_temp: Starting temperature for action selection sampling.
            final_temp: Final temperature value.
            annealing_steps: Number of moves over which temperature anneals.
            temp_clamp_fraction: Fraction of annealing_steps before clamping temperature.
            mcts_metrics: Optional MCTSMetrics collector instance.
            game_state_pool: Optional GameStatePool for memory management.
        """
        self.network = network
        # Use network's encoder to ensure consistent encoding (basic or enhanced)
        self.state_encoder = network.state_encoder
        self.logger = logging.getLogger("MCTS") # Use module-level logger

        # Evaluation Mode Configuration
        self.evaluation_mode = evaluation_mode.lower()
        self.heuristic_evaluator = heuristic_evaluator
        self.heuristic_weight = np.clip(heuristic_weight, 0.0, 1.0)  # Ensure weight is in [0, 1]

        # Validate evaluation mode configuration
        if self.evaluation_mode not in ["pure_neural", "pure_heuristic", "hybrid"]:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}. Must be 'pure_neural', 'pure_heuristic', or 'hybrid'.")
        if self.evaluation_mode in ["pure_heuristic", "hybrid"] and heuristic_evaluator is None:
            raise ValueError(f"heuristic_evaluator required for evaluation_mode='{evaluation_mode}'")

        # Memory Pool Management
        self.game_state_pool = game_state_pool
        self._pool_enabled = game_state_pool is not None

        # MCTS Core Parameters
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        # *** Store using the new name 'value_weight' ***
        self.value_weight = value_weight
        self.max_depth = max_depth # Store max_depth

        # Rollout Budget Scheduling
        self.early_simulations = num_simulations
        # Default late_simulations to early_simulations if None is passed
        self.late_simulations = late_simulations if late_simulations is not None else num_simulations
        self.switch_ply = simulation_switch_ply

        # Temperature Schedule Parameters (stored for get_temperature)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_steps = annealing_steps
        self.temp_clamp_frac = temp_clamp_fraction

        # Metrics Tracking
        self.metrics = mcts_metrics if mcts_metrics is not None else MCTSMetrics()
        self.current_iteration = 0 # Can be updated externally if needed

        self.logger.info(f"MCTS Initialized:")
        self.logger.info(f"  Evaluation Mode: {self.evaluation_mode} (heuristic_weight={self.heuristic_weight:.3f})")
        self.logger.info(f"  Memory: Pool enabled={self._pool_enabled}")
        self.logger.info(f"  Sims: Early={self.early_simulations}, Late={self.late_simulations} (Switch Ply {self.switch_ply})")
        self.logger.info(f"  Search: cPUCT={self.c_puct:.2f}, Alpha={self.dirichlet_alpha:.2f}, Value Weight={self.value_weight:.2f}, Max Depth={self.max_depth}")
        self.logger.info(f"  Temperature: Initial={self.initial_temp:.2f}, Final={self.final_temp:.2f}, Steps={self.annealing_steps}, Clamp Frac={self.temp_clamp_frac:.2f}")

        self.debug_config()

    def debug_config(self):
        """Log all configuration parameters to verify they're being set correctly."""
        self.logger.info("====== MCTS Configuration Debug ======")
        self.logger.info(f"Early Simulations: {self.early_simulations}")
        self.logger.info(f"Late Simulations: {self.late_simulations}")
        self.logger.info(f"Switch Ply: {self.switch_ply}")
        self.logger.info(f"c_puct: {self.c_puct}")
        self.logger.info(f"Dirichlet Alpha: {self.dirichlet_alpha}")
        self.logger.info(f"Value Weight: {self.value_weight}")
        self.logger.info(f"Max Depth: {self.max_depth}")
        self.logger.info("======================================")

    def _acquire_state_copy(self, original_state: GameState) -> GameState:
        """Get a GameState copy from pool or create new one."""
        if self._pool_enabled:
            try:
                pooled_state = self.game_state_pool.get()
                # Copy state data from original to pooled state
                # This is more efficient than creating a brand new state
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

    def _create_child_node(self, state: GameState, parent, prior_prob: float) -> 'Node':
        """Create a child node with proper memory management."""
        if self._pool_enabled:
            # Get a state from the pool for the child node
            child_state = self._acquire_state_copy(state)
            child_node = Node(
                child_state, 
                parent=parent, 
                prior_prob=prior_prob, 
                c_puct=self.c_puct, 
                state_pool=self.game_state_pool
            )
            child_node._owns_state = True  # Mark that this node owns the state
            return child_node
        else:
            # Fall back to regular copy
            return Node(
                state.copy(), 
                parent=parent, 
                prior_prob=prior_prob, 
                c_puct=self.c_puct
            )

    def get_temperature(self, move_number: int) -> float:
        """
        Calculate temperature based on linear annealing schedule with clamping.
        """
        # Ensure annealing_steps is positive to avoid division by zero
        if self.annealing_steps <= 0:
            return self.final_temp

        clamp_moves = int(self.annealing_steps * self.temp_clamp_frac)

        # Ensure clamp_moves is at least 1 if annealing_steps > 0
        clamp_moves = max(1, clamp_moves)

        if move_number >= clamp_moves:
            temperature = self.final_temp
        else:
            progress = move_number / clamp_moves # Progress from 0 to 1
            temperature = self.initial_temp - (self.initial_temp - self.final_temp) * progress

        # Add debug logging for temperature calculation if needed
        # self.logger.debug(f"Move: {move_number}, Anneal Steps: {self.annealing_steps}, Clamp Frac: {self.temp_clamp_frac}, Clamp Moves: {clamp_moves}, Temp: {temperature:.3f}")
        return temperature

    def search(self, state: GameState, move_number: int) -> np.ndarray:
        """
        Run MCTS simulations for the given state and move number.
        """
        # Choose rollout budget based on the current move number
        budget = self.early_simulations if move_number < self.switch_ply else self.late_simulations
        # self.logger.debug(f"Move {move_number}, Using budget: {budget} (Early: {self.early_simulations}, Late: {self.late_simulations}, Switch: {self.switch_ply})")

        root = Node(state, c_puct=self.c_puct) # Pass c_puct to root node
        # Initialize policy vector
        move_probs = np.zeros(self.state_encoder.total_moves, dtype=np.float32)

        # --- Run Simulations ---
        simulation_states = []  # Track states acquired from pool for cleanup
        
        for sim in range(budget):
            node = root
            search_path = [node]
            current_state = self._acquire_state_copy(state)
            simulation_states.append(current_state)  # Track for cleanup
            depth = 0
            action = None

            # 1. Selection ...
            while node.is_expanded and node.children:
                action = self._select_action(node)
                if action not in node.children:
                     self.logger.error(f"MCTS Error: Selected action {action} not in node children {list(node.children.keys())} during Selection phase. State: {current_state}")
                     break
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                depth += 1
                if depth >= self.max_depth:
                     break

            # Check if simulation was broken early (now action will always be defined)
            # *** Modify the check slightly: check if action is None OR not in children ***
            if action is None or (action not in node.children and node.is_expanded and node.children):
                 # If action is None, the loop didn't even run once (root had no children?)
                 # If action not in children, the error log above triggered.
                 self.logger.debug(f"Skipping simulation {sim+1} due to selection error or early break.")
                 continue # Skip to next simulation

            # 2. Expansion & Evaluation: If a leaf node is reached
            value = self._get_value(current_state) # Check if terminal first

            if value is None and depth < self.max_depth:
                policy, net_value = self._evaluate_state(current_state) # Get NN prediction
                value = net_value # Use network's value if not terminal

                # *** Add check: Ensure state didn't become terminal between _get_value and _evaluate_state ***
                if current_state.is_terminal():
                     self.logger.debug(f"State became terminal unexpectedly before expansion at depth {depth}. Using terminal value.")
                     value = self._get_value(current_state) # Re-get terminal value
                else:
                    # Proceed with expansion only if truly non-terminal
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

                    # Apply Dirichlet noise at the root node ONCE per search call
                    # Use a flag on the root node to track if noise was applied
                    if node is root and not hasattr(root, "dirichlet_applied"):
                        if self.dirichlet_alpha > 0 and len(valid_moves) > 0:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
                            epsilon_mix = 0.25 # Standard mixing fraction
                            for i, move in enumerate(valid_moves):
                                child_node = root.children[move]
                                child_node.prior_prob = (1 - epsilon_mix) * child_node.prior_prob + epsilon_mix * noise[i]
                            root.dirichlet_applied = True
                            # self.logger.debug(f"Applied Dirichlet noise (alpha={self.dirichlet_alpha}) at root.")

                # else: # No valid moves means state is terminal (should have been caught by _get_value)
                #     self.logger.warning(f"MCTS Warning: Reached non-terminal state with no valid moves during expansion? State:\n{current_state}")
                #     value = self._get_value(current_state) or 0.0 # Assign terminal value if possible

            # elif value is None and depth >= self.max_depth:
                 # Reached max depth, use network evaluation for value, but don't expand
                 # _, value = self._evaluate_state(current_state)
                 # self.logger.debug(f"MCTS Search hit max depth ({self.max_depth}) during expansion. Using value {value:.3f}.")


            # 3. Backpropagation: Update values and visit counts along the path
            if value is None: # Should not happen if logic above is correct, but safeguard
                 self.logger.error(f"MCTS Error: Value is None before backpropagation. State terminal: {current_state.is_terminal()}")
                 value = 0.0 # Default value if something went wrong
            self._backpropagate(search_path, value)

        # Clean up simulation states
        for state in simulation_states:
            self._release_state(state)

        # --- After all simulations ---
        # Calculate final move probabilities based on visit counts
        temp = self.get_temperature(move_number)
        # self.logger.debug(f"Move {move_number}: Calculating policy with Temp={temp:.3f}")

        # Ensure root has children before proceeding
        if not root.children:
            self.logger.debug(f"MCTS Warning: Root node has no children after {budget} simulations. Returning uniform policy over valid moves (if any). State:\n{state}")
            valid_moves = state.get_valid_moves()
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for move in valid_moves:
                    move_idx = self.state_encoder.move_to_index(move)
                    if 0 <= move_idx < len(move_probs):
                         move_probs[move_idx] = prob
                    else:
                         self.logger.error(f"Invalid move index {move_idx} for move {move} generated by state.get_valid_moves(). Max index: {len(move_probs)-1}")
            return move_probs # Return uniform or zero vector


        valid_moves_at_root = list(root.children.keys())
        visit_counts = np.array([
            root.children[move].visit_count for move in valid_moves_at_root
        ], dtype=np.float32)

        if visit_counts.sum() == 0:
             self.logger.warning(f"MCTS Warning: Root node children have zero total visits after {budget} simulations. Returning uniform policy.")
             prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
             visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)
        else:
            # Apply temperature
            if temp == 0:
                # Greedy selection: highest visit count gets probability 1
                visit_probs = np.zeros_like(visit_counts)
                visit_probs[np.argmax(visit_counts)] = 1.0
            else:
                # Apply temperature scaling
                visit_counts_temp = np.power(visit_counts, 1.0 / temp)
                # Normalize to get probabilities
                total_visits_temp = visit_counts_temp.sum()
                if total_visits_temp > 1e-6 : # Avoid division by zero
                     visit_probs = visit_counts_temp / total_visits_temp
                else: # Handle case where all temp-scaled visits are near zero
                     self.logger.warning("MCTS Warning: Sum of temperature-scaled visit counts is near zero. Falling back to uniform.")
                     prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
                     visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)


        # Store probabilities in the full policy vector
        for move, prob in zip(valid_moves_at_root, visit_probs):
            move_idx = self.state_encoder.move_to_index(move)
            if 0 <= move_idx < len(move_probs):
                 move_probs[move_idx] = prob
            else:
                 self.logger.error(f"Invalid move index {move_idx} for move {move} from root children. Max index: {len(move_probs)-1}")

        # MEMORY FIX: Break circular references in MCTS tree to allow garbage collection
        root.clear_tree()

        return move_probs

    def search_batch(self, state: GameState, move_number: int, batch_size: int = 32) -> np.ndarray:
        """
        Run MCTS simulations with batched leaf node evaluation for improved throughput.

        Instead of evaluating each leaf node individually, this method collects multiple
        leaf nodes and evaluates them all at once using the network's predict_batch method.
        This provides 10-20x speedup on M2 by utilizing CPU/Neural Engine more efficiently.

        Args:
            state: The current game state
            move_number: Current move number (for temperature scheduling)
            batch_size: Number of leaf nodes to collect before batching evaluation (default: 32)

        Returns:
            Policy vector of move probabilities
        """
        # Choose rollout budget based on the current move number
        budget = self.early_simulations if move_number < self.switch_ply else self.late_simulations

        root = Node(state, c_puct=self.c_puct)
        move_probs = np.zeros(self.state_encoder.total_moves, dtype=np.float32)

        # Track states acquired from pool for cleanup
        simulation_states = []

        # Collect leaves for batched evaluation
        batch_leaves = []  # List of (node, search_path, current_state, depth)

        for sim in range(budget):
            node = root
            search_path = [node]
            current_state = self._acquire_state_copy(state)
            simulation_states.append(current_state)
            depth = 0
            action = None

            # 1. Selection phase - traverse tree until we reach a leaf
            while node.is_expanded and node.children:
                action = self._select_action(node)
                if action not in node.children:
                    self.logger.error(f"MCTS Error: Selected action {action} not in node children during batched selection.")
                    break
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                depth += 1
                if depth >= self.max_depth:
                    break

            # Skip if selection had errors
            if action is None or (action not in node.children and node.is_expanded and node.children):
                continue

            # 2. Check if this is a terminal state
            terminal_value = self._get_value(current_state)
            if terminal_value is not None:
                # Terminal state - backpropagate immediately
                self._backpropagate(search_path, terminal_value)
                continue

            # 3. Check if we hit max depth
            if depth >= self.max_depth:
                # At max depth - need to evaluate but don't expand
                # Add to batch for evaluation
                node.add_virtual_loss()  # Mark as in-flight
                batch_leaves.append((node, search_path, current_state, depth, False))  # False = don't expand
            else:
                # Normal leaf node - add to batch for expansion and evaluation
                node.add_virtual_loss()  # Mark as in-flight
                batch_leaves.append((node, search_path, current_state, depth, True))  # True = can expand

            # 4. Process batch when it's full or we're at the end
            if len(batch_leaves) >= batch_size or sim == budget - 1:
                if batch_leaves:
                    self._evaluate_and_backup_batch(batch_leaves, root)
                    batch_leaves = []

        # Process any remaining leaves
        if batch_leaves:
            self._evaluate_and_backup_batch(batch_leaves, root)

        # Clean up simulation states
        for s in simulation_states:
            self._release_state(s)

        # Calculate final move probabilities (same as serial version)
        temp = self.get_temperature(move_number)

        if not root.children:
            self.logger.debug(f"MCTS Warning: Root node has no children after {budget} simulations.")
            valid_moves = state.get_valid_moves()
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for move in valid_moves:
                    move_idx = self.state_encoder.move_to_index(move)
                    if 0 <= move_idx < len(move_probs):
                        move_probs[move_idx] = prob
            return move_probs

        valid_moves_at_root = list(root.children.keys())
        visit_counts = np.array([
            root.children[move].visit_count for move in valid_moves_at_root
        ], dtype=np.float32)

        if visit_counts.sum() == 0:
            prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
            visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)
        else:
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

        for move, prob in zip(valid_moves_at_root, visit_probs):
            move_idx = self.state_encoder.move_to_index(move)
            if 0 <= move_idx < len(move_probs):
                move_probs[move_idx] = prob

        # Store root value for use as training target (Fix #1)
        self.last_root_value = root.value() if root.visit_count > 0 else 0.0

        # MEMORY FIX: Break circular references in MCTS tree to allow garbage collection
        root.clear_tree()

        return move_probs

    def _evaluate_and_backup_batch(self, batch_leaves: List[Tuple], root: Node):
        """
        Evaluate a batch of leaf nodes and back up the results.

        Args:
            batch_leaves: List of tuples (node, search_path, current_state, depth, can_expand)
            root: Root node (for Dirichlet noise application)
        """
        if not batch_leaves:
            return

        # Separate states for batch evaluation
        states_to_evaluate = [item[2] for item in batch_leaves]

        # Batch evaluate all states
        if self.evaluation_mode in ["pure_neural", "hybrid"]:
            # Use neural network for batch evaluation
            policy_logits_batch, values_batch = self.network.predict_batch(states_to_evaluate)

            # Convert to numpy
            policy_logits_batch = policy_logits_batch.cpu().numpy()
            values_batch = values_batch.cpu().numpy().flatten()
        else:
            # Pure heuristic mode - evaluate individually (heuristics don't support batching)
            policy_logits_batch = None
            values_batch = np.array([
                self.heuristic_evaluator.evaluate_position(s, s.current_player)
                for s in states_to_evaluate
            ])

        # Process each leaf in the batch
        for i, (node, search_path, current_state, depth, can_expand) in enumerate(batch_leaves):
            # Get policy and value for this state
            if self.evaluation_mode in ["pure_neural", "hybrid"]:
                policy_logits = policy_logits_batch[i]
                policy_nn = torch.softmax(torch.from_numpy(policy_logits), dim=0).numpy()
                value_nn = values_batch[i]
            else:
                policy_nn = None
                value_nn = 0.0

            # Combine with heuristics if in hybrid mode
            if self.evaluation_mode == "hybrid":
                value_heuristic = self.heuristic_evaluator.evaluate_position(
                    current_state, current_state.current_player
                )
                valid_moves = current_state.get_valid_moves()
                policy_heuristic = np.zeros_like(policy_nn)
                if len(valid_moves) > 0:
                    for move in valid_moves:
                        idx = self.state_encoder.move_to_index(move)
                        if 0 <= idx < len(policy_heuristic):
                            policy_heuristic[idx] = 1.0 / len(valid_moves)

                w_h = self.heuristic_weight
                w_n = 1.0 - w_h
                policy = w_n * policy_nn + w_h * policy_heuristic
                value = w_n * value_nn + w_h * value_heuristic
            elif self.evaluation_mode == "pure_heuristic":
                valid_moves = current_state.get_valid_moves()
                policy = np.zeros(self.state_encoder.total_moves)
                if len(valid_moves) > 0:
                    for move in valid_moves:
                        idx = self.state_encoder.move_to_index(move)
                        if 0 <= idx < len(policy):
                            policy[idx] = 1.0 / len(valid_moves)
                value = values_batch[i]
            else:  # pure_neural
                policy = policy_nn
                value = value_nn

            # Expand node if allowed
            if can_expand and not current_state.is_terminal():
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

                    # Apply Dirichlet noise at root node (once per search call)
                    if node is root and not hasattr(root, "dirichlet_applied"):
                        if self.dirichlet_alpha > 0 and len(valid_moves) > 0:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
                            epsilon_mix = 0.25
                            for j, move in enumerate(valid_moves):
                                child_node = root.children[move]
                                child_node.prior_prob = (
                                    (1 - epsilon_mix) * child_node.prior_prob + epsilon_mix * noise[j]
                                )
                            root.dirichlet_applied = True

            # Remove virtual loss and backpropagate
            node.remove_virtual_loss()
            self._backpropagate(search_path, value)

    def _select_action(self, node: Node) -> Move:
        """Select action using UCB formula with configured value_weight."""
        valid_moves = list(node.children.keys())
        parent_visit_count = node.visit_count # Total visits to the parent node

        # Handle case where node might not have been visited yet (shouldn't happen if called after expansion)
        if parent_visit_count == 0:
             self.logger.warning("MCTS _select_action called on node with zero visits. Selecting randomly.")
             return random.choice(valid_moves) if valid_moves else None


        best_score = -float('inf')
        best_move = None

        # Add small random noise to break ties consistently
        epsilon = 1e-8

        for move in valid_moves:
            child = node.children[move]

            # Q-value (Exploitation term), scaled by value_weight
            # If child hasn't been visited, its value is 0.
            q_value = child.value()
            scaled_q = self.value_weight * q_value # Apply the weighting factor

            # U-value (Exploration term)
            # Use child's c_puct and prior_prob
            if child.visit_count == 0:
                u_value = child.c_puct * child.prior_prob * np.sqrt(parent_visit_count + epsilon) # Add epsilon for sqrt(0) case
            else:
                u_value = child.c_puct * child.prior_prob * np.sqrt(parent_visit_count) / (1 + child.visit_count)

            score = scaled_q + u_value + np.random.uniform(0, epsilon)

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None and valid_moves: # Fallback if something went wrong
            self.logger.warning("MCTS _select_action failed to find best move. Selecting randomly.")
            return random.choice(valid_moves)

        return best_move

    def _evaluate_state(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Get policy and value from neural network and/or heuristic evaluator.

        Supports three evaluation modes:
        - pure_neural: Only use neural network
        - pure_heuristic: Only use heuristic evaluator
        - hybrid: Weighted combination of neural network and heuristic
        """
        # Get neural network evaluation
        if self.evaluation_mode in ["pure_neural", "hybrid"]:
            with torch.no_grad():
                policy_tensor, value_tensor = self.network.predict_from_state(state)
                # Convert to numpy and extract values
                policy_nn = torch.softmax(policy_tensor, dim=1).squeeze(0).cpu().numpy()
                value_nn = value_tensor.item()  # Get scalar value
        else:
            policy_nn = None
            value_nn = 0.0

        # Get heuristic evaluation
        if self.evaluation_mode in ["pure_heuristic", "hybrid"]:
            # Heuristic evaluator returns differential score for current player
            current_player = state.current_player
            value_heuristic = self.heuristic_evaluator.evaluate_position(state, current_player)

            # For policy, use uniform distribution over legal moves (heuristics don't provide policy guidance)
            valid_moves = state.get_valid_moves()
            policy_heuristic = np.zeros_like(policy_nn) if policy_nn is not None else np.zeros(self.state_encoder.total_moves)
            if len(valid_moves) > 0:
                for move in valid_moves:
                    idx = self.state_encoder.move_to_index(move)
                    if 0 <= idx < len(policy_heuristic):
                        policy_heuristic[idx] = 1.0 / len(valid_moves)
        else:
            policy_heuristic = None
            value_heuristic = 0.0

        # Combine based on evaluation mode
        if self.evaluation_mode == "pure_neural":
            return policy_nn, value_nn
        elif self.evaluation_mode == "pure_heuristic":
            return policy_heuristic, value_heuristic
        else:  # hybrid mode
            # Weighted combination
            w_h = self.heuristic_weight
            w_n = 1.0 - w_h

            # Combine policy (weighted sum)
            policy_combined = w_n * policy_nn + w_h * policy_heuristic

            # Combine value (weighted sum)
            value_combined = w_n * value_nn + w_h * value_heuristic

            return policy_combined, value_combined

    def _get_value(self, state: GameState) -> Optional[float]:
        """Get terminal value if game ended, None otherwise. Uses normalized margin."""
        if not state.is_terminal():
            return None

        # Calculate normalized margin based on final scores
        score_diff = state.white_score - state.black_score
        normalized_margin = score_diff / 3.0 # Max score difference is 3 (3-0 or 0-3)
        # Clamp the value to be strictly within [-1, 1]
        final_value = np.clip(normalized_margin, -1.0, 1.0)

        # Perspective matters: MCTS backpropagates the negative of the child's value.
        # The value returned here should be from the perspective of the *player whose turn it would be*
        # if the game hadn't ended. However, AlphaZero often uses the value from the perspective
        # of the player who *made the last move* leading to this terminal state. Let's clarify.
        # If the game ends, the outcome is fixed. Let's return the objective outcome.
        # The backpropagation step (value = -value) handles the perspective switch.
        return final_value


    def _mask_invalid_moves(self, policy: np.ndarray, valid_moves: List[Move]) -> np.ndarray:
        """Mask out invalid moves from the policy vector and re-normalize."""
        mask = np.zeros_like(policy, dtype=bool)
        valid_indices = []
        for move in valid_moves:
             idx = self.state_encoder.move_to_index(move)
             if 0 <= idx < len(policy):
                  mask[idx] = True
                  valid_indices.append(idx)
             else:
                  self.logger.error(f"Invalid move index {idx} encountered in _mask_invalid_moves for move {move}. Max index: {len(policy)-1}")

        # Apply mask
        masked_policy = np.where(mask, policy, 0.0)

        # Re-normalize
        policy_sum = masked_policy.sum()
        if policy_sum > 1e-6: # Use a small epsilon for floating point comparison
            normalized_policy = masked_policy / policy_sum
        else:
            # If the sum is zero (e.g., network assigned zero probability to all valid moves),
            # assign uniform probability to valid moves.
            self.logger.warning("MCTS: All valid moves have near-zero probability in policy. Using uniform distribution.")
            num_valid = len(valid_indices)
            if num_valid > 0:
                uniform_prob = 1.0 / num_valid
                normalized_policy = np.zeros_like(policy)
                for idx in valid_indices:
                     normalized_policy[idx] = uniform_prob
            else: # Should not happen if valid_moves is not empty
                 normalized_policy = masked_policy # Return the zero vector

        return normalized_policy


    def _backpropagate(self, path: List[Node], value: float):
        """Backpropagate the evaluated value up the search path."""
        # Value is from the perspective of the player whose turn is *at the end* of the path.
        # As we go up, we flip the sign for the parent node.
        for node in reversed(path):
            node.visit_count += 1
            # Ensure the value perspective is correct for the node's state
            # The value should be added from the perspective of the player *whose turn it is* at that node.
            # Since 'value' comes from the child state, it's from the opponent's perspective relative to the current node.
            # So, we add -value to the node's value_sum.
            node.value_sum += -value
            value = -value # Flip perspective for the next parent node


    def _get_move_prob(self, policy: np.ndarray, move: Move) -> float:
        """Get move probability from the (normalized) policy array."""
        move_idx = self.state_encoder.move_to_index(move)
        if 0 <= move_idx < len(policy):
            return policy[move_idx]
        else:
            self.logger.error(f"Invalid move index {move_idx} requested in _get_move_prob for move {move}. Max index: {len(policy)-1}")
            return 0.0


class SelfPlay:
    """Handles self-play game generation using configured MCTS and temperature."""

    def __init__(self,
                 network: NetworkWrapper,
                 num_workers: int,
                 # --- Evaluation Mode ---
                 evaluation_mode: str = "pure_neural",
                 heuristic_weight: float = 0.5,
                 # --- MCTS Params (to be passed to MCTS instance) ---
                 num_simulations: int = 100, # Early game sims
                 late_simulations: Optional[int] = None,
                 simulation_switch_ply: int = 20,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 # *** Use consistent 'value_weight' naming ***
                 value_weight: float = 1.0,
                 max_depth: int = 500,
                 # --- Batched MCTS Params ---
                 use_batched_mcts: bool = True,  # Enable batched MCTS for 10-20x speedup
                 mcts_batch_size: int = 32,  # Number of leaves to collect before batching
                 # --- Temp Params (to be passed to MCTS instance) ---
                 initial_temp: float = 1.0,
                 final_temp: float = 0.1,
                 annealing_steps: int = 30,
                 temp_clamp_fraction: float = 0.6,
                 # --- Optional Metrics ---
                 metrics_logger: Optional[MetricsLogger] = None,
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 # --- Memory Pools ---
                 game_state_pool: Optional[GameStatePool] = None,
                ):
        """
        Initialize SelfPlay with explicit MCTS and temperature parameters.
        """
        self.network = network
        self.num_workers = num_workers # Use the calculated number passed in
        self.metrics_logger = metrics_logger
        # Use network's encoder to ensure consistent encoding (basic or enhanced)
        self.state_encoder = network.state_encoder
        self.logger = logging.getLogger("SelfPlay") # Use module logger
        self.current_iteration = 0 # Track current training iteration if needed

        # Store memory pools
        self.game_state_pool = game_state_pool

        # Initialize heuristic evaluator
        self.evaluation_mode = evaluation_mode
        self.heuristic_weight = heuristic_weight
        if evaluation_mode in ["pure_heuristic", "hybrid"]:
            self.heuristic_evaluator = YinshHeuristics()
            self.logger.info(f"Initialized YinshHeuristics evaluator for {evaluation_mode} mode")
        else:
            self.heuristic_evaluator = None

        # Store batched MCTS configuration
        self.use_batched_mcts = use_batched_mcts
        self.mcts_batch_size = mcts_batch_size

        # Store all parameters needed by MCTS and workers explicitly
        self.mcts_config = {
            'evaluation_mode': evaluation_mode,
            'heuristic_evaluator': self.heuristic_evaluator,
            'heuristic_weight': heuristic_weight,
            'num_simulations': num_simulations,
            'late_simulations': late_simulations,
            'simulation_switch_ply': simulation_switch_ply,
            'c_puct': c_puct,
            'dirichlet_alpha': dirichlet_alpha,
            'value_weight': value_weight, # Store with consistent name
            'max_depth': max_depth,
            'initial_temp': initial_temp,
            'final_temp': final_temp,
            'annealing_steps': annealing_steps,
            'temp_clamp_fraction': temp_clamp_fraction,
            'use_batched_mcts': use_batched_mcts,  # Add batched MCTS flag
            'mcts_batch_size': mcts_batch_size,  # Add batch size
            'use_enhanced_encoding': getattr(self.network, 'use_enhanced_encoding', False),  # Enhanced encoding flag
        }
        # Store optional metrics instances
        self.mcts_metrics = mcts_metrics # Store the instance

        # Instantiate the main MCTS object for the SelfPlay process itself (if needed, e.g., single-threaded play)
        # This MCTS instance is primarily used if play_game is called directly,
        # worker processes will create their own MCTS instances.

        # Filter out batched MCTS params and encoding flag that don't belong in MCTS.__init__()
        mcts_init_config = {k: v for k, v in self.mcts_config.items()
                           if k not in ['use_batched_mcts', 'mcts_batch_size', 'use_enhanced_encoding']}

        self.mcts = MCTS(
            network=self.network,
            mcts_metrics=self.mcts_metrics, # Pass metrics instance
            game_state_pool=self.game_state_pool, # Pass memory pool
            **mcts_init_config # Unpack the filtered config dict
        )

        self.logger.info("SelfPlay Initialized:")
        self.logger.info(f"  Workers: {self.num_workers}")
        self.logger.info(f"  Evaluation Mode: {self.evaluation_mode} (heuristic_weight={self.heuristic_weight:.3f})")
        self.logger.info(f"  Batched MCTS: {'ENABLED' if self.use_batched_mcts else 'DISABLED'} (batch_size={self.mcts_batch_size})")
        self.logger.info(f"  Enhanced Encoding: {'ENABLED (15 channels)' if self.mcts_config.get('use_enhanced_encoding', False) else 'DISABLED (6 channels)'}")
        self.logger.info(f"  MCTS Config: {self.mcts_config}")


    def generate_games(self, num_games: int) -> List[Tuple[List[np.ndarray], List[np.ndarray], List[float], List[Dict]]]:
        """Generate self-play games in parallel using multiple workers.

        Fix #1: Now returns MCTS root values (List[float]) instead of single game outcome (float)
        """
        self.logger.info(f"Starting generation of {num_games} games using {self.num_workers} workers...")
        # Fix #1: Changed from outcome (float) to values (List[float]) - MCTS root values per position
        games_data: List[Tuple[List[np.ndarray], List[np.ndarray], List[float], List[Dict]]] = []

        # Ensure the main network is on the CPU before saving state dict for workers
        self.network.network.cpu()
        network_params = self.network.network.state_dict()
        # Move network back to original device if needed
        self.network.network.to(self.network.device)


        # Create a temporary file to share the model state_dict
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                torch.save(network_params, tmp.name)
                model_path = tmp.name
                self.logger.debug(f"Saved temporary model state_dict for workers to: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to create temporary model file: {e}", exc_info=True)
            return [] # Return empty list if model cannot be shared


        start_time = time.time()
        games_completed = 0
        futures = []

        self.logger.info(f"MCTS config being used: sims={self.mcts.early_simulations}, "
                         f"late_sims={self.mcts.late_simulations}, "
                         f"switch_ply={self.mcts.switch_ply}, c_puct={self.mcts.c_puct}")

        # Handle sequential generation when workers=0 (memory-optimized mode)
        if self.num_workers <= 0:
            self.logger.info("Running sequential game generation (workers=0, memory-optimized mode)")
            for game_id in range(num_games):
                try:
                    result = play_game_worker(
                        model_path=model_path,
                        game_id=game_id,
                        mcts_config=self.mcts_config,
                    )
                    if result is not None:
                        states, policies, values, temp_data, game_history = result
                        games_data.append((states, policies, values, game_history))
                        games_completed += 1

                        # Log progress
                        if games_completed % max(1, num_games // 10) == 0 or games_completed == num_games:
                            elapsed = time.time() - start_time
                            rate = games_completed / elapsed if elapsed > 0 else 0
                            self.logger.info(f"Games Generated: {games_completed}/{num_games} ({rate:.2f} games/s)")
                    else:
                        self.logger.warning(f"Game {game_id} returned None result.")
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}", exc_info=True)

            # Cleanup temp model file
            if 'model_path' in locals() and os.path.exists(model_path):
                try:
                    os.unlink(model_path)
                except OSError:
                    pass

            total_time = time.time() - start_time
            final_rate = games_completed / total_time if total_time > 0 else 0
            self.logger.info(f"\nGame generation complete:")
            self.logger.info(f"- Games generated: {games_completed}/{num_games}")
            self.logger.info(f"- Total time: {total_time:.1f} seconds")
            self.logger.info(f"- Final rate: {final_rate:.2f} games/second")
            return games_data

        # Parallel generation with workers > 0
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for game_id in range(num_games):
                    # Submit worker jobs, passing the *explicit* MCTS configuration
                    futures.append(
                        executor.submit(
                            play_game_worker,
                            model_path=model_path,
                            game_id=game_id,
                            # --- Pass all necessary config parameters ---
                            mcts_config=self.mcts_config, # Pass the whole config dict
                            # mcts_metrics=self.mcts_metrics # Pass metrics collector if needed by worker
                            # --- Pass other necessary args ---
                        )
                    )

                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        # Check if worker returned valid data (not None or error)
                        if result is not None:
                             # Fix #1: Unpack values (MCTS root values) instead of outcome
                             states, policies, values, temp_data, game_history = result
                             games_data.append((states, policies, values, game_history))
                             games_completed += 1

                             # Log progress periodically
                             if games_completed % max(1, num_games // 10) == 0 or games_completed == num_games:
                                 elapsed = time.time() - start_time
                                 rate = games_completed / elapsed if elapsed > 0 else 0
                                 self.logger.info(f"Games Generated: {games_completed}/{num_games} ({rate:.2f} games/s)")
                                 # Log CPU usage (optional)
                                 # cpu_percent = psutil.cpu_percent(percpu=True)
                                 # avg_cpu = sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0
                                 # self.logger.debug(f"CPU Usage: {avg_cpu:.1f}%")

                             # --- Optional: Log detailed game metrics ---
                             # if self.metrics_logger:
                             #     game_metrics = GameMetrics(...) # Populate with data from result
                             #     self.metrics_logger.log_game(game_metrics)

                        else:
                             self.logger.warning(f"Worker returned None result for a game.")

                    except Exception as e:
                        # Log errors from worker processes
                        self.logger.error(f"Error processing game result from worker: {e}", exc_info=True)

        except Exception as e:
             self.logger.error(f"Error during parallel game generation: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Ensure the temporary model file is deleted
            if 'model_path' in locals() and os.path.exists(model_path):
                try:
                    os.unlink(model_path)
                    self.logger.debug(f"Deleted temporary model file: {model_path}")
                except OSError as e:
                    self.logger.error(f"Error deleting temporary model file {model_path}: {e}")

        total_time = time.time() - start_time
        final_rate = games_completed / total_time if total_time > 0 else 0
        self.logger.info(f"\nGame generation complete:")
        self.logger.info(f"- Games generated: {games_completed}/{num_games}")
        self.logger.info(f"- Total time: {total_time:.1f} seconds")
        if games_completed > 0:
            self.logger.info(f"- Average time per game: {total_time / games_completed:.2f} seconds")
        self.logger.info(f"- Final rate: {final_rate:.2f} games/second")

        # Return collected game data (states, policies, outcome, history)
        return games_data


    # Removed play_game method as it's superseded by play_game_worker
    # Removed export_games method (can be handled by supervisor or runner if needed)
    # Removed _collect_phase_values (can be done by supervisor/runner from game history)
    # Removed _get_optimal_workers (now done by supervisor)


def play_game_worker(
        model_path: str,
        game_id: int,
        mcts_config: Dict,
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], float, Dict, List[Dict]]]:
    """
    Memory-optimized worker function for self-play game generation.
    """
    # Configure minimal logging to reduce memory overhead
    worker_logger = logging.getLogger(f"Worker-{game_id}")
    if not worker_logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
        worker_logger.propagate = False
    worker_logger.setLevel(logging.INFO)

    worker_logger.info(f"Starting game {game_id}")

    try:
        # Initialize with CPU device for worker
        device = torch.device('cpu')

        # Create local tensor pool for network operations
        from ..memory import TensorPool, TensorPoolConfig
        tensor_pool_config = TensorPoolConfig(
            initial_size=30,   # Enough for several concurrent predictions
            enable_statistics=False,  # Keep overhead low in workers
            enable_adaptive_sizing=True,  # Enable adaptive sizing
            enable_tensor_reshaping=True,  # Enable tensor reshaping
            auto_device_selection=True    # Enable device-specific pooling
        )
        local_tensor_pool = TensorPool(tensor_pool_config)

        # Create network with tensor pool (use enhanced encoding if specified)
        use_enhanced_encoding = mcts_config.get('use_enhanced_encoding', False)
        network = NetworkWrapper(device=device, tensor_pool=local_tensor_pool,
                                 use_enhanced_encoding=use_enhanced_encoding)
        network.load_model(model_path)
        network.network.eval()

        # Create local memory pool for this worker
        from ..memory import GameStatePool, GameStatePoolConfig
        from ..game import GameState
        pool_config = GameStatePoolConfig(
            initial_size=50,  # Start with moderate pool size for worker
            enable_statistics=False,  # Disable stats in workers for performance
            factory_func=GameState  # Use GameState constructor as factory
        )
        local_game_state_pool = GameStatePool(pool_config)

        # Use network's encoder to ensure consistent encoding (basic or enhanced)
        state_encoder = network.state_encoder

        # Extract batched MCTS settings from config (with defaults)
        use_batched_mcts = mcts_config.get('use_batched_mcts', True)
        mcts_batch_size = mcts_config.get('mcts_batch_size', 32)

        # Remove these from mcts_config before passing to MCTS __init__
        mcts_init_config = {k: v for k, v in mcts_config.items()
                           if k not in ['use_batched_mcts', 'mcts_batch_size', 'use_enhanced_encoding']}

        mcts = MCTS(network=network, game_state_pool=local_game_state_pool, **mcts_init_config)

        # --- Memory-Efficient Game Data Structures ---
        state = local_game_state_pool.get()  # Get initial state from pool
        from ..memory import reset_game_state
        reset_game_state(state)  # Ensure it's properly initialized

        # Use lists instead of deques (less overhead)
        states = []
        policies = []
        players = []  # Track which player is to move at each position (for outcome perspective)

        # Keep minimal game history - just essentials for debugging
        game_history = []

        # Use a lightweight dict for temperature data
        temp_data = {'temperatures': [], 'entropies': [], 'search_times': []}

        move_count = 0
        max_game_moves = 300  # Safety limit

        # --- Main Game Loop ---
        while not state.is_terminal() and move_count < max_game_moves:
            # MCTS search with timing (use batched or serial based on configuration)
            search_start = time.time()
            if use_batched_mcts:
                move_probs = mcts.search_batch(state, move_count, batch_size=mcts_batch_size)
            else:
                move_probs = mcts.search(state, move_count)
            search_time = time.time() - search_start

            # Record search time
            temp_data['search_times'].append(search_time)

            # --- Action Selection ---
            temp = mcts.get_temperature(move_count)
            valid_moves = state.get_valid_moves()

            if not valid_moves:
                worker_logger.warning(f"No valid moves at move {move_count}. Ending game.")
                break

            # Get valid move probabilities
            valid_indices = [state_encoder.move_to_index(move) for move in valid_moves]
            valid_move_probs = move_probs[valid_indices]

            # Normalize probabilities
            prob_sum = valid_move_probs.sum()
            if prob_sum < 1e-6:
                valid_move_probs = np.ones(len(valid_moves), dtype=np.float32) / len(valid_moves)
            else:
                valid_move_probs /= prob_sum

            # Sample action based on temperature
            if temp == 0 or temp < 0.01:  # Greedy selection for very low temp
                selected_idx_in_valid = np.argmax(valid_move_probs)
            else:
                # Sample move based on probabilities
                try:
                    selected_idx_in_valid = np.random.choice(len(valid_moves), p=valid_move_probs)
                except ValueError as e:
                    worker_logger.error(f"Sampling error: {e}")
                    selected_idx_in_valid = np.argmax(valid_move_probs)

            selected_move = valid_moves[selected_idx_in_valid]

            # --- Record State and Policy ---
            # Store encoded state - using np.float32 instead of float64 reduces memory by half
            encoded_state = state_encoder.encode_state(state).astype(np.float32)
            states.append(encoded_state)
            policies.append(move_probs)
            players.append(state.current_player)  # Track player for outcome perspective

            # Calculate entropy for temperature adaptation
            entropy = -np.sum(valid_move_probs * np.log(valid_move_probs + 1e-9))

            # Record temperature data - just essentials
            temp_data['temperatures'].append(temp)
            temp_data['entropies'].append(entropy)

            # Only store minimal state info in history to save memory
            if move_count % 5 == 0:  # Only record every 5th move to history to save memory
                game_history.append({
                    'move_number': move_count,
                    'move': str(selected_move),
                    'phase': str(state.game_phase) if hasattr(state, 'game_phase') else "UNKNOWN",
                })

            # Make move
            state.make_move(selected_move)
            move_count += 1

            # Perform occasional garbage collection during very long games
            if move_count > 100 and move_count % 50 == 0:
                pass  # Memory pools handle cleanup automatically

            # Check for terminal state
            if state.is_terminal():
                worker_logger.debug(f"Terminal state reached after {move_count} moves")
                break

        # --- After Game Loop Ends ---
        # Record final state
        final_encoded_state = state_encoder.encode_state(state).astype(np.float32)
        states.append(final_encoded_state)

        # Add dummy policy for final state
        dummy_policy = np.zeros_like(policies[0]) if policies else np.zeros(state_encoder.total_moves, dtype=np.float32)
        policies.append(dummy_policy)
        players.append(state.current_player)  # Track player for final state too

        # Add final game info
        game_history.append({
            'move_number': move_count,
            'final_state': True,
            'white_score': state.white_score,
            'black_score': state.black_score
        })

        # Calculate outcome from White's perspective (+1 = White wins, -1 = Black wins)
        score_diff = state.white_score - state.black_score
        outcome_white = float(np.clip(score_diff / 3.0, -1.0, 1.0))

        # Backfill values for all positions based on game outcome
        # Each position's value = outcome from that player's perspective
        from ..game.types import Player
        values = []
        for player in players:
            if player == Player.WHITE:
                values.append(outcome_white)
            else:
                values.append(-outcome_white)  # Flip for Black's perspective

        outcome = outcome_white  # For logging

        # Log final summary - minimal logging to reduce overhead
        worker_logger.info(
            f"Game {game_id} finished in {move_count} moves. "
            f"Score: W={state.white_score}, B={state.black_score}. Outcome={outcome:.3f}"
        )

        # Clean up memory before returning
        # Release state if possible
        if 'local_game_state_pool' in locals() and 'state' in locals() and state is not None:
            local_game_state_pool.return_game_state(state)
            
        del network
        del mcts.network  # Explicitly clear network reference
        del mcts

        # Memory pools handle cleanup automatically

        # Return outcome-based values (standard AlphaZero approach)
        # Each position's value = game outcome from that player's perspective
        return states, policies, values, temp_data, game_history

    except Exception as e:
        worker_logger.error(f"Error in game {game_id}: {e}", exc_info=True)
        # Ensure cleanup even on error
        try:
            # Release state if possible
            if 'local_game_state_pool' in locals() and 'state' in locals() and state is not None:
                local_game_state_pool.return_game_state(state)
            del network
            del mcts
            # Memory pools handle cleanup automatically
        except:
            pass
        return None