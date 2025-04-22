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

# Configure the logger at the module level
# Use getLogger to ensure consistency if already configured elsewhere
logger = logging.getLogger("SelfPlay")
# Set default level if not configured by runner/supervisor
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     logger.addHandler(logging.StreamHandler()) # Add console handler if none exist
     logger.setLevel(logging.INFO)


class Node:
    """Monte Carlo Tree Search node."""
    # --- NO CHANGES NEEDED ---
    def __init__(self, state: GameState, parent=None, prior_prob=0.0, c_puct=1.0):
        self.state = state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False
        self.c_puct = c_puct # Store c_puct as instance variable

    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, parent_visit_count: int) -> float:
        """Calculate Upper Confidence Bound score."""
        q_value = self.value()
        # Add small epsilon to prevent division by zero
        u_value = (self.c_puct * self.prior_prob *
                  np.sqrt(parent_visit_count) / (1 + self.visit_count))
        return q_value + u_value

class MCTS:
    """Monte-Carlo Tree-Search with temperature annealing and ply-adaptive rollouts."""

    def __init__(self,
                 network: NetworkWrapper,
                 # --- MCTS Search Params ---
                 num_simulations: int, # Base simulations (early game)
                 c_puct: float,
                 dirichlet_alpha: float,
                 # *** Rename 'initial_value_weight' to 'value_weight' for clarity ***
                 value_weight: float, # Weighting for value in UCB selection
                 max_depth: int,
                 # --- Rollout Scheduling Params ---
                 late_simulations: Optional[int], # Use Optional[int] for type hint clarity
                 simulation_switch_ply: int,
                 # --- Temperature Params (passed down) ---
                 initial_temp: float,
                 final_temp: float,
                 annealing_steps: int,
                 temp_clamp_fraction: float,
                 # --- Optional Metrics ---
                 mcts_metrics: Optional[MCTSMetrics] = None,
                ):
        """
        Initialize MCTS.

        Args:
            network: The neural network wrapper.
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
        """
        self.network = network
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("MCTS") # Use module-level logger

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
        self.logger.info(f"  Sims: Early={self.early_simulations}, Late={self.late_simulations} (Switch Ply {self.switch_ply})")
        self.logger.info(f"  Search: cPUCT={self.c_puct:.2f}, Alpha={self.dirichlet_alpha:.2f}, Value Weight={self.value_weight:.2f}, Max Depth={self.max_depth}")
        self.logger.info(f"  Temperature: Initial={self.initial_temp:.2f}, Final={self.final_temp:.2f}, Steps={self.annealing_steps}, Clamp Frac={self.temp_clamp_frac:.2f}")


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
        for sim in range(budget):
            node = root
            search_path = [node]
            current_state = state.copy()
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
                            node.children[move] = Node(
                                current_state.copy(),
                                parent=node,
                                prior_prob=self._get_move_prob(policy, move),
                                c_puct=self.c_puct
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


        return move_probs


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
        """Get policy and value from neural network."""
        state_tensor = self.state_encoder.encode_state(state)
        # Ensure tensor is float and add batch dimension
        state_tensor = torch.from_numpy(state_tensor).float().unsqueeze(0).to(self.network.device)

        with torch.no_grad():
            policy_logits, value = self.network.predict(state_tensor)
            # Apply softmax to logits to get probabilities, move to CPU, remove batch dim
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value_scalar = value.item() # Get scalar value
            return policy_probs, value_scalar

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
                 # --- MCTS Params (to be passed to MCTS instance) ---
                 num_simulations: int, # Early game sims
                 late_simulations: Optional[int],
                 simulation_switch_ply: int,
                 c_puct: float,
                 dirichlet_alpha: float,
                 # *** Use consistent 'value_weight' naming ***
                 value_weight: float,
                 max_depth: int,
                 # --- Temp Params (to be passed to MCTS instance) ---
                 initial_temp: float,
                 final_temp: float,
                 annealing_steps: int,
                 temp_clamp_fraction: float,
                 # --- Optional Metrics ---
                 metrics_logger: Optional[MetricsLogger] = None,
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 ):
        """
        Initialize SelfPlay with explicit MCTS and temperature parameters.
        """
        self.network = network
        self.num_workers = num_workers # Use the calculated number passed in
        self.metrics_logger = metrics_logger
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("SelfPlay") # Use module logger
        self.current_iteration = 0 # Track current training iteration if needed

        # Store all parameters needed by MCTS and workers explicitly
        self.mcts_config = {
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
        }
        # Store optional metrics instances
        self.mcts_metrics = mcts_metrics # Store the instance

        # Instantiate the main MCTS object for the SelfPlay process itself (if needed, e.g., single-threaded play)
        # This MCTS instance is primarily used if play_game is called directly,
        # worker processes will create their own MCTS instances.
        self.mcts = MCTS(
            network=self.network,
            mcts_metrics=self.mcts_metrics, # Pass metrics instance
            **self.mcts_config # Unpack the config dict
        )

        self.logger.info("SelfPlay Initialized:")
        self.logger.info(f"  Workers: {self.num_workers}")
        self.logger.info(f"  MCTS Config: {self.mcts_config}")


    def generate_games(self, num_games: int) -> List[Tuple[List[np.ndarray], List[np.ndarray], float, List[Dict]]]:
        """Generate self-play games in parallel using multiple workers."""
        self.logger.info(f"Starting generation of {num_games} games using {self.num_workers} workers...")
        games_data: List[Tuple[List[np.ndarray], List[np.ndarray], float, List[Dict]]] = []

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
                             states, policies, outcome, temp_data, game_history = result
                             games_data.append((states, policies, outcome, game_history))
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
    # --- Receive MCTS configuration as a dictionary ---
    mcts_config: Dict,
    # --- Optional metrics ---
    # mcts_metrics: Optional[MCTSMetrics] = None # Can be passed if MCTS/worker needs to log metrics
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], float, Dict, List[Dict]]]:
    """
    Worker function executed in a separate process to play a single self-play game.

    It initializes its own MCTS instance based on the provided configuration and
    simulates a game from start to finish, collecting state, policy, and outcome data.

    Args:
        model_path: Path to the saved model state_dict file (usually temporary).
        game_id: Identifier for the game being played (for logging).
        mcts_config: Dictionary containing all necessary parameters for initializing
                     the MCTS instance (e.g., simulation counts, c_puct, alpha, temp settings).
        # mcts_metrics: Optional MCTSMetrics collector instance (if shared metrics are needed).

    Returns:
        A tuple containing:
            - states (List[np.ndarray]): List of encoded game states encountered.
            - policies (List[np.ndarray]): List of full MCTS policy vectors corresponding to each state.
            - outcome (float): The normalized game outcome margin (score_diff / 3.0) in [-1, 1].
            - temp_data (Dict): Dictionary containing temperature and entropy metrics during the game.
            - game_history (List[Dict]): A detailed log of each step, including move, probabilities, times, etc.
        Returns None if a critical error occurs during game play.
    """
    # Configure logging for the worker process independently.
    # This helps isolate logs from different game simulations.
    worker_logger = logging.getLogger(f"Worker-{game_id}")
    # Set level - inheriting from the root logger configuration is often best practice.
    # If the root logger isn't configured (e.g., running this worker standalone),
    # we set a default level and handler.
    if not worker_logger.hasHandlers() and not logging.getLogger().hasHandlers():
         # Setup basic console logging if no handlers are configured at all
         handler = logging.StreamHandler()
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         handler.setFormatter(formatter)
         # Add handler to this specific worker logger
         worker_logger.addHandler(handler)
         worker_logger.propagate = False # Prevent duplicate logging if root also gets configured later
    worker_logger.setLevel(logging.INFO) # Set desired level (INFO or DEBUG)

    worker_logger.info(f"Initializing worker and starting game simulation...")

    try:
        # --- Initialization within the Worker Process ---

        # Device: Workers typically run on CPU to avoid GPU contention/memory issues
        # when running many workers, unless specifically designed for multi-GPU setups.
        device = torch.device('cpu')

        # Network: Load the model architecture and weights from the provided path.
        # Create a new NetworkWrapper instance within the worker.
        network = NetworkWrapper(device=device) # Initialize on the worker's device (CPU)
        network.load_model(model_path) # Load the state dict
        network.network.eval() # Set the network to evaluation mode (important for consistent predictions)
        worker_logger.debug(f"Network loaded from {model_path} onto {device}.")

        # Utilities: Initialize state encoder.
        state_encoder = StateEncoder()

        # MCTS: Instantiate the MCTS engine using the configuration passed from the supervisor.
        # The '**mcts_config' unpacks the dictionary into keyword arguments for MCTS.__init__.
        mcts = MCTS(
            network=network, # Pass the worker's network instance
            # mcts_metrics=mcts_metrics, # Pass metrics collector if provided and needed
            **mcts_config
        )
        # Log key MCTS parameters being used by this worker for verification.
        worker_logger.info(f"Worker MCTS check: Sims={mcts.early_simulations}, cPUCT={mcts.c_puct}, Alpha={mcts.dirichlet_alpha}, ValueWeight={mcts.value_weight}")

        # --- Game Simulation Data Structures ---
        state = GameState() # Initialize a new game state
        states: List[np.ndarray] = [] # To store encoded states before each move
        policies: List[np.ndarray] = [] # To store the full MCTS policy vector for each state
        game_history: List[Dict] = [] # To store detailed step-by-step game information
        temp_data: Dict[str, List] = defaultdict(list) # To store temperature/entropy related metrics
        mcts_search_times: List[float] = [] # To store the duration of each MCTS search call

        move_count = 0
        max_game_moves = 500 # Safety limit to prevent potential infinite loops

        # --- Main Game Loop ---
        # Continue as long as the game is not terminal and the move limit hasn't been reached.
        while not state.is_terminal() and move_count < max_game_moves:

            # --- MCTS Search and Timing ---
            search_start_time = time.time()
            # Perform MCTS search from the current state.
            # The mcts.search method handles simulation budget switching internally based on move_count.
            move_probs = mcts.search(state, move_count) # Returns the full policy vector over all possible moves
            search_time = time.time() - search_start_time
            mcts_search_times.append(search_time) # Record the search duration

            # --- Action Selection ---
            # Get the current temperature for sampling.
            temp = mcts.get_temperature(move_count)
            # Get the list of currently valid moves from the game state.
            valid_moves = state.get_valid_moves()

            # Handle the case where no valid moves are available (should signal a terminal state, but check defensively).
            if not valid_moves:
                worker_logger.warning(f"No valid moves available at move {move_count}. Game state indicates terminal: {state.is_terminal()}. Ending game.")
                break # Exit the loop if no moves can be made

            # Filter the full policy vector to only include probabilities for valid moves.
            valid_indices = [state_encoder.move_to_index(move) for move in valid_moves]
            # Use advanced indexing to get probabilities for valid moves efficiently.
            valid_move_probs = move_probs[valid_indices]

            # Normalize the probabilities for valid moves to ensure they sum to 1.
            prob_sum = valid_move_probs.sum()
            if prob_sum < 1e-6: # Check if the sum is close to zero
                worker_logger.warning(f"Sum of probabilities for valid moves is near zero at move {move_count}. Using uniform distribution.")
                num_valid = len(valid_moves)
                # Assign uniform probability if possible, otherwise handle the zero-move case (already checked by 'if not valid_moves').
                valid_move_probs = np.ones(num_valid, dtype=np.float32) / num_valid if num_valid > 0 else np.array([], dtype=np.float32)
            else:
                 # Normalize the valid probabilities.
                 valid_move_probs /= prob_sum

            # Sample the action to take based on the normalized, valid probabilities and temperature.
            if temp == 0:
                 # If temperature is zero, choose the move with the highest probability (greedy).
                 selected_idx_in_valid = np.argmax(valid_move_probs)
            else:
                 # If temperature is non-zero, sample probabilistically.
                 # MCTS search output (move_probs) should already be temperature-adjusted if designed that way.
                 # If not, apply power scaling here: probs^(1/temp) and re-normalize.
                 # Assuming MCTS output is ready for sampling:
                 if not np.isclose(valid_move_probs.sum(), 1.0): # Final check for normalization
                      worker_logger.warning(f"Probabilities for valid moves do not sum to 1 ({valid_move_probs.sum()}) before sampling. Renormalizing.")
                      valid_move_probs /= valid_move_probs.sum()

                 # Ensure there are probabilities to sample from
                 if valid_move_probs.size == 0:
                     worker_logger.error(f"Cannot sample move at move {move_count}: valid_move_probs is empty.")
                     break # Exit loop if sampling is impossible

                 try:
                      selected_idx_in_valid = np.random.choice(len(valid_moves), p=valid_move_probs)
                 except ValueError as e:
                      worker_logger.error(f"Error sampling move at move {move_count}: {e}. Probs sum: {valid_move_probs.sum()}, Probs: {valid_move_probs}", exc_info=True)
                      # Fallback: Choose the best move greedily if sampling fails
                      selected_idx_in_valid = np.argmax(valid_move_probs)


            # Get the selected Move object.
            selected_move = valid_moves[selected_idx_in_valid]

            # --- Record State and Policy Data (BEFORE the move is made) ---
            # Store the encoded state representation.
            encoded_state = state_encoder.encode_state(state)
            states.append(encoded_state)
            # Store the full policy vector generated by MCTS for this state.
            policies.append(move_probs)

            # --- Calculate Metrics for this Step ---
            # Calculate entropy of the probability distribution over valid moves.
            entropy = -np.sum(valid_move_probs * np.log(valid_move_probs + 1e-9)) if valid_move_probs.size > 0 else 0.0

            # --- Append Detailed Step Information to Game History ---
            # This history is useful for detailed analysis and debugging.
            game_history.append({
                # Optional: Store full GameState object (can consume significant memory if kept for many games)
                # 'state': state.copy(),
                'state_encoded': encoded_state, # Store lightweight encoded state
                'move': str(selected_move), # Store string representation of the chosen move
                'policy_valid': valid_move_probs.tolist() if valid_move_probs.size > 0 else [], # Policy distribution over valid moves
                'policy_full': move_probs.tolist(), # Full policy vector from MCTS
                'temperature': temp, # Temperature used for sampling this move
                'entropy': entropy, # Entropy of the valid move distribution
                'mcts_search_time': search_time, # Time taken for MCTS search for this move
                'value_pred': None # Placeholder: Could fetch MCTS root value here if needed
            })

            # --- Record Temperature Data ---
            temp_data['temperatures'].append(temp)
            temp_data['entropies'].append(entropy)
            temp_data['move_stats'].append({ # More detailed stats per move
                'move_number': move_count + 1, # Move number (starting from 1)
                'temperature': temp,
                'entropy': entropy,
                'top_prob': float(np.max(valid_move_probs)) if valid_move_probs.size > 0 else 0.0,
                'selected_prob': float(valid_move_probs[selected_idx_in_valid]) if valid_move_probs.size > 0 else 0.0,
                'search_time': search_time # Record the search time here as well
            })

            # --- Apply the Selected Move to the Game State ---
            state.make_move(selected_move)
            move_count += 1 # Increment move counter

            # --- Check for Terminal State After Move ---
            if state.is_terminal():
                worker_logger.debug(f"Terminal state reached after {move_count} moves.")
                break # Exit the loop immediately if game ends

        # --- After Game Loop Ends (Terminal State or Max Moves) ---

        if move_count >= max_game_moves:
            worker_logger.warning(f"Game reached max move limit ({max_game_moves}).")
            # The game state might not be technically terminal according to rules,
            # but we treat it as finished for training purposes.

        # --- Record the Final State ---
        # Append the final encoded state reached after the last move (or at max moves).
        final_encoded_state = state_encoder.encode_state(state)
        states.append(final_encoded_state)
        # Append a dummy policy vector (e.g., zeros) for the final state, as no action
        # is taken from it. This keeps the lengths of states and policies aligned.
        dummy_policy = np.zeros_like(policies[0]) if policies else np.zeros(state_encoder.total_moves, dtype=np.float32)
        policies.append(dummy_policy)

        # Add a final entry to game_history representing the end state.
        game_history.append({
            'state_encoded': final_encoded_state,
            'move': None, 'policy_valid': None, 'policy_full': None,
            'temperature': None, 'entropy': None, 'mcts_search_time': None,
            'value_pred': None # Could potentially store final actual value here
        })

        # --- Determine Final Game Outcome ---
        # Calculate the normalized margin based on the final scores.
        score_diff = state.white_score - state.black_score
        # Normalize by the maximum possible score difference (3-0 vs 0-3 => range 6, centered at 0). Max absolute score is 3.
        normalized_margin = np.clip(score_diff / 3.0, -1.0, 1.0) # Ensure value is within [-1, 1]
        outcome = float(normalized_margin)

        # --- Log Final Game Summary ---
        avg_search_time_ms = np.mean(mcts_search_times) * 1000 if mcts_search_times else 0
        worker_logger.info(
            f"Game finished in {move_count} moves. "
            f"Final Score: W={state.white_score}, B={state.black_score}. Outcome={outcome:.3f}. "
            f"Avg MCTS Search Time: {avg_search_time_ms:.2f} ms"
        )

        # Add final collected timings to temp_data if needed by the caller.
        temp_data['mcts_search_times'] = mcts_search_times

        # --- Return Collected Data ---
        # Return the list of states, policies, the final outcome, temperature data, and detailed history.
        return states, policies, outcome, temp_data, game_history

    except Exception as e:
        # Log any unexpected exceptions that occur within the worker process.
        worker_logger.error(f"Critical error during game play: {e}", exc_info=True)
        # Return None to signal that this worker failed to produce a valid game.
        return None