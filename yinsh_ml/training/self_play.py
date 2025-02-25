"""Self-play implementation for YINSH ML training."""

import numpy as np
import torch
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

from yinsh_ml.utils.mcts_metrics import MCTSMetrics
from ..utils.TemperatureMetrics import TemperatureMetrics
from ..utils.metrics_logger import MetricsLogger, GameMetrics
from ..utils.encoding import StateEncoder
from ..game.game_state import GameState
from ..game.constants import Player
from ..network.wrapper import NetworkWrapper
from ..game.moves import Move

# Configure the logger at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the desired logging level

# Create handlers (console and file handlers can be added as needed)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the console handler level

# Optionally, add a file handler
# file_handler = logging.FileHandler('play_game_worker.log')
# file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
# logger.addHandler(file_handler)

class Node:
    """Monte Carlo Tree Search node."""

    def __init__(self, state: GameState, parent=None, prior_prob=0.0, c_puct=1.0):  # Add c_puct parameter
        self.state = state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False
        self.c_puct = c_puct  # Store c_puct as instance variable

    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, parent_visit_count: int) -> float:  # Remove c_puct parameter
        """Calculate Upper Confidence Bound score."""
        q_value = self.value()
        # Add small epsilon to prevent division by zero
        u_value = (self.c_puct * self.prior_prob *  # Use instance variable
                  np.sqrt(parent_visit_count) / (1 + self.visit_count))
        return q_value + u_value

class MCTS:
    """Monte Carlo Tree Search implementation."""

    def __init__(self, network: NetworkWrapper,
                 num_simulations: int = 100,
                 initial_temp: float = 1.0,
                 final_temp: float = 0.2,
                 annealing_steps: int = 30,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 initial_value_weight: float = 0.5,
                 max_depth: int = 20,
                 mcts_metrics: Optional[MCTSMetrics] = None
                 ):
        self.network = network
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("MCTS")

        # Temperature annealing parameters
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_steps = annealing_steps

        # Use the passed MCTSMetrics object or create a new one
        self.metrics = mcts_metrics if mcts_metrics is not None else MCTSMetrics()
        self.current_iteration = 0

        # MCTS exploration parameter
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha  # Dirichlet noise parameter

        # Value weight scaling for Q-values in UCB
        self.value_weight = initial_value_weight

        print(
            f"[MCTS] Initialized with c_puct={self.c_puct}, dirichlet_alpha={self.dirichlet_alpha}, value_weight={self.value_weight}")

    def get_temperature(self, move_number: int) -> float:
        """Calculate temperature based on move number."""
        if move_number >= self.annealing_steps:
            return self.final_temp

        # Linear annealing
        progress = move_number / self.annealing_steps
        return self.initial_temp - (self.initial_temp - self.final_temp) * progress

    def search(self, state: GameState, move_number: int) -> np.ndarray:
        """Perform MCTS search with temperature-adjusted move probabilities."""
        root = Node(state)
        move_probs = np.zeros(self.state_encoder.total_moves, dtype=np.float32)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            depth = 0  # Track depth

            # Selection
            while node.is_expanded and node.children:
                action = self._select_action(node)
                # print(f"Selected action: {action}")  # Debug print
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                depth += 1

            # Expansion and evaluation
            value = self._get_value(current_state)
            # print(f"Value from _get_value: {value}")  # Debug print
            if value is None:
                policy, value = self._evaluate_state(current_state)
                # print(f"Policy from network: {policy}")  # Debug print
                # print(f"Value from network: {value}")  # Debug print
                valid_moves = current_state.get_valid_moves()

                if valid_moves:
                    node.is_expanded = True
                    policy = self._mask_invalid_moves(policy, valid_moves)
                    for move in valid_moves:
                        node.children[move] = Node(
                            current_state.copy(),
                            parent=node,
                            prior_prob=self._get_move_prob(policy, move),
                            c_puct=self.c_puct  # Use dynamic c_puct
                        )
                    # Apply Dirichlet noise at the root if not already applied
                    if node.parent is None and not hasattr(node, "dirichlet_applied"):
                        epsilon_mix = 0.25  # Mixing fraction for Dirichlet noise
                        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
                        for i, move in enumerate(valid_moves):
                            original = node.children[move].prior_prob
                            node.children[move].prior_prob = (1 - epsilon_mix) * original + epsilon_mix * noise[i]
                        node.dirichlet_applied = True
                        # print(f"[MCTS] Applied Dirichlet noise at root with alpha {self.dirichlet_alpha}")
                else:
                    print("No valid moves to expand!")

            # Backpropagation
            self._backpropagate(search_path, value)
            # print(f"Value after backpropagation: {value}")

            # Collect MCTS metrics during search
            # print(f"Adding search depth: {depth}")  # Debug print
            self.metrics.add_search_depth(depth)
            # print(f"Recording branching factor: {len(node.children)}")  # Debug print
            self.metrics.record_branching_factor(len(node.children))

        # Calculate visit count distribution
        temp = self.get_temperature(move_number)
        # self.logger.debug(f"Using temperature {temp:.2f} at move {move_number}")

        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return move_probs

        visit_counts = np.array([
            root.children[move].visit_count
            for move in valid_moves
        ], dtype=np.float32)

        # Apply temperature
        if temp != 0:
            visit_counts = np.power(visit_counts, 1 / temp)

        # Normalize
        total_visits = visit_counts.sum()
        if total_visits > 0:
            visit_probs = visit_counts / total_visits
        else:
            visit_probs = np.ones_like(visit_counts) / len(visit_counts)

        # Store probabilities
        for move, prob in zip(valid_moves, visit_probs):
            move_idx = self.state_encoder.move_to_index(move)
            move_probs[move_idx] = prob

        return move_probs

    def _select_action(self, node: Node) -> Move:
        """Select action according to UCB formula with weighted Q-values."""
        valid_moves = list(node.children.keys())
        total_visits = node.visit_count
        epsilon = 1e-8
        ucb_scores = []
        for move in valid_moves:
            child = node.children[move]
            # Scale the Q-value using the configured value_weight
            scaled_q = self.value_weight * child.value()
            # Compute exploration term as usual
            u_value = child.c_puct * child.prior_prob * np.sqrt(total_visits) / (1 + child.visit_count)
            score = scaled_q + u_value + np.random.uniform(0, epsilon)
            ucb_scores.append(score)
        ucb_scores = np.array(ucb_scores)

        # Optional logging for debugging and health metrics
        # if len(ucb_scores) > 0:
        #     values = np.array([child.value() for child in node.children.values()])
        #     value_range = np.max(values) - np.min(values)
        #     best_by_value = valid_moves[np.argmax(values)]
        #     best_by_ucb = valid_moves[np.argmax(ucb_scores)]
        #     visit_counts = np.array([node.children[move].visit_count for move in valid_moves])
        #     max_visits = np.max(visit_counts)
        #     if value_range > 0.01 or best_by_value != best_by_ucb:
        #         print(f"[MCTS] _select_action: value_range={value_range:.4f}, max_visits={max_visits}")
        #         print(f"[MCTS] _select_action: best_by_value={best_by_value}, best_by_ucb={best_by_ucb}")
        #         self.metrics.record_position(self.current_iteration, {
        #             'value_range': value_range,
        #             'max_visits': max_visits,
        #             'moves': [
        #                 (str(valid_moves[i]), values[i], visit_counts[i], ucb_scores[i])
        #                 for i in np.argsort(values)[-3:]
        #             ],
        #             'value_ucb_disagreement': best_by_value != best_by_ucb,
        #             'game_phase': str(node.state.phase)
        #         })

        best_move = valid_moves[np.argmax(ucb_scores)]
        # print(f"[MCTS] _select_action: Selected move {best_move} with UCB score {np.max(ucb_scores):.4f}")
        return best_move

    def _evaluate_state(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Get policy and value from neural network."""
        state_tensor = self.state_encoder.encode_state(state)
        state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(
            self.network.device)  # Ensure tensor is on the correct device

        with torch.no_grad():
            policy, value = self.network.predict(state_tensor)
            return policy.squeeze().cpu().numpy(), value.item()  # Move policy to CPU before conversion

    def _get_value(self, state: GameState) -> Optional[float]:
        """Get value if game is terminal, None otherwise."""
        winner = state.get_winner()
        if winner is None:
            return None
        return 1.0 if winner == state.current_player else -1.0

    def _mask_invalid_moves(self, policy: np.ndarray, valid_moves: List[Move]) -> np.ndarray:
        """Mask out invalid moves and renormalize policy."""
        valid_indices = [self.state_encoder.move_to_index(move) for move in valid_moves]
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_indices] = policy[valid_indices]
        sum_valid = masked_policy.sum()
        if sum_valid > 0:
            masked_policy /= sum_valid
        else:
            # If all valid moves have zero probability, use uniform distribution
            masked_policy[valid_indices] = 1.0 / len(valid_indices)
        return masked_policy

    def _backpropagate(self, path: List[Node], value: float):
        """Backpropagate value through the tree."""
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Switch perspective between players

    def _get_move_prob(self, policy: np.ndarray, move: Move) -> float:
        """Get move probability from policy array."""
        move_idx = self.state_encoder.move_to_index(move)
        return policy[move_idx]


class SelfPlay:
    """Handles self-play game generation with temperature annealing."""

    def __init__(self, network: NetworkWrapper, metrics_logger: MetricsLogger, num_simulations: int = 100,
                 initial_temp: float = 1.0, final_temp: float = 0.2,
                 annealing_steps: int = 30, c_puct: float = 1.0,
                 max_depth: int = 20, mcts_metrics: Optional[MCTSMetrics] = None):
        self.network = network
        self.num_simulations = num_simulations
        self.metrics_logger = metrics_logger

        # M2-specific optimal worker count
        if platform.processor() == 'arm':  # Check for Apple Silicon
            # Optimal for M2: use 6 workers (leave 2 cores for system)
            self.num_workers = 6
        else:
            # Original logic for other CPUs
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count >= 32:
                self.num_workers = min(24, cpu_count - 4)
            elif cpu_count >= 16:
                self.num_workers = min(12, cpu_count - 2)
            else:
                self.num_workers = max(4, psutil.cpu_count(logical=False) - 1)

        self.current_iteration = 0

        # Temperature annealing parameters
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_steps = annealing_steps

        self.mcts = MCTS(
            network,
            num_simulations=num_simulations,
            initial_temp=initial_temp,
            final_temp=final_temp,
            annealing_steps=annealing_steps,
            c_puct=c_puct,         # Add new parameter
            max_depth=max_depth,    # Add new parameter
            mcts_metrics=mcts_metrics  # Pass MCTSMetrics to MCTS
        )
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("SelfPlay")
        self.temp_metrics = TemperatureMetrics()
        self.logger.setLevel(logging.ERROR)

    def _get_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on CPU cores."""
        cpu_count = psutil.cpu_count(logical=True)  # Get total cores including hyperthreading
        physical_cores = psutil.cpu_count(logical=False)  # Get physical cores only

        if cpu_count >= 32:  # For A10G and similar
            return min(24, cpu_count - 4)  # Leave some cores for overhead
        elif cpu_count >= 16:  # For T4-16CPU
            return min(12, cpu_count - 2)
        else:  # For smaller instances or local machine
            return max(4, physical_cores - 1)  # Leave one core free


    def save_draw_game(self, state: GameState, game_id: int, move_count: int, save_dir: str = "training_draw_games"):
        """Save details of a draw game from training."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save game details to text file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(save_dir / f"draw_game_{game_id}_{timestamp}.txt", "w") as f:
            f.write(f"Training Draw Game {game_id}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Moves: {move_count}\n")
            f.write(f"Final Score - White: {state.white_score}, Black: {state.black_score}\n")
            f.write("\nFinal Board State:\n")
            f.write(str(state.board))

            # Count pieces
            for piece_type in [PieceType.WHITE_RING, PieceType.BLACK_RING,
                               PieceType.WHITE_MARKER, PieceType.BLACK_MARKER]:
                count = len(state.board.get_pieces_positions(piece_type))
                f.write(f"\n{piece_type}: {count}")

    def generate_games(self, num_games: int = 100) -> List[Tuple[List[np.ndarray], List[np.ndarray], int]]:
        """Generate self-play games in parallel using multiple workers."""
        print(f"\nStarting generation of {num_games} games using {self.num_workers} workers")
        games = []
        num_simulations = self.num_simulations

        network_params = self.network.network.state_dict()

        # Reset temperature metrics for new batch
        self.temp_metrics = TemperatureMetrics()

        # Record metrics for mcts metric tracker
        self.mcts.current_iteration = self.current_iteration

        # Initialize MCTS statistics tracking
        mcts_stats = {
            'simulation_time': [],
            'tree_depth': [],
            'value_confidence': []
        }

        # Save model to temporary file for workers
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(network_params, tmp.name)
            model_path = tmp.name

        start_time = time.time()
        games_completed = 0

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(
                        play_game_worker,
                        model_path,
                        game_id,
                        num_simulations,
                        self.initial_temp,
                        self.final_temp,
                        self.annealing_steps,
                        self.mcts.metrics  # Pass mcts_metrics
                    )
                    for game_id in range(1, num_games + 1)
                ]

                while futures:
                    done, futures = concurrent.futures.wait(
                        futures,
                        timeout=10,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # Print CPU usage
                    cpu_percent = psutil.cpu_percent(percpu=True)
                    avg_cpu = sum(cpu_percent) / len(cpu_percent)
                    print(f"\nCPU Usage: {avg_cpu:.1f}% (across {len(cpu_percent)} cores)")
                    print(f"Active workers: {len(futures)}")

                    # Process completed games
                    for future in done:
                        try:
                            states, policies, outcome, temp_data, game_history = future.result()
                            games_completed += 1
                            games.append((states, policies, outcome, game_history))
                            elapsed = time.time() - start_time
                            rate = games_completed / elapsed

                            # Get phase-specific value predictions
                            phase_values = self._collect_phase_values(states)

                            # Calculate final confidence from last state
                            final_state = states[-1]
                            _, final_value = self.network.predict(
                                torch.FloatTensor(final_state).unsqueeze(0).to(self.network.device)
                            )
                            final_confidence = abs(final_value.item())

                            # if self.metrics_logger:
                            #     # Pass game history to metrics_logger for each game
                            #     self.metrics_logger.record_game_history(game_history)

                            # Enhanced game metrics
                            game_metrics = GameMetrics(
                                length=len(states),
                                outcome=outcome,
                                duration=elapsed,
                                avg_move_time=elapsed / len(states),
                                phase_values=phase_values,
                                final_confidence=final_confidence,
                                temperature_data=temp_data
                            )
                            self.metrics_logger.log_game(game_metrics)

                            # Simplified logging
                            self.logger.info(
                                f"\nGame {games_completed}/{num_games}: "
                                f"{len(states)} moves, "
                                f"{'White' if outcome == 1 else 'Black'} win, "
                                f"{rate:.2f} games/s"
                            )

                            # Update temperature metrics
                            for move_stat in temp_data['move_stats']:
                                self.temp_metrics.add_move_data(
                                    move_number=move_stat['move_number'],
                                    temperature=move_stat['temperature'],
                                    move_probs=np.array([0.8, 0.2]),
                                    selected_move_idx=0
                                )
                        except Exception as e:
                            self.logger.error(f"Error in game generation: {e}")

        finally:
            os.unlink(model_path)

        total_time = time.time() - start_time
        print(f"\nGame generation complete:")
        print(f"- Total time: {total_time:.1f} seconds")
        print(f"- Average time per game: {total_time / num_games:.1f} seconds")
        print(f"- Final rate: {num_games / total_time:.2f} games/second")

        return games

    def _collect_phase_values(self, states: List[np.ndarray]) -> Dict[str, List[float]]:
        """Collect value predictions for each game phase."""
        phase_values = defaultdict(list)

        for state in states:
            # Get game phase from state
            phase_channel = state[5]  # Assuming phase info is in channel 5
            phase = "placement" if np.mean(phase_channel) < 0.33 else \
                "main_game" if np.mean(phase_channel) < 0.66 else \
                    "ring_removal"

            # Get value prediction for this state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.network.device)
            _, value = self.network.predict(state_tensor)
            phase_values[phase].append(value.item())

        return dict(phase_values)

    def play_game(self, game_id: int) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Play a single game with temperature annealing."""
        state = GameState()
        states = []
        policies = []
        move_count = 0
        game_history = []

        self.logger.info(f"\nStarting game {game_id}")

        while not state.is_terminal() and move_count < 5000:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                self.logger.warning(f"Game {game_id} move {move_count}: No valid moves available.")
                break

            temp = self.mcts.get_temperature(move_count)
            move_probs = self.mcts.search(state, move_count)

            valid_indices = [self.state_encoder.move_to_index(move) for move in valid_moves]
            valid_probs = move_probs[valid_indices]
            if valid_probs.sum() == 0:
                self.logger.error(f"Game {game_id} move {move_count}: All valid moves have zero probability!")
                break
            valid_probs = valid_probs / valid_probs.sum()

            selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
            selected_move = valid_moves[selected_idx]

            # Make the move first.
            success = state.make_move(selected_move)
            if not success:
                self.logger.error(f"Game {game_id} move {move_count}: Failed to make move: {selected_move}")
                break

            move_count += 1

            # Record the post-move state (which reflects any updates from the move)
            encoded_state = self.state_encoder.encode_state(state)
            states.append(encoded_state)

            # The move_probs or valid_probs from MCTS search
            policies.append(move_probs)  # Make sure move_probs has shape [total_moves]
            game_history.append({
                'state': state.copy(),
                'move': selected_move,
                'move_probs': valid_probs,
                'temperature': temp,
                'value_pred': None  # Optionally fill in if needed
            })

            # Check immediately if the move resulted in a terminal state.
            if state.is_terminal():
                self.logger.debug(f"Game {game_id}: Terminal state reached after {move_count} moves.")
                break

        # After exiting the loop, ensure the final state is recorded.
        if state.is_terminal():
            # Check if the last recorded state in game_history is terminal.
            if not game_history or not game_history[-1]['state'].is_terminal():
                self.logger.debug(f"Game {game_id}: Appending final terminal state to history.")
                final_encoded_state = self.state_encoder.encode_state(state)

                states.append(final_encoded_state)

                # Append a dummy policy to keep lengths in sync
                dummy_policy = np.zeros_like(move_probs)
                policies.append(dummy_policy)

                game_history.append({
                    'state': state.copy(),
                    'move': None,
                    'move_probs': None,
                    'temperature': None,
                    'value_pred': None
                })

        self.logger.debug(f"Game {game_id}: Final state - Phase: {state.phase}, "
                          f"White score: {state.white_score}, Black score: {state.black_score}, "
                          f"Terminal: {state.is_terminal()}")

        winner = state.get_winner()
        outcome = 1 if winner == Player.WHITE else (-1 if winner == Player.BLACK else 0)
        return states, policies, outcome

    def export_games(self, games: List[Tuple[List[np.ndarray], List[np.ndarray], int]],
                    path: str):
        """Export games to file."""
        data = {
            'states': [state for game in games for state in game[0]],
            'policies': [policy for game in games for policy in game[1]],
            'outcomes': [game[2] for game in games]
        }
        np.save(path, data)
        self.logger.info(f"Exported {len(games)} games to {path}")


def play_game_worker(
    model_path: str,
    game_id: int,
    num_simulations: int,
    initial_temp: float = 1.0,
    final_temp: float = 0.2,
    annealing_steps: int = 30,
    mcts_metrics: Optional[MCTSMetrics] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], int, Dict, List[Dict]]:
    """
    Worker function to play a single game with temperature metrics and robust logging.
    Returns:
        Tuple containing:
         - List of encoded game states.
         - List of policy vectors.
         - Game outcome.
         - Temperature metrics dictionary.
         - Game history (list of dicts, each with a 'state' key that is a GameState).
    """
    logger.info(f"Worker {game_id} starting game...")
    device = torch.device('cpu')

    try:
        network = NetworkWrapper(model_path=model_path, device=device)
        mcts = MCTS(
            network,
            num_simulations=num_simulations,
            initial_temp=initial_temp,
            final_temp=final_temp,
            annealing_steps=annealing_steps,
            mcts_metrics=mcts_metrics
        )
        state = GameState()
        state_encoder = StateEncoder()

        states: List[np.ndarray] = []
        policies: List[np.ndarray] = []
        move_count = 0
        temp_data: Dict[str, List] = {
            'temperatures': [],
            'entropies': [],
            'move_stats': []
        }
        game_history: List[Dict] = []

        move_start = time.time()

        # Main loop
        while not state.is_terminal() and move_count < 5000:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                logger.warning(f"Worker {game_id} move {move_count}: No valid moves available.")
                break

            # 1) Get temperature and MCTS policy
            temp = mcts.get_temperature(move_count)
            move_probs = mcts.search(state, move_count)  # shape [action_size] (e.g., 7395)

            # 2) Filter for valid_moves
            valid_indices = [state_encoder.move_to_index(move) for move in valid_moves]
            valid_probs = move_probs[valid_indices]
            if valid_probs.sum() == 0:
                logger.warning(f"Worker {game_id} move {move_count}: All valid moves have zero probability.")
                break
            valid_probs = valid_probs / valid_probs.sum()

            # 3) Select move from the valid set
            selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
            selected_move = valid_moves[selected_idx]

            # 4) Make the move
            success = state.make_move(selected_move)
            if not success:
                logger.error(f"Worker {game_id} move {move_count}: Failed to make move: {selected_move}")
                break

            move_count += 1

            # 5) Encode the post-move state and record policy
            encoded_state = state_encoder.encode_state(state)
            states.append(encoded_state)
            policies.append(move_probs)  # <-- IMPORTANT: store the *full* policy vector

            # Record extras in game_history
            game_history.append({
                'state': state.copy(),  # Copy the GameState object
                'move': selected_move,
                'move_probs': valid_probs,  # Probability distribution over valid moves only
                'temperature': temp,
                'value_pred': None  # optionally fill in if desired
            })

            # 6) Collect temperature/entropy info
            entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
            temp_data['temperatures'].append((move_count, temp))
            temp_data['entropies'].append((move_count, entropy))
            temp_data['move_stats'].append({
                'move_number': move_count,
                'temperature': temp,
                'entropy': entropy,
                'top_prob': float(np.max(valid_probs)),
                'selected_prob': float(valid_probs[selected_idx]),
                'move_time': time.time()
            })

            # If that move led to a terminal state, break
            if state.is_terminal():
                logger.debug(f"Worker {game_id}: Terminal state reached after {move_count} moves.")
                break

        # Final check (optional, in case we want to store one more “terminal state”).
        state._update_game_phase()
        if state.is_terminal():
            # Append *one* terminal state plus a dummy policy to keep lengths aligned.
            final_encoded_state = state_encoder.encode_state(state)
            states.append(final_encoded_state)
            # Use a zero vector of same shape as move_probs
            dummy_policy = np.zeros_like(move_probs, dtype=np.float32)
            policies.append(dummy_policy)
            game_history.append({
                'state': state.copy(),
                'move': None,
                'move_probs': None,
                'temperature': None,
                'value_pred': None
            })

        # Debug:
        logger.debug(
            f"Worker {game_id}: Final #states={len(states)}, #policies={len(policies)}"
        )
        logger.debug(
            f"Worker {game_id}: Final state - Phase={state.phase}, "
            f"Score: W={state.white_score}, B={state.black_score}, Terminal={state.is_terminal()}"
        )

        # Determine outcome
        winner = state.get_winner()
        if winner == Player.WHITE:
            outcome = 1
            logger.info(f"Worker {game_id}: White wins")
        elif winner == Player.BLACK:
            outcome = -1
            logger.info(f"Worker {game_id}: Black wins")
        else:
            outcome = 0
            logger.info(f"Worker {game_id}: Draw or no valid outcome")

        return states, policies, outcome, temp_data, game_history

    except Exception as e:
        logger.exception(f"Worker {game_id} error: {e}")
        raise