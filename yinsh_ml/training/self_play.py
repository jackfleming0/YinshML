"""Self-play implementation for YINSH ML training."""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import random
import logging
from concurrent.futures import ProcessPoolExecutor
import time
import tempfile
import os
from pathlib import Path

from ..utils.TemperatureMetrics import TemperatureMetrics
from ..utils.encoding import StateEncoder
from ..game.game_state import GamePhase, GameState
from ..game.constants import Player, Position
from ..network.wrapper import NetworkWrapper
from ..game.moves import Move, MoveType

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
                 max_depth: int = 20):
        self.network = network
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("MCTS")

        # Temperature annealing parameters
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_steps = annealing_steps

        # MCTS exploration parameter
        self.c_puct = c_puct


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
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                depth += 1

            # Expansion and evaluation
            value = self._get_value(current_state)
            if value is None:
                policy, value = self._evaluate_state(current_state)
                valid_moves = current_state.get_valid_moves()

                if valid_moves:
                    node.is_expanded = True
                    policy = self._mask_invalid_moves(policy, valid_moves)

                    for move in valid_moves:
                        node.children[move] = Node(
                            current_state.copy(),
                            parent=node,
                            prior_prob=self._get_move_prob(policy, move),
                            c_puct=self.c_puct  # Pass the c_puct value
                        )

            # Backpropagation
            self._backpropagate(search_path, value)

        # Calculate visit count distribution
        temp = self.get_temperature(move_number)
        self.logger.debug(f"Using temperature {temp:.2f} at move {move_number}")

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
        """Select action according to UCB formula."""
        valid_moves = list(node.children.keys())
        visit_counts = np.array([node.children[move].visit_count for move in valid_moves])
        total_visits = node.visit_count

        # Add small epsilon to all UCB scores to break ties randomly
        epsilon = 1e-8
        ucb_scores = np.array([
            node.children[move].get_ucb_score(total_visits) + np.random.uniform(0, epsilon)
            for move in valid_moves
        ])

        best_move = valid_moves[np.argmax(ucb_scores)]
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

    def __init__(self, network: NetworkWrapper, num_simulations: int = 100, num_workers: int = 4,
                 initial_temp: float = 1.0, final_temp: float = 0.2, annealing_steps: int = 30,
                 c_puct: float = 1.0, max_depth: int = 20):  # Add new parameters
        self.network = network
        self.num_simulations = num_simulations
        self.num_workers = num_workers

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
            max_depth=max_depth    # Add new parameter
        )
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("SelfPlay")
        self.temp_metrics = TemperatureMetrics()
        self.logger.setLevel(logging.ERROR)


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
        games = []
        num_simulations = self.num_simulations
        network_params = self.network.network.state_dict()

        # Reset temperature metrics for new batch
        self.temp_metrics = TemperatureMetrics()

        # Save the model to a temporary file for workers to load
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(network_params, tmp.name)
            model_path = tmp.name

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
                        self.annealing_steps
                    )
                    for game_id in range(1, num_games + 1)
                ]
                for future in futures:
                    try:
                        states, policies, outcome, temp_data = future.result()
                        games.append((states, policies, outcome))

                        # Create a dummy probability distribution for metrics
                        dummy_probs = np.array([0.8, 0.2])  # Just for metrics tracking

                        # Update temperature metrics from this game's data
                        for move_stat in temp_data['move_stats']:
                            self.temp_metrics.add_move_data(
                                move_number=move_stat['move_number'],
                                temperature=move_stat['temperature'],
                                move_probs=dummy_probs,  # Using dummy probs instead of entropy
                                selected_move_idx=0
                            )

                        self.logger.info("Game completed successfully.")
                    except Exception as e:
                        self.logger.error(f"Error in game generation: {e}")
        finally:
            # Clean up the temporary model file
            os.unlink(model_path)

        return games

    def play_game(self, game_id: int) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Play a single game with temperature annealing."""
        state = GameState()
        states = []
        policies = []
        move_count = 0

        self.logger.info(f"\nStarting game {game_id}")

        while not state.is_terminal() and move_count < 5000:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                break

            # Get current temperature
            temp = self.mcts.get_temperature(move_count)

            # Run MCTS with current temperature
            move_probs = self.mcts.search(state, move_count)

            # Get probabilities for valid moves
            valid_indices = [self.state_encoder.move_to_index(move) for move in valid_moves]
            valid_probs = move_probs[valid_indices]

            if valid_probs.sum() == 0:
                self.logger.error("All valid moves have zero probability!")
                break

            # Normalize probabilities
            valid_probs = valid_probs / valid_probs.sum()

            # Select move
            selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
            selected_move = valid_moves[selected_idx]

            # Track temperature metrics
            self.temp_metrics.add_move_data(
                move_number=move_count,
                temperature=temp,
                move_probs=valid_probs,
                selected_move_idx=selected_idx
            )

            # Store state and policy before making move
            encoded_state = self.state_encoder.encode_state(state)
            states.append(encoded_state)
            policies.append(move_probs)

            # Make the move
            success = state.make_move(selected_move)
            if not success:
                self.logger.error(f"Failed to make move: {selected_move}")
                break

            move_count += 1

        # Determine outcome
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


def play_game_worker(model_path: str, game_id: int, num_simulations: int,
                     initial_temp: float = 1.0, final_temp: float = 0.2,
                     annealing_steps: int = 30) -> Tuple[List[np.ndarray], List[np.ndarray], int, Dict]:
    """Worker function to play a single game with temperature metrics."""
    # Initialize the network and MCTS in the worker
    device = torch.device('cpu')
    network = NetworkWrapper(model_path=model_path, device=device)
    mcts = MCTS(
        network,
        num_simulations=num_simulations,
        initial_temp=initial_temp,
        final_temp=final_temp,
        annealing_steps=annealing_steps
    )
    state_encoder = StateEncoder()
    state = GameState()

    # Initialize game data
    states = []
    policies = []
    move_count = 0

    # Initialize temperature metrics for this game
    temp_data = {
        'temperatures': [],  # List of (move_number, temp) tuples
        'entropies': [],  # List of (move_number, entropy) tuples
        'move_stats': []  # List of detailed move statistics
    }

    while not state.is_terminal() and move_count < 5000:
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            break

        # Get current temperature
        temp = mcts.get_temperature(move_count)

        # Run MCTS to get move probabilities
        move_probs = mcts.search(state, move_count)

        # Get probabilities for valid moves
        valid_indices = [state_encoder.move_to_index(move) for move in valid_moves]
        valid_probs = move_probs[valid_indices]

        if valid_probs.sum() == 0:
            break

        # Normalize probabilities
        valid_probs = valid_probs / valid_probs.sum()

        # Apply temperature
        if temp > 0:
            adj_probs = np.power(valid_probs, 1 / temp)
            adj_probs = adj_probs / adj_probs.sum()
        else:
            adj_probs = valid_probs

        # Select move
        selected_idx = np.random.choice(len(valid_moves), p=adj_probs)
        selected_move = valid_moves[selected_idx]

        # Record temperature metrics
        entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))
        temp_data['temperatures'].append((move_count, temp))
        temp_data['entropies'].append((move_count, entropy))
        temp_data['move_stats'].append({
            'move_number': move_count,
            'temperature': temp,
            'entropy': entropy,
            'top_prob': np.max(valid_probs),
            'selected_prob': valid_probs[selected_idx]
        })

        # Store state and policy
        encoded_state = state_encoder.encode_state(state)
        states.append(encoded_state)
        policies.append(move_probs)

        # Make move
        state.make_move(selected_move)
        move_count += 1

    # Determine outcome
    winner = state.get_winner()
    outcome = 1 if winner == Player.WHITE else (-1 if winner == Player.BLACK else 0)

    return states, policies, outcome, temp_data