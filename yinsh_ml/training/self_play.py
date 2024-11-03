"""Self-play implementation for YINSH ML training."""

import numpy as np
import torch
from typing import List, Tuple, Optional
from collections import defaultdict
import random
import logging
from concurrent.futures import ProcessPoolExecutor
import time
import tempfile
import os


from ..utils.encoding import StateEncoder
from ..game.game_state import GamePhase, GameState
from ..game.constants import Player, Position
from ..network.wrapper import NetworkWrapper
from ..game.moves import Move, MoveType

class Node:
    """Monte Carlo Tree Search node."""

    def __init__(self, state: GameState, parent=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False

    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, parent_visit_count: int, c_puct: float = 1.0) -> float:
        """Calculate Upper Confidence Bound score."""
        q_value = self.value()
        # Add small epsilon to prevent division by zero
        u_value = (c_puct * self.prior_prob *
                  np.sqrt(parent_visit_count) / (1 + self.visit_count))
        return q_value + u_value

class MCTS:
    """Monte Carlo Tree Search implementation."""

    def __init__(self, network: NetworkWrapper, num_simulations: int = 100):
        self.network = network
        self.num_simulations = num_simulations
        self.state_encoder = StateEncoder()  # Initialize StateEncoder here as well
        self.logger = logging.getLogger("MCTS")
        self.logger.setLevel(logging.INFO)

    def search(self, state: GameState) -> np.ndarray:
        """Perform MCTS search and return move probabilities."""
        root = Node(state)

        # Ensure move_probs is float32
        move_probs = np.zeros(self.state_encoder.num_positions ** 2 + 2 * self.state_encoder.num_positions,
                              dtype=np.float32)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()

            # Selection
            while node.is_expanded and node.children:
                action = self._select_action(node)
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)

            # Evaluation
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
                            prior_prob=self._get_move_prob(policy, move)
                        )

            # Backpropagation
            self._backpropagate(search_path, value)

        # Calculate move probabilities from visit counts
        valid_moves = state.get_valid_moves()

        if not valid_moves:
            return move_probs

        visit_counts = np.array([
            root.children[move].visit_count
            for move in valid_moves
        ], dtype=np.float32)  # Ensure float32

        total_visits = visit_counts.sum()
        if total_visits > 0:
            visit_probs = visit_counts / total_visits
        else:
            visit_probs = np.ones_like(visit_counts, dtype=np.float32) / len(visit_counts)

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
    """Handles self-play game generation."""

    def __init__(self, network: NetworkWrapper, num_simulations: int = 100, num_workers: int = 4):
        """
        Initialize the SelfPlay instance.

        Args:
            network (NetworkWrapper): The neural network wrapper.
            num_simulations (int): Number of MCTS simulations per move.
            num_workers (int): Number of parallel workers for self-play.
        """
        self.network = network
        self.num_simulations = num_simulations
        self.num_workers = num_workers

        self.mcts = MCTS(network, num_simulations=num_simulations)
        self.state_encoder = StateEncoder()
        self.logger = logging.getLogger("SelfPlay")
        self.logger.setLevel(logging.INFO)

    def generate_games(self, num_games: int = 100) -> List[Tuple[List[np.ndarray], List[np.ndarray], int]]:
        """Generate self-play games in parallel using multiple workers."""
        games = []
        num_simulations = self.num_simulations
        network_params = self.network.network.state_dict()

        # Save the model to a temporary file for workers to load
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            torch.save(network_params, tmp.name)
            model_path = tmp.name

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(play_game_worker, model_path, game_id, num_simulations)
                    for game_id in range(1, num_games + 1)
                ]
                for future in futures:
                    try:
                        result = future.result()
                        games.append(result)
                        self.logger.info("Game completed successfully.")
                    except Exception as e:
                        self.logger.error(f"Error in game generation: {e}")
        finally:
            # Clean up the temporary model file
            os.unlink(model_path)

        return games

    def play_game(self, game_id: int) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Play a single game."""
        state = GameState()
        states = []
        policies = []

        max_moves = 500
        move_count = 0

        # Track game progress metrics
        rings_placed = {Player.WHITE: 0, Player.BLACK: 0}
        markers_placed = {Player.WHITE: 0, Player.BLACK: 0}
        markers_removed = {Player.WHITE: 0, Player.BLACK: 0}
        rings_removed = {Player.WHITE: 0, Player.BLACK: 0}

        self.logger.info(f"\nStarting game {game_id}")
        self.logger.info(f"Initial phase: {state.phase}")
        self.logger.info(f"Initial player: {state.current_player}")

        while not state.is_terminal() and move_count < max_moves:
            self.logger.debug(f"\nMove {move_count}")
            self.logger.debug(f"Current phase: {state.phase}")
            self.logger.debug(f"Current player: {state.current_player}")

            # Get valid moves
            valid_moves = state.get_valid_moves()
            self.logger.debug(f"Found {len(valid_moves)} valid moves")

            if not valid_moves:
                self.logger.error("No valid moves available!")
                break

            # Run MCTS
            move_probs = self.mcts.search(state)

            # Get probabilities for valid moves
            valid_indices = [self.state_encoder.move_to_index(move) for move in valid_moves]
            valid_probs = move_probs[valid_indices]

            self.logger.debug(f"Valid move probabilities sum: {valid_probs.sum()}")

            if valid_probs.sum() == 0:
                self.logger.error("All valid moves have zero probability!")
                break

            # Normalize probabilities
            valid_probs = valid_probs / valid_probs.sum()

            # Select move
            selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
            selected_move = valid_moves[selected_idx]
            self.logger.debug(f"Selected move: {selected_move}")

            # Try to make the move
            success = state.make_move(selected_move)
            if not success:
                self.logger.error(f"Failed to make move: {selected_move}")
                break

            # Store state and policy
            encoded_state = self.state_encoder.encode_state(state)
            states.append(encoded_state)
            policies.append(move_probs)

            # Update metrics
            if selected_move.type == MoveType.PLACE_RING:
                rings_placed[state.current_player] += 1
            elif selected_move.type == MoveType.MOVE_RING:
                markers_placed[state.current_player] += 1
            elif selected_move.type == MoveType.REMOVE_MARKERS:
                markers_removed[state.current_player] += 5
            elif selected_move.type == MoveType.REMOVE_RING:
                rings_removed[state.current_player] += 1

            move_count += 1

        # Log final game state
        self.logger.info(f"\nGame {game_id} completed in {move_count} moves")
        self.logger.debug(f"Final state:\n{state}")
        self.logger.info(f"Final score - White: {state.white_score}, Black: {state.black_score}")

        # Determine outcome
        winner = state.get_winner()
        if winner == Player.WHITE:
            outcome = 1
        elif winner == Player.BLACK:
            outcome = -1
        else:
            outcome = 0

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

def play_game_worker(model_path, game_id, num_simulations):
    """Worker function to play a single game."""
    # Initialize the network in the worker
    device = torch.device('cpu')  # Use 'cuda' or 'mps' if appropriate
    network = NetworkWrapper(model_path=model_path, device=device)
    mcts = MCTS(network, num_simulations=num_simulations)
    state_encoder = StateEncoder()
    state = GameState()
    states = []
    policies = []
    max_moves = 500
    move_count = 0

    while not state.is_terminal() and move_count < max_moves:
        # Run MCTS to get move probabilities
        move_probs = mcts.search(state)
        valid_moves = state.get_valid_moves()

        if not valid_moves:
            break

        # Get probabilities for valid moves
        valid_indices = [state_encoder.move_to_index(move) for move in valid_moves]
        valid_probs = move_probs[valid_indices]

        if valid_probs.sum() == 0:
            break

        # Normalize probabilities
        valid_probs = valid_probs / valid_probs.sum()

        # Select move based on probabilities
        selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
        selected_move = valid_moves[selected_idx]

        # Make the move
        state.make_move(selected_move)

        # Store state and policy
        encoded_state = state_encoder.encode_state(state)
        states.append(encoded_state)
        policies.append(move_probs)

        move_count += 1

    # Determine game outcome
    winner = state.get_winner()
    if winner == Player.WHITE:
        outcome = 1
    elif winner == Player.BLACK:
        outcome = -1
    else:
        outcome = 0

    return states, policies, outcome