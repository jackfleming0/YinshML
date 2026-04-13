"""Convert standardized game JSON to training pairs (state tensor, policy, value).

Takes the standardized game format and produces training data suitable
for supervised pre-training of the neural network.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from ..game.game_state import GameState
from ..game.constants import Player, Position
from ..game.types import Move, MoveType, GamePhase
from ..utils.encoding import StateEncoder

logger = logging.getLogger(__name__)


class GameConverter:
    """Converts standardized game JSON into training pairs.

    Each position in a game produces:
    - state: np.ndarray of shape (6, 11, 11)
    - policy: np.ndarray of shape (total_moves,) — one-hot at the expert move
    - value: float — game outcome from current player's perspective
    """

    def __init__(self, encoder: Optional[StateEncoder] = None):
        self.encoder = encoder or StateEncoder()

    def convert_game(self, game_data: dict) -> List[Dict]:
        """Convert a single game to training pairs.

        Args:
            game_data: Standardized game dict with 'moves', 'result', etc.

        Returns:
            List of dicts with keys 'state', 'policy', 'value', 'metadata'.
        """
        result = game_data.get('result')  # 'white', 'black', or 'draw'
        moves = game_data.get('moves', [])

        if not moves or result not in ('white', 'black', 'draw'):
            logger.warning(f"Skipping game {game_data.get('game_id', '?')}: "
                          f"invalid result '{result}' or no moves")
            return []

        gs = GameState()
        training_pairs = []

        for move_data in moves:
            move = self._parse_move(move_data, gs)
            if move is None:
                logger.warning(f"Failed to parse move in game "
                              f"{game_data.get('game_id', '?')}: {move_data}")
                return []  # Abort on parse failure

            # Record training pair BEFORE making the move
            state = self.encoder.encode_state(gs)
            policy = self._make_policy(move)
            value = self._outcome_value(result, gs.current_player)

            training_pairs.append({
                'state': state,
                'policy': policy,
                'value': value,
                'metadata': {
                    'game_id': game_data.get('game_id', ''),
                    'move_num': len(gs.move_history),
                    'phase': gs.phase.name,
                }
            })

            if not gs.make_move(move):
                logger.warning(f"Illegal move in game "
                              f"{game_data.get('game_id', '?')}: {move}")
                return []  # Abort — data is corrupted

        return training_pairs

    def convert_file(self, path: str) -> List[Dict]:
        """Convert a JSON file (single game or list of games) to training pairs."""
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, list):
            games = data
        else:
            games = [data]

        all_pairs = []
        for game_data in games:
            pairs = self.convert_game(game_data)
            if pairs:
                all_pairs.extend(pairs)
                logger.info(f"Converted game {game_data.get('game_id', '?')}: "
                           f"{len(pairs)} positions")
            else:
                logger.warning(f"Failed to convert game "
                              f"{game_data.get('game_id', '?')}")

        return all_pairs

    def convert_directory(self, directory: str,
                          min_rating: int = 0) -> List[Dict]:
        """Convert all JSON files in a directory."""
        path = Path(directory)
        all_pairs = []

        for json_file in sorted(path.glob('*.json')):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                games = data if isinstance(data, list) else [data]
                for game in games:
                    # Quality filter
                    if min_rating > 0:
                        players = game.get('players', {})
                        w_rating = players.get('white', {}).get('rating', 0)
                        b_rating = players.get('black', {}).get('rating', 0)
                        if w_rating < min_rating or b_rating < min_rating:
                            continue

                    pairs = self.convert_game(game)
                    all_pairs.extend(pairs)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error processing {json_file}: {e}")

        logger.info(f"Converted {len(all_pairs)} total positions from {directory}")
        return all_pairs

    def save_training_data(self, pairs: List[Dict], output_path: str):
        """Save training pairs as compressed numpy arrays."""
        if not pairs:
            logger.warning("No training pairs to save")
            return

        states = np.array([p['state'] for p in pairs], dtype=np.float32)
        policies = np.array([p['policy'] for p in pairs], dtype=np.float32)
        values = np.array([p['value'] for p in pairs], dtype=np.float32)

        np.savez_compressed(
            output_path,
            states=states,
            policies=policies,
            values=values,
        )
        logger.info(f"Saved {len(pairs)} training pairs to {output_path}")

    @staticmethod
    def load_training_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load training pairs from a .npz file.

        Returns:
            (states, policies, values) arrays.
        """
        data = np.load(path)
        return data['states'], data['policies'], data['values']

    def _parse_move(self, move_data: dict, gs: GameState) -> Optional[Move]:
        """Parse a standardized move dict into a Move object."""
        move_type_str = move_data.get('move_type', '')
        player_str = move_data.get('player', '')
        player = Player.WHITE if player_str == 'white' else Player.BLACK

        try:
            if move_type_str == 'PLACE_RING':
                pos = Position.from_string(move_data['position'])
                return Move(type=MoveType.PLACE_RING, player=player, source=pos)

            elif move_type_str == 'MOVE_RING':
                src = Position.from_string(move_data['source'])
                dst = Position.from_string(move_data['destination'])
                return Move(type=MoveType.MOVE_RING, player=player,
                           source=src, destination=dst)

            elif move_type_str == 'REMOVE_MARKERS':
                markers = tuple(Position.from_string(p)
                               for p in move_data['markers'])
                return Move(type=MoveType.REMOVE_MARKERS, player=player,
                           markers=markers)

            elif move_type_str == 'REMOVE_RING':
                pos = Position.from_string(move_data['position'])
                return Move(type=MoveType.REMOVE_RING, player=player,
                           source=pos)

        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing move: {e}")
            return None

        return None

    def _make_policy(self, move: Move) -> np.ndarray:
        """Create a one-hot policy vector for the given move."""
        policy = np.zeros(self.encoder.total_moves, dtype=np.float32)
        try:
            idx = self.encoder.move_to_index(move)
            policy[idx] = 1.0
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not encode move {move}: {e}")
        return policy

    @staticmethod
    def _outcome_value(result: str, current_player: Player) -> float:
        """Convert game outcome to value from current player's perspective.

        Returns:
            +1.0 if current player won, -1.0 if lost, 0.0 for draw.
        """
        if result == 'draw':
            return 0.0
        winner = Player.WHITE if result == 'white' else Player.BLACK
        return 1.0 if current_player == winner else -1.0
