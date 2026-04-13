"""Validate scraped YINSH games by replaying through GameState.

Ensures every move in a scraped game is legal, catching parsing bugs
that would silently corrupt training data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..game.game_state import GameState
from ..game.constants import Player, Position
from ..game.types import Move, MoveType, GamePhase

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validating a single game."""

    def __init__(self, game_id: str, source: str):
        self.game_id = game_id
        self.source = source
        self.valid = True
        self.total_moves = 0
        self.error_move = -1
        self.error_message = ""
        self.warnings: List[str] = []

    def __repr__(self):
        status = "VALID" if self.valid else f"INVALID at move {self.error_move}"
        return f"ValidationResult({self.game_id}: {status}, {self.total_moves} moves)"


class GameValidator:
    """Validates scraped games by replaying through the game engine.

    Usage:
        validator = GameValidator()
        result = validator.validate_game(game_data)
        if result.valid:
            # game is safe for training
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, reject games on any warning. If False, only
                    reject on illegal moves.
        """
        self.strict = strict

    def validate_game(self, game_data: dict) -> ValidationResult:
        """Validate a single game by replaying all moves.

        Args:
            game_data: Standardized game dict with 'moves', 'result', etc.

        Returns:
            ValidationResult with validity status and error details.
        """
        game_id = game_data.get('game_id', 'unknown')
        source = game_data.get('source', 'unknown')
        result = ValidationResult(game_id, source)

        moves = game_data.get('moves', [])
        game_result = game_data.get('result')

        if not moves:
            result.valid = False
            result.error_message = "No moves in game"
            return result

        gs = GameState()
        result.total_moves = len(moves)

        for i, move_data in enumerate(moves):
            move = self._parse_move(move_data, gs)
            if move is None:
                result.valid = False
                result.error_move = i
                result.error_message = f"Failed to parse move: {move_data}"
                return result

            # Check the move is legal
            valid_moves = gs.get_valid_moves()
            if not self._move_in_list(move, valid_moves):
                result.valid = False
                result.error_move = i
                result.error_message = (
                    f"Illegal move at step {i}: {move} "
                    f"(phase={gs.phase.name}, player={gs.current_player.name})"
                )
                return result

            if not gs.make_move(move):
                result.valid = False
                result.error_move = i
                result.error_message = f"make_move returned False for: {move}"
                return result

        # Resolve result. If the scraper couldn't parse RE, derive it from
        # final game state. This is necessary for ~70% of Boardspace SGFs
        # which predate the RE property.
        actual_winner = gs.get_winner()

        if game_result in ('white', 'black'):
            expected_winner = Player.WHITE if game_result == 'white' else Player.BLACK
            if actual_winner != expected_winner:
                result.warnings.append(
                    f"Expected winner {game_result} but got "
                    f"{actual_winner.name if actual_winner else 'None'}"
                )
                if self.strict:
                    result.valid = False
                    result.error_message = result.warnings[-1]
        elif game_result == 'draw':
            if actual_winner is not None:
                result.warnings.append(
                    f"Expected draw but got winner {actual_winner.name}"
                )
                if self.strict:
                    result.valid = False
                    result.error_message = result.warnings[-1]
        else:
            if actual_winner == Player.WHITE:
                game_data['result'] = 'white'
            elif actual_winner == Player.BLACK:
                game_data['result'] = 'black'
            else:
                result.valid = False
                result.error_message = (
                    f"Unresolved result: no RE in source and final state "
                    f"has no winner (likely truncated game)"
                )
                return result

        return result

    def validate_file(self, path: str) -> List[ValidationResult]:
        """Validate all games in a JSON file."""
        with open(path) as f:
            data = json.load(f)

        games = data if isinstance(data, list) else [data]
        results = []
        for game in games:
            r = self.validate_game(game)
            results.append(r)
            if r.valid:
                logger.info(f"VALID: {r.game_id} ({r.total_moves} moves)")
            else:
                logger.warning(f"INVALID: {r.game_id} - {r.error_message}")

        valid_count = sum(1 for r in results if r.valid)
        logger.info(f"Validated {len(results)} games: "
                    f"{valid_count} valid, {len(results) - valid_count} invalid")
        return results

    def validate_directory(self, directory: str,
                           min_rating: int = 0) -> Dict[str, List[ValidationResult]]:
        """Validate all JSON files in a directory.

        Returns:
            Dict mapping filenames to lists of ValidationResults.
        """
        path = Path(directory)
        all_results = {}

        for json_file in sorted(path.glob('*.json')):
            results = self.validate_file(str(json_file))

            if min_rating > 0:
                # Re-check rating filter
                with open(json_file) as f:
                    data = json.load(f)
                games = data if isinstance(data, list) else [data]
                filtered = []
                for game, result in zip(games, results):
                    players = game.get('players', {})
                    w_rating = players.get('white', {}).get('rating', 0)
                    b_rating = players.get('black', {}).get('rating', 0)
                    if w_rating >= min_rating and b_rating >= min_rating:
                        filtered.append(result)
                results = filtered

            all_results[json_file.name] = results

        total = sum(len(r) for r in all_results.values())
        valid = sum(1 for rs in all_results.values() for r in rs if r.valid)
        logger.info(f"Directory validation: {valid}/{total} games valid")
        return all_results

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

    @staticmethod
    def _move_in_list(move: Move, valid_moves: List[Move]) -> bool:
        """Check if a move matches any in the valid moves list.

        Handles the case where marker removal order may differ.
        """
        for vm in valid_moves:
            if vm.type != move.type or vm.player != move.player:
                continue
            if move.type == MoveType.REMOVE_MARKERS:
                if (move.markers and vm.markers and
                        set(move.markers) == set(vm.markers)):
                    return True
            else:
                if vm.source == move.source and vm.destination == move.destination:
                    return True
        return False
