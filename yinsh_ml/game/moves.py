"""Move generation and validation for YINSH."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .constants import Position, Player, PieceType, is_valid_position
from .types import Move, MoveType, GamePhase
import logging

# Setup logger
logger = logging.getLogger(__name__)


class MoveGenerator:
    """Generates and validates moves for YINSH."""

    def get_valid_moves(board: 'Board', game_state: 'GameState') -> List[Move]:
        """Get all valid moves for the current game state."""
        logger.debug("\nEntering get_valid_moves")
        phase = game_state.phase
        player = game_state.current_player

        logger.debug(f"Finding moves for {player} in phase {phase}")
        logger.debug(f"Phase type: {type(phase)}")
        logger.debug(f"Phase value: {phase.value}")

        moves = []
        if phase.value == GamePhase.RING_PLACEMENT.value:
            logger.debug("Getting ring placement moves")
            moves = MoveGenerator._get_ring_placement_moves(board, player)
        elif phase.value == GamePhase.MAIN_GAME.value:
            logger.debug("Getting ring movement moves")
            moves = MoveGenerator._get_ring_movement_moves(board, player)
        elif phase.value == GamePhase.ROW_COMPLETION.value:
            logger.debug("Getting marker removal moves")
            moves = MoveGenerator._get_marker_removal_moves(board, player)
        elif phase.value == GamePhase.RING_REMOVAL.value:
            logger.debug("Getting ring removal moves")
            moves = MoveGenerator._get_ring_removal_moves(board, player)

        logger.debug(f"Found {len(moves)} valid moves")
        return moves

    @staticmethod
    def _get_ring_placement_moves(board: 'Board', player: Player) -> List[Move]:
        """Get all valid ring placement moves."""
        logger.debug("Starting ring placement generation")
        moves = []
        valid_count = 0
        empty_count = 0

        logger.debug(f"Board state:\n{board}")

        # Check each position
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                logger.debug(f"\nChecking {pos}")
                if is_valid_position(pos):
                    valid_count += 1
                    logger.debug(f"Valid position")
                    if board.is_empty(pos):
                        empty_count += 1
                        logger.debug(f"Empty position - creating move")
                        moves.append(Move(
                            type=MoveType.PLACE_RING,
                            player=player,
                            source=pos
                        ))
                else:
                    logger.debug(f"Invalid position")

        logger.debug(f"\nRing placement generation complete:")

        return moves

    @staticmethod
    def _get_ring_movement_moves(board: 'Board', player: Player) -> List[Move]:
        """Get all valid ring movement moves."""
        moves = []
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING

        # Find all rings of the current player
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if not is_valid_position(pos):
                    continue

                piece = board.get_piece(pos)
                if piece == ring_type:
                    # Get valid destinations for this ring
                    destinations = board.valid_move_positions(pos)
                    for dest in destinations:
                        moves.append(Move(
                            type=MoveType.MOVE_RING,
                            player=player,
                            source=pos,
                            destination=dest
                        ))

        return moves

    @staticmethod
    def _get_marker_removal_moves(board: 'Board', player: Player) -> List[Move]:
        """Get all valid marker removal moves.

        Real YINSH: when a player completes a row of 5+ same-color
        markers, they choose ANY 5 consecutive markers from that run to
        remove. ``find_marker_rows`` now returns the full run of
        contiguous markers along a hex axis (length 5, 6, or 7), and
        this function enumerates every length-5 window across it.

        Windows are sorted into a canonical (column, row) order before
        deduping via a ``frozenset`` so the same 5-window reachable from
        overlapping 6/7-runs on the same axis is only emitted once.
        """
        moves = []
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER

        rows = board.find_marker_rows(marker_type)
        logger.debug(f"Found {len(rows)} rows of {marker_type}")

        seen_windows = set()
        for row in rows:
            if len(row.positions) < 5:
                continue
            # row.positions is sorted (column, row); every 5-consecutive
            # slice along the run is a legal REMOVE_MARKERS candidate.
            for i in range(len(row.positions) - 4):
                window = tuple(row.positions[i:i + 5])
                key = frozenset(window)
                if key in seen_windows:
                    continue
                seen_windows.add(key)
                logger.debug(
                    f"Creating move with markers: {[str(m) for m in window]}"
                )
                moves.append(Move(
                    type=MoveType.REMOVE_MARKERS,
                    player=player,
                    markers=window,
                ))

        logger.debug(f"Generated {len(moves)} valid marker removal moves")
        return moves

    @staticmethod
    def _get_ring_removal_moves(board: 'Board', player: Player) -> List[Move]:
        """Get all valid ring removal moves."""
        moves = []
        ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING

        # Get all positions with player's rings
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if not is_valid_position(pos):
                    continue
                if board.get_piece(pos) == ring_type:
                    moves.append(Move(
                        type=MoveType.REMOVE_RING,
                        player=player,
                        source=pos
                    ))

        return moves

    @staticmethod
    def validate_move(move: Move, board: 'Board', game_state: 'GameState') -> bool:
        """Validate if a move is legal in the current game state."""
        if move.player != game_state.current_player:
            return False

        if move.type == MoveType.PLACE_RING:
            return (game_state.phase == GamePhase.RING_PLACEMENT and
                    is_valid_position(move.source) and
                    board.is_empty(move.source))

        elif move.type == MoveType.MOVE_RING:
            return (game_state.phase == GamePhase.MAIN_GAME and
                    board.is_valid_ring_move(move.source, move.destination))

        elif move.type == MoveType.REMOVE_MARKERS:
            if game_state.phase != GamePhase.ROW_COMPLETION:
                return False
            if not move.markers or len(move.markers) != 5:
                return False
            return board.is_valid_marker_sequence(move.markers, move.player)

        elif move.type == MoveType.REMOVE_RING:
            if game_state.phase != GamePhase.RING_REMOVAL:
                return False
            ring_type = PieceType.WHITE_RING if move.player == Player.WHITE else PieceType.BLACK_RING
            return board.get_piece(move.source) == ring_type

        return False

    @staticmethod
    def get_affected_positions(move: Move) -> Set[Position]:
        """Get all board positions affected by a move."""
        affected = set()

        if move.type == MoveType.PLACE_RING:
            affected.add(move.source)

        elif move.type == MoveType.MOVE_RING:
            affected.add(move.source)
            affected.add(move.destination)
            # Add positions of markers that would be flipped
            affected.update(MoveGenerator._get_flipped_markers(move.source, move.destination))

        elif move.type == MoveType.REMOVE_MARKERS:
            affected.update(move.markers)

        elif move.type == MoveType.REMOVE_RING:
            affected.add(move.source)

        return affected

    @staticmethod
    def _get_flipped_markers(start: Position, end: Position) -> Set[Position]:
        """Get positions of markers that would be flipped by a ring move."""
        flipped = set()

        # Calculate direction of movement
        col_diff = ord(end.column) - ord(start.column)
        row_diff = end.row - start.row

        if col_diff == 0 and row_diff == 0:
            return flipped

        # Normalize direction
        steps = max(abs(col_diff), abs(row_diff))
        direction = (
            col_diff // steps if col_diff != 0 else 0,
            row_diff // steps if row_diff != 0 else 0
        )

        # Get all positions along the movement path
        current = start
        for _ in range(steps - 1):
            current = get_next_position(current, direction)
            if current:
                flipped.add(current)

        return flipped
