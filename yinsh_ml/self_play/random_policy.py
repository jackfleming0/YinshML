"""Random move selection policy for Yinsh self-play."""

import random
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..game import GameState, Move, MoveType, Player, Position, PieceType
from ..game.moves import MoveGenerator

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for the random move policy."""
    rule_based_probability: float = 0.1  # 10% chance of rule-based moves
    random_seed: Optional[int] = None


class RandomMovePolicy:
    """Random move selection policy with optional rule-based moves."""
    
    def __init__(self, config: PolicyConfig = None):
        """Initialize the random move policy.
        
        Args:
            config: Policy configuration. If None, uses default config.
        """
        self.config = config or PolicyConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        logger.info(f"Initialized RandomMovePolicy with rule-based probability: {self.config.rule_based_probability}")
    
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """Select a move for the current game state.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move, or None if no valid moves available
        """
        # Get all valid moves for current state
        valid_moves = MoveGenerator.get_valid_moves(game_state.board, game_state)
        
        if not valid_moves:
            logger.warning("No valid moves available")
            return None
        
        logger.debug(f"Found {len(valid_moves)} valid moves for {game_state.current_player}")
        
        # Decide between random and rule-based move
        if random.random() < self.config.rule_based_probability:
            move = self._select_rule_based_move(valid_moves, game_state)
            if move:
                logger.debug(f"Selected rule-based move: {move}")
                return move
        
        # Fall back to random selection
        move = random.choice(valid_moves)
        logger.debug(f"Selected random move: {move}")
        return move
    
    def _select_rule_based_move(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Select a rule-based move from valid moves.
        
        Args:
            valid_moves: List of valid moves
            game_state: Current game state
            
        Returns:
            Rule-based move if found, None otherwise
        """
        if not valid_moves:
            return None
        
        # Rule-based strategies based on game phase
        if game_state.phase.value == 0:  # RING_PLACEMENT
            return self._rule_based_ring_placement(valid_moves, game_state)
        elif game_state.phase.value == 1:  # MAIN_GAME
            return self._rule_based_ring_movement(valid_moves, game_state)
        elif game_state.phase.value == 2:  # ROW_COMPLETION
            return self._rule_based_marker_removal(valid_moves, game_state)
        elif game_state.phase.value == 3:  # RING_REMOVAL
            return self._rule_based_ring_removal(valid_moves, game_state)
        
        # Fall back to random if no rule applies
        return random.choice(valid_moves)
    
    def _rule_based_ring_placement(self, valid_moves: List[Move], game_state: GameState) -> Move:
        """Rule-based ring placement strategy.
        
        Prefer center positions and avoid edges early in the game.
        """
        # Filter moves by type
        ring_placement_moves = [m for m in valid_moves if m.type == MoveType.PLACE_RING]
        if not ring_placement_moves:
            return random.choice(valid_moves)
        
        # Score positions based on centrality
        scored_moves = []
        for move in ring_placement_moves:
            score = self._calculate_position_score(move.source)
            scored_moves.append((score, move))
        
        # Sort by score (higher is better) and select from top candidates
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Select from top 30% of moves
        top_count = max(1, len(scored_moves) // 3)
        top_moves = [move for _, move in scored_moves[:top_count]]
        
        return random.choice(top_moves)
    
    def _rule_based_ring_movement(self, valid_moves: List[Move], game_state: GameState) -> Move:
        """Rule-based ring movement strategy.
        
        Prefer moves that create or extend marker lines.
        """
        # Filter moves by type
        ring_movement_moves = [m for m in valid_moves if m.type == MoveType.MOVE_RING]
        if not ring_movement_moves:
            return random.choice(valid_moves)
        
        # Score moves based on potential marker line creation
        scored_moves = []
        for move in ring_movement_moves:
            score = self._calculate_movement_score(move, game_state)
            scored_moves.append((score, move))
        
        # Sort by score and select from top candidates
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Select from top 40% of moves
        top_count = max(1, len(scored_moves) // 2)
        top_moves = [move for _, move in scored_moves[:top_count]]
        
        return random.choice(top_moves)
    
    def _rule_based_marker_removal(self, valid_moves: List[Move], game_state: GameState) -> Move:
        """Rule-based marker removal strategy.
        
        Prefer removing markers that create the longest lines.
        """
        # Filter moves by type
        marker_removal_moves = [m for m in valid_moves if m.type == MoveType.REMOVE_MARKERS]
        if not marker_removal_moves:
            return random.choice(valid_moves)
        
        # Score moves based on line length
        scored_moves = []
        for move in marker_removal_moves:
            score = len(move.markers) if move.markers else 0
            scored_moves.append((score, move))
        
        # Sort by score and select from top candidates
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Select from top 50% of moves
        top_count = max(1, len(scored_moves) // 2)
        top_moves = [move for _, move in scored_moves[:top_count]]
        
        return random.choice(top_moves)
    
    def _rule_based_ring_removal(self, valid_moves: List[Move], game_state: GameState) -> Move:
        """Rule-based ring removal strategy.
        
        Prefer removing rings that are less central.
        """
        # Filter moves by type
        ring_removal_moves = [m for m in valid_moves if m.type == MoveType.REMOVE_RING]
        if not ring_removal_moves:
            return random.choice(valid_moves)
        
        # Score moves based on ring centrality (lower is better for removal)
        scored_moves = []
        for move in ring_removal_moves:
            score = -self._calculate_position_score(move.source)  # Negative for removal
            scored_moves.append((score, move))
        
        # Sort by score and select from top candidates
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Select from top 50% of moves
        top_count = max(1, len(scored_moves) // 2)
        top_moves = [move for _, move in scored_moves[:top_count]]
        
        return random.choice(top_moves)
    
    def _calculate_position_score(self, position: Position) -> float:
        """Calculate a score for a position based on centrality.
        
        Args:
            position: Board position
            
        Returns:
            Score (higher is more central)
        """
        # Convert position to coordinates (approximate)
        col = ord(position.column) - ord('A')
        row = position.row - 1
        
        # Calculate distance from center (approximate center at D4)
        center_col = 3  # D
        center_row = 3  # 4
        
        distance = ((col - center_col) ** 2 + (row - center_row) ** 2) ** 0.5
        
        # Return inverse distance (higher is better)
        return 1.0 / (1.0 + distance)
    
    def _calculate_movement_score(self, move: Move, game_state: GameState) -> float:
        """Calculate a score for a ring movement based on potential marker line creation.
        
        Args:
            move: The move to score
            game_state: Current game state
            
        Returns:
            Score (higher is better)
        """
        if not move.source or not move.destination:
            return 0.0
        
        # Simple heuristic: prefer moves that go towards existing markers
        source_piece = game_state.board.get_piece(move.source)
        if not source_piece or not source_piece.is_ring():
            return 0.0
        
        # Check if destination is near existing markers of the same color
        player = source_piece.get_player()
        if player is None:
            return 0.0
            
        player_marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        nearby_markers = 0
        
        # Check positions around destination
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                try:
                    check_pos = Position(
                        chr(ord(move.destination.column) + dx),
                        move.destination.row + dy
                    )
                    piece = game_state.board.get_piece(check_pos)
                    if piece == player_marker_type:
                        nearby_markers += 1
                except (ValueError, OverflowError):
                    # Invalid position
                    continue
        
        return nearby_markers
    
    def get_move_statistics(self, game_state: GameState) -> dict:
        """Get statistics about available moves.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with move statistics
        """
        valid_moves = MoveGenerator.get_valid_moves(game_state.board, game_state)
        
        if not valid_moves:
            return {
                'total_moves': 0,
                'move_types': {},
                'rule_based_probability': self.config.rule_based_probability
            }
        
        # Count moves by type
        move_types = {}
        for move in valid_moves:
            move_type = move.type.value
            move_types[move_type] = move_types.get(move_type, 0) + 1
        
        return {
            'total_moves': len(valid_moves),
            'move_types': move_types,
            'rule_based_probability': self.config.rule_based_probability,
            'current_phase': game_state.phase.value,
            'current_player': game_state.current_player.value
        }
