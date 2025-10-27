"""Move selection policies for self-play."""

import random
import logging
from typing import List, Optional
from dataclasses import dataclass

from ..game.types import Move, MoveType, GamePhase
from ..game.constants import Player, Position
from ..game.board import Board
from ..game.game_state import GameState

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for move selection policies."""
    rule_based_probability: float = 0.1  # 10% chance of using rule-based moves
    random_seed: Optional[int] = None


class RandomMovePolicy:
    """Random move selection policy for self-play."""
    
    def __init__(self, config: PolicyConfig = None):
        """Initialize the random move policy.
        
        Args:
            config: Policy configuration
        """
        self.config = config or PolicyConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        logger.info(f"Initialized RandomMovePolicy with rule-based probability: {self.config.rule_based_probability}")
    
    def select_move(self, game_state: GameState) -> Move:
        """Select a move using random policy with optional rule-based moves.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move
            
        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Use rule-based move with configured probability
        if random.random() < self.config.rule_based_probability:
            move = self._select_rule_based_move(valid_moves, game_state)
            if move is not None:
                logger.debug(f"Selected rule-based move: {move}")
                return move
        
        # Fall back to random selection
        move = random.choice(valid_moves)
        logger.debug(f"Selected random move: {move}")
        return move
    
    def _select_rule_based_move(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Select a move using simple rule-based heuristics.
        
        Args:
            valid_moves: List of valid moves
            game_state: Current game state
            
        Returns:
            Rule-based move if found, None otherwise
        """
        if not valid_moves:
            return None
        
        # Rule-based strategies based on game phase
        if game_state.phase == GamePhase.RING_PLACEMENT:
            return self._rule_based_ring_placement(valid_moves, game_state)
        elif game_state.phase == GamePhase.MAIN_GAME:
            return self._rule_based_ring_movement(valid_moves, game_state)
        elif game_state.phase == GamePhase.ROW_COMPLETION:
            return self._rule_based_marker_removal(valid_moves, game_state)
        elif game_state.phase == GamePhase.RING_REMOVAL:
            return self._rule_based_ring_removal(valid_moves, game_state)
        
        return None
    
    def _rule_based_ring_placement(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based ring placement strategy.
        
        Prefer center positions and avoid edges early in the game.
        """
        # Filter moves by type
        ring_placement_moves = [m for m in valid_moves if m.type == MoveType.PLACE_RING]
        if not ring_placement_moves:
            return None
        
        # Prefer center positions (D4-E5-F6 area)
        center_moves = []
        edge_moves = []
        
        for move in ring_placement_moves:
            pos = move.source
            if pos is None:
                continue
                
            # Check if position is in center area
            if self._is_center_position(pos):
                center_moves.append(move)
            elif self._is_edge_position(pos):
                edge_moves.append(move)
        
        # Prefer center moves, fall back to non-edge moves, then any move
        if center_moves:
            return random.choice(center_moves)
        elif len(ring_placement_moves) > len(edge_moves):
            non_edge_moves = [m for m in ring_placement_moves if m not in edge_moves]
            if non_edge_moves:
                return random.choice(non_edge_moves)
        
        return random.choice(ring_placement_moves)
    
    def _rule_based_ring_movement(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based ring movement strategy.
        
        Prefer moves that create or block potential runs.
        """
        # Filter moves by type
        ring_moves = [m for m in valid_moves if m.type == MoveType.MOVE_RING]
        if not ring_moves:
            return None
        
        # Simple heuristic: prefer moves that advance toward opponent's side
        opponent = Player.BLACK if game_state.current_player == Player.WHITE else Player.WHITE
        
        scoring_moves = []
        for move in ring_moves:
            if move.destination is None:
                continue
            
            score = self._score_ring_move(move, game_state, opponent)
            scoring_moves.append((score, move))
        
        if scoring_moves:
            # Sort by score (higher is better) and pick from top moves
            scoring_moves.sort(key=lambda x: x[0], reverse=True)
            top_moves = [m for score, m in scoring_moves if score == scoring_moves[0][0]]
            return random.choice(top_moves)
        
        return random.choice(ring_moves)
    
    def _rule_based_marker_removal(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based marker removal strategy.
        
        Prefer removing opponent's markers when possible.
        """
        # Filter moves by type
        marker_moves = [m for m in valid_moves if m.type == MoveType.REMOVE_MARKERS]
        if not marker_moves:
            return None
        
        # Simple heuristic: prefer moves that remove opponent's markers
        opponent = Player.BLACK if game_state.current_player == Player.WHITE else Player.WHITE
        
        opponent_marker_moves = []
        for move in marker_moves:
            if move.markers is None:
                continue
            
            # Count opponent markers in this move
            opponent_count = sum(1 for pos in move.markers 
                              if game_state.board.get_piece(pos) == 
                              (PieceType.BLACK_MARKER if opponent == Player.BLACK else PieceType.WHITE_MARKER))
            
            if opponent_count > 0:
                opponent_marker_moves.append((opponent_count, move))
        
        if opponent_marker_moves:
            # Sort by opponent marker count and pick from top moves
            opponent_marker_moves.sort(key=lambda x: x[0], reverse=True)
            top_moves = [m for count, m in opponent_marker_moves 
                        if count == opponent_marker_moves[0][0]]
            return random.choice(top_moves)
        
        return random.choice(marker_moves)
    
    def _rule_based_ring_removal(self, valid_moves: List[Move], game_state: GameState) -> Optional[Move]:
        """Rule-based ring removal strategy.
        
        Prefer removing rings that are less strategically important.
        """
        # Filter moves by type
        ring_removal_moves = [m for m in valid_moves if m.type == MoveType.REMOVE_RING]
        if not ring_removal_moves:
            return None
        
        # Simple heuristic: prefer removing rings from edge positions
        edge_moves = []
        center_moves = []
        
        for move in ring_removal_moves:
            if move.source is None:
                continue
            
            if self._is_edge_position(move.source):
                edge_moves.append(move)
            elif self._is_center_position(move.source):
                center_moves.append(move)
        
        # Prefer edge moves, fall back to center moves, then any move
        if edge_moves:
            return random.choice(edge_moves)
        elif center_moves:
            return random.choice(center_moves)
        
        return random.choice(ring_removal_moves)
    
    def _is_center_position(self, pos: Position) -> bool:
        """Check if position is in the center area of the board."""
        # Center area: D4-E5-F6 region
        center_positions = {
            'D4', 'D5', 'D6',
            'E4', 'E5', 'E6', 'E7',
            'F4', 'F5', 'F6', 'F7', 'F8'
        }
        return str(pos) in center_positions
    
    def _is_edge_position(self, pos: Position) -> bool:
        """Check if position is on the edge of the board."""
        # Edge positions are those with minimal neighbors
        edge_positions = {
            'A2', 'A3', 'A4', 'A5',
            'B1', 'B7',
            'C1', 'C8',
            'D1', 'D9',
            'E1', 'E10',
            'F2', 'F10',
            'G2', 'G11',
            'H3', 'H11',
            'I4', 'I11',
            'J5', 'J11',
            'K7', 'K8', 'K9', 'K10'
        }
        return str(pos) in edge_positions
    
    def _score_ring_move(self, move: Move, game_state: GameState, opponent: Player) -> int:
        """Score a ring move based on simple heuristics.
        
        Args:
            move: The move to score
            game_state: Current game state
            opponent: Opponent player
            
        Returns:
            Score for the move (higher is better)
        """
        if move.destination is None:
            return 0
        
        score = 0
        
        # Prefer moves toward opponent's side
        if opponent == Player.BLACK:
            # Black starts at bottom, so prefer moves toward higher rows
            if move.destination.row > 6:
                score += 2
        else:
            # White starts at top, so prefer moves toward lower rows
            if move.destination.row < 6:
                score += 2
        
        # Prefer center positions
        if self._is_center_position(move.destination):
            score += 1
        
        # Avoid edge positions
        if self._is_edge_position(move.destination):
            score -= 1
        
        return score


class PolicyFactory:
    """Factory for creating move selection policies."""
    
    @staticmethod
    def create_random_policy(config: PolicyConfig = None) -> RandomMovePolicy:
        """Create a random move policy.
        
        Args:
            config: Policy configuration
            
        Returns:
            Random move policy instance
        """
        return RandomMovePolicy(config)
    
    @staticmethod
    def create_policy(policy_type: str, config: PolicyConfig = None) -> RandomMovePolicy:
        """Create a policy by type.
        
        Args:
            policy_type: Type of policy to create
            config: Policy configuration
            
        Returns:
            Policy instance
            
        Raises:
            ValueError: If policy type is not supported
        """
        if policy_type == "random":
            return PolicyFactory.create_random_policy(config)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
