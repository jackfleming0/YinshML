"""Feature extraction functions for YINSH heuristic evaluation.

This module contains all feature extraction functions used by the heuristic
evaluator. Features are extracted from game states and returned as differential
scores (my_value - opponent_value) for use in position evaluation.

Features are based on statistical analysis of 100K+ games, with correlation
validated patterns including:
- Completed runs differential (r=0.116)
- Potential runs count (r=0.109)
- Connected marker chains (r=0.071)
- Ring positioning and spread
- Board control and territorial advantage

All feature extraction functions follow the pattern:
    feature_value = my_player_value - opponent_value

This ensures all features are differential and can be directly combined
with weights for position evaluation.
"""

from typing import Dict, List, Tuple, Set
from ..game.game_state import GameState
from ..game.constants import Player, PieceType, MARKERS_FOR_ROW


def completed_runs_differential(game_state: GameState, player: Player) -> int:
    """Calculate differential of completed runs between players.
    
    This feature measures the difference in completed 5-marker rows
    between the specified player and their opponent. Completed runs
    are highly correlated with winning (r=0.116).
    
    A completed run is a row of exactly 5 consecutive markers of the
    same color in a line (horizontal, vertical, or diagonal).
    
    Args:
        game_state: The current game state
        player: The player to calculate differential for (my_player)
        
    Returns:
        Integer differential: my_completed_runs - opponent_completed_runs
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Get marker types for both players
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Find all marker rows for both players
    my_rows = game_state.board.find_marker_rows(my_marker)
    opponent_rows = game_state.board.find_marker_rows(opponent_marker)
    
    # Count completed runs (rows with exactly 5 markers)
    my_completed_runs = sum(1 for row in my_rows if row.length >= MARKERS_FOR_ROW)
    opponent_completed_runs = sum(1 for row in opponent_rows if row.length >= MARKERS_FOR_ROW)
    
    # Return differential
    return my_completed_runs - opponent_completed_runs


def potential_runs_count(game_state: GameState, player: Player) -> int:
    """Count potential runs (near-complete rows) for the player.
    
    This feature counts how many potential 5-marker rows exist for
    the specified player. Potential runs are indicators of tactical
    opportunities and are correlated with success (r=0.109).
    
    A potential run is a row of 3 or 4 consecutive markers that could
    become a completed run with 1-2 more markers.
    
    Args:
        game_state: The current game state
        player: The player to count potential runs for
        
    Returns:
        Integer differential: my_potential_runs - opponent_potential_runs
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Get marker types for both players
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Find all marker rows for both players
    my_rows = game_state.board.find_marker_rows(my_marker)
    opponent_rows = game_state.board.find_marker_rows(opponent_marker)
    
    # Count potential runs (rows with 3 or 4 markers, but not yet completed)
    min_potential = 3
    my_potential = sum(
        1 for row in my_rows 
        if min_potential <= row.length < MARKERS_FOR_ROW
    )
    opponent_potential = sum(
        1 for row in opponent_rows 
        if min_potential <= row.length < MARKERS_FOR_ROW
    )
    
    # Return differential
    return my_potential - opponent_potential


def connected_marker_chains(game_state: GameState, player: Player) -> int:
    """Calculate connected marker chain length differential.
    
    This feature measures the difference in connected marker chain
    lengths between players. Connected chains represent strategic
    positioning and are correlated with winning (r=0.071).
    
    A connected chain is a group of markers that are adjacent to each
    other (horizontally, vertically, or diagonally). This function finds
    the longest such chain for each player.
    
    Args:
        game_state: The current game state
        player: The player to calculate differential for
        
    Returns:
        Integer differential: my_longest_chain_length - opponent_longest_chain_length
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Get marker types for both players
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Find longest chain for each player using DFS
    my_longest_chain = _find_longest_chain(game_state.board, my_marker)
    opponent_longest_chain = _find_longest_chain(game_state.board, opponent_marker)
    
    # Return differential
    return my_longest_chain - opponent_longest_chain


def _find_longest_chain(board, marker_type: PieceType) -> int:
    """Find the length of the longest connected marker chain.
    
    Uses depth-first search to find connected components of markers.
    
    Args:
        board: The board to search
        marker_type: The type of marker to search for
        
    Returns:
        Length of the longest connected chain (0 if no markers found)
    """
    from ..game.constants import Position, is_valid_position
    
    # Get all positions with this marker type
    marker_positions = [
        pos for pos, piece in board.pieces.items()
        if piece == marker_type
    ]
    
    if not marker_positions:
        return 0
    
    # Directions for adjacency (hexagonal board directions)
    directions = [
        (0, 1),   # Vertical up
        (0, -1),  # Vertical down
        (1, 0),   # Horizontal right
        (-1, 0),  # Horizontal left
        (1, 1),   # Diagonal up-right
        (-1, -1), # Diagonal down-left
        (1, -1),  # Diagonal down-right
        (-1, 1),  # Diagonal up-left
    ]
    
    visited = set()
    max_chain_length = 0
    
    # DFS to find chain length
    def dfs_chain_length(pos):
        """DFS helper to calculate chain length from a starting position."""
        if pos in visited:
            return 0
        
        visited.add(pos)
        chain_length = 1  # Count current position
        
        # Check all adjacent positions
        for dx, dy in directions:
            col_idx = ord(pos.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = pos.row + dy
            next_pos = Position(new_col, new_row)
            
            if (is_valid_position(next_pos) and 
                next_pos in marker_positions and 
                next_pos not in visited):
                chain_length += dfs_chain_length(next_pos)
        
        return chain_length
    
    # Find longest chain by checking all unvisited markers
    for pos in marker_positions:
        if pos not in visited:
            chain_length = dfs_chain_length(pos)
            max_chain_length = max(max_chain_length, chain_length)
    
    return max_chain_length


def ring_positioning(game_state: GameState, player: Player) -> float:
    """Calculate ring positioning advantage.
    
    This feature evaluates the strategic value of ring positions,
    considering factors like centrality, mobility, and control
    of key board areas.
    
    The positioning score considers:
    - Centrality: How close rings are to the board center
    - Mobility: Number of valid moves available from ring positions
    - Control: Strategic positioning in key board areas
    
    Args:
        game_state: The current game state
        player: The player to evaluate ring positioning for
        
    Returns:
        Float differential score: my_ring_positioning_score - opponent_score
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Get ring positions for both players
    my_rings = game_state.board.get_rings_positions(player)
    opponent = Player.BLACK if player == Player.WHITE else Player.WHITE
    opponent_rings = game_state.board.get_rings_positions(opponent)
    
    # Calculate positioning scores
    my_score = _calculate_ring_positioning_score(game_state.board, my_rings)
    opponent_score = _calculate_ring_positioning_score(game_state.board, opponent_rings)
    
    # Return differential
    return my_score - opponent_score


def _calculate_ring_positioning_score(board, ring_positions: List) -> float:
    """Calculate positioning score for a set of ring positions.
    
    Args:
        board: The board state
        ring_positions: List of Position objects for rings
        
    Returns:
        Float score representing strategic positioning value
    """
    if not ring_positions:
        return 0.0
    
    from ..game.constants import Position, is_valid_position
    
    score = 0.0
    board_center_col = 'E'
    board_center_row = 5
    
    for pos in ring_positions:
        # 1. Centrality score (closer to center is better)
        col_dist = abs(ord(pos.column) - ord(board_center_col))
        row_dist = abs(pos.row - board_center_row)
        distance_from_center = max(col_dist, row_dist)
        centrality_score = max(0.0, 5.0 - distance_from_center) / 5.0
        score += centrality_score
        
        # 2. Mobility score (more valid moves is better)
        valid_moves = board.valid_move_positions(pos)
        mobility_score = min(len(valid_moves) / 10.0, 1.0)  # Normalize to 0-1
        score += mobility_score
        
        # 3. Control score (position in strategic areas)
        # Center region (D-F, 4-6) is more valuable
        if pos.column in ['D', 'E', 'F'] and 4 <= pos.row <= 6:
            control_score = 1.0
        elif pos.column in ['C', 'G'] and 3 <= pos.row <= 7:
            control_score = 0.5
        else:
            control_score = 0.2
        score += control_score
    
    # Average score per ring
    return score / len(ring_positions) if ring_positions else 0.0


def ring_spread(game_state: GameState, player: Player) -> float:
    """Calculate ring spread differential between players.
    
    This feature measures how well-distributed rings are across
    the board, which affects mobility and strategic flexibility.
    
    A good spread means rings are distributed across different board
    regions, providing more strategic options and reducing vulnerability.
    Poor spread (clustered rings) limits mobility and tactical options.
    
    Args:
        game_state: The current game state
        player: The player to calculate spread for
        
    Returns:
        Float differential: my_ring_spread_score - opponent_spread_score
        Higher values indicate better distribution
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Get ring positions for both players
    my_rings = game_state.board.get_rings_positions(player)
    opponent = Player.BLACK if player == Player.WHITE else Player.WHITE
    opponent_rings = game_state.board.get_rings_positions(opponent)
    
    # Calculate spread scores (higher = better distribution)
    my_spread = _calculate_ring_spread_score(my_rings)
    opponent_spread = _calculate_ring_spread_score(opponent_rings)
    
    # Return differential
    return my_spread - opponent_spread


def _calculate_ring_spread_score(ring_positions: List) -> float:
    """Calculate spread score for a set of ring positions.
    
    Measures how well-distributed rings are. Uses average pairwise
    distance - higher distances indicate better spread.
    
    Args:
        ring_positions: List of Position objects for rings
        
    Returns:
        Float score representing distribution quality (0-10 scale)
    """
    if len(ring_positions) <= 1:
        return 0.0
    
    # Calculate average pairwise distance
    total_distance = 0.0
    pair_count = 0
    
    for i in range(len(ring_positions)):
        for j in range(i + 1, len(ring_positions)):
            pos1, pos2 = ring_positions[i], ring_positions[j]
            
            # Calculate Manhattan-like distance on hexagonal board
            col_dist = abs(ord(pos1.column) - ord(pos2.column))
            row_dist = abs(pos1.row - pos2.row)
            distance = max(col_dist, row_dist)  # Chebyshev distance
            
            total_distance += distance
            pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    avg_distance = total_distance / pair_count
    
    # Normalize to 0-10 scale (typical max distance on board is ~10)
    spread_score = min(avg_distance, 10.0)
    
    return spread_score


def board_control(game_state: GameState, player: Player) -> float:
    """Calculate board control and territorial advantage.
    
    This feature evaluates territorial control by analyzing marker
    density, center control, and influence across board regions.
    
    The control score considers:
    - Marker density in strategic regions (center, inner, outer, edge)
    - Center dominance (markers in center region)
    - Regional influence (weighted by strategic importance)
    
    Args:
        game_state: The current game state
        player: The player to evaluate board control for
        
    Returns:
        Float differential: my_control_score - opponent_control_score
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Get marker types for both players
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Calculate control scores for both players
    my_control = _calculate_board_control_score(game_state.board, my_marker)
    opponent_control = _calculate_board_control_score(game_state.board, opponent_marker)
    
    # Return differential
    return my_control - opponent_control


def _calculate_board_control_score(board, marker_type: PieceType) -> float:
    """Calculate board control score for a marker type.
    
    Analyzes territorial control by evaluating marker distribution
    across board regions with strategic weighting.
    
    Args:
        board: The board state
        marker_type: The marker type to analyze
        
    Returns:
        Float score representing board control (0-10 scale)
    """
    from ..game.constants import Position, is_valid_position
    
    # Get all marker positions
    marker_positions = [
        pos for pos, piece in board.pieces.items()
        if piece == marker_type
    ]
    
    if not marker_positions:
        return 0.0
    
    # Define board regions with strategic weights
    board_center_col = 'E'
    board_center_row = 5
    control_score = 0.0
    
    for pos in marker_positions:
        # Calculate distance from center
        col_dist = abs(ord(pos.column) - ord(board_center_col))
        row_dist = abs(pos.row - board_center_row)
        distance = max(col_dist, row_dist)
        
        # Assign region weights (center is most valuable)
        if distance <= 1:
            region_weight = 3.0  # Center region
        elif distance <= 2:
            region_weight = 2.0  # Inner region
        elif distance <= 3:
            region_weight = 1.0  # Outer region
        else:
            region_weight = 0.5  # Edge region
        
        control_score += region_weight
    
    # Normalize by total possible markers and scale to 0-10
    # Typical maximum markers on board ~50-60
    normalized_score = (control_score / len(marker_positions)) * min(len(marker_positions) / 6.0, 1.0)
    
    return min(normalized_score * 10.0, 10.0)


def extract_all_features(game_state: GameState, player: Player) -> Dict[str, float]:
    """Extract all features for a given game state and player.
    
    This function is a convenience wrapper that extracts all heuristic
    features at once and returns them as a dictionary.
    
    Optimized version that computes marker rows once and reuses them
    to avoid redundant calculations.
    
    Args:
        game_state: The current game state
        player: The player to extract features for
        
    Returns:
        Dictionary mapping feature names to their differential values.
        Keys match the function names (e.g., 'completed_runs_differential',
        'potential_runs_count', etc.)
        
    Raises:
        TypeError: If game_state is not a GameState instance
        ValueError: If player is invalid
        
    Example:
        >>> features = extract_all_features(game_state, Player.WHITE)
        >>> print(f"Runs differential: {features['completed_runs_differential']}")
    """
    if not isinstance(game_state, GameState):
        raise TypeError(f"game_state must be a GameState instance, got {type(game_state)}")
    
    if not isinstance(player, Player):
        raise ValueError(f"player must be a Player enum value, got {type(player)}")
    
    # Optimize: Compute marker rows once and reuse for multiple features
    from ..game.constants import PieceType
    
    my_marker = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
    opponent_marker = PieceType.BLACK_MARKER if player == Player.WHITE else PieceType.WHITE_MARKER
    
    # Compute marker rows once (used by completed_runs and potential_runs)
    my_rows = game_state.board.find_marker_rows(my_marker)
    opponent_rows = game_state.board.find_marker_rows(opponent_marker)
    
    # Extract features using pre-computed rows where possible
    # For features that use marker rows, compute directly
    my_completed_runs = sum(1 for row in my_rows if row.length >= MARKERS_FOR_ROW)
    opponent_completed_runs = sum(1 for row in opponent_rows if row.length >= MARKERS_FOR_ROW)
    completed_runs_diff = my_completed_runs - opponent_completed_runs
    
    min_potential = 3
    my_potential = sum(1 for row in my_rows if min_potential <= row.length < MARKERS_FOR_ROW)
    opponent_potential = sum(1 for row in opponent_rows if min_potential <= row.length < MARKERS_FOR_ROW)
    potential_runs_diff = my_potential - opponent_potential
    
    return {
        'completed_runs_differential': completed_runs_diff,
        'potential_runs_count': float(potential_runs_diff),
        'connected_marker_chains': float(connected_marker_chains(game_state, player)),
        'ring_positioning': ring_positioning(game_state, player),
        'ring_spread': ring_spread(game_state, player),
        'board_control': board_control(game_state, player),
    }



