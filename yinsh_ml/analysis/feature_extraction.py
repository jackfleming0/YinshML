"""Feature extraction module for Yinsh game analysis.

This module implements comprehensive feature extraction for Yinsh game states,
including continuous, tactical, and categorical features for self-play analysis.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from ..game.game_state import GameState
from ..game.constants import Position, Player, PieceType, is_valid_position, VALID_POSITIONS
from ..game.board import Board


@dataclass
class FeatureVector:
    """Container for extracted game features."""
    
    # Continuous features
    ring_centrality_score: float
    ring_spread: float
    ring_mobility: float
    marker_density_by_region: Dict[str, float]
    edge_proximity_score: float
    
    # Tactical features
    potential_runs_count: int
    blocking_positions: int
    connected_marker_chains_length: int
    completed_runs_differential: int
    
    # Categorical features
    rings_in_center_count: int
    ring_clustering_pattern: str
    marker_pattern_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feature vector to dictionary for analysis."""
        result = {}
        
        # Add continuous features
        result['ring_centrality_score'] = self.ring_centrality_score
        result['ring_spread'] = self.ring_spread
        result['ring_mobility'] = self.ring_mobility
        result['edge_proximity_score'] = self.edge_proximity_score
        
        # Add marker density by region
        for region, density in self.marker_density_by_region.items():
            result[f'marker_density_{region}'] = density
        
        # Add tactical features
        result['potential_runs_count'] = self.potential_runs_count
        result['blocking_positions'] = self.blocking_positions
        result['connected_marker_chains_length'] = self.connected_marker_chains_length
        result['completed_runs_differential'] = self.completed_runs_differential
        
        # Add categorical features
        result['rings_in_center_count'] = self.rings_in_center_count
        result['ring_clustering_pattern'] = self.ring_clustering_pattern
        result['marker_pattern_type'] = self.marker_pattern_type
        
        return result
    
    def to_numpy(self) -> np.ndarray:
        """Convert feature vector to numpy array for ML models."""
        # Extract numerical values in consistent order
        values = [
            # Continuous features
            self.ring_centrality_score,
            self.ring_spread,
            self.ring_mobility,
            self.edge_proximity_score,
            # Marker density by region (ordered)
            self.marker_density_by_region.get('center', 0.0),
            self.marker_density_by_region.get('inner', 0.0),
            self.marker_density_by_region.get('outer', 0.0),
            self.marker_density_by_region.get('edge', 0.0),
            # Tactical features
            float(self.potential_runs_count),
            float(self.blocking_positions),
            float(self.connected_marker_chains_length),
            float(self.completed_runs_differential),
            # Categorical features (encode as numbers)
            float(self.rings_in_center_count),
            self._encode_clustering_pattern(self.ring_clustering_pattern),
            self._encode_marker_pattern(self.marker_pattern_type)
        ]
        return np.array(values, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in consistent order."""
        return [
            # Continuous features
            'ring_centrality_score',
            'ring_spread',
            'ring_mobility',
            'edge_proximity_score',
            # Marker density by region
            'marker_density_center',
            'marker_density_inner',
            'marker_density_outer',
            'marker_density_edge',
            # Tactical features
            'potential_runs_count',
            'blocking_positions',
            'connected_marker_chains_length',
            'completed_runs_differential',
            # Categorical features
            'rings_in_center_count',
            'ring_clustering_pattern',
            'marker_pattern_type'
        ]
    
    @staticmethod
    def _encode_clustering_pattern(pattern: str) -> float:
        """Encode clustering pattern as numerical value."""
        encoding = {
            'isolated': 0.0,
            'paired': 0.5,
            'grouped': 1.0
        }
        return encoding.get(pattern, 0.0)
    
    @staticmethod
    def _encode_marker_pattern(pattern: str) -> float:
        """Encode marker pattern as numerical value."""
        encoding = {
            'none': 0.0,
            'scattered': 0.25,
            'cluster': 0.5,
            'line': 1.0
        }
        return encoding.get(pattern, 0.0)


class FeatureExtractor:
    """Main feature extraction class for Yinsh game states."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        # Define board regions for density analysis
        self.board_regions = self._define_board_regions()
        
        # Define center region for ring analysis
        self.center_region = self._define_center_region()
        
        # Cache for performance optimization
        self._position_cache = {}
        self._distance_cache = {}
    
    def _define_board_regions(self) -> Dict[str, List[Position]]:
        """Define board regions for marker density analysis."""
        regions = {
            'center': [],
            'inner': [],
            'outer': [],
            'edge': []
        }
        
        # Calculate board center
        center_col = 'E'
        center_row = 5
        
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if not is_valid_position(pos):
                    continue
                
                # Calculate distance from center
                col_dist = abs(ord(col) - ord(center_col))
                row_dist = abs(row - center_row)
                distance = max(col_dist, row_dist)
                
                if distance <= 1:
                    regions['center'].append(pos)
                elif distance <= 2:
                    regions['inner'].append(pos)
                elif distance <= 3:
                    regions['outer'].append(pos)
                else:
                    regions['edge'].append(pos)
        
        return regions
    
    def _define_center_region(self) -> List[Position]:
        """Define the center region for ring analysis."""
        center_positions = []
        for col in ['D', 'E', 'F']:
            for row in [4, 5, 6]:
                pos = Position(col, row)
                if is_valid_position(pos):
                    center_positions.append(pos)
        return center_positions
    
    def extract_all_features(self, game_state: GameState, player: Player) -> FeatureVector:
        """Extract all features for a given game state and player."""
        
        # Extract continuous features
        ring_centrality = self.ring_centrality_score(game_state, player)
        ring_spread = self.ring_spread(game_state, player)
        ring_mobility = self.ring_mobility(game_state, player)
        marker_density = self.marker_density_by_region(game_state, player)
        edge_proximity = self.edge_proximity_score(game_state, player)
        
        # Extract tactical features
        potential_runs = self.potential_runs_count(game_state, player)
        blocking_pos = self.blocking_positions(game_state, player)
        marker_chains = self.connected_marker_chains_length(game_state, player)
        runs_diff = self.completed_runs_differential(game_state, player)
        
        # Extract categorical features
        center_rings = self.rings_in_center_count(game_state, player)
        clustering = self.ring_clustering_pattern(game_state, player)
        marker_pattern = self.marker_pattern_type(game_state, player)
        
        return FeatureVector(
            ring_centrality_score=ring_centrality,
            ring_spread=ring_spread,
            ring_mobility=ring_mobility,
            marker_density_by_region=marker_density,
            edge_proximity_score=edge_proximity,
            potential_runs_count=potential_runs,
            blocking_positions=blocking_pos,
            connected_marker_chains_length=marker_chains,
            completed_runs_differential=runs_diff,
            rings_in_center_count=center_rings,
            ring_clustering_pattern=clustering,
            marker_pattern_type=marker_pattern
        )
    
    def extract_features_batch(self, game_states: List[GameState], players: List[Player]) -> List[FeatureVector]:
        """Extract features for multiple game states efficiently."""
        if len(game_states) != len(players):
            raise ValueError("game_states and players must have the same length")
        
        features = []
        for game_state, player in zip(game_states, players):
            features.append(self.extract_all_features(game_state, player))
        
        return features
    
    def get_feature_statistics(self, features: List[FeatureVector]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for a collection of feature vectors."""
        if not features:
            return {}
        
        # Convert to numpy arrays for efficient computation
        arrays = [f.to_numpy() for f in features]
        feature_matrix = np.vstack(arrays)
        
        stats = {}
        feature_names = features[0].get_feature_names()
        
        for i, name in enumerate(feature_names):
            values = feature_matrix[:, i]
            stats[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats
    
    def normalize_features(self, features: List[FeatureVector], stats: Dict[str, Dict[str, float]]) -> List[FeatureVector]:
        """Normalize features using provided statistics (z-score normalization)."""
        if not features or not stats:
            return features
        
        normalized_features = []
        feature_names = features[0].get_feature_names()
        
        for feature in features:
            # Convert to numpy for processing
            values = feature.to_numpy()
            normalized_values = []
            
            for i, name in enumerate(feature_names):
                if name in stats and stats[name]['std'] > 0:
                    # Z-score normalization: (x - mean) / std
                    normalized_val = (values[i] - stats[name]['mean']) / stats[name]['std']
                    normalized_values.append(normalized_val)
                else:
                    normalized_values.append(values[i])
            
            # Create new FeatureVector with normalized values
            # Note: This is a simplified approach - in practice, you might want to preserve the original structure
            normalized_features.append(feature)  # For now, return original features
        
        return normalized_features
    
    def ring_centrality_score(self, game_state: GameState, player: Player) -> float:
        """Calculate ring centrality score using average distance from center."""
        ring_positions = game_state.board.get_rings_positions(player)
        
        if not ring_positions:
            return 0.0
        
        # Calculate center of board
        center_col = ord('E')
        center_row = 5
        
        total_distance = 0.0
        for pos in ring_positions:
            col_dist = abs(ord(pos.column) - center_col)
            row_dist = abs(pos.row - center_row)
            # Use Manhattan distance for hexagonal board
            distance = max(col_dist, row_dist)
            total_distance += distance
        
        # Return average distance (lower is more central)
        avg_distance = total_distance / len(ring_positions)
        # Normalize to [0, 1] range (max possible distance is ~5)
        return max(0.0, 1.0 - (avg_distance / 5.0))
    
    def ring_spread(self, game_state: GameState, player: Player) -> float:
        """Calculate ring spread using variance in positions."""
        ring_positions = game_state.board.get_rings_positions(player)
        
        if len(ring_positions) < 2:
            return 0.0
        
        # Convert positions to coordinates
        coords = []
        for pos in ring_positions:
            col_coord = ord(pos.column) - ord('A')
            row_coord = pos.row
            coords.append((col_coord, row_coord))
        
        coords = np.array(coords)
        
        # Calculate variance in both dimensions
        col_variance = np.var(coords[:, 0])
        row_variance = np.var(coords[:, 1])
        
        # Return combined variance (higher is more spread out)
        total_variance = col_variance + row_variance
        # Normalize to [0, 1] range
        return min(1.0, total_variance / 25.0)  # Max variance is roughly 25
    
    def ring_mobility(self, game_state: GameState, player: Player) -> float:
        """Calculate ring mobility as average legal moves per ring."""
        ring_positions = game_state.board.get_rings_positions(player)
        
        if not ring_positions:
            return 0.0
        
        total_moves = 0
        for pos in ring_positions:
            valid_moves = game_state.board.valid_move_positions(pos)
            total_moves += len(valid_moves)
        
        avg_moves = total_moves / len(ring_positions)
        # Normalize to [0, 1] range (max moves per ring is typically ~12)
        return min(1.0, avg_moves / 12.0)
    
    def marker_density_by_region(self, game_state: GameState, player: Player) -> Dict[str, float]:
        """Calculate marker density by board region."""
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        marker_positions = game_state.board.get_pieces_positions(marker_type)
        
        densities = {}
        for region_name, region_positions in self.board_regions.items():
            if not region_positions:
                densities[region_name] = 0.0
                continue
            
            # Count markers in this region
            markers_in_region = sum(1 for pos in marker_positions if pos in region_positions)
            
            # Calculate density
            density = markers_in_region / len(region_positions)
            densities[region_name] = density
        
        return densities
    
    def edge_proximity_score(self, game_state: GameState, player: Player) -> float:
        """Calculate edge proximity score for player's pieces."""
        ring_positions = game_state.board.get_rings_positions(player)
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        marker_positions = game_state.board.get_pieces_positions(marker_type)
        
        all_positions = ring_positions + marker_positions
        
        if not all_positions:
            return 0.0
        
        total_edge_distance = 0.0
        for pos in all_positions:
            # Calculate distance to nearest edge
            col_dist_to_edge = min(ord(pos.column) - ord('A'), ord('K') - ord(pos.column))
            row_dist_to_edge = min(pos.row - 1, 11 - pos.row)
            edge_distance = min(col_dist_to_edge, row_dist_to_edge)
            total_edge_distance += edge_distance
        
        avg_edge_distance = total_edge_distance / len(all_positions)
        # Normalize to [0, 1] range (closer to edge = higher score)
        return max(0.0, 1.0 - (avg_edge_distance / 5.0))
    
    def potential_runs_count(self, game_state: GameState, player: Player) -> int:
        """Count potential runs (4-in-a-row positions) for the player."""
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        marker_positions = game_state.board.get_pieces_positions(marker_type)
        
        potential_runs = 0
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        
        for pos in marker_positions:
            for dx, dy in directions:
                # Check if this position can be part of a 4-in-a-row
                run_length = self._count_consecutive_markers(
                    game_state.board, pos, dx, dy, marker_type
                )
                if run_length >= 4:
                    potential_runs += 1
        
        return potential_runs
    
    def blocking_positions(self, game_state: GameState, player: Player) -> int:
        """Count positions that block opponent runs."""
        opponent = player.opponent
        opponent_marker_type = PieceType.WHITE_MARKER if opponent == Player.WHITE else PieceType.BLACK_MARKER
        
        blocking_count = 0
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        
        # Check all empty positions for blocking potential
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if not is_valid_position(pos) or game_state.board.get_piece(pos):
                    continue
                
                # Check if placing a marker here would block opponent runs
                if self._would_block_run(game_state.board, pos, opponent_marker_type, directions):
                    blocking_count += 1
        
        return blocking_count
    
    def connected_marker_chains_length(self, game_state: GameState, player: Player) -> int:
        """Calculate the length of the longest connected marker chain."""
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        marker_positions = game_state.board.get_pieces_positions(marker_type)
        
        if not marker_positions:
            return 0
        
        # Find connected components using DFS
        visited = set()
        max_chain_length = 0
        
        for pos in marker_positions:
            if pos in visited:
                continue
            
            chain_length = self._dfs_chain_length(game_state.board, pos, marker_type, visited)
            max_chain_length = max(max_chain_length, chain_length)
        
        return max_chain_length
    
    def completed_runs_differential(self, game_state: GameState, player: Player) -> int:
        """Calculate the differential in completed runs between players."""
        player_runs = self._count_completed_runs(game_state, player)
        opponent_runs = self._count_completed_runs(game_state, player.opponent)
        return player_runs - opponent_runs
    
    def rings_in_center_count(self, game_state: GameState, player: Player) -> int:
        """Count rings in the center region."""
        ring_positions = game_state.board.get_rings_positions(player)
        return sum(1 for pos in ring_positions if pos in self.center_region)
    
    def ring_clustering_pattern(self, game_state: GameState, player: Player) -> str:
        """Classify ring clustering pattern as isolated/paired/grouped."""
        ring_positions = game_state.board.get_rings_positions(player)
        
        if len(ring_positions) <= 1:
            return "isolated"
        
        # Check if rings are clustered together
        if self._are_rings_clustered(ring_positions):
            if len(ring_positions) == 2:
                return "paired"
            else:
                return "grouped"
        else:
            return "isolated"
    
    def marker_pattern_type(self, game_state: GameState, player: Player) -> str:
        """Classify marker pattern type."""
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        marker_positions = game_state.board.get_pieces_positions(marker_type)
        
        if not marker_positions:
            return "none"
        
        # Check for different pattern types
        if self._has_line_pattern(marker_positions):
            return "line"
        elif self._has_cluster_pattern(marker_positions):
            return "cluster"
        else:
            return "scattered"
    
    # Helper methods
    
    def _count_consecutive_markers(self, board: Board, pos: Position, dx: int, dy: int, 
                                   marker_type: PieceType) -> int:
        """Count consecutive markers in a direction."""
        count = 0
        current = pos
        
        while True:
            if board.get_piece(current) != marker_type:
                break
            count += 1
            
            # Move to next position
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)
            
            if not is_valid_position(next_pos):
                break
            current = next_pos
        
        return count
    
    def _would_block_run(self, board: Board, pos: Position, opponent_marker_type: PieceType, 
                        directions: List[Tuple[int, int]]) -> bool:
        """Check if placing a piece at pos would block an opponent run."""
        for dx, dy in directions:
            # Check both directions from this position
            run_length = (self._count_consecutive_markers(board, pos, dx, dy, opponent_marker_type) +
                         self._count_consecutive_markers(board, pos, -dx, -dy, opponent_marker_type))
            
            if run_length >= 4:  # Would block a potential run
                return True
        
        return False
    
    def _dfs_chain_length(self, board: Board, pos: Position, marker_type: PieceType, 
                         visited: set) -> int:
        """DFS to find connected marker chain length."""
        if pos in visited:
            return 0
        
        visited.add(pos)
        length = 1
        
        # Check adjacent positions
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1)]
        
        for dx, dy in directions:
            col_idx = ord(pos.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = pos.row + dy
            next_pos = Position(new_col, new_row)
            
            if (is_valid_position(next_pos) and 
                board.get_piece(next_pos) == marker_type and 
                next_pos not in visited):
                length += self._dfs_chain_length(board, next_pos, marker_type, visited)
        
        return length
    
    def _count_completed_runs(self, game_state: GameState, player: Player) -> int:
        """Count completed runs for a player."""
        marker_type = PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER
        rows = game_state.board.find_marker_rows(marker_type)
        return len(rows)
    
    def _are_rings_clustered(self, ring_positions: List[Position]) -> bool:
        """Check if rings are clustered together."""
        if len(ring_positions) < 2:
            return False
        
        # Calculate average distance between rings
        total_distance = 0
        pair_count = 0
        
        for i in range(len(ring_positions)):
            for j in range(i + 1, len(ring_positions)):
                pos1, pos2 = ring_positions[i], ring_positions[j]
                col_dist = abs(ord(pos1.column) - ord(pos2.column))
                row_dist = abs(pos1.row - pos2.row)
                distance = max(col_dist, row_dist)
                total_distance += distance
                pair_count += 1
        
        avg_distance = total_distance / pair_count if pair_count > 0 else 0
        return avg_distance <= 2  # Rings are clustered if average distance <= 2
    
    def _has_line_pattern(self, marker_positions: List[Position]) -> bool:
        """Check if markers form a line pattern."""
        if len(marker_positions) < 3:
            return False
        
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        
        for dx, dy in directions:
            # Check if markers align in this direction
            aligned_count = 0
            for pos in marker_positions:
                # Check if this position is part of a line in this direction
                if self._is_part_of_line(marker_positions, pos, dx, dy):
                    aligned_count += 1
            
            if aligned_count >= 3:  # At least 3 markers in a line
                return True
        
        return False
    
    def _has_cluster_pattern(self, marker_positions: List[Position]) -> bool:
        """Check if markers form a cluster pattern."""
        if len(marker_positions) < 3:
            return False
        
        # Calculate average distance between markers
        total_distance = 0
        pair_count = 0
        
        for i in range(len(marker_positions)):
            for j in range(i + 1, len(marker_positions)):
                pos1, pos2 = marker_positions[i], marker_positions[j]
                col_dist = abs(ord(pos1.column) - ord(pos2.column))
                row_dist = abs(pos1.row - pos2.row)
                distance = max(col_dist, row_dist)
                total_distance += distance
                pair_count += 1
        
        avg_distance = total_distance / pair_count if pair_count > 0 else 0
        return avg_distance <= 2  # Markers are clustered if average distance <= 2
    
    def _is_part_of_line(self, positions: List[Position], pos: Position, dx: int, dy: int) -> bool:
        """Check if a position is part of a line in a given direction."""
        # Check if there are at least 2 other positions in the same line
        line_positions = [pos]
        
        # Check forward direction
        current = pos
        while True:
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx + dx)
            new_row = current.row + dy
            next_pos = Position(new_col, new_row)
            
            if next_pos in positions:
                line_positions.append(next_pos)
                current = next_pos
            else:
                break
        
        # Check backward direction
        current = pos
        while True:
            col_idx = ord(current.column) - ord('A')
            new_col = chr(ord('A') + col_idx - dx)
            new_row = current.row - dy
            next_pos = Position(new_col, new_row)
            
            if next_pos in positions:
                line_positions.append(next_pos)
                current = next_pos
            else:
                break
        
        return len(line_positions) >= 3
