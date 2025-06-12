#!/usr/bin/env python3
"""
Yinsh-specific visualizations for TensorBoard integration.

This module provides specialized visualization functions for the Yinsh game,
including board state rendering, game trajectories, and model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import io
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import torch

from ..game.constants import Position, Player, PieceType, VALID_POSITIONS
from ..game.board import Board
from ..game.game_state import GameState, GamePhase
from ..game.types import Move, MoveType
from ..utils.encoding import StateEncoder


class YinshBoardVisualizer:
    """Visualizes Yinsh board states and game analysis for TensorBoard."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize the board visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        self.state_encoder = StateEncoder()
        
        # Board layout constants
        self.HEX_SIZE = 0.4
        self.HEX_SPACING = 0.9
        
        # Color scheme
        self.colors = {
            'background': '#f8f9fa',
            'board_line': '#495057',
            'valid_position': '#e9ecef',
            'white_ring': '#ffffff',
            'black_ring': '#212529',
            'white_marker': '#ffc107',
            'black_marker': '#fd7e14',
            'valid_move': '#28a745',
            'attention': '#dc3545',
            'ring_border': '#6c757d'
        }
        
        # Pre-calculate position coordinates
        self._position_coords = self._calculate_position_coordinates()
    
    def _calculate_position_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Calculate x, y coordinates for each valid board position."""
        coords = {}
        
        for col_letter in 'ABCDEFGHIJK':
            col_idx = ord(col_letter) - ord('A')  # 0-10
            valid_rows = VALID_POSITIONS[col_letter]
            
            for row in valid_rows:
                # Hexagonal grid layout
                x = col_idx * self.HEX_SPACING
                # Offset rows for hexagonal pattern
                y_offset = (col_idx % 2) * 0.5
                y = (row - 1) * self.HEX_SPACING + y_offset
                
                position_str = f"{col_letter}{row}"
                coords[position_str] = (x, y)
        
        return coords
    
    def _create_hexagon(self, center: Tuple[float, float], size: float) -> patches.RegularPolygon:
        """Create a hexagonal patch for board positions."""
        return patches.RegularPolygon(
            center, 6, radius=size,
            orientation=np.pi/6,  # Flat-top hexagon
            facecolor=self.colors['valid_position'],
            edgecolor=self.colors['board_line'],
            linewidth=1
        )
    
    def render_board_state(self, 
                          board: Board, 
                          game_state: Optional[GameState] = None,
                          valid_moves: Optional[List[Position]] = None,
                          attention_weights: Optional[Dict[str, float]] = None,
                          title: str = "Yinsh Board State") -> np.ndarray:
        """
        Render the current board state as an image.
        
        Args:
            board: Board object with current piece positions
            game_state: Optional game state for additional context
            valid_moves: Optional list of valid move positions to highlight
            attention_weights: Optional attention weights for positions
            title: Title for the visualization
            
        Returns:
            numpy array representing the rendered image
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect('equal')
        ax.set_facecolor(self.colors['background'])
        
        # Draw board positions
        for position_str, (x, y) in self._position_coords.items():
            # Draw hexagonal position
            hex_patch = self._create_hexagon((x, y), self.HEX_SIZE)
            
            # Apply attention coloring if available
            if attention_weights and position_str in attention_weights:
                attention_val = attention_weights[position_str]
                # Blend attention color with base color
                alpha = min(1.0, attention_val)
                hex_patch.set_facecolor(self.colors['attention'])
                hex_patch.set_alpha(alpha)
            
            ax.add_patch(hex_patch)
            
            # Draw position label
            ax.text(x, y, position_str, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color=self.colors['board_line'])
        
        # Draw pieces
        for position_str, (x, y) in self._position_coords.items():
            try:
                pos = Position.from_string(position_str)
                piece = board.get_piece(pos)
                
                if piece is not None:
                    if piece == PieceType.WHITE_RING:
                        self._draw_ring(ax, x, y, 'white')
                    elif piece == PieceType.BLACK_RING:
                        self._draw_ring(ax, x, y, 'black')
                    elif piece == PieceType.WHITE_MARKER:
                        self._draw_marker(ax, x, y, 'white')
                    elif piece == PieceType.BLACK_MARKER:
                        self._draw_marker(ax, x, y, 'black')
            except:
                continue  # Skip invalid positions
        
        # Highlight valid moves
        if valid_moves:
            for pos in valid_moves:
                pos_str = str(pos)
                if pos_str in self._position_coords:
                    x, y = self._position_coords[pos_str]
                    highlight = patches.Circle((x, y), self.HEX_SIZE * 0.8, 
                                             facecolor='none', 
                                             edgecolor=self.colors['valid_move'],
                                             linewidth=3)
                    ax.add_patch(highlight)
        
        # Add game state information
        info_text = self._generate_game_info(game_state)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set title and clean up axes
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-1, max(x for x, y in self._position_coords.values()) + 1)
        ax.set_ylim(-1, max(y for x, y in self._position_coords.values()) + 1)
        ax.axis('off')
        
        # Convert to numpy array
        img_array = self._fig_to_array(fig)
        plt.close(fig)
        
        return img_array
    
    def _draw_ring(self, ax, x: float, y: float, color: str):
        """Draw a ring piece at the specified position."""
        ring_color = self.colors[f'{color}_ring']
        border_color = self.colors['ring_border']
        
        # Outer circle (ring)
        outer_ring = patches.Circle((x, y), self.HEX_SIZE * 0.6, 
                                   facecolor=ring_color, 
                                   edgecolor=border_color, 
                                   linewidth=2)
        ax.add_patch(outer_ring)
        
        # Inner circle (hole)
        inner_hole = patches.Circle((x, y), self.HEX_SIZE * 0.3, 
                                   facecolor=self.colors['background'])
        ax.add_patch(inner_hole)
    
    def _draw_marker(self, ax, x: float, y: float, color: str):
        """Draw a marker piece at the specified position."""
        marker_color = self.colors[f'{color}_marker']
        
        marker = patches.Circle((x, y), self.HEX_SIZE * 0.4, 
                               facecolor=marker_color, 
                               edgecolor=self.colors['board_line'], 
                               linewidth=1)
        ax.add_patch(marker)
    
    def _generate_game_info(self, game_state: Optional[GameState]) -> str:
        """Generate informational text about the current game state."""
        if not game_state:
            return "Board State"
        
        info_lines = [
            f"Phase: {game_state.phase.name}",
            f"Current Player: {game_state.current_player.name}",
            f"White Score: {game_state.white_score}",
            f"Black Score: {game_state.black_score}"
        ]
        
        if game_state.rings_placed:
            white_rings = game_state.rings_placed.get(Player.WHITE, 0)
            black_rings = game_state.rings_placed.get(Player.BLACK, 0)
            info_lines.extend([
                f"Rings Placed:",
                f"  White: {white_rings}/5",
                f"  Black: {black_rings}/5"
            ])
        
        return "\n".join(info_lines)
    
    def render_move_trajectory(self, 
                              moves: List[Move], 
                              board_states: List[Board],
                              title: str = "Game Trajectory") -> np.ndarray:
        """
        Render a sequence of moves as a trajectory visualization.
        
        Args:
            moves: List of moves in chronological order
            board_states: List of board states corresponding to each move
            title: Title for the visualization
            
        Returns:
            numpy array representing the rendered image
        """
        if not moves or not board_states:
            return self._create_empty_image("No trajectory data")
        
        # Create subplots for key game moments
        n_states = min(6, len(board_states))  # Show up to 6 key states
        step_size = max(1, len(board_states) // n_states)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(n_states):
            idx = i * step_size
            if idx >= len(board_states):
                break
                
            ax = axes[i]
            
            # Render mini board state
            self._render_mini_board(ax, board_states[idx], moves[idx] if idx < len(moves) else None)
            ax.set_title(f"Move {idx + 1}", fontsize=10)
        
        # Hide unused subplots
        for i in range(n_states, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        img_array = self._fig_to_array(fig)
        plt.close(fig)
        
        return img_array
    
    def _render_mini_board(self, ax, board: Board, move: Optional[Move] = None):
        """Render a simplified board state for trajectory visualization."""
        ax.set_aspect('equal')
        ax.set_facecolor(self.colors['background'])
        
        # Simplified grid representation
        for position_str, (x, y) in list(self._position_coords.items())[::3]:  # Sample positions
            try:
                pos = Position.from_string(position_str)
                piece = board.get_piece(pos)
                
                # Simple dot representation
                if piece == PieceType.WHITE_RING:
                    ax.plot(x, y, 'o', color='white', markersize=8, 
                           markeredgecolor='black', markeredgewidth=1)
                elif piece == PieceType.BLACK_RING:
                    ax.plot(x, y, 'o', color='black', markersize=8)
                elif piece == PieceType.WHITE_MARKER:
                    ax.plot(x, y, 's', color='gold', markersize=6)
                elif piece == PieceType.BLACK_MARKER:
                    ax.plot(x, y, 's', color='orange', markersize=6)
            except:
                continue
        
        # Highlight move if provided
        if move and move.source:
            source_str = str(move.source)
            if source_str in self._position_coords:
                x, y = self._position_coords[source_str]
                ax.plot(x, y, '*', color='red', markersize=12)
        
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
        ax.axis('off')
    
    def render_attention_heatmap(self, 
                                attention_weights: Dict[str, float],
                                board: Optional[Board] = None,
                                title: str = "Attention Heatmap") -> np.ndarray:
        """
        Render attention weights as a heatmap overlay on the board.
        
        Args:
            attention_weights: Dictionary mapping position strings to attention values
            board: Optional board state to show pieces
            title: Title for the visualization
            
        Returns:
            numpy array representing the rendered image
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect('equal')
        ax.set_facecolor(self.colors['background'])
        
        # Normalize attention weights
        if attention_weights:
            max_attention = max(attention_weights.values())
            min_attention = min(attention_weights.values())
            attention_range = max_attention - min_attention
            
            if attention_range > 0:
                normalized_weights = {
                    pos: (weight - min_attention) / attention_range
                    for pos, weight in attention_weights.items()
                }
            else:
                normalized_weights = {pos: 0.5 for pos in attention_weights}
        else:
            normalized_weights = {}
        
        # Draw positions with attention coloring
        for position_str, (x, y) in self._position_coords.items():
            attention_val = normalized_weights.get(position_str, 0.0)
            
            # Create heat-mapped hexagon
            hex_patch = self._create_hexagon((x, y), self.HEX_SIZE)
            
            # Color based on attention
            intensity = attention_val
            color = plt.cm.Reds(intensity)
            hex_patch.set_facecolor(color)
            hex_patch.set_alpha(0.7 + 0.3 * intensity)  # Vary transparency
            
            ax.add_patch(hex_patch)
            
            # Add attention value text
            if attention_val > 0.1:  # Only show significant values
                ax.text(x, y, f"{attention_val:.2f}", ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        # Draw pieces if board provided
        if board:
            for position_str, (x, y) in self._position_coords.items():
                try:
                    pos = Position.from_string(position_str)
                    piece = board.get_piece(pos)
                    
                    if piece is not None:
                        # Draw pieces with slight transparency
                        if piece == PieceType.WHITE_RING:
                            self._draw_ring(ax, x, y, 'white')
                        elif piece == PieceType.BLACK_RING:
                            self._draw_ring(ax, x, y, 'black')
                        elif piece == PieceType.WHITE_MARKER:
                            self._draw_marker(ax, x, y, 'white')
                        elif piece == PieceType.BLACK_MARKER:
                            self._draw_marker(ax, x, y, 'black')
                except:
                    continue
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(-1, max(x for x, y in self._position_coords.values()) + 1)
        ax.set_ylim(-1, max(y for x, y in self._position_coords.values()) + 1)
        ax.axis('off')
        
        img_array = self._fig_to_array(fig)
        plt.close(fig)
        
        return img_array
    
    def render_phase_analysis(self, 
                            phase_metrics: Dict[str, Dict[str, float]],
                            title: str = "Phase Analysis") -> np.ndarray:
        """
        Render analysis of performance across different game phases.
        
        Args:
            phase_metrics: Dictionary with phase names as keys and metric dictionaries as values
            title: Title for the visualization
            
        Returns:
            numpy array representing the rendered image
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        phases = list(phase_metrics.keys())
        metrics = ['accuracy', 'confidence', 'loss', 'games_played']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            values = [phase_metrics[phase].get(metric, 0) for phase in phases]
            
            bars = ax.bar(phases, values, color=plt.cm.viridis(np.linspace(0, 1, len(phases))))
            ax.set_title(f"{metric.title()} by Phase")
            ax.set_ylabel(metric.title())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        img_array = self._fig_to_array(fig)
        plt.close(fig)
        
        return img_array
    
    def _fig_to_array(self, fig) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        # Save figure to bytes buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Load as PIL Image and convert to numpy
        pil_image = Image.open(buffer)
        img_array = np.array(pil_image)
        buffer.close()
        
        return img_array
    
    def _create_empty_image(self, message: str = "No data available") -> np.ndarray:
        """Create an empty placeholder image with a message."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        
        img_array = self._fig_to_array(fig)
        plt.close(fig)
        
        return img_array