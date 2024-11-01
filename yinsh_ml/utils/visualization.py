"""Visualization utilities for YINSH training and analysis."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path
import logging
from datetime import datetime
import json

from ..game.constants import Position, PieceType, VALID_POSITIONS
from ..game.board import Board


class TrainingVisualizer:
    """Visualizes training progress and metrics."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger("TrainingVisualizer")
        self.logger.setLevel(logging.INFO)

        # Style configuration
        plt.style.use('seaborn')
        self.colors = {
            'policy_loss': '#2ecc71',
            'value_loss': '#e74c3c',
            'total_loss': '#3498db',
            'win_rate': '#f1c40f'
        }

    def plot_training_history(self, metrics: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot training metrics over time.

        Args:
            metrics: Dictionary containing lists of metric values
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))

        # Plot each metric
        for metric_name, values in metrics.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values,
                     label=metric_name.replace('_', ' ').title(),
                     color=self.colors.get(metric_name, '#000000'),
                     linewidth=2)

        plt.title('Training Metrics Over Time', fontsize=14, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_win_rates(self, win_rates: List[float], draw_rates: List[float],
                       save_path: Optional[str] = None):
        """Plot win rates over training iterations."""
        plt.figure(figsize=(12, 6))

        epochs = range(1, len(win_rates) + 1)

        plt.plot(epochs, win_rates,
                 label='Win Rate',
                 color=self.colors['win_rate'],
                 linewidth=2)
        plt.plot(epochs, draw_rates,
                 label='Draw Rate',
                 color='#95a5a6',
                 linewidth=2,
                 linestyle='--')

        plt.title('Game Outcomes Over Training', fontsize=14, pad=20)
        plt.xlabel('Training Iteration', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.ylim(0, 1)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def visualize_state_tensor(self, state_tensor: torch.Tensor,
                               save_path: Optional[str] = None):
        """
        Visualize the board state tensor channels.

        Args:
            state_tensor: Tensor of shape (6, 11, 11)
            save_path: Optional path to save the visualization
        """
        channel_names = [
            'White Rings', 'Black Rings',
            'White Markers', 'Black Markers',
            'Valid Moves', 'Game Phase'
        ]

        plt.figure(figsize=(15, 10))

        for idx, (channel, name) in enumerate(zip(state_tensor, channel_names), 1):
            plt.subplot(2, 3, idx)

            # Create hexagonal mask for valid positions
            mask = np.ones_like(channel)
            for col in range(11):
                for row in range(11):
                    pos = Position(chr(ord('A') + col), row + 1)
                    if not self._is_valid_position(str(pos)):
                        mask[row, col] = np.nan

            # Apply mask to channel
            masked_channel = channel.numpy() * mask

            # Plot heatmap
            sns.heatmap(masked_channel,
                        cmap='YlOrRd' if 'Valid' in name else 'Blues',
                        square=True,
                        cbar_kws={'label': name})

            plt.title(name)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_move_probabilities(self, probs: np.ndarray, valid_moves: List[Tuple[Position, Position]],
                                save_path: Optional[str] = None):
        """
        Visualize move probabilities.

        Args:
            probs: Array of move probabilities (14641,)
            valid_moves: List of valid (start, end) position tuples
            save_path: Optional path to save visualization
        """
        plt.figure(figsize=(12, 8))

        # Reshape probabilities to 121x121 matrix
        prob_matrix = probs.reshape(121, 121)

        # Create move labels
        valid_indices = [(self._pos_to_index(str(start)), self._pos_to_index(str(end)))
                         for start, end in valid_moves]

        # Plot heatmap
        sns.heatmap(prob_matrix,
                    cmap='viridis',
                    square=True,
                    cbar_kws={'label': 'Probability'})

        # Highlight valid moves
        for start_idx, end_idx in valid_indices:
            plt.plot(end_idx + 0.5, start_idx + 0.5, 'r+', markersize=10)

        plt.title('Move Probability Distribution')
        plt.xlabel('Destination Position')
        plt.ylabel('Source Position')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def save_training_snapshot(self, metrics: Dict[str, List[float]],
                               win_rates: List[float],
                               draw_rates: List[float],
                               epoch: int):
        """
        Save a complete snapshot of training progress.

        Args:
            metrics: Training metrics
            win_rates: List of win rates
            draw_rates: List of draw rates
            epoch: Current training epoch
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.log_dir / f"snapshot_{timestamp}"
        snapshot_dir.mkdir(exist_ok=True)

        # Save metrics plot
        self.plot_training_history(
            metrics,
            save_path=str(snapshot_dir / "training_metrics.png")
        )

        # Save win rates plot
        self.plot_win_rates(
            win_rates,
            draw_rates,
            save_path=str(snapshot_dir / "win_rates.png")
        )

        # Save metrics data
        metrics_data = {
            'epoch': epoch,
            'metrics': metrics,
            'win_rates': win_rates,
            'draw_rates': draw_rates,
            'timestamp': timestamp
        }

        with open(snapshot_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=4)

    def _is_valid_position(self, pos_str: str) -> bool:
        """Check if a position string represents a valid board position."""
        col = pos_str[0]
        row = int(pos_str[1:])
        return col in VALID_POSITIONS and row in VALID_POSITIONS[col]

    def _pos_to_index(self, pos_str: str) -> int:
        """Convert position string to linear index."""
        col = ord(pos_str[0]) - ord('A')
        row = int(pos_str[1:]) - 1
        return row * 11 + col


class GameVisualizer:
    """Visualizes game states and moves."""

    def __init__(self):
        self.piece_colors = {
            PieceType.WHITE_RING: '#ffffff',
            PieceType.BLACK_RING: '#000000',
            PieceType.WHITE_MARKER: '#cccccc',
            PieceType.BLACK_MARKER: '#333333'
        }

        self.background_color = '#f0f0f0'
        self.grid_color = '#666666'

    def draw_board(self, board: Board, selected_pos: Optional[Position] = None,
                   valid_moves: Optional[List[Position]] = None,
                   last_move: Optional[Tuple[Position, Position]] = None,
                   save_path: Optional[str] = None):
        """
        Draw the current board state.

        Args:
            board: Current board state
            selected_pos: Currently selected position (if any)
            valid_moves: List of valid move destinations (if any)
            last_move: Source and destination of last move (if any)
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(12, 12))

        # Draw board grid
        self._draw_grid()

        # Draw pieces
        for pos, piece in board.pieces.items():
            self._draw_piece(pos, piece)

        # Highlight selected position
        if selected_pos:
            self._highlight_position(selected_pos, color='yellow', alpha=0.3)

        # Highlight valid moves
        if valid_moves:
            for pos in valid_moves:
                self._highlight_position(pos, color='green', alpha=0.2)

        # Show last move
        if last_move:
            start, end = last_move
            self._draw_move_arrow(start, end)

        plt.axis('equal')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def _draw_grid(self):
        """Draw the hexagonal board grid."""
        for col in "ABCDEFGHIJK":
            for row in range(1, 12):
                pos = Position(col, row)
                if self._is_valid_position(str(pos)):
                    x, y = self._get_hex_center(pos)
                    self._draw_hexagon(x, y)

    def _draw_piece(self, pos: Position, piece: PieceType):
        """Draw a game piece at the specified position."""
        x, y = self._get_hex_center(pos)

        if piece.is_ring():
            circle = plt.Circle(
                (x, y), 0.35,
                facecolor='none',
                edgecolor=self.piece_colors[piece],
                linewidth=3
            )
            plt.gca().add_artist(circle)
        else:  # Marker
            circle = plt.Circle(
                (x, y), 0.25,
                facecolor=self.piece_colors[piece],
                edgecolor='none'
            )
            plt.gca().add_artist(circle)

    def _highlight_position(self, pos: Position, color: str, alpha: float):
        """Highlight a board position."""
        x, y = self._get_hex_center(pos)
        hexagon = self._get_hexagon_path(x, y)
        plt.fill(hexagon[:, 0], hexagon[:, 1], color=color, alpha=alpha)

    def _draw_move_arrow(self, start: Position, end: Position):
        """Draw an arrow indicating a move."""
        start_x, start_y = self._get_hex_center(start)
        end_x, end_y = self._get_hex_center(end)

        plt.arrow(
            start_x, start_y,
            end_x - start_x, end_y - start_y,
            color='blue',
            alpha=0.5,
            head_width=0.2,
            head_length=0.3,
            length_includes_head=True
        )

    def _get_hex_center(self, pos: Position) -> Tuple[float, float]:
        """Get the center coordinates of a hexagon for a board position."""
        col_idx = ord(pos.column) - ord('A')
        row_idx = pos.row - 1

        # Use hexagonal geometry to calculate coordinates
        x = col_idx * 1.5
        y = row_idx * np.sqrt(3) / 2

        return x, y

    def _draw_hexagon(self, x: float, y: float):
        """Draw a single hexagon of the board grid."""
        hexagon = self._get_hexagon_path(x, y)
        plt.plot(
            hexagon[:, 0],
            hexagon[:, 1],
            color=self.grid_color,
            linewidth=1
        )

    def _get_hexagon_path(self, x: float, y: float) -> np.ndarray:
        """Get the vertices of a hexagon centered at (x,y)."""
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        vertices = np.column_stack([
            x + 0.5 * np.cos(angles),
            y + 0.5 * np.sin(angles)
        ])
        return vertices

    def _is_valid_position(self, pos_str: str) -> bool:
        """Check if a position string represents a valid board position."""
        col = pos_str[0]
        row = int(pos_str[1:])
        return col in VALID_POSITIONS and row in VALID_POSITIONS[col]