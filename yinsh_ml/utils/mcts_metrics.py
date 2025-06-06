"""MCTS metrics tracking and analysis."""

import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import (Player, PieceType)
from yinsh_ml.game.moves import Move

class MCTSMetrics:
    def __init__(self):
        self.iteration_data = {}  # Keyed by iteration number
        self.current_iteration = 0  # Need to set this when starting new iterations
        self.search_depths = []  # new
        self.branching_factors = []  # new

    def record_position(self, iteration: int, position_data: dict):
        """Record interesting position data during MCTS."""
        print(f"record_position called with iteration: {iteration}")  # Debug print
        print(f"position_data: {position_data}")  # Debug print

        if iteration not in self.iteration_data:
            self.iteration_data[iteration] = []

        self.iteration_data[iteration].append({
            'value_range': position_data['value_range'],
            'max_visits': position_data['max_visits'],
            'top_moves': [
                {
                    'move': str(move),
                    'value': value,
                    'visits': visits,
                    'ucb': ucb
                }
                for move, value, visits, ucb in position_data['moves']
            ],
            'value_ucb_disagreement': position_data.get('value_ucb_disagreement', False),
            'game_phase': position_data.get('game_phase', 'unknown')
        })

    def add_search_depth(self, depth: int): # new
        """Record the depth of a search."""
        # print(f"Adding search depth: {depth}")  # Debug print
        self.search_depths.append(depth)


    def record_branching_factor(self, branching_factor: int): # new
        """Record the branching factor at a node."""
        # print(f"Recording branching factor: {branching_factor}")  # Debug print
        self.branching_factors.append(branching_factor)

    def analyze_iteration(self, iteration: int) -> dict:
        """Analyze metrics for a specific iteration."""
        positions = self.iteration_data.get(iteration, [])
        if not positions:
            return {}

        return {
            'avg_value_range': np.mean([p['value_range'] for p in positions]),
            'max_value_range': max(p['value_range'] for p in positions),
            'avg_visits': np.mean([p['max_visits'] for p in positions]),
            'value_ucb_disagreements': sum(1 for p in positions if p['value_ucb_disagreement']),
            'num_positions': len(positions)
        }

    def analyze_training_progression(self):
        """Analyze how MCTS behavior changes across iterations."""
        iterations = sorted(self.iteration_data.keys())

        stats = pd.DataFrame([
            self.analyze_iteration(it) for it in iterations
        ], index=iterations)

        # Plot trends
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        stats['avg_value_range'].plot(ax=axes[0, 0], title='Average Value Range')
        stats['avg_visits'].plot(ax=axes[0, 1], title='Average Visits')
        stats['value_ucb_disagreements'].plot(ax=axes[1, 0],
                                              title='Value-UCB Disagreements')

        plt.tight_layout()
        return stats

    def save(self, path: str):
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.iteration_data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MCTSMetrics':
        """Load metrics from file."""
        metrics = cls()
        with open(path) as f:
            metrics.iteration_data = json.load(f)
        return metrics