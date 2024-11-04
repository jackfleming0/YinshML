from typing import Dict, List
import numpy as np
import logging
from dataclasses import dataclass, field  # Optional, if you want to use @dataclass
import matplotlib.pyplot as plt  # If you want to use the plotting functionality

class TemperatureMetrics:
    """Tracks the effectiveness of temperature annealing."""

    def __init__(self):
        self.move_temps = []  # List of (move_number, temperature) tuples
        self.move_entropies = []  # List of (move_number, entropy) tuples
        self.early_game_stats = []  # Stats for moves 0-10
        self.mid_game_stats = []  # Stats for moves 11-30
        self.late_game_stats = []  # Stats for moves 31+

    def add_move_data(self, move_number: int, temperature: float,
                      move_probs: np.ndarray, selected_move_idx: int):
        """Record data for a single move."""
        # Store temperature
        self.move_temps.append((move_number, temperature))

        # Calculate and store policy entropy
        entropy = -np.sum(move_probs * np.log(move_probs + 1e-10))
        self.move_entropies.append((move_number, entropy))

        # Record move statistics by game stage
        move_stats = {
            'temperature': temperature,
            'entropy': entropy,
            'top_prob': np.max(move_probs),
            'selected_prob': move_probs[selected_move_idx] if selected_move_idx < len(move_probs) else 0.0
        }

        if move_number <= 10:
            self.early_game_stats.append(move_stats)
        elif move_number <= 30:
            self.mid_game_stats.append(move_stats)
        else:
            self.late_game_stats.append(move_stats)

    def add_game_result(self, states: List[np.ndarray], policies: List[np.ndarray],
                        outcome: int, annealing_steps: int):
        """Record game statistics based on game phase."""
        game_length = len(states)

        # Split game into phases
        early_moves = states[:annealing_steps // 3]
        mid_moves = states[annealing_steps // 3:annealing_steps]
        late_moves = states[annealing_steps:]

        # Calculate policy sharpness (lower entropy = more focused decisions)
        def get_policy_entropy(policies_subset):
            if not policies_subset:
                return 0.0
            entropy = -np.sum(policies_subset * np.log(policies_subset + 1e-8), axis=1)
            return np.mean(entropy)

        # Record statistics for each phase
        if early_moves:
            self.early_game_stats.append({
                'policy_entropy': get_policy_entropy(policies[:len(early_moves)]),
                'outcome': outcome
            })

        if mid_moves:
            self.mid_game_stats.append({
                'policy_entropy': get_policy_entropy(policies[len(early_moves):len(early_moves) + len(mid_moves)]),
                'outcome': outcome
            })

        if late_moves:
            self.late_game_stats.append({
                'policy_entropy': get_policy_entropy(policies[len(early_moves) + len(mid_moves):]),
                'outcome': outcome
            })

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for different game stages."""

        def summarize_stats(stats_list):
            if not stats_list:
                return {}
            return {
                'avg_temp': np.mean([s['temperature'] for s in stats_list]),
                'avg_entropy': np.mean([s['entropy'] for s in stats_list]),
                'avg_top_prob': np.mean([s['top_prob'] for s in stats_list]),
                'avg_selected_prob': np.mean([s['selected_prob'] for s in stats_list])
            }

        return {
            'early_game': summarize_stats(self.early_game_stats),
            'mid_game': summarize_stats(self.mid_game_stats),
            'late_game': summarize_stats(self.late_game_stats)
        }