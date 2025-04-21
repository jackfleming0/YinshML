import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    # Game statistics
    game_lengths: List[float] = field(default_factory=list)
    ring_mobility: List[float] = field(default_factory=list)
    win_rates: List[float] = field(default_factory=list)
    draw_rates: List[float] = field(default_factory=list)

    # Training performance
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    # **mutable default must use default_factory**
    extra_series  : Dict[str, List[float]] = field(default_factory=dict)

    def add_iteration_metrics(self,
                              avg_game_length : float,
                              avg_ring_mobility: float,
                              win_rate        : float,
                              draw_rate       : float,
                              policy_loss     : float,
                              value_loss      : float,
                              **extras):
        logger.debug(f"Adding metrics – win:{win_rate:.3f}, extras:{list(extras.keys())}")

        self.game_lengths .append(float(avg_game_length))
        self.ring_mobility.append(float(avg_ring_mobility))
        self.win_rates    .append(float(win_rate))
        self.draw_rates   .append(float(draw_rate))
        self.policy_losses.append(float(policy_loss))
        self.value_losses .append(float(value_loss))

        for k, v in extras.items():
            self.extra_series.setdefault(k, []).append(float(v) if v is not None else None)

    # NEW helper demanded by TrainingSupervisor
    def get_latest_metrics(self) -> Dict[str, any]:
        """Return a flat dict with the most recent entry of every series."""
        latest: Dict[str, any] = {
            "avg_game_length":  self.game_lengths[-1]   if self.game_lengths   else None,
            "avg_ring_mobility":self.ring_mobility[-1]  if self.ring_mobility  else None,
            "win_rate":         self.win_rates[-1]      if self.win_rates      else None,
            "draw_rate":        self.draw_rates[-1]     if self.draw_rates     else None,
            "policy_loss":      self.policy_losses[-1]  if self.policy_losses  else None,
            "value_loss":       self.value_losses[-1]   if self.value_losses   else None,
        }
        # include any dynamic extras
        for key, series in self.extra_series.items():
            latest[key] = series[-1] if series else None
        return latest

    def assess_stability(self, window_size: int = 5) -> Dict[str, bool]:
        """
        Assess training stability based on loss trends, stability metrics,
        and ELO progression.
        """
        if len(self.policy_losses) < window_size:
            return {'stable': False, 'reason': 'Insufficient data'}

        # Calculate trends using linear regression
        x = np.arange(window_size)
        policy_trend = np.polyfit(x, self.policy_losses[-window_size:], 1)[0]
        value_trend = np.polyfit(x, self.value_losses[-window_size:], 1)[0]

        # Calculate variances to detect oscillation/instability
        policy_variance = np.var(self.policy_losses[-window_size:])
        value_variance = np.var(self.value_losses[-window_size:])

        # Assess convergence - are we still making meaningful improvements?
        policy_relative_change = abs(policy_trend) / np.mean(self.policy_losses[-window_size:])
        value_relative_change = abs(value_trend) / np.mean(self.value_losses[-window_size:])

        checks = {
            'policy_improving': policy_trend < 0,  # Policy loss is decreasing
            'value_improving': value_trend < 0,  # Value loss is decreasing
            'policy_stable': policy_variance < 0.1,  # Not oscillating too much
            'value_stable': value_variance < 0.1,  # Not oscillating too much
            'meaningful_changes': (policy_relative_change > 0.001 or value_relative_change > 0.001),
            # Still making progress
        }

        # Only assess ELO if we have tournament data
        tournament_data = self.get_latest_tournament_summary()
        if tournament_data:
            # Compare current model's ELO with previous iteration
            current_elo = tournament_data['current_elo']
            previous_elo = tournament_data['previous_elo']
            checks['elo_improving'] = current_elo > previous_elo

        return checks

    def get_latest_tournament_summary(self) -> Optional[Dict]:
        """Get ELO comparison data from the most recent tournament."""
        if not hasattr(self, 'tournament_manager'):
            return None

        latest_results = self.tournament_manager.get_latest_tournament_summary()
        if not latest_results:
            return None

        return {
            'current_elo': latest_results['current_elo'],
            'previous_elo': latest_results['previous_elo']
        }