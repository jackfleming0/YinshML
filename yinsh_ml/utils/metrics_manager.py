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

    def add_iteration_metrics(self,
                              avg_game_length: float,
                              avg_ring_mobility: float,
                              win_rate: float,
                              draw_rate: float,
                              policy_loss: float,
                              value_loss: float):
        # Add logging to verify data
        logger.debug(f"Adding metrics - mobility: {avg_ring_mobility}, win_rate: {win_rate}")
        self.game_lengths.append(float(avg_game_length))  # Ensure float conversion
        self.ring_mobility.append(float(avg_ring_mobility))
        self.win_rates.append(float(win_rate))
        self.draw_rates.append(float(draw_rate))
        self.policy_losses.append(float(policy_loss))
        self.value_losses.append(float(value_loss))

    def assess_stability(self, window_size: int = 5) -> Dict[str, bool]:
        """Assess if training metrics are stable."""
        if len(self.policy_losses) < window_size:
            return {'stable': False, 'reason': 'Insufficient data'}

        recent_policy = self.policy_losses[-window_size:]
        recent_value = self.value_losses[-window_size:]
        recent_mobility = self.ring_mobility[-window_size:]

        checks = {
            'policy_loss': np.mean(np.diff(recent_policy)) < 0,  # Decreasing
            'value_loss': np.mean(np.diff(recent_value)) < 0,  # Decreasing
            'mobility_healthy': np.mean(recent_mobility) > 2.0,  # Reasonable mobility
            'game_quality': 20 < np.mean(self.game_lengths[-window_size:]) < 200
        }

        logger.info("Stability check results:")
        for metric, is_stable in checks.items():
            logger.info(f"{metric}: {'PASS' if is_stable else 'FAIL'}")

        return checks