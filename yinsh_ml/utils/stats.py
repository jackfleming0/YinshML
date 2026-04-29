"""Lightweight statistics helpers for win-rate confidence intervals.

Used by the tournament gate (yinsh_ml/training/supervisor.py) and the round-robin
reporter (yinsh_ml/utils/tournament.py) so both report CIs from the same formula.
"""

import math
from typing import Tuple


def wilson_bounds(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score (lower, upper) confidence interval for a binomial proportion.

    Default z=1.96 is the 95% CI. With total==0 returns (0, 1) — the widest
    possible bound, correctly flagging "no data" so downstream straddle-checks
    classify as inconclusive rather than asserting a wrong-narrow interval.
    """
    if total == 0:
        return (0.0, 1.0)
    p_hat = wins / total
    term = max(0.0, (p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    half = z * math.sqrt(term)
    center = p_hat + z**2 / (2 * total)
    denom = 1 + z**2 / total
    lower = max(0.0, (center - half) / denom)
    upper = min(1.0, (center + half) / denom)
    return (lower, upper)


def standard_error(wins: int, total: int) -> float:
    """Binomial SE = √(p(1-p)/n). Zero when total is zero (no data)."""
    if total == 0:
        return 0.0
    p_hat = wins / total
    return math.sqrt(p_hat * (1 - p_hat) / total)
