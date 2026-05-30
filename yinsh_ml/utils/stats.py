"""Lightweight statistics helpers for win-rate confidence intervals.

Used by the tournament gate (yinsh_ml/training/supervisor.py) and the round-robin
reporter (yinsh_ml/utils/tournament.py) so both report CIs from the same formula.

Also hosts the sequential test (SPRT) used by the evaluation funnel
(yinsh_ml/orchestration/funnel.py) so a decisive match can stop early instead of
grinding a fixed sample. Wilson is the fixed-N estimator; SPRT is its sequential
sibling — both live here so the gate and the funnel share one source of truth.
"""

import math
from dataclasses import dataclass
from typing import Literal, Tuple


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


SPRTVerdict = Literal["accept_h1", "accept_h0", "continue"]


@dataclass
class SPRTResult:
    """Outcome of a sequential probability ratio test over decisive games.

    - ``accept_h1``: challenger is at/above the strength threshold (promote-worthy).
    - ``accept_h0``: challenger is no better than the null (kill-worthy).
    - ``continue``: not enough evidence yet — play more games.

    ``llr`` is the cumulative log-likelihood ratio; ``lower``/``upper`` are the
    Wald decision boundaries. Draws are excluded (they carry no information about
    the win/loss odds), so callers should keep counting them only for reporting.
    """
    verdict: SPRTVerdict
    llr: float
    lower: float
    upper: float
    wins: int
    losses: int
    draws: int

    @property
    def decisive(self) -> bool:
        """True once the test has crossed a boundary (no longer ``continue``)."""
        return self.verdict != "continue"


def sprt_decision(
    wins: int,
    losses: int,
    draws: int = 0,
    p0: float = 0.5,
    p1: float = 0.55,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> SPRTResult:
    """Wald SPRT on decisive games, testing H1 (win-prob ``p1``) vs H0 (``p0``).

    Classic binomial SPRT: each decisive game is a Bernoulli trial (win=1, loss=0)
    and draws are ignored. The cumulative log-likelihood ratio is compared against
    the Wald boundaries derived from the error rates ``alpha`` (Type I) and
    ``beta`` (Type II). This lets a clearly-stronger or clearly-weaker challenger
    resolve in tens of games instead of a fixed couple hundred.

    With no decisive games yet the result is ``continue`` (no evidence either way).
    """
    if not 0.0 < p0 < 1.0 or not 0.0 < p1 < 1.0:
        raise ValueError(f"p0/p1 must be in (0, 1), got p0={p0}, p1={p1}")
    if p1 <= p0:
        raise ValueError(f"p1 ({p1}) must be greater than p0 ({p0})")

    # Per-game log-likelihood contributions under H1 vs H0.
    win_term = math.log(p1 / p0)
    loss_term = math.log((1 - p1) / (1 - p0))
    llr = wins * win_term + losses * loss_term

    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr >= upper:
        verdict: SPRTVerdict = "accept_h1"
    elif llr <= lower:
        verdict = "accept_h0"
    else:
        verdict = "continue"

    return SPRTResult(
        verdict=verdict, llr=llr, lower=lower, upper=upper,
        wins=wins, losses=losses, draws=draws,
    )
