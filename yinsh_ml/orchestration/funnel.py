"""The tiered evaluation funnel.

Cheap screens gate access to expensive validation, so most candidates die cheap
and only the promising few earn the full gauntlet. This slice ships **Tier 0**:

  1. the failure-mode panel (nearly free — reads signals the run already produced)
  2. a small SPRT match vs the baseline, played in batches with early-stop

Tiers 1 (anchor ladder + heuristic gauntlet) and 2 (full ladder + GH engines +
human milestones) are deliberately not built yet — ``run_tier1``/``run_tier2`` are
the seams where they'll hang.

The SPRT loop is the whole point of the funnel's frugality: a decisively stronger
or weaker candidate crosses a Wald boundary in tens of decisive games instead of
grinding a fixed couple hundred (the "20h-validation-on-2h-experiments" trap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..utils.stats import SPRTResult, sprt_decision, wilson_bounds
from .failure_panel import FailurePanel, PanelInput, PanelResult
from .match_runner import MatchOutcome, MatchRunner


@dataclass
class Tier0Result:
    """Outcome of Tier-0 evaluation for one candidate."""

    panel: PanelResult
    outcome: MatchOutcome
    sprt: Optional[SPRTResult]
    wilson_lower: float
    wilson_upper: float
    had_baseline: bool

    @property
    def panel_green(self) -> bool:
        return self.panel.green

    @property
    def sprt_decisive(self) -> bool:
        return self.sprt is not None and self.sprt.decisive

    @property
    def is_clear_win(self) -> bool:
        """Decisively stronger than baseline *and* no failure-mode flags."""
        return self.panel_green and self.sprt is not None and self.sprt.verdict == "accept_h1"

    @property
    def is_clear_loss(self) -> bool:
        """Decisively no better than baseline (cheap to auto-kill)."""
        return self.sprt is not None and self.sprt.verdict == "accept_h0"

    @property
    def is_ambiguous(self) -> bool:
        """Inconclusive SPRT, or a failure-mode flag — needs human judgement."""
        if not self.panel_green:
            return True
        if not self.had_baseline:
            return True
        return self.sprt is not None and not self.sprt.decisive


class EvaluationFunnel:
    """Runs Tier-0 evaluation. Tiers 1-2 are stubbed seams."""

    def __init__(
        self,
        panel: Optional[FailurePanel] = None,
        p0: float = 0.5,
        p1: float = 0.55,
        alpha: float = 0.05,
        beta: float = 0.05,
        batch_size: int = 10,
        max_games: int = 200,
    ):
        self.panel = panel or FailurePanel()
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.max_games = max_games

    def run_tier0(
        self,
        panel_input: PanelInput,
        match_runner: Optional[MatchRunner],
    ) -> Tier0Result:
        """Run the panel, then (if a baseline exists) the early-stopping SPRT match.

        ``match_runner`` is ``None`` for a first candidate with no opponent: the
        panel still runs, but there's nothing to play against, so the result is
        left for review rather than auto-advanced.
        """
        panel = self.panel.evaluate(panel_input)

        if match_runner is None:
            return Tier0Result(
                panel=panel,
                outcome=MatchOutcome(),
                sprt=None,
                wilson_lower=0.0,
                wilson_upper=1.0,
                had_baseline=False,
            )

        outcome, sprt = self._sprt_match(match_runner)
        lower, upper = wilson_bounds(outcome.wins, outcome.decisive)
        return Tier0Result(
            panel=panel,
            outcome=outcome,
            sprt=sprt,
            wilson_lower=lower,
            wilson_upper=upper,
            had_baseline=True,
        )

    def _sprt_match(self, runner: MatchRunner) -> tuple[MatchOutcome, SPRTResult]:
        """Play in batches, checking the SPRT after each, stopping when decisive."""
        total = MatchOutcome()
        sprt = sprt_decision(0, 0, 0, self.p0, self.p1, self.alpha, self.beta)
        while total.total < self.max_games:
            n = min(self.batch_size, self.max_games - total.total)
            total = total + runner.play_batch(n)
            sprt = sprt_decision(
                total.wins, total.losses, total.draws,
                self.p0, self.p1, self.alpha, self.beta,
            )
            if sprt.decisive:
                break
        return total, sprt

    # --- Tier 1 / Tier 2 seams (not built in this slice) -----------------

    def run_tier1(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError(
            "Tier 1 (anchor ladder + heuristic gauntlet) is a planned seam; "
            "this slice ships Tier 0 only."
        )

    def run_tier2(self, *args, **kwargs):  # pragma: no cover - stub
        raise NotImplementedError(
            "Tier 2 (full ladder + GH engines + human milestones) is a planned seam; "
            "this slice ships Tier 0 only."
        )
