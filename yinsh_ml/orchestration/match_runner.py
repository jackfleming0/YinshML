"""Pluggable match execution for the evaluation funnel.

The funnel needs games *incrementally* — play a small batch, ask the SPRT whether
it can stop, play more only if it can't. That requires an interface narrower than
``ModelTournament``: just "play N more games and tell me the tally". This protocol
is also the seam that keeps the funnel testable without a GPU (tests inject a fake
runner) and defers the heavy tournament/model code until a real match is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class MatchOutcome:
    """Win/draw/loss tally from the candidate's perspective."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def decisive(self) -> int:
        """Games that weren't draws (the ones the SPRT consumes)."""
        return self.wins + self.losses

    def __add__(self, other: "MatchOutcome") -> "MatchOutcome":
        return MatchOutcome(
            wins=self.wins + other.wins,
            draws=self.draws + other.draws,
            losses=self.losses + other.losses,
        )


@runtime_checkable
class MatchRunner(Protocol):
    """Plays games between a candidate and a baseline, a batch at a time."""

    def play_batch(self, n: int) -> MatchOutcome:
        """Play ``n`` more games, alternating colors, candidate-vs-baseline.

        Returns the tally for *this batch only* (the funnel accumulates).
        """
        ...


class TournamentMatchRunner:
    """Real adapter over ``ModelTournament._play_match``.

    Loads the candidate and baseline checkpoints once, then plays in chunks so the
    SPRT can short-circuit. The heavy ``ModelTournament``/``NetworkWrapper`` code is
    imported lazily inside ``__init__`` so importing the orchestration package
    doesn't drag the tournament engine (torch itself still arrives transitively via
    ``yinsh_ml.utils``).

    Not exercised in CI here (no GPU/checkpoints in the container) — its job is to
    be the production wiring behind the same interface the fake runner satisfies.
    """

    def __init__(
        self,
        candidate_ckpt: str,
        baseline_ckpt: str,
        training_dir: str,
        device: str = "mps",
        use_enhanced_encoding: bool = False,
        value_head_type: Optional[str] = None,
        eval_seed: Optional[int] = 0,
    ):
        from pathlib import Path
        from ..utils.tournament import ModelTournament

        self._candidate_path = Path(candidate_ckpt)
        self._baseline_path = Path(baseline_ckpt)
        # games_per_match is set per-batch in play_batch via a fresh tournament call,
        # so seed it small here; the real per-batch count comes from the funnel.
        self._tournament = ModelTournament(
            training_dir=Path(training_dir),
            device=device,
            games_per_match=2,
            eval_seed=eval_seed,
            use_enhanced_encoding=use_enhanced_encoding,
            value_head_type=value_head_type,
        )
        self._candidate = self._tournament._load_model(self._candidate_path)
        self._baseline = self._tournament._load_model(self._baseline_path)
        self._played = 0

    def play_batch(self, n: int) -> MatchOutcome:
        """Play ``n`` games, splitting colors evenly to cancel first-move bias."""
        half = max(1, n // 2)
        cand_id, base_id = self._candidate_path.stem, self._baseline_path.stem

        self._tournament.games_per_match = half
        # Candidate as white.
        r1 = self._tournament._play_match(
            self._candidate, self._baseline, cand_id, base_id
        )
        # Candidate as black (swap seats); its wins are the match's black_wins.
        r2 = self._tournament._play_match(
            self._baseline, self._candidate, base_id, cand_id
        )

        wins = _white_wins(r1) + _black_wins(r2)
        losses = _black_wins(r1) + _white_wins(r2)
        draws = _draws(r1) + _draws(r2)
        self._played += wins + losses + draws
        return MatchOutcome(wins=wins, draws=draws, losses=losses)


def _white_wins(match_result) -> int:
    return getattr(match_result, "white_wins", 0)


def _black_wins(match_result) -> int:
    return getattr(match_result, "black_wins", 0)


def _draws(match_result) -> int:
    return getattr(match_result, "draws", 0)
