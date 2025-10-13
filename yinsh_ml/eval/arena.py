import logging
from typing import Dict, Any, Tuple

from pathlib import Path

from yinsh_ml.utils.tournament import ModelTournament, _canon


class ArenaResult(Tuple[int, int]):
    __slots__ = ()


class Arena:
    """Thin wrapper over ModelTournament to provide gating-oriented APIs."""

    def __init__(self, training_dir: Path, device: str, games_per_match: int) -> None:
        self.logger = logging.getLogger('Arena')
        self.tournament = ModelTournament(training_dir=training_dir, device=device, games_per_match=games_per_match)

    def evaluate_candidate(self, candidate_ckpt: Path) -> Dict[str, Any]:
        """Run a mini-tournament and summarize candidate stats."""
        self.tournament.run_full_round_robin_tournament(0)
        model_id = _canon(str(candidate_ckpt))
        stats = self.tournament.get_model_performance(model_id) or {}
        return stats

    def head_to_head(self, candidate_ckpt: Path, best_ckpt: Path) -> Tuple[int, int]:
        cand = _canon(str(candidate_ckpt))
        best = _canon(str(best_ckpt))
        try:
            wins, total = self.tournament.get_head_to_head(cand, best)
        except Exception:
            wins, total = 0, 0
        return wins, total


