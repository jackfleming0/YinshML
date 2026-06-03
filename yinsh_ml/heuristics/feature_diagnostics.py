"""Diagnostics for catching dead/inert heuristic features.

The review of BGA game 862307561 found two features that were silently
constant during real play: ``potential_runs_count`` (a latent bug — it read
length-3/4 runs from a >=5-only scanner) and ``completed_runs_differential``
(legitimately ~0, because completed rows are removed within the same turn).

A constant feature contributes nothing to evaluation no matter its weight. This
module makes that condition detectable: run a feature set across the states of a
real game and report which features never move.
"""

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple

from ..game.game_state import GameState
from ..game.constants import Player
from .features import extract_all_features


def feature_liveness_report(
    states: Iterable[GameState],
    player: Player,
    feature_fn: Callable[[GameState, Player], Dict[str, float]] = extract_all_features,
    *,
    round_ndigits: int = 6,
) -> Tuple[Dict[str, int], List[str]]:
    """Count distinct values each feature takes across ``states``.

    Args:
        states: iterable of game states (e.g. one per ply of a real game).
        player: perspective to evaluate features from.
        feature_fn: ``(state, player) -> {name: value}``; defaults to the
            production ``extract_all_features``. Pass
            ``extract_experimental_features`` to vet the experimental palette.
        round_ndigits: rounding applied before counting distinct values, so
            floating-point noise doesn't masquerade as variation.

    Returns:
        ``(report, dead)`` where ``report`` maps feature name -> number of
        distinct values observed, and ``dead`` is the sorted list of features
        that never varied (<=1 distinct value) — the inert signals.
    """
    series: Dict[str, set] = defaultdict(set)
    for state in states:
        for name, value in feature_fn(state, player).items():
            series[name].add(round(float(value), round_ndigits))
    report = {name: len(values) for name, values in series.items()}
    dead = sorted(name for name, count in report.items() if count <= 1)
    return report, dead
