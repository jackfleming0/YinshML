"""Canonical registry of heuristic features.

Single source of truth for *which* features exist and how to compute the ones
beyond the optimized production six. Lets the evaluator run on a configurable
feature set (the 6 production features by default, plus any opted-in palette
features) without each call site re-listing names.

- The 6 production features are computed by the optimized
  ``features.extract_all_features`` (one marker-row scan shared across them), so
  the registry does not re-list their individual extractors for the hot path.
- Palette ("experimental") features are individually computed via
  ``EXTRA_FEATURE_FNS`` only when an experiment opts them into the active set.
"""

from typing import Callable, Dict, Tuple

from ..game.game_state import GameState
from ..game.constants import Player
from .experimental_features import EXPERIMENTAL_FEATURE_FNS

# The 6 weighted production features, in canonical order. Mirrors
# WeightManager.VALID_FEATURES and evaluator._feature_names.
PRODUCTION_FEATURES: Tuple[str, ...] = (
    "completed_runs_differential",
    "potential_runs_count",
    "connected_marker_chains",
    "ring_positioning",
    "ring_spread",
    "board_control",
)

# Opt-in palette features (computed on demand). Sorted for deterministic order.
EXPERIMENTAL_FEATURES: Tuple[str, ...] = tuple(sorted(EXPERIMENTAL_FEATURE_FNS))

# Every feature name the system knows about.
KNOWN_FEATURES: Tuple[str, ...] = PRODUCTION_FEATURES + EXPERIMENTAL_FEATURES

# name -> extractor, for the palette features only (production six come from
# the optimized extract_all_features).
EXTRA_FEATURE_FNS: Dict[str, Callable[[GameState, Player], float]] = dict(
    EXPERIMENTAL_FEATURE_FNS
)


def is_known_feature(name: str) -> bool:
    return name in KNOWN_FEATURES


def order_feature_set(names) -> list:
    """Return ``names`` ordered production-first then palette, de-duplicated,
    restricted to known features. Raises on an unknown name."""
    wanted = set(names)
    unknown = wanted - set(KNOWN_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s): {sorted(unknown)}")
    return [f for f in KNOWN_FEATURES if f in wanted]
