"""Transcribed strong-human YINSH games, for use as engine regression
fixtures and as analysis/curriculum material.

Each game module exposes the move sequence plus replay helpers that drive the
moves through our own ``GameState`` engine. The primary value is twofold:

1. **Engine correctness** — a real adversarial human game that the move
   generator + row/removal logic must reproduce exactly (legal at every ply,
   correct final score).
2. **Analysis material** — per-ply state snapshots for scoring with the
   heuristic feature set (see ``scripts/review_human_game.py``).
"""

from . import bga_862307561

__all__ = ["bga_862307561"]
