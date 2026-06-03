"""Regression tests for the potential_runs_count fix.

Before the fix, ``potential_runs_count`` (and the inline copy in
``extract_all_features``) read maximal runs from ``Board.find_marker_rows``,
which only returns runs of length >= 5. The feature is defined over runs of
length 3-4, so the filter ``3 <= length < 5`` could never match and the
feature was identically 0 everywhere. These tests pin the corrected behavior.
"""

import pytest

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.heuristics.features import (
    potential_runs_count,
    extract_all_features,
    _maximal_marker_runs,
)


def _place(state, cells, piece):
    for c in cells:
        state.board.place_piece(Position(c[0], int(c[1:])), piece)


def test_counts_three_and_four_runs_but_not_five():
    state = GameState()
    # Black: one 3-run (C1-C3) and one 4-run (G2-G5) -> 2 potential runs.
    _place(state, ["C1", "C2", "C3"], PieceType.BLACK_MARKER)
    _place(state, ["G2", "G3", "G4", "G5"], PieceType.BLACK_MARKER)
    # White: one completed 5-run (E1-E5) -> 0 potential runs.
    _place(state, ["E1", "E2", "E3", "E4", "E5"], PieceType.WHITE_MARKER)

    # From Black's perspective: 2 (mine) - 0 (opp) = 2.
    assert potential_runs_count(state, Player.BLACK) == 2
    # Symmetry: from White's perspective it's the negative.
    assert potential_runs_count(state, Player.WHITE) == -2

    feats = extract_all_features(state, Player.BLACK)
    assert feats["potential_runs_count"] == 2.0
    # The completed white 5-run shows up in completed_runs, not potential.
    assert feats["completed_runs_differential"] == -1


def test_singletons_and_pairs_are_not_potential_runs():
    state = GameState()
    _place(state, ["C1", "E5", "G8"], PieceType.BLACK_MARKER)  # isolated singles
    _place(state, ["D1", "D2"], PieceType.BLACK_MARKER)        # a pair (len 2)
    assert potential_runs_count(state, Player.BLACK) == 0


def test_maximal_runs_helper_buckets_by_length():
    state = GameState()
    _place(state, ["C1", "C2", "C3", "C4"], PieceType.BLACK_MARKER)
    runs = _maximal_marker_runs(state.board, PieceType.BLACK_MARKER)
    # Exactly one maximal run, of length 4 (not also reported as 2s/3s).
    assert len(runs) == 1
    assert len(next(iter(runs))) == 4


def test_feature_is_live_across_a_real_game():
    """Guard against silent re-death: the feature must vary in a real game."""
    from yinsh_ml.data.human_games import bga_862307561 as game
    values = [potential_runs_count(s, Player.BLACK) for _, _, s in game.iter_states()]
    assert len(set(values)) > 1, "potential_runs_count is constant — likely dead again"
    assert any(v != 0 for v in values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
