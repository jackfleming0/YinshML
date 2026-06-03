"""Guard against silently-dead heuristic features.

Runs the production feature set across every ply of a real human game and
asserts the features that *should* respond to play actually vary. This is the
regression that would have caught the ``potential_runs_count`` bug.
"""

import pytest

from yinsh_ml.game.constants import Player
from yinsh_ml.heuristics.feature_diagnostics import feature_liveness_report
from yinsh_ml.heuristics.experimental_features import extract_experimental_features
from yinsh_ml.data.human_games import bga_862307561 as game


# Features that must respond to play. completed_runs_differential is
# deliberately excluded: it is legitimately ~constant because completed rows
# are removed within the same turn, so a standing 5-row is never observed
# between moves. That is a known, documented limitation — not a bug to assert
# away here (see docs/game_reviews/bga_862307561_review.md).
SHOULD_BE_LIVE = {
    "potential_runs_count",        # was the silently-dead bug
    "connected_marker_chains",
    "ring_positioning",
    "ring_spread",
    "board_control",
}


def _game_states():
    return [state for _turn, _mover, state in game.iter_states()]


def test_production_features_that_should_vary_do_vary():
    report, dead = feature_liveness_report(_game_states(), Player.BLACK)
    inert = SHOULD_BE_LIVE.intersection(dead)
    assert not inert, (
        f"these features never varied across a full real game (likely dead): "
        f"{sorted(inert)}; full distinct-value report: {report}"
    )


def test_potential_runs_count_specifically_is_live():
    """Pin the exact bug we fixed so it can't silently regress."""
    report, _dead = feature_liveness_report(_game_states(), Player.BLACK)
    assert report["potential_runs_count"] > 1


def test_completed_runs_is_known_inert_in_normal_play():
    """Document (don't 'fix') the legitimate inertness of completed_runs.

    If this ever starts varying, the eval/removal timing changed and the
    review notes + any replacement feature should be revisited.
    """
    report, _dead = feature_liveness_report(_game_states(), Player.BLACK)
    assert report["completed_runs_differential"] == 1


def test_experimental_palette_is_all_live():
    report, dead = feature_liveness_report(
        _game_states(), Player.BLACK, feature_fn=extract_experimental_features
    )
    assert not dead, f"experimental features that never varied: {dead}; report: {report}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
