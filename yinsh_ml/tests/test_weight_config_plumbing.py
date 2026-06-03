"""Verify re-fit weights JSON flows into the evaluator (plumbing A).

These exercise the artifact contract the training plumbing relies on:
``YinshHeuristics(weight_config_file=...)`` loads the JSON and the loaded
weights actually change the evaluation. (The supervisor/self_play wiring that
carries the path is torch-heavy and not importable here; this pins the load
contract those layers depend on.)
"""

import json

import pytest

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.heuristics import weight_fitting as wf


def _weights(potential_weight):
    """All-zero weights except potential_runs_count in every phase."""
    return {
        phase: {f: (potential_weight if f == "potential_runs_count" else 0.0)
                for f in wf.PRODUCTION_FEATURES}
        for phase in wf.PHASES
    }


def _quiet_state_with_black_potential_run():
    state = GameState()
    # One black 3-run (a potential run); nothing tactical/terminal.
    for c in ["C1", "C2", "C3"]:
        state.board.place_piece(Position(c[0], int(c[1:])), PieceType.BLACK_MARKER)
    return state


def test_weight_config_file_changes_evaluation(tmp_path):
    zero_path = tmp_path / "zero.json"
    live_path = tmp_path / "live.json"
    zero_path.write_text(json.dumps(_weights(0.0)))
    live_path.write_text(json.dumps(_weights(10.0)))

    state = _quiet_state_with_black_potential_run()

    zero_eval = YinshHeuristics(
        weight_config_file=str(zero_path), enable_forced_sequence_detection=False
    ).evaluate_position(state, Player.BLACK)
    live_eval = YinshHeuristics(
        weight_config_file=str(live_path), enable_forced_sequence_detection=False
    ).evaluate_position(state, Player.BLACK)

    # With all weights zero the quiet position scores 0; giving
    # potential_runs_count a positive weight makes black's 3-run register.
    assert zero_eval == pytest.approx(0.0, abs=1e-6)
    assert live_eval > zero_eval


def test_none_path_falls_back_to_default_weights():
    """weight_config_file=None must not error and must use default weights."""
    default = YinshHeuristics(enable_forced_sequence_detection=False)
    explicit_none = YinshHeuristics(weight_config_file=None,
                                    enable_forced_sequence_detection=False)
    state = _quiet_state_with_black_potential_run()
    assert (default.evaluate_position(state, Player.BLACK)
            == explicit_none.evaluate_position(state, Player.BLACK))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
