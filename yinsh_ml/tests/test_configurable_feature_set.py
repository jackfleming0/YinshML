"""Configurable heuristic feature set: palette features become testable.

Verifies that a weights JSON which includes an experimental palette feature
self-activates that feature in the evaluator (so it actually influences play),
that WeightManager accepts the extended set, and that the default 6-feature
behavior is unchanged.
"""

import json

import pytest

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.heuristics.weight_manager import WeightManager
from yinsh_ml.heuristics.feature_registry import PRODUCTION_FEATURES, EXPERIMENTAL_FEATURES
from yinsh_ml.heuristics import weight_fitting as wf


def _place(state, cells, piece):
    for c in cells:
        state.board.place_piece(Position(c[0], int(c[1:])), piece)


def _zeros(extra=None):
    """All-zero weights for the 6 production features, plus optional extras."""
    base = {f: 0.0 for f in PRODUCTION_FEATURES}
    if extra:
        base.update(extra)
    return {phase: dict(base) for phase in ("early", "mid", "late")}


def _disruption_state():
    """White 3-run E3-E5; black ring E1 can jump it (defensive_disruption=+1 for BLACK)."""
    state = GameState()
    _place(state, ["E3", "E4", "E5"], PieceType.WHITE_MARKER)
    _place(state, ["E1"], PieceType.BLACK_RING)
    return state


def test_default_evaluator_uses_only_production_features():
    ev = YinshHeuristics(enable_forced_sequence_detection=False)
    assert ev._feature_names == list(PRODUCTION_FEATURES)
    assert ev._extra_feature_fns == {}


def test_palette_feature_in_weights_self_activates(tmp_path):
    zero_path = tmp_path / "zero.json"
    palette_path = tmp_path / "palette.json"
    zero_path.write_text(json.dumps(_zeros()))
    palette_path.write_text(json.dumps(_zeros(extra={"defensive_disruption": 10.0})))

    state = _disruption_state()

    zero_ev = YinshHeuristics(weight_config_file=str(zero_path),
                              enable_forced_sequence_detection=False)
    palette_ev = YinshHeuristics(weight_config_file=str(palette_path),
                                 enable_forced_sequence_detection=False)

    # The palette evaluator auto-activated the feature...
    assert "defensive_disruption" in palette_ev._feature_names
    assert "defensive_disruption" in palette_ev._extra_feature_fns
    assert "defensive_disruption" not in zero_ev._feature_names

    # ...and it changes the evaluation (BLACK can disrupt one white run = +1,
    # weighted 10 -> +10), while the zero set scores it at 0.
    assert zero_ev.evaluate_position(state, Player.BLACK) == pytest.approx(0.0, abs=1e-6)
    assert palette_ev.evaluate_position(state, Player.BLACK) > 9.0


def test_explicit_feature_set_arg():
    ev = YinshHeuristics(
        feature_set=list(PRODUCTION_FEATURES) + ["ring_mobility_differential"],
        enable_forced_sequence_detection=False,
    )
    assert "ring_mobility_differential" in ev._extra_feature_fns


def test_weightmanager_accepts_palette_rejects_unknown(tmp_path):
    ext = tmp_path / "ext.json"
    bad = tmp_path / "bad.json"
    ext.write_text(json.dumps(_zeros(extra={"near_completion_threats": 3.0})))
    bad.write_text(json.dumps(_zeros(extra={"totally_bogus_feature": 1.0})))

    loaded = WeightManager().load_from_file(str(ext))
    assert loaded["early"]["near_completion_threats"] == 3.0

    with pytest.raises(ValueError, match="bogus|Unknown"):
        WeightManager().load_from_file(str(bad))


def test_fitter_extended_set_round_trips_into_evaluator(tmp_path):
    import numpy as np
    rng = np.random.default_rng(3)
    feats = wf.default_feature_set(with_experimental=True)
    samples = []
    for phase in wf.PHASES:
        for _ in range(200):
            fd = {f: float(rng.normal()) for f in feats}
            label = int(fd["defensive_disruption"] + 0.3 * rng.normal() > 0)
            samples.append((phase, fd, label))

    weights = wf.fit_weights_from_samples(samples, method="logreg", features=feats)
    path = tmp_path / "refit_ext.json"
    path.write_text(json.dumps(weights))

    # loads + activates the palette in a real evaluator
    ev = YinshHeuristics(weight_config_file=str(path),
                         enable_forced_sequence_detection=False)
    for f in EXPERIMENTAL_FEATURES:
        assert f in ev._feature_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
