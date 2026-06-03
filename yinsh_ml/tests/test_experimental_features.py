"""Tests for the experimental heuristic feature palette.

These pin the intended behavior of each candidate ("flower") feature on crafted
positions, verify the house differential/antisymmetry convention, and smoke-run
the whole palette over a real human game. The palette is intentionally NOT part
of the production evaluator, so these tests stand alone.
"""

import pytest

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, Position, PieceType
from yinsh_ml.heuristics import experimental_features as ef


def _place(state, cells, piece):
    for c in cells:
        state.board.place_piece(Position(c[0], int(c[1:])), piece)


def test_marker_tempo_differential_counts_markers():
    state = GameState()
    _place(state, ["C1", "C2", "C3"], PieceType.BLACK_MARKER)
    _place(state, ["E5"], PieceType.WHITE_MARKER)
    assert ef.marker_tempo_differential(state, Player.BLACK) == 2.0
    assert ef.marker_tempo_differential(state, Player.WHITE) == -2.0


def _box_white_ring_at_a2(state):
    """White ring at A2 with all three on-board exits (A3, B3, B2) blocked by
    BLACK rings -> A2 has 0 legal moves. NB: only rings/edges block a ring;
    markers don't (a ring jumps over markers)."""
    _place(state, ["A2"], PieceType.WHITE_RING)
    _place(state, ["A3", "B3", "B2"], PieceType.BLACK_RING)


def test_ring_mobility_differential_open_vs_cramped():
    state = GameState()
    _place(state, ["F6"], PieceType.BLACK_RING)  # open center, many moves
    _box_white_ring_at_a2(state)                 # white ring fully boxed (0 moves)
    # Black (F6 + the mobile blocker rings) is strictly more mobile than the
    # single immobilized white ring.
    assert ef.ring_mobility_differential(state, Player.BLACK) > 0
    assert ef.ring_mobility_differential(state, Player.WHITE) < 0


def test_ring_confinement_pressure_rewards_boxing_opponent():
    state = GameState()
    _place(state, ["F6"], PieceType.BLACK_RING)  # mobile black ring
    _box_white_ring_at_a2(state)                 # white A2 -> 0 moves (confined)
    # The blocker rings (A3/B3/B2) each have >=2 moves, so black has no confined
    # ring; white has exactly one. Black perspective -> +1, white -> -1.
    assert ef.ring_confinement_pressure(state, Player.BLACK) == 1.0
    assert ef.ring_confinement_pressure(state, Player.WHITE) == -1.0


def test_near_completion_threats_detects_ring_on_extension():
    state = GameState()
    # Black 4-run C2-C5; black ring sitting on the C6 extension cell with
    # open space above (C7/C8 empty) -> moving it completes a 5-row.
    _place(state, ["C2", "C3", "C4", "C5"], PieceType.BLACK_MARKER)
    _place(state, ["C6"], PieceType.BLACK_RING)
    assert ef.near_completion_threats(state, Player.BLACK) == 1.0
    assert ef.near_completion_threats(state, Player.WHITE) == -1.0


def test_near_completion_threats_needs_a_ring_not_just_space():
    state = GameState()
    # Same 4-run but extension cells empty (no ring to drop a marker) -> 0.
    _place(state, ["C2", "C3", "C4", "C5"], PieceType.BLACK_MARKER)
    assert ef.near_completion_threats(state, Player.BLACK) == 0.0


def test_defensive_disruption_counts_flippable_opponent_runs():
    state = GameState()
    # White 3-run E3-E5 (maximal: E2/E6 empty).
    _place(state, ["E3", "E4", "E5"], PieceType.WHITE_MARKER)
    # Black ring at E1 can jump the three white markers and land on E6,
    # flipping E3/E4/E5 -> the white run is disruptable.
    _place(state, ["E1"], PieceType.BLACK_RING)
    # Black can break one white run; white can break none -> differential +/-1.
    assert ef.defensive_disruption(state, Player.BLACK) == 1.0
    assert ef.defensive_disruption(state, Player.WHITE) == -1.0


def test_all_features_are_antisymmetric_on_a_generic_position():
    """f(state, WHITE) == -f(state, BLACK) for every palette feature."""
    state = GameState()
    _place(state, ["F6", "G7"], PieceType.BLACK_RING)
    _place(state, ["D4", "E5"], PieceType.WHITE_RING)
    _place(state, ["C2", "C3", "C4"], PieceType.BLACK_MARKER)
    _place(state, ["H8", "H9", "H10", "H11"], PieceType.WHITE_MARKER)
    for name, fn in ef.EXPERIMENTAL_FEATURE_FNS.items():
        w = fn(state, Player.WHITE)
        b = fn(state, Player.BLACK)
        assert w == -b, f"{name} not antisymmetric: white={w} black={b}"


def test_palette_runs_over_real_game_and_varies():
    from yinsh_ml.data.human_games import bga_862307561 as game
    series = {name: [] for name in ef.EXPERIMENTAL_FEATURE_FNS}
    for _turn, _mover, state in game.iter_states():
        feats = ef.extract_experimental_features(state, Player.BLACK)
        assert set(feats) == set(ef.EXPERIMENTAL_FEATURE_FNS)
        for name, val in feats.items():
            assert isinstance(val, float)
            series[name].append(val)
    # Every flower should actually move at least once over a full game;
    # a constant feature would be a dead signal (the very bug we're guarding).
    for name, vals in series.items():
        assert len(set(vals)) > 1, f"experimental feature {name} is constant over the game"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
