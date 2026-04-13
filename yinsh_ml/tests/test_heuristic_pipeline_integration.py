"""Integration tests for the Yinsh heuristic evaluation pipeline.

These scenarios validate the full flow from board state preparation to
final score output, ensuring that feature extraction, phase detection,
weight application, and evaluator helpers interact correctly. The
coverage here complements the focused unit-test suites in
`yinsh_ml/tests/test_heuristics.py`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

import pytest

from yinsh_ml.game.constants import PieceType, Player, Position
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.heuristics.features import extract_all_features
from yinsh_ml.heuristics.phase_detection import GamePhaseCategory, detect_phase
from yinsh_ml.heuristics.terminal_detection import detect_terminal_position


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


RING_POSITIONS = [
    ("E5", Player.WHITE),
    ("F6", Player.BLACK),
    ("E7", Player.WHITE),
    ("F8", Player.BLACK),
    ("E9", Player.WHITE),
    ("F10", Player.BLACK),
    ("G5", Player.WHITE),
    ("H5", Player.BLACK),
    ("C3", Player.WHITE),
    ("D4", Player.BLACK),
]


def _place_initial_rings(state: GameState) -> None:
    """Populate the board with a balanced set of ring placements."""

    for pos_str, player in RING_POSITIONS:
        state.current_player = player
        move = Move(
            type=MoveType.PLACE_RING,
            player=player,
            source=Position.from_string(pos_str),
        )
        state.make_move(move)


def _add_move_history(state: GameState, count: int) -> None:
    """Append placeholder moves to the history to simulate game progress."""

    for idx in range(count):
        move = Move(
            type=MoveType.PLACE_RING,
            player=Player.WHITE if idx % 2 == 0 else Player.BLACK,
            source=Position.from_string("E5"),
        )
        state.move_history.append(move)


def _create_midgame_state() -> GameState:
    """Craft a representative mid-game position with markers on the board."""

    state = GameState()
    _place_initial_rings(state)
    _add_move_history(state, 24)

    placements = {
        "B1": PieceType.WHITE_MARKER,
        "B2": PieceType.WHITE_MARKER,
        "B3": PieceType.WHITE_MARKER,
        "C1": PieceType.BLACK_MARKER,
        "C2": PieceType.BLACK_MARKER,
        "D3": PieceType.WHITE_MARKER,
        "E4": PieceType.BLACK_MARKER,
    }

    for coord, piece in placements.items():
        state.board.place_piece(Position.from_string(coord), piece)

    return state


def _create_terminal_state(winner: Player) -> GameState:
    """Return a terminal position for the requested winner."""

    state = GameState()
    if winner is Player.WHITE:
        state.white_score = 3
        state.black_score = 0
    else:
        state.white_score = 0
        state.black_score = 3
    return state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def evaluator() -> YinshHeuristics:
    """Provide a fresh heuristic evaluator for each test."""

    return YinshHeuristics()


@pytest.fixture()
def midgame_state() -> GameState:
    """Provide a cached mid-game board used across multiple assertions."""

    return _create_midgame_state()


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def test_full_pipeline_smoke(evaluator: YinshHeuristics, midgame_state: GameState) -> None:
    """End-to-end smoke test for evaluation pipeline."""

    score_white = evaluator.evaluate_position(midgame_state, Player.WHITE)
    score_black = evaluator.evaluate_position(midgame_state, Player.BLACK)

    assert isinstance(score_white, float)
    assert isinstance(score_black, float)
    # Symmetry check ensures weights/features were blended correctly
    assert pytest.approx(score_white, rel=1e-6) == -score_black

    # Phase detection should classify this as midgame given move history
    phase = detect_phase(midgame_state)
    assert phase == GamePhaseCategory.MID

    # Features should include every registered feature name
    features = extract_all_features(midgame_state, Player.WHITE)
    expected = {
        "completed_runs_differential",
        "potential_runs_count",
        "connected_marker_chains",
        "ring_positioning",
        "ring_spread",
        "board_control",
    }
    assert expected.issubset(features.keys())

    # Terminal detection should not flag this position
    assert detect_terminal_position(midgame_state, Player.WHITE) is None


def test_weight_update_changes_score(evaluator: YinshHeuristics, midgame_state: GameState) -> None:
    """Adjusting weights must influence the resulting evaluation."""

    phase = detect_phase(midgame_state).value
    features = extract_all_features(midgame_state, Player.WHITE)
    active_feature = next(
        (name for name, value in features.items() if abs(value) > 1e-6),
        None,
    )
    assert active_feature is not None, "Expected a non-zero feature in mid-game state"

    baseline = evaluator.evaluate_position(midgame_state, Player.WHITE)

    original_weight = evaluator.get_weight(phase, active_feature)
    assert original_weight is not None

    tweaked_weight = original_weight * 1.75 if original_weight != 0 else 10.0
    evaluator.update_weight(phase, active_feature, tweaked_weight)

    try:
        adjusted = evaluator.evaluate_position(midgame_state, Player.WHITE)
    finally:
        evaluator.update_weight(phase, active_feature, original_weight)

    assert isinstance(adjusted, float)
    assert adjusted != pytest.approx(
        baseline
    ), f"Weight adjustment should modify the score for feature '{active_feature}'"


def test_batch_evaluation_matches_individual(evaluator: YinshHeuristics) -> None:
    """Batch evaluation should be consistent with per-position scoring."""

    positions: List[GameState] = []
    players: List[Player] = []

    for moves in (0, 10, 20, 35, 45):
        state = GameState()
        _place_initial_rings(state)
        _add_move_history(state, moves)
        positions.append(state)
        players.append(Player.WHITE)

    batch_scores = evaluator.evaluate_batch(positions, players)
    individual_scores = [evaluator.evaluate_position(state, player) for state, player in zip(positions, players)]

    assert len(batch_scores) == len(individual_scores)
    for batch, individual in zip(batch_scores, individual_scores):
        assert pytest.approx(batch, rel=1e-6) == individual


def test_terminal_positions_short_circuit(evaluator: YinshHeuristics) -> None:
    """Terminal states should return decisive scores without feature work."""

    white_win = _create_terminal_state(Player.WHITE)
    black_win = _create_terminal_state(Player.BLACK)

    white_score = evaluator.evaluate_position(white_win, Player.WHITE)
    black_score = evaluator.evaluate_position(black_win, Player.BLACK)

    assert white_score > 1000.0
    assert black_score > 1000.0


def test_threaded_evaluation_consistency() -> None:
    """Concurrent evaluations should remain deterministic."""

    def _evaluate_players(players: Iterable[Player]) -> List[float]:
        local_results: List[float] = []
        for player in players:
            local_state = _create_midgame_state()
            local_evaluator = YinshHeuristics()
            local_results.append(local_evaluator.evaluate_position(local_state, player))
        return local_results

    players = [Player.WHITE, Player.BLACK] * 4

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_evaluate_players, players) for _ in range(2)]
        results = [future.result() for future in futures]

    first_run = results[0]
    for run in results[1:]:
        assert run == first_run

    for score_white, score_black in zip(first_run[::2], first_run[1::2]):
        assert pytest.approx(score_white, rel=1e-6) == -score_black


def test_phase_transition_continuity(evaluator: YinshHeuristics) -> None:
    """Scores around phase boundaries should transition smoothly."""

    def _state_with_history(move_count: int) -> GameState:
        state = GameState()
        for idx in range(move_count):
            move = Move(
                type=MoveType.PLACE_RING,
                player=Player.WHITE if idx % 2 == 0 else Player.BLACK,
                source=Position.from_string("E5"),
            )
            state.move_history.append(move)
        return state

    move_counts = [14, 15, 16, 17, 34, 35, 36, 37]
    scores = []
    phases = []

    for moves in move_counts:
        state = _state_with_history(moves)
        scores.append(evaluator.evaluate_position(state, Player.WHITE))
        phases.append(detect_phase(state))

    assert phases[:4] == [
        GamePhaseCategory.EARLY,
        GamePhaseCategory.EARLY,
        GamePhaseCategory.MID,
        GamePhaseCategory.MID,
    ]
    assert phases[4:] == [
        GamePhaseCategory.MID,
        GamePhaseCategory.MID,
        GamePhaseCategory.LATE,
        GamePhaseCategory.LATE,
    ]

    deltas = [abs(b - a) for a, b in zip(scores, scores[1:])]
    assert all(delta < 1500.0 for delta in deltas)


