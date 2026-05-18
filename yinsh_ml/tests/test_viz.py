"""Tests for the live game viewer modules.

Covers:
  - Hex-board renderer geometry — the most-likely-silent-regression
    surface, since the transform encodes a non-obvious coordinate
    convention (matching-sign diagonal as the third hex axis).
  - GameReplay parquet roundtrip — moves serialize and deserialize
    cleanly, and replayed Board states match the originals.
  - Capture detection via score deltas — the unambiguous audit signal.
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from yinsh_ml.game.board import Board
from yinsh_ml.game.constants import PieceType, Player, Position, VALID_POSITIONS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.self_play.data_storage import ParquetDataStorage, StorageConfig
from yinsh_ml.self_play.game_recorder import GameRecorder
from yinsh_ml.viz import (
    Annotator,
    annotate,
    captures_and_threats_annotator,
    heuristic_features_annotator,
    list_games,
    load_game,
    position_to_xy,
    render_board,
    replay_from_dataframe,
    replay_from_moves,
)


# ---------------------------------------------------------------------------
# Renderer geometry
# ---------------------------------------------------------------------------
def test_position_to_xy_origin() -> None:
    """A1 is off the hex board but the function still resolves;
    A2 sits at the origin (col_idx=0, row=2 → y = 2-1 = 1)."""
    assert position_to_xy("A", 2) == (0.0, 1.0)


def test_hex_axes_are_unit_distance() -> None:
    """All three hex-axis neighbours of any valid position must be at
    unit screen distance. This is the renderer's load-bearing
    invariant — if the transform drifts, the grid lines stop being
    60° apart and the rendered board ceases to be hexagonal."""
    axes = [(0, 1), (1, 0), (1, 1)]  # forward set of the 3 hex axes
    for col, rows in VALID_POSITIONS.items():
        for row in rows:
            x0, y0 = position_to_xy(col, row)
            for dcol, drow in axes:
                ncol = chr(ord(col) + dcol)
                nrow = row + drow
                if ncol not in VALID_POSITIONS or nrow not in VALID_POSITIONS[ncol]:
                    continue
                x1, y1 = position_to_xy(ncol, nrow)
                d = math.hypot(x1 - x0, y1 - y0)
                assert d == pytest.approx(1.0), (
                    f"axis ({dcol},{drow}) from {col}{row} produced "
                    f"distance {d:.4f} (expected 1.0)"
                )


def test_render_board_smoke() -> None:
    """A render call against a non-trivial board should not throw and
    must return a figure with at least one axis."""
    board = Board()
    board.place_piece(Position(column="E", row=5), PieceType.WHITE_RING)
    board.place_piece(Position(column="F", row=6), PieceType.BLACK_RING)
    board.place_piece(Position(column="G", row=7), PieceType.WHITE_MARKER)
    fig = render_board(
        board,
        last_move=(Position(column="E", row=5), Position(column="F", row=6)),
        highlight=[Position(column="G", row=7)],
        title="smoke",
    )
    assert fig is not None
    assert len(fig.axes) >= 1


# ---------------------------------------------------------------------------
# Replay roundtrip
# ---------------------------------------------------------------------------
def _play_random_game(seed: int, max_moves: int = 30) -> tuple:
    """Generate a random-play game and return (recorder, final_state)."""
    rng = random.Random(seed)
    recorder = GameRecorder(output_dir="/tmp/_yinsh_viz_test_scratch", save_json=False)
    state = GameState()
    recorder.start_game(f"test_game_{seed}")
    while not state.is_terminal() and len(recorder.current_game.turns) < max_moves:
        valid = list(state.get_valid_moves())
        if not valid:
            break
        move = rng.choice(valid)
        recorder.record_turn(state, move, state.current_player)
        state.make_move(move)
    return recorder, state


def test_replay_roundtrip_states_match() -> None:
    """Recorded moves replayed through GameState must reproduce the
    same Board snapshots as the original game generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = StorageConfig(output_dir=tmpdir, parquet_dir="parquet_data", batch_size=1)
        storage = ParquetDataStorage(config=cfg)

        recorder, final_state = _play_random_game(seed=11, max_moves=20)
        record = recorder.end_game(final_state, winner=None)
        storage.store_game_record(record)
        storage.flush()

        replay = load_game(storage.parquet_dir, record.game_id)
        assert replay.replay_truncated_at is None
        assert len(replay.moves) == len(record.turns)

        # Walk the moves, comparing against a fresh game.
        ref = GameState()
        for i, move in enumerate(replay.moves):
            ref.make_move(move)
            assert ref.board.pieces == replay.board_after(i).pieces, (
                f"Board divergence at turn {i + 1}"
            )


def test_list_games_summary_columns() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = StorageConfig(output_dir=tmpdir, parquet_dir="parquet_data", batch_size=1)
        storage = ParquetDataStorage(config=cfg)
        recorder, final_state = _play_random_game(seed=42, max_moves=15)
        storage.store_game_record(recorder.end_game(final_state, winner=None))
        storage.flush()

        summary = list_games(storage.parquet_dir)
        assert not summary.empty
        for col in ["game_id", "winner", "total_turns",
                    "white_score", "black_score", "source_file"]:
            assert col in summary.columns


# ---------------------------------------------------------------------------
# Capture detection via score deltas
# ---------------------------------------------------------------------------
def test_capture_detection_from_score_delta() -> None:
    """The dashboard's capture-tagging logic — synthesize a sequence of
    (white_score, black_score) tuples and verify the score-delta
    classification matches expectation. This mirrors the inline logic in
    scripts/dashboard_games.py::_compute_trajectory."""
    sequence = [
        (0, 0),  # turn 1
        (0, 0),  # turn 2
        (1, 0),  # turn 3 — WHITE capture
        (1, 0),  # turn 4
        (1, 1),  # turn 5 — BLACK capture
        (2, 1),  # turn 6 — WHITE capture
        (2, 1),  # turn 7
    ]
    expected = ["", "", "WHITE", "", "BLACK", "WHITE", ""]

    captures = []
    prev_w = prev_b = 0
    for w, b in sequence:
        if w > prev_w:
            captures.append("WHITE")
        elif b > prev_b:
            captures.append("BLACK")
        else:
            captures.append("")
        prev_w, prev_b = w, b

    assert captures == expected


def test_iter_states_full_pass() -> None:
    """GameReplay.iter_states should yield one (turn_idx, GameState)
    per recorded move, with phase + scores reachable on the state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = StorageConfig(output_dir=tmpdir, parquet_dir="parquet_data", batch_size=1)
        storage = ParquetDataStorage(config=cfg)
        recorder, final_state = _play_random_game(seed=3, max_moves=12)
        record = recorder.end_game(final_state, winner=None)
        storage.store_game_record(record)
        storage.flush()

        replay = load_game(storage.parquet_dir, record.game_id)
        seen = list(replay.iter_states())
        assert len(seen) == len(replay.moves)
        for ti, st in seen:
            # Every yielded state must be queryable for phase + scores.
            assert st.phase is not None
            assert isinstance(st.white_score, int)
            assert isinstance(st.black_score, int)


# ---------------------------------------------------------------------------
# Defensive-miss detector
# ---------------------------------------------------------------------------
def test_defensive_miss_logic_classification() -> None:
    """The detector classifies a turn as a miss iff opponent had a
    standing 4-row at the start AND it survived the player's move.
    Mirror the inline logic from dashboard_games.py::_compute_trajectory."""

    def classify(threats_before: int, threats_after: int) -> bool:
        return bool(threats_before > 0 and threats_after >= threats_before)

    # No threat → no miss possible.
    assert classify(0, 0) is False
    assert classify(0, 1) is False  # player CREATED an opponent threat (case b
                                    # path-flip side-effect); detector ignores
    # Threat present, player broke it → not a miss.
    assert classify(1, 0) is False
    assert classify(2, 1) is False  # broke one of two
    # Threat present, player did not reduce → miss.
    assert classify(1, 1) is True
    assert classify(2, 2) is True
    assert classify(1, 2) is True   # got worse


# ---------------------------------------------------------------------------
# Source-agnostic replay + annotator framework
# ---------------------------------------------------------------------------
def test_replay_from_moves_smoke() -> None:
    """`replay_from_moves` builds a usable GameReplay from any move list."""
    recorder, final_state = _play_random_game(seed=7, max_moves=15)
    # Pull moves directly out of the recorder; the parquet round-trip
    # isn't needed here — that's exactly the abstraction we're testing.
    raw_moves = []
    state = GameState()
    rng = random.Random(7)
    while not state.is_terminal() and len(raw_moves) < 15:
        valid = list(state.get_valid_moves())
        if not valid:
            break
        m = rng.choice(valid)
        raw_moves.append(m)
        state.make_move(m)

    replay = replay_from_moves(raw_moves, game_id="from_moves_test")
    assert replay.game_id == "from_moves_test"
    assert len(replay.moves) == len(raw_moves)
    assert len(replay.states) == len(raw_moves) + 1
    assert replay.replay_truncated_at is None
    # Annotations bag exists but starts empty.
    assert replay.annotations == []


def test_annotate_runs_multiple_annotators_in_one_pass() -> None:
    """`annotate` should produce one merged annotation dict per turn."""
    rng = random.Random(11)
    state = GameState()
    moves = []
    while not state.is_terminal() and len(moves) < 10:
        valid = list(state.get_valid_moves())
        if not valid:
            break
        m = rng.choice(valid)
        moves.append(m)
        state.make_move(m)

    replay = replay_from_moves(moves, game_id="ann_test")
    annotate(replay, [
        captures_and_threats_annotator(),
        heuristic_features_annotator(feature_keys=["completed_runs_differential"]),
    ])

    assert len(replay.annotations) == len(moves)
    for turn_ann in replay.annotations:
        # captures_and_threats keys
        assert "capture" in turn_ann
        assert "white_score" in turn_ann
        assert "defensive_miss" in turn_ann
        # heuristic_features keys (prefixed)
        assert "hf_completed_runs_differential" in turn_ann


def test_annotator_failure_does_not_break_replay() -> None:
    """A broken annotator should log a warning but not abort the run."""
    def bad_annotator(replay, turn_idx, state):
        raise RuntimeError("synthetic failure")

    rng = random.Random(3)
    state = GameState()
    moves = []
    while len(moves) < 6:
        valid = list(state.get_valid_moves())
        if not valid:
            break
        m = rng.choice(valid)
        moves.append(m)
        state.make_move(m)

    replay = replay_from_moves(moves, game_id="bad_ann_test")
    # Bad annotator first, working one second — should still get the
    # working one's output.
    annotate(replay, [bad_annotator, captures_and_threats_annotator()])
    assert len(replay.annotations) == len(moves)
    for turn_ann in replay.annotations:
        assert "capture" in turn_ann  # working annotator ran
