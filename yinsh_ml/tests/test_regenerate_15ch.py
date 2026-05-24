"""Tests for scripts/regenerate_npz_with_enhanced_encoder.py.

The script decodes 6-channel encoded states back to GameState (via
StateEncoder.decode_state), then re-encodes via EnhancedStateEncoder
to produce 15-channel states. Path B of Branch D.2 prep (see D2_PREP.md).

These tests pin:
  1. The script runs end-to-end on a tiny synthetic corpus
  2. Output is mmap-compatible (matches convert_npz_to_mmap_shards format)
  3. Channels 0-12 + 14 round-trip cleanly (information-preserving from
     the basic encoder's perspective)
  4. Channel 13 (turn_number) is all-zeros (documented limitation)
  5. policy_indices, values, total_moves are passed through unchanged
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from yinsh_ml.game.game_state import GameState
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "regenerate_npz_with_enhanced_encoder.py"


@pytest.fixture
def tiny_6ch_corpus(tmp_path):
    """Make a tiny 5-position 6-channel npz to feed the regenerator.

    Each state is a freshly-encoded GameState — round-trip target is
    that the regenerator should reproduce these encoder outputs (minus
    channel 13, which is unrecoverable through decode).
    """
    basic = StateEncoder()
    states = []
    policy_indices = []
    values = []
    for i in range(5):
        gs = GameState()
        # Apply 0..4 placement moves to vary the positions
        from yinsh_ml.game.types import Move, MoveType
        from yinsh_ml.game.constants import Player, Position
        # Just place i rings as white (first-i moves of opening)
        opening_cells = [
            Position('F', 6), Position('E', 6), Position('G', 6),
            Position('F', 5), Position('F', 7),
        ]
        for j in range(i):
            mv = Move(type=MoveType.PLACE_RING, player=gs.current_player,
                      source=opening_cells[j])
            ok = gs.make_move(mv)
            assert ok, f"setup move {j} rejected for fixture state {i}"
        states.append(basic.encode_state(gs).astype(np.float32))
        policy_indices.append(i)  # arbitrary slot indices
        values.append(0.0 if i % 2 == 0 else 1.0)

    npz_path = tmp_path / "tiny_6ch.npz"
    np.savez_compressed(
        npz_path,
        states=np.asarray(states, dtype=np.float32),
        policy_indices=np.asarray(policy_indices, dtype=np.int32),
        values=np.asarray(values, dtype=np.float32),
        total_moves=np.int32(basic.total_moves),
    )
    return npz_path


def test_regenerator_runs_end_to_end(tiny_6ch_corpus, tmp_path):
    """Subprocess invocation succeeds and produces all expected outputs."""
    out_dir = tmp_path / "out_15ch"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(tiny_6ch_corpus),
         "--output", str(out_dir),
         "--workers", "2",
         "--chunk-size", "2"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"script failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    # Expected output files
    for name in ("states.npy", "policy_indices.npy", "values.npy",
                 "total_moves.npy", "NOTES.md"):
        assert (out_dir / name).exists(), f"missing output: {name}"


def test_output_is_mmap_compatible(tiny_6ch_corpus, tmp_path):
    """Output .npy files load via np.load(..., mmap_mode='r')."""
    out_dir = tmp_path / "out_15ch"
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(tiny_6ch_corpus), "--output", str(out_dir),
         "--workers", "1", "--chunk-size", "2"],
        check=True, capture_output=True, cwd=str(REPO_ROOT))

    states = np.load(out_dir / "states.npy", mmap_mode="r")
    assert states.shape == (5, 15, 11, 11), f"got {states.shape}"
    assert states.dtype == np.float32

    policy = np.load(out_dir / "policy_indices.npy", mmap_mode="r")
    assert policy.shape == (5,) and policy.dtype == np.int32

    values = np.load(out_dir / "values.npy", mmap_mode="r")
    assert values.shape == (5,) and values.dtype == np.float32

    total_moves = np.load(out_dir / "total_moves.npy").item()
    assert total_moves == 7433


def test_recoverable_channels_round_trip(tiny_6ch_corpus, tmp_path):
    """Channels 0-12 + 14 should match what EnhancedStateEncoder would
    produce on the decoded GameState. Channel 13 should be all zeros.

    This pins the documented limitation: decode-then-reencode is lossy
    only on channel 13 (turn_number, unrecoverable from 6ch).
    """
    out_dir = tmp_path / "out_15ch"
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(tiny_6ch_corpus), "--output", str(out_dir),
         "--workers", "1", "--chunk-size", "2"],
        check=True, capture_output=True, cwd=str(REPO_ROOT))

    out_states = np.load(out_dir / "states.npy")  # (5, 15, 11, 11)

    # Re-derive the expected output ourselves: load each 6ch input,
    # decode → re-encode, compare against script output.
    with np.load(tiny_6ch_corpus) as data:
        in_states = data['states']

    basic = StateEncoder()
    enh = EnhancedStateEncoder()
    for i in range(len(in_states)):
        gs = basic.decode_state(in_states[i].astype(np.float32))
        expected = enh.encode_state(gs)
        np.testing.assert_array_almost_equal(
            out_states[i], expected, decimal=5,
            err_msg=f"state {i}: regenerator output disagrees with direct "
                    f"decode→encode"
        )
        # Independent assertion: channel 13 must be all zeros (no move_count
        # survives the 6ch round-trip; encoder falls back to 0).
        assert (out_states[i, 13] == 0).all(), (
            f"state {i}: channel 13 (turn_number) expected all-zeros after "
            f"6ch round-trip, got nonzero values"
        )


def test_metadata_passed_through(tiny_6ch_corpus, tmp_path):
    """policy_indices, values, total_moves must match the input exactly."""
    out_dir = tmp_path / "out_15ch"
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(tiny_6ch_corpus), "--output", str(out_dir),
         "--workers", "1", "--chunk-size", "5"],
        check=True, capture_output=True, cwd=str(REPO_ROOT))

    with np.load(tiny_6ch_corpus) as data:
        in_policy = data['policy_indices']
        in_values = data['values']
        in_total = int(data['total_moves'].item())

    out_policy = np.load(out_dir / "policy_indices.npy")
    out_values = np.load(out_dir / "values.npy")
    out_total = int(np.load(out_dir / "total_moves.npy").item())

    np.testing.assert_array_equal(in_policy, out_policy)
    np.testing.assert_array_equal(in_values, out_values)
    assert in_total == out_total


def test_rejects_wrong_channel_count(tmp_path):
    """If someone hands the regenerator an already-15ch corpus (or any
    non-6ch input), it should fail cleanly with a clear error rather than
    silently producing garbage."""
    bad_npz = tmp_path / "bad_15ch.npz"
    np.savez_compressed(
        bad_npz,
        states=np.zeros((3, 15, 11, 11), dtype=np.float32),
        policy_indices=np.zeros(3, dtype=np.int32),
        values=np.zeros(3, dtype=np.float32),
        total_moves=np.int32(7433),
    )

    out_dir = tmp_path / "should_not_exist"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(bad_npz), "--output", str(out_dir),
         "--workers", "1"],
        capture_output=True, text=True, cwd=str(REPO_ROOT))

    assert result.returncode != 0, (
        "script should reject non-6ch input but exited 0"
    )
    combined = result.stdout + result.stderr
    assert "6-channel" in combined or "shape" in combined, (
        f"expected error message about 6-channel input, got:\n{combined}"
    )


def test_max_positions_truncates(tiny_6ch_corpus, tmp_path):
    """--max-positions should process only the first N positions."""
    out_dir = tmp_path / "out_truncated"
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--input", str(tiny_6ch_corpus), "--output", str(out_dir),
         "--workers", "1", "--chunk-size", "10",
         "--max-positions", "3"],
        check=True, capture_output=True, cwd=str(REPO_ROOT))

    states = np.load(out_dir / "states.npy")
    assert states.shape == (3, 15, 11, 11), (
        f"--max-positions=3 should produce 3 states, got {states.shape[0]}"
    )
