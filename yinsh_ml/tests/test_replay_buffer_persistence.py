"""Tests for GameExperience replay buffer save/load persistence.

Covers the round-trip (save -> load) and the `.pkl` -> `.pkl.gz` fallback
that `YinshTrainer.__init__` relies on when locating a previously persisted
buffer. See commit fb7fb10.
"""

import os
import numpy as np
import pytest

from yinsh_ml.training.trainer import GameExperience


def _make_fake_buffer(num_entries: int = 100, policy_size: int = 7433) -> GameExperience:
    """Populate a GameExperience directly (bypassing add_game_experience
    subsampling / phase decoding) so contents are deterministic and easy to
    compare after a round-trip."""
    buf = GameExperience(max_size=num_entries * 2, subsample_long_games=False)
    rng = np.random.default_rng(42)
    for i in range(num_entries):
        state = rng.standard_normal((6, 11, 11)).astype(np.float32)
        policy = rng.random(policy_size).astype(np.float16)
        policy /= policy.sum()
        value = float(rng.uniform(-1.0, 1.0))
        buf.states.append(state)
        buf.move_probs.append(policy)
        buf.values.append(value)
        buf.phases.append("MAIN_GAME")
        buf.move_numbers.append(i)
    return buf


def _assert_buffers_equal(a: GameExperience, b: GameExperience) -> None:
    assert a.size() == b.size()
    for s1, s2 in zip(a.states, b.states):
        np.testing.assert_array_equal(s1, s2)
    for p1, p2 in zip(a.move_probs, b.move_probs):
        np.testing.assert_array_equal(p1, p2)
    assert list(a.values) == list(b.values)
    assert list(a.phases) == list(b.phases)
    assert list(a.move_numbers) == list(b.move_numbers)


def test_save_and_load_roundtrip_compressed(tmp_path):
    original = _make_fake_buffer(num_entries=100)
    save_path = str(tmp_path / "replay_buffer.pkl")

    original.save_buffer(save_path, compress=True)

    # compress=True appends .gz; base path should NOT exist but .gz should.
    assert not os.path.exists(save_path)
    assert os.path.exists(save_path + ".gz")

    restored = GameExperience(max_size=500, subsample_long_games=False)
    restored.load_buffer(save_path + ".gz")
    _assert_buffers_equal(original, restored)


def test_save_and_load_roundtrip_uncompressed(tmp_path):
    original = _make_fake_buffer(num_entries=50)
    save_path = str(tmp_path / "replay_buffer.pkl")

    original.save_buffer(save_path, compress=False)

    assert os.path.exists(save_path)

    restored = GameExperience(max_size=500, subsample_long_games=False)
    restored.load_buffer(save_path)
    _assert_buffers_equal(original, restored)


def test_trainer_init_fallback_pkl_to_gz(tmp_path):
    """Supervisor passes '<save_dir>/replay_buffer.pkl' (no .gz), but
    save_buffer writes '<save_dir>/replay_buffer.pkl.gz'. Simulate the
    fallback logic used in YinshTrainer.__init__."""
    original = _make_fake_buffer(num_entries=30)
    base_path = str(tmp_path / "replay_buffer.pkl")

    original.save_buffer(base_path, compress=True)
    # Only the .gz exists
    assert not os.path.exists(base_path)
    assert os.path.exists(base_path + ".gz")

    # Replicate the trainer's fallback resolution (trainer.py:637-648)
    candidate = base_path
    if not os.path.exists(candidate):
        gz = base_path if base_path.endswith(".gz") else base_path + ".gz"
        if os.path.exists(gz):
            candidate = gz

    assert candidate == base_path + ".gz"

    restored = GameExperience(max_size=500, subsample_long_games=False)
    restored.load_buffer(candidate)
    _assert_buffers_equal(original, restored)


def test_load_missing_file_is_nonfatal(tmp_path, caplog):
    """load_buffer swallows exceptions and logs an error — a missing file
    must not crash trainer init."""
    buf = GameExperience(max_size=100, subsample_long_games=False)
    bogus_path = str(tmp_path / "does_not_exist.pkl.gz")

    # Should not raise.
    buf.load_buffer(bogus_path)
    assert buf.size() == 0


def test_load_buffer_respects_new_max_size(tmp_path):
    """Regression: load_buffer uses self.max_size, so a buffer saved at
    size N can be loaded into a buffer with a larger cap without
    silently capping at N."""
    original = _make_fake_buffer(num_entries=20)
    save_path = str(tmp_path / "replay_buffer.pkl")
    original.save_buffer(save_path, compress=True)

    # New buffer with a much bigger cap.
    restored = GameExperience(max_size=10_000, subsample_long_games=False)
    restored.load_buffer(save_path + ".gz")

    assert restored.size() == 20
    assert restored.states.maxlen == 10_000
