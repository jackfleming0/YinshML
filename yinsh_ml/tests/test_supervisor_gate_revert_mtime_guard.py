"""T4.11: gate-revert mtime guard.

The Wilson-gate revert path in ``TrainingSupervisor`` reloads
``best_model.pt`` (or, more precisely, the per-iteration checkpoint
tracked by ``best_model_path``) when a candidate fails its gate.
``NetworkWrapper.load_model`` validates encoder shape, value-head mode,
and policy-head size — but not whether the bytes on disk match the
recorded best. If something rewrote the checkpoint between promotion
and revert, the supervisor would silently load the wrong weights.

This test suite exercises the mtime-guard helpers added in T4.11:

* ``_record_best_model_mtimes`` — captures the post-write mtime.
* ``_check_best_model_mtime_for_revert`` — compares against on-disk.
* ``_handle_mtime_mismatch`` — applies the configured policy.

We don't construct a full TrainingSupervisor (heavy: NetworkWrapper +
SelfPlay + ModelTournament). We bind the methods to a hand-built
namespace that has the few attributes they read.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from yinsh_ml.training.supervisor import TrainingSupervisor


def _make_stub(
    tmp_path: Path,
    *,
    policy: str = "fail",
    record_mtimes: bool = True,
) -> SimpleNamespace:
    """Build a minimal namespace that satisfies the T4.11 helpers."""
    save_dir = tmp_path
    save_dir.mkdir(parents=True, exist_ok=True)

    # Per-iteration checkpoint — what gets reloaded on revert.
    iter_dir = save_dir / "iteration_0"
    iter_dir.mkdir(parents=True, exist_ok=True)
    best_path = iter_dir / "checkpoint_iteration_0.pt"
    best_path.write_bytes(b"fake-checkpoint-bytes")

    # Canonical best_model.pt — kept as a parallel artifact.
    best_save_path = save_dir / "best_model.pt"
    best_save_path.write_bytes(b"fake-best-model-bytes")

    stub = SimpleNamespace(
        save_dir=save_dir,
        best_model_path=best_path,
        best_model_save_path=best_save_path,
        best_model_iteration=0,
        best_model_path_mtime=None,
        best_model_save_path_mtime=None,
        gate_revert_on_mtime_mismatch=policy,
        logger=logging.getLogger(f"test.gate_revert_guard.{policy}"),
    )
    # _check_best_model_mtime_for_revert calls self._handle_mtime_mismatch;
    # bind the real implementation onto the stub.
    stub._handle_mtime_mismatch = (  # type: ignore[attr-defined]
        lambda *, actual, recorded, reason: (
            TrainingSupervisor._handle_mtime_mismatch(
                stub, actual=actual, recorded=recorded, reason=reason
            )
        )
    )

    if record_mtimes:
        # Use the real method via the bound-method dance.
        TrainingSupervisor._record_best_model_mtimes(stub)
        assert stub.best_model_path_mtime is not None
        assert stub.best_model_save_path_mtime is not None
    return stub


# ---------------------------------------------------------------------- #
# Happy path: untouched files → check passes                             #
# ---------------------------------------------------------------------- #


def test_check_passes_when_mtime_unchanged(tmp_path):
    stub = _make_stub(tmp_path, policy="fail")
    proceed = TrainingSupervisor._check_best_model_mtime_for_revert(stub)
    assert proceed is True


# ---------------------------------------------------------------------- #
# Backward compat: pre-T4.11 state file → no mtime recorded → skip       #
# ---------------------------------------------------------------------- #


def test_check_skips_when_no_mtime_recorded(tmp_path):
    stub = _make_stub(tmp_path, policy="fail", record_mtimes=False)
    # Even with policy='fail', a missing record must not raise — that
    # would break legacy state files on first revert post-upgrade.
    proceed = TrainingSupervisor._check_best_model_mtime_for_revert(stub)
    assert proceed is True


# ---------------------------------------------------------------------- #
# Mismatch behavior under each policy                                    #
# ---------------------------------------------------------------------- #


def _touch_file(path: Path) -> None:
    """Advance the file's mtime by enough that float comparisons see it.

    Some filesystems (HFS+, older ext4) only have 1 s mtime resolution,
    so we sleep a bit before the touch. APFS (modern macOS) is ns-level
    and the sleep is overkill, but harmless.
    """
    time.sleep(1.1)
    os.utime(path, None)  # touch — set mtime to now


def test_mtime_mismatch_fail_raises(tmp_path):
    stub = _make_stub(tmp_path, policy="fail")
    _touch_file(stub.best_model_path)
    with pytest.raises(RuntimeError, match="mtime mismatch"):
        TrainingSupervisor._check_best_model_mtime_for_revert(stub)


def test_mtime_mismatch_keep_rejected_returns_false(tmp_path):
    stub = _make_stub(tmp_path, policy="keep_rejected")
    _touch_file(stub.best_model_path)
    proceed = TrainingSupervisor._check_best_model_mtime_for_revert(stub)
    assert proceed is False, (
        "keep_rejected policy must skip the revert (return False)"
    )


def test_mtime_mismatch_force_revert_returns_true(tmp_path, caplog):
    stub = _make_stub(tmp_path, policy="force_revert")
    _touch_file(stub.best_model_path)
    with caplog.at_level(logging.WARNING):
        proceed = TrainingSupervisor._check_best_model_mtime_for_revert(stub)
    assert proceed is True
    # The mismatch must still be logged (loud warning) so the operator
    # sees the inconsistency in their logs.
    assert any("mtime mismatch" in rec.message for rec in caplog.records), (
        "force_revert must still log the mismatch"
    )


# ---------------------------------------------------------------------- #
# Missing-file path: best_model_path vanished                            #
# ---------------------------------------------------------------------- #


def test_missing_best_model_path_under_fail_raises(tmp_path):
    stub = _make_stub(tmp_path, policy="fail")
    stub.best_model_path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        TrainingSupervisor._check_best_model_mtime_for_revert(stub)


def test_missing_best_model_path_under_keep_rejected_returns_false(tmp_path):
    stub = _make_stub(tmp_path, policy="keep_rejected")
    stub.best_model_path.unlink()
    proceed = TrainingSupervisor._check_best_model_mtime_for_revert(stub)
    assert proceed is False


# ---------------------------------------------------------------------- #
# JSON round-trip preserves mtime fields                                 #
# ---------------------------------------------------------------------- #


def test_save_load_roundtrip_preserves_mtime(tmp_path):
    """_save_best_model_state → _load_best_model_state → recovers mtimes."""
    stub = _make_stub(tmp_path, policy="fail")
    # Stash the recorded values so we can compare after the round-trip.
    expected_path_mtime = stub.best_model_path_mtime
    expected_save_path_mtime = stub.best_model_save_path_mtime

    # Add the bits _save_best_model_state needs.
    stub.best_model_elo = 1234.5
    stub._iteration_counter = 7

    TrainingSupervisor._save_best_model_state(stub)

    # Now reset and reload into a fresh stub to prove the JSON carried it.
    stub2 = SimpleNamespace(
        save_dir=stub.save_dir,
        best_model_iteration=-1,
        best_model_elo=-float("inf"),
        best_model_path=None,
        best_model_path_mtime=None,
        best_model_save_path_mtime=None,
        _iteration_counter=0,
        logger=logging.getLogger("test.gate_revert_guard.roundtrip"),
    )
    # _reset_best_model_state must be available because _load may call it.
    stub2._reset_best_model_state = lambda: None  # type: ignore[attr-defined]

    TrainingSupervisor._load_best_model_state(stub2)

    assert stub2.best_model_path_mtime == pytest.approx(expected_path_mtime)
    assert stub2.best_model_save_path_mtime == pytest.approx(
        expected_save_path_mtime
    )
    assert stub2.best_model_iteration == 0
    assert stub2.best_model_elo == 1234.5


def test_load_legacy_state_file_warns_and_leaves_mtime_none(tmp_path, caplog):
    """A pre-T4.11 state file (no mtime fields) → load logs a warning,
    leaves the mtime fields as None, and the run continues.
    """
    save_dir = tmp_path
    save_dir.mkdir(parents=True, exist_ok=True)
    iter_dir = save_dir / "iteration_3"
    iter_dir.mkdir()
    chk = iter_dir / "checkpoint_iteration_3.pt"
    chk.write_bytes(b"x")

    # Write a legacy-shape state file (no mtime fields).
    import json
    (save_dir / "best_model_state.json").write_text(json.dumps({
        "best_model_elo": 1500.0,
        "best_model_iteration": 3,
        "best_model_path": "iteration_3/checkpoint_iteration_3.pt",
        "_iteration_counter": 4,
    }))

    stub = SimpleNamespace(
        save_dir=save_dir,
        best_model_iteration=-1,
        best_model_elo=-float("inf"),
        best_model_path=None,
        best_model_path_mtime=None,
        best_model_save_path_mtime=None,
        _iteration_counter=0,
        logger=logging.getLogger("test.gate_revert_guard.legacy"),
    )
    stub._reset_best_model_state = lambda: None  # type: ignore[attr-defined]

    with caplog.at_level(logging.WARNING):
        TrainingSupervisor._load_best_model_state(stub)

    assert stub.best_model_path_mtime is None
    assert stub.best_model_save_path_mtime is None
    assert any(
        "T4.11" in rec.message and "no" in rec.message.lower()
        for rec in caplog.records
    ), "legacy state file load should log a T4.11 warning"
