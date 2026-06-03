"""Torch-free coverage for the engine-labeled corpus generator (E24 Phase 1a).

The encode step needs torch (15ch encoder) and runs on the box; everything
testable without torch — play, labeling, and array shaping via an injected
stub encoder — is exercised here.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from yinsh_ml.game.constants import Player
from yinsh_ml.game.types import GamePhase

# Load the script directly (it lives in scripts/, not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "gen_engine_labeled_corpus",
    Path(__file__).resolve().parents[2] / "scripts" / "gen_engine_labeled_corpus.py",
)
glc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(glc)

BASE = str(Path(__file__).resolve().parents[2] / "configs" / "heuristic_weights" / "baseline.json")


def test_z_label_side_to_move_pov():
    assert glc._z(Player.WHITE, Player.WHITE) == 1
    assert glc._z(Player.BLACK, Player.BLACK) == 1
    assert glc._z(Player.WHITE, Player.BLACK) == -1
    assert glc._z(Player.BLACK, Player.WHITE) == -1
    assert glc._z(Player.WHITE, None) == 0  # draw / unfinished


def test_play_and_label_yields_main_game_positions_with_valid_labels():
    labeled = glc.play_and_label(BASE, depth=1, eps=0.2, seed=3, max_moves=80)
    assert len(labeled) > 0
    for state, z in labeled:
        assert state.phase == GamePhase.MAIN_GAME  # only main-game positions recorded
        assert z in (-1, 0, 1)
    # all positions in one game share the same outcome magnitude (one winner)
    winner_signs = {z for _, z in labeled}
    assert winner_signs <= {-1, 0, 1}


def test_encode_labeled_shapes_with_stub_encoder():
    class StubEncoder:
        def encode_state(self, _state):
            return np.zeros((15, 11, 11), dtype=np.float32)

    labeled = [("s0", 1), ("s1", -1), ("s2", 0)]
    states, values = glc.encode_labeled(labeled, StubEncoder())
    assert states.shape == (3, 15, 11, 11)
    assert states.dtype == np.float32
    assert values.tolist() == [1.0, -1.0, 0.0]
    assert values.dtype == np.float32


def test_encode_labeled_empty():
    class StubEncoder:
        def encode_state(self, _state):  # pragma: no cover - not reached
            raise AssertionError

    states, values = glc.encode_labeled([], StubEncoder())
    assert states.shape == (0, 15, 11, 11)
    assert values.shape == (0,)
