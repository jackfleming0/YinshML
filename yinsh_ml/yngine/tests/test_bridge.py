"""Integration tests for the yngine_driver subprocess bridge.

Skipped automatically if the driver binary hasn't been built. Build with::

    bash third_party/yngine_driver/build.sh

These tests are marked ``slow`` because each test spawns a subprocess and
exercises a real MCTS search; the cluster as a whole runs in <30s on Apple
Silicon but pulls in real compiled code so we don't want it on every
unit-test run.
"""

from __future__ import annotations

import pytest

from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.yngine import Yngine, YngineError, default_binary_path

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def _binary_ready():
    bin_path = default_binary_path()
    if not bin_path.exists():
        pytest.skip(
            f"yngine_driver not built at {bin_path}. "
            "Run `bash third_party/yngine_driver/build.sh` first."
        )
    return bin_path


def test_handshake_and_clean_shutdown(_binary_ready):
    eng = Yngine.start()
    try:
        st = eng.state()
        assert st["next"] == "place"
        assert st["turn"] == "W"
        assert st["result"] == "ongoing"
    finally:
        rc = eng.stop()
    assert rc == 0


def test_context_manager(_binary_ready):
    with Yngine.start() as eng:
        assert eng.state()["next"] == "place"
    # Implicit stop on exit — no exception.


def test_search_returns_legal_placement(_binary_ready):
    with Yngine.start() as eng:
        mv, wire = eng.get_move(player=Player.WHITE, sims=50)
        assert mv.type == MoveType.PLACE_RING
        assert mv.source is not None
        # Round-trip the wire form back through apply_wire should succeed.
        eng.apply_wire(wire)
        # State should now be black's turn to place.
        st = eng.state()
        assert st["next"] == "place"
        assert st["turn"] == "B"


def test_full_placement_phase_against_our_engine(_binary_ready):
    """Drive 10 alternating placements: yngine picks 5, we pick 5, both
    boards stay in sync. Validates the codec round-trip on a real flow."""
    our_state = GameState()
    with Yngine.start() as eng:
        for i in range(10):
            if i % 2 == 0:
                # yngine plays for the side to move
                mv, wire = eng.get_move(player=our_state.current_player, sims=20)
                assert our_state.make_move(mv), f"yngine move {mv} rejected"
                eng.apply_wire(wire)
            else:
                # We pick the lexicographically first valid placement
                valid = our_state.get_valid_moves()
                place = sorted(
                    (m for m in valid if m.type == MoveType.PLACE_RING),
                    key=lambda m: (m.source.column, m.source.row),
                )[0]
                assert our_state.make_move(place)
                eng.apply(place)
        # Both sides agree about whose turn it is.
        st = eng.state()
        expected_turn = "W" if our_state.current_player == Player.WHITE else "B"
        assert st["turn"] == expected_turn


def test_invalid_move_returns_err(_binary_ready):
    with Yngine.start() as eng:
        # Place a ring at A2 to seed state, then try to place again at the
        # same spot — yngine should reject as illegal.
        mv = Move(type=MoveType.PLACE_RING, player=Player.WHITE,
                  source=Position('A', 2))
        eng.apply(mv)
        with pytest.raises(YngineError, match="illegal"):
            eng.apply(mv)


def test_garbage_move_returns_err(_binary_ready):
    """A wire string that doesn't parse should produce 'err parse'."""
    with Yngine.start() as eng:
        with pytest.raises(YngineError, match="parse"):
            # Skip the codec and inject malformed wire text directly.
            eng.apply_wire("Z 999 999")


def test_new_resets_position(_binary_ready):
    with Yngine.start() as eng:
        mv = Move(type=MoveType.PLACE_RING, player=Player.WHITE,
                  source=Position('E', 5))
        eng.apply(mv)
        assert eng.state()["turn"] == "B"
        eng.new_game()
        assert eng.state()["turn"] == "W"
