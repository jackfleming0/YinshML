"""Tests for the reworked move encoder (Track A polish item).

Pins:
  * Collision-freeness for ring movement — every (src, dst) pair with
    src != dst maps to a distinct slot. This was the original motivation
    for the rework: the legacy `((src*31+dst) % 5848)` produced only 2687
    distinct values across 7140 pairs, a 62% collision rate.
  * Round-trip: `move_to_index` followed by `index_to_move` reproduces an
    equivalent Move for all four move types.
  * Layout: slots are disjoint, contiguous, and sum to `total_moves`.
  * Boundary-slot behavior: first + last slot of each sub-range round-trip
    cleanly.
  * Size invariant: `StateEncoder().total_moves` matches
    `YinshNetwork().total_moves` (single source of truth).
"""

import itertools
import random

import numpy as np
import pytest

from yinsh_ml.game.constants import (
    DIRECTIONS,
    MARKERS_FOR_ROW,
    PieceType,
    Player,
    is_valid_position,
)
from yinsh_ml.game.game_state import GamePhase, GameState
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.game.types import Position
from yinsh_ml.utils.encoding import (
    StateEncoder,
    _REMOVE_MARKERS_LINES,
    _LINE_TO_INDEX,
)


@pytest.fixture
def encoder():
    return StateEncoder()


class TestLayout:
    def test_total_moves_is_7433(self, encoder):
        """Pinned: 85 (placement) + 7140 (ring move) + 85 (ring removal) +
        123 (REMOVE_MARKERS one-slot-per-line) = 7433.

        Updated from 8390 when REMOVE_MARKERS moved from the collision-prone
        1080-slot sequence hash to a collision-free enumeration of the 123
        valid hex 5-in-a-row lines.
        """
        assert encoder.total_moves == 7433

    def test_ranges_are_disjoint_and_contiguous(self, encoder):
        """Every slot falls into exactly one sub-range; the ranges stack up
        to exactly `total_moves`."""
        ranges = [
            encoder.ring_place_range,
            encoder.move_ring_range,
            encoder.remove_ring_range,
            encoder.remove_markers_range,
        ]
        # Contiguous: each range starts where the last ended.
        assert ranges[0][0] == 0
        for i in range(1, len(ranges)):
            assert ranges[i][0] == ranges[i - 1][1]
        # Ends at total_moves.
        assert ranges[-1][1] == encoder.total_moves

    def test_ring_move_space_is_exactly_85x84(self, encoder):
        """Collision-free sizing: exactly `num_positions · (num_positions-1)`."""
        assert encoder.move_ring_space == encoder.num_positions * (encoder.num_positions - 1)
        assert encoder.move_ring_space == 7140

    def test_network_reads_total_moves_from_encoder(self):
        """Single-source-of-truth guard: the network's policy-head size
        comes from StateEncoder so layout bumps don't go out of sync."""
        from yinsh_ml.network.model import YinshNetwork
        net = YinshNetwork(num_blocks=1, num_channels=8)  # small for test speed
        assert net.total_moves == StateEncoder().total_moves


class TestRingMoveCollisionFreeness:
    """The original motivation: every distinct (src, dst) pair must land on
    a distinct policy slot so MCTS training targets can encode arbitrary
    move preferences without collision."""

    def test_all_pairs_distinct_indices(self, encoder):
        positions = sorted(encoder.position_to_index.keys())  # 85 valid positions
        seen = set()
        count = 0
        for src in positions:
            for dst in positions:
                if src == dst:
                    continue
                move = Move(
                    type=MoveType.MOVE_RING,
                    player=None,
                    source=Position.from_string(src),
                    destination=Position.from_string(dst),
                )
                idx = encoder.move_to_index(move)
                assert idx not in seen, (
                    f"collision: ({src}->{dst}) mapped to slot {idx} "
                    f"already seen"
                )
                seen.add(idx)
                count += 1
        assert count == 7140
        assert len(seen) == 7140  # every one distinct

    def test_indices_fill_move_ring_range_exactly(self, encoder):
        """Not just distinct — also every slot in `move_ring_range` is
        *reachable*. Legacy encoding left 3161 slots structurally
        unreachable; pin against regression."""
        positions = sorted(encoder.position_to_index.keys())
        seen = set()
        for src in positions:
            for dst in positions:
                if src == dst:
                    continue
                move = Move(
                    type=MoveType.MOVE_RING, player=None,
                    source=Position.from_string(src),
                    destination=Position.from_string(dst),
                )
                seen.add(encoder.move_to_index(move))
        expected = set(range(encoder.move_ring_range[0], encoder.move_ring_range[1]))
        assert seen == expected

    def test_same_src_rejected(self, encoder):
        """Encoding a self-loop must raise — otherwise it'd collide with
        a real move on one of the diagonal-skip boundary slots."""
        pos = Position.from_string("E5")
        with pytest.raises(ValueError, match="src == dst"):
            encoder.move_to_index(Move(
                type=MoveType.MOVE_RING, player=None,
                source=pos, destination=pos,
            ))


class TestRoundTrip:
    """Every encoded move, decoded, must reproduce an equivalent move. The
    legacy ring-movement inversion used a search loop (quadratic) because
    the hash wasn't cleanly invertible; the new one is O(1)."""

    def test_placement_roundtrip(self, encoder):
        positions = sorted(encoder.position_to_index.keys())
        for pos_str in positions:
            move = Move(
                type=MoveType.PLACE_RING, player=Player.WHITE,
                source=Position.from_string(pos_str),
            )
            idx = encoder.move_to_index(move)
            decoded = encoder.index_to_move(idx, Player.WHITE)
            assert decoded.type == MoveType.PLACE_RING
            assert str(decoded.source) == pos_str

    def test_ring_movement_roundtrip_all_pairs(self, encoder):
        positions = sorted(encoder.position_to_index.keys())
        for src, dst in itertools.permutations(positions, 2):
            move = Move(
                type=MoveType.MOVE_RING, player=Player.WHITE,
                source=Position.from_string(src),
                destination=Position.from_string(dst),
            )
            idx = encoder.move_to_index(move)
            decoded = encoder.index_to_move(idx, Player.WHITE)
            assert decoded.type == MoveType.MOVE_RING
            assert str(decoded.source) == src, f"src drift at ({src}->{dst})"
            assert str(decoded.destination) == dst, f"dst drift at ({src}->{dst})"

    def test_ring_removal_roundtrip(self, encoder):
        positions = sorted(encoder.position_to_index.keys())
        for pos_str in positions:
            move = Move(
                type=MoveType.REMOVE_RING, player=Player.BLACK,
                source=Position.from_string(pos_str),
            )
            idx = encoder.move_to_index(move)
            decoded = encoder.index_to_move(idx, Player.BLACK)
            assert decoded.type == MoveType.REMOVE_RING
            assert str(decoded.source) == pos_str


class TestBoundarySlots:
    """First and last slot of each sub-range. Off-by-one bugs in range
    dispatch live here."""

    def test_first_placement_slot(self, encoder):
        assert encoder.move_to_index(Move(
            type=MoveType.PLACE_RING, player=None,
            source=Position.from_string(sorted(encoder.position_to_index.keys())[0]),
        )) == 0

    def test_first_ring_move_slot(self, encoder):
        # Use actual integer indices, not sorted-by-string order — column-then-
        # row insertion gives indices in its own deterministic order, so
        # `sorted(keys())` would put "K10" before "K9".
        pos_by_idx = {v: k for k, v in encoder.position_to_index.items()}
        src = Position.from_string(pos_by_idx[0])
        dst = Position.from_string(pos_by_idx[1])
        # src_idx=0, dst_idx=1 → adjusted_dst = 1 - 1 = 0 (since 1 ≥ 0).
        # slot within range = 0·84 + 0 = 0, absolute = move_ring_base.
        move = Move(type=MoveType.MOVE_RING, player=None, source=src, destination=dst)
        assert encoder.move_to_index(move) == encoder.move_ring_base

    def test_last_ring_move_slot(self, encoder):
        pos_by_idx = {v: k for k, v in encoder.position_to_index.items()}
        src = Position.from_string(pos_by_idx[84])
        dst = Position.from_string(pos_by_idx[83])
        # src_idx=84, dst_idx=83 → adjusted_dst = 83 (since 83 < 84).
        # slot within range = 84·84 + 83 = 7139, absolute = 85 + 7139 = 7224
        # which is move_ring_range[1] - 1.
        move = Move(type=MoveType.MOVE_RING, player=None, source=src, destination=dst)
        assert encoder.move_to_index(move) == encoder.move_ring_range[1] - 1

    def test_first_ring_removal_slot(self, encoder):
        move = Move(
            type=MoveType.REMOVE_RING, player=None,
            source=Position.from_string(sorted(encoder.position_to_index.keys())[0]),
        )
        assert encoder.move_to_index(move) == encoder.remove_ring_base


class TestPolicyHeadSizeGuard:
    """Pin the NetworkWrapper.load_model guard that prevents silent filtering
    of a policy-head shape mismatch — otherwise an old-layout checkpoint
    would silently load with a randomly-initialized policy output layer."""

    def test_guard_logic_exists(self):
        import inspect
        from yinsh_ml.network.wrapper import NetworkWrapper
        src = inspect.getsource(NetworkWrapper.load_model)
        # Require the guard mentions both the key invariant (policy-head size)
        # and the past sizes, so a future refactor that rips it out will
        # trip this test first.
        assert "Policy-head size mismatch" in src
        assert "7395" in src   # pre-rework legacy size
        assert "8390" in src   # intermediate collision-prone REMOVE_MARKERS size


# ---------------------------------------------------------------------------
# Regression tests for the REMOVE_MARKERS collision-free rewrite
# ---------------------------------------------------------------------------


class TestRemoveMarkersLayout:
    """Pin the new REMOVE_MARKERS encoding: every valid 5-in-a-row line gets a
    unique slot, the reverse lookup returns the real line (not a fabricated
    diagonal), and the slot range matches the precomputed table."""

    def test_line_count_is_123(self):
        """Recount after the pseudo-diagonal fix reduced the axis set: there
        are exactly 123 valid hex 5-in-a-row lines on the YINSH board."""
        assert len(_REMOVE_MARKERS_LINES) == 123

    def test_all_lines_lie_on_hex_axes(self):
        """Every enumerated line is 5 valid positions stepping along one of
        the 3 matching-sign hex axes."""
        valid_directions = set(DIRECTIONS)
        for line in _REMOVE_MARKERS_LINES:
            assert len(line) == MARKERS_FOR_ROW
            for p in line:
                assert is_valid_position(p)
            p0 = line[0]
            p1 = line[1]
            step = (ord(p1.column) - ord(p0.column), p1.row - p0.row)
            assert step in valid_directions, (
                f"Line {line} has non-hex-axis step {step}"
            )
            for i, pos in enumerate(line):
                assert ord(pos.column) - ord(p0.column) == step[0] * i
                assert pos.row - p0.row == step[1] * i

    def test_all_lines_unique(self):
        """No two distinct hex-axis 5-lines share a frozenset of positions."""
        assert len({frozenset(l) for l in _REMOVE_MARKERS_LINES}) == 123

    def test_encode_round_trip_all_lines(self, encoder):
        """For every valid 5-line: move_to_index → index_to_move returns the
        exact same line (no fabrication, no collision)."""
        for line in _REMOVE_MARKERS_LINES:
            move = Move(
                type=MoveType.REMOVE_MARKERS,
                player=Player.WHITE,
                markers=tuple(line),
            )
            idx = encoder.move_to_index(move)
            decoded = encoder.index_to_move(idx, Player.WHITE)
            assert decoded.type == MoveType.REMOVE_MARKERS
            assert frozenset(decoded.markers) == frozenset(line)
            # Re-encoding the decoded move returns the same slot
            assert encoder.move_to_index(decoded) == idx

    def test_encode_distinct_for_distinct_lines(self, encoder):
        """All 123 lines map to 123 distinct slots — no collisions."""
        slots = set()
        for line in _REMOVE_MARKERS_LINES:
            move = Move(
                type=MoveType.REMOVE_MARKERS, player=None, markers=tuple(line)
            )
            slots.add(encoder.move_to_index(move))
        assert len(slots) == 123

    def test_remove_markers_range_fills_exactly(self, encoder):
        """Every slot in `remove_markers_range` is reachable by exactly one
        valid 5-line."""
        slots = set()
        for line in _REMOVE_MARKERS_LINES:
            move = Move(
                type=MoveType.REMOVE_MARKERS, player=None, markers=tuple(line)
            )
            slots.add(encoder.move_to_index(move))
        expected = set(range(*encoder.remove_markers_range))
        assert slots == expected

    def test_non_line_markers_rejected(self, encoder):
        """A 5-marker set that is NOT a hex line must raise."""
        bogus = Move(
            type=MoveType.REMOVE_MARKERS,
            player=None,
            markers=(
                Position('A', 2), Position('B', 3), Position('C', 4),
                Position('E', 5), Position('F', 6),
            ),
        )
        with pytest.raises(ValueError, match="valid 5-in-a-row"):
            encoder.move_to_index(bogus)

    def test_index_to_move_returns_valid_line(self, encoder):
        """For every slot in the REMOVE_MARKERS range, the decoded move's
        markers form a valid hex line (all on-board, along a single axis)."""
        valid_directions = set(DIRECTIONS)
        for idx in range(*encoder.remove_markers_range):
            move = encoder.index_to_move(idx, Player.BLACK)
            assert move.type == MoveType.REMOVE_MARKERS
            assert len(move.markers) == MARKERS_FOR_ROW
            for p in move.markers:
                assert is_valid_position(p)
            p0, p1 = move.markers[0], move.markers[1]
            step = (ord(p1.column) - ord(p0.column), p1.row - p0.row)
            assert step in valid_directions


# ---------------------------------------------------------------------------
# Regression tests for side-aware decode_state (Bug 1)
# ---------------------------------------------------------------------------


def _make_random_state(rng: random.Random) -> GameState:
    """Build a plausible mid-game state: each side has 3-5 rings and a handful
    of markers, in MAIN_GAME phase, either player to move."""
    gs = GameState()
    gs.phase = GamePhase.MAIN_GAME
    gs.current_player = Player.WHITE if rng.random() < 0.5 else Player.BLACK

    all_positions = [
        Position(c, r)
        for c in "ABCDEFGHIJK"
        for r in range(1, 12)
        if is_valid_position(Position(c, r))
    ]
    rng.shuffle(all_positions)

    n_white_rings = rng.randint(3, 5)
    n_black_rings = rng.randint(3, 5)
    n_white_markers = rng.randint(0, 6)
    n_black_markers = rng.randint(0, 6)

    cursor = 0
    for _ in range(n_white_rings):
        gs.board.place_piece(all_positions[cursor], PieceType.WHITE_RING)
        cursor += 1
    for _ in range(n_black_rings):
        gs.board.place_piece(all_positions[cursor], PieceType.BLACK_RING)
        cursor += 1
    for _ in range(n_white_markers):
        gs.board.place_piece(all_positions[cursor], PieceType.WHITE_MARKER)
        cursor += 1
    for _ in range(n_black_markers):
        gs.board.place_piece(all_positions[cursor], PieceType.BLACK_MARKER)
        cursor += 1

    gs.rings_placed = {Player.WHITE: n_white_rings, Player.BLACK: n_black_rings}
    return gs


def _boards_equal(a: GameState, b: GameState) -> bool:
    """Absolute-colour equality of placed pieces across two boards."""
    for pt in (PieceType.WHITE_RING, PieceType.BLACK_RING,
               PieceType.WHITE_MARKER, PieceType.BLACK_MARKER):
        if set(a.board.get_pieces_positions(pt)) != set(b.board.get_pieces_positions(pt)):
            return False
    return True


class TestEncodeDecodeRoundTrip:
    """Pin the side-aware decode_state fix.

    Before the fix, decode_state unconditionally labelled channel 0 as WHITE,
    but encode_state side-normalizes (channel 0 = current player's rings).
    For BLACK-to-move states (~half of self-play samples), this silently
    inverted colours, and augmentation._base_move_encoding enumerated valid
    moves for a swapped-colour board → policy mass got redistributed onto
    wrong squares, corrupting training labels.
    """

    def test_black_to_move_state_preserves_colors(self, encoder):
        """Minimal black-to-move state with mixed rings + markers: round-trip
        must preserve the absolute colour of every piece."""
        gs = GameState()
        gs.phase = GamePhase.MAIN_GAME
        gs.current_player = Player.BLACK
        gs.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        gs.board.place_piece(Position('F', 6), PieceType.BLACK_RING)
        gs.board.place_piece(Position('D', 4), PieceType.WHITE_MARKER)
        gs.board.place_piece(Position('G', 7), PieceType.BLACK_MARKER)
        gs.rings_placed = {Player.WHITE: 1, Player.BLACK: 1}

        encoded = encoder.encode_state(gs)
        decoded = encoder.decode_state(encoded)

        assert decoded.current_player == Player.BLACK
        assert _boards_equal(gs, decoded), \
            "Black-to-move colour round-trip must preserve absolute colours"

    def test_white_to_move_state_preserves_colors(self, encoder):
        gs = GameState()
        gs.phase = GamePhase.MAIN_GAME
        gs.current_player = Player.WHITE
        gs.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        gs.board.place_piece(Position('F', 6), PieceType.BLACK_RING)
        gs.rings_placed = {Player.WHITE: 1, Player.BLACK: 1}

        encoded = encoder.encode_state(gs)
        decoded = encoder.decode_state(encoded)
        assert decoded.current_player == Player.WHITE
        assert _boards_equal(gs, decoded)

    def test_random_states_round_trip(self, encoder):
        """100 random plausible states: board, current_player, phase, and
        rings_placed all survive round-trip."""
        rng = random.Random(1234)
        for i in range(100):
            gs = _make_random_state(rng)
            encoded = encoder.encode_state(gs)
            decoded = encoder.decode_state(encoded)

            assert decoded.current_player == gs.current_player, (
                f"sample {i}: current_player {decoded.current_player} != "
                f"{gs.current_player}"
            )
            assert decoded.phase == gs.phase, (
                f"sample {i}: phase {decoded.phase} != {gs.phase}"
            )
            assert _boards_equal(gs, decoded), (
                f"sample {i}: board round-trip failed "
                f"(current_player={gs.current_player})"
            )
            assert decoded.rings_placed == gs.rings_placed, (
                f"sample {i}: rings_placed {decoded.rings_placed} != "
                f"{gs.rings_placed}"
            )

    def test_both_perspectives_same_board_different_current_player(self, encoder):
        """Same board, flip current_player: encode→decode must recover the
        correct current_player in both cases. Guards against the sentinel
        cell being overwritten or swapped by the rest of encode_state."""
        gs = GameState()
        gs.phase = GamePhase.MAIN_GAME
        gs.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        gs.board.place_piece(Position('F', 6), PieceType.BLACK_RING)
        gs.board.place_piece(Position('D', 4), PieceType.WHITE_MARKER)

        gs.current_player = Player.WHITE
        enc_w = encoder.encode_state(gs)
        assert encoder.decode_state(enc_w).current_player == Player.WHITE

        gs.current_player = Player.BLACK
        enc_b = encoder.encode_state(gs)
        assert encoder.decode_state(enc_b).current_player == Player.BLACK
