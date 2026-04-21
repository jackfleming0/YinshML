"""Tests for the BGA replay parser.

Exercises `BGAScraper.parse_raw` purely offline against checked-in
fixtures — no network, no cookies. Protects the parser from regressions
when the raw BGA payload structure shifts.
"""

import json
from pathlib import Path

import pytest

from yinsh_ml.data.scrapers.bga import (
    BGACapHit,
    BGAScraper,
    _parse_bga_replay,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'bga'


def _load(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


class TestParseMinimalGame:
    """Happy-path: one notification of each YINSH move kind."""

    def setup_method(self):
        self.raw = _load('raw_minimal_game.json')
        self.parsed = BGAScraper.parse_raw(self.raw, table_id=1234)

    def test_not_none(self):
        assert self.parsed is not None

    def test_game_metadata(self):
        assert self.parsed['source'] == 'bga'
        assert self.parsed['game_id'] == 'bga_1234'
        assert self.parsed['quality_tier'] == 'human'

    def test_color_mapping(self):
        # #ffffff → white, #000000 → black — the inverted mapping
        # regression caught in the 2026-04-13 parser rewrite.
        players = self.parsed['players']
        assert players['white']['name'] == 'Alice'
        assert players['black']['name'] == 'Bob'

    def test_result_from_gameend(self):
        assert self.parsed['result'] == 'white'

    def test_move_sequence(self):
        moves = self.parsed['moves']
        # PLACE_RING x2, MOVE_RING x1, REMOVE_MARKERS x1, REMOVE_RING x1
        # (the implicit PLACE_MARKER is intentionally dropped)
        assert len(moves) == 5
        types = [m['move_type'] for m in moves]
        assert types == [
            'PLACE_RING', 'PLACE_RING',
            'MOVE_RING', 'REMOVE_MARKERS', 'REMOVE_RING',
        ]

    def test_place_marker_is_dropped(self):
        # Regression guard: BGA emits a 'places a marker on' notification
        # immediately before every MOVE_RING; the marker square is
        # redundant with MOVE_RING.locationFrom, so the parser skips it.
        assert not any(m['move_type'] == 'PLACE_MARKER'
                       for m in self.parsed['moves'])

    def test_move_ring_payload(self):
        move_ring = next(m for m in self.parsed['moves']
                         if m['move_type'] == 'MOVE_RING')
        assert move_ring['player'] == 'white'
        assert move_ring['source'] == 'J6'
        assert move_ring['destination'] == 'J8'

    def test_remove_ring_uses_location_from(self):
        # BGA puts the board square in locationFrom; locationTo is a
        # reserve sentinel like '@0'. Regression caught in the
        # 2026-04-13 rewrite.
        ring_rem = next(m for m in self.parsed['moves']
                        if m['move_type'] == 'REMOVE_RING')
        assert ring_rem['position'] == 'J6'

    def test_remove_markers_expands_to_line(self):
        row = next(m for m in self.parsed['moves']
                   if m['move_type'] == 'REMOVE_MARKERS')
        # locationFrom/locationTo are the row endpoints; parser fills in
        # the five squares between them.
        assert row['markers'][0] == 'E5'
        assert row['markers'][-1] == 'I5'
        assert len(row['markers']) == 5


class TestRestartUndo:
    """restartUndo pops the most recent move by the same player."""

    def test_undone_move_is_absent(self):
        raw = _load('raw_with_undo.json')
        parsed = BGAScraper.parse_raw(raw, table_id=99)
        assert parsed is not None
        # Bob placed B2, undid it, then placed K10. B2 should not appear.
        positions = [
            m.get('position') for m in parsed['moves']
            if m['move_type'] == 'PLACE_RING' and m['player'] == 'black'
        ]
        assert 'B2' not in positions
        assert positions == ['K10']


class TestRestartsTurnMultiAction:
    """'restarts their turn' pops the whole current-turn run of actions.

    Regression: the parser used to treat "restarts their turn" like
    "takes back their last move" and pop only a single action, leaving
    2-4 stale notifications that corrupted the replayed game state.
    Evidence: bga_835096709 notif #532 where a full place-marker + move-
    ring + remove-row + remove-ring sequence was only half-undone.
    """

    def setup_method(self):
        raw = _load('raw_restarts_turn.json')
        self.parsed = BGAScraper.parse_raw(raw, table_id=42)

    def test_parsed_not_none(self):
        assert self.parsed is not None

    def test_undone_full_turn_is_absent(self):
        # Black's first turn (B2 → B4, remove A1..A5 row, remove K11 ring)
        # was entirely restart-undone. None of those squares should appear
        # in the output move list.
        moves = self.parsed['moves']
        for m in moves:
            if m['move_type'] == 'MOVE_RING':
                assert m['source'] != 'B2'
                assert m['destination'] != 'B4'
            if m['move_type'] == 'REMOVE_RING':
                assert m['position'] != 'K11'
            if m['move_type'] == 'REMOVE_MARKERS':
                # The undone row removal was A1..A5.
                assert 'A1' not in m['markers']

    def test_replayed_turn_is_present(self):
        # After the restart, black replayed: place marker C3, move ring C3→C5.
        # Only MOVE_RING survives (place-marker is implicit in MOVE_RING.source).
        moves = self.parsed['moves']
        black_ring_moves = [
            m for m in moves
            if m['move_type'] == 'MOVE_RING' and m['player'] == 'black'
        ]
        assert len(black_ring_moves) == 1
        assert black_ring_moves[0]['source'] == 'C3'
        assert black_ring_moves[0]['destination'] == 'C5'

    def test_no_stale_remove_ring_or_remove_markers(self):
        # The half-undone path left orphan REMOVE_ROW / REMOVE_RING events
        # in the move list; with the fix, black emits only the replayed
        # ring move (no removes at all in this fixture).
        moves = self.parsed['moves']
        assert not any(m['move_type'] == 'REMOVE_RING' for m in moves)
        assert not any(m['move_type'] == 'REMOVE_MARKERS' for m in moves)

    def test_take_back_behavior_unchanged(self):
        # Sanity: the existing single-pop path for "takes back their last move"
        # still works after the multi-pop fix.
        raw = _load('raw_with_undo.json')
        parsed = BGAScraper.parse_raw(raw, table_id=100)
        positions = [
            m.get('position') for m in parsed['moves']
            if m['move_type'] == 'PLACE_RING' and m['player'] == 'black'
        ]
        assert positions == ['K10']


class TestRestartsTurnReplaysCleanly:
    """End-to-end: parsed output of a 'restarts their turn' stream feeds
    through GameState.make_move without rejection.

    This mirrors the audit harness in scripts/audit_bga_replays.py on a
    small synthetic case, so the regression is caught even if the raw
    BGA fixture directory goes out of sync.
    """

    def test_synthetic_restart_place_rings_replay(self):
        from yinsh_ml.game.constants import Player, Position
        from yinsh_ml.game.game_state import GameState
        from yinsh_ml.game.types import Move, MoveType

        raw = _load('raw_restarts_turn.json')
        parsed = BGAScraper.parse_raw(raw, table_id=42)
        assert parsed is not None

        def _p(s): return Position.from_string(s)
        def _player(c): return Player.WHITE if c == 'white' else Player.BLACK

        # Only replay the ring-placement opening from our fixture —
        # full mid-game replay requires all 10 rings placed first. The
        # key regression fact is that the parser emits no stale
        # REMOVE_RING / REMOVE_MARKERS from the undone half-turn
        # (covered by TestRestartsTurnMultiAction). Here we verify the
        # two place-rings at the top still apply cleanly, i.e. the
        # multi-pop heuristic doesn't also chew into the opening.
        state = GameState()
        place_rings = [m for m in parsed['moves']
                       if m['move_type'] == 'PLACE_RING']
        assert len(place_rings) == 2
        for m in place_rings:
            mv = Move(type=MoveType.PLACE_RING, player=_player(m['player']),
                      source=_p(m['position']))
            assert state.make_move(mv), (
                f"PLACE_RING {m['position']} was rejected after restart-undo "
                f"parsing — multi-pop heuristic may have eaten too much"
            )


class TestParseRawErrorHandling:
    """parse_raw is pure — no network, no exceptions on bad input."""

    def test_empty_returns_none(self):
        assert BGAScraper.parse_raw({}, table_id=1) is None

    def test_non_dict_data_returns_none(self):
        assert BGAScraper.parse_raw({'data': 'not a dict'}, table_id=1) is None

    def test_no_moves_returns_none(self):
        raw = {
            'status': 'OK',
            'data': {
                'players': [
                    {'id': '1', 'name': 'x', 'color': '#ffffff'},
                    {'id': '2', 'name': 'y', 'color': '#000000'},
                ],
                'logs': [],
            },
        }
        assert BGAScraper.parse_raw(raw, table_id=1) is None

    def test_module_level_helper_matches_classmethod(self):
        # The classmethod is a thin wrapper; both should agree.
        raw = _load('raw_minimal_game.json')
        assert BGAScraper.parse_raw(raw, 7) == _parse_bga_replay(raw, 7)


class TestCapHitException:
    """BGACapHit is exported for bulk crawlers to catch."""

    def test_is_exception_subclass(self):
        assert issubclass(BGACapHit, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(BGACapHit):
            raise BGACapHit('daily limit exceeded')


class TestFetchRawCapDetection:
    """`fetch_raw` must raise BGACapHit on rate-limit payloads.

    Regression: BGA returns `{"status": "0", ...}` (string, not int) when
    the per-account replay cap is hit. An earlier version compared against
    int 0, which silently let the error response through as a "successful"
    raw dump. Bulk crawlers then wrote garbage files and marked tids as
    seen — poisoning seen.json so valid tids would never be retried.
    """

    def _make_scraper(self, monkeypatch, fake_payload):
        scraper = BGAScraper(delay=0)
        scraper._logged_in = True
        monkeypatch.setattr(scraper, '_fetch', lambda url, **kw: None)
        monkeypatch.setattr(scraper, '_fetch_json', lambda url, **kw: fake_payload)
        return scraper

    def test_string_status_zero_with_limit_raises(self, monkeypatch):
        payload = {
            'status': '0',
            'exception': 'feException',
            'error': 'You have reached a limit (replay)',
            'code': 100,
        }
        scraper = self._make_scraper(monkeypatch, payload)
        with pytest.raises(BGACapHit):
            scraper.fetch_raw(999)

    def test_int_status_zero_with_limit_raises(self, monkeypatch):
        payload = {'status': 0, 'error': 'replay limit reached'}
        scraper = self._make_scraper(monkeypatch, payload)
        with pytest.raises(BGACapHit):
            scraper.fetch_raw(999)

    def test_string_status_zero_without_limit_returns_none(self, monkeypatch):
        payload = {'status': '0', 'error': 'table not found'}
        scraper = self._make_scraper(monkeypatch, payload)
        assert scraper.fetch_raw(999) is None

    def test_successful_payload_passes_through(self, monkeypatch):
        payload = {'status': 1, 'data': {'logs': [], 'players': {}}}
        scraper = self._make_scraper(monkeypatch, payload)
        assert scraper.fetch_raw(999) == payload
