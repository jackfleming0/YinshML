"""Tests for the BGA Review-mode import endpoint.

Exercises the pure helpers in ``analysis_board.bga_import`` (URL parsing,
cache, rate-limit) and the Flask glue in ``analysis_board.server.api_import_bga``
end-to-end via a monkey-patched ``BGAScraper`` so the suite never touches the
network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from analysis_board import bga_import
from analysis_board.bga_import import (
    BGAImportError,
    cache_load,
    cache_save,
    check_rate_limit,
    parse_url_or_id,
    record_rate_limit,
)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

class TestParseUrlOrId:
    def test_full_url(self):
        url = "https://boardgamearena.com/5/yinsh?table=859379688"
        assert parse_url_or_id(url) == 859379688

    def test_gamereview_url(self):
        url = "https://boardgamearena.com/gamereview?table=12345&player=678"
        assert parse_url_or_id(url) == 12345

    def test_bare_integer_string(self):
        assert parse_url_or_id("859379688") == 859379688

    def test_bare_integer(self):
        assert parse_url_or_id(859379688) == 859379688

    def test_table_equals_only(self):
        assert parse_url_or_id("table=42") == 42

    def test_url_with_whitespace(self):
        assert parse_url_or_id("  https://boardgamearena.com/5/yinsh?table=100  ") == 100

    def test_rejects_empty_string(self):
        with pytest.raises(BGAImportError):
            parse_url_or_id("")

    def test_rejects_garbage(self):
        with pytest.raises(BGAImportError):
            parse_url_or_id("not a url, not a number")

    def test_rejects_non_string(self):
        with pytest.raises(BGAImportError):
            parse_url_or_id({"table": 42})

    def test_rejects_negative_int(self):
        with pytest.raises(BGAImportError):
            parse_url_or_id(-1)

    def test_url_no_table_param(self):
        # Player profile URL — no game id to extract.
        with pytest.raises(BGAImportError):
            parse_url_or_id("https://boardgamearena.com/player?id=123")


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_load_miss_returns_none(self, tmp_path):
        assert cache_load(tmp_path, 12345) is None

    def test_save_then_load(self, tmp_path):
        payload = {"steps": [{"step_index": 0}], "meta": {"x": 1}}
        cache_save(tmp_path, 12345, payload)
        loaded = cache_load(tmp_path, 12345)
        assert loaded == payload

    def test_save_is_atomic(self, tmp_path):
        # After a successful save there should be exactly one *.json file —
        # no leftover .json.tmp. Atomic rename, not write-in-place.
        cache_save(tmp_path, 12345, {"k": "v"})
        files = list(tmp_path.iterdir())
        assert [p.name for p in files] == ["12345.json"]

    def test_lru_eviction(self, tmp_path, monkeypatch):
        # Squash the cap so we don't have to save 1000 files.
        monkeypatch.setattr(bga_import, "MAX_CACHE_ENTRIES", 3)
        for i in range(5):
            cache_save(tmp_path, 1000 + i, {"i": i})
            # Tick mtime so sort order is deterministic across fast saves.
            time.sleep(0.01)
        # Only the most-recent 3 should remain.
        remaining = sorted(p.name for p in tmp_path.glob("*.json"))
        assert remaining == ["1002.json", "1003.json", "1004.json"]

    def test_load_bumps_mtime_for_lru(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bga_import, "MAX_CACHE_ENTRIES", 2)
        cache_save(tmp_path, 1, {"x": 1}); time.sleep(0.02)
        cache_save(tmp_path, 2, {"x": 2}); time.sleep(0.02)
        # Touch entry 1 — it should now look freshest.
        assert cache_load(tmp_path, 1) is not None
        time.sleep(0.02)
        cache_save(tmp_path, 3, {"x": 3})
        # Entry 2 is now the oldest (1 was bumped, 3 just written) so 2 evicts.
        remaining = sorted(p.name for p in tmp_path.glob("*.json"))
        assert remaining == ["1.json", "3.json"]


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------

class TestRateLimit:
    def setup_method(self):
        bga_import._rate_reset_for_tests()

    def test_under_limit(self):
        for _ in range(5):
            assert check_rate_limit("1.2.3.4") is True
            record_rate_limit("1.2.3.4")
        assert check_rate_limit("1.2.3.4") is False

    def test_separate_ips_independent(self):
        for _ in range(5):
            record_rate_limit("1.2.3.4")
        assert check_rate_limit("1.2.3.4") is False
        assert check_rate_limit("5.6.7.8") is True

    def test_window_expires(self):
        now = time.time()
        # Stuff 5 records dated 2 hours in the past — outside the 1h window.
        for _ in range(5):
            record_rate_limit("1.2.3.4", now=now - 7200)
        assert check_rate_limit("1.2.3.4", now=now) is True


# ---------------------------------------------------------------------------
# Flask endpoint — mocks the scraper so no network is touched.
# ---------------------------------------------------------------------------

# A tiny but legal BGA-parsed-game fixture: 10 ring placements + a couple of
# main-game moves. Covers PLACE_RING / MOVE_RING (the most common paths).
_FIXTURE_PARSED = {
    "source": "bga",
    "game_id": "bga_42",
    "players": {
        "white": {"name": "alice", "rating": 0},
        "black": {"name": "bob", "rating": 0},
    },
    "result": "white",
    "moves": [
        {"move_type": "PLACE_RING", "player": "white", "position": "E5"},
        {"move_type": "PLACE_RING", "player": "black", "position": "F6"},
        {"move_type": "PLACE_RING", "player": "white", "position": "D4"},
        {"move_type": "PLACE_RING", "player": "black", "position": "G7"},
        {"move_type": "PLACE_RING", "player": "white", "position": "E7"},
        {"move_type": "PLACE_RING", "player": "black", "position": "H8"},
        {"move_type": "PLACE_RING", "player": "white", "position": "C3"},
        {"move_type": "PLACE_RING", "player": "black", "position": "I9"},
        {"move_type": "PLACE_RING", "player": "white", "position": "B2"},
        {"move_type": "PLACE_RING", "player": "black", "position": "J10"},
    ],
}


@pytest.fixture
def server_client(tmp_path, monkeypatch):
    """Flask test client with BGA cache pointed at a tmp dir, rate-limit
    cleared, and the scraper stubbed via a configurable mock."""
    bga_import._rate_reset_for_tests()
    from analysis_board import server as _server

    monkeypatch.setattr(_server, "BGA_CACHE_DIR", tmp_path / "cache")

    state = {
        "scraper_result": _FIXTURE_PARSED,
        "scraper_raises": None,    # exception class to raise from scrape_game
        "builder_returns_none": False,
        "call_count": 0,
    }

    def fake_builder():
        if state["builder_returns_none"]:
            return None
        scraper = MagicMock()

        def scrape(table_id):
            state["call_count"] += 1
            if state["scraper_raises"] is not None:
                raise state["scraper_raises"]
            return state["scraper_result"]

        scraper.scrape_game.side_effect = scrape
        return scraper

    monkeypatch.setattr(_server, "_build_bga_scraper", fake_builder)
    client = _server.app.test_client()
    return client, state


class TestImportEndpoint:
    def test_missing_input_400(self, server_client):
        client, _ = server_client
        res = client.post("/api/import_bga", json={})
        assert res.status_code == 400
        body = res.get_json()
        assert body["ok"] is False

    def test_bad_url_400(self, server_client):
        client, _ = server_client
        res = client.post("/api/import_bga", json={"url_or_table_id": "garbage"})
        assert res.status_code == 400
        body = res.get_json()
        assert body["ok"] is False

    def test_missing_cookies_returns_friendly_error(self, server_client):
        client, state = server_client
        state["builder_returns_none"] = True
        res = client.post("/api/import_bga",
                          json={"url_or_table_id": "https://boardgamearena.com/5/yinsh?table=42"})
        assert res.status_code == 200
        body = res.get_json()
        assert body["ok"] is False
        assert any("cookies" in e.lower() for e in body["errors"])

    def test_successful_import(self, server_client):
        client, state = server_client
        res = client.post("/api/import_bga", json={"url_or_table_id": "42"})
        assert res.status_code == 200, res.get_json()
        body = res.get_json()
        assert body["ok"] is True
        assert body["table_id"] == 42
        assert body["cached"] is False
        assert body["metadata"]["players"][0]["name"] == "alice"
        # 1 starting step + 10 replayed plies = 11 steps.
        assert len(body["steps"]) == 11
        assert body["steps"][0]["move"] is None
        assert body["steps"][1]["move"]["type"] == "PLACE_RING"
        assert state["call_count"] == 1

    def test_cache_hit_skips_scraper(self, server_client):
        client, state = server_client
        # Prime the cache.
        client.post("/api/import_bga", json={"url_or_table_id": "42"})
        assert state["call_count"] == 1
        # Second call should hit cache.
        res = client.post("/api/import_bga", json={"url_or_table_id": "42"})
        body = res.get_json()
        assert body["ok"] is True
        assert body["cached"] is True
        # Scraper not called again.
        assert state["call_count"] == 1

    def test_bga_cap_hit_returns_friendly_error(self, server_client):
        client, state = server_client

        # Build the real exception class so the server's
        # `e.__class__.__name__ == "BGACapHit"` check fires.
        from yinsh_ml.data.scrapers.bga import BGACapHit
        state["scraper_raises"] = BGACapHit("daily limit reached")
        res = client.post("/api/import_bga", json={"url_or_table_id": "42"})
        body = res.get_json()
        assert body["ok"] is False
        assert any("cap" in e.lower() or "limit" in e.lower() for e in body["errors"])

    def test_failed_import_does_not_burn_rate_limit(self, server_client):
        client, state = server_client
        state["builder_returns_none"] = True
        # 6 failures should not exhaust the rate limit (which is 5/hr).
        for _ in range(6):
            res = client.post("/api/import_bga", json={"url_or_table_id": "42"})
            assert res.get_json()["ok"] is False
        # Now restore cookies and confirm the first success still goes through.
        state["builder_returns_none"] = False
        res = client.post("/api/import_bga", json={"url_or_table_id": "42"})
        assert res.get_json()["ok"] is True

    def test_rate_limit_blocks_after_5_novel_imports(self, server_client):
        client, state = server_client
        # Each tableid different so each is novel (no cache short-circuit).
        for i in range(5):
            res = client.post("/api/import_bga",
                              json={"url_or_table_id": str(1000 + i)})
            assert res.get_json()["ok"] is True
        # 6th distinct table from same IP → 429.
        res = client.post("/api/import_bga", json={"url_or_table_id": "9999"})
        assert res.status_code == 429
        assert res.get_json()["ok"] is False
