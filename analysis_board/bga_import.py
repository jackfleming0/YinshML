"""BGA game-import for the Review-mode UI.

Pure-Python helpers that the Flask layer wires into ``/api/import_bga``:
  * ``parse_url_or_id`` — extracts a BGA table id from either a full URL
    or a bare integer string.
  * ``cache_load`` / ``cache_save`` — JSON-on-disk cache keyed by table id,
    with LRU eviction by mtime once the cache passes ``MAX_CACHE_ENTRIES``.
  * ``check_rate_limit`` / ``record_rate_limit`` — per-IP throttle on novel
    (non-cache-hit) imports. Failures are not recorded; the cache and the
    rate limit together keep us well under BGA's 200/day per-account cap.
  * ``replay_to_steps`` — replays a parsed BGA game through ``GameState``
    and emits a step-per-ply payload the frontend can step through with
    prev/next buttons.

The heavy engine imports (GameState, Move, ...) live inside the function
that needs them so the module stays cheap to import in tests.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Hard upper bound on entries kept on disk. ~50 KB per game × 1000 = ~50 MB —
# fine on the Mac mini. Eviction is LRU by mtime: oldest gets unlinked when a
# new save pushes us above the bound.
MAX_CACHE_ENTRIES = 1000

# Per-IP throttle. The cache absorbs repeat hits on the same game; this caps
# how many *novel* fetches a single IP can spend per window. Failures don't
# count — so a 401-on-missing-cookies doesn't burn the user's allowance.
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW_SECONDS = 3600

# ``table=<digits>`` in a query string is how BGA encodes table ids in both
# the player-facing URL (``/5/yinsh?table=...``) and the developer-facing
# replay URL (``/gamereview?table=...``). Anchoring on the literal "table="
# keeps us robust to upstream URL-shape changes that don't touch that param.
_TABLE_ID_PAT = re.compile(r"\btable=(\d+)")


class BGAImportError(Exception):
    """Raised by the import pipeline with a user-facing message.

    The Flask layer renders ``user_message`` as a JSON ``errors`` entry. The
    default 200 status code mirrors the rest of the analysis-board API: we
    surface caller-facing errors in the body, not via HTTP status, so the
    frontend has a single code path for parsing responses.
    """

    def __init__(self, message: str, status: int = 200):
        super().__init__(message)
        self.user_message = message
        self.status = status


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def parse_url_or_id(value: Any) -> int:
    """Extract a BGA table id from a URL or bare integer string.

    Accepts (illustrative — anything containing ``table=<digits>`` works):
        * ``"https://boardgamearena.com/5/yinsh?table=859379688"``
        * ``"https://boardgamearena.com/gamereview?table=859379688&...&"``
        * ``"table=859379688"``
        * ``"859379688"``
        * ``859379688``    (int passed through)

    Raises ``BGAImportError`` (400) for anything else.
    """
    if isinstance(value, int):
        if value <= 0:
            raise BGAImportError("table id must be positive", status=400)
        return value
    if not isinstance(value, str):
        raise BGAImportError("url_or_table_id must be a string", status=400)
    s = value.strip()
    if not s:
        raise BGAImportError("url_or_table_id is empty", status=400)
    m = _TABLE_ID_PAT.search(s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    raise BGAImportError(
        f"could not extract a BGA table id from {value!r}",
        status=400,
    )


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def cache_load(cache_dir: Path, table_id: int) -> Optional[Dict[str, Any]]:
    """Return the cached payload for ``table_id``, or None on miss.

    Bumps the file's mtime on every hit so the LRU eviction in
    ``cache_save`` sees recently-accessed games as fresh.
    """
    path = cache_dir / f"{int(table_id)}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        os.utime(path, None)
    except OSError:
        pass
    return data


def cache_save(cache_dir: Path, table_id: int, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to the cache and evict any LRU overflow.

    Uses atomic write-then-rename (``.json.tmp`` → ``.json``) so a crash
    mid-write doesn't leave a partial file that ``cache_load`` would later
    return as garbage.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{int(table_id)}.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    tmp.replace(path)
    _evict_lru(cache_dir)


def _evict_lru(cache_dir: Path, max_entries: Optional[int] = None) -> None:
    # Read MAX_CACHE_ENTRIES from the module at call time, not as a default-arg
    # binding — that way tests can monkeypatch ``bga_import.MAX_CACHE_ENTRIES``
    # to shrink the cap without re-importing the module.
    if max_entries is None:
        max_entries = MAX_CACHE_ENTRIES
    if not cache_dir.exists():
        return
    entries = sorted(
        cache_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(entries) > max_entries:
        old = entries.pop(0)
        try:
            old.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-IP rate limiting
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
# ip → list of timestamps (UNIX seconds) of accepted novel imports
_rate_tracker: Dict[str, List[float]] = {}


def check_rate_limit(ip: str, *, now: Optional[float] = None) -> bool:
    """Return True if ``ip`` is under the limit, False if throttled.

    Does *not* record the attempt — ``record_rate_limit`` does. Two-step
    so failed imports (missing cookies, BGACapHit, bad URL) can be skipped
    without burning the user's quota.
    """
    if now is None:
        now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_lock:
        recent = [t for t in _rate_tracker.get(ip, []) if t >= cutoff]
        _rate_tracker[ip] = recent
        return len(recent) < RATE_LIMIT_MAX


def record_rate_limit(ip: str, *, now: Optional[float] = None) -> None:
    if now is None:
        now = time.time()
    with _rate_lock:
        recent = [t for t in _rate_tracker.get(ip, []) if t >= now - RATE_LIMIT_WINDOW_SECONDS]
        recent.append(now)
        _rate_tracker[ip] = recent


def _rate_reset_for_tests() -> None:
    """Clear the per-IP tracker. Test-only."""
    with _rate_lock:
        _rate_tracker.clear()


# ---------------------------------------------------------------------------
# Parsed-replay → step-by-step playback
# ---------------------------------------------------------------------------

def replay_to_steps(
    parsed: Dict[str, Any],
    serialize_state: Callable[..., Dict[str, Any]],
    serialize_move: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    """Replay every move in ``parsed`` through ``GameState`` and emit a
    step-per-ply payload.

    Step 0 is the starting empty board (move=null). Step k (k≥1) records
    the move that landed us there plus the position-after, in the same
    schema ``/api/evaluate`` expects — the frontend can hand each step's
    ``position`` straight back to the evaluator for engine commentary.

    Serializers are injected (instead of imported) to avoid a circular
    import with ``analysis_board.server``.
    """
    # Lazy imports keep this module light for tests that don't need the
    # engine. The full chain pulls torch via the network wrapper.
    from yinsh_ml.game.constants import Player, Position
    from yinsh_ml.game.game_state import GameState
    from yinsh_ml.game.types import Move, MoveType

    PLAYER_MAP = {"white": Player.WHITE, "black": Player.BLACK}

    moves_in = parsed.get("moves") or []
    if not moves_in:
        raise BGAImportError("parsed replay contained no moves")

    gs = GameState()
    steps: List[Dict[str, Any]] = [
        {
            "step_index": 0,
            "move": None,
            "player": None,
            "position": serialize_state(gs),
            "ply_label": "start",
        }
    ]

    for idx, mv in enumerate(moves_in, start=1):
        engine_move = _bga_to_engine_move(mv, PLAYER_MAP, Position, Move, MoveType)
        try:
            ok = gs.make_move(engine_move)
        except Exception as e:  # noqa: BLE001
            raise BGAImportError(
                f"replay failed at ply {idx} ({engine_move}): {e}"
            )
        if not ok:
            raise BGAImportError(
                f"replay rejected ply {idx} ({engine_move}) — parsed moves "
                f"may have diverged from engine rules"
            )
        steps.append({
            "step_index": idx,
            "move": serialize_move(engine_move),
            "player": engine_move.player.name,
            "position": serialize_state(gs),
            "ply_label": _ply_label(idx, engine_move.player.name),
        })

    metadata = _build_metadata(parsed, gs)
    return {"metadata": metadata, "steps": steps}


def _bga_to_engine_move(mv: Dict[str, Any], player_map, Position, Move, MoveType):
    mt = str(mv.get("move_type", "")).upper()
    if mt not in MoveType.__members__:
        raise BGAImportError(f"unknown move_type in parsed replay: {mt!r}")
    type_ = MoveType[mt]
    player_str = str(mv.get("player", "")).lower()
    if player_str not in player_map:
        raise BGAImportError(f"unknown player in parsed replay: {player_str!r}")
    player = player_map[player_str]

    if type_.name == "PLACE_RING":
        return Move(type=type_, player=player,
                    source=Position.from_string(mv["position"]))
    if type_.name == "MOVE_RING":
        return Move(type=type_, player=player,
                    source=Position.from_string(mv["source"]),
                    destination=Position.from_string(mv["destination"]))
    if type_.name == "REMOVE_MARKERS":
        markers = tuple(Position.from_string(p) for p in (mv.get("markers") or ()))
        if not markers:
            raise BGAImportError("REMOVE_MARKERS move has no markers")
        return Move(type=type_, player=player, markers=markers)
    if type_.name == "REMOVE_RING":
        return Move(type=type_, player=player,
                    source=Position.from_string(mv["position"]))
    raise BGAImportError(f"unhandled move_type: {type_.name}")


def _ply_label(idx: int, player_name: str) -> str:
    short = "W" if player_name == "WHITE" else "B"
    return f"{idx} {short}"


def _build_metadata(parsed: Dict[str, Any], gs: Any) -> Dict[str, Any]:
    players_in = parsed.get("players") or {}
    out_players: List[Dict[str, Any]] = []
    for color in ("white", "black"):
        side = players_in.get(color) or {}
        out_players.append({
            "color": color.upper(),
            "name": side.get("name", "unknown"),
            # BGA's /archive endpoint doesn't return ELO; rating=0 means
            # "unknown" rather than "0 ELO".
            "rating": int(side.get("rating", 0) or 0),
        })
    result_str = str(parsed.get("result") or "").lower()
    if result_str in ("white", "black"):
        winner = result_str.upper()
    else:
        winner = None
    return {
        "players": out_players,
        "result": {
            "winner": winner,
            "score": f"{int(gs.white_score)}-{int(gs.black_score)}",
        },
        "source": parsed.get("source"),
        "game_id": parsed.get("game_id"),
    }
