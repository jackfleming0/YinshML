"""Shared helpers for analysis-board game-log deep dives.

The analysis board (analysis_board/server.py) appends one JSONL line per
/api/move to ``games/YYYY-MM-DD.jsonl``. Each line is self-contained (pre + post
position, applied move, scores, game_over, winner) and grouped into games by
``play_session_id``. From 2026-06-08 each line also carries ``mover``
("human"|"engine"); older lines don't, so engine side falls back to a
move-timing heuristic.

This module centralizes everything the deep-dive scripts share: log loading,
engine-side detection, GameState reconstruction, and marker-run / completability
primitives that reuse the engine's own geometry.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# Make the repo root importable regardless of CWD, so scripts run as
# `python scripts/foo.py` without needing PYTHONPATH=.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import analysis_board.server as _srv  # noqa: E402
from yinsh_ml.heuristics.features import _maximal_marker_runs  # noqa: E402
from yinsh_ml.game.constants import (  # noqa: E402
    Player, PieceType, Position, is_valid_position, MARKERS_FOR_ROW,
)

DEFAULT_LOG_DIR = os.path.join(
    _ROOT, "analysis_board", "multiplayer", "deploy", "games"
)


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_sessions(
    log_dir: str = DEFAULT_LOG_DIR,
    since: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {play_session_id: [events...]} in chronological order. Malformed
    lines (partial writes from a live server) are skipped. ``since`` is an ISO
    date/datetime string; events strictly before it are dropped.
    """
    cutoff = None
    if since:
        cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if cutoff is not None:
                    ts = _parse_ts(ev.get("ts"))
                    if ts is not None and ts < cutoff:
                        continue
                sessions[ev.get("play_session_id") or "__no_session__"].append(ev)
    return sessions


def completed_only(sessions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    return {sid: evs for sid, evs in sessions.items() if evs and evs[-1].get("game_over")}


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Engine-side detection (prefer provenance, fall back to timing)
# ---------------------------------------------------------------------------

def engine_side(rows: List[Dict[str, Any]]) -> Optional[str]:
    """"WHITE"|"BLACK"|None. Uses the `mover` field when present (any engine-
    tagged move pins the side); otherwise infers from move-timing regularity.
    """
    eng = {r["pre_position"]["side_to_move"] for r in rows if r.get("mover") == "engine"}
    if len(eng) == 1:
        return next(iter(eng))
    return engine_side_by_timing(rows)


def engine_side_by_timing(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Engine has a fixed per-move sim budget -> low coefficient of variation on
    MAIN_GAME move deltas; humans are erratic. Lower-CoV side is the engine.
    """
    prev = None
    wd: List[float] = []
    bd: List[float] = []
    for r in rows:
        ts = _parse_ts(r.get("ts"))
        if ts is None:
            continue
        if prev is not None:
            mv = r["pre_position"]["side_to_move"]
            ph = r["pre_position"].get("phase")
            d = (ts - prev).total_seconds()
            if ph != "RING_PLACEMENT" and d < 300:
                (wd if mv == "WHITE" else bd).append(d)
        prev = ts

    def cov(xs: List[float]) -> Optional[float]:
        if len(xs) < 2:
            return None
        m = sum(xs) / len(xs)
        return st.pstdev(xs) / m if m else None

    wc, bc = cov(wd), cov(bd)
    if wc is None or bc is None:
        return None
    return "WHITE" if wc < bc else "BLACK"


# ---------------------------------------------------------------------------
# State reconstruction + marker primitives
# ---------------------------------------------------------------------------

def build_state(position: Dict[str, Any]):
    """Reconstruct a GameState from a logged position dict (may raise)."""
    return _srv.build_state(position)


def marker_for(side: str) -> PieceType:
    return PieceType.WHITE_MARKER if side == "WHITE" else PieceType.BLACK_MARKER


def player_for(side: str) -> Player:
    return Player.WHITE if side == "WHITE" else Player.BLACK


def runs_by_length(board, marker: PieceType) -> Tuple[Dict[int, int], int]:
    """({run_length: count}, longest_run) for a color's maximal marker runs."""
    counts: Dict[int, int] = defaultdict(int)
    longest = 0
    for run in _maximal_marker_runs(board, marker):
        counts[len(run)] += 1
        longest = max(longest, len(run))
    return counts, longest


def four_runs_with_liveness(board, marker: PieceType) -> Tuple[Set[Tuple], Set[Tuple]]:
    """(all_4runs, live_4runs) where a 4-run is 'live' (geometric proxy) iff at
    least one axis-extension cell is on-board and empty. Necessary-but-not-
    sufficient for completability — use ``completing_moves`` for the real check.
    """
    all4: Set[Tuple] = set()
    live: Set[Tuple] = set()
    for run in _maximal_marker_runs(board, marker):
        if len(run) != 4:
            continue
        all4.add(run)
        pts = sorted(run, key=lambda p: (p[0], p[1]))
        (c1, r1), (c2, r2) = pts[0], pts[-1]
        n = len(pts) - 1
        dcol = (ord(c2) - ord(c1)) // n
        drow = (r2 - r1) // n
        for cc, rr in [(chr(ord(c1) - dcol), r1 - drow), (chr(ord(c2) + dcol), r2 + drow)]:
            try:
                p = Position(cc, rr)
            except Exception:
                continue
            if is_valid_position(p) and board.get_piece(p) is None:
                live.add(run)
                break
    return all4, live


def completing_moves(gs, side: str) -> List[Any]:
    """Rigorous completability: the legal moves available to ``side`` (must be
    to move) that immediately create a 5+ marker row. Simulates each candidate
    on a copy. Empty unless it's ``side``'s turn in MAIN_GAME.
    """
    if gs.current_player != player_for(side):
        return []
    marker = marker_for(side)
    out = []
    for mv in gs.get_valid_moves():
        g2 = gs.copy()
        try:
            if not g2.make_move(mv):
                continue
        except Exception:
            continue
        if any(row.length >= MARKERS_FOR_ROW for row in g2.board.find_marker_rows(marker)):
            out.append(mv)
    return out


def can_complete(gs, side: str) -> bool:
    return bool(completing_moves(gs, side))
