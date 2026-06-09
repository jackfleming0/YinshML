#!/usr/bin/env python3
"""Analyze JSONL game logs produced by the analysis board (server.py).

Reads every ``*.jsonl`` file in ``analysis_board/multiplayer/deploy/games/``
(or a directory specified via ``--log-dir``), groups events by
``play_session_id``, and emits per-session summaries.

The server writes one line per successful ``/api/move`` call. Each line is
self-contained (pre + post position, applied move, game_over, winner, etc.),
and a client-generated ``play_session_id`` groups moves into discrete games.

Common usage::

    # summary table — one line per session, oldest first
    python scripts/analyze_game_logs.py

    # add opening / capture / move-mix detail per session
    python scripts/analyze_game_logs.py -v

    # only show sessions that reached game_over
    python scripts/analyze_game_logs.py --completed-only

    # deep dive on one session — prints every move
    python scripts/analyze_game_logs.py --session-id 5b9bce1d-...

    # machine-readable
    python scripts/analyze_game_logs.py --json > summaries.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_LOG_DIR = (
    Path(__file__).resolve().parent.parent
    / "analysis_board" / "multiplayer" / "deploy" / "games"
)


# ---------------------------------------------------------------------------
# Event ingest
# ---------------------------------------------------------------------------

def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def iter_events(log_dir: Path, since: Optional[datetime]) -> Iterator[Dict[str, Any]]:
    """Yield every event in chronological-by-file order. Malformed lines
    (partial writes from a live server append) are skipped silently.
    """
    for path in sorted(log_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since is not None:
                    ts = _parse_ts(ev.get("ts"))
                    if ts is not None and ts < since:
                        continue
                yield ev


def group_by_session(events: Iterator[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        sid = ev.get("play_session_id") or "__no_session__"
        sessions[sid].append(ev)
    return sessions


# ---------------------------------------------------------------------------
# Per-session summary computation
# ---------------------------------------------------------------------------

def _centroid(positions: List[str]) -> Optional[Tuple[float, float]]:
    if not positions:
        return None
    cols = [ord(p[0]) - ord("A") for p in positions]
    rows = [int(p[1:]) for p in positions]
    return (statistics.mean(cols), statistics.mean(rows))


def _spread(positions: List[str]) -> Optional[float]:
    """Mean Euclidean distance from the centroid — how dispersed the rings
    are. Low = clustered, high = spread across the board."""
    if not positions or len(positions) < 2:
        return None
    cx, cy = _centroid(positions)
    distances = []
    for p in positions:
        x = ord(p[0]) - ord("A")
        y = int(p[1:])
        distances.append(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)
    return statistics.mean(distances)


def _format_centroid(c: Optional[Tuple[float, float]]) -> str:
    if c is None:
        return "?"
    col = chr(int(round(c[0])) + ord("A"))
    row = int(round(c[1]))
    return f"~{col}{row}"


def summarize_session(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {}
    first = events[0]
    last = events[-1]
    start_ts = _parse_ts(first.get("ts"))
    end_ts = _parse_ts(last.get("ts"))
    duration_s = (end_ts - start_ts).total_seconds() if start_ts and end_ts else None

    move_types: Dict[str, int] = defaultdict(int)
    placements_white: List[str] = []
    placements_black: List[str] = []
    captures: List[Dict[str, Any]] = []

    for i, ev in enumerate(events):
        applied = ev.get("applied_move") or {}
        mtype = applied.get("type")
        if mtype:
            move_types[mtype] += 1
        # Who's moving — read from the pre-move side_to_move (always correct,
        # unlike the applied_move's player field which can be None).
        side = (ev.get("pre_position") or {}).get("side_to_move")
        if mtype == "PLACE_RING":
            src = applied.get("source")
            if src:
                if side == "WHITE":
                    placements_white.append(src)
                elif side == "BLACK":
                    placements_black.append(src)
        elif mtype == "REMOVE_MARKERS":
            captures.append({
                "move_num": i + 1,
                "type": "row_capture",
                "markers": applied.get("markers") or [],
                "side": side,
            })
        elif mtype == "REMOVE_RING":
            captures.append({
                "move_num": i + 1,
                "type": "ring_capture",
                "source": applied.get("source"),
                "side": side,
            })

    last_winner = last.get("winner")
    last_game_over = bool(last.get("game_over"))
    final_position = last.get("new_position") or {}
    final_scores = final_position.get("scores") or {}

    # Move provenance (logged from 2026-06-08; absent in older logs). Derive
    # which color the engine played from the per-move `mover` tags, so we can
    # report engine win/loss directly instead of guessing from move timing.
    engine_sides: set = set()
    engine_meta: Optional[Dict[str, Any]] = None
    have_provenance = False
    for ev in events:
        mover = ev.get("mover")
        if mover is None:
            continue
        have_provenance = True
        side = (ev.get("pre_position") or {}).get("side_to_move")
        if mover == "engine":
            if side:
                engine_sides.add(side)
            if engine_meta is None and ev.get("mover_meta"):
                engine_meta = ev.get("mover_meta")
    if len(engine_sides) == 1:
        engine_side: Optional[str] = next(iter(engine_sides))
    elif engine_sides:
        engine_side = "MIXED"  # engine moved as both colors — unexpected
    else:
        engine_side = None
    engine_result: Optional[str] = None
    if last_game_over and engine_side in ("WHITE", "BLACK"):
        if last_winner == engine_side:
            engine_result = "engine_won"
        elif last_winner is None:
            engine_result = "draw"
        else:
            engine_result = "engine_lost"

    return {
        "play_session_id": first.get("play_session_id"),
        "start": start_ts.isoformat() if start_ts else None,
        "end": end_ts.isoformat() if end_ts else None,
        "duration_s": duration_s,
        "num_moves": len(events),
        "move_types": dict(move_types),
        "white_placements": placements_white,
        "black_placements": placements_black,
        "white_opening_centroid": _centroid(placements_white),
        "black_opening_centroid": _centroid(placements_black),
        "white_opening_spread": _spread(placements_white),
        "black_opening_spread": _spread(placements_black),
        "captures": captures,
        "completed": last_game_over,
        "winner": last_winner,
        "final_scores": final_scores,
        "has_provenance": have_provenance,
        "engine_side": engine_side,
        "engine_result": engine_result,
        "engine_meta": engine_meta,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _outcome_str(summary: Dict[str, Any]) -> str:
    if not summary.get("completed"):
        return "(incomplete)"
    winner = summary.get("winner")
    scores = summary.get("final_scores") or {}
    ws = scores.get("WHITE", "?")
    bs = scores.get("BLACK", "?")
    base = f"{winner or 'DRAW'} {ws}-{bs}"
    # Append the engine verdict when the log carries move provenance.
    er = summary.get("engine_result")
    es = summary.get("engine_side")
    if er == "engine_won":
        base += f"  [engine WON as {es}]"
    elif er == "engine_lost":
        base += f"  [engine LOST as {es}]"
    elif er == "draw" and es in ("WHITE", "BLACK"):
        base += f"  [engine drew as {es}]"
    return base


def format_one_line(summary: Dict[str, Any]) -> str:
    sid_short = (summary.get("play_session_id") or "?")[:8]
    start = (summary.get("start") or "?")[:19].replace("T", " ")
    dur = summary.get("duration_s")
    dur_str = f"{dur:.0f}s" if dur is not None else "?"
    nm = summary.get("num_moves") or 0
    return f"{sid_short:<8}  {start:<19}  {dur_str:>6}  {nm:>3}  {_outcome_str(summary)}"


def format_verbose_block(summary: Dict[str, Any]) -> str:
    wp = summary.get("white_placements") or []
    bp = summary.get("black_placements") or []
    wc = _format_centroid(summary.get("white_opening_centroid"))
    bc = _format_centroid(summary.get("black_opening_centroid"))
    ws = summary.get("white_opening_spread")
    bs = summary.get("black_opening_spread")

    lines: List[str] = []
    lines.append(f"  White opening: {','.join(wp) or '(none)'}")
    lines.append(f"    centroid {wc}, spread {ws:.2f}" if ws is not None else f"    centroid {wc}, spread ?")
    lines.append(f"  Black opening: {','.join(bp) or '(none)'}")
    lines.append(f"    centroid {bc}, spread {bs:.2f}" if bs is not None else f"    centroid {bc}, spread ?")

    captures = summary.get("captures") or []
    if captures:
        lines.append(f"  Captures ({len(captures)}):")
        for cap in captures:
            if cap["type"] == "row_capture":
                markers = ",".join(cap.get("markers") or [])
                lines.append(f"    move {cap['move_num']:>3}  {cap.get('side', '?')}: row [{markers}]")
            else:
                lines.append(f"    move {cap['move_num']:>3}  {cap.get('side', '?')}: ring @ {cap.get('source')}")
    else:
        lines.append("  Captures: (none)")

    types = summary.get("move_types") or {}
    type_str = ", ".join(f"{k}={v}" for k, v in sorted(types.items()))
    lines.append(f"  Move mix: {type_str}")
    return "\n".join(lines)


def format_full_move_list(events: List[Dict[str, Any]]) -> str:
    lines = ["  Moves:"]
    for i, ev in enumerate(events):
        applied = ev.get("applied_move") or {}
        side = (ev.get("pre_position") or {}).get("side_to_move") or "?"
        desc = applied.get("description") or json.dumps(applied, separators=(",", ":"))
        lines.append(f"    {i+1:>3}  {side[:1]}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate stats across sessions
# ---------------------------------------------------------------------------

def aggregate_stats(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(summaries)
    completed = [s for s in summaries if s.get("completed")]
    white_wins = sum(1 for s in completed if s.get("winner") == "WHITE")
    black_wins = sum(1 for s in completed if s.get("winner") == "BLACK")
    draws = sum(1 for s in completed if s.get("winner") is None)
    avg_dur = (
        statistics.mean(s["duration_s"] for s in completed if s.get("duration_s") is not None)
        if completed else None
    )
    avg_moves = (
        statistics.mean(s["num_moves"] for s in completed if s.get("num_moves") is not None)
        if completed else None
    )
    # Engine record over completed games that carry provenance (mover tags).
    engine_games = [s for s in completed if s.get("engine_side") in ("WHITE", "BLACK")]
    engine_wins = sum(1 for s in engine_games if s.get("engine_result") == "engine_won")
    engine_losses = sum(1 for s in engine_games if s.get("engine_result") == "engine_lost")
    engine_draws = sum(1 for s in engine_games if s.get("engine_result") == "draw")
    return {
        "total_sessions": total,
        "completed": len(completed),
        "incomplete": total - len(completed),
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_completed_duration_s": avg_dur,
        "avg_completed_move_count": avg_moves,
        "engine_games": len(engine_games),
        "engine_wins": engine_wins,
        "engine_losses": engine_losses,
        "engine_draws": engine_draws,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help=f"Directory of *.jsonl logs. Default: {DEFAULT_LOG_DIR}")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Deep-dive a single session (matched by prefix).")
    parser.add_argument("--since", type=str, default=None,
                        help="ISO timestamp; skip events before this. e.g. 2026-05-28 or 2026-05-28T20:00:00Z")
    parser.add_argument("--completed-only", action="store_true",
                        help="Hide sessions that didn't reach game_over.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print opening/capture/move-mix detail per session.")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON (overrides text formatting).")
    args = parser.parse_args()

    log_dir: Path = args.log_dir
    if not log_dir.exists():
        print(f"No log dir at {log_dir}")
        return

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    all_events = list(iter_events(log_dir, since))
    if not all_events:
        print(f"No events found in {log_dir}" + (f" since {args.since}" if since else ""))
        return

    sessions = group_by_session(iter(all_events))

    # Single-session deep dive
    if args.session_id:
        matched = [(sid, evs) for sid, evs in sessions.items() if sid.startswith(args.session_id)]
        if not matched:
            print(f"No session matching prefix {args.session_id!r}.")
            return
        if len(matched) > 1:
            print(f"Prefix {args.session_id!r} matches {len(matched)} sessions; please disambiguate:")
            for sid, _ in matched:
                print(f"  {sid}")
            return
        sid, evs = matched[0]
        summary = summarize_session(evs)
        if args.json:
            out = {"summary": summary, "events": evs}
            print(json.dumps(out, indent=2, default=str))
            return
        print(format_one_line(summary))
        print(format_verbose_block(summary))
        print(format_full_move_list(evs))
        return

    # Multi-session summary
    summaries = [summarize_session(evs) for evs in sessions.values()]
    if args.completed_only:
        summaries = [s for s in summaries if s.get("completed")]
    summaries.sort(key=lambda s: s.get("start") or "")

    if args.json:
        print(json.dumps({
            "stats": aggregate_stats(summaries),
            "sessions": summaries,
        }, indent=2, default=str))
        return

    stats = aggregate_stats(summaries)
    print(f"\n{log_dir}  ({stats['total_sessions']} sessions)")
    print(
        f"  completed: {stats['completed']}  "
        f"incomplete: {stats['incomplete']}  "
        f"white wins: {stats['white_wins']}  "
        f"black wins: {stats['black_wins']}  "
        f"draws: {stats['draws']}"
    )
    if stats["avg_completed_duration_s"] is not None:
        print(
            f"  avg completed: {stats['avg_completed_duration_s']:.0f}s, "
            f"{stats['avg_completed_move_count']:.0f} moves"
        )
    if stats["engine_games"]:
        print(
            f"  engine record (provenance-tagged): "
            f"{stats['engine_wins']}W-{stats['engine_losses']}L-{stats['engine_draws']}D "
            f"over {stats['engine_games']} completed games"
        )
    print()
    print(f"{'session':<8}  {'start':<19}  {'dur':>6}  {'moves':>4}  outcome")
    print(f"{'-'*8}  {'-'*19}  {'-'*6}  {'-'*5}  {'-'*30}")
    for s in summaries:
        print(format_one_line(s))
        if args.verbose:
            print(format_verbose_block(s))
            print()


if __name__ == "__main__":
    main()
