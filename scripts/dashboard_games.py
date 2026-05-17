"""Live game viewer for YINSH self-play parquet output.

Streamlit dashboard that reads parquet files written by
``yinsh_ml.self_play.data_storage``, replays games turn-by-turn, and
surfaces per-turn metrics. Designed to be left running while a
self-play harness writes games — refresh the page to pick up new games.

Run with:

    streamlit run scripts/dashboard_games.py

Then point the sidebar at your parquet directory (default:
``self_play_data/parquet_data`` relative to repo root).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _AUTOREFRESH_AVAILABLE = True
except ImportError:
    _AUTOREFRESH_AVAILABLE = False
    _st_autorefresh = None  # type: ignore

# Make `yinsh_ml` importable when running via `streamlit run` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.game.constants import PieceType, Player  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.game.types import MoveType  # noqa: E402
from yinsh_ml.heuristics.features import extract_all_features  # noqa: E402
from yinsh_ml.viz import render_board  # noqa: E402
from yinsh_ml.viz.game_replay import GameReplay, list_games, load_game  # noqa: E402

# Heuristic features that we particularly care about for the offense-only
# audit. Surfaced separately from the parquet's analysis features.
_AUDIT_FEATURES = [
    "completed_runs_differential",
    "potential_runs_count",
    "connected_marker_chains",
    "ring_positioning",
    "ring_spread",
]


@st.cache_data(show_spinner=False)
def _list_games_cached(parquet_dir: str, mtime_key: float):
    """Cached game list. ``mtime_key`` invalidates when files change."""
    return list_games(Path(parquet_dir))


def _dir_mtime(parquet_dir: Path) -> float:
    if not parquet_dir.exists():
        return 0.0
    return max(
        (fp.stat().st_mtime for fp in parquet_dir.glob("*.parquet")),
        default=0.0,
    )


@st.cache_data(show_spinner=False)
def _load_game_cached(parquet_dir: str, game_id: str, mtime_key: float) -> GameReplay:
    return load_game(Path(parquet_dir), game_id)


def _count_threats(state: GameState, marker_type: PieceType) -> int:
    """Number of 4-length contiguous marker rows for ``marker_type``.

    A 4-length row is one marker short of capture — the defining
    "immediate threat" condition for case (a) of YINSH capture
    formation: gradual marker buildup over multiple turns. On their
    next turn the threat-holder can complete the row, OR the defender
    can break it by moving a ring whose path crosses one of the four
    markers (flipping it to the defender's color).

    Caveat — case (b): a 5-row can also form via a single ring move
    whose path-flips convert scattered markers or a 3-row directly
    into a 5-row of the mover's color. No visible 4-row state
    precedes such a capture, so this metric is blind to case (b).
    Use score-delta capture-event tracking + ``_count_potential``
    for the full picture.
    """
    rows = state.board.find_marker_rows(marker_type)
    return sum(1 for r in rows if r.length == 4)


def _count_potential(state: GameState, marker_type: PieceType) -> int:
    """Number of 3+ length contiguous marker rows — broader early warning.

    Captures earlier-stage row-building activity. Persists across more
    turns than the strict ``_count_threats == 4`` definition, so
    sustained one-sided growth is a cleaner offense-only signal.
    """
    rows = state.board.find_marker_rows(marker_type)
    return sum(1 for r in rows if r.length >= 3)


@st.cache_data(show_spinner=True)
def _compute_trajectory(
    parquet_dir: str, game_id: str, mtime_key: float
) -> pd.DataFrame:
    """Per-turn audit features, threat counts, captures, defensive misses.

    Single forward pass (``replay.iter_states``) so cost is linear in
    game length. Cached by (game_id, parquet-dir mtime) — re-runs only
    when underlying data changes.

    ``capture`` column is "WHITE"/"BLACK" on turns where that player's
    running score went up by ≥1 (captures from the score delta, which
    is the unambiguous signal — row-length matching is unreliable
    around RING_REMOVAL because the captured markers are immediately
    cleared from the board).

    ``defensive_miss`` column is True on turns where the opponent had
    a 4-row at the start of the player's turn AND the same opponent
    still has at least the same number of 4-rows after the player's
    move. Necessary-but-not-sufficient detector — catches "player
    ignored the threat entirely" but not subtler issues like "player
    flipped a marker that the opponent can trivially flip back."
    See TODO_baseline.md viz section for the richer multi-dim metric.
    """
    replay = _load_game_cached(parquet_dir, game_id, mtime_key)

    def threats_for(state: GameState, player: Player) -> int:
        marker = (PieceType.WHITE_MARKER if player == Player.WHITE
                  else PieceType.BLACK_MARKER)
        return _count_threats(state, marker)

    rows: List[dict] = []
    prev_white_score = prev_black_score = 0
    # Both players' 4-row counts at the prior state — needed to detect
    # whether the player whose turn it is now actually responded to a
    # standing threat. Initialized to 0 because the empty starting
    # board has no markers.
    prev_threats = {Player.WHITE: 0, Player.BLACK: 0}

    for turn_idx, state in replay.iter_states():
        feats = extract_all_features(state, Player.WHITE)
        capture = ""
        if state.white_score > prev_white_score:
            capture = "WHITE"
        elif state.black_score > prev_black_score:
            capture = "BLACK"
        prev_white_score, prev_black_score = state.white_score, state.black_score

        # state.current_player is whoever is about to move next, i.e.
        # the OPPONENT of whoever just moved. Their threats are what
        # the just-moved player should have been defending against.
        opponent_of_mover = state.current_player
        threats_before = prev_threats[opponent_of_mover]
        threats_after = threats_for(state, opponent_of_mover)
        defensive_miss = bool(threats_before > 0 and threats_after >= threats_before)

        # Refresh both colors' threats for the next iteration.
        white_after = threats_for(state, Player.WHITE)
        black_after = threats_for(state, Player.BLACK)
        prev_threats = {Player.WHITE: white_after, Player.BLACK: black_after}

        rows.append({
            "turn": turn_idx + 1,
            "player": state.current_player.name,
            "white_score": state.white_score,
            "black_score": state.black_score,
            "capture": capture,
            "defensive_miss": defensive_miss,
            **{k: float(feats.get(k, 0.0)) for k in _AUDIT_FEATURES},
            "white_threats": white_after,
            "black_threats": black_after,
            "white_potential": _count_potential(state, PieceType.WHITE_MARKER),
            "black_potential": _count_potential(state, PieceType.BLACK_MARKER),
        })
    return pd.DataFrame(rows)


def _format_move(move) -> str:
    parts = [move.player.name, move.type.value]
    if move.type == MoveType.PLACE_RING:
        parts.append(f"@ {move.source}")
    elif move.type == MoveType.MOVE_RING:
        parts.append(f"{move.source} → {move.destination}")
    elif move.type == MoveType.REMOVE_MARKERS:
        markers = ",".join(str(p) for p in (move.markers or ()))
        parts.append(f"[{markers}]")
    elif move.type == MoveType.REMOVE_RING:
        parts.append(f"@ {move.source}")
    return " ".join(parts)


def main() -> None:
    st.set_page_config(page_title="YINSH game viewer", layout="wide")
    st.title("YINSH game viewer")

    # ----- Sidebar: directory + game selection ---------------------------
    with st.sidebar:
        st.header("Source")
        default_dir = str(ROOT / "self_play_data" / "parquet_data")
        parquet_dir_str = st.text_input("Parquet directory", value=default_dir)
        parquet_dir = Path(parquet_dir_str)

        if not parquet_dir.exists():
            st.error(f"Directory not found: {parquet_dir}")
            st.stop()

        mtime = _dir_mtime(parquet_dir)
        summary = _list_games_cached(str(parquet_dir), mtime)

        if summary.empty:
            st.warning(f"No parquet files in {parquet_dir}")
            st.stop()

        st.caption(f"{len(summary)} games · last write {mtime:.0f}")
        st.dataframe(
            summary[["game_id", "winner", "total_turns",
                     "white_score", "black_score"]],
            height=240, hide_index=True,
        )

        # Game picker. Default to the most recent.
        game_ids: List[str] = summary["game_id"].tolist()
        default_idx = len(game_ids) - 1
        game_id = st.selectbox("Game", game_ids, index=default_idx)

        st.divider()
        show_coords = st.checkbox("Show coordinate labels", value=False)
        show_audit = st.checkbox("Compute heuristic features on-the-fly", value=True)

        st.divider()
        st.caption("Live mode")
        if _AUTOREFRESH_AVAILABLE:
            live = st.checkbox("Auto-refresh", value=False,
                               help="Re-read parquet directory on a timer "
                                    "so new games appear as the harness "
                                    "writes them. Pair with "
                                    "`generate_heuristic_games.py "
                                    "--batch-size 1`.")
            interval = st.slider("Interval (seconds)", min_value=2,
                                 max_value=30, value=5, disabled=not live)
            if live:
                _st_autorefresh(interval=interval * 1000, key="game_viewer_refresh")
        else:
            st.caption("Install `streamlit-autorefresh` for live mode.")

        if st.button("Reload"):
            st.cache_data.clear()

    # ----- Load the selected game ---------------------------------------
    try:
        replay = _load_game_cached(str(parquet_dir), game_id, mtime)
    except KeyError:
        st.error(f"Game {game_id!r} no longer in parquet directory.")
        st.stop()

    n = len(replay.moves)
    if n == 0:
        st.warning("Game has no moves recorded.")
        st.stop()

    # ----- Turn navigation ----------------------------------------------
    state_key = f"turn_idx:{game_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = n - 1
    # Clamp in case the game length changed.
    st.session_state[state_key] = min(st.session_state[state_key], n - 1)

    nav_l, nav_m, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
    with nav_r:
        if st.button("Next ▶", use_container_width=True):
            st.session_state[state_key] = min(n - 1, st.session_state[state_key] + 1)
    with nav_m:
        turn_idx = st.slider(
            "Turn", min_value=0, max_value=n - 1,
            value=st.session_state[state_key], key=f"slider:{game_id}",
        )
        st.session_state[state_key] = turn_idx

    # ----- Compute trajectory once for both tabs ------------------------
    trajectory: Optional[pd.DataFrame] = None
    if show_audit:
        try:
            trajectory = _compute_trajectory(str(parquet_dir), game_id, mtime)
        except Exception as e:
            st.warning(f"Could not compute trajectory: {e}")

    # ----- Tabs: Board view + Trajectory view ---------------------------
    tab_board, tab_traj = st.tabs(["Board", "Trajectory"])

    with tab_board:
        board_col, info_col = st.columns([3, 2])

        with board_col:
            move = replay.moves[turn_idx]
            last_move = None
            if move.source is not None and move.destination is not None:
                last_move = (move.source, move.destination)
            fig = render_board(
                replay.board_after(turn_idx),
                last_move=last_move,
                title=f"{replay.game_id} — turn {turn_idx + 1}/{n}",
                show_coords=show_coords,
                figsize=(7.5, 7.5),
            )
            st.pyplot(fig)

            # Badges: captures (unambiguous), defensive misses, threats.
            if trajectory is not None and turn_idx < len(trajectory):
                row = trajectory.iloc[turn_idx]
                badges: List[str] = []
                cap = row.get("capture", "") or ""
                if cap:
                    icon = "⚪" if cap == "WHITE" else "⚫"
                    badges.append(f":green[{icon} CAPTURE — {cap}]")
                if bool(row.get("defensive_miss", False)):
                    badges.append(":red[🚨 DEFENSIVE MISS]")
                wt = int(row["white_threats"])
                bt = int(row["black_threats"])
                if wt:
                    badges.append(f":orange[⚠ White has {wt} 4-row(s)]")
                if bt:
                    badges.append(f":orange[⚠ Black has {bt} 4-row(s)]")
                if badges:
                    st.markdown(" · ".join(badges))

        with info_col:
            st.subheader("Move")
            st.text(_format_move(move))

            st.subheader("Game state")
            meta = {
                "Winner": replay.winner or "—",
                "White score": replay.white_score,
                "Black score": replay.black_score,
                "Final phase": replay.final_phase or "—",
                "Total turns": replay.total_turns,
            }
            if replay.replay_truncated_at is not None:
                meta["⚠ Replay truncated at"] = replay.replay_truncated_at
            st.table({"": meta})

            if replay.features and replay.features[turn_idx]:
                st.subheader("Recorded features (parquet)")
                feats = {
                    k: f"{v:.3f}" if isinstance(v, float) else v
                    for k, v in replay.features[turn_idx].items()
                }
                st.table({"value": feats})

            if show_audit and trajectory is not None and turn_idx < len(trajectory):
                st.subheader("Heuristic features (this turn, White POV)")
                tr_row = trajectory.iloc[turn_idx]
                audit = {
                    k: f"{tr_row[k]:+.2f}"
                    for k in _AUDIT_FEATURES if k in tr_row
                }
                st.table({"diff": audit})

    with tab_traj:
        if trajectory is None or trajectory.empty:
            st.info("Enable 'Compute heuristic features on-the-fly' in the "
                    "sidebar to see trajectories.")
        else:
            st.subheader("Score progression")
            captures = trajectory[trajectory["capture"] != ""]
            cap_txt = " · ".join(
                f"turn {int(r.turn)}: {r.capture}" for _, r in captures.iterrows()
            ) if not captures.empty else "—"
            misses = trajectory[trajectory["defensive_miss"] == True]  # noqa: E712
            miss_txt = " · ".join(
                f"turn {int(r.turn)}" for _, r in misses.iterrows()
            ) if not misses.empty else "—"
            st.caption(f"Captures: {cap_txt}")
            st.caption(
                f"Defensive misses ({len(misses)}): {miss_txt} "
                "— necessary-but-not-sufficient detector (opponent had a "
                "4-row at the start of player's turn AND it survived the "
                "player's move). See the README for tactical caveats."
            )
            st.line_chart(
                trajectory.set_index("turn")[["white_score", "black_score"]]
            )

            st.subheader("Heuristic features over the game (White POV)")
            st.caption(
                "Positive = White favoured. Watch for one side's "
                "`potential_runs_count` or `completed_runs_differential` "
                "climbing without the other side responding — the canonical "
                "offense-only collapse signature."
            )
            st.line_chart(trajectory.set_index("turn")[_AUDIT_FEATURES])

            st.subheader("Row-building over time")
            st.caption(
                "**Threats** = rows of exactly 4 markers (one move from "
                "capture; defender can break by flipping a marker in the "
                "row). **Potential** = rows of ≥3 markers — broader "
                "early-warning of one side building unchecked. Neither "
                "metric catches single-move path-flip captures; see the "
                "capture-event list above for those."
            )
            st.line_chart(
                trajectory.set_index("turn")[[
                    "white_threats", "black_threats",
                    "white_potential", "black_potential",
                ]]
            )

            with st.expander("Per-turn data"):
                st.dataframe(
                    trajectory[[
                        "turn", "player",
                        "white_score", "black_score", "capture",
                        "defensive_miss",
                        "white_threats", "black_threats",
                        "white_potential", "black_potential",
                        *_AUDIT_FEATURES,
                    ]],
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
