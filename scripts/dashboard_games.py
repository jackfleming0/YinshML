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
import streamlit as st

# Make `yinsh_ml` importable when running via `streamlit run` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.game.constants import Player  # noqa: E402
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


def _replay_to_turn(replay: GameReplay, turn_index: int) -> Optional[GameState]:
    """Re-run the game up to ``turn_index`` to get a full GameState.

    Needed because GameReplay stores Board snapshots but the heuristic
    feature functions take a GameState (they need phase + scores).
    """
    state = GameState()
    for i in range(min(turn_index + 1, len(replay.moves))):
        try:
            state.make_move(replay.moves[i])
        except Exception:
            return None
    return state


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

    # ----- Main columns: board + metrics --------------------------------
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

        if show_audit:
            state = _replay_to_turn(replay, turn_idx)
            if state is not None:
                st.subheader("Heuristic features (computed, White POV)")
                computed = extract_all_features(state, Player.WHITE)
                audit = {
                    k: f"{computed.get(k, 0):+.2f}"
                    for k in _AUDIT_FEATURES if k in computed
                }
                st.table({"diff": audit})


if __name__ == "__main__":
    main()
