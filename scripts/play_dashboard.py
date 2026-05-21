#!/usr/bin/env python3
"""Interactive visual play-vs-model dashboard (Streamlit + clickable board).

Play YINSH against a trained checkpoint on a clickable hex board. The AI
uses the same pure-neural MCTS as eval_vs_heuristic.py. Click a position to
act:
  - RING_PLACEMENT: click an empty spot to place.
  - MAIN_GAME: click your ring to pick it up (valid destinations light up),
    then click a destination to move. Click the selected ring again to cancel.
  - RING_REMOVAL: click your ring to remove it.
  - ROW_COMPLETION: pick the 5-marker row from the selector (discrete choice).

The board is a Plotly figure; every position carries its coordinate as
customdata, so clicks resolve to an exact hex with no pixel mapping. The
AI's top-3 considered moves (MCTS visit share) are shown each turn.

Run:
    streamlit run scripts/play_dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.game.constants import Position, is_valid_position  # noqa: E402
from yinsh_ml.game.game_state import GameState, GamePhase  # noqa: E402
from yinsh_ml.game.moves import Move, MoveType  # noqa: E402
from yinsh_ml.game.types import Player  # noqa: E402
from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.training.self_play import MCTS  # noqa: E402
from yinsh_ml.viz.board_render import position_to_xy, _all_valid_positions  # noqa: E402

# Match the viz module's palette.
BG = "#fbfaf6"
GRID = "#9aa0a6"
DOT = "#5f6368"
W_RING_FACE, W_RING_EDGE = "#ffffff", "#202124"
B_RING_FACE, B_RING_EDGE = "#202124", "#202124"
W_MARK, B_MARK, MARK_EDGE = "#dadce0", "#3c4043", "#202124"
HILITE = "#4285f4"
SELECTED = "#34a853"
LAST_FROM, LAST_TO = "#fbbc04", "#34a853"

try:
    from yinsh_ml.game.constants import PieceType
except Exception:  # pragma: no cover
    PieceType = None


# ----------------------------- formatting -----------------------------

def fmt_move(m: Move) -> str:
    if m.type == MoveType.PLACE_RING:
        return f"place {m.source}"
    if m.type == MoveType.MOVE_RING:
        return f"{m.source} → {m.destination}"
    if m.type == MoveType.REMOVE_MARKERS:
        return "remove markers " + " ".join(str(p) for p in (m.markers or []))
    if m.type == MoveType.REMOVE_RING:
        return f"remove ring {m.source}"
    return str(m)


# ----------------------------- AI -----------------------------

def build_mcts(network: NetworkWrapper, sims: int) -> MCTS:
    return MCTS(
        network=network, evaluation_mode="pure_neural", heuristic_evaluator=None,
        num_simulations=sims, late_simulations=sims, simulation_switch_ply=10_000,
        enable_subtree_reuse=True, epsilon_mix_start=0.0, epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=0, initial_temp=1.0, final_temp=1.0, annealing_steps=1,
    )


def ai_choose(game: GameState) -> Tuple[Optional[Move], float, List[Tuple[str, float]]]:
    mcts: MCTS = st.session_state.pd_mcts
    network: NetworkWrapper = st.session_state.pd_network
    valid = game.get_valid_moves()
    if not valid:
        return None, 0.0, []
    t0 = time.time()
    visit = mcts.search_batch(game, st.session_state.pd_ai_movecount, batch_size=32)
    elapsed = time.time() - t0
    probs_t = torch.from_numpy(np.asarray(visit)).to(network.device)
    chosen = network.select_move(probs_t, valid, temperature=0.0)
    del probs_t
    scored: List[Tuple[str, float]] = []
    for mv in valid:
        try:
            idx = network.state_encoder.move_to_index(mv)
            if idx < len(visit):
                scored.append((fmt_move(mv), float(visit[idx])))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1], reverse=True)
    return chosen, elapsed, scored[:3]


def advance_mcts_root(played: Move) -> None:
    st.session_state.pd_mcts.advance_root(played)
    st.session_state.pd_ai_movecount += 1


# ----------------------------- game flow -----------------------------

def new_game(checkpoint: str, color: str, sims: int, device: str) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = (checkpoint, device)
    if st.session_state.get("pd_net_cfg") != cfg or "pd_network" not in st.session_state:
        net = NetworkWrapper(device=device)
        net.load_model(checkpoint)
        st.session_state.pd_network = net
        st.session_state.pd_net_cfg = cfg

    st.session_state.pd_mcts = build_mcts(st.session_state.pd_network, sims)
    st.session_state.pd_ai_movecount = 0
    st.session_state.pd_game = GameState()
    st.session_state.pd_human_color = Player.WHITE if color == "white" else Player.BLACK
    st.session_state.pd_ai_color = (
        Player.BLACK if st.session_state.pd_human_color == Player.WHITE else Player.WHITE)
    st.session_state.pd_history = []
    st.session_state.pd_last_move = None
    st.session_state.pd_selected = None
    st.session_state.pd_ai_info = []
    st.session_state.pd_error = None
    st.session_state.pd_board_nonce = 0
    run_ai_turn_if_needed()


def _record(mv: Move, by_ai: bool) -> None:
    st.session_state.pd_history.append(("AI: " if by_ai else "You: ") + fmt_move(mv))
    if mv.type == MoveType.MOVE_RING:
        st.session_state.pd_last_move = (str(mv.source), str(mv.destination))


def run_ai_turn_if_needed() -> None:
    game: GameState = st.session_state.pd_game
    ai_color: Player = st.session_state.pd_ai_color
    infos = []
    while not game.is_terminal() and game.current_player == ai_color:
        mv, elapsed, top3 = ai_choose(game)
        if mv is None or not game.make_move(mv):
            st.session_state.pd_error = "AI produced no legal move."
            break
        advance_mcts_root(mv)
        _record(mv, by_ai=True)
        infos.append({"move": fmt_move(mv), "elapsed": elapsed, "top3": top3})
    st.session_state.pd_ai_info = infos


def submit_human_move(mv: Move) -> None:
    game: GameState = st.session_state.pd_game
    st.session_state.pd_error = None
    if not game.make_move(mv):
        st.session_state.pd_error = f"Illegal move: {fmt_move(mv)}"
        return
    advance_mcts_root(mv)
    _record(mv, by_ai=False)
    st.session_state.pd_selected = None
    run_ai_turn_if_needed()


def handle_click(pos_str: str) -> None:
    """Drive the per-phase state machine from a board click."""
    game: GameState = st.session_state.pd_game
    if game.is_terminal() or game.current_player != st.session_state.pd_human_color:
        return
    valid = game.get_valid_moves()
    phase = game.phase
    player = game.current_player

    if phase == GamePhase.RING_PLACEMENT:
        if any(m.type == MoveType.PLACE_RING and str(m.source) == pos_str for m in valid):
            submit_human_move(Move(type=MoveType.PLACE_RING, player=player,
                                   source=Position.from_string(pos_str)))
        else:
            st.session_state.pd_error = f"Can't place on {pos_str}."

    elif phase == GamePhase.MAIN_GAME:
        sel = st.session_state.pd_selected
        ring_sources = {str(m.source) for m in valid if m.type == MoveType.MOVE_RING}
        if sel is None:
            if pos_str in ring_sources:
                st.session_state.pd_selected = pos_str
            else:
                st.session_state.pd_error = f"{pos_str} isn't one of your movable rings."
        else:
            if pos_str == sel:  # click the selected ring again -> cancel
                st.session_state.pd_selected = None
                return
            dests = {str(m.destination) for m in valid
                     if m.type == MoveType.MOVE_RING and str(m.source) == sel}
            if pos_str in dests:
                submit_human_move(Move(type=MoveType.MOVE_RING, player=player,
                                       source=Position.from_string(sel),
                                       destination=Position.from_string(pos_str)))
            elif pos_str in ring_sources:  # reselect a different ring
                st.session_state.pd_selected = pos_str
            else:
                st.session_state.pd_error = f"{pos_str} isn't a legal destination."

    elif phase == GamePhase.RING_REMOVAL:
        if any(m.type == MoveType.REMOVE_RING and str(m.source) == pos_str for m in valid):
            submit_human_move(Move(type=MoveType.REMOVE_RING, player=player,
                                   source=Position.from_string(pos_str)))
        else:
            st.session_state.pd_error = f"{pos_str} isn't a removable ring."
    # ROW_COMPLETION handled via selector, not clicks.


# ----------------------------- board figure -----------------------------

def _piece_kind(board, pos: Position) -> Optional[str]:
    piece = board.get_piece(pos)
    if piece is None:
        return None
    is_ring = piece.is_ring() if hasattr(piece, "is_ring") else (
        PieceType is not None and piece in (PieceType.WHITE_RING, PieceType.BLACK_RING))
    if is_ring:
        return "white_ring" if "WHITE" in str(piece) else "black_ring"
    return "white_marker" if "WHITE" in str(piece) else "black_marker"


def build_board_figure(game: GameState, *, selected: Optional[str],
                       highlight: set, last_move: Optional[Tuple[str, str]]) -> go.Figure:
    board = game.board
    fig = go.Figure()

    # Grid lines (3 forward hex axes), one trace with None separators.
    lx, ly = [], []
    for pos in _all_valid_positions():
        x0, y0 = position_to_xy(pos.column, pos.row)
        for dcol, drow in ((0, 1), (1, 0), (1, 1)):
            nb = Position(column=chr(ord(pos.column) + dcol), row=pos.row + drow)
            if is_valid_position(nb):
                x1, y1 = position_to_xy(nb.column, nb.row)
                lx += [x0, x1, None]
                ly += [y0, y1, None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                             line=dict(color=GRID, width=1), hoverinfo="skip",
                             showlegend=False))

    # Per-position marker styling. customdata = position string on EVERY point
    # so a click anywhere on a position resolves to that hex.
    xs, ys, cd, syms, sizes, colors, line_colors, line_widths, holes = (
        [], [], [], [], [], [], [], [], [])
    for pos in _all_valid_positions():
        s = str(pos)
        x, y = position_to_xy(pos.column, pos.row)
        kind = _piece_kind(board, pos)
        xs.append(x); ys.append(y); cd.append(s)
        if kind == "white_ring":
            syms.append("circle"); sizes.append(30); colors.append(W_RING_FACE)
            line_colors.append(W_RING_EDGE); line_widths.append(2); holes.append((x, y))
        elif kind == "black_ring":
            syms.append("circle"); sizes.append(30); colors.append(B_RING_FACE)
            line_colors.append(B_RING_EDGE); line_widths.append(2); holes.append((x, y))
        elif kind == "white_marker":
            syms.append("circle"); sizes.append(18); colors.append(W_MARK)
            line_colors.append(MARK_EDGE); line_widths.append(1.5)
        elif kind == "black_marker":
            syms.append("circle"); sizes.append(18); colors.append(B_MARK)
            line_colors.append(MARK_EDGE); line_widths.append(1.5)
        else:  # empty
            syms.append("circle"); sizes.append(9); colors.append(DOT)
            line_colors.append(DOT); line_widths.append(0)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", customdata=cd,
        marker=dict(symbol=syms, size=sizes, color=colors,
                    line=dict(color=line_colors, width=line_widths)),
        hovertemplate="%{customdata}<extra></extra>", showlegend=False, name="cells"))

    # Ring "holes" — cream centers so rings read as hollow. customdata kept so
    # clicks on a ring center still resolve to the position.
    if holes:
        hx, hy = zip(*holes)
        hcd = [s for s, k in zip(cd, [_piece_kind(board, Position.from_string(s)) for s in cd])
               if k in ("white_ring", "black_ring")]
        fig.add_trace(go.Scatter(
            x=list(hx), y=list(hy), mode="markers", customdata=hcd,
            marker=dict(symbol="circle", size=16, color=BG),
            hoverinfo="skip", showlegend=False, name="holes"))

    # Highlight valid targets / selection.
    if highlight:
        hxs, hys, hcd = [], [], []
        for s in highlight:
            p = Position.from_string(s)
            x, y = position_to_xy(p.column, p.row)
            hxs.append(x); hys.append(y); hcd.append(s)
        fig.add_trace(go.Scatter(
            x=hxs, y=hys, mode="markers", customdata=hcd,
            marker=dict(symbol="circle-open", size=40,
                        line=dict(color=HILITE, width=3)),
            hoverinfo="skip", showlegend=False, name="hl"))
    if selected:
        p = Position.from_string(selected)
        x, y = position_to_xy(p.column, p.row)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", customdata=[selected],
            marker=dict(symbol="circle-open", size=44,
                        line=dict(color=SELECTED, width=4)),
            hoverinfo="skip", showlegend=False, name="sel"))
    if last_move:
        for s, col in ((last_move[0], LAST_FROM), (last_move[1], LAST_TO)):
            p = Position.from_string(s)
            x, y = position_to_xy(p.column, p.row)
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers", customdata=[s],
                marker=dict(symbol="circle-open", size=48, line=dict(color=col, width=2)),
                hoverinfo="skip", showlegend=False, name="lm"))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), height=680, dragmode=False,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False), clickmode="event+select",
    )
    return fig


# ----------------------------- main -----------------------------

def main() -> None:
    st.set_page_config(page_title="YINSH — play vs model", layout="wide")
    st.title("YINSH — play vs model")

    with st.sidebar:
        st.header("Setup")
        default_ckpt = "models/branchC_volume_pretrain/best_iter_4.pt"
        if not Path(default_ckpt).exists():
            default_ckpt = "models/yngine_volume_pretrain/best_supervised.pt"
        checkpoint = st.text_input("Checkpoint .pt", value=default_ckpt)
        color = st.radio("Your color", ["white", "black"], horizontal=True)
        sims = st.slider("AI MCTS sims/move", 32, 400, 100, step=32)
        device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"])
        if st.button("New game", type="primary"):
            if not Path(checkpoint).exists():
                st.error(f"Checkpoint not found: {checkpoint}")
            else:
                with st.spinner("Loading model + starting game..."):
                    new_game(checkpoint, color, sims, device)
                st.rerun()
        st.caption("Click the board to move. AI uses the same pure-neural MCTS "
                   "as the eval harness; its top-3 considered moves are shown each turn.")

    if "pd_game" not in st.session_state:
        st.info("Set a checkpoint and click **New game** in the sidebar to begin.")
        return

    game: GameState = st.session_state.pd_game
    human_color: Player = st.session_state.pd_human_color
    ai_color: Player = st.session_state.pd_ai_color
    selected = st.session_state.pd_selected

    # Compute highlight set for the current human turn.
    highlight: set = set()
    if not game.is_terminal() and game.current_player == human_color:
        valid = game.get_valid_moves()
        if game.phase == GamePhase.RING_PLACEMENT:
            highlight = {str(m.source) for m in valid if m.type == MoveType.PLACE_RING}
        elif game.phase == GamePhase.MAIN_GAME:
            if selected is None:
                highlight = {str(m.source) for m in valid if m.type == MoveType.MOVE_RING}
            else:
                highlight = {str(m.destination) for m in valid
                             if m.type == MoveType.MOVE_RING and str(m.source) == selected}
        elif game.phase == GamePhase.RING_REMOVAL:
            highlight = {str(m.source) for m in valid if m.type == MoveType.REMOVE_RING}

    board_col, side_col = st.columns([3, 2])

    with side_col:
        st.subheader("Status")
        st.write(f"**Phase:** {game.phase.name}")
        st.write(f"**Score** — White: {game.white_score} / Black: {game.black_score}")
        st.write(f"**You:** {human_color.name}  •  **AI:** {ai_color.name}")
        st.write(f"**To move:** {game.current_player.name}")
        if st.session_state.pd_error:
            st.error(st.session_state.pd_error)

        if game.is_terminal():
            w = game.get_winner()
            if w is None:
                st.success("Game over — draw.")
            elif w == human_color:
                st.success("Game over — you win! 🎉")
            else:
                st.warning("Game over — AI wins.")
        elif game.current_player == human_color:
            if game.phase == GamePhase.MAIN_GAME and selected:
                st.info(f"Ring **{selected}** selected — click a highlighted "
                        f"destination, or click {selected} again to cancel.")
            elif game.phase == GamePhase.ROW_COMPLETION:
                # 5-marker row is a discrete pick → selector, not clicks.
                opts = [m for m in game.get_valid_moves()
                        if m.type == MoveType.REMOVE_MARKERS]
                labels = [" ".join(str(p) for p in (m.markers or [])) for m in opts]
                idx = st.selectbox("Pick a row of 5 markers to remove:",
                                   range(len(labels)), format_func=lambda i: labels[i])
                if st.button("Remove markers", type="primary") and opts:
                    submit_human_move(opts[idx])
                    st.session_state.pd_board_nonce += 1
                    st.rerun()
            else:
                st.info("Your move — click a highlighted position.")

        if st.session_state.pd_ai_info:
            st.markdown("### AI's last turn")
            for info in st.session_state.pd_ai_info:
                st.write(f"played **{info['move']}**  _( {info['elapsed']:.1f}s )_")
                if info["top3"]:
                    st.caption("considered: " + " · ".join(
                        f"{lbl} {share:.0%}" for lbl, share in info["top3"]))

        with st.expander("Move history", expanded=False):
            for line in st.session_state.pd_history:
                st.text(line)

    with board_col:
        fig = build_board_figure(game, selected=selected, highlight=highlight,
                                 last_move=st.session_state.pd_last_move)
        key = f"pd_board_{st.session_state.pd_board_nonce}"
        event = st.plotly_chart(fig, use_container_width=True, key=key,
                                on_select="rerun", selection_mode="points")

        clicked = None
        try:
            pts = (event or {}).get("selection", {}).get("points", [])
            for p in pts:
                c = p.get("customdata")
                if c:
                    clicked = c[0] if isinstance(c, (list, tuple)) else c
                    break
        except Exception:
            clicked = None

        if clicked and not game.is_terminal() and game.current_player == human_color:
            handle_click(str(clicked))
            st.session_state.pd_board_nonce += 1  # reset chart selection
            st.rerun()


if __name__ == "__main__":
    main()
