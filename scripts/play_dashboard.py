#!/usr/bin/env python3
"""Interactive visual play-vs-model dashboard (Streamlit).

Play a full game of YINSH against a trained checkpoint on the rendered hex
board, with the AI driven by the same pure-neural MCTS path as
eval_vs_heuristic.py / play_vs_model_mcts.py. The board uses the viz
renderer (yinsh_ml.viz.render_board); valid targets for the current phase
are highlighted, and move input is via phase-aware selectboxes populated
from GameState.get_valid_moves() (so only legal moves are offered).

The AI's top-3 considered moves (by MCTS visit share) are shown each turn,
so you can eyeball *why* it plays what it plays — the qualitative
gut-check the win-rate metrics can't give you.

Run:
    streamlit run scripts/play_dashboard.py
    # then pick a checkpoint in the sidebar (default: the Branch C best iter)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

# Make `yinsh_ml` importable when launched via `streamlit run` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.game.constants import PieceType, Position  # noqa: E402
from yinsh_ml.game.game_state import GameState, GamePhase  # noqa: E402
from yinsh_ml.game.moves import Move, MoveType  # noqa: E402
from yinsh_ml.game.types import Player  # noqa: E402
from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.training.self_play import MCTS  # noqa: E402
from yinsh_ml.viz import render_board  # noqa: E402


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
        network=network,
        evaluation_mode="pure_neural",
        heuristic_evaluator=None,
        num_simulations=sims,
        late_simulations=sims,
        simulation_switch_ply=10_000,
        enable_subtree_reuse=True,
        epsilon_mix_start=0.0,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=0,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
    )


def ai_choose(game: GameState) -> Tuple[Optional[Move], float, List[Tuple[str, float]]]:
    """Run MCTS for the side to move; return (move, elapsed_s, top3 [(label, share)])."""
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
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    # Only reload the network if checkpoint/device changed (it's the slow part).
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
        Player.BLACK if st.session_state.pd_human_color == Player.WHITE else Player.WHITE
    )
    st.session_state.pd_history = []
    st.session_state.pd_last_move = None
    st.session_state.pd_sel_ring = None
    st.session_state.pd_ai_info = []
    st.session_state.pd_error = None
    st.session_state.pd_sims = sims
    # If the AI is White it moves first.
    run_ai_turn_if_needed()


def _record_move(mv: Move, by_ai: bool) -> None:
    prefix = "AI: " if by_ai else "You: "
    st.session_state.pd_history.append(prefix + fmt_move(mv))
    if mv.type == MoveType.MOVE_RING:
        st.session_state.pd_last_move = (mv.source, mv.destination)


def run_ai_turn_if_needed() -> None:
    """Play the AI's full turn (it may span sub-phases) until it's the human's
    turn again or the game ends."""
    game: GameState = st.session_state.pd_game
    ai_color: Player = st.session_state.pd_ai_color
    infos = []
    while not game.is_terminal() and game.current_player == ai_color:
        mv, elapsed, top3 = ai_choose(game)
        if mv is None:
            break
        if not game.make_move(mv):
            st.session_state.pd_error = f"AI produced an illegal move: {fmt_move(mv)}"
            break
        advance_mcts_root(mv)
        _record_move(mv, by_ai=True)
        infos.append({"move": fmt_move(mv), "elapsed": elapsed, "top3": top3})
    st.session_state.pd_ai_info = infos


def submit_human_move(mv: Move) -> None:
    game: GameState = st.session_state.pd_game
    st.session_state.pd_error = None
    if not game.make_move(mv):
        st.session_state.pd_error = f"Illegal move: {fmt_move(mv)}"
        return
    advance_mcts_root(mv)
    _record_move(mv, by_ai=False)
    st.session_state.pd_sel_ring = None
    # If it's now the AI's turn (human's sub-phases are done), let the AI play.
    run_ai_turn_if_needed()


# ----------------------------- UI: input panels -----------------------------

def human_input_panel(game: GameState) -> Optional[List[Position]]:
    """Render phase-appropriate input widgets. Returns positions to highlight
    on the board (valid targets / selection), and submits moves via buttons."""
    valid = game.get_valid_moves()
    phase = game.phase
    highlight: List[Position] = []

    if phase == GamePhase.RING_PLACEMENT:
        spots = sorted({str(m.source) for m in valid if m.type == MoveType.PLACE_RING})
        highlight = [Position.from_string(s) for s in spots]
        choice = st.selectbox("Place a ring at:", spots, key="pd_place_sel")
        if st.button("Place ring", type="primary"):
            submit_human_move(Move(type=MoveType.PLACE_RING,
                                   player=game.current_player,
                                   source=Position.from_string(choice)))
            st.rerun()

    elif phase == GamePhase.MAIN_GAME:
        move_rings = sorted({str(m.source) for m in valid if m.type == MoveType.MOVE_RING})
        sel = st.session_state.pd_sel_ring
        if sel is None:
            highlight = [Position.from_string(s) for s in move_rings]
            pick = st.selectbox("Select one of your rings:", move_rings, key="pd_ring_sel")
            if st.button("Select ring", type="primary"):
                st.session_state.pd_sel_ring = pick
                st.rerun()
        else:
            dests = sorted({str(m.destination) for m in valid
                            if m.type == MoveType.MOVE_RING and str(m.source) == sel})
            highlight = [Position.from_string(sel)] + [Position.from_string(d) for d in dests]
            st.write(f"Ring **{sel}** selected.")
            dest = st.selectbox("Move it to:", dests, key="pd_dest_sel")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Move here", type="primary"):
                    submit_human_move(Move(type=MoveType.MOVE_RING,
                                           player=game.current_player,
                                           source=Position.from_string(sel),
                                           destination=Position.from_string(dest)))
                    st.rerun()
            with c2:
                if st.button("Cancel selection"):
                    st.session_state.pd_sel_ring = None
                    st.rerun()

    elif phase == GamePhase.ROW_COMPLETION:
        opts = [m for m in valid if m.type == MoveType.REMOVE_MARKERS]
        labels = [" ".join(str(p) for p in (m.markers or [])) for m in opts]
        idx = st.selectbox("Pick a row of 5 markers to remove:",
                           range(len(labels)), format_func=lambda i: labels[i],
                           key="pd_markers_sel")
        if opts:
            highlight = list(opts[idx].markers or [])
        if st.button("Remove markers", type="primary") and opts:
            submit_human_move(opts[idx])
            st.rerun()

    elif phase == GamePhase.RING_REMOVAL:
        rings = sorted({str(m.source) for m in valid if m.type == MoveType.REMOVE_RING})
        highlight = [Position.from_string(s) for s in rings]
        choice = st.selectbox("Remove one of your rings:", rings, key="pd_remring_sel")
        if st.button("Remove ring", type="primary"):
            submit_human_move(Move(type=MoveType.REMOVE_RING,
                                   player=game.current_player,
                                   source=Position.from_string(choice)))
            st.rerun()

    return highlight


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
        st.caption("AI uses the same pure-neural MCTS as the eval harness. "
                   "Top-3 considered moves shown each turn.")

    if "pd_game" not in st.session_state:
        st.info("Set a checkpoint and click **New game** in the sidebar to begin.")
        return

    game: GameState = st.session_state.pd_game
    human_color: Player = st.session_state.pd_human_color
    ai_color: Player = st.session_state.pd_ai_color

    board_col, side_col = st.columns([3, 2])

    with side_col:
        st.subheader("Status")
        st.write(f"**Phase:** {game.phase.name}")
        st.write(f"**Score** — White: {game.white_score} / Black: {game.black_score}")
        st.write(f"**You:** {human_color.name}  •  **AI:** {ai_color.name}")
        st.write(f"**To move:** {game.current_player.name}")

        if st.session_state.pd_error:
            st.error(st.session_state.pd_error)

        terminal = game.is_terminal()
        highlight: List[Position] = []
        if terminal:
            winner = game.get_winner()
            if winner is None:
                st.success("Game over — draw.")
            elif winner == human_color:
                st.success("Game over — you win! 🎉")
            else:
                st.warning("Game over — AI wins.")
        elif game.current_player == human_color:
            st.markdown("### Your move")
            highlight = human_input_panel(game) or []
        else:
            st.markdown("### AI is thinking...")
            # Shouldn't normally land here (AI turn runs inline), but guard anyway.
            run_ai_turn_if_needed()
            st.rerun()

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
        last_move = st.session_state.pd_last_move
        fig = render_board(
            game.board,
            last_move=last_move,
            highlight=highlight if not game.is_terminal() else None,
            title=None,
        )
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
