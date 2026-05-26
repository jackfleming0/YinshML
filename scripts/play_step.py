#!/usr/bin/env python3
"""Stateful one-move-at-a-time wrapper around play_vs_model_mcts.py's logic.

Designed for an external orchestrator (e.g. a Claude subagent) to drive
play turn-by-turn via Bash invocations. Each call loads the model + game
state from disk, applies one move, runs the AI's response if it's the
AI's turn, persists state, prints a JSON status block.

Subcommands:
    new      Start a new game. Prints the starting state.
    move     Apply one human move. Prints the resulting state, including
             any AI move that follows.
    status   Reprint the current state without changing it.
    end      Delete the session.

Game state is stored in /tmp/yinsh_session_<session>/state.pkl. Reloading
the model on every move costs ~1-2s but keeps the script stateless.

Usage:
    # Start a game
    python scripts/play_step.py new \\
        --session game1 --color white --mcts-sims 100 \\
        --checkpoint models/supervised_seed_humans_only/best_supervised.pt

    # Make a move (returns AI's reply if it's AI's turn)
    python scripts/play_step.py move --session game1 --command "place F5"

    # Anytime — reprint state
    python scripts/play_step.py status --session game1
"""

import argparse
import json
import pickle
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.types import Player
from yinsh_ml.game.constants import Position, PieceType, is_valid_position
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.training.self_play import MCTS


# ----------------------------- session storage -----------------------------

@dataclass
class Session:
    session_id: str
    checkpoint: str
    device: str
    mcts_sims: int
    human_color: str  # 'white' or 'black'
    game_state: GameState
    move_count: int = 0
    move_log: List[str] = field(default_factory=list)


def session_dir(session_id: str) -> Path:
    return Path(tempfile.gettempdir()) / f"yinsh_session_{session_id}"


def state_file(session_id: str) -> Path:
    return session_dir(session_id) / "state.pkl"


def save_session(s: Session) -> None:
    d = session_dir(s.session_id)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "state.pkl", "wb") as f:
        pickle.dump(s, f)


def load_session(session_id: str) -> Session:
    p = state_file(session_id)
    if not p.exists():
        sys.exit(f"No session at {p}. Did you run `new` first?")
    with open(p, "rb") as f:
        return pickle.load(f)


# ----------------------------- helpers -----------------------------

def parse_position(s: str) -> Optional[Position]:
    s = s.strip()
    if len(s) < 2:
        return None
    try:
        col = s[0].upper()
        row = int(s[1:])
        pos = Position(col, row)
        return pos if is_valid_position(pos) else None
    except (ValueError, IndexError):
        return None


def fmt_move(m: Move) -> str:
    if m.type == MoveType.PLACE_RING:
        return f"PLACE_RING {m.source}"
    if m.type == MoveType.MOVE_RING:
        return f"MOVE_RING {m.source}->{m.destination}"
    if m.type == MoveType.REMOVE_MARKERS:
        return f"REMOVE_MARKERS " + " ".join(str(p) for p in (m.markers or []))
    if m.type == MoveType.REMOVE_RING:
        return f"REMOVE_RING {m.source}"
    return str(m)


def parse_command(text: str, current_player: Player, phase: GamePhase,
                  selected_ring: Optional[Position] = None) -> Optional[Move]:
    """Convert a 'place F5' / 'select F6 move H8' / 'markers F6 F7 F8 F9 F10'
    / 'remove F6' command string into a Move. Returns None if invalid; caller
    handles the error reporting.

    Designed to accept an explicit chained format `select F6 then move H8`
    so subagents can submit the full ring-move in one shot, avoiding a
    two-step round-trip per ring-move turn.
    """
    text = text.strip().lower()
    parts = text.split()
    if not parts:
        return None
    cmd = parts[0]

    if cmd == "place" and len(parts) == 2 and phase == GamePhase.RING_PLACEMENT:
        pos = parse_position(parts[1])
        if pos is None:
            return None
        return Move(type=MoveType.PLACE_RING, player=current_player, source=pos)

    # 'move D2 to E2' or 'move D2 E2' for combined select+move
    if cmd == "move" and phase == GamePhase.MAIN_GAME:
        # Variants:
        #  move D2 E2
        #  move D2 to E2
        toks = [p for p in parts[1:] if p != "to"]
        if len(toks) == 2:
            src = parse_position(toks[0])
            dst = parse_position(toks[1])
            if src and dst:
                return Move(type=MoveType.MOVE_RING, player=current_player,
                            source=src, destination=dst)
        return None

    if cmd == "markers" and phase == GamePhase.ROW_COMPLETION and len(parts) == 6:
        positions = [parse_position(p) for p in parts[1:]]
        if all(positions):
            return Move(type=MoveType.REMOVE_MARKERS,
                        player=current_player, markers=positions)
        return None

    if cmd == "remove" and len(parts) == 2 and phase == GamePhase.RING_REMOVAL:
        pos = parse_position(parts[1])
        if pos is None:
            return None
        return Move(type=MoveType.REMOVE_RING, player=current_player, source=pos)

    return None


def status_block(s: Session) -> dict:
    g = s.game_state
    human_color = Player.WHITE if s.human_color == "white" else Player.BLACK
    valid_moves = []
    if not g.is_terminal():
        valid_moves = [fmt_move(m) for m in g.get_valid_moves()[:20]]

    # Render the board as text — same path the interactive script uses
    board_str = str(g.board)

    return {
        "session": s.session_id,
        "phase": g.phase.name,
        "current_player": g.current_player.name,
        "human_color": s.human_color,
        "ai_color": "black" if s.human_color == "white" else "white",
        "is_human_turn": g.current_player == human_color,
        "is_terminal": g.is_terminal(),
        "winner": (g.get_winner().name if g.get_winner() else None) if g.is_terminal() else None,
        "white_score": g.white_score,
        "black_score": g.black_score,
        "move_count": s.move_count,
        "board": board_str,
        "valid_moves_sample": valid_moves,
        "recent_log": s.move_log[-10:],
    }


# ----------------------------- AI driver -----------------------------

def _build_mcts(net: NetworkWrapper, sims: int) -> MCTS:
    return MCTS(
        network=net,
        evaluation_mode="pure_neural",
        heuristic_evaluator=None,
        num_simulations=sims,
        late_simulations=sims,
        simulation_switch_ply=10_000,
        enable_subtree_reuse=False,  # Stateless wrapper — subtree reuse is moot
        epsilon_mix_start=0.0,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=0,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
    )


def play_ai_response(s: Session) -> Optional[str]:
    """If it's the AI's turn, run MCTS to pick a move, apply it. Returns
    a fmt_move string or None (if not AI's turn or terminal)."""
    g = s.game_state
    if g.is_terminal():
        return None
    human_color = Player.WHITE if s.human_color == "white" else Player.BLACK
    if g.current_player == human_color:
        return None  # Not AI's turn

    net = NetworkWrapper(model_path=s.checkpoint, device=s.device)
    mcts = _build_mcts(net, s.mcts_sims)
    valid = g.get_valid_moves()
    if not valid:
        return None
    visit_probs = mcts.search_batch(g, s.move_count, batch_size=32)
    probs_t = torch.from_numpy(np.asarray(visit_probs)).to(net.device)
    move = net.select_move(probs_t, valid, temperature=0.0)
    del probs_t
    if move is None:
        return None
    if not g.make_move(move):
        return None
    s.move_count += 1
    s.move_log.append(f"AI: {fmt_move(move)}")
    return fmt_move(move)


# ----------------------------- subcommands -----------------------------

def cmd_new(args) -> int:
    if state_file(args.session).exists():
        print(json.dumps({"error": f"Session {args.session!r} already exists. "
                          f"Use --session <other> or `end` it first."}, indent=2))
        return 2

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device

    if not Path(args.checkpoint).exists():
        print(json.dumps({"error": f"Checkpoint not found: {args.checkpoint}"}))
        return 2

    s = Session(
        session_id=args.session,
        checkpoint=args.checkpoint,
        device=device,
        mcts_sims=args.mcts_sims,
        human_color=args.color,
        game_state=GameState(),
    )

    # If the human is Black, the AI (White) needs to go first.
    if args.color == "black":
        ai_move = play_ai_response(s)
        if ai_move:
            s.move_log.append(f"(opening) {ai_move}")

    save_session(s)
    print(json.dumps(status_block(s), indent=2))
    return 0


def cmd_move(args) -> int:
    s = load_session(args.session)
    g = s.game_state

    if g.is_terminal():
        print(json.dumps({**status_block(s), "error": "Game is already terminal."}, indent=2))
        return 1

    human_color = Player.WHITE if s.human_color == "white" else Player.BLACK
    if g.current_player != human_color:
        print(json.dumps({**status_block(s),
                          "error": "Not your turn — call `move` only when is_human_turn is true."},
                         indent=2))
        return 1

    move = parse_command(args.command, g.current_player, g.phase)
    if move is None:
        print(json.dumps({**status_block(s),
                          "error": f"Could not parse command: {args.command!r}. "
                          f"For ring moves, use 'move D2 E2'. For placements, "
                          f"use 'place F5'. For row removal, use 'markers a b c d e'. "
                          f"For ring removal after capture, use 'remove F5'."}, indent=2))
        return 1

    if not g.make_move(move):
        # Invalid move per game rules. Provide useful diagnostic.
        valid_dests = []
        if g.phase == GamePhase.MAIN_GAME and move.type == MoveType.MOVE_RING:
            valid_dests = [str(p) for p in g.board.valid_move_positions(move.source)]
        print(json.dumps({**status_block(s),
                          "error": f"Move rejected by game engine: {fmt_move(move)}",
                          "valid_destinations_from_source": valid_dests}, indent=2))
        return 1

    s.move_count += 1
    s.move_log.append(f"YOU: {fmt_move(move)}")

    # Now the AI plays until either (a) it's our turn again, (b) terminal
    while not s.game_state.is_terminal():
        cp = s.game_state.current_player
        if cp == human_color:
            break
        ai_move = play_ai_response(s)
        if ai_move is None:
            break

    save_session(s)
    print(json.dumps(status_block(s), indent=2))
    return 0


def cmd_status(args) -> int:
    s = load_session(args.session)
    print(json.dumps(status_block(s), indent=2))
    return 0


def cmd_end(args) -> int:
    d = session_dir(args.session)
    if d.exists():
        for p in d.glob("*"):
            p.unlink()
        d.rmdir()
        print(json.dumps({"ended": args.session}))
    else:
        print(json.dumps({"warn": f"No session at {d}"}))
    return 0


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new")
    p_new.add_argument("--session", required=True)
    p_new.add_argument("--color", default="white", choices=["white", "black"])
    p_new.add_argument("--mcts-sims", type=int, default=100)
    p_new.add_argument("--checkpoint", required=True)
    p_new.add_argument("--device", default="auto")
    p_new.set_defaults(func=cmd_new)

    p_move = sub.add_parser("move")
    p_move.add_argument("--session", required=True)
    p_move.add_argument("--command", required=True,
                        help="e.g. 'place F5', 'move D2 E2', 'markers F6 F7 F8 F9 F10', 'remove F5'")
    p_move.set_defaults(func=cmd_move)

    p_status = sub.add_parser("status")
    p_status.add_argument("--session", required=True)
    p_status.set_defaults(func=cmd_status)

    p_end = sub.add_parser("end")
    p_end.add_argument("--session", required=True)
    p_end.set_defaults(func=cmd_end)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
