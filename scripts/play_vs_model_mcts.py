#!/usr/bin/env python3
"""Play YINSH against a trained model, using MCTS for the AI.

Text-based I/O. Designed for sanity-checking model strength against a human
who has a board nearby. Uses the same MCTS path as `eval_vs_heuristic.py`
(pure-neural MCTS, subtree reuse, no root noise) so what you play against
matches what the eval scored.

Usage:
    # Default: play as White against the supervised seed at deployment budget
    python scripts/play_vs_model_mcts.py

    # Play as Black, fewer sims for faster moves while you experiment
    python scripts/play_vs_model_mcts.py --color black --mcts-simulations 64

    # Use a specific checkpoint
    python scripts/play_vs_model_mcts.py \\
        --checkpoint models/supervised_seed/best_supervised_cloud.pt
"""

import argparse
import logging
import sys
import time
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

logger = logging.getLogger("play_vs_model")


# ----------------------------- helpers -----------------------------

def parse_position(pos_str: str) -> Optional[Position]:
    """Parse 'F6' / 'g7' style strings into a Position. Returns None on bad input."""
    s = pos_str.strip()
    if len(s) < 2:
        return None
    try:
        col = s[0].upper()
        row = int(s[1:])
        pos = Position(col, row)
        if not is_valid_position(pos):
            return None
        return pos
    except (ValueError, IndexError):
        return None


def show_state(game: GameState, ai_color: Player) -> None:
    """Print board and key state lines."""
    print()
    print(f"Phase: {game.phase.name}   "
          f"Score W:{game.white_score} / B:{game.black_score}   "
          f"To move: {game.current_player.name}   "
          f"(AI plays {ai_color.name})")
    print(game.board)


def fmt_move(m: Move) -> str:
    """Compact human-readable move string."""
    if m.type == MoveType.PLACE_RING:
        return f"PLACE_RING {m.source}"
    if m.type == MoveType.MOVE_RING:
        return f"MOVE_RING  {m.source} → {m.destination}"
    if m.type == MoveType.REMOVE_MARKERS:
        return f"REMOVE_MARKERS {' '.join(str(p) for p in (m.markers or []))}"
    if m.type == MoveType.REMOVE_RING:
        return f"REMOVE_RING {m.source}"
    return str(m)


# ----------------------------- AI -----------------------------

class MCTSPlayer:
    """AI driven by pure-neural MCTS with subtree reuse."""

    def __init__(self, network: NetworkWrapper, simulations: int):
        self.network = network
        self.simulations = simulations
        self.mcts = MCTS(
            network=network,
            evaluation_mode="pure_neural",
            heuristic_evaluator=None,
            num_simulations=simulations,
            late_simulations=simulations,
            simulation_switch_ply=10_000,
            enable_subtree_reuse=True,
            epsilon_mix_start=0.0,
            epsilon_mix_end=0.0,
            epsilon_mix_taper_moves=0,
            initial_temp=1.0,
            final_temp=1.0,
            annealing_steps=1,
        )
        self.move_count = 0

    def choose_move(self, game: GameState) -> Optional[Move]:
        """Run MCTS, pick the highest-visit move via select_move with temperature=0."""
        valid = game.get_valid_moves()
        if not valid:
            return None
        t0 = time.time()
        visit_probs = self.mcts.search_batch(game, self.move_count, batch_size=32)
        elapsed = time.time() - t0
        probs_t = torch.from_numpy(np.asarray(visit_probs)).to(self.network.device)
        chosen = self.network.select_move(probs_t, valid, temperature=0.0)
        del probs_t
        # Show the AI's top considered moves so the human gets a feel for its preferences.
        print(f"\n[AI thought {elapsed:.1f}s @ {self.simulations} sims]")
        # Top-3 by visit count among valid moves
        scored = []
        for mv in valid:
            try:
                idx = self.network.state_encoder.move_to_index(mv)
                if idx < len(visit_probs):
                    scored.append((mv, float(visit_probs[idx])))
            except Exception:
                continue
        scored.sort(key=lambda x: x[1], reverse=True)
        print("  Top considered:")
        for mv, p in scored[:3]:
            print(f"    {p:.1%}  {fmt_move(mv)}")
        return chosen

    def update_after_move(self, played: Move) -> None:
        self.mcts.advance_root(played)
        self.move_count += 1


# ----------------------------- Human input -----------------------------

class HumanPlayer:
    """Text-driven human input. Tracks selected ring across two-step inputs."""

    def __init__(self):
        self.selected_ring: Optional[Position] = None
        self.valid_destinations: List[Position] = []

    def _print_help(self, game: GameState) -> None:
        print("\nCommands:")
        if game.phase == GamePhase.RING_PLACEMENT:
            print("  place F6           Place a ring at F6")
        elif game.phase == GamePhase.MAIN_GAME:
            print("  select F6          Pick up the ring at F6")
            print("  move H8            Move the selected ring to H8")
            print("  cancel             Unselect the ring")
        elif game.phase == GamePhase.ROW_COMPLETION:
            print("  markers F6 F7 F8 F9 F10   Pick 5 markers in a row to remove")
        elif game.phase == GamePhase.RING_REMOVAL:
            print("  remove F6          Remove your ring at F6")
        print("  show               Reprint the board")
        print("  hint               Ask the model what it'd play")
        print("  quit               Exit the game (forfeit)")

    def _suggest_with_ai(self, game: GameState, ai: MCTSPlayer) -> None:
        """Optional 'hint' command — show what the AI would play. Doesn't advance the game."""
        valid = game.get_valid_moves()
        if not valid:
            return
        probs = ai.mcts.search_batch(game, ai.move_count, batch_size=32)
        scored = []
        for mv in valid:
            try:
                idx = ai.network.state_encoder.move_to_index(mv)
                if idx < len(probs):
                    scored.append((mv, float(probs[idx])))
            except Exception:
                continue
        scored.sort(key=lambda x: x[1], reverse=True)
        print("\n  [Hint] Top moves by AI visits:")
        for mv, p in scored[:5]:
            print(f"    {p:.1%}  {fmt_move(mv)}")
        # Reset the cached root because the hint search mutated it.
        ai.mcts.clear_cached_root() if hasattr(ai.mcts, "clear_cached_root") else None

    def choose_move(self, game: GameState, ai: Optional[MCTSPlayer] = None) -> Optional[Move]:
        """Loop until a valid move is constructed or 'quit' is entered."""
        while True:
            if game.phase == GamePhase.RING_PLACEMENT:
                prompt = f"\n[{game.current_player.name}] place a ring (e.g. 'place F6'): "
            elif game.phase == GamePhase.MAIN_GAME:
                if self.selected_ring is None:
                    prompt = f"\n[{game.current_player.name}] select a ring (e.g. 'select F6'): "
                else:
                    dests = ", ".join(str(p) for p in self.valid_destinations)
                    prompt = (f"\n[{game.current_player.name}] ring at {self.selected_ring} selected.\n"
                             f"  Valid destinations: {dests}\n"
                             f"  Enter 'move <pos>' or 'cancel': ")
            elif game.phase == GamePhase.ROW_COMPLETION:
                prompt = f"\n[{game.current_player.name}] pick 5 markers in a row to remove: "
            elif game.phase == GamePhase.RING_REMOVAL:
                prompt = f"\n[{game.current_player.name}] remove a ring (e.g. 'remove F6'): "
            else:
                return None

            raw = input(prompt).strip().lower()
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0]

            if cmd in ("quit", "q", "exit"):
                return None
            if cmd in ("help", "?"):
                self._print_help(game)
                continue
            if cmd == "show":
                show_state(game, ai_color=Player.WHITE)  # color fyi for header only
                continue
            if cmd == "hint" and ai is not None:
                self._suggest_with_ai(game, ai)
                continue
            if cmd == "cancel":
                self.selected_ring = None
                self.valid_destinations = []
                continue

            # Phase-specific
            try:
                if cmd == "place" and len(parts) == 2 and game.phase == GamePhase.RING_PLACEMENT:
                    pos = parse_position(parts[1])
                    if pos is None:
                        print(f"  Bad position: {parts[1]}")
                        continue
                    return Move(type=MoveType.PLACE_RING,
                                player=game.current_player, source=pos)

                if cmd == "select" and len(parts) == 2 and game.phase == GamePhase.MAIN_GAME:
                    pos = parse_position(parts[1])
                    if pos is None:
                        print(f"  Bad position: {parts[1]}")
                        continue
                    piece = game.board.get_piece(pos)
                    if piece is None or not piece.is_ring():
                        print(f"  No ring at {pos}.")
                        continue
                    expected_ring = (PieceType.WHITE_RING
                                     if game.current_player == Player.WHITE
                                     else PieceType.BLACK_RING)
                    if piece != expected_ring:
                        print(f"  That ring isn't yours.")
                        continue
                    self.selected_ring = pos
                    self.valid_destinations = list(game.board.valid_move_positions(pos))
                    if not self.valid_destinations:
                        print(f"  Ring at {pos} has no legal moves; pick a different one.")
                        self.selected_ring = None
                    continue

                if cmd == "move" and len(parts) == 2 and game.phase == GamePhase.MAIN_GAME:
                    if self.selected_ring is None:
                        print("  Select a ring first.")
                        continue
                    dest = parse_position(parts[1])
                    if dest is None:
                        print(f"  Bad position: {parts[1]}")
                        continue
                    if dest not in self.valid_destinations:
                        print(f"  Not a valid destination. Choices: "
                              f"{', '.join(str(p) for p in self.valid_destinations)}")
                        continue
                    src = self.selected_ring
                    self.selected_ring = None
                    self.valid_destinations = []
                    return Move(type=MoveType.MOVE_RING,
                                player=game.current_player,
                                source=src, destination=dest)

                if cmd == "markers" and game.phase == GamePhase.ROW_COMPLETION:
                    if len(parts) != 6:
                        print("  Need exactly 5 marker positions.")
                        continue
                    positions = [parse_position(p) for p in parts[1:]]
                    if not all(positions):
                        print("  One or more positions invalid.")
                        continue
                    return Move(type=MoveType.REMOVE_MARKERS,
                                player=game.current_player,
                                markers=positions)

                if cmd == "remove" and len(parts) == 2 and game.phase == GamePhase.RING_REMOVAL:
                    pos = parse_position(parts[1])
                    if pos is None:
                        print(f"  Bad position: {parts[1]}")
                        continue
                    return Move(type=MoveType.REMOVE_RING,
                                player=game.current_player, source=pos)

                print(f"  Command '{cmd}' not valid in phase {game.phase.name}. "
                      f"Type 'help' for options.")
            except Exception as e:
                print(f"  Input error: {e}")


# ----------------------------- Game loop -----------------------------

def play(args) -> None:
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    print(f"Device: {device}")

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    print(f"Loading model: {args.checkpoint}")
    network = NetworkWrapper(device=device)
    network.load_model(str(args.checkpoint))

    human_color = Player.WHITE if args.color.lower() == "white" else Player.BLACK
    ai_color = Player.BLACK if human_color == Player.WHITE else Player.WHITE

    ai = MCTSPlayer(network=network, simulations=args.mcts_simulations)
    human = HumanPlayer()

    print("\n" + "=" * 60)
    print(f"YINSH — you are {human_color.name}, AI is {ai_color.name}")
    print(f"AI: pure-neural MCTS @ {args.mcts_simulations} sims/move")
    print(f"Type 'help' at any prompt for commands. 'hint' asks the AI.")
    print("=" * 60)

    game = GameState()
    move_num = 0
    show_state(game, ai_color)

    while not game.is_terminal() and move_num < args.max_moves:
        if game.current_player == human_color:
            mv = human.choose_move(game, ai=ai)
            if mv is None:
                print("\nQuitting (forfeit).")
                return
        else:
            mv = ai.choose_move(game)
            if mv is None:
                print("\nAI returned no move — game ends.")
                break
            print(f"\n[AI plays] {fmt_move(mv)}")

        if not game.make_move(mv):
            print(f"  ✗ Move rejected: {fmt_move(mv)}")
            if game.current_player != human_color:
                print("  AI made an illegal move; ending game.")
                break
            continue

        if game.current_player != human_color:
            ai.update_after_move(mv)
        else:
            # AI just moved (state already advanced to human turn) — also advance MCTS root.
            ai.update_after_move(mv)
        move_num += 1
        show_state(game, ai_color)

    print("\n" + "=" * 60)
    if game.is_terminal():
        winner = game.get_winner()
        if winner is None:
            print("Game ended in a draw.")
        elif winner == human_color:
            print(f"You won as {human_color.name}!")
        else:
            print(f"AI won as {ai_color.name}.")
    else:
        print(f"Move cap ({args.max_moves}) reached.")
    print(f"Final score — W:{game.white_score} / B:{game.black_score}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Play YINSH against a trained model (MCTS).")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("models/supervised_seed/best_supervised_cloud.pt"),
        help="Path to model checkpoint .pt. Default expects the cloud-trained 10-epoch seed; "
             "falls back to best_supervised.pt if cloud copy missing.",
    )
    parser.add_argument("--color", type=str, default="white",
                        choices=["white", "black"], help="Your color")
    parser.add_argument("--mcts-simulations", type=int, default=100,
                        help="MCTS sims/move. 64=fast, 100=normal (eval used 100), "
                             "400=deployment 'hard' preset (slow but stronger)")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / mps / cpu / auto (default: auto)")
    parser.add_argument("--max-moves", type=int, default=400)
    args = parser.parse_args()

    # Fallback: if user kept default but cloud checkpoint is missing, try the local one.
    if not args.checkpoint.exists():
        fallback = Path("models/supervised_seed/best_supervised.pt")
        if fallback.exists():
            print(f"NOTE: {args.checkpoint} not found; falling back to {fallback} (1-epoch local seed).")
            args.checkpoint = fallback

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    play(args)


if __name__ == "__main__":
    main()
