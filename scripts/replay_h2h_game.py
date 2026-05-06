#!/usr/bin/env python3
"""Play a single H2H game between two checkpoints with verbose move-by-move output.

Companion to eval_head_to_head.py — when the H2H aggregate stat looks weird
(e.g. white-wins-100% under deterministic argmax), this script lets you
actually watch one game unfold and inspect what each side is doing on each ply.

Usage:
    # Watch iter_19 (white) vs iter_19 (black) — same checkpoint both sides.
    # If the network plays "score ASAP and ignore defense", you'll see white
    # building toward a row while black makes uncoordinated/non-defensive moves.
    python scripts/replay_h2h_game.py \\
        --run-dir runs/20260503_140258 \\
        --white-iter 19 \\
        --black-iter 19 \\
        --device cuda

    # Cross-checkpoint replay
    python scripts/replay_h2h_game.py \\
        --run-dir runs/20260503_140258 \\
        --white-iter 9 --black-iter 19 \\
        --temperature 0.0 \\
        --num-games 3 \\
        --board-every 5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Player

logger = logging.getLogger("replay_h2h")


def find_checkpoint(run_dir: Path, iter_idx: int) -> Path:
    ckpt = run_dir / f"iteration_{iter_idx}" / f"checkpoint_iteration_{iter_idx}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def play_one_verbose(
    white_net: NetworkWrapper,
    black_net: NetworkWrapper,
    game_seed: int,
    max_moves: int = 200,
    temperature: float = 0.0,
    board_every: int = 0,
) -> dict:
    """Play one game and print every move. Returns final summary dict."""
    torch.manual_seed(game_seed)
    np.random.seed(game_seed)

    game = GameState()
    move_log = []
    move_count = 0

    while not game.is_terminal() and move_count < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            print(f"  [ply {move_count}] {game.current_player.name}: NO VALID MOVES (stalemate)")
            break

        is_white_turn = (game.current_player == Player.WHITE)
        net = white_net if is_white_turn else black_net

        state_array = net.state_encoder.encode_state(game)
        inp = torch.from_numpy(np.array(state_array)).unsqueeze(0)
        move_probs, value_est = net.predict(inp)
        selected = net.select_move(move_probs, valid_moves, temperature=temperature)

        if selected is None:
            print(f"  [ply {move_count}] {game.current_player.name}: select_move returned None")
            break

        # Capture pre-move scores for delta tracking
        ws_before, bs_before = game.white_score, game.black_score

        # Print move with phase + value-head estimate
        v = float(value_est.item()) if hasattr(value_est, 'item') else float(value_est)
        phase_str = game.phase.name[:4]
        print(
            f"  [ply {move_count:3d} {phase_str}] {game.current_player.name:5s} "
            f"v={v:+.3f} | {selected}"
        )

        if not game.make_move(selected):
            print(f"  [ply {move_count}] make_move REJECTED {selected}")
            break

        # Note score changes (= row capture or scoring event)
        if game.white_score != ws_before or game.black_score != bs_before:
            print(
                f"           ↳ SCORE: White={game.white_score} Black={game.black_score}"
            )

        move_log.append(str(selected))
        move_count += 1

        # Periodic board snapshot
        if board_every > 0 and move_count % board_every == 0:
            print(f"  --- board after ply {move_count} ---")
            print("  " + str(game.board).replace("\n", "\n  "))

    winner = game.get_winner()
    winner_str = winner.name if winner else "DRAW"
    print(f"  RESULT: winner={winner_str} | total_plies={move_count} | "
          f"final_score={game.white_score}-{game.black_score}")
    print(f"  --- final board ---")
    print("  " + str(game.board).replace("\n", "\n  "))
    return {
        "winner": winner_str,
        "plies": move_count,
        "white_score": game.white_score,
        "black_score": game.black_score,
        "moves": move_log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--white-iter", type=int, required=True)
    parser.add_argument("--black-iter", type=int, required=True)
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--board-every", type=int, default=0,
                        help="Print board every N plies (0 = only at end)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    white_ckpt = find_checkpoint(args.run_dir, args.white_iter)
    black_ckpt = find_checkpoint(args.run_dir, args.black_iter)

    logger.info(f"White: iter_{args.white_iter} from {white_ckpt}")
    logger.info(f"Black: iter_{args.black_iter} from {black_ckpt}")

    white_net = NetworkWrapper(device=device)
    white_net.load_model(str(white_ckpt))
    black_net = NetworkWrapper(device=device)
    black_net.load_model(str(black_ckpt))

    summaries = []
    for g in range(args.num_games):
        seed = args.seed + g
        print(f"\n{'=' * 70}")
        print(f"GAME {g+1}/{args.num_games}  seed={seed}  temperature={args.temperature}")
        print(f"  WHITE = iter_{args.white_iter}   BLACK = iter_{args.black_iter}")
        print(f"{'=' * 70}")
        s = play_one_verbose(
            white_net, black_net, seed,
            max_moves=args.max_moves,
            temperature=args.temperature,
            board_every=args.board_every,
        )
        summaries.append(s)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    w_wins = sum(1 for s in summaries if s["winner"] == "WHITE")
    b_wins = sum(1 for s in summaries if s["winner"] == "BLACK")
    draws = len(summaries) - w_wins - b_wins
    avg_plies = np.mean([s["plies"] for s in summaries])
    print(f"  WHITE wins: {w_wins} | BLACK wins: {b_wins} | DRAW/inconclusive: {draws}")
    print(f"  Avg plies: {avg_plies:.1f}")


if __name__ == "__main__":
    main()
