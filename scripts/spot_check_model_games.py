#!/usr/bin/env python3
"""Generate a small batch of model-vs-HeuristicAgent games for spot-check.

Mirrors the player loop in tournament.run_anchor_eval (pure-neural MCTS for
the model side, HeuristicAgent at the chosen depth for the anchor) but adds
a GameRecorder around each game so output drops straight into the existing
dashboard viewer (`scripts/dashboard_games.py`).

Use when you want eyes on actual play — e.g. spot-check a pretrained
checkpoint while waiting for the aggregate validation gate to finish.

Usage:
    python scripts/spot_check_model_games.py \\
        --checkpoint models/yngine_volume_pretrain/best_supervised.pt \\
        --num-games 5 \\
        --depth 1 \\
        --mcts-simulations 64 \\
        --output-dir self_play_data/spot_check_yngine_v1/ \\
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.game.types import Player  # noqa: E402
from yinsh_ml.heuristics import YinshHeuristics  # noqa: E402
from yinsh_ml.network.wrapper import NetworkWrapper  # noqa: E402
from yinsh_ml.self_play.game_recorder import GameRecorder  # noqa: E402
from yinsh_ml.training.self_play import MCTS  # noqa: E402

logger = logging.getLogger("spot_check_model_games")


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


def play_one_game(
    network: NetworkWrapper,
    candidate_is_white: bool,
    depth: int,
    mcts_sims: int,
    game_id: str,
    output_dir: Path,
    max_moves: int,
    temperature: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    heuristic = YinshHeuristics()
    anchor = HeuristicAgent(config=HeuristicAgentConfig(
        max_depth=depth,
        random_tiebreak=False,
        random_seed=seed,
        use_transposition_table=True,
    ), evaluator=heuristic)
    anchor.clear_transposition_table()

    mcts = build_mcts(network, mcts_sims)

    recorder = GameRecorder(output_dir=str(output_dir / "_worker_scratch"),
                            save_json=False)
    recorder.start_game(game_id)

    state = GameState()
    move_count = 0
    started = time.time()

    while not state.is_terminal() and move_count < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break

        cand_to_move = (
            (state.current_player == Player.WHITE and candidate_is_white)
            or (state.current_player == Player.BLACK and not candidate_is_white)
        )

        if cand_to_move:
            visit_probs = mcts.search_batch(state, move_count, batch_size=32)
            visit_probs_t = torch.from_numpy(np.asarray(visit_probs)).to(network.device)
            move = network.select_move(visit_probs_t, list(valid),
                                       temperature=temperature)
            del visit_probs_t
        else:
            move = anchor.select_move(state)

        if move is None:
            break

        recorder.record_turn(state, move, state.current_player)
        ok = state.make_move(move)
        if not ok:
            logger.warning(f"{game_id}: invalid move at move {move_count}, aborting")
            break
        move_count += 1

    winner = state.get_winner() if state.is_terminal() else None
    record = recorder.end_game(state, winner=winner)
    elapsed = time.time() - started

    cand_color = "white" if candidate_is_white else "black"
    return {
        "game_id": game_id,
        "moves": move_count,
        "winner": str(record.winner) if record and record.winner else None,
        "candidate_color": cand_color,
        "white_score": record.final_score.get("white", 0) if record else 0,
        "black_score": record.final_score.get("black", 0) if record else 0,
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--depth", type=int, default=1,
                        help="HeuristicAgent search depth")
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed-base", type=int, default=20260519)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    network = NetworkWrapper(model_path=str(args.checkpoint), device=device)
    network.network.eval()

    # Split candidate side half white / half black so we surface any
    # side-asymmetry in the model's play.
    half = args.num_games // 2
    color_order = [True] * half + [False] * (args.num_games - half)

    results = []
    for i, cand_white in enumerate(color_order):
        game_id = f"spot_check_{args.checkpoint.stem}_{i:03d}"
        logger.info(f"Game {i + 1}/{args.num_games} "
                    f"(candidate={'white' if cand_white else 'black'})")
        info = play_one_game(
            network=network,
            candidate_is_white=cand_white,
            depth=args.depth,
            mcts_sims=args.mcts_simulations,
            game_id=game_id,
            output_dir=args.output_dir,
            max_moves=args.max_moves,
            temperature=args.temperature,
            seed=args.seed_base + i,
        )
        results.append(info)
        logger.info(f"  → {info}")

    cand_wins = sum(1 for r in results
                    if r["winner"] and (
                        (r["winner"] == "Player.WHITE" and r["candidate_color"] == "white")
                        or (r["winner"] == "Player.BLACK" and r["candidate_color"] == "black")))
    logger.info("")
    logger.info(f"Summary: {cand_wins}/{len(results)} candidate wins; "
                f"see {args.output_dir} for parquet files.")


if __name__ == "__main__":
    main()
