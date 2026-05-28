"""Measure model self-play opening distribution and write JSON in the same
format as expert_games for direct comparison.

Run from repo root:
    python scripts/measure_model_openings.py \
        --checkpoint models/iter1_ema_2026-05-27/iter1_ema.pt \
        --games 200 \
        --output analysis_board/multiplayer/model_openings_iter1_ema.json \
        --protocol deployed_sampled

`deployed_sampled` protocol — closest to "what the friend saw, plus enough
stochasticity to recover distribution shape": pure_neural MCTS, no
Dirichlet noise, temperature 0.5 (sample from MCTS visits), 96 sims/move.

`training` protocol — what the policy is actually trained against: hybrid
MCTS w/ heuristic_weight=0.5, Dirichlet 0.25→0 over 20 plies, temp 1.0→0.1
annealed.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase, MoveType
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import MCTS
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.utils.encoding import StateEncoder


def move_to_dict(move):
    player_str = "white" if move.player.name == "WHITE" else "black"
    if move.type == MoveType.PLACE_RING:
        return {"move_type": "PLACE_RING", "player": player_str, "position": str(move.source)}
    if move.type == MoveType.MOVE_RING:
        return {"move_type": "MOVE_RING", "player": player_str,
                "source": str(move.source), "destination": str(move.destination)}
    if move.type == MoveType.REMOVE_MARKERS:
        return {"move_type": "REMOVE_MARKERS", "player": player_str,
                "markers": [str(m) for m in (move.markers or ())]}
    return {"move_type": "REMOVE_RING", "player": player_str, "position": str(move.source)}


def select_move_from_visits(mcts_policy, valid_moves, encoder, temperature, rng):
    """Sample a move from MCTS visit distribution at the given temperature."""
    probs = np.zeros(len(valid_moves), dtype=np.float64)
    for i, mv in enumerate(valid_moves):
        idx = encoder.move_to_index(mv)
        if 0 <= idx < len(mcts_policy):
            probs[i] = float(mcts_policy[idx])
    if probs.sum() <= 0:
        # MCTS returned no mass on any valid move — fall back to uniform.
        return valid_moves[rng.integers(0, len(valid_moves))]
    if temperature <= 1e-3:
        return valid_moves[int(np.argmax(probs))]
    probs = probs ** (1.0 / temperature)
    probs /= probs.sum()
    return valid_moves[rng.choice(len(valid_moves), p=probs)]


PROTOCOLS = {
    "deployed_sampled": dict(
        evaluation_mode="pure_neural",
        heuristic_weight=0.0,
        num_simulations=96,
        late_simulations=64,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        epsilon_mix_start=0.0,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=1,
        initial_temp=0.5,
        final_temp=0.1,
        annealing_steps=20,
        temp_clamp_fraction=0.6,
        use_batched_mcts=True,
        mcts_batch_size=32,
        max_depth=300,
        fpu_reduction=0.25,
    ),
    "training": dict(
        evaluation_mode="hybrid",
        heuristic_weight=0.5,
        num_simulations=96,
        late_simulations=64,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        epsilon_mix_start=0.25,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=20,
        initial_temp=1.0,
        final_temp=0.1,
        annealing_steps=30,
        temp_clamp_fraction=0.6,
        use_batched_mcts=True,
        mcts_batch_size=32,
        max_depth=300,
        fpu_reduction=0.25,
    ),
}


def build_mcts(network, protocol, heuristic):
    cfg = dict(PROTOCOLS[protocol])
    # MCTS doesn't accept use_batched_mcts / mcts_batch_size as direct kwargs —
    # batched search is the default code path in search_batch(). Strip them.
    cfg.pop("use_batched_mcts", None)
    batch_size = cfg.pop("mcts_batch_size", 32)
    needs_heuristic = cfg["evaluation_mode"] in ("hybrid", "pure_heuristic")
    return MCTS(
        network=network,
        heuristic_evaluator=heuristic if needs_heuristic else None,
        enable_subtree_reuse=True,
        **cfg,
    ), batch_size


def play_game(network, protocol, heuristic, rng, game_id, max_moves=300):
    mcts, batch_size = build_mcts(network, protocol, heuristic)
    encoder = StateEncoder()
    state = GameState()
    moves = []
    move_count = 0
    while not state.is_terminal() and move_count < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break
        policy = mcts.search_batch(state, move_count, batch_size=batch_size)
        temp = mcts.get_temperature(move_count)
        selected = select_move_from_visits(policy, valid, encoder, temp, rng)
        moves.append(move_to_dict(selected))
        state.make_move(selected)
        mcts.advance_root(selected)
        move_count += 1
    if state.white_score > state.black_score:
        result = "white"
    elif state.black_score > state.white_score:
        result = "black"
    else:
        result = "draw"
    return {
        "source": "model_selfplay",
        "game_id": f"selfplay_{game_id}",
        "players": {
            "white": {"name": "model", "rating": 0},
            "black": {"name": "model", "rating": 0},
        },
        "result": result,
        "moves": moves,
        "white_score": state.white_score,
        "black_score": state.black_score,
        "n_moves": move_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="deployed_sampled")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    print(f"[{time.strftime('%H:%M:%S')}] Loading checkpoint: {args.checkpoint}")
    network = NetworkWrapper(model_path=args.checkpoint)
    heuristic = YinshHeuristics()
    rng = np.random.default_rng(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games = []
    t0 = time.time()
    for i in range(args.games):
        g = play_game(network, args.protocol, heuristic, rng, game_id=i)
        games.append(g)
        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.games - i - 1) / rate / 60.0
            print(f"[{time.strftime('%H:%M:%S')}] game {i+1}/{args.games} "
                  f"({rate*60:.1f} games/min, ETA {eta:.1f}m)")
            with out_path.open("w") as f:
                json.dump(games, f)
    with out_path.open("w") as f:
        json.dump(games, f)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {len(games)} games to {out_path}")


if __name__ == "__main__":
    main()
