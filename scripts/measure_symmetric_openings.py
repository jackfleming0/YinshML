"""Symmetric MCTS self-play — tests whether enforcing D2 symmetry at
inference eliminates the asymmetric A5-style modal that we measured in
iter1_ema, and whether it changes the F6-cluster behavior of the
dropout+LS model.

For each move:
1. Run MCTS once at the current state s — get policy_0 over actions at s
2. Run MCTS once at each of 3 D2-transformed states (T1(s), T2(s), T3(s))
   — get policy_i over actions at T_i(s)
3. Inverse-transform each policy_i back to actions at s (since T_i is
   involution in D2, inverse = forward)
4. Average the 4 policies → policy_sym
5. Sample/argmax move from policy_sym

Cost: 4x slower than vanilla MCTS per move. Roughly 3-4 min per game on
MPS at 96 sims.

Output JSON matches the format of scripts/measure_model_openings.py for
easy comparison.

Hypothesis: if iter1_ema's A5 72% / A2 0% asymmetry comes from
path-dependent MCTS visit counts (not from network weights), symmetric
MCTS averaging should produce a symmetric distribution where A5, K7,
E1, G11 each appear at ~18% (since they're in the same D2 orbit).

If asymmetry comes from network weights, the symmetric averaging will
only partially help.
"""

import argparse, json, time, traceback
from pathlib import Path

import numpy as np
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import MoveType
from yinsh_ml.game.constants import Player
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import MCTS
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.training.augmentation import YinshSymmetryAugmenter


def make_mcts(net, sims):
    """Create a fresh MCTS instance. Subtree reuse is disabled because each
    symmetric variant needs a fresh search; we can't carry state across the
    4 transform variants of the same move."""
    return MCTS(
        network=net, evaluation_mode='pure_neural', heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=sims, late_simulations=sims,
        simulation_switch_ply=20, c_puct=1.0, dirichlet_alpha=0.0,
        value_weight=1.0, max_depth=300, epsilon_mix_start=0.0,
        epsilon_mix_end=0.0, epsilon_mix_taper_moves=1, initial_temp=0.5,
        final_temp=0.1, annealing_steps=20, temp_clamp_fraction=0.6,
        enable_subtree_reuse=False, fpu_reduction=0.25,
    )


def symmetric_search(state, net, augmenter, basic_encoder, sims, move_num):
    """Run 4 D2-symmetric MCTS searches and return policy averaged into the
    original action space."""
    basic_state = basic_encoder.encode_state(state)
    avg_policy = np.zeros(basic_encoder.total_moves, dtype=np.float64)
    n_valid = 0

    for tid in range(4):
        try:
            if tid == 0:
                transformed_game = state
                transformed_basic = basic_state
            else:
                transformed_basic = augmenter._transform_state(basic_state, tid)
                transformed_game = basic_encoder.decode_state(transformed_basic)

            mcts = make_mcts(net, sims)
            policy_i = mcts.search_batch(transformed_game, move_num, batch_size=32)

            if tid == 0:
                avg_policy += policy_i
            else:
                # The augmenter's _transform_policy maps a policy defined on
                # state's action space to T_tid(state)'s action space.
                # We have policy_i defined on transformed_basic's action space,
                # so we pass (transformed_basic, policy_i, tid) to get a policy
                # on T_tid(transformed_basic) = T_tid(T_tid(basic_state)) =
                # basic_state (since T_tid is involution in D2).
                policy_orig = augmenter._transform_policy(transformed_basic, policy_i, tid)
                avg_policy += policy_orig
            n_valid += 1
        except Exception as e:
            # Transform might fail at terminal/edge states — skip that variant
            continue

    if n_valid > 0:
        avg_policy /= n_valid
    return avg_policy


def select_move(policy, valid, encoder, temp, rng):
    probs = np.zeros(len(valid))
    for i, mv in enumerate(valid):
        idx = encoder.move_to_index(mv)
        if 0 <= idx < len(policy):
            probs[i] = float(policy[idx])
    if probs.sum() <= 0:
        return valid[rng.integers(0, len(valid))]
    if temp <= 1e-3:
        return valid[int(np.argmax(probs))]
    probs = probs ** (1.0 / temp)
    probs /= probs.sum()
    return valid[rng.choice(len(valid), p=probs)]


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


def get_temp(move_count, annealing_steps=20, initial=0.5, final=0.1):
    """Match the temperature schedule used in measure_model_openings.py."""
    clamp = int(annealing_steps * 0.6)
    clamp = max(1, clamp)
    if move_count >= clamp:
        return final
    return initial - (initial - final) * (move_count / clamp)


def play_game(net, augmenter, basic_encoder, sims, rng, game_id, max_moves=300):
    state = GameState()
    moves = []
    move_count = 0
    while not state.is_terminal() and move_count < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break
        policy = symmetric_search(state, net, augmenter, basic_encoder, sims, move_count)
        temp = get_temp(move_count)
        sel = select_move(policy, valid, basic_encoder, temp, rng)
        moves.append(move_to_dict(sel))
        state.make_move(sel)
        move_count += 1

    if state.white_score > state.black_score:
        result = 'white'
    elif state.black_score > state.white_score:
        result = 'black'
    else:
        result = 'draw'
    return {
        "source": "model_symmetric_selfplay",
        "game_id": f"sym_{game_id}",
        "result": result,
        "white_score": state.white_score,
        "black_score": state.black_score,
        "n_moves": move_count,
        "moves": moves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--use-enhanced-encoding', action='store_true', default=True)
    ap.add_argument('--no-enhanced', dest='use_enhanced_encoding', action='store_false')
    ap.add_argument('--games', type=int, default=50)
    ap.add_argument('--sims', type=int, default=96)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20260530)
    args = ap.parse_args()

    print(f'[{time.strftime("%H:%M:%S")}] Loading: {args.checkpoint}')
    print(f'  enhanced_encoding: {args.use_enhanced_encoding}')
    print(f'  games: {args.games}, sims: {args.sims}')
    net = NetworkWrapper(model_path=args.checkpoint, use_enhanced_encoding=args.use_enhanced_encoding)

    # Augmenter uses basic encoder for geometric transforms (the spatial layout
    # is the same regardless of whether enhanced/basic; only channel count differs,
    # which the augmenter handles by iterating over state.shape[0])
    basic_encoder = StateEncoder()
    augmenter = YinshSymmetryAugmenter(include_reflections=True, state_encoder=basic_encoder)

    rng = np.random.default_rng(args.seed)
    games = []
    t0 = time.time()
    for i in range(args.games):
        try:
            g = play_game(net, augmenter, basic_encoder, args.sims, rng, i)
        except Exception as e:
            print(f'[{time.strftime("%H:%M:%S")}] game {i+1} FAILED: {e}')
            traceback.print_exc()
            continue
        games.append(g)
        elapsed = time.time() - t0
        rate = (i+1) / elapsed * 60
        eta = (args.games - i - 1) / (rate/60) / 60 if rate > 0 else 0
        print(f'[{time.strftime("%H:%M:%S")}] game {i+1}/{args.games}: '
              f'result={g["result"]} (W{g["white_score"]}-B{g["black_score"]}, {g["n_moves"]}mv) '
              f'| {rate:.2f}/min ETA {eta:.1f}m', flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(games, f)

    print(f'[{time.strftime("%H:%M:%S")}] Done. Wrote {len(games)} games to {args.output}')


if __name__ == '__main__':
    main()
