"""Two-model head-to-head with MCTS for both sides. Records outcomes."""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import MoveType
from yinsh_ml.game.constants import Player
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import MCTS
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.utils.encoding import StateEncoder


def make_mcts(net):
    return MCTS(
        network=net, evaluation_mode='pure_neural', heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=96, late_simulations=64,
        simulation_switch_ply=20, c_puct=1.0, dirichlet_alpha=0.0,
        value_weight=1.0, max_depth=300, epsilon_mix_start=0.0,
        epsilon_mix_end=0.0, epsilon_mix_taper_moves=1, initial_temp=0.5,
        final_temp=0.1, annealing_steps=20, temp_clamp_fraction=0.6,
        enable_subtree_reuse=True, fpu_reduction=0.25,
    )


def select_move(policy, valid, encoder, temp, rng):
    probs = np.zeros(len(valid), dtype=np.float64)
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


def play_game(white_mcts, black_mcts, rng, max_moves=300):
    encoder = StateEncoder()
    state = GameState()
    moves = []
    move_count = 0
    while not state.is_terminal() and move_count < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break
        mcts = white_mcts if state.current_player == Player.WHITE else black_mcts
        policy = mcts.search_batch(state, move_count, batch_size=32)
        temp = mcts.get_temperature(move_count)
        sel = select_move(policy, valid, encoder, temp, rng)
        moves.append({
            'move_number': move_count,
            'player': 'white' if state.current_player == Player.WHITE else 'black',
            'move_type': sel.type.name,
            'position': str(sel.source) if sel.type == MoveType.PLACE_RING else f'{sel.source}->{sel.destination}'
        })
        state.make_move(sel)
        white_mcts.advance_root(sel)
        black_mcts.advance_root(sel)
        move_count += 1
    if state.white_score > state.black_score:
        result = 'white'
    elif state.black_score > state.white_score:
        result = 'black'
    else:
        result = 'draw'
    return {
        'result': result,
        'white_score': state.white_score,
        'black_score': state.black_score,
        'n_moves': move_count,
        'moves': moves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--white', required=True)
    ap.add_argument('--black', required=True)
    ap.add_argument('--white-label', default='white')
    ap.add_argument('--black-label', default='black')
    ap.add_argument('--games', type=int, default=15)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20260528)
    args = ap.parse_args()

    print(f'[{time.strftime("%H:%M:%S")}] Loading {args.white_label}: {args.white}')
    nw_w = NetworkWrapper(model_path=args.white)
    print(f'[{time.strftime("%H:%M:%S")}] Loading {args.black_label}: {args.black}')
    nw_b = NetworkWrapper(model_path=args.black)

    rng = np.random.default_rng(args.seed)
    games = []
    wins = {'white': 0, 'black': 0, 'draw': 0}
    t0 = time.time()
    for i in range(args.games):
        wm = make_mcts(nw_w)
        bm = make_mcts(nw_b)
        g = play_game(wm, bm, rng)
        g['game_id'] = i
        g['white_model'] = args.white_label
        g['black_model'] = args.black_label
        games.append(g)
        wins[g['result']] += 1
        elapsed = time.time() - t0
        rate = (i+1) / elapsed * 60
        eta = (args.games - i - 1) / (rate/60) / 60 if rate > 0 else 0
        print(f'[{time.strftime("%H:%M:%S")}] game {i+1}/{args.games}: result={g["result"]} '
              f'(W{g["white_score"]}-B{g["black_score"]}, {g["n_moves"]}mv) | '
              f'totals: W={wins["white"]} B={wins["black"]} D={wins["draw"]} '
              f'| {rate:.1f}/min ETA {eta:.1f}m')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'config': {k:v for k,v in vars(args).items() if k != 'output' or True}, 'wins': wins, 'games': games}, f)
    print(f'[{time.strftime("%H:%M:%S")}] Final: W={wins["white"]} B={wins["black"]} D={wins["draw"]}')


if __name__ == '__main__':
    main()
