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


class PriorOnlyPlayer:
    """search=0 baseline: play straight from the network's raw policy prior,
    no MCTS. Tests whether a better-distilled POLICY helps in play *before* any
    search washes out prior differences — the purest read on policy quality.
    Duck-types the MCTS surface play_game uses (search_batch / get_temperature /
    advance_root). A mild temperature gives game variety (deterministic nets
    would otherwise replay one game); the schedule matches the MCTS sides."""

    def __init__(self, net):
        self.net = net

    def search_batch(self, state, move_number, batch_size=32):
        probs, _ = self.net.predict_from_state(state)
        if torch.is_tensor(probs):
            probs = probs.detach().cpu().numpy()
        return np.asarray(probs, dtype=np.float32).reshape(-1)

    def get_temperature(self, move_number):
        return max(0.1, 0.5 - 0.4 * (move_number / 20.0))  # 0.5 -> 0.1 over 20 plies

    def advance_root(self, move):
        pass


def make_player(net, sims, late_sims):
    """sims<=0 -> prior-only (no search). Otherwise MCTS at a FIXED budget
    (`late_sims=None` => same as `sims`); pass `late_sims` only if you
    deliberately want an early/late split (see the --sims help)."""
    if sims is None or sims <= 0:
        return PriorOnlyPlayer(net)
    return MCTS(
        network=net, evaluation_mode='pure_neural', heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=sims,
        late_simulations=(late_sims if late_sims is not None else sims),
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
    ap.add_argument('--sims', type=int, default=96,
                    help='MCTS budget per move (FIXED for the whole game by default — a '
                         'clean x-axis for a budget sweep). 0 = prior-only, no search. '
                         'For a sweep: 0, 200, 400, 800, 1600, 3200.')
    ap.add_argument('--late-sims', type=int, default=None,
                    help='optional early/late split: sims for ply>=20. Default None = fixed '
                         '(= --sims). Only set this if you deliberately want a split (e.g. to '
                         'mirror a self-play recipe); a sweep should leave it unset.')
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20260528)
    args = ap.parse_args()

    # --white / --black are MODEL SLOTS, not fixed board colors: colors are
    # balanced across games (each model plays white in half), so neither model
    # gets the first-player edge. The verdict is by MODEL (`by_model`); `wins`
    # (by board color) is kept as the first-player diagnostic — note that's the
    # first-player win rate AT THESE MODELS' STRENGTH, not an inherent property
    # of YINSH.
    print(f'[{time.strftime("%H:%M:%S")}] Loading {args.white_label}: {args.white}')
    nw_a = NetworkWrapper(model_path=args.white)
    print(f'[{time.strftime("%H:%M:%S")}] Loading {args.black_label}: {args.black}')
    nw_b = NetworkWrapper(model_path=args.black)

    rng = np.random.default_rng(args.seed)
    games = []
    wins = {'white': 0, 'black': 0, 'draw': 0}                       # by board color
    by_model = {args.white_label: 0, args.black_label: 0, 'draw': 0}  # color-balanced verdict
    if args.games % 2 != 0:
        print(f'[warn] --games={args.games} is odd; colors will be off by one. '
              f'Use an even count for an exactly balanced verdict.')
    t0 = time.time()
    for i in range(args.games):
        a_is_white = (i % 2 == 0)   # alternate which model plays white
        if a_is_white:
            wm, bm = make_player(nw_a, args.sims, args.late_sims), make_player(nw_b, args.sims, args.late_sims)
            wlabel, blabel = args.white_label, args.black_label
        else:
            wm, bm = make_player(nw_b, args.sims, args.late_sims), make_player(nw_a, args.sims, args.late_sims)
            wlabel, blabel = args.black_label, args.white_label
        g = play_game(wm, bm, rng)
        g['game_id'] = i
        g['white_model'] = wlabel
        g['black_model'] = blabel
        games.append(g)
        wins[g['result']] += 1
        if g['result'] == 'draw':
            by_model['draw'] += 1
        else:
            by_model[wlabel if g['result'] == 'white' else blabel] += 1
        elapsed = time.time() - t0
        rate = (i+1) / elapsed * 60
        eta = (args.games - i - 1) / (rate/60) / 60 if rate > 0 else 0
        print(f'[{time.strftime("%H:%M:%S")}] game {i+1}/{args.games}: result={g["result"]} '
              f'({wlabel[:8]}=white W{g["white_score"]}-B{g["black_score"]}, {g["n_moves"]}mv) | '
              f'by-model: {args.white_label}={by_model[args.white_label]} '
              f'{args.black_label}={by_model[args.black_label]} D={by_model["draw"]} '
              f'| {rate:.1f}/min ETA {eta:.1f}m')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'config': {k: v for k, v in vars(args).items()},
                       'by_model': by_model, 'color_balanced': True,
                       'wins': wins,  # by board color = first-player diagnostic
                       'games': games}, f)
    n = max(1, args.games)
    a, b, dr = by_model[args.white_label], by_model[args.black_label], by_model['draw']
    budget = 'prior-only (sims=0)' if args.sims <= 0 else (
        f'sims={args.sims}' + (f'/{args.late_sims} (split)' if args.late_sims is not None else ' (fixed)'))
    print(f'[{time.strftime("%H:%M:%S")}] Final (color-balanced, {budget}): '
          f'{args.white_label} {a} - {b} {args.black_label} (draws {dr})')
    print(f'  {args.white_label} score = {a/n:.3f}  |  first-player(white) win rate '
          f'= {wins["white"]/n:.3f} (at these models\' strength, not an inherent game property)')


if __name__ == '__main__':
    main()
