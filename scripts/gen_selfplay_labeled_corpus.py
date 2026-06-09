#!/usr/bin/env python
"""Generate an IN-DISTRIBUTION-STRONG held-out value corpus for E25 Part 1.

Plays NEURAL-MCTS self-play with a single net (the champion, iter1_ema), records
every MAIN_GAME position, and labels it by the game's final outcome from the
side-to-move POV (z in {-1, 0, +1}). Saves a `states`/`values` .npz consumable by
`scripts/value_head_calibration.py`.

WHY THIS, NOT gen_engine_labeled_corpus.py: the value head we measure derives from
iter1_ema, which was trained on NEURAL-MCTS self-play positions. E24's corpus was
HeuristicAgent (negamax depth-2) self-play — a DIFFERENT position distribution, so
the 0.737 AUC partly measured out-of-distribution shift, not the head's limit. To
ask "what is iter1_ema's true discrimination on strong, clean, in-distribution
positions?" we must label the positions the net itself produces under search.

Held-out by construction: freshly generated games, disjoint from any training corpus.

Self-play diversity: a touch of root Dirichlet noise + the MCTS temperature schedule
keeps games from collapsing into one deterministic line, so the corpus spans many
strong positions rather than N copies of the same game.

Usage (box, CPU, parallel):
  python scripts/gen_selfplay_labeled_corpus.py \
      --model models/iter1_ema_2026-05-27/iter1_ema.pt \
      --out expert_games/selfplay_strong_15ch.npz \
      --games 300 --sims 400 --workers 16 --device cpu --max-positions 12000

Usage (local smoke, single process, MPS):
  python scripts/gen_selfplay_labeled_corpus.py \
      --model models/iter1_ema_2026-05-27/iter1_ema.pt \
      --out /tmp/smoke_corpus.npz --games 2 --sims 16 --max-positions 200
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase
from yinsh_ml.game.constants import Player


def make_mcts(net, sims, dirichlet_alpha):
    """A strong neural-MCTS player. Mirrors measure_h2h.make_mcts but with
    configurable budget and a little root noise for self-play diversity."""
    from yinsh_ml.training.self_play import MCTS
    return MCTS(
        network=net, evaluation_mode='pure_neural', heuristic_evaluator=None,
        heuristic_weight=0.0, num_simulations=sims, late_simulations=sims,
        simulation_switch_ply=20, c_puct=1.0, dirichlet_alpha=dirichlet_alpha,
        value_weight=1.0, max_depth=300, epsilon_mix_start=0.25 if dirichlet_alpha > 0 else 0.0,
        epsilon_mix_end=0.0, epsilon_mix_taper_moves=20, initial_temp=1.0,
        final_temp=0.1, annealing_steps=20, temp_clamp_fraction=0.6,
        enable_subtree_reuse=True, fpu_reduction=0.25,
    )


def select_move(policy, valid, encoder, temp, rng):
    """Temperature move selection over the MCTS visit-count policy."""
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


def play_and_encode(net, sims, dirichlet_alpha, seed, max_moves=400):
    """Play one neural-MCTS self-play game with `net` on both sides.

    Returns (states[N,C,11,11] float32, values[N] float32) — every MAIN_GAME
    position, encoded via the net's own (15ch) encoder, labeled by final outcome
    from that position's side-to-move POV. Encoding happens at record time, so no
    deepcopy of GameState is needed.
    """
    encoder = net.state_encoder
    mcts = make_mcts(net, sims, dirichlet_alpha)
    rng = np.random.default_rng(seed)
    state = GameState()
    enc_buf, player_buf = [], []
    move_count = 0
    while not state.is_terminal() and move_count < max_moves:
        valid = state.get_valid_moves()
        if not valid:
            break
        if state.phase == GamePhase.MAIN_GAME:
            enc_buf.append(np.asarray(encoder.encode_state(state), np.float32))
            player_buf.append(state.current_player)
        policy = mcts.search_batch(state, move_count, batch_size=32)
        temp = mcts.get_temperature(move_count)
        sel = select_move(policy, valid, encoder, temp, rng)
        if not state.make_move(sel):
            break
        mcts.advance_root(sel)
        move_count += 1

    # Outcome by score (matches measure_h2h); winner is the higher score, else draw.
    if state.white_score > state.black_score:
        winner = Player.WHITE
    elif state.black_score > state.white_score:
        winner = Player.BLACK
    else:
        winner = None

    if not enc_buf:
        return (np.empty((0, 15, 11, 11), np.float32), np.empty((0,), np.float32))
    states = np.stack(enc_buf)
    values = np.asarray(
        [0 if winner is None else (1 if p == winner else -1) for p in player_buf],
        np.float32,
    )
    return states, values


# --- parallel worker: one NetworkWrapper per process ---
_NET = None


def _init_worker(model_path, device):
    global _NET
    import torch
    torch.set_num_threads(1)  # cap each worker to 1 thread; N workers x all-cores oversubscribes
    from yinsh_ml.network.wrapper import NetworkWrapper
    _NET = NetworkWrapper(model_path=model_path, device=device)
    _NET.network.eval()


def _worker(task):
    sims, dirichlet_alpha, seed, max_moves = task
    return play_and_encode(_NET, sims, dirichlet_alpha, seed, max_moves)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="champion checkpoint (e.g. iter1_ema.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--sims", type=int, default=400,
                    help="MCTS budget per move (strong: 400+; smoke: 16)")
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3,
                    help="root noise for self-play diversity (0 = deterministic)")
    ap.add_argument("--max-moves", type=int, default=250,
                    help="hard cap per game; keeps dirichlet-noised games that never "
                         "close a 3-row win from dragging out (and stalling the pool)")
    ap.add_argument("--max-positions", type=int, default=12000,
                    help="target/subsample cap; generation STOPS once this many positions exist")
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="re-save the corpus every N completed games (crash/kill safety)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda (cpu recommended for workers>1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = None if args.device == "auto" else args.device
    if args.workers > 1 and args.device == "auto":
        device = "cpu"  # MPS/CUDA across forked workers is unsafe; CPU fan-out is the box model.
        print("workers>1: forcing device=cpu (pass --device explicitly to override)")

    tasks = [(args.sims, args.dirichlet_alpha, args.seed + i, args.max_moves)
             for i in range(args.games)]
    print(f"generating {args.games} neural self-play games @ {args.sims} sims "
          f"(dir_alpha={args.dirichlet_alpha}, workers={args.workers}, device={device or 'auto'})...")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def save_corpus(states_list, values_list, tag):
        """Concatenate, subsample to the cap, and write. Safe to call repeatedly."""
        if not states_list:
            return 0
        states = np.concatenate(states_list, axis=0)
        values = np.concatenate(values_list, axis=0)
        if len(states) > args.max_positions:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(states), size=args.max_positions, replace=False)
            states, values = states[idx], values[idx]
        np.savez_compressed(args.out, states=states, values=values)
        dec = int((values != 0).sum())
        print(f"  [checkpoint {tag}] saved {len(states)} positions "
              f"({dec} decisive, {len(states) - dec} draws) -> {args.out}", flush=True)
        return len(states)

    # imap_unordered streams results as each game finishes, so we can log progress,
    # checkpoint incrementally, and STOP EARLY once we have enough positions — none
    # of which pool.map() allowed (the original blind, all-or-nothing run).
    t0 = time.time()
    states_list, values_list, total_pos, games_done = [], [], 0, 0
    pool = None
    if args.workers > 1:
        pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(args.model, device))
        result_iter = pool.imap_unordered(_worker, tasks)
    else:
        _init_worker(args.model, device)
        result_iter = (_worker(t) for t in tasks)

    try:
        for st, vl in result_iter:
            games_done += 1
            if len(st):
                states_list.append(st)
                values_list.append(vl)
                total_pos += len(st)
            print(f"  game {games_done}/{args.games}: +{len(st)} positions "
                  f"(total {total_pos}/{args.max_positions}) | {(time.time()-t0)/60:.1f}m elapsed",
                  flush=True)
            if games_done % args.checkpoint_every == 0:
                save_corpus(states_list, values_list, f"g{games_done}")
            if total_pos >= args.max_positions:
                print(f"  reached target {args.max_positions} positions at game "
                      f"{games_done}; stopping early", flush=True)
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    n = save_corpus(states_list, values_list, "final")
    print(f"done: {n} positions from {games_done} games | {(time.time()-t0)/60:.1f}m total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
