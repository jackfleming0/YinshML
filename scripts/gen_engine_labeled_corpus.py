#!/usr/bin/env python
"""Generate an ENGINE-labeled held-out value corpus for E24 Phase 1a.

Plays HeuristicAgent (negamax) self-play — a CONSISTENT, model-INDEPENDENT
labeler — encodes each main-game position to 15ch, and labels it by the game's
final outcome from the side-to-move POV (z in {-1, 0, +1}). Saves a
`states`/`values` .npz consumable by `scripts/value_head_calibration.py`.

WHY HeuristicAgent (not the net): the value head we measure derives from
iter1_ema, so labeling with iter1_ema/MCTS would be circular. HeuristicAgent is
independent and plays *consistently*, which is the whole point — it removes the
human-blunder noise that inflates the human-corpus AUC floor (~0.70). For an even
cleaner corpus, relabel later with a stronger external engine (yngine).

Held-out by construction: these are freshly generated games, disjoint from any
training corpus.

Usage (on the box; needs torch for the 15ch encoder):
  python scripts/gen_engine_labeled_corpus.py \
      --out expert_games/engine_labeled_15ch.npz \
      --games 300 --depth 2 --epsilon 0.12 --workers 80 --max-positions 12000

The play+label logic is torch-free and unit-tested; only the encode step (and
thus the real run) needs torch.
"""
import argparse
import copy
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
import validate_weights as vw  # torch-free; _make_agent builds a full-game heuristic engine

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase


def _z(player, winner) -> int:
    """Outcome from this position's side-to-move POV."""
    if winner is None:
        return 0
    return 1 if player == winner else -1


def play_and_label(base_weights, depth, eps, seed, max_moves=400):
    """Play one HeuristicAgent game; return [(GameState, z)] for MAIN_GAME positions.

    Torch-free. epsilon-greedy random moves inject position diversity so the corpus
    isn't hundreds of near-identical deterministic games.
    """
    rng = random.Random(seed)
    engine = vw._make_agent(base_weights, depth, seed)
    state = GameState()
    recs = []  # (deepcopy(state), side-to-move player)
    for _ in range(max_moves):
        if state.phase == GamePhase.GAME_OVER or state.is_stalemate():
            break
        if state.phase == GamePhase.MAIN_GAME:
            recs.append((copy.deepcopy(state), state.current_player))
        moves = state.get_valid_moves()
        if not moves:
            break
        move = rng.choice(moves) if rng.random() < eps else engine.select_move(state)
        if move is None or not state.make_move(move):
            break
        if state.white_score >= 3 or state.black_score >= 3:
            break
    winner = state.get_winner()
    return [(s, _z(player, winner)) for s, player in recs]


def encode_labeled(labeled, encoder):
    """Encode [(GameState, z)] -> (states[N,15,11,11] float32, values[N] float32).

    `encoder` is injected (EnhancedStateEncoder on the box; a stub in tests).
    """
    if not labeled:
        return (np.empty((0, 15, 11, 11), np.float32), np.empty((0,), np.float32))
    states = np.stack([np.asarray(encoder.encode_state(s), np.float32) for s, _ in labeled])
    values = np.asarray([z for _, z in labeled], np.float32)
    return states, values


# --- parallel worker: one EnhancedStateEncoder per process (torch, lazy) ---
_ENC = None


def _init_worker():
    global _ENC
    from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder
    _ENC = EnhancedStateEncoder()


def _worker(task):
    base_weights, depth, eps, seed, max_moves = task
    return encode_labeled(play_and_label(base_weights, depth, eps, seed, max_moves), _ENC)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-weights", default="configs/heuristic_weights/baseline.json",
                    help="heuristic weights for the labeling engine (the standard engine)")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--depth", type=int, default=2,
                    help="negamax depth of the labeling engine (2 = good/consistent; 3 = stronger, slower)")
    ap.add_argument("--epsilon", type=float, default=0.12, help="exploration for position diversity")
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--max-positions", type=int, default=12000, help="subsample cap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    tasks = [(args.base_weights, args.depth, args.epsilon, args.seed + i, args.max_moves)
             for i in range(args.games)]
    print(f"generating {args.games} depth-{args.depth} engine games "
          f"(eps={args.epsilon}, workers={args.workers})...")
    if args.workers > 1:
        with mp.Pool(args.workers, initializer=_init_worker) as pool:
            chunks = pool.map(_worker, tasks)
    else:
        _init_worker()
        chunks = [_worker(t) for t in tasks]

    states = np.concatenate([c[0] for c in chunks if len(c[0])], axis=0)
    values = np.concatenate([c[1] for c in chunks if len(c[1])], axis=0)

    if len(states) > args.max_positions:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(states), size=args.max_positions, replace=False)
        states, values = states[idx], values[idx]

    dec = int((values != 0).sum())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, states=states, values=values)
    print(f"wrote {args.out}: {len(states)} positions "
          f"({dec} decisive, {len(states) - dec} draws), states{states.shape} {states.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
