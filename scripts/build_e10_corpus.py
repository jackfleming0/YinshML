"""E10 — build a placement-diversified 15-channel pretraining corpus.

Assembles a from-scratch supervised corpus that EXPOSES multiple opening styles
so self-play (with a value head grounded by E2) can fairly arbitrate which wins,
instead of the corpus pre-deciding by omission:

  placement positions = mix(human / yngine-engine / uniform-random), each
                        expanded by the full 4x D2 augmentation (symmetry signal,
                        synergy with E16 — reuses the E16 move permutation so the
                        corpus and the regularizer agree exactly)
  main-game positions  = the engine (yngine) corpus, used directly

Why these choices (see SYMMETRY_FIXES_RUNBOOK.md / EXPERIMENT_BACKLOG E10):
- Engine (yngine) is a NON-DOMINANT but solid slice (default 35%): we don't know
  the engine's wall-clustering is bad — excluding it would pre-judge a possible
  edge. Capping it stops the degenerate over-concentration.
- Random placements are state-coverage / an entropy prior. They carry value 0
  (an early random opening is ~neutral) and the random move as their index — a
  weak imitation signal by design, mainly there to broaden coverage. Set
  --random-frac 0 to drop them.

Output is a mmap directory (states.npy / policy_indices.npy / values.npy /
total_moves.npy) directly consumable by
`run_supervised_pretraining.py --data-dir <out>`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from yinsh_ml.utils.encoding import StateEncoder, decode_phase_from_state
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.training.augmentation import YinshSymmetryAugmenter
from yinsh_ml.training import symmetric_reg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--engine-corpus', required=True,
                   help='15ch yngine corpus: a .npz or an mmap dir (states/policy_indices/values).')
    p.add_argument('--human-placements', default='expert_games/hvh_placement_only_15ch.npz')
    p.add_argument('--output', required=True, help='output mmap dir')
    p.add_argument('--human-frac', type=float, default=0.40)
    p.add_argument('--engine-frac', type=float, default=0.35)
    p.add_argument('--random-frac', type=float, default=0.25)
    p.add_argument('--no-augment', dest='augment', action='store_false', default=True,
                   help='disable the 4x D2 augmentation of placement positions')
    p.add_argument('--max-main-game', type=int, default=None, help='cap main-game positions')
    p.add_argument('--max-placement', type=int, default=None,
                   help='target total placement positions BEFORE augmentation (default: as many as the mix allows)')
    p.add_argument('--ram-budget-gb', type=float, default=40.0,
                   help='refuse to build a corpus needing more than this in RAM (guards OOM; raise --max-main-game to fit)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _load(path):
    """Load (states, policy_indices, values) from a .npz or mmap dir."""
    path = Path(path)
    if path.is_dir():
        states = np.load(path / 'states.npy', mmap_mode='r')
        pol = np.load(path / 'policy_indices.npy', mmap_mode='r')
        vals = np.load(path / 'values.npy', mmap_mode='r')
        return states, pol, vals
    d = np.load(path, mmap_mode='r')
    pol = d['policy_indices'] if 'policy_indices' in d.files else d['policies']
    if pol.ndim > 1:
        raise ValueError(f"{path} has soft policy targets; this builder needs hard policy_indices")
    return d['states'], pol, d['values']


def _placement_mask(states, enc):
    """Boolean mask of RING_PLACEMENT positions, read from the phase channel
    (vectorized — no per-position python decode)."""
    from yinsh_ml.utils.encoding import phase_channel_index, _PHASE_READ_ROW, _PHASE_READ_COL
    ch = phase_channel_index(states.shape[1])
    phase_vals = np.asarray(states[:, ch, _PHASE_READ_ROW, _PHASE_READ_COL])
    # RING_PLACEMENT encodes to phase_idx 0 → value 0.0; next phase is 1/(n-1).
    # Threshold at half the first step to classify placement robustly.
    from yinsh_ml.game.types import GamePhase
    step = 1.0 / (len(GamePhase) - 1)
    return phase_vals < (step * 0.5)


def _generate_random_placements(n, enc, rng):
    """n random placement tuples: (state, random_move_index, value=0)."""
    states, idxs = [], []
    n_pos = enc.position_to_index
    while len(states) < n:
        gs = GameState()
        # random number of rings already placed (1..9), then one more random ring
        k = int(rng.integers(0, 9))
        ok = True
        for _ in range(k):
            valid = gs.get_valid_moves()
            placements = [m for m in valid if m.type == MoveType.PLACE_RING]
            if not placements:
                ok = False
                break
            gs.make_move(placements[int(rng.integers(0, len(placements)))])
        if not ok:
            continue
        valid = gs.get_valid_moves()
        placements = [m for m in valid if m.type == MoveType.PLACE_RING]
        if not placements:
            continue
        mv = placements[int(rng.integers(0, len(placements)))]
        states.append(enc.encode_state(gs).astype(np.float32))
        idxs.append(enc.move_to_index(mv))
    return np.stack(states), np.array(idxs, dtype=np.int64), np.zeros(n, dtype=np.float32)


def _sample(states, pol, vals, k, rng):
    """Take k rows (without replacement if possible) from a (states,pol,vals) set."""
    n = states.shape[0]
    if k >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=k, replace=False)
    idx = np.sort(idx)  # sorted → mmap-friendly fancy indexing
    return np.asarray(states[idx]).astype(np.float32), np.asarray(pol[idx]).astype(np.int64), np.asarray(vals[idx]).astype(np.float32)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    enc = EnhancedStateEncoder()
    basic = StateEncoder()
    aug = YinshSymmetryAugmenter(include_reflections=True, state_encoder=basic)
    # E16 move permutations: perm[tid][old_idx] = new_idx (reused so corpus &
    # regularizer use identical geometry). tid 0 is identity.
    perms = [np.arange(enc.total_moves, dtype=np.int64)] + [
        symmetric_reg.build_full_policy_permutation(aug, enc, tid) for tid in (1, 2, 3)
    ]

    print('Loading engine corpus...')
    e_states, e_pol, e_vals = _load(args.engine_corpus)
    assert e_states.shape[1] == 15, f"engine corpus must be 15ch, got {e_states.shape[1]}"
    print(f'  {e_states.shape[0]:,} engine positions')

    pmask = _placement_mask(e_states, enc)
    eng_place_idx = np.nonzero(pmask)[0]
    eng_main_idx = np.nonzero(~pmask)[0]
    print(f'  engine split: {eng_place_idx.size:,} placement / {eng_main_idx.size:,} main-game')

    # --- placement mix ---
    h_states, h_pol, h_vals = _load(args.human_placements)
    print(f'  {h_states.shape[0]:,} human placements')

    # Decide placement budget: scale to the smallest binding source given fracs.
    fr = {'human': args.human_frac, 'engine': args.engine_frac, 'random': args.random_frac}
    avail = {'human': h_states.shape[0], 'engine': eng_place_idx.size,
             'random': 10 ** 9}  # random is generated on demand
    budget = args.max_placement or min(
        int(avail[s] / fr[s]) for s in fr if fr[s] > 0 and s != 'random'
    )
    counts = {s: int(round(budget * fr[s])) for s in fr}
    print(f'  placement budget={budget:,} → human={counts["human"]:,} '
          f'engine={counts["engine"]:,} random={counts["random"]:,}')

    parts_s, parts_p, parts_v = [], [], []
    hs, hp, hv = _sample(h_states, h_pol, h_vals, counts['human'], rng)
    parts_s.append(hs); parts_p.append(hp); parts_v.append(hv)
    es, ep, ev = _sample(e_states[eng_place_idx], e_pol[eng_place_idx], e_vals[eng_place_idx],
                         counts['engine'], rng) if counts['engine'] else (None, None, None)
    if es is not None:
        parts_s.append(es); parts_p.append(ep); parts_v.append(ev)
    if counts['random'] > 0:
        rs, rp, rv = _generate_random_placements(counts['random'], enc, rng)
        parts_s.append(rs); parts_p.append(rp); parts_v.append(rv)

    place_s = np.concatenate(parts_s); place_p = np.concatenate(parts_p); place_v = np.concatenate(parts_v)
    print(f'  placement mix (pre-aug): {place_s.shape[0]:,}')

    # --- 4x D2 augmentation of the placement mix ---
    if args.augment:
        aug_s, aug_p, aug_v = [place_s], [place_p], [place_v]
        for tid in (1, 2, 3):
            ts = np.stack([aug._transform_state(s, tid) for s in place_s]).astype(np.float32)
            tp = perms[tid][place_p]
            aug_s.append(ts); aug_p.append(tp); aug_v.append(place_v.copy())
        place_s = np.concatenate(aug_s); place_p = np.concatenate(aug_p); place_v = np.concatenate(aug_v)
        print(f'  placement after 4x D2 aug: {place_s.shape[0]:,}')

    # --- main-game (engine), optionally capped ---
    mi = eng_main_idx
    if args.max_main_game and args.max_main_game < mi.size:
        mi = np.sort(rng.choice(mi, size=args.max_main_game, replace=False))
    # RAM guard: the assembled corpus is held in memory before the mmap write.
    # 15ch float32 ≈ 7.26 KB/position. Refuse to silently OOM the box.
    est_gb = (place_s.shape[0] + mi.size) * 15 * 121 * 4 / 1e9
    if est_gb > args.ram_budget_gb:
        raise SystemExit(
            f"Corpus would need ~{est_gb:.0f} GB in RAM (> --ram-budget-gb "
            f"{args.ram_budget_gb}). Pass --max-main-game to cap main-game "
            f"positions (e.g. --max-main-game {int(args.ram_budget_gb * 1e9 / (15*121*4) - place_s.shape[0]):,})."
        )
    main_s = np.asarray(e_states[mi]).astype(np.float32)
    main_p = np.asarray(e_pol[mi]).astype(np.int64)
    main_v = np.asarray(e_vals[mi]).astype(np.float32)
    print(f'  main-game: {main_s.shape[0]:,}')

    # --- combine + shuffle ---
    all_s = np.concatenate([place_s, main_s])
    all_p = np.concatenate([place_p, main_p])
    all_v = np.concatenate([place_v, main_v])
    order = rng.permutation(all_s.shape[0])
    all_s, all_p, all_v = all_s[order], all_p[order], all_v[order]

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'states.npy', all_s)
    np.save(out / 'policy_indices.npy', all_p)
    np.save(out / 'values.npy', all_v)
    np.save(out / 'total_moves.npy', np.array(enc.total_moves, dtype=np.int64))
    meta = {'total': int(all_s.shape[0]), 'placement': int(place_s.shape[0]),
            'main_game': int(main_s.shape[0]), 'fracs': fr, 'augmented': args.augment,
            'channels': 15, 'total_moves': int(enc.total_moves)}
    (out / 'NOTES.md').write_text(
        '# E10 placement-diversified corpus\n\n```\n' + json.dumps(meta, indent=2) + '\n```\n')
    print(f'\nWrote {all_s.shape[0]:,} positions to {out}/  (consume with '
          f'run_supervised_pretraining.py --data-dir {out})')


if __name__ == '__main__':
    main()
