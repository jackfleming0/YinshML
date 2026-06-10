#!/usr/bin/env python3
"""Engine-label HUMAN positions into the e26 teacher format, for the
"watch the engine crush a human position" puzzle pool.

The human-game corpora (e.g. ``hvh_full_game_15ch.npz``) only store the human
move played + outcome — no per-move search distribution. The puzzle curator
(``curate_puzzles.py``) needs the same labels the e26 self-play teacher has:
``policy_idx / policy_prob / policy_q / policy_prior / values``. This script
produces them by running the *same* high-budget teacher MCTS over a sample of
real human positions, so the curator can run unchanged.

Output npz matches ``e26_teacher_*sim.npz`` exactly, so:
    python scripts/puzzles/curate_puzzles.py --npz <this output> --out <bank>

reuses the entire find_win/hold/sharpness/trap pipeline. The resulting puzzles
are real positions humans steered into, with the engine's (often unintuitive)
best move as the answer — and Task-1's reveal-line plays out the crush.

Usage:
    python scripts/puzzles/label_human_positions.py \
        --input expert_games/hvh_full_game_15ch.npz \
        --model models/iter1_ema_2026-05-27/iter1_ema.pt \
        --out expert_games/human_labeled_800sim.npz \
        --sims 800 --max-positions 4000 --device auto
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yinsh_ml.game.types import GamePhase
from scripts.gen_distill_corpus import make_mcts, _topk, TOPK


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="expert_games/hvh_full_game_15ch.npz",
                   help="Human-game npz with a 'states' array (15ch).")
    p.add_argument("--model", required=True, help="Network checkpoint to label with.")
    p.add_argument("--out", required=True)
    p.add_argument("--sims", type=int, default=800, help="Teacher search budget per position.")
    p.add_argument("--max-positions", type=int, default=4000,
                   help="How many MAIN_GAME human positions to label (sampled).")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    import logging
    logging.getLogger("MCTS").setLevel(logging.WARNING)  # silence per-position config spam
    rng = np.random.default_rng(args.seed)
    device = None if args.device == "auto" else args.device

    print(f"Loading human positions from {args.input} ...")
    d = np.load(args.input, allow_pickle=True)
    states = d["states"]
    # The actual move the human played at each position (single index). Carried
    # through so the puzzle reveal can show "human played X, engine crushes with
    # Y" — the gap. Absent => -1 (reveal just omits the human move).
    human_moves = d["policy_indices"] if "policy_indices" in d else np.full(states.shape[0], -1, np.int32)
    n_total = states.shape[0]
    print(f"  {n_total:,} positions, {states.shape[1]}ch")

    # Filter to MAIN_GAME (phase channel 12, broadcast scalar -> phase idx).
    # Placement / capture-substep positions aren't puzzle material.
    ch12 = np.median(states[:, 12].reshape(n_total, -1), axis=1)
    phase_idx = np.rint(ch12 * (len(GamePhase) - 1)).astype(int)
    main_game = np.where(phase_idx == GamePhase.MAIN_GAME.value)[0]
    print(f"  {len(main_game):,} MAIN_GAME positions")

    take = min(args.max_positions, len(main_game))
    sample = rng.choice(main_game, take, replace=False)
    print(f"  labeling {take:,} sampled positions @ {args.sims} sims ...")

    import torch
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    from yinsh_ml.network.wrapper import NetworkWrapper
    net = NetworkWrapper(model_path=args.model, device=device)
    net.network.eval()
    encoder = net.state_encoder

    S, PI, PP, PQ, PPR, V, HM = [], [], [], [], [], [], []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(tag):
        if not S:
            return
        np.savez_compressed(
            args.out,
            states=np.stack(S), policy_idx=np.stack(PI), policy_prob=np.stack(PP),
            policy_q=np.stack(PQ), policy_prior=np.stack(PPR), values=np.asarray(V, np.float32),
            human_move_idx=np.asarray(HM, np.int32),
        )
        print(f"  [{tag}] saved {len(S):,} labeled positions -> {args.out}", flush=True)

    t0 = time.time()
    done = 0
    for gi in sample:
        state15 = states[gi]
        gs = encoder.decode_state(state15)
        if gs.phase != GamePhase.MAIN_GAME:
            continue
        valid = gs.get_valid_moves()
        if not valid:
            continue
        # Fresh MCTS per position — these are independent, no subtree carryover.
        # dirichlet_alpha=0 => clean search policy (no root exploration noise).
        mcts = make_mcts(net, args.sims, dirichlet_alpha=0.0)
        policy = mcts.search_batch(gs, 30, batch_size=args.batch_size)
        ti, tp = _topk(np.asarray(policy, np.float32))
        cstats = mcts.root_child_stats()   # {idx: (q, prior)}
        tq = np.zeros(TOPK, np.float32); tpr = np.zeros(TOPK, np.float32)
        for j, idx in enumerate(ti):
            if idx >= 0 and idx in cstats:
                tq[j], tpr[j] = cstats[idx]
        # Store the re-encoded state so it round-trips through curation's
        # decode_state identically to the e26 corpus.
        S.append(np.asarray(encoder.encode_state(gs), np.float32))
        PI.append(ti); PP.append(tp); PQ.append(tq); PPR.append(tpr)
        V.append(np.float32(getattr(mcts, "last_root_value", 0.0)))
        HM.append(np.int32(human_moves[gi]))
        done += 1
        if done % args.checkpoint_every == 0:
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{take:,}  ({rate:.1f} pos/s, ~{(take-done)/max(rate,1e-9)/60:.0f} min left)", flush=True)
            checkpoint(f"ckpt {done}")

    checkpoint("final")
    print(f"Done: {len(S):,} positions in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
