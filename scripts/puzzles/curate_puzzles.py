#!/usr/bin/env python3
"""Curate a YINSH puzzle bank from a teacher self-play corpus (.npz).

Turns the e26 teacher distillation data (states + per-move visit/q/prior + value)
into a balanced, inspectable bank of single-move "tactics" puzzles for the
analysis board.

Design (see session plan):
  * Every corpus position is already MAIN_GAME (placement excluded upstream).
  * A puzzle needs a *decision*: drop "forced" positions where one move owns the
    visits, and require a punishing gap between the accepted moves and the rest.
  * Two pools:
      - find_win : there is a clearly strong move (best_q >= --win-q).
      - hold     : you are under pressure (best_q below that), but there is still
                   one clearly-best move and a cliff down to the alternatives.
  * Acceptance honours Jack's "any of the top 5" rule: accept the <=5 highest-visit
    moves that are within --accept-delta of the best q (and q>0 for find_win).
  * "Sharpness" = the cliff between the worst accepted move and the best rejected
    move. "Trap strength" = the highest network prior among clearly-bad moves
    (a move the net itself wants to play but that loses) -> drives difficulty.
  * Final set is stratified across {game stage x score x centrality x difficulty}
    so the bank is not dominated by one regime.

Scoring/filtering is fully vectorised over the whole corpus; only the final
sampled positions are reconstructed (decode_state) and serialised.

Usage:
    python scripts/puzzles/curate_puzzles.py \
        --npz expert_games/e26_teacher_800sim_full.npz \
        --out data/puzzles/puzzle_bank.json \
        --target-per-pool 400
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder
from yinsh_ml.game.constants import Player


# ---------------------------------------------------------------------------
# Channel layout (EnhancedStateEncoder, 15ch)
CH_CUR_RINGS, CH_OPP_RINGS = 0, 1
CH_CUR_MARK, CH_OPP_MARK = 2, 3
CH_CENTER_DIST = 9
CH_SCORE_DIFF = 14


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default="expert_games/e26_teacher_800sim_full.npz")
    p.add_argument("--out", default="analysis_board/data/puzzle_bank.json",
                   help="Bank output. Default lives under analysis_board/ (tracked) so "
                        "`yinsh-redeploy`'s git pull ships it; data/ is gitignored.")
    p.add_argument("--target-per-pool", type=int, default=400,
                   help="Approx number of puzzles to keep per pool after stratification.")
    p.add_argument("--seed", type=int, default=0)

    # --- decision / quality thresholds (tunable) ---
    p.add_argument("--forced-thresh", type=float, default=0.95,
                   help="Drop positions where the top move's visit share exceeds this.")
    p.add_argument("--min-legal", type=int, default=6,
                   help="Need at least this many legal moves (room for distractors).")
    p.add_argument("--win-q", type=float, default=0.25,
                   help="best_q >= this -> 'find_win' pool; below -> 'hold' candidate.")
    p.add_argument("--hold-q-floor", type=float, default=-0.55,
                   help="hold pool: ignore near-lost positions (best_q below this).")
    p.add_argument("--accept-delta", type=float, default=0.10,
                   help="A move is 'accepted' if its q is within this of best_q (cap 5, by visits).")
    p.add_argument("--min-cliff", type=float, default=0.12,
                   help="Require accept_floor - best_reject_q >= this (puzzle actually punishes a wrong move).")
    p.add_argument("--trap-gap", type=float, default=0.15,
                   help="A 'trap' is a rejected move with q <= best_q - this gap.")
    p.add_argument("--max-distractors", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Debug: only load first N positions.")

    # --- human-origin pool ("watch the engine crush a human position") ---
    p.add_argument("--source-tag", default="selfplay",
                   help="Stamp each puzzle's origin: 'selfplay' (e26) or 'human'.")
    p.add_argument("--require-divergence", action="store_true",
                   help="Human pool only: keep positions where the human's actual move "
                        "is NOT in the engine's accept set. The obvious human move is the "
                        "foil; the engine's deeper move is the answer — that contrast is "
                        "the teaching. Needs a 'human_move_idx' array in the npz.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz, allow_pickle=True)
    S = d["states"]
    idx = d["policy_idx"].astype(np.int32)
    prob = d["policy_prob"].astype(np.float64)
    q = d["policy_q"].astype(np.float64)
    prior = d["policy_prior"].astype(np.float64)
    values = d["values"].astype(np.float64)
    # The human's actual move per position (same encoder index space as policy_idx),
    # present only in human-labeled corpora. -1 where unknown.
    human_move_idx = d["human_move_idx"].astype(np.int32) if "human_move_idx" in d.files \
        else np.full(S.shape[0], -1, np.int32)
    if args.limit:
        S, idx, prob, q, prior, values, human_move_idx = (
            a[:args.limit] for a in (S, idx, prob, q, prior, values, human_move_idx))
    N = S.shape[0]
    print(f"  {N:,} positions, {S.shape[1]}ch states  (source={args.source_tag})")

    # -----------------------------------------------------------------------
    # Stage A: position metadata (vectorised, straight off the state tensor)
    markers = (S[:, CH_CUR_MARK] + S[:, CH_OPP_MARK]).reshape(N, -1).sum(1)
    score_diff = np.rint(np.median(S[:, CH_SCORE_DIFF].reshape(N, -1), axis=1) * 3.0).astype(int)
    rings = (S[:, CH_CUR_RINGS] + S[:, CH_OPP_RINGS]).reshape(N, -1)
    cdist = S[:, CH_CENTER_DIST].reshape(N, -1)
    ring_count = np.maximum(rings.sum(1), 1)
    centrality = (rings * cdist).sum(1) / ring_count  # 1=central rings, 0=edge rings

    # -----------------------------------------------------------------------
    # Stage B: per-move decision scoring (rank slots by visit share)
    valid = idx >= 0
    order = np.argsort(-prob, axis=1, kind="stable")

    def take(a):
        return np.take_along_axis(a, order, axis=1)

    P, Q, PR, ID, VAL = take(prob), take(q), take(prior), take(idx), take(valid)
    Qm = np.where(VAL, Q, -np.inf)              # masked q for max/argmax
    p0 = P[:, 0]
    nlegal = VAL.sum(1)
    best_q = Qm.max(1)

    near = Q >= (best_q[:, None] - args.accept_delta)
    top5 = np.zeros_like(VAL)
    top5[:, :5] = VAL[:, :5]

    def build_accept(require_positive):
        m = VAL & top5 & near
        if require_positive:
            m = m & (Q > 0.0)
        return m

    def cliff_and_traps(accept_mask):
        accept_floor = np.where(accept_mask, Q, np.inf).min(1)          # inf if no accept
        reject = VAL & ~accept_mask
        best_reject = np.where(reject, Q, -np.inf).max(1)               # -inf if none
        cliff = accept_floor - best_reject
        # trap strength: highest prior among clearly-bad rejected moves
        bad = reject & (Q <= (best_q[:, None] - args.trap_gap))
        trap_strength = np.where(bad, PR, 0.0).max(1)
        n_accept = accept_mask.sum(1)
        return accept_floor, best_reject, cliff, trap_strength, n_accept

    # Pool assignment ------------------------------------------------------
    not_forced = p0 <= args.forced_thresh
    enough = nlegal >= args.min_legal
    base = not_forced & enough

    is_win = base & (best_q >= args.win_q)
    is_hold = base & (best_q < args.win_q) & (best_q >= args.hold_q_floor)

    pools = {}
    for name, sel, require_pos in (("find_win", is_win, True), ("hold", is_hold, False)):
        acc = build_accept(require_pos)
        af, br, cliff, trap, n_acc = cliff_and_traps(acc)
        worthy = sel & (n_acc >= 1) & (cliff >= args.min_cliff)
        # Teaching contrast: the human's obvious move ≠ the engine's accept set.
        # acc is in ranked-slot order, so are ID; check membership of the human
        # index among the accepted slots' move indices.
        acc_move_idx = np.where(acc, ID, -1)                       # -1 for non-accepted slots
        human_in_accept = (acc_move_idx == human_move_idx[:, None]).any(1)
        diverges = (~human_in_accept) & (human_move_idx >= 0)
        if args.require_divergence:
            worthy = worthy & diverges
        pools[name] = dict(sel=worthy, accept=acc, accept_floor=af, best_reject=br,
                           cliff=cliff, trap=trap, n_accept=n_acc, diverges=diverges)
        extra = f"  divergent={int((worthy & diverges).sum()):,}" if (human_move_idx >= 0).any() else ""
        print(f"\n[{name}] candidates: pool={int(sel.sum()):,}  puzzle-worthy={int(worthy.sum()):,}{extra}")

    # -----------------------------------------------------------------------
    # Stage C: stratified sampling within each pool
    def stage_bucket(m):
        return np.select([m < 12, m < 20, m < 28], ["early", "mid", "late"], default="end")

    stage = stage_bucket(markers)
    score_b = np.clip(np.abs(score_diff), 0, 2)

    encoder = EnhancedStateEncoder()
    out_puzzles = []

    for name, info in pools.items():
        sel_idx = np.where(info["sel"])[0]
        if len(sel_idx) == 0:
            print(f"[{name}] no puzzle-worthy positions; skipping")
            continue
        # difficulty tertiles from a combined hardness score (within this pool)
        hard = info["trap"][sel_idx] + (1.0 - np.clip(info["cliff"][sel_idx], 0, 1))
        diff_lab = np.array(["easy", "med", "hard"])[
            np.clip(np.digitize(hard, np.quantile(hard, [1/3, 2/3])), 0, 2)]
        cent = centrality[sel_idx]
        cent_lab = np.array(["c_lo", "c_mid", "c_hi"])[
            np.clip(np.digitize(cent, np.quantile(cent, [1/3, 2/3])), 0, 2)]

        cells = defaultdict(list)
        for k, gi in enumerate(sel_idx):
            key = (stage[gi], int(score_b[gi]), cent_lab[k], diff_lab[k])
            cells[key].append((gi, diff_lab[k]))
        for v in cells.values():
            rng.shuffle(v)

        # round-robin across cells until target reached
        keys = list(cells.keys())
        rng.shuffle(keys)
        picked = []
        target = args.target_per_pool
        cursor = {k: 0 for k in keys}
        while len(picked) < target and keys:
            progressed = False
            for k in list(keys):
                if cursor[k] < len(cells[k]):
                    picked.append(cells[k][cursor[k]])
                    cursor[k] += 1
                    progressed = True
                    if len(picked) >= target:
                        break
            if not progressed:
                break
        print(f"[{name}] sampled {len(picked)} from {len(cells)} strata")

        # Stage D: reconstruct + serialise picked positions.
        # All per-move arrays (ID, P, Q, PR, VAL, accept) are in visit-rank order.
        for gi, dlab in picked:
            out_puzzles.append(_serialise_puzzle(
                encoder, name, int(gi), dlab,
                state=S[gi], ID_row=ID[gi], P_row=P[gi], Q_row=Q[gi], PR_row=PR[gi],
                VAL_row=VAL[gi], accept_ranked=info["accept"][gi],
                markers=markers[gi], score_diff=score_diff[gi], centrality=centrality[gi],
                value=values[gi], best_q=best_q[gi], accept_floor=info["accept_floor"][gi],
                cliff=info["cliff"][gi], trap=info["trap"][gi],
                max_distractors=args.max_distractors,
                source_tag=args.source_tag, human_move_idx=int(human_move_idx[gi]),
                diverges=bool(info["diverges"][gi])))

    # -----------------------------------------------------------------------
    out = {
        "schema": "yinsh-puzzle-bank/v1",
        "source": args.npz,
        "n_puzzles": len(out_puzzles),
        "params": {k: getattr(args, k) for k in
                   ("forced_thresh", "min_legal", "win_q", "hold_q_floor",
                    "accept_delta", "min_cliff", "trap_gap", "target_per_pool", "seed")},
        "puzzles": out_puzzles,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote {len(out_puzzles)} puzzles -> {args.out}")
    _report(out_puzzles)


def _serialise_puzzle(encoder, pool, gi, diff_lab, *, state, ID_row, P_row, Q_row, PR_row,
                      VAL_row, accept_ranked, markers, score_diff, centrality, value,
                      best_q, accept_floor, cliff, trap, max_distractors,
                      source_tag="selfplay", human_move_idx=-1, diverges=False):
    gs = encoder.decode_state(state)
    side = gs.current_player                       # true side to move (decoded from sentinel)
    pieces = [{"pos": str(pos), "piece": piece.name} for pos, piece in gs.board.pieces.items()]
    # A ring leaves the board only by scoring, so score = 5 - rings on board (verified exact).
    white_rings = sum(1 for pc in gs.board.pieces.values() if pc.name == "WHITE_RING")
    black_rings = sum(1 for pc in gs.board.pieces.values() if pc.name == "BLACK_RING")
    scores = {"WHITE": 5 - white_rings, "BLACK": 5 - black_rings}

    def move_dict(rank):
        mv = encoder.index_to_move(int(ID_row[rank]), side)
        return {
            "type": mv.type.name,
            "source": str(mv.source),
            "destination": str(mv.destination) if mv.destination is not None else None,
            "visits_share": round(float(P_row[rank]), 4),
            "q": round(float(Q_row[rank]), 4),
            "prior": round(float(PR_row[rank]), 4),
        }

    accept_ranks = [r for r in range(len(VAL_row)) if VAL_row[r] and accept_ranked[r]]
    accept_moves = [move_dict(r) for r in accept_ranks]

    # distractors: clearly-bad rejected moves, highest prior first (the tempting traps)
    reject_ranks = [r for r in range(len(VAL_row))
                    if VAL_row[r] and not accept_ranked[r] and Q_row[r] <= best_q - 0.15]
    reject_ranks.sort(key=lambda r: -PR_row[r])
    distractors = [move_dict(r) for r in reject_ranks[:max_distractors]]

    # The human's actual move (the "obvious" foil) — decoded for the reveal so
    # the UI can show "a human played X; the engine's deeper move is Y" and the
    # line makes the why visible. None for self-play puzzles.
    human_move = None
    if human_move_idx is not None and human_move_idx >= 0:
        try:
            hmv = encoder.index_to_move(int(human_move_idx), side)
            human_move = {
                "type": hmv.type.name,
                "source": str(hmv.source),
                "destination": str(hmv.destination) if hmv.destination is not None else None,
            }
        except Exception:  # noqa: BLE001 — out-of-range/odd index, just omit
            human_move = None

    return {
        "id": f"{source_tag[:1]}{'fw' if pool=='find_win' else 'hl'}_{gi:06d}",
        "pool": pool,
        "source": source_tag,
        "source_index": gi,
        "difficulty": diff_lab,
        "side_to_move": side.name,
        "phase": "MAIN_GAME",
        "scores": scores,
        "pieces": pieces,
        "accept_moves": accept_moves,
        "distractors": distractors,
        "human_move": human_move,
        "human_diverges": bool(diverges),
        "metrics": {
            "best_q": round(float(best_q), 4),
            "accept_floor": round(float(accept_floor), 4),
            "cliff": round(float(cliff), 4),
            "trap_strength": round(float(trap), 4),
        },
        "tags": {
            "markers": int(markers),
            "score_diff": int(score_diff),
            "centrality": round(float(centrality), 3),
            "position_value": round(float(value), 4),
        },
    }


def _report(puzzles):
    print("\n=== puzzle bank report ===")
    by_pool = Counter(p["pool"] for p in puzzles)
    print("pool:      ", dict(by_pool))
    print("difficulty:", dict(Counter(p["difficulty"] for p in puzzles)))
    stages = Counter()
    for p in puzzles:
        m = p["tags"]["markers"]
        stages["early" if m < 12 else "mid" if m < 20 else "late" if m < 28 else "end"] += 1
    print("stage:     ", dict(stages))
    print("score_diff:", dict(Counter(p["tags"]["score_diff"] for p in puzzles)))
    n_acc = [len(p["accept_moves"]) for p in puzzles]
    n_dis = [len(p["distractors"]) for p in puzzles]
    print(f"accept moves/puzzle: mean={np.mean(n_acc):.2f} (min {min(n_acc)}, max {max(n_acc)})")
    print(f"distractors/puzzle:  mean={np.mean(n_dis):.2f}")


if __name__ == "__main__":
    main()
