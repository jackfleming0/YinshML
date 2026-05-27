"""Run raw-policy + multi-budget MCTS on a JSONL of positions.

For each position, computes:
- raw policy entropy + top-K probabilities
- raw policy value
- raw policy best-move-value (network value after playing the policy argmax)
- For each requested sim budget: top-K visit distribution, root-Q value,
  best-move-value, and ``rank_of_final_best`` — where the MCTS-chosen move
  ranks in the *raw* policy (the headline misalignment metric).

Output: a JSONL with one record per position, ready for report.py.

Usage:
    python analysis_board/loop/measure.py \
        --positions analysis_board/loop/runs/<ts>/positions.jsonl \
        --model-id yngine_volume_15ch_pretrain/best_supervised.pt \
        --sims 0,100,400,1200,3200 \
        --out analysis_board/loop/runs/<ts>/measurements.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the analysis-board server's machinery — same model registry, same
# state-construction code, same MCTS init. Avoids drift across loop + UI.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "server_mod", ROOT / "analysis_board" / "server.py"
)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)
server._models = server.discover_models()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("measure")


def _policy_entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats — over only the non-zero entries."""
    p = probs[probs > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def measure_one(
    payload: Dict[str, Any],
    model_id: str,
    sim_budgets: List[int],
    top_k: int = 8,
) -> Dict[str, Any]:
    """Run all budgets on one position; return a flat record."""
    # Build state via server.build_state (same code path the UI uses).
    payload_for_eval = {
        "model_id": model_id,
        "pieces": payload["pieces"],
        "phase": payload["phase"],
        "side_to_move": payload["side_to_move"],
        "scores": payload["scores"],
    }
    try:
        gs = server.build_state(payload_for_eval)
        valid_moves = gs.get_valid_moves()
    except Exception as e:  # noqa: BLE001
        return {
            "id": payload.get("id"),
            "ok": False,
            "error": str(e),
            "meta": payload.get("meta", {}),
        }
    if not valid_moves:
        return {
            "id": payload.get("id"),
            "ok": False,
            "error": "no legal moves",
            "meta": payload.get("meta", {}),
        }

    encoder = server._encoder
    n_legal = len(valid_moves)
    valid_idx = [encoder.move_to_index(m) for m in valid_moves]

    # --- Raw policy pass (always) ---
    wrapper = server.get_wrapper(model_id)
    move_probs_t, value_t = wrapper.predict_from_state(gs)
    probs_np = move_probs_t.detach().cpu().numpy()
    if probs_np.ndim > 1:
        probs_np = probs_np[0]
    raw_masked = np.zeros_like(probs_np)
    for i in valid_idx:
        raw_masked[i] = probs_np[i]
    s = raw_masked.sum()
    raw_norm = raw_masked / s if s > 1e-12 else raw_masked
    raw_value = float(value_t.detach().cpu().reshape(-1)[0].item())
    raw_entropy = _policy_entropy(raw_norm[raw_norm > 0])
    raw_sorted = sorted(valid_idx, key=lambda i: -raw_norm[i])
    raw_top = [
        {"idx": int(i), "prob": float(raw_norm[i])}
        for i in raw_sorted[:top_k]
    ]
    raw_argmax_idx = raw_sorted[0]
    raw_argmax_move = next(m for m in valid_moves if encoder.move_to_index(m) == raw_argmax_idx)
    raw_best_value = server._best_move_value(wrapper, gs, raw_argmax_move)

    record: Dict[str, Any] = {
        "id": payload.get("id"),
        "ok": True,
        "meta": payload.get("meta", {}),
        "phase": payload["phase"],
        "side_to_move": payload["side_to_move"],
        "scores": payload["scores"],
        "n_pieces": len(payload["pieces"]),
        "n_legal_moves": n_legal,
        "raw_policy": {
            "value": raw_value,
            "best_move_value": raw_best_value,
            "entropy": raw_entropy,
            "top": raw_top,
            "argmax_idx": int(raw_argmax_idx),
        },
        "mcts": {},
    }

    # Build a fast lookup from policy slot index → rank in the raw policy.
    raw_rank_by_idx = {int(i): r for r, i in enumerate(raw_sorted)}

    # --- MCTS at each sim budget ---
    for sims in sim_budgets:
        if sims <= 0:
            continue
        mcts = server.get_mcts(model_id, sims)
        t0 = time.perf_counter()
        try:
            mcts_probs = mcts.search(gs, move_number=1)
        except Exception as e:  # noqa: BLE001
            record["mcts"][str(sims)] = {"error": str(e)}
            continue
        wall_s = time.perf_counter() - t0

        root_q = float(getattr(mcts, "last_root_value", 0.0))
        sorted_idx = sorted(valid_idx, key=lambda i: -mcts_probs[i])
        top_idx = sorted_idx[0]
        top_move = next(m for m in valid_moves if encoder.move_to_index(m) == top_idx)
        best_val = server._best_move_value(wrapper, gs, top_move)
        # Rank in raw policy: this is the headline misalignment metric.
        rank_of_final_best = raw_rank_by_idx.get(int(top_idx), -1)
        # Top-K visit distribution
        top_records = [
            {
                "idx": int(i),
                "prob": float(mcts_probs[i]),
                "approx_visits": int(round(float(mcts_probs[i]) * sims)),
            }
            for i in sorted_idx[:top_k]
        ]
        # Visit-distribution entropy
        mcts_entropy = _policy_entropy(np.array([mcts_probs[i] for i in valid_idx]))
        # Gap between #1 and #2 (in fractional probability)
        gap_1_2 = (
            float(mcts_probs[sorted_idx[0]]) - float(mcts_probs[sorted_idx[1]])
            if len(sorted_idx) > 1 else 1.0
        )
        # Value cost of misalignment: how much value would we have lost by
        # following raw policy instead of MCTS? Positive = MCTS overruled
        # favorably. Near zero = raw and MCTS picked moves of equivalent
        # value (the "harmless misalignment" case — different move, same
        # outcome). Large positive = the policy head was leading us into
        # measurably worse positions.
        value_gain_over_raw = (
            best_val - raw_best_value
            if best_val is not None and raw_best_value is not None
            else None
        )
        # Opposite-sign divergence between root_q (search-averaged value
        # from side-to-move POV) and best_move_value (network value at the
        # post-move position, same POV). This is the diagnostic for "MCTS
        # confident position is good but value head one ply deeper says
        # no" — exactly the failure pattern that hit the row-10 counter-
        # capture position 2026-05-26. Skip when either side is near-zero
        # (sign noise on undecided positions isn't meaningful).
        NEAR_ZERO = 0.05
        opposite_sign_divergence = None
        if best_val is not None and abs(root_q) > NEAR_ZERO and abs(best_val) > NEAR_ZERO:
            opposite_sign_divergence = bool(
                (root_q > 0) != (best_val > 0)
            )
        record["mcts"][str(sims)] = {
            "root_q": root_q,
            "best_move_value": best_val,
            "value_gain_over_raw": value_gain_over_raw,
            "opposite_sign_divergence": opposite_sign_divergence,
            "entropy": mcts_entropy,
            "top": top_records,
            "rank_of_final_best": rank_of_final_best,
            "gap_1_2": gap_1_2,
            "wall_seconds": wall_s,
        }
    return record


def parse_sims(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positions", required=True, type=Path)
    p.add_argument("--model-id", required=True)
    p.add_argument("--sims", default="0,100,400,1200,3200",
                   help="comma-separated sim budgets (0 = raw policy)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only process the first N positions (debug)")
    args = p.parse_args()

    sims = parse_sims(args.sims)
    log.info("sim budgets: %s", sims)

    positions = []
    with args.positions.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            positions.append(json.loads(line))
    if args.limit:
        positions = positions[: args.limit]
    log.info("measuring %d positions with model %s", len(positions), args.model_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_ok = 0
    n_err = 0
    with args.out.open("w") as f:
        for i, pos in enumerate(positions, 1):
            rec = measure_one(pos, args.model_id, sims, top_k=args.top_k)
            f.write(json.dumps(rec) + "\n")
            if rec.get("ok"):
                n_ok += 1
            else:
                n_err += 1
            if i % 10 == 0 or i == len(positions):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                log.info("  %d/%d (%.1f pos/s)  ok=%d  err=%d", i, len(positions), rate, n_ok, n_err)
    log.info("done — ok=%d err=%d in %.1fs", n_ok, n_err, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
