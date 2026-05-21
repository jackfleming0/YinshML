#!/usr/bin/env python3
"""Run eval_vs_expert across many checkpoints, sharing the position cache.

Loads BGA expert positions once, then iterates over a list of checkpoints
emitting one JSON per checkpoint plus a CSV summary. ~30× faster than
calling eval_vs_expert.py once per checkpoint when there are many.
"""

import argparse
import csv
import glob
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.encoding import StateEncoder
from scripts.eval_vs_expert import evaluate, format_report, load_expert_positions

logger = logging.getLogger(__name__)


def discover_checkpoints(run_globs):
    found = []
    for g in run_globs:
        for p in sorted(glob.glob(g)):
            found.append(Path(p))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Glob patterns for checkpoints, e.g. 'runs_ablation_b/*/iteration_*/checkpoint_iteration_*_ema.pt'",
    )
    parser.add_argument("--games-dir", default="expert_games/bga/parsed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", default="expert_eval_reports")
    parser.add_argument("--summary-csv", default="expert_eval_reports/summary.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = StateEncoder()

    t0 = time.time()
    positions = load_expert_positions(args.games_dir, encoder)
    logger.info(f"Position load: {time.time() - t0:.1f}s, {len(positions)} positions")

    checkpoints = discover_checkpoints(args.runs)
    logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")

    rows = []
    for i, ckpt in enumerate(checkpoints):
        # ckpt path: runs_<X>/<timestamp>/iteration_<N>/checkpoint_iteration_<N>_ema.pt
        try:
            iter_num = int(ckpt.parent.name.split("_")[1])
            run_label = ckpt.parts[-4]
        except (IndexError, ValueError):
            iter_num = -1
            run_label = ckpt.parent.parent.name

        label = f"{run_label}_iter{iter_num}"
        logger.info(f"[{i + 1}/{len(checkpoints)}] {label} ...")

        ts = time.time()
        try:
            model = NetworkWrapper(model_path=str(ckpt), device=args.device)
            report = evaluate(model, positions, batch_size=args.batch_size)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

        # Write per-checkpoint JSON
        out_path = out_dir / f"{label}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": str(ckpt),
                    "run_label": run_label,
                    "iter": iter_num,
                    "report": report,
                },
                f,
                indent=2,
            )

        # Aggregate row for CSV
        all_b = report.get("ALL", {})
        main_b = report.get("MAIN_GAME", {})
        rc_b = report.get("ROW_COMPLETION", {})
        ring_b = report.get("RING_REMOVAL", {})
        rp_b = report.get("RING_PLACEMENT", {})
        rows.append(
            {
                "run": run_label,
                "iter": iter_num,
                "all_top1": all_b.get("top1_acc"),
                "all_top3": all_b.get("top3_acc"),
                "all_val_mse": all_b.get("value_mse"),
                "main_top1": main_b.get("top1_acc"),
                "main_top3": main_b.get("top3_acc"),
                "main_val_mse": main_b.get("value_mse"),
                "row_completion_top1": rc_b.get("top1_acc"),
                "ring_removal_top1": ring_b.get("top1_acc"),
                "ring_placement_top1": rp_b.get("top1_acc"),
                "elapsed_s": round(time.time() - ts, 1),
            }
        )
        logger.info(
            f"  ALL top1={all_b.get('top1_acc', 0):.3f} "
            f"MAIN top1={main_b.get('top1_acc', 0):.3f} "
            f"val_mse={all_b.get('value_mse', 0):.3f} "
            f"({time.time() - ts:.1f}s)"
        )

    # Write summary CSV
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {args.summary_csv}")
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
