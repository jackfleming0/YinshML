#!/usr/bin/env python
"""Calibrate the failure-mode panel thresholds from REAL run metrics.

The panel's defaults are deliberately loose because healthy thresholds can't be
guessed — they depend on the value-head mode, outcome encoding, and this codebase's
metric scales. This tool derives them from actual training runs.

It ingests the per-iteration metrics JSON that ``run_training.py`` writes
(``<run>/**/metrics/iteration_*.json``), summarizes the distribution of each panel
signal, and writes a calibration file the panel loads via
``FailurePanel.from_calibration``.

Usage:
    # Calibrate from the runs already in experiments/ (default):
    python scripts/calibrate_panel.py

    # Point at specific run dirs (e.g. on your Mac, your real training output):
    python scripts/calibrate_panel.py --runs experiments/ runs/ --out configs/panel_calibration.json

Re-run this on your own hardware with your real configs — thresholds tuned to a
CPU smoke are not the thresholds for a Mac-trained model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

# Signals we can read from the supervisor's metrics JSON today. value_variance and a
# network policy-entropy signal aren't emitted, so they're not calibrated here.
_SIGNALS = ["value_accuracy", "value_loss", "policy_loss"]


def _final_iter_signals(metrics_json: Path) -> Optional[Dict[str, float]]:
    try:
        data = json.loads(metrics_json.read_text())
    except Exception:
        return None
    training = (data.get("metrics", {}) or {}).get("training")
    if isinstance(training, list):
        last = training[-1] if training else None
    elif isinstance(training, dict):
        last = training
    else:
        last = None
    if not last:
        return None
    return {s: last[s] for s in _SIGNALS if isinstance(last.get(s), (int, float))}


def _collect(run_dirs: List[str]) -> List[Dict[str, float]]:
    """Final-iteration signals, one row per metrics file found under the dirs."""
    rows: List[Dict[str, float]] = []
    seen = set()
    for d in run_dirs:
        for fp in sorted(glob.glob(os.path.join(d, "**", "metrics", "iteration_*.json"), recursive=True)):
            if fp in seen:
                continue
            seen.add(fp)
            sig = _final_iter_signals(Path(fp))
            if sig:
                rows.append(sig)
    return rows


def _summary(values: List[float]) -> Dict[str, float]:
    values = sorted(values)
    n = len(values)
    def pct(p):
        if n == 1:
            return values[0]
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return values[k]
    return {
        "n": n, "min": values[0], "p10": pct(0.10), "median": st.median(values),
        "max": values[-1], "mean": st.mean(values),
    }


def calibrate(run_dirs: List[str]) -> Dict:
    rows = _collect(run_dirs)
    observed = {}
    for s in _SIGNALS:
        vals = [r[s] for r in rows if s in r]
        if vals:
            observed[s] = _summary(vals)

    thresholds: Dict[str, float] = {}
    notes: List[str] = []
    if "value_accuracy" in observed:
        # Floor at half the lowest observed value — only flags a candidate that's
        # clearly below the worst real run, never a normal one.
        floor = round(max(0.0, observed["value_accuracy"]["min"] * 0.5), 4)
        thresholds["min_value_accuracy"] = floor
        notes.append(
            f"min_value_accuracy={floor} = 0.5 x observed min "
            f"({observed['value_accuracy']['min']:.4f}); value_accuracy spans "
            f"[{observed['value_accuracy']['min']:.3f}, {observed['value_accuracy']['max']:.3f}] "
            "across the reference runs — a weak health signal at this scale, so this "
            "is a loose collapse guard, not a quality bar."
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dirs": run_dirs,
        "n_metric_files": len(rows),
        "observed": observed,
        "thresholds": thresholds,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate failure-panel thresholds")
    parser.add_argument("--runs", nargs="+", default=["experiments"],
                        help="Directories to scan for run metrics (recursive).")
    parser.add_argument("--out", default="configs/panel_calibration.json")
    args = parser.parse_args()

    result = calibrate(args.runs)
    if result["n_metric_files"] == 0:
        print(f"No run metrics found under {args.runs}. Nothing to calibrate.", file=sys.stderr)
        print("Run some experiments first, or point --runs at a dir with metrics/iteration_*.json.")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("=" * 64)
    print("PANEL CALIBRATION")
    print("=" * 64)
    print(f"  reference metric files: {result['n_metric_files']}  (from {', '.join(args.runs)})")
    for sig, s in result["observed"].items():
        print(f"  {sig:16} n={s['n']:>3}  min={s['min']:.4f}  median={s['median']:.4f}  max={s['max']:.4f}")
    print("\n  recommended thresholds:")
    for k, v in result["thresholds"].items():
        print(f"    {k} = {v}")
    for note in result["notes"]:
        print(f"  note: {note}")
    print(f"\n  written to {out}")
    print("  the panel auto-loads this via FailurePanel.from_calibration().")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
