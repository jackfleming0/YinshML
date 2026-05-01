#!/usr/bin/env python3
"""Paired stress benchmark for the bitboard-port branch's pre-merge gate.

Runs cloud_smoke_cpp_stress.yaml and cloud_smoke_py_stress.yaml back to
back (both 2 iterations × 10 games × sim=400, identical knobs except
``self_play.use_cpp_engine``) and reports the iter-2 self-play
wall-clock ratio. iter-2 not iter-1 — iter-1 includes per-process model
load + spawn + first-batch warmup that swamps the steady-state signal.

The number this prints is what goes in the merge PR description for
the bitboard-port branch. See BITBOARD_FOLLOWUP_PLAN.md "Pre-merge
validation".

Usage on cloud:

    git pull
    python scripts/paired_stress_benchmark.py

The supervisor itself logs "Generated and processed N games in T s"
once per iteration. We grep stdout for those lines, take the 2nd
occurrence per run, and ratio them.

Each variant takes a few minutes on a 4090 (10 games × ~5s/game ÷ 6
workers ≈ 80s/iter × 2 iters ≈ 3 min, plus model setup overhead).
Total runtime ~8-10 minutes for both variants.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_CPP = ROOT / "configs" / "cloud_smoke_cpp_stress.yaml"
CFG_PY = ROOT / "configs" / "cloud_smoke_py_stress.yaml"

# Matches the supervisor log line in
# yinsh_ml/training/supervisor.py:879 — "Generated and processed
# {num_games_generated} games in {game_time:.1f}s". A trailing 's' is
# part of the format; allow optional fractional seconds for safety.
_SP_TIME_RE = re.compile(
    r"Generated and processed\s+(\d+)\s+games in\s+([0-9]+(?:\.[0-9]+)?)s"
)


def run_variant(label: str, config: Path, log_dir: Path) -> dict:
    """Run one config end-to-end, tee stdout, return parsed iter-2 time.

    The supervisor process prints to stdout — we capture it as it
    streams so the user sees progress AND we can parse later. tee'ing
    to a file under log_dir/ gives a durable artifact for the PR.
    """
    log_path = log_dir / f"{label}.log"
    print(f"\n[{label}] config={config.name} → {log_path}", flush=True)
    start = time.time()

    cmd = [sys.executable, "scripts/run_training.py", "--config", str(config)]
    sp_times: list[tuple[int, float]] = []
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stdout.write(f"  [{label}] {line}")
            sys.stdout.flush()
            m = _SP_TIME_RE.search(line)
            if m:
                sp_times.append((int(m.group(1)), float(m.group(2))))
        rc = proc.wait()

    elapsed = time.time() - start
    result = {
        "label": label,
        "config": str(config),
        "log": str(log_path),
        "rc": rc,
        "elapsed_total": elapsed,
        "self_play_times": sp_times,
    }
    if rc != 0:
        print(f"[{label}] WARNING: run_training.py exited rc={rc}", flush=True)
    print(f"[{label}] done in {elapsed:.1f}s; iters captured: {len(sp_times)}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-cpp", action="store_true", help="skip the C++ run (debug only)"
    )
    ap.add_argument(
        "--skip-py", action="store_true",
        help="skip the Python-engine run (faster check, no ratio reported)",
    )
    ap.add_argument(
        "--log-dir", default="paired_stress_logs",
        help="where to tee the run logs (default: paired_stress_logs/)",
    )
    args = ap.parse_args()

    log_dir = ROOT / args.log_dir
    log_dir.mkdir(exist_ok=True)

    results: dict[str, dict] = {}
    if not args.skip_cpp:
        if not CFG_CPP.exists():
            sys.exit(f"missing config: {CFG_CPP}")
        results["cpp"] = run_variant("cpp", CFG_CPP, log_dir)
    if not args.skip_py:
        if not CFG_PY.exists():
            sys.exit(f"missing config: {CFG_PY}")
        results["py"] = run_variant("py", CFG_PY, log_dir)

    print("\n" + "=" * 72)
    print("Paired stress benchmark — iter-2 self-play wall clock")
    print("=" * 72)

    rows = []
    for label, r in results.items():
        sp = r["self_play_times"]
        if len(sp) >= 2:
            iter2_time = sp[1][1]
            iter2_games = sp[1][0]
            rows.append((label, iter2_games, iter2_time, r["elapsed_total"]))
            print(f"  {label}: iter-2 self-play = {iter2_time:.1f}s "
                  f"({iter2_games} games), total run = {r['elapsed_total']:.1f}s")
        else:
            print(f"  {label}: only captured {len(sp)} iteration(s); expected 2. "
                  f"Check {r['log']}.")

    cpp_row = next((r for r in rows if r[0] == "cpp"), None)
    py_row = next((r for r in rows if r[0] == "py"), None)
    if cpp_row and py_row:
        ratio = py_row[2] / cpp_row[2]
        print()
        print(f"  Speedup ratio (py / cpp) on iter-2: {ratio:.2f}x")
        print(f"  → paste this number in the merge PR description.")
    elif cpp_row:
        print()
        print("  Skipping ratio (Python run was skipped or failed).")

    print("=" * 72)


if __name__ == "__main__":
    main()
