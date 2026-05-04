"""End-to-end GPU scaling protocol for the questions in GPU_SCALING_PLAN.md.

The plan asks three things that all need to be answered with one experiment:

  1. Are we actually using the GPU? (`nvidia-smi dmon` answers this)
  2. Which combination of `num_workers` × `mcts_batch_size` saturates a 4090?
  3. Does raising `num_workers` reintroduce the historical zombie-process /
     RSS-leak issue that motivated `MAX_WORKERS = 0` in supervisor.py?

This script runs a sweep of (num_workers, mcts_batch_size) cells against a
base config. For each cell it:

  - writes a temporary config with the two knobs overridden
  - launches `scripts/run_training.py --config <tmp> --iterations 1`
  - in parallel, polls `nvidia-smi` once per second for sm%, mem%, power
  - in parallel, polls process RSS once per second (zombie/leak detection)
  - parses the supervisor's
        "Generated and processed N games in Ts"
    line out of stdout to compute games/hour
  - writes a per-cell directory with raw samples + parsed summary

When the full sweep finishes it writes:

  - results/sweep_summary.csv         — one row per cell, sortable
  - results/sweep_summary.md          — markdown table for the PR
  - results/winner.txt                — best cell by games/hour
                                        (with a "no RSS leak" sanity gate)

Run on the 4090 box, not your Mac. Locally it will fall back to no-GPU mode
and still record wall-clock + RSS, which is occasionally useful for sanity
testing the harness itself but does not answer the doc's question.

Usage:
  python scripts/gpu_scaling_sweep.py \
      --base-config configs/cloud_smoke.yaml \
      --output-dir results/sweep_$(date +%Y%m%d_%H%M%S) \
      --workers 6 12 \
      --batch-sizes 32 128

Defaults run a 2x2 grid (the doc's recommendation: pulls both cheap levers
once each so we can isolate which one mattered).

Hardware-budget note: each cell runs `num_iterations: 1` of the base config.
With cloud_smoke.yaml's `games_per_iteration: 10` that's typically 5-15min
per cell on a 4090. The default 4-cell sweep finishes in ~30-60min.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # RSS sampling becomes a no-op


ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Sampler threads                                                             #
# --------------------------------------------------------------------------- #


class GPUSampler(threading.Thread):
    """Polls `nvidia-smi --query-gpu=...` once per second.

    Records: utilization.gpu (sm%), utilization.memory, power.draw,
    memory.used. Skips silently if nvidia-smi is missing.
    """

    QUERY = "utilization.gpu,utilization.memory,power.draw,memory.used"

    def __init__(self, output_path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.interval = interval
        self._stop_event = threading.Event()
        self.available = shutil.which("nvidia-smi") is not None
        self.samples: List[dict] = []

    def run(self):
        if not self.available:
            return
        with self.output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "sm_pct", "mem_pct", "pwr_w", "mem_used_mb"])
            t0 = time.time()
            while not self._stop_event.is_set():
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--query-gpu={self.QUERY}",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=2.0,
                    )
                    line = out.strip().splitlines()[0]
                    sm, mem, pwr, mem_used = [s.strip() for s in line.split(",")]
                    row = [round(time.time() - t0, 2), sm, mem, pwr, mem_used]
                    writer.writerow(row)
                    f.flush()
                    self.samples.append(
                        dict(sm=float(sm), mem=float(mem), pwr=float(pwr),
                             mem_used=float(mem_used))
                    )
                except Exception:
                    pass
                self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()

    def summary(self) -> dict:
        if not self.samples:
            return dict(sm_avg=None, sm_p95=None, pwr_avg=None, mem_used_max=None)
        sm = sorted(s["sm"] for s in self.samples)
        pwr = [s["pwr"] for s in self.samples]
        mem_used = [s["mem_used"] for s in self.samples]
        return dict(
            sm_avg=round(sum(sm) / len(sm), 1),
            sm_p95=round(sm[int(0.95 * (len(sm) - 1))], 1),
            pwr_avg=round(sum(pwr) / len(pwr), 1),
            mem_used_max=round(max(mem_used), 1),
        )


class RSSSampler(threading.Thread):
    """Polls process RSS (and that of children) once per second.

    The doc explicitly flags the historical zombie-process / leak issue
    that motivated `MAX_WORKERS = 0`. Watch for monotonic growth here, not
    just the peak.
    """

    def __init__(self, pid: int, output_path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.output_path = output_path
        self.interval = interval
        self._stop_event = threading.Event()
        self.available = psutil is not None
        self.samples: List[Tuple[float, float]] = []  # (t_sec, rss_mb)

    def _total_rss_mb(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            proc = psutil.Process(self.pid)
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def run(self):
        if not self.available:
            return
        with self.output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "rss_mb"])
            t0 = time.time()
            while not self._stop_event.is_set():
                rss = self._total_rss_mb()
                if rss is None:
                    break  # process gone
                t = round(time.time() - t0, 2)
                writer.writerow([t, round(rss, 1)])
                f.flush()
                self.samples.append((t, rss))
                self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()

    def summary(self) -> dict:
        if not self.samples:
            return dict(rss_initial_mb=None, rss_peak_mb=None, rss_growth_mb=None)
        initial = self.samples[0][1]
        peak = max(s[1] for s in self.samples)
        # Use last 25% vs first 25% as the leak-vs-warmup signal.
        n = len(self.samples)
        head = [s[1] for s in self.samples[: max(1, n // 4)]]
        tail = [s[1] for s in self.samples[-max(1, n // 4):]]
        growth = (sum(tail) / len(tail)) - (sum(head) / len(head))
        return dict(
            rss_initial_mb=round(initial, 1),
            rss_peak_mb=round(peak, 1),
            rss_growth_mb=round(growth, 1),
        )


# --------------------------------------------------------------------------- #
# Cell runner                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    num_workers: int
    mcts_batch_size: int
    wall_clock_s: float
    games: Optional[int]
    games_per_hour: Optional[float]
    sm_avg: Optional[float]
    sm_p95: Optional[float]
    pwr_avg: Optional[float]
    mem_used_max: Optional[float]
    rss_initial_mb: Optional[float]
    rss_peak_mb: Optional[float]
    rss_growth_mb: Optional[float]
    return_code: int
    log_path: str
    failed: bool = False
    error: str = ""


def _write_cell_config(base_cfg_path: Path, num_workers: int,
                       mcts_batch_size: int, save_dir: Path,
                       cell_dir: Path) -> Path:
    """Deep-merge knob overrides into a copy of the base config."""
    with base_cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("self_play", {})["num_workers"] = num_workers
    cfg["self_play"]["mcts_batch_size"] = mcts_batch_size
    cfg["num_iterations"] = 1
    cfg["save_dir"] = str(save_dir)

    # Don't let arena/eval cost dominate the timing signal — the doc names
    # self-play as the dominant phase. Keep games_per_match modest.
    cfg.setdefault("arena", {}).setdefault("games_per_match", 4)

    out = cell_dir / "config.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _parse_games_and_time(log_text: str) -> Tuple[Optional[int], Optional[float]]:
    """Extract `Generated and processed N games in Ts` from supervisor log."""
    import re

    last = None
    for m in re.finditer(
        r"Generated and processed (\d+) games in ([\d.]+)s", log_text
    ):
        last = (int(m.group(1)), float(m.group(2)))
    return last if last else (None, None)


def run_cell(base_cfg_path: Path, num_workers: int, mcts_batch_size: int,
             cell_dir: Path, timeout_s: int) -> CellResult:
    """Run one (num_workers, mcts_batch_size) cell and collect metrics."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    save_dir = cell_dir / "run"
    config_path = _write_cell_config(
        base_cfg_path, num_workers, mcts_batch_size, save_dir, cell_dir
    )

    log_path = cell_dir / "stdout.log"
    gpu_csv = cell_dir / "gpu_samples.csv"
    rss_csv = cell_dir / "rss_samples.csv"

    print(
        f"\n=== cell num_workers={num_workers} "
        f"mcts_batch_size={mcts_batch_size} ===",
        flush=True,
    )
    print(f"  config: {config_path}", flush=True)
    print(f"  logs:   {log_path}", flush=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_training.py"),
        "--config", str(config_path),
        "--iterations", "1",
    ]

    # Stream stdout to BOTH the log file and the parent's terminal — flying
    # blind for 30+ minutes per cell is what the previous run cost us.
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        bufsize=1,
        text=True,
    )

    log_f = log_path.open("w")
    prefix = f"[w={num_workers} b={mcts_batch_size}] "

    def _tee():
        try:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(prefix + line)
                sys.stdout.flush()
        except Exception:
            pass

    tee_thread = threading.Thread(target=_tee, daemon=True)
    tee_thread.start()

    gpu_sampler = GPUSampler(gpu_csv)
    rss_sampler = RSSSampler(proc.pid, rss_csv)
    gpu_sampler.start()
    rss_sampler.start()

    failed = False
    error = ""
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        failed = True
        error = f"timeout after {timeout_s}s"
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
        raise
    finally:
        gpu_sampler.stop()
        rss_sampler.stop()
        gpu_sampler.join(timeout=5)
        rss_sampler.join(timeout=5)
        tee_thread.join(timeout=5)
        log_f.close()

    wall = time.time() - t0
    log_text = log_path.read_text(errors="replace")
    games, gen_time = _parse_games_and_time(log_text)
    games_per_hour = (games / gen_time * 3600.0) if (games and gen_time) else None

    if rc != 0 and not failed:
        failed = True
        error = f"return code {rc}"

    gs = gpu_sampler.summary()
    rs = rss_sampler.summary()

    result = CellResult(
        num_workers=num_workers,
        mcts_batch_size=mcts_batch_size,
        wall_clock_s=round(wall, 1),
        games=games,
        games_per_hour=round(games_per_hour, 1) if games_per_hour else None,
        sm_avg=gs["sm_avg"],
        sm_p95=gs["sm_p95"],
        pwr_avg=gs["pwr_avg"],
        mem_used_max=gs["mem_used_max"],
        rss_initial_mb=rs["rss_initial_mb"],
        rss_peak_mb=rs["rss_peak_mb"],
        rss_growth_mb=rs["rss_growth_mb"],
        return_code=rc,
        log_path=str(log_path),
        failed=failed,
        error=error,
    )

    (cell_dir / "result.json").write_text(json.dumps(asdict(result), indent=2))
    print(f"  done: games/hr={result.games_per_hour} sm_avg={result.sm_avg} "
          f"rss_growth_mb={result.rss_growth_mb} {'[FAILED: ' + error + ']' if failed else ''}",
          flush=True)
    return result


# --------------------------------------------------------------------------- #
# Sweep + reporting                                                           #
# --------------------------------------------------------------------------- #


def write_summary(results: List[CellResult], output_dir: Path) -> None:
    """CSV + markdown summary + winner pick."""
    csv_path = output_dir / "sweep_summary.csv"
    md_path = output_dir / "sweep_summary.md"
    winner_path = output_dir / "winner.txt"

    fields = list(asdict(results[0]).keys()) if results else []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    lines = ["# GPU scaling sweep results", "",
             f"Cells: {len(results)}, "
             f"completed: {sum(1 for r in results if not r.failed)}", ""]
    lines.append(
        "| workers | batch | games | wall(s) | games/hr | sm_avg | sm_p95 | "
        "pwr_avg | rss_peak_mb | rss_growth_mb | status |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in results:
        status = "ok" if not r.failed else f"FAIL: {r.error}"
        lines.append(
            f"| {r.num_workers} | {r.mcts_batch_size} | {r.games} | "
            f"{r.wall_clock_s} | {r.games_per_hour} | {r.sm_avg} | "
            f"{r.sm_p95} | {r.pwr_avg} | {r.rss_peak_mb} | "
            f"{r.rss_growth_mb} | {status} |"
        )

    # Winner: best games/hr among cells that didn't fail and didn't grow
    # RSS by more than 500MB (rough leak-detector threshold; tune later).
    candidates = [r for r in results
                  if not r.failed
                  and r.games_per_hour is not None
                  and (r.rss_growth_mb is None or r.rss_growth_mb < 500)]
    candidates.sort(key=lambda r: r.games_per_hour or 0, reverse=True)

    if candidates:
        w = candidates[0]
        verdict = (
            f"Winner: num_workers={w.num_workers} "
            f"mcts_batch_size={w.mcts_batch_size} "
            f"@ {w.games_per_hour} games/hr "
            f"(sm_avg={w.sm_avg}%, rss_growth={w.rss_growth_mb}MB)"
        )
    else:
        verdict = "No cell qualified — all failed or showed RSS growth >500MB."

    lines.append("")
    lines.append(f"**{verdict}**")

    md_path.write_text("\n".join(lines) + "\n")
    winner_path.write_text(verdict + "\n")
    print("\n" + verdict, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", type=Path,
                   default=ROOT / "configs" / "cloud_smoke.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / f"sweep_{int(time.time())}")
    p.add_argument("--workers", type=int, nargs="+", default=[6, 12],
                   help="num_workers values to sweep")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128],
                   help="mcts_batch_size values to sweep")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-cell timeout in seconds (default 30min)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sweep plan and exit")
    args = p.parse_args()

    if not args.base_config.exists():
        sys.exit(f"base config not found: {args.base_config}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = [(w, b) for w in args.workers for b in args.batch_sizes]

    print(f"Sweep over {len(cells)} cells:")
    for w, b in cells:
        print(f"  num_workers={w} mcts_batch_size={b}")
    print(f"Output: {args.output_dir}")
    print(f"GPU sampling: {'enabled' if shutil.which('nvidia-smi') else 'DISABLED (nvidia-smi missing)'}")
    print(f"RSS sampling: {'enabled' if psutil else 'DISABLED (psutil missing)'}")

    if args.dry_run:
        return

    results: List[CellResult] = []
    try:
        for w, b in cells:
            cell_dir = args.output_dir / f"w{w}_b{b}"
            results.append(run_cell(args.base_config, w, b, cell_dir, args.timeout))
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial summary.", flush=True)

    if results:
        write_summary(results, args.output_dir)
    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
