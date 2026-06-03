#!/usr/bin/env python
"""Launch the arms of a weight experiment in parallel.

Reads a manifest produced by gen_weight_experiment.py and runs each arm's
training as a subprocess, capped at ``--max-parallel`` concurrent jobs (set this
to your GPU count on a single node; use a job array / multiple invocations
across nodes). Per-arm stdout+stderr stream to ``<save_dir>/train.log``.

The actual training (scripts/run_training.py) requires torch/GPU, so real runs
happen on the cloud. The orchestration itself is generic: ``--train-cmd`` lets
you substitute any command (e.g. for a dry run or a smoke test).

Examples:
  # real cloud run, 2 GPUs
  python scripts/experiments/run_experiment.py \
      --manifest configs/experiments/refit_v1/manifest.json --max-parallel 2

  # dry run (print the commands only)
  python scripts/experiments/run_experiment.py \
      --manifest configs/experiments/refit_v1/manifest.json --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_TRAIN_CMD = "python scripts/run_training.py --config {config} --save-dir {save_dir}"


def build_commands(manifest: dict, train_cmd: str) -> dict:
    """Return ``{arm: command_string}`` for each arm in the manifest."""
    cmds = {}
    for name, arm in manifest["arms"].items():
        cmds[name] = train_cmd.format(config=arm["config"], save_dir=arm["save_dir"])
    return cmds


def _run_one(name: str, cmd: str, save_dir: str) -> dict:
    log_dir = Path(save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    start = time.time()
    with open(log_path, "w") as log:
        log.write(f"$ {cmd}\n")
        log.flush()
        proc = subprocess.run(cmd, shell=True, stdout=log,
                              stderr=subprocess.STDOUT)
    return {"arm": name, "returncode": proc.returncode,
            "seconds": round(time.time() - start, 1), "log": str(log_path)}


def run_arms(manifest: dict, train_cmd: str, max_parallel: int,
             dry_run: bool = False) -> list:
    cmds = build_commands(manifest, train_cmd)
    if dry_run:
        for name, cmd in cmds.items():
            print(f"[{name}] {cmd}")
        return [{"arm": n, "command": c, "dry_run": True} for n, c in cmds.items()]

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(_run_one, name, cmd, manifest["arms"][name]["save_dir"]): name
            for name, cmd in cmds.items()
        }
        for fut in as_completed(futures):
            res = fut.result()
            status = "OK" if res["returncode"] == 0 else f"FAIL({res['returncode']})"
            print(f"[{res['arm']}] {status} in {res['seconds']}s -> {res['log']}")
            results.append(res)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--train-cmd", default=DEFAULT_TRAIN_CMD,
                    help="command template with {config} and {save_dir} placeholders")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    results = run_arms(manifest, args.train_cmd, args.max_parallel, args.dry_run)
    if not args.dry_run:
        out = Path(args.manifest).with_name("run_status.json")
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"status -> {out}")
        failed = [r for r in results if r.get("returncode", 0) != 0]
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
