#!/usr/bin/env python
"""Production shakedown for the experiment-orchestration layer.

Runs the real pipeline end-to-end on a small config so you can confirm everything
works *before* trusting it with real compute. Two stages:

  1. Train a baseline through the real launcher (``scripts/run_training.py``),
     producing a checkpoint and registry row.
  2. Self-vs-self integrity check: play the baseline checkpoint against a copy of
     itself through the evaluation funnel. A correct eval must NOT call a model
     stronger than itself — the SPRT should land at ``continue`` or ``accept_h0``
     and the win rate should straddle 50%. If a model "beats" itself, the eval is
     broken and nothing downstream can be trusted.

Optionally (``--full``) it then trains a fresh candidate against that baseline,
exercising the entire train → checkpoint → SPRT-match → route → gate path.

Usage:
    python scripts/bootstrap_orchestration.py                       # stages 1-2
    python scripts/bootstrap_orchestration.py --full               # + candidate run
    python scripts/bootstrap_orchestration.py --config configs/smoke.yaml --iterations 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestration production shakedown")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output-dir", default="experiments/bootstrap")
    parser.add_argument("--device", default=None, help="Override auto-detected device.")
    parser.add_argument("--games", type=int, default=20, help="Self-vs-self game cap.")
    parser.add_argument("--full", action="store_true",
                        help="Also train a candidate vs the baseline (full pipeline).")
    args = parser.parse_args()

    from yinsh_ml.orchestration import (
        EvaluationFunnel, ExperimentSpec, Journal, OrchestrationStore, Scheduler,
    )
    from yinsh_ml.orchestration.failure_panel import PanelInput
    from yinsh_ml.orchestration.match_runner import TournamentMatchRunner
    from yinsh_ml.orchestration.scheduler import _latest_checkpoint

    device = args.device or _detect_device()
    checks: list[tuple[str, bool, str]] = []

    store = OrchestrationStore(str(Path(args.output_dir) / "experiments.db"))
    funnel = EvaluationFunnel(batch_size=10, max_games=args.games)
    scheduler = Scheduler(
        store=store, journal=Journal(args.output_dir), funnel=funnel,
        output_dir=args.output_dir,
    )

    # --- Stage 1: train a baseline through the real launcher --------------
    print(f"\n[1/2] Training baseline ({args.config}, {args.iterations} iter, device={device})...")
    baseline = scheduler.process(ExperimentSpec(
        config_path=args.config, iterations=args.iterations, name="bootstrap-baseline",
    ))
    ok = baseline.launch_status == "completed"
    checks.append(("baseline training completed", ok, baseline.launch_status))
    if not ok:
        return _report(checks)

    baseline_ckpt = _latest_checkpoint(str(Path(args.output_dir) / baseline.experiment_id))
    checks.append(("baseline checkpoint produced", baseline_ckpt is not None, str(baseline_ckpt)))
    if baseline_ckpt is None:
        return _report(checks)

    # --- Stage 2: self-vs-self eval integrity -----------------------------
    print(f"\n[2/2] Self-vs-self integrity check (up to {args.games} games, device={device})...")
    runner = TournamentMatchRunner(
        candidate_ckpt=baseline_ckpt, baseline_ckpt=baseline_ckpt,
        training_dir=str(Path(args.output_dir) / baseline.experiment_id),
        device=device, eval_seed=0,
    )
    tier0 = funnel.run_tier0(PanelInput(), runner)
    o = tier0.outcome
    verdict = tier0.sprt.verdict if tier0.sprt else "n/a"
    print(f"      self-vs-self record: {o.wins}W/{o.draws}D/{o.losses}L; "
          f"SPRT={verdict}; win-rate CI [{tier0.wilson_lower:.2f}, {tier0.wilson_upper:.2f}]")
    # A model must not be judged stronger than itself.
    integrity_ok = verdict != "accept_h1"
    checks.append(("eval does not call a model stronger than itself", integrity_ok, f"SPRT={verdict}"))

    # --- Optional Stage 3: full candidate-vs-baseline ---------------------
    if args.full:
        print(f"\n[3/3] Training a candidate vs baseline {baseline.experiment_id} (full pipeline)...")
        cand = scheduler.process(ExperimentSpec(
            config_path=args.config, iterations=args.iterations,
            baseline_id=baseline.experiment_id, name="bootstrap-candidate",
        ))
        ok = cand.launch_status == "completed" and cand.decision is not None
        route = f"{cand.decision.route} -> {cand.decision.next_status}" if cand.decision else "n/a"
        checks.append(("candidate ran + routed against baseline", ok, route))

    return _report(checks, args.output_dir)


def _report(checks, output_dir: str | None = None) -> int:
    print("\n" + "=" * 64)
    print("ORCHESTRATION SHAKEDOWN")
    print("=" * 64)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name}  ({detail})")
    if output_dir:
        print(f"\n  Journals + feed under: {output_dir}/")
    print("=" * 64)
    print("RESULT:", "ALL CHECKS PASSED — production path is sound." if all_ok
          else "FAILURES ABOVE — fix before running real experiments.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
