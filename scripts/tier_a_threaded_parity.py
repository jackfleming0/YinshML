"""Tier-A parity check: serial vs process-pool vs thread-pool-shared-evaluator.

Question: does `use_shared_evaluator=True` + `num_workers>0` (the threaded
BatchedEvaluator path) produce self-play targets statistically equivalent
to the serial / process-pool paths, on a real network at production sim
counts?

Existing unit tests already pin:
  * `test_mcts_serial_vs_batch_parity.py` — serial `search` and batched
    `search_batch` produce byte-identical visit distributions on a fake
    constant-output net (T1.1 fix verified).
  * `test_batched_evaluator.py::test_mcts_with_evaluator_matches_direct_path`
    — `search_batch(evaluator=None)` ≡ `search_batch(evaluator=BatchedEvaluator(net))`
    in a single thread.

This script closes the remaining gap: real network (`best_supervised.pt`)
+ thread fan-in (N concurrent threads through ONE BatchedEvaluator) +
production-relevant sim counts. The thing that could be wrong but isn't
covered above: a race / drop / reorder in the multi-thread coalescer that
only manifests under real fan-in.

Method: 3 dispatch paths, each generates K games on identical config.
All three forced to CPU so arithmetic is identical (no MPS-vs-CPU noise).
Compare aggregate target distributions via two-sample KS.

Verdict: PARITY iff all (A↔C, B↔C) KS p-values > α on the policy-entropy
and top-1-mass distributions, AND zero worker crashes on path C.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# Repo root on path so we can import yinsh_ml without `pip install -e .` quirks
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import SelfPlay


PATHS = ("serial", "process_pool", "threaded_shared_eval")


@dataclass
class PathStats:
    name: str
    num_games: int
    wall_time_s: float
    worker_crash_count: int
    game_lengths: List[int] = field(default_factory=list)
    policy_entropies: List[float] = field(default_factory=list)
    top1_mass: List[float] = field(default_factory=list)
    terminal_values: List[float] = field(default_factory=list)

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "num_games": self.num_games,
            "wall_time_s": round(self.wall_time_s, 2),
            "worker_crash_count": self.worker_crash_count,
            "games_per_second": round(self.num_games / self.wall_time_s, 3) if self.wall_time_s > 0 else 0.0,
            "mean_game_length": float(np.mean(self.game_lengths)) if self.game_lengths else 0.0,
            "mean_entropy": float(np.mean(self.policy_entropies)) if self.policy_entropies else 0.0,
            "mean_top1_mass": float(np.mean(self.top1_mass)) if self.top1_mass else 0.0,
            "mean_terminal_value": float(np.mean(self.terminal_values)) if self.terminal_values else 0.0,
            "total_moves": len(self.policy_entropies),
        }


def _make_selfplay(network: NetworkWrapper,
                   num_workers: int,
                   use_shared_evaluator: bool,
                   sims: int,
                   late_sims: int,
                   mcts_batch_size: int,
                   max_depth: int) -> SelfPlay:
    """Construct a SelfPlay matching the production Branch C MCTS-200 recipe
    with the *only* deltas being (1) sim count and (2) the dispatch-path
    flags. All quality knobs (fpu, epsilon-mix taper, subtree reuse,
    dirichlet alpha, c_puct, temperature schedule) match `configs/wave3_branchC_mcts200.yaml`.
    """
    return SelfPlay(
        network=network,
        num_workers=num_workers,
        evaluation_mode="pure_neural",   # cut the heuristic out so we measure
                                          # the threaded MCTS path cleanly
        heuristic_weight=0.0,
        num_simulations=sims,
        late_simulations=late_sims,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        value_weight=1.0,
        max_depth=max_depth,
        use_batched_mcts=True,
        mcts_batch_size=mcts_batch_size,
        enable_subtree_reuse=True,
        fpu_reduction=0.25,
        epsilon_mix_start=0.25,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=20,
        initial_temp=1.0,
        final_temp=0.1,
        annealing_steps=30,
        temp_clamp_fraction=0.6,
        use_shared_evaluator=use_shared_evaluator,
    )


def _collect_stats(name: str, games_data, wall_time: float, crash_count: int) -> PathStats:
    """Distill per-move policy targets into the comparable aggregates."""
    stats_obj = PathStats(name=name, num_games=len(games_data),
                          wall_time_s=wall_time, worker_crash_count=crash_count)
    for states, policies, values, _history in games_data:
        stats_obj.game_lengths.append(len(states))
        if len(values) > 0:
            stats_obj.terminal_values.append(float(values[-1]))
        for p in policies:
            p_arr = np.asarray(p, dtype=np.float64)
            s = p_arr.sum()
            if s <= 0:
                continue
            p_norm = p_arr / s
            mask = p_norm > 0
            ent = float(-np.sum(p_norm[mask] * np.log(p_norm[mask])))
            stats_obj.policy_entropies.append(ent)
            stats_obj.top1_mass.append(float(p_norm.max()))
    return stats_obj


def _run_path(name: str, network: NetworkWrapper, args) -> PathStats:
    print(f"\n--- {name} ---", flush=True)
    if name == "serial":
        sp = _make_selfplay(network, num_workers=0, use_shared_evaluator=False,
                            sims=args.sims, late_sims=args.late_sims,
                            mcts_batch_size=args.batch_size, max_depth=args.max_depth)
    elif name == "process_pool":
        sp = _make_selfplay(network, num_workers=args.num_workers, use_shared_evaluator=False,
                            sims=args.sims, late_sims=args.late_sims,
                            mcts_batch_size=args.batch_size, max_depth=args.max_depth)
    elif name == "threaded_shared_eval":
        sp = _make_selfplay(network, num_workers=args.num_workers, use_shared_evaluator=True,
                            sims=args.sims, late_sims=args.late_sims,
                            mcts_batch_size=args.batch_size, max_depth=args.max_depth)
    else:
        raise ValueError(f"unknown path: {name}")

    t0 = time.time()
    games = sp.generate_games(num_games=args.num_games)
    wall = time.time() - t0
    stats_obj = _collect_stats(name, games, wall, sp.worker_crash_count)
    s = stats_obj.summary()
    print(f"  games={s['num_games']:>3d}  crashes={s['worker_crash_count']}  "
          f"wall={s['wall_time_s']:>6.1f}s  games/s={s['games_per_second']:>5.3f}  "
          f"mean_len={s['mean_game_length']:>6.2f}  "
          f"mean_entropy={s['mean_entropy']:>5.3f}  "
          f"mean_top1={s['mean_top1_mass']:>5.3f}", flush=True)
    return stats_obj


def _ks(a: List[float], b: List[float]) -> Tuple[float, float]:
    if not a or not b:
        return (float("nan"), float("nan"))
    res = stats.ks_2samp(a, b)
    return (float(res.statistic), float(res.pvalue))


def _verdict(comparisons: Dict[str, Dict[str, Dict]], alpha: float) -> str:
    """PARITY iff (A↔C, B↔C) on entropy AND top1_mass all clear alpha,
    AND crashes==0 on path C. DIVERGENCE otherwise."""
    critical_pairs = [
        ("serial", "threaded_shared_eval"),
        ("process_pool", "threaded_shared_eval"),
    ]
    critical_metrics = ["policy_entropies", "top1_mass"]
    failures = []
    for a_name, b_name in critical_pairs:
        key = f"{a_name}_vs_{b_name}"
        if key not in comparisons:
            continue
        for m in critical_metrics:
            p = comparisons[key].get(m, {}).get("p_value", float("nan"))
            if np.isnan(p) or p < alpha:
                failures.append(f"{key}.{m} p={p:.4f}")
    return "PARITY" if not failures else "DIVERGENCE: " + "; ".join(failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yngine_volume_pretrain/best_supervised.pt")
    parser.add_argument("--num-games", type=int, default=8,
                        help="Games per dispatch path (3 paths total).")
    parser.add_argument("--num-workers", type=int, default=3,
                        help="Workers for process_pool + threaded paths. Serial is fixed at 0.")
    parser.add_argument("--sims", type=int, default=48,
                        help="Early-game MCTS simulations.")
    parser.add_argument("--late-sims", type=int, default=24,
                        help="Late-game MCTS simulations (after switch_ply).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="mcts_batch_size — leaf-eval flush threshold.")
    parser.add_argument("--max-depth", type=int, default=80,
                        help="Hard cap on plies per game (keeps the test bounded).")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="KS significance threshold. PARITY if all p > alpha.")
    parser.add_argument("--output", default="logs/tier_a_parity.json")
    parser.add_argument("--skip-paths", default="",
                        help="Comma-separated paths to skip (e.g. 'process_pool').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Silence the noisy SelfPlay info logs — the per-game progress prints
    # would otherwise drown the per-path summary lines.
    logging.getLogger("SelfPlay").setLevel(logging.WARNING)
    logging.getLogger("YinshTrainer").setLevel(logging.WARNING)

    skip = {s.strip() for s in args.skip_paths.split(",") if s.strip()}
    paths = [p for p in PATHS if p not in skip]
    if "threaded_shared_eval" in skip:
        print("WARNING: skipping the path under test — verdict will be uninformative.",
              file=sys.stderr)

    print(f"Tier-A parity check")
    print(f"  model={args.model}")
    print(f"  num_games={args.num_games}  num_workers={args.num_workers}")
    print(f"  sims={args.sims} late_sims={args.late_sims} batch_size={args.batch_size}")
    print(f"  max_depth={args.max_depth}  alpha={args.alpha}")
    print(f"  paths={paths}  device=cpu (forced — fair-arithmetic comparison)")

    # Force CPU on the parent network so all three paths execute identical
    # arithmetic. Workers default to CPU on Mac (no CUDA branch), and the
    # threaded path inherits the parent device.
    network = NetworkWrapper(model_path=args.model, device="cpu")
    network.network.eval()

    results: Dict[str, PathStats] = {}
    for name in paths:
        try:
            results[name] = _run_path(name, network, args)
        except Exception as e:
            print(f"  EXCEPTION in path '{name}': {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[name] = PathStats(name=name, num_games=0, wall_time_s=0.0,
                                      worker_crash_count=-1)

    # Pairwise KS on the metrics that would catch a real bug.
    comparisons: Dict[str, Dict[str, Dict]] = {}
    metric_names = ["game_lengths", "policy_entropies", "top1_mass", "terminal_values"]
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            key = f"{a_name}_vs_{b_name}"
            comparisons[key] = {}
            for m in metric_names:
                a_vals = getattr(results[a_name], m)
                b_vals = getattr(results[b_name], m)
                D, p = _ks(a_vals, b_vals)
                comparisons[key][m] = {"ks_statistic": D, "p_value": p,
                                       "n_a": len(a_vals), "n_b": len(b_vals)}

    verdict = _verdict(comparisons, args.alpha)

    print("\n=== KS two-sample tests (p-values) ===")
    print(f"{'pair':<42s} {'game_len':>10s} {'entropy':>10s} {'top1':>10s} {'term_val':>10s}")
    for key, cmp in comparisons.items():
        row = f"{key:<42s}"
        for m in metric_names:
            p = cmp[m]["p_value"]
            row += f" {p:>10.4f}" if not np.isnan(p) else " " + "nan".rjust(10)
        print(row)

    print(f"\n=== VERDICT === {verdict}")
    print(f"(alpha={args.alpha}; critical pairs are serial↔threaded and process_pool↔threaded "
          f"on policy_entropies + top1_mass)")

    # Crash check
    threaded = results.get("threaded_shared_eval")
    if threaded is not None and threaded.worker_crash_count != 0:
        print(f"\nWARNING: threaded path reported {threaded.worker_crash_count} worker crashes — "
              f"correctness verdict may be misleading.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump({
            "args": vars(args),
            "per_path": {n: r.summary() for n, r in results.items()},
            "comparisons": comparisons,
            "verdict": verdict,
            "alpha": args.alpha,
        }, fh, indent=2)
    print(f"\nWrote {args.output}")

    return 0 if verdict.startswith("PARITY") else 1


if __name__ == "__main__":
    sys.exit(main())
