#!/usr/bin/env python3
"""
Profile a single self-play game with the C++ engine to find the new
post-bitboard-port bottleneck. Backs the BITBOARD_FOLLOWUP_PLAN.md
"Step 0 — profile first" task.

Run on the cloud GPU box (CUDA + built `.so`):

    python scripts/profile_cpp_self_play.py --sims 400
    python scripts/profile_cpp_self_play.py --sims 48 --tag warmup

Drops `profile_cpp_<tag>.prof` next to the run; scp back for pstats
interactive use. Prints top-30 by tottime AND cumtime to stdout.
"""

import argparse
import cProfile
import os
import pstats
import sys
import tempfile
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400,
                    help="num_simulations (and late_simulations). 400 = steady-state, 48 = warmup case.")
    ap.add_argument("--tag", type=str, default=None,
                    help="Filename suffix for the .prof dump (default: sims<N>).")
    ap.add_argument("--device", type=str, default=None,
                    help="Force device (cuda / cpu). Default: cuda if available.")
    ap.add_argument("--top", type=int, default=30, help="Top-N rows printed per view.")
    ap.add_argument("--no-cpp", action="store_true",
                    help="Profile the Python engine instead (sanity check / baseline).")
    args = ap.parse_args()

    tag = args.tag or f"sims{args.sims}{'_py' if args.no_cpp else ''}"
    out_path = f"profile_cpp_{tag}.prof"

    # cProfile doesn't see other threads; pin BLAS to 1 so we measure
    # the worker's own time and not contention from torch threadpools.
    torch.set_num_threads(1)

    # Sanity: confirm the C++ extension is actually importable when we
    # claim to use it. A silent fallback would invalidate the profile.
    if not args.no_cpp:
        try:
            from yinsh_ml.game_cpp import _engine  # noqa: F401
            print(f"[profile] C++ engine loaded: {_engine.__file__}")
        except ImportError as e:
            print(f"[profile] FATAL: --no-cpp not set but _engine import failed: {e}",
                  file=sys.stderr)
            sys.exit(2)

    from yinsh_ml.network.wrapper import NetworkWrapper
    from yinsh_ml.training.self_play import play_game_worker

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[profile] device={device} sims={args.sims} use_cpp_engine={not args.no_cpp}")

    network = NetworkWrapper(device=device)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        torch.save(network.network.state_dict(), tmp.name)
        model_path = tmp.name

    cfg = dict(
        evaluation_mode="hybrid",
        heuristic_evaluator=None,
        heuristic_weight=0.5,
        num_simulations=args.sims,
        late_simulations=args.sims,
        simulation_switch_ply=20,
        fast_simulations=0,
        fast_sim_prob=0.0,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        value_weight=1.0,
        max_depth=200,
        initial_temp=1.0,
        final_temp=0.1,
        annealing_steps=30,
        temp_clamp_fraction=0.6,
        use_batched_mcts=True,
        mcts_batch_size=32,
        use_enhanced_encoding=False,
        enable_subtree_reuse=True,
        fpu_reduction=0.25,
        epsilon_mix_start=0.25,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=20,
        root_policy_temp=1.0,
        use_cpp_engine=not args.no_cpp,
    )

    print("[profile] starting cProfile run...")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    try:
        play_game_worker(model_path=model_path, game_id=0, mcts_config=cfg)
    finally:
        pr.disable()
        elapsed = time.perf_counter() - t0
        try:
            os.unlink(model_path)
        except OSError:
            pass

    print(f"[profile] worker finished in {elapsed:.1f}s")
    pr.dump_stats(out_path)
    print(f"[profile] dumped: {out_path} ({os.path.getsize(out_path)/1024:.0f} KiB)")

    stats = pstats.Stats(pr)

    print(f"\n{'='*70}\nTop {args.top} by TOTTIME (self-time, the bottleneck list)\n{'='*70}")
    stats.sort_stats("tottime").print_stats(args.top)

    print(f"\n{'='*70}\nTop {args.top} by CUMTIME (where wall-clock goes)\n{'='*70}")
    stats.sort_stats("cumulative").print_stats(args.top)

    # Filtered view: just our code, not torch/numpy internals — easier
    # to spot which wrapper / encoder calls dominate.
    print(f"\n{'='*70}\nTop {args.top} by TOTTIME, filtered to yinsh_ml/*\n{'='*70}")
    stats.sort_stats("tottime").print_stats("yinsh_ml", args.top)


if __name__ == "__main__":
    main()
