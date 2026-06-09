"""Self-play throughput micro-benchmark (E20).

Isolates games/hr for the self-play STAGE under different parallelism
strategies, holding the model, sims, batch size, and engine fixed. No
training / eval / tournament — just `SelfPlay.generate_games` timed. This is
the controlled experiment behind the inference-server work: does routing N
worker processes through one GPU-resident coalescing server beat (a) the
per-worker ProcessPool path that serializes on the GPU command queue and
(b) the threaded shared-evaluator path that stalls on the GIL?

Modes:
  serial            num_workers=0, in-process (the honest peak before this work)
  process_pool      N processes, each its own predict_batch on the GPU (the ceiling)
  threaded          1 model, N threads via BatchedEvaluator (GIL-bound)
  inference_server  N processes -> 1 GPU server that coalesces (this experiment)

Weights are random — throughput is weight-independent (forward-pass cost is
identical), so no checkpoint is needed.

Example:
  python scripts/bench_selfplay_throughput.py \
      --modes serial,process_pool,threaded,inference_server \
      --workers 4,8,16 --games 48 --sims 48 --batch-size 64 --eval-mode pure_neural
"""

import argparse
import logging
import time

import torch

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import SelfPlay


def build_selfplay(network, mode, workers, sims, batch_size, use_cpp, eval_mode, inference_dtype="fp32"):
    kwargs = dict(
        network=network,
        num_workers=workers,
        evaluation_mode=eval_mode,
        heuristic_weight=0.3,
        num_simulations=sims,
        late_simulations=sims,
        use_batched_mcts=True,
        mcts_batch_size=batch_size,
        use_cpp_engine=use_cpp,
    )
    if mode == "serial":
        kwargs["num_workers"] = 0
    elif mode == "process_pool":
        pass
    elif mode == "threaded":
        kwargs["use_shared_evaluator"] = True
    elif mode == "inference_server":
        kwargs["use_inference_server"] = True
        kwargs["inference_server_dtype"] = inference_dtype
    else:
        raise ValueError(f"unknown mode {mode}")
    return SelfPlay(**kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="serial,process_pool,inference_server")
    p.add_argument("--workers", default="8", help="comma list; ignored for serial")
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--sims", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-mode", default="pure_neural", choices=["pure_neural", "hybrid"])
    p.add_argument("--inference-dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="server forward precision (inference_server mode only)")
    p.add_argument("--no-cpp", action="store_true", help="disable the C++ engine (default: on)")
    p.add_argument("--use-enhanced-encoding", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cpp = not args.no_cpp

    network = NetworkWrapper(device=device, use_enhanced_encoding=args.use_enhanced_encoding)
    network.network.eval()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    worker_counts = [int(w) for w in args.workers.split(",") if w.strip()]

    print(f"\n# device={device} cpp_engine={use_cpp} eval_mode={args.eval_mode} "
          f"sims={args.sims} batch_size={args.batch_size} games/run={args.games}")
    print(f"{'mode':<18}{'workers':>8}{'games':>8}{'time_s':>10}{'games/hr':>12}")
    print("-" * 56)

    results = []
    for mode in modes:
        wlist = [0] if mode == "serial" else worker_counts
        for workers in wlist:
            sp = build_selfplay(network, mode, workers, args.sims, args.batch_size, use_cpp,
                                args.eval_mode, inference_dtype=args.inference_dtype)
            t0 = time.time()
            games = sp.generate_games(args.games)
            dt = time.time() - t0
            n = len(games)
            rate = (n / dt * 3600) if dt > 0 else 0.0
            label = mode if mode == "serial" else f"{mode}"
            print(f"{label:<18}{workers:>8}{n:>8}{dt:>10.1f}{rate:>12.0f}")
            results.append((mode, workers, n, dt, rate))

    # Summary: best rate and speedup vs serial (if measured).
    serial_rate = next((r for (m, w, n, dt, r) in results if m == "serial"), None)
    best = max(results, key=lambda x: x[4]) if results else None
    print("-" * 56)
    if best:
        speedup = f" ({best[4] / serial_rate:.2f}x vs serial)" if serial_rate else ""
        print(f"BEST: {best[0]} workers={best[1]} -> {best[4]:.0f} games/hr{speedup}")


if __name__ == "__main__":
    main()
