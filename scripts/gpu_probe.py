"""Five-second sanity check that the model + inference path actually use the GPU.

The GPU scaling sweep was producing 0 games per cell with `sm=0%` on a 4090
even after a torch+driver reinstall. The harness was silent for minutes
between log lines, so we couldn't tell which layer (driver, torch wheel,
NetworkWrapper, or supervisor init) was at fault.

This script bypasses the supervisor and self-play machinery entirely. It:

  1. Builds a NetworkWrapper(device='cuda').
  2. Confirms the model parameters actually landed on cuda:0.
  3. Runs single-state and batch-64 inference loops, syncing CUDA so the
     timing reflects real GPU work.
  4. Reports peak CUDA memory.

Three possible outcomes, in order of likelihood:

  - All output looks healthy → GPU path is fine; the harness hang is in
    supervisor init (dataset, experiment tracker, etc.) — investigate there.
  - `first param device: cpu` → NetworkWrapper isn't honoring device='cuda'
    on this image. Bug in the wrapper.
  - Script hangs or raises a CUDA error → the torch+driver pairing isn't
    actually working in practice, despite `torch.cuda.is_available()=True`.
    Get a different image / wheel.

Run with:
    python scripts/gpu_probe.py
"""

import time

import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.network.wrapper import NetworkWrapper


def main() -> None:
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ABORT: cuda not available — fix the torch/driver install before proceeding")
        return
    print(f"  device name:    {torch.cuda.get_device_name(0)}")

    print("building wrapper on cuda...")
    t = time.time()
    nw = NetworkWrapper(device="cuda")
    print(f"  built in {time.time() - t:.2f}s")
    print(f"  wrapper device:    {nw.device}")
    print(f"  first param device: {next(nw.network.parameters()).device}")
    print(f"  cuda mem after build: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    gs = GameState()

    print("warming up...")
    t = time.time()
    nw.predict_from_state(gs)
    torch.cuda.synchronize()
    print(f"  first call: {time.time() - t:.3f}s  (includes JIT/kernel cache)")

    print("100 single-state predict_from_state calls...")
    t = time.time()
    for _ in range(100):
        nw.predict_from_state(gs)
    torch.cuda.synchronize()
    elapsed = time.time() - t
    print(f"  total: {elapsed:.3f}s  ({1000 * elapsed / 100:.2f} ms/call)")

    print("100 batch-64 predict_batch calls...")
    batch = [GameState() for _ in range(64)]
    t = time.time()
    for _ in range(100):
        nw.predict_batch(batch)
    torch.cuda.synchronize()
    elapsed = time.time() - t
    print(f"  total: {elapsed:.3f}s  ({1000 * elapsed / 100:.2f} ms/call, "
          f"{6400 / elapsed:.0f} positions/s)")

    print(f"cuda mem peak: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
