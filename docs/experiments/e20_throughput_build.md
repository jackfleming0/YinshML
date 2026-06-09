# E20 — Self-play throughput build

**Status:** BUILT + MEASURED on a rented 4090 (2026-06-09). ~20× serial / ~9.5× the
old process-pool ceiling, measured. Chasing the last ~1.6× to the GPU roofline.
**Date(s):** scoped 2026-06-01; built + measured 2026-06-09.
**Cost:** engineering build + a few 4090-hours of micro-benchmarking (no full training run).
**Branch / artifacts:** landed on `main` (commits `f5d0d14` inference server →
`…` virtual-loss fix). Config flag `self_play.use_inference_server: true`
(`configs/inference_server_smoke.yaml`). Code: `yinsh_ml/network/inference_server.py`,
`yinsh_ml/network/wrapper.py::predict_batch_encoded`,
`yinsh_ml/training/self_play.py` (dispatch + `play_games_inference_client` +
`_select_action` virtual-loss fix). Benchmark: `scripts/bench_selfplay_throughput.py`.
Tests: `yinsh_ml/tests/test_inference_server.py`.

---

## TL;DR (2026-06-09)

Self-play was bounded by **N worker processes each serializing `predict_batch` on
the GPU command queue** (per `GPU_SCALING_RESULTS.md`): the per-worker path peaked
at ~677 games/hr (4 workers) then *regressed*. Two levers fixed it, measured on a
4090 (pure-neural, sims=48, C++ engine on):

| config | games/hr | vs serial (326) | mean coalesced GPU batch |
|---|---:|---:|---:|
| serial (`num_workers=0`) | 326 | 1× | — |
| process_pool @4 (old peak) | 677 | 2.1× | n/a (per-worker) |
| process_pool @8 | 636 | *regresses* | n/a |
| inference_server @32 (coalescing only) | 2,351 | 7.2× | 16 |
| inference_server @32 + real virtual loss | 6,447 | 19.8× | 125 |
| **inference_server @48 + real virtual loss (peak)** | **7,140** | **21.9×** | **127** |

Confirmed on the **real (hybrid) recipe**: 3,834 g/hr @32w = **8.65× serial** (smaller
than pure-neural because per-worker heuristic CPU work slows request generation).

GPU roofline (pure forward, no MCTS) ≈ **12,800 evals/sec at batch ~256**; at the 48w
peak we feed ~7,690 evals/sec ≈ **64% of roofline**, GPU ~64% duty cycle. The coalesced
batch plateaus at ~127 and **more workers regress** (64w < 48w) — so the remaining ~1.6×
is **CPU/IPC/single-process-server bound, not GPU-batch bound** (see Open Levers).

**The win reinvests two ways** ("data-dense training in the same wall-clock"): more
games/hr *or* more sims/move at fixed wall-clock. Per E25 (value-head representational
ceiling), the high-value spend is **more sims/move** — sharper value targets — not just
more games. The unifying metric is **MCTS-sims/sec with the GPU fed**, which also makes
this extensible to single-game search latency (analysis board).

---

## What was actually built (and why it diverged from the 2026-06-01 scope)

The original scope (below) assumed the **threaded** shared evaluator
(`use_shared_evaluator`). That path was abandoned for a **process-based** server.
The reason is the decisive finding of this build:

> **The C++ engine's pybind bindings only release the GIL in the `Bench*`
> microbenchmark functions** (`yinsh_ml/game_cpp/src/bindings.cpp`) — the
> production hot-path methods (`GetValidMovesPy`, `ApplyMovePy`, clone) hold the
> GIL for their whole call. So a threaded evaluator can coalesce inference but its
> MCTS tree work + C++ board calls still serialize on the GIL. This is why the
> May 2026 threaded `BatchedEvaluator` measured *slower than serial* (470–541 vs
> 702 g/hr). Threads were never going to work without also rewriting the bindings.

So we went **process-based** (the fallback the design doc itself named):

- **One GPU-resident server process** (`run_inference_server`) owns the only model
  and the only CUDA context → no command-queue contention by construction. It drains
  a shared queue and **coalesces ragged per-worker leaf batches** into one forward
  pass (`NetworkWrapper.predict_batch_encoded`).
- **Workers stay separate processes** (no GIL between them), CPU-only — they build a
  NetworkWrapper on CPU purely for its encoder + MCTS handle, never run a forward
  pass, and **`torch.set_num_threads(1)`** so scaling to 32–64 workers on a big-core
  box doesn't oversubscribe (mirrors commit `4a9d3be` for the gen scripts — without
  it, 8 workers ran 609 g/hr; with it, 1,025).
- **Encode worker-side, ship arrays.** `CppGameState` doesn't pickle, so workers
  encode to compact (6×11×11) float arrays and ship those; the server never sees a
  game-state object. `predict_batch_encoded` is the encode/forward split.
- **`ProcessEvaluatorClient.evaluate_batch`** is a drop-in for the `evaluator=` seam
  MCTS already had — no MCTS call-site change for the transport.

### The second lever (the 2.7× multiplier): virtual loss was a no-op

Coalescing alone only got mean GPU batch ≈ 16 because **each worker's in-search
flush carried ~1 leaf**, not the configured 64. Root cause in
`MCTS._select_action`: for **unvisited** children it ignored virtual loss entirely
(Q fell back to a constant FPU baseline; the U-term had no visit denominator). So
putting an unexpanded leaf in-flight didn't change its score → every concurrent sim
re-selected the same top-prior leaf → the duplicate-leaf guard flushed at batch 1.

Fix: make virtual loss a **real loss** in a unified Q/U formula —
`q = (value_sum − virtual_losses) / (visits + virtual_losses)`, U denominator uses
`1 + adjusted_visits`. In-flight nodes are now repelled, so sims spread across
distinct leaves. **It reduces exactly to the prior formula when no virtual loss is
in flight** (every serial selection + the first sim of each batch), so serial search
is byte-unchanged — all 51 MCTS parity/consistency/FPU/subtree tests pass. Result:
per-worker flush 1 → ~14 leaves, server batch 16 → 125, **2.7× throughput**, and 7×
fewer GPU launches for the same work. This lever also lowers single-game search
latency, so it pays into the analysis board too.

---

## Measured results (4090, pure-neural, sims=48, C++ engine on, 64–96 games/run)

**GPU roofline** (pure `predict_batch_encoded`, no MCTS):

| batch | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| evals/sec | 6,328 | 6,485 | 7,294 | 12,117 | **12,847** | 10,715 |

**Inference-server worker sweep, before vs after the virtual-loss fix:**

| workers | g/hr (coalesce only) | g/hr (+ real VL) | batch (before→after) |
|---:|---:|---:|---:|
| 8 | 1,025 | 4,542 | 5.0 → 44.4 |
| 16 | 1,778 | 5,861 | 9.5 → 83.0 |
| 32 | 2,351 | 6,447 | 16.3 → 125.0 |

**48/64 workers (roofline chase):**

| workers | g/hr | mean coalesced batch | vs serial |
|---:|---:|---:|---:|
| 32 | 6,447 | 125.0 | 19.8× |
| **48** | **7,140** | 126.5 | **21.9×** |
| 64 | 6,909 | 128.8 | 21.2× (*regresses*) |

**Peak is 48 workers (7,140 g/hr).** Two things the higher sweep revealed:
- **Coalesced batch plateaus at ~127, not 256** — more workers don't grow the batch,
  they just add CPU/IPC contention (64w < 48w). Batch is gated by per-worker flush
  size (~14 leaves) × workers-flushing-per-`max_wait`-window, not by worker count.
- At 48w we feed ~7,690 evals/sec vs the ~12,100 batch-128 roofline ≈ **64% of
  roofline**, and the GPU sits at ~64% duty cycle (≈36% idle waiting for requests).
  **So the remaining gap is now CPU / IPC / single-process-server bound, NOT
  GPU-batch bound.** Adding workers is tapped out; closing it needs *faster* workers
  or a *faster server*, not *more* workers.

**Hybrid real-recipe confirmation (the actual training config):**

| config | g/hr | vs serial (443 hybrid) | batch |
|---|---:|---:|---:|
| serial, hybrid | 443 | 1× | — |
| inference_server @32, hybrid | **3,834** | **8.65×** | 36.0 |

The win **holds in production** (8.65× serial) but is smaller than pure-neural (3,834
vs 6,447 @32w) — the per-worker heuristic CPU work slows request generation, so fewer
flushes coalesce (batch 36 vs 125). This reinforces the diagnosis: at the real recipe
we are **worker-CPU bound**, which is exactly what the next levers target.

---

## How to reproduce (box runbook)

On a fresh vast.ai box (`/venv/main` conda env has torch+CUDA; system python does NOT):

```bash
cd ~/YinshML && git checkout main && git pull
source /venv/main/bin/activate
pip install pybind11                              # build dep, not preinstalled
python setup.py build_ext --inplace               # build the C++ .so for THIS python (3.12)
pip install -e . --no-deps --no-build-isolation   # put yinsh_ml on the path for scripts/
python -m pytest yinsh_ml/tests/test_inference_server.py -q   # parity + roundtrip

# Throughput sweep (modes: serial, process_pool, threaded, inference_server)
python scripts/bench_selfplay_throughput.py \
    --modes inference_server --workers 8,16,32,48 \
    --games 64 --sims 48 --batch-size 64 --eval-mode pure_neural
```

Gotchas hit this build: `pybind11` missing; the `.so` is per-python-version (gitignored,
must rebuild on the box); `predict_batch_encoded` parity test needs GPU/CPU fp32
tolerance (argmax + `atol=1e-2`), not bit-equality.

---

## Open levers toward the roofline (64% → ~100%)

The gap is **no longer GPU-batch bound** — at 48w the GPU is ~64% duty cycle, idle
~36% waiting for requests, and more workers regress. The remaining 1.6× is
**worker-CPU + IPC + single-process-server overhead**. Before the invasive lever,
**profile where a worker's per-leaf time goes** (tree CPU vs encode vs IPC round-trip
wait) — isolate the variable, then attack it:

1. **`inference_server_max_wait_ms`** (cheap, test first) — currently 1.0. Won't grow
   the batch much if workers can't supply it, but quick to falsify. Add `--max-wait`
   to the bench and sweep {1, 3, 5}.
2. **Faster IPC transport** — the server pickles a ~127-array batch in and scatters 48
   responses out *per forward*, single-threaded Python, at ~61 forward/sec. Shared-memory
   ring buffers (fixed-size slots) instead of `mp.Queue` pickling would cut both the
   server's scatter overhead and per-worker round-trip latency. Likely the biggest
   remaining lever now that we're IPC/server-bound.
3. **Vectorize the server scatter** / consider 2 server threads (GPU forward releases
   the GIL, so a drain thread + a scatter thread can overlap).
4. **Reduce per-worker tree CPU** — Gemini's integer-keyed-tree refactor (kill the
   ~23s/game Python `cpp_move_to_py`/`Position.__init__` glue, `BITBOARD_FOLLOWUP_PLAN.md`).
   Now **justified** (we're worker-CPU bound at the real recipe), and it doubles as the
   single-game-latency lever for the analysis board. Invasive — do it last, after
   profiling confirms tree CPU (not IPC) is the worker's dominant cost.

**Reinvestment (don't lose the plot):** the ~20× already banked converts to more
sims/move at fixed wall-clock. Per [[e25]], that (sharper value targets) is the
higher-value spend than raw games/hr — so chasing roofline and reinvesting in depth
are the same goal.

---

## Original scope (2026-06-01) — preserved for provenance

> **Status at the time:** QUEUED, gated on a proven depth lever. Framed as
> *threaded* shared evaluator + bitboards, in service of DEEPER search.
>
> Self-play is hard CPU-bound: 5090 at mean 32% util, ~40% of wall-clock at 0%,
> bursting to ~90%. Two levers, both meant to buy deeper search, not more shallow
> games: (a) shared/centralized evaluator to consolidate ragged batches; (b)
> bitboards in `board.py`/`moves.py`. The gate: only build E20 once a depth rung
> ([[e19]]) shows a real per-iteration learning rate — "throughput multiplies
> improvement, it does not create it."

**How the gate was resolved:** bitboards already landed (the C++ engine). The build
proceeded under a reframed goal — *support more data-dense training in the same
wall-clock* — which is itself the enabler [[e26]] called out as a prerequisite for
high-sim self-play. Throughput here is the means to afford depth (more sims/move),
not a substitute for it, so it's consistent with the original principle.

## Provenance & links

- Build session: 2026-06-09, on a rented vast.ai 4090 (192-core box).
- Related: `GPU_SCALING_RESULTS.md` (the 2026-05 diagnosis this fixes),
  `BATCHED_EVALUATOR_DESIGN.md` (the threaded path that lost to the GIL),
  [[e25]] (value-head ceiling → spend throughput on sims/move), [[e26]] (high-sim
  self-play needed this build).
