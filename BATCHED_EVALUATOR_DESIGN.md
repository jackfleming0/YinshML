# Shared `BatchedEvaluator` — Design Sketch

> **2026-05-06 postscript — read this first.** The design below was
> implemented in PR #12 and measured on a 4090 cloud box. **It does
> not deliver the predicted 3-5× speedup.** Serial mode (no threads,
> no evaluator) measured 702 games/hr; the threaded path measured
> 470-541 across `num_workers ∈ {1, 4, 8, 16}` — slower, not faster.
> Root cause: GIL contention in MCTS Python code that the bitboard
> port did not eliminate. The shipped evaluator code itself is
> correct (7/7 unit tests, deterministic vs the direct path) and
> kept in-tree as a robustness path (it's the only parallel mode
> that runs on Python 3.12 + the cloud image we tested), but the
> design hypothesis below was wrong on this hardware. See "Threaded
> BatchedEvaluator: results vs. prediction" in
> `GPU_SCALING_RESULTS.md` for the post-mortem and the three forward
> options (more bitboard, `torch.multiprocessing` with shared model,
> or fix Python 3.12 spawn).

> **Status**: implemented (PR #12), measured (negative result above).
> Companion to `GPU_SCALING_PLAN.md` (PR #11) and
> `GPU_SCALING_RESULTS.md`. The
> 2026-05-05 sweep showed every other lever (`num_workers`,
> `mcts_batch_size`) is exhausted; this was the next piece of work
> that *plausibly* could have moved games/hr — and it didn't.

## Why we need it

Empirical numbers from `GPU_SCALING_RESULTS.md`:

```
sm_p95  = 73-75%   (GPU bursts at near-full speed when fed)
sm_avg  = 10-15%   (idle most of the time)
peak    = num_workers=4, ~1250 games/hr
8/16 workers actively hurt
```

The gap between `sm_p95` and `sm_avg` is the prize: ~5× more compute is
sitting on the table. The mechanism is N independent worker processes
each calling `predict_batch` with their own small batch — these
serialize on the GPU's command queue rather than coalescing into one
big call. More workers = more contention.

A shared evaluator turns N small calls into 1 big call. Expected
ceiling based on the `sm_p95`/`sm_avg` gap: **3-5× games/hr** (target
4000-6000 on the same hardware), with `sm_avg` finally landing in the
50-70% range.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Main process                      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │   BatchedEvaluator (single instance)        │   │
│  │   - owns the only NetworkWrapper on cuda    │   │
│  │   - request queue (states from workers)     │   │
│  │   - per-worker reply queues                 │   │
│  │   - drain loop: flush every Nms or hit B    │   │
│  └─────────────────────────────────────────────┘   │
│         ▲                       ▼                   │
│         │  state               │ (logits, value)    │
│         │                       │                    │
│   ┌─────┴───┐ ┌───────┐ ┌─────┴───┐ ┌───────┐     │
│   │ MCTS    │ │ MCTS  │ │ MCTS    │ │ MCTS  │     │
│   │ Worker  │ │Worker │ │ Worker  │ │Worker │     │
│   │ (thread)│ │(thread│ │ (thread)│ │(thread│     │
│   └─────────┘ └───────┘ └─────────┘ └───────┘     │
└─────────────────────────────────────────────────────┘
```

Key structural choices:

- **Threads, not processes.** This wasn't true a year ago — pure-Python
  MCTS code was where the hot loop lived, GIL contention killed
  threading. The bitboard work moved most of that into C++; threads are
  now viable. No IPC, no model duplication, no spawn overhead. Falls
  back to processes only if benchmarking shows GIL contention is
  meaningful.
- **One model on the GPU.** The current architecture has *N* — that's
  the wrong shape. 4090 has 24GB; one copy is plenty.
- **Per-worker reply queues, not a shared one.** Workers block on their
  own future; routing is O(1) by request ID instead of O(N) scan.

## API surface (sketch)

```python
class BatchedEvaluator:
    def __init__(self, network: NetworkWrapper,
                 max_batch: int = 128,
                 max_wait_ms: float = 1.0):
        self.network = network
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self._req_queue: Queue[_Request] = Queue()
        self._drain_thread = Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def evaluate(self, state) -> Tuple[np.ndarray, float]:
        """Submit one state, block on its result. Called from MCTS workers."""
        reply = Future()
        self._req_queue.put(_Request(state, reply))
        return reply.result()  # blocks until drain loop fills it

    def evaluate_batch(self, states) -> List[Tuple[np.ndarray, float]]:
        """For workers that already have multiple states ready."""
        ...

    def _drain_loop(self):
        """Pull requests off queue, flush every max_wait_ms or at max_batch."""
        while not self._stop:
            batch = self._collect(timeout=self.max_wait_ms)
            if not batch: continue
            states = [r.state for r in batch]
            policy_logits, values = self.network.predict_batch(states)
            for r, pl, v in zip(batch, policy_logits, values):
                r.reply.set_result((pl, v))

    def shutdown(self): ...
```

MCTS-side change is minimal — `MCTS` constructor takes an
`evaluator: Optional[BatchedEvaluator]`. If given, leaves go to
`evaluator.evaluate_batch(...)` instead of `network.predict_batch(...)`.
The existing `search_batch` per-game flush logic remains; it just
becomes the "first level of batching" with the evaluator as the second.

## Sharp edges

- **Backpressure.** With max_wait_ms=1ms and 4 workers each generating
  ~50 leaves/game-step, request queue could spike to 200+ entries
  between flushes. Need a high-water mark that forces a flush
  regardless of timer. Easy: `if len(batch) >= max_batch: flush_now()`.
- **First-flush latency.** A solo worker's first leaf could wait the
  full max_wait_ms. For training that's fine (50-100µs of real wait
  time when max_wait_ms=1.0); for interactive eval it might matter and
  we want a `bypass_for_size_one` knob.
- **Cleanup at shutdown.** Reply futures pending when `shutdown()` is
  called must be cancelled, not left dangling — workers will deadlock
  otherwise.
- **GIL with very small networks.** The drain loop calls `predict_batch`
  which is mostly CUDA work — releases the GIL. But the encode loop
  inside `predict_batch` is Python. If profiling shows that's a wall,
  move it into the workers (they encode and submit pre-encoded
  tensors).
- **Configuration backwards-compat.** Existing configs set
  `num_workers` and `mcts_batch_size`. Map them: `num_workers` becomes
  the thread count, `mcts_batch_size` becomes the per-worker pre-flush
  collect size, `BatchedEvaluator.max_batch` is a new field
  (default 128).

## Validation plan

Three tests of decreasing scope, before merge:

1. **Unit**: `BatchedEvaluator` with a recording fake network — confirm
   N submitted requests produce one `predict_batch` call of size N (or
   `max_batch`, whichever is smaller). Mirror the C1 test in
   `test_gpu_scaling_investigation.py`.
2. **Integration**: 1-iteration self-play with the evaluator vs.
   without on the same config. Should produce equivalent
   game outcomes (both use the same RNG seed) and *measurably*
   higher games/hr.
3. **Sweep**: re-run `scripts/gpu_scaling_sweep.py` and add a
   `--use-evaluator` flag. The new winner should beat 1250 games/hr by
   the predicted multiplier (target: ≥3000) on the same 4090 box.
   Update `GPU_SCALING_RESULTS.md` with the new table.

## Out of scope

- Multi-GPU. Worth it once we've validated single-GPU. Same shape, more
  evaluators.
- Async/await rewrite of MCTS. Tempting but unnecessary — threads +
  blocking futures are simpler and good enough.
- Mixed-precision in the evaluator. Already toggled via
  `enable_autocast: true` in trainer configs; same flag flows through.
  Don't add a separate evaluator-side knob.

## Sizing

This is roughly a week of work for one person who knows the codebase.
Most of the risk is in the sharp-edges section above; the happy path is
straightforward. Don't bundle it with other changes — if the speedup is
real, you want it isolated so you can measure cleanly.
