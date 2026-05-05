# BatchedEvaluator — Implementation Plan

> **Goal**: land the shared `BatchedEvaluator` (designed in
> `BATCHED_EVALUATOR_DESIGN.md`) on the `gpu-scaling-harness` branch
> and merge to `main` within 2 days.
>
> **Constraint**: keep every phase independently committable so we can
> ship a partial result if a later phase invalidates an assumption.
> Phases are sized in hours, not story points.

## Where this slots into existing code

Read alongside these files — the plan refers to specific entry points:

| File | What's there now | What changes |
|---|---|---|
| `yinsh_ml/training/self_play.py:132` (`class MCTS`) | constructor takes `network` directly | new optional `evaluator` arg |
| `yinsh_ml/training/self_play.py:889` (`_evaluate_and_backup_batch`) | calls `self.network.predict_batch(states)` | route through `evaluator.evaluate_batch(...)` when set |
| `yinsh_ml/training/self_play.py:1530` (`play_game_worker`) | each worker creates its own `NetworkWrapper` | new sibling function `play_game_thread` that takes an `evaluator` instead |
| `yinsh_ml/training/self_play.py:1407` (`generate_games` serial branch) | calls `play_game_worker` directly | unchanged |
| `yinsh_ml/training/self_play.py:1448` (`generate_games` pool branch) | `ProcessPoolExecutor` of `play_game_worker` | new third branch: `ThreadPoolExecutor` of `play_game_thread` against shared evaluator |

## Phases

### Phase 0 — `BatchedEvaluator` skeleton + unit tests  *(~2h)*

**New file**: `yinsh_ml/network/batched_evaluator.py`

```python
class BatchedEvaluator:
    def __init__(self, network: NetworkWrapper,
                 max_batch: int = 128,
                 max_wait_ms: float = 1.0)
    def evaluate(self, state) -> Tuple[np.ndarray, float]
    def evaluate_batch(self, states) -> Tuple[np.ndarray, np.ndarray]
    def shutdown(self)
    def __enter__(self) / __exit__(self)
```

Internals:
- `queue.Queue` of `_Request(state, Future)`
- `Thread(target=_drain_loop, daemon=True)` started in `__init__`
- Drain loop: pull until queue empty OR `max_wait_ms` elapsed OR len ≥ `max_batch`, then call `network.predict_batch(states)` and resolve futures.

**New test file**: `yinsh_ml/tests/test_batched_evaluator.py`

Three tests, all using a recording fake network:

1. `test_n_requests_collapse_to_one_predict_batch_call` — submit 100 requests fast; assert ≤2 calls to `predict_batch`, total positions = 100.
2. `test_max_batch_caps_call_size` — submit 500 requests with `max_batch=64`; assert no single call exceeds 64.
3. `test_evaluate_returns_correct_value_per_request` — fake network returns `value = i * 0.1`; submit 50 requests; assert each future resolves to its own value (no cross-talk in the routing).

**Also**: when this lands, the existing
`test_no_shared_batched_evaluator_module` in
`test_gpu_scaling_investigation.py` will fail. That's the trigger to
remove it and add a one-line update to `GPU_SCALING_PLAN.md` saying
the gap is closed.

**Exit criteria**: 3 new tests pass; old C4 test fails for the right
reason; no other test regressions.

**Commit**: `network: add BatchedEvaluator (no integration yet)`

### Phase 1 — Wire evaluator into `self_play.MCTS`  *(~2h)*

Single integration point. In `self_play.py`:

```python
# self_play.py:132 — class MCTS.__init__
def __init__(self, network, ..., evaluator=None):
    self.evaluator = evaluator  # optional shared evaluator

# self_play.py:889 — _evaluate_and_backup_batch
if self.evaluation_mode in ["pure_neural", "hybrid"]:
    if self.evaluator is not None:
        # one-line swap
        policy_logits_batch, values_batch = self.evaluator.evaluate_batch(states_to_evaluate)
    else:
        policy_logits_batch, values_batch = self.network.predict_batch(states_to_evaluate)
```

`evaluator.evaluate_batch` returns the same `(torch.Tensor, torch.Tensor)` shape as `predict_batch` so the call site is otherwise unchanged.

**New test**: `test_mcts_with_evaluator_matches_direct_path`. Run a 32-sim `search_batch` twice with a fixed seed: once with `evaluator=None`, once with `evaluator=BatchedEvaluator(network)`. Both should produce identical visit counts on the root's children. Pin determinism.

**Exit criteria**: new test passes; existing
`test_search_batch_calls_predict_batch_in_batches` (parametrized) still
passes (uses the no-evaluator path).

**Commit**: `self_play: optional shared evaluator in MCTS`

### Phase 2 — Thread-based `SelfPlay` mode  *(~3h)*

**Config**: new field `self_play.use_shared_evaluator: bool` in
`mode_settings`. Default `false` so existing configs are unchanged.

**New worker**: `play_game_thread(game_id, mcts_config, evaluator, ...)` —
a near-clone of `play_game_worker` minus the `NetworkWrapper` /
model-load lines. Takes the shared evaluator, otherwise identical.

**New branch in `generate_games`**:

```python
if self.use_shared_evaluator and self.num_workers > 0:
    # one network, one evaluator, N threads
    with BatchedEvaluator(self.network, max_batch=self.mcts_batch_size * 2) as ev:
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(play_game_thread, ...) for _ in range(num_games)]
            ...
elif self.num_workers > 0:
    # existing ProcessPoolExecutor path (unchanged)
    ...
else:
    # existing serial path (unchanged)
    ...
```

Default `max_batch = mcts_batch_size * num_workers` (so it can absorb a
full per-worker flush from every worker simultaneously without
truncating).

**Smoke test**: `test_threaded_selfplay_completes_smoke`. 2 games, 2
threads, 16 sims/move. Asserts `games_completed == 2` and that the
shared evaluator received `predict_batch` calls. Slow test — mark
`@pytest.mark.slow`.

**Exit criteria**: smoke test passes locally on Mac/MPS; 2-game run
finishes in <60s.

**Commit**: `self_play: thread-based mode using shared evaluator`

### Phase 3 — Validate end-to-end  *(~3h)*

Two checks before merging.

**3a. Mac smoke (~10 min wall time)**:

```bash
sed -E 's/^( *use_shared_evaluator: ).*/\1true/; \
        s/^( *num_workers: ).*/\1 4/' \
    configs/cloud_smoke.yaml > /tmp/smoke_threaded.yaml
# Add the line if it doesn't exist:
echo "  use_shared_evaluator: true" >> /tmp/smoke_threaded.yaml

python scripts/run_training.py --config /tmp/smoke_threaded.yaml --iterations 1
```

Pass criterion: `Generated and processed 10 games in <T>s` appears, no
errors. We're not measuring throughput on Mac — just confirming the
threaded path doesn't deadlock or crash.

**3b. Cloud sweep (~30 min wall time)**:

Add a `--use-evaluator` flag to `gpu_scaling_sweep.py` that injects
`use_shared_evaluator: true` into the cell config. Then:

```bash
python scripts/gpu_scaling_sweep.py \
    --base-config configs/cloud_smoke.yaml \
    --output-dir results/sweep_evaluator_$(date +%Y%m%d_%H%M%S) \
    --workers 4 8 16 \
    --batch-sizes 64 \
    --use-evaluator \
    --timeout 600
```

Pass criterion: best games/hr beats 1250 (current peak) by at least
2×, with `sm_avg ≥ 30%`. Append the new table to
`GPU_SCALING_RESULTS.md` and call out the multiplier.

**If 3b underperforms** (<2× speedup): don't merge. The hypothesis
that "command-queue contention is the bottleneck" was wrong; we'd
need to instrument `predict_batch` itself before deciding what's next.

**Commit**: `docs: BatchedEvaluator validation results — Nx games/hr on 4090`

### Phase 4 — Cleanup + merge  *(~1h)*

- Delete `test_no_shared_batched_evaluator_module` (it now fails for
  the right reason — gap closed).
- Update `GPU_SCALING_PLAN.md` Part 1 future-work note to "delivered
  in PR #12, see results below."
- Squash review-style fixups before merging.

## Total wall-clock budget

| Phase | Time |
|---|---|
| 0 — skeleton + tests | 2h |
| 1 — MCTS wiring | 2h |
| 2 — threaded SelfPlay | 3h |
| 3 — validation (Mac + cloud) | 3h |
| 4 — cleanup | 1h |
| **Total** | **~11h** |

Two-day window has plenty of slack. If a phase blows up, drop Phase 4
cleanup off the critical path and merge with the test left in place
(it can come out in a follow-up).

## Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| GIL contention from MCTS Python loops | medium | Bitboard work moved most CPU work to C++. If profiling shows GIL is the wall, fall back to `ProcessPoolExecutor` of N processes that each share *one* model on a single CUDA device via `torch.multiprocessing` shared tensors — uglier but doable. |
| First-flush latency hurts at low concurrency | low | `max_wait_ms` is configurable. Set lower for N ≤ 2 workers. |
| Determinism drift between threaded & process modes | medium | Phase 1 test pins this on the search level. If it breaks at the SelfPlay level, the divergence is in worker-side state (likely Dirichlet noise per worker) — not catastrophic, just needs documenting. |
| Evaluator deadlock on shutdown | low | `shutdown()` cancels pending futures; `__exit__` always called. Test by interrupting Phase 3 mid-run — should clean up. |
| Cloud sweep underperforms | medium | Already have an off-ramp: don't merge, file an issue, keep `num_workers: 4` as the lock-in. |

## What we're not doing in this PR

- Multi-GPU. Same shape, deferred.
- Async/await rewrite. Threads + blocking futures are sufficient.
- Per-evaluator metrics dashboard. Eyeballing `nvidia-smi dmon` is enough.
- Backporting the evaluator into the inference-time `yinsh_ml/search/mcts.py`. Separate concern — that path is for tournaments, not training.

## After this lands

Move to long-run config. See `configs/cloud_run_v1.yaml` (added in
the same PR as a follow-on commit). That's the file you'd actually
launch overnight.
