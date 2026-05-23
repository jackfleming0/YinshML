# Tech Debt — MCTS Consolidation (RESOLVED 2026-05-21)

Seeded from the spaghetti surfaced while building the frozen-anchor yardstick
(see `VOLUME_PRETRAIN_RESULTS.md`), then resolved the same day. Kept as a record
of what was wrong and what was done. The lesson worth carrying forward is in §5.

## Outcome

The duplicate-`MCTS` problem is **resolved by excision.** There is now **one MCTS
engine**: `yinsh_ml/training/self_play.py::MCTS` (`search_batch()` for batched
leaf eval, plus a working singleton `search()`). The dead, broken
`yinsh_ml/search/mcts.py::MCTS` and everything that depended on it are gone.

---

## 1. What was wrong — duplicate MCTS, one fully dead

Two classes named `MCTS`:
- `yinsh_ml/search/mcts.py::MCTS` — **dead and broken at multiple layers.**
  `search()` never expanded the root (`action` stayed `None` on the first sim and
  the guard `continue`d before expansion) → returned a **uniform-random, net-blind**
  policy (`last_effective_child_visits = 0.0`; verified two different nets gave
  byte-identical policies). Fixing *that* then exposed a second bug: the eval path
  passed a numpy array to a torch model and crashed. **The engine never worked
  end-to-end with a real network.** Its tests passed only because they asserted on
  timing/telemetry/shape, never move quality.
- `yinsh_ml/training/self_play.py::MCTS` — the real, validated engine. Net-dependent,
  concentrated policies. Used by self-play training and `run_anchor_eval`.

**Blast radius (verified): the project's headline results were NEVER contaminated.**
Self-play training and the HA-ladder anchor gates both use the working engine
(`use_batched_mcts` defaults `True`). The dead engine only ever fed diagnostic
scripts and a test-only policy.

## 2. What was done (excision, 2026-05-21)

**Deleted:**
- `yinsh_ml/search/mcts.py` (the dead engine: `MCTS`, `MCTSConfig`, `EvaluationMode`,
  `MCTSNode`).
- `yinsh_ml/search/performance_profiler.py` — module-level hard-import of the dead
  engine; only consumer was the deleted perf test. (Resurfacing a real profiler
  against `self_play.MCTS` is a fresh-script job if ever wanted.)
- `scripts/eval_head_to_head_mcts.py` — broken diagnostic, superseded by
  `scripts/eval_vs_frozen_anchor.py`. Its old "white-wins-pattern" finding was an
  argmax-on-uniform artifact.
- `scripts/profile_mcts_selfplay.py`, `scripts/diagnose_value_predictions.py` —
  profiled/diagnosed the dead engine.
- `yinsh_ml/tests/test_heuristic_mcts_performance.py` — perf tests on the dead engine.

**Edited:**
- `yinsh_ml/search/__init__.py` — dropped the `.mcts` and `performance_profiler`
  re-exports; kept `training_tracker`, `transposition_table`, `node_type`.
- `yinsh_ml/self_play/policies.py` — `MCTSPolicy.__init__` now raises
  `NotImplementedError` pointing to `self_play.SelfPlay` / `MCTS.search_batch`;
  removed the dead import. (`SelfPlayRunner`/`game_runner.py`, which wired it, is
  never instantiated in production — only tests.)
- 4 test files surgically pruned of their dead-engine portions while **preserving
  every `self_play.MCTS` test** (B1 telemetry, epsilon-mix taper, backprop sign
  convention, GPU-scaling batch behavior): `test_telemetry_safeguards.py`,
  `test_epsilon_mix.py`, `test_mcts_backprop_perspective.py`,
  `test_gpu_scaling_investigation.py`.
- `yinsh_ml/tests/test_heuristic_selfplay.py` — `@pytest.mark.skip` on the
  `MCTSPolicy`/`AdaptivePolicy` tests (AdaptivePolicy builds MCTSPolicy); kept all
  HeuristicPolicy/quality/factory coverage.

**Test result:** full suite green vs baseline — no new failures introduced; the
drop in passing count is exactly the deleted dead-engine tests. Pre-existing,
MCTS-unrelated failures remain (tensor-pool / zero-copy / memory-pool macOS
backend issues; a `StubGameState` mismatch in `test_heuristic_agent`).

## 3. (Correction to the original draft of this doc)

An earlier draft claimed `use_batched_mcts=False` routed to the *broken* `search()`
— **wrong.** That branch calls `self_play.MCTS.search()` (the working singleton
path, parity-tested against `search_batch`). Only `search/mcts.py::MCTS.search()`
was broken, and it's now deleted. No landmine there.

## 4. Follow-ups (minor, not urgent)

- The dead engine's `epsilon_mix` / backprop tests are gone; equivalent logic on the
  surviving `self_play.MCTS` *is* still covered (those test files test both engines
  and kept the self_play side). No coverage of the real engine was lost.
- If a real MCTS profiler is ever wanted, write one against `self_play.MCTS`
  (the deleted `performance_profiler.py` only knew the dead API).
- **TensorPool device-string mismatch on MPS (discovered 2026-05-23):**
  `TensorPool._detect_default_device` returns `torch.device('mps')` (no index)
  but a tensor allocated on that device records as `torch.device('mps:0')`,
  so `_get_tensor_key` writes `'mps:0'` into the pool key while `get(device=None)`
  queries with `'mps'`. The `device_str` filter in `_find_compatible_tensor`
  (and the exact-match key lookup in `get`) consequently never matches on
  MPS — **the pool effectively allocates fresh every call on Apple Silicon.**
  Pre-existing, harmless to correctness, missed perf opportunity. Fix by
  normalizing both sides through `torch.device(str).type + ':0'` (or always
  call `.cuda(0)` / `.mps(0)` style at storage). Out of scope for the
  reshape-warning fix that surfaced it.
- **TensorPool API drift (pre-existing):** 23 tests in `test_tensor_pool.py`
  call `pool.get_tensor` / `pool.return_tensor` / etc. (old API); the
  production API is `pool.get` / `pool.release`. Tests fail uniformly with
  `AttributeError: 'TensorPool' object has no attribute 'get_tensor'`. Tests
  need to be ported; production code is unaffected.

## 5. Process note (the durable lesson)

The bug surfaced *only* because the yardstick build included a **positive control**
(known-weaker checkpoints expected to read WEAKER), not just a negative control
(same-strength expected ~0.5). The negative control passed cleanly on broken code.
When building any measurement instrument, include a known-*different* control, not
only a known-same one — a flat "all inconclusive" is the signature of a dead
instrument.
