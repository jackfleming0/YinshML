# Bitboard port — follow-up bottlenecks

> Self-contained plan for a new session. Paste this whole file as the
> first message, or `cat` it. Don't read prior conversation context —
> this doc has everything.

## Where we are

The bitboard engine port (branch `bitboard-port`) shipped and is
passing parity tests. As of **2026-05-01 with Candidate A landed**,
single-game cProfile on cloud 4090 + Linux x86 at sim=400 shows:

| profile (single-game cProfile, sim=400) | wall time |
|---|---:|
| Pre-bitboard-port (Python engine) | ~340s* |
| Bitboard port shipped | 235s |
| Bitboard port + Candidate A (heuristic eval cache) | **110s** |

(\* estimated from the 1.46× steady-state ratio — the Python baseline
hasn't been re-profiled since A landed; both engines benefit from A
since the heuristic is engine-agnostic.)

**Cumulative wall-clock reduction: ~68% from the pre-port baseline,
53% from the bitboard-port baseline.** Brief's 5× target is no longer
the right framing; recipe cost is what matters and we've cut it
substantially.

## Candidate A: LANDED 2026-05-01

**Measured outcome on cloud 4090:**
- Cache hit rate: **95.8%** (30,025 hits / 31,335 calls per game)
- `evaluate_position` cum time: 146s → **4.46s** (97% reduction)
- Total game time: 235s → **110s** (53% wall-clock reduction)
- Beat the projection (37%) by ~16 percentage points because MCTS
  revisits positions far more aggressively than the simulated
  benchmark predicted (60% projected vs 95.8% actual).

Implementation: cache `evaluate_position` results by
`(position_fingerprint, white_score, black_score, move_count, player)`.
Position fingerprint is Zobrist for Python `GameState` and a direct
bitboard tuple for `CppGameState`. See commits `f6d3617`, `7690cff`,
`09beec3`. Parity tests in `yinsh_ml/heuristics/test_eval_cache.py`.

## Profile after Candidate A (2026-05-01, cloud 4090, 110s/game)

The original heuristic-dominated profile is gone. New top costs:

| # | Cost center | Self | Cum | What |
|--:|---|---:|---:|---|
| 1 | **MCTS `_select_action` (UCB)** | **24.5s** | 50.2s | 201k pure-Python tree walks |
| 2 | **`Move.__hash__` chain** (Move + enum + recursive `builtins.hash`) | **~30s** | ~30s | 12M Move hashes/game from MCTS dict ops |
| 3 | `_evaluate_and_backup_batch` (incl. NN forward) | 3.7s | 57.8s | The whole MCTS leaf-eval batch |
| 4 | `cpp_move_to_py` | 6.9s | 20.2s | C++ wrapper conversion |
| 5 | NN `predict_batch` (incl. `torch.conv2d`) | 0.8s | 19.9s | The GPU floor |
| 6 | `cell_to_position` / `position_to_cell` | ~5s | ~9s | Wrapper conversion helpers |
| 7 | Encoder `encode_state` | 1.4s | 11.2s | Stable; not worth touching |
| 8 | `evaluate_position` (cached) | 0.05s | 4.5s | DONE |

Heuristic features (`connected_marker_chains`, `tactical_patterns`)
are no longer in the top 30. **Old Candidate B is dead.**

## Candidate ranking after A

The new bottlenecks are MCTS internals (UCB walk + Move hashing), not
the heuristic. Rerank in `(expected_gain / implementation_cost)`:

### Candidate A' (NEXT, top priority): cache `Move.__hash__`

**Hypothesis.** `Move` is a dataclass; its `__hash__` recursively
hashes `MoveType` (enum), `Player` (enum), `Position` (dataclass with
column/row), and an optional markers tuple every call. 12M of these
per game = ~30s cumulative. MCTS uses Move objects as dict keys on
every child lookup and edge update, so the same Move's hash is
recomputed thousands of times.

**Fix.** Add a `_hash` slot to `Move` (and `Position` if it's also
hot). Compute on first `__hash__` call, return cached int thereafter.
Move is treated as immutable everywhere — caching is safe.

**Expected gain.** ~25s of 110s = **~23% wall-clock** if hash chain
shrinks to a single int read. Conservatively ~15% even if the constant
overhead is bigger than expected.

**Risk.** Low. Hash semantics preserved (same input → same int);
equality unchanged. A parity test that constructs many Move objects
and asserts `hash(a) == hash(b) iff a == b` covers the invariant.

**Implementation.**
1. Read `yinsh_ml/game/types.py::Move` to confirm dataclass shape.
2. Add `_hash: Optional[int]` slot, lazy-compute in `__hash__`.
3. Check whether `Position` deserves the same treatment.
4. Run `pytest yinsh_ml/heuristics/ yinsh_ml/tests/test_game_logic.py
   yinsh_ml/tests/test_move_encoder.py`.
5. Re-run profile; expect `Move.__hash__` to drop out of top-10.

### Candidate B' (after A'): vectorize MCTS `_select_action`

**Hypothesis.** `_select_action` iterates child stats in pure Python
to compute UCB and pick argmax — 24.5s self-time across 201k calls.
At ~80 children per state, that's ~120µs per call, dominated by
Python loop overhead.

**Fix.** Store children's `(visit_count, prior, q_value)` as numpy
arrays inside the node. Compute UCB as a vectorized op; argmax is one
numpy call.

**Expected gain.** 30-40s cum savings = **~25% wall-clock**. Effort:
~3-4 hours including careful regression testing.

**Risk.** Medium. MCTS correctness sensitive to numerical details
(NaN handling, FPU reduction, Dirichlet noise mix-in). Diff smoke
test against current MCTS over a few hundred games — same final
move distribution.

### Candidate C (was old Candidate 2): move-list conversion

**Hypothesis.** `cpp_move_to_py` (6.9s self / 20.2s cum) +
`cell_to_position` (~5s self / ~9s cum) = ~30s cum from C++→Python
move conversion alone.

**Status.** Now ~14% of game time after A. Worth picking up after A'
and B' if we re-profile and it's still in the top 5.

### Dropped candidates

- **Old Candidate B (port heuristic features to C++)** — DROP.
  `connected_marker_chains` was 80s/game pre-cache; post-cache it's
  ~2s/game. Not worth a C++ port.
- **Old Candidate 1 (encoder fast-path)** — already dropped, still dropped.
- **Old Candidate 3 (move history defer)** — already dropped.

## Step 0 — profile first (always)

```bash
# On the cloud box (4090, .so built):
git pull
python scripts/profile_cpp_self_play.py --sims 400          # steady-state target
python scripts/profile_cpp_self_play.py --sims 48 --tag warmup   # the 1.2× case
```

Re-profile after every candidate lands. The plan keeps getting
overturned by data — only profile-driven choices have stuck.

## Stop conditions

- If A' lands and the ratio is ≥2.5× steady-state vs the Python
  engine baseline, **stop and ship**. Recipe cost is well within
  acceptable range.
- If A' wall-clock win is <10%, Move hashing wasn't the bottleneck I
  thought — re-profile and check whether the work moved elsewhere
  (e.g. dict ops directly, not hashing).
- If profiling shows the new dominant cost is `torch.*` GPU
  primitives, stop. That's the GPU floor + Amdahl's law on NN.

## Validation strategy

For every candidate landed:

1. **Parity tests pass.** `pytest yinsh_ml/heuristics/`,
   `yinsh_ml/tests/test_game_logic.py`,
   `yinsh_ml/tests/test_move_encoder.py`,
   `yinsh_ml/tests/test_heuristic_integration.py` stay green. Add a
   new parity test for any non-trivial behavioral change.
2. **Paired stress run.** Run `cloud_smoke_cpp_stress.yaml` and
   `cloud_smoke_py_stress.yaml`, compare iter-2 self-play time. The
   ratio number goes in the commit message.
3. **Regression catch.** Re-run sim=48 profile; over-optimizing the
   high-sim path can sometimes regress the low-sim path.

## Out of scope

- **Port MCTS to C++.** Yngine has lock-free MCTS. Weeks of work,
  separate brief.
- **Change the move encoding scheme.** 7433-slot scheme is fixed by
  the network's policy head.
- **Change the network architecture.** Separate workstream.
- **Modify game semantics in `yinsh_ml/game/`.** Reference
  implementation; semantic changes break parity-test foundation.
  Non-semantic changes (e.g. caching `__hash__` for Candidate A') are
  fine.

## Repo state

- Branch: `bitboard-port`.
- Build: `pip install -e . && python setup.py build_ext --inplace`.
- Sanity: `pytest yinsh_ml/game_cpp/tests/ -q` should report ~140
  passing.
- The `.so` is gitignored.
- 43 pre-existing test failures (down to 38) are unrelated to this
  plan; see commits `c9c83a8`, `539b9ee` for context.
