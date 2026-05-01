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
| + Candidate A (heuristic eval cache) | 110s |
| + Candidate A' (Move.__hash__ cache) | **95s** |

(\* estimated from the 1.46× steady-state ratio — the Python baseline
hasn't been re-profiled since A landed; both engines benefit from A
since the heuristic is engine-agnostic.)

**Cumulative wall-clock reduction: ~72% from the pre-port baseline,
60% from the bitboard-port baseline.** Brief's 5× target is no longer
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

## Candidate A': LANDED 2026-05-01

**Measured outcome on cloud 4090:**
- `Move.__hash__` cum: 29.5s → **8.8s** (-21s recursive chain savings)
- `builtins.hash` self: 25.5s → 3.3s
- `enum.__hash__` self: 5.2s → 0.6s
- Total game time: 110s → **95s** (14% additional wall-clock; net 60%
  vs bitboard-port baseline)
- Cache `dict.get` overhead: ~2s/game (acceptable)

Implementation: lazy-memoize `Move.__hash__` in `__dict__['_hash']`,
strip on pickle (per-process `PYTHONHASHSEED`). 11 parity tests in
`yinsh_ml/tests/test_move_hash_cache.py`. See commits `a3cfc9a`,
`cdc63c1`.

## Profile after Candidate A' (2026-05-01, cloud 4090, 95s/game)

| # | Cost center | Self | Cum | What |
|--:|---|---:|---:|---|
| 1 | **MCTS `_select_action` (UCB)** | **20.4s** | 24.5s | 152k pure-Python tree walks (cum dropped from 50s with hash cache) |
| 2 | `cpp_move_to_py` | 8.5s | 24.9s | C++ wrapper conversion (Candidate C) |
| 3 | `cell_to_position` | 4.4s | 9.6s | Conversion helper |
| 4 | `Move.__hash__` (cached) | 4.06s | 8.8s | Self unchanged (still 12M calls), cum collapsed from 29.5s |
| 5 | `move_to_index` | 4.5s | 6.2s | Encoder side |
| 6 | `_evaluate_and_backup_batch` | 4.6s | 71.5s | Includes NN + select_action |
| 7 | `predict_batch` | 0.9s | 24.1s | GPU floor |
| 8 | `encode_state` | 1.8s | 13.7s | Stable |
| 9 | `evaluate_position` (cached) | 0.06s | 4.7s | DONE |

`_select_action` is now the unambiguous #1 self-time hotspot.
Heuristic features remain absent from the top 30. **Candidate B'
(vectorize UCB) is the obvious next move.**

## Candidate ranking after A

The new bottlenecks are MCTS internals (UCB walk + Move hashing), not
the heuristic. Rerank in `(expected_gain / implementation_cost)`:

### Candidate B' (NEXT, top priority): vectorize MCTS `_select_action`

**Hypothesis.** `_select_action` iterates child stats in pure Python
to compute UCB and pick argmax — 20.4s self-time across 152k calls
post-A'. At ~80 children per state, that's ~130µs per call dominated
by per-iter `np.sqrt` / `np.random.uniform` C-call overhead.

**Fix.** Materialize children's `(visit_count, value_sum,
virtual_losses, prior_prob)` into numpy arrays per call (c_puct is
constant per MCTS instance, take from `self.c_puct`). Compute Q-vector
+ U-vector + ε-noise vector with vectorized ops; argmax once.
Preserve the FPU branch: visited children use `child.value()`,
unvisited use `q_parent_pov - fpu_reduction × √(visited_policy_sum)`.
Preserve the U formula split: visited uses `c_puct × prior × √parent
/ (1 + visits)`, unvisited uses `c_puct × prior × √parent` (no
division).

**Expected gain.** Materialization ~5-10µs per call; vectorized math
~5µs. Total ~10-15µs vs current 130µs → ~15s wall-clock savings.
**~16% wall-clock**.

**Risk.** Medium. UCB selection is numerically sensitive — wrong
formula breaks training silently (slow regression in policy quality
that only shows up after many iterations). Mitigations:
1. Numerical-parity test: build many synthetic nodes with random
   stats; assert vector and scalar implementations pick the same
   move with the same numpy RNG state.
2. Keep the scalar implementation under a fallback flag for at least
   one cycle so we can A/B compare on the cloud if needed.
3. Run heuristic_mcts_performance + selfplay tests before commit.

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

### A' is LANDED

Hash cache shipped 2026-05-01 (commits `a3cfc9a`, `cdc63c1`). See the
"Candidate A': LANDED" section above for measured outcome.

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
