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
| + Candidate A' (Move.__hash__ cache) | 95s |
| + Candidate B' (vectorize UCB) | **85s** |

(\* estimated from the 1.46× steady-state ratio — the Python baseline
hasn't been re-profiled since A landed; both engines benefit from A
since the heuristic is engine-agnostic.)

**Cumulative wall-clock reduction: ~75% from the pre-port baseline,
64% from the bitboard-port baseline.** Brief's 5× target is no longer
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

## Candidate B': LANDED 2026-05-01

**Measured outcome on cloud 4090:**
- `_select_action` self: 20.4s → **10.6s** (-48%)
- `_select_action` cum: 24.5s → **14.9s** (-39%)
- Per-call cost: 130µs → 76µs (less than the projected 10µs — the
  Python materialization loop over `node.children` is now the gating
  factor; further wins require storing stats as numpy arrays directly
  in `Node`, a bigger refactor)
- Total game time: 95s → **85s** (10% additional wall-clock; net 64%
  vs bitboard-port baseline)

Implementation: materialize children's `(visit_count, value_sum,
virtual_losses, prior_prob)` into 4 numpy arrays per call, compute
Q+U+ε vectorized, argmax once. Preserves the visited/unvisited UCB
formula split exactly. 7 parity tests in
`yinsh_ml/tests/test_select_action_vector.py` covering 192 random
nodes (varied visited fractions, child counts 2-80) plus 4 edge
cases. See commits `3bb42ff`, `b80e28d`.

## Profile after Candidate B' (2026-05-01, cloud 4090, 85s/game)

| # | Cost center | Self | Cum | What |
|--:|---|---:|---:|---|
| 1 | `_select_action` (vectorized) | 10.6s | 14.9s | Halved by B' but still #1; further wins need stats-as-arrays in Node |
| 2 | **`cpp_move_to_py`** | **8.5s** | 24.9s | C++ → Python Move conversion, 3.85M calls |
| 3 | `Position.__init__` (`<string>:2`) | 5.5s | 6.0s | dataclass __init__, 3.85M calls — the auto-generated Move ctor |
| 4 | **`cell_to_position`** | **4.4s** | 9.9s | 7.17M calls, only 99 unique outputs — top cache target |
| 5 | `move_to_index` | 4.5s | 6.2s | Encoder side, Python Move → int |
| 6 | `_evaluate_and_backup_batch` | 4.5s | 70.6s | Includes NN + select + materialization |
| 7 | `Move.__hash__` (cached) | 3.0s | 7.2s | Cache-hit `dict.get` — could be slot-based for ~2s more |
| 8 | `predict_batch` | 0.9s | 23.4s | GPU floor |
| 9 | `encode_state` | 1.7s | 13.6s | Stable |
| 10 | `evaluate_position` (cached) | 0.06s | 5.3s | DONE |

The C++ wrapper conversion path now dominates yinsh_ml self-time:
`cpp_move_to_py` + `cell_to_position` + `Position.__init__` +
`move_to_index` ≈ 23s self-time across ~4M calls per game. **Candidate
C territory.**

## Candidate ranking after A

The new bottlenecks are MCTS internals (UCB walk + Move hashing), not
the heuristic. Rerank in `(expected_gain / implementation_cost)`:

### Candidate C-1 (NEXT, top priority): cache `cell_to_position`

**Hypothesis.** 7.17M `cell_to_position` calls per game, each
constructing a fresh `Position` dataclass — but there are only 99
valid YINSH positions (121 with the 11×11 lattice). Same Position
returned repeatedly is wasted work.

**Fix.** Build a 121-slot list of `Position` instances at module load
in `yinsh_ml/game_cpp/_convert.py`. `cell_to_position(cell)` becomes
a `_POSITION_BY_CELL[cell]` lookup. Position is `@dataclass(frozen=True)`
so sharing instances across callers is safe.

**Expected gain.** `cell_to_position` self drops 4.4s → ~0.5s; the
dataclass `Position.__init__` cost (currently scattered through
`<string>:2(__init__)`) drops to zero for cached lookups. Combined
estimated savings ~6-7s wall-clock = **~7-8%**.

**Risk.** Very low. Same `Position` values returned (verified by
parity test against the legacy divmod implementation across all 121
cells). Frozen dataclass — instance sharing is safe.

**Cost.** ~30 minutes including the parity test.

### Candidate C-2 (after C-1): port move conversion to C++ or skip Python Move objects entirely

**Hypothesis.** `cpp_move_to_py` (8.5s self) constructs a Python
`Move` for every move enumerated, even though MCTS often only needs
the C++ representation. `move_to_index` (4.5s self) then takes that
Python Move and converts to an integer for policy indexing.

**Fix.** Either (a) add `_engine.Move.to_index()` in C++ and skip
Python Move materialization entirely on the encoder path, or (b)
keep moves as opaque C++ tokens through MCTS and only convert at
state-encoder boundaries.

**Expected gain.** ~10-15s wall-clock = ~12-18%. Bigger lift than
C-1 — defer until C-1 lands and we re-profile.

### Candidate B'' (parking lot): stats-as-arrays in Node

**Hypothesis.** `_select_action` post-B' is gated by the
materialization loop over `node.children`. Storing children's stats
as numpy arrays directly in the parent Node would eliminate that
loop entirely; selection becomes a few vectorized ops on
already-allocated arrays.

**Cost / risk.** Significant Node refactor — touches expansion,
backpropagation, virtual loss tracking, subtree reuse. Not worth it
unless we exhaust C-1 / C-2.

<!-- B' was here pre-2026-05-01; LANDED, see "Candidate B': LANDED"
section above for the measured outcome. -->

<!-- Old Candidate C split into C-1 and C-2 above; the C-1 sub-task
(cache cell_to_position) is the new top priority. -->


### Dropped candidates

- **Old Candidate B (port heuristic features to C++)** — DROP.
  `connected_marker_chains` was 80s/game pre-cache; post-cache it's
  ~2s/game. Not worth a C++ port.
- **Old Candidate 1 (encoder fast-path)** — already dropped, still dropped.
- **Old Candidate 3 (move history defer)** — already dropped.

### A' and B' both LANDED

See "Candidate A': LANDED" and "Candidate B': LANDED" sections above
for measured outcomes.

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
