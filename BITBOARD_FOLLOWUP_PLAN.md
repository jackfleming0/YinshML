# Bitboard port — follow-up bottlenecks

> Self-contained plan for a new session. Paste this whole file as the
> first message, or `cat` it. Don't read prior conversation context —
> this doc has everything.

## Where we are

The bitboard engine port (branch `bitboard-port`, ~6 commits) shipped
and is passing parity tests. Speedups measured on cloud 4090 + Linux
x86 vs the Python engine:

| benchmark | speedup |
|---|---:|
| Random playouts (engine-only, no NN) | 332× |
| State.clone() vs copy.deepcopy(GameState) | ~134,000× |
| MCTS self-play, sim=48, GPU NN | 1.2× |
| MCTS self-play, sim=400, GPU NN, **steady-state** | **1.46×** |

Real-world recipe cost reduction is ~35%. Below the brief's 5× target.

The engine layer is no longer the dominant cost. Per-move time at
sim=400 on a 4090 is 290 ms (C++) vs 480 ms (Python). The 190 ms gap
is what we saved. The remaining 290 ms is the new headroom — and as
the profile in the next section shows, the dominant cost is no longer
in the engine wrapper at all.

## Profile results (2026-05-01, cloud 4090)

A single-game cProfile run with `--sims 400` (`scripts/profile_cpp_self_play.py`)
broke down the 235s/game self-time as follows. **The plan's original
hypotheses were wrong** — encoder is 5.7%, not the bottleneck; the
heuristic evaluator is 62%.

| # | Cost center | Cum time | % of game | What |
|--:|---|---:|---:|---|
| 1 | **`YinshHeuristics.evaluate_position`** | **146s** | **62%** | Per-leaf heuristic eval (37k calls) |
| 2 | └── `connected_marker_chains` / `_find_longest_chain` / `dfs_chain_length` | 80s | 34% | Python DFS over markers |
| 3 | └── `tactical_patterns.detect_immediate_*` | 41s | 17% | Near-complete row pattern matching |
| 4 | MCTS `_select_action` (UCB) | 62s | 26% | Pure-Python tree walk |
| 5 | C++ wrapper `cpp_move_to_py` + `get_piece` + `cell↔pos` | ~75s | ~32%* | Plan original Candidate 2 |
| 6 | Python hashing (`Position.__hash__`, enum hash) | ~50s | ~21%* | 80M `builtins.hash` calls |
| 7 | NN `predict_batch` (forward + GPU) | 24s | 10% | The "GPU floor" |
| 8 | `encode_state` | 13s | 5.7% | Plan original Candidate 1 — much smaller than guessed |

(\* nested with #1, percentages overlap)

**The headline:** the heuristic costs 6× more than the neural network
(146s vs 24s) per game. For a `hybrid` mode that's supposed to be a
50/50 blend, the cost ratio is wildly out of balance. The C++ engine
port already extracted most of the engine-layer wins; what's left is
overwhelmingly the Python heuristic.

## Updated candidate ranking

The plan was originally written from wrapper-code intuition. Profile
data overrides that. Updated ranking, in order of `(expected_gain /
implementation_cost)`:

### Candidate A (NEW, top priority): cache `evaluate_position` by Zobrist hash

**Hypothesis.** 37,248 evaluate_position calls per game; MCTS visits
many positions multiple times via tree exploration and transpositions.
Existing transposition-table evidence suggests 60–80% hit rates on
position keys. Caching the heuristic by `(zobrist, score, move_count,
player)` should slash 50–70% off the heuristic cost.

**Fix.** Inside `YinshHeuristics`, add a dict keyed by
`(position_fingerprint, white_score, black_score, move_count,
player_value)` mapping to the cached float result. Position
fingerprint is the Zobrist hash for Python `GameState` and a
bitboard-tuple fast-path for `CppGameState` (avoids triggering
`CppBoard.pieces` materialization which itself is non-trivial). Cache
size capped, FIFO drop on overflow. `clear_cache()` for between-game
reuse if the evaluator outlives a single game.

**Expected gain.** If hit rate matches the existing transposition
table at 60%, savings = 0.60 × 146s = ~88s of 235s per game = **~37%
wall-clock**. Even at a conservative 40% hit rate, ~25% wall-clock.
Should push the steady-state speedup ratio from 1.46× toward 2.0×+.

**Risk.** Low. Cache key includes every state input the heuristic
reads from; a parity test verifies cached vs. uncached results are
bit-equal across many random positions. Worst case if logic is wrong:
training data is stale, caught by the parity test.

**Implementation.**
1. Refactor `evaluate_position` body into `_evaluate_position_uncached`.
2. Make `evaluate_position` a cache-aware wrapper.
3. Add `_position_fingerprint(state)` helper handling both engine paths.
4. Add `clear_cache()` and `cache_stats()` for visibility.
5. Add parity test: 200 random states, cached == uncached.
6. Re-run profile; expect `evaluate_position` cum time to drop ~50%+.

### Candidate B (was #2): port hot heuristic features to C++

**Hypothesis.** `connected_marker_chains` (80s/game) is a Python DFS
over markers. Connected-component on a bitboard is O(popcount + a few
shifts) in C++ — should be ~50–100× faster.
`tactical_patterns._find_near_complete_rows` (41s/game) is similarly
amenable to bitboard tricks.

**Fix.** Add `_engine.connected_chain_length(markers_bb)` and
`_engine.find_near_complete_rows(markers_bb)` bindings. Wire through
features.py / tactical_patterns.py to dispatch on isinstance(state.board,
CppBoard).

**Expected gain.** Stacks on top of A, since A's cache misses still
pay this cost. If A reaches 60% hit rate, miss-path heuristic is 0.4
× 121s = 48s; B could halve that = ~10% additional wall-clock.

**Risk.** Medium. Connected-component on a hex bitboard requires
correct neighbor masks. Parity test against the Python implementation
is mandatory.

**Cost.** ~6–10 hours.

### Candidate C (was Candidate 2): move-list conversion

**Hypothesis.** `cpp_move_to_py` self-time is 11s/game; cell↔position
conversion adds another ~10s. ~9% wall-clock if both go.

**Status.** Demoted from #2 to #3. Worth doing, but A and B should
land first.

### Demoted / dropped

- **Original Candidate 1 (encoder fast-path)** — DROP. Encoder is
  5.7% of game; full elimination is well below the doc's "10%+"
  estimate. Skip unless A and B both land and re-profile shows
  encoder rising in the rankings.
- **Original Candidate 3 (move history defer)** — DROP. Not in
  top-30 self-time. Skip definitively.

## Step 0 — profile first (don't skip)

The first time this plan ran (2026-05-01) it produced the table
above. Re-run after every candidate lands so the next decision is
data-driven, not intuition-driven.

```bash
# On the cloud box (4090, .so built):
git pull
python scripts/profile_cpp_self_play.py --sims 400          # steady-state target
python scripts/profile_cpp_self_play.py --sims 48 --tag warmup   # the 1.2× case
```

Expect ~90–280s wall time per run (235s on the 2026-05-01 baseline).
The script prints top-30 by tottime/cumtime/yinsh_ml-filtered and
dumps `profile_cpp_<tag>.prof` for offline pstats.

## Stop conditions

- If Candidate A lands and the new ratio is ≥2× steady-state, ship it
  and consider B/C optional. Recipe cost reduction is now substantial
  enough to justify the work.
- If Candidate A speedup is <10% measured wall-clock, the cache hit
  rate was lower than expected — investigate via `cache_stats()` and
  reconsider the key design before moving to B.
- If profiling shows the new dominant cost is `torch.*` GPU
  primitives, stop. The remaining gap is GPU floor + Amdahl's law on
  NN.

## Validation strategy

For every candidate landed:

1. **Parity tests pass.** `pytest yinsh_ml/game_cpp/tests/` and
   `pytest yinsh_ml/heuristics/tests/` stay green. Any change to a
   shared output (heuristic score, move encoding, encoder layout)
   gets a new parity test.
2. **Paired stress run.** Run `cloud_smoke_cpp_stress.yaml` and
   `cloud_smoke_py_stress.yaml`, compare iter-2 self-play time. The
   ratio number is what goes in the commit message.
3. **Regression catch.** Run a single iteration with
   `cloud_smoke_cpp.yaml` (sim=48). Make sure speedup at low sims
   didn't go negative — over-optimizing the high-sim path can
   sometimes regress the low-sim path.

## Out of scope (don't do these in this plan)

- **Port MCTS to C++.** Yngine has lock-free MCTS. Different project,
  weeks of work, will need its own brief.
- **Change the move encoding scheme.** The 7433-slot scheme is fixed
  by the network's policy head.
- **Change the network architecture.** Separate workstream.
- **Touch yinsh_ml/game/.** The Python engine remains the reference
  implementation; mutations there break parity-test foundation.

## Repo state when this plan starts

- Branch: `bitboard-port` (or whatever supersedes it after merge).
- Build: `pip install -e . && python setup.py build_ext --inplace`.
- Sanity: `pytest yinsh_ml/game_cpp/tests/ -q` should report ~140
  passing.
- The `.so` is gitignored.
- The 43 pre-existing test failures (down to 38 after the small fixes
  in this branch) are unrelated to this plan; see commits
  `c9c83a8`, `539b9ee` for context.
