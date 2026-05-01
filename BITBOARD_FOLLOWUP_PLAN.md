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

The engine layer is no longer the dominant cost — when sim=400 on a
4090, per-move time is 290 ms (C++) vs 480 ms (Python). The 190 ms
gap is what we saved. The remaining 290 ms is now the new headroom.

This doc proposes the next ~3 attacks to keep pushing the ratio up,
ranked by expected gain per hour of work.

## Step 0 — profile first (don't skip)

Bottleneck claims below are hypotheses from the wrapper code, not
measured. **Validate before optimizing.** Profile a representative
training iteration with the C++ engine and rank actual self-time:

```bash
# On the cloud box (or any with the .so built):
git checkout bitboard-port
git pull

# Make sure the stress configs from the previous session exist, or
# regenerate from cloud_smoke.yaml:
python -c "import yaml; cfg=yaml.safe_load(open('configs/cloud_smoke.yaml')); sp=cfg.setdefault('self_play',{}); sp['use_cpp_engine']=True; sp['num_simulations']=400; sp['late_simulations']=400; yaml.safe_dump(cfg, open('configs/cloud_smoke_cpp_stress.yaml','w'), sort_keys=False)"

# Profile a single worker run (no multiprocessing — keeps cProfile clean):
python -c "
import cProfile, pstats, os, tempfile, torch
torch.set_num_threads(1)
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.training.self_play import play_game_worker

network = NetworkWrapper(device='cuda')
with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
    torch.save(network.network.state_dict(), tmp.name)
    model_path = tmp.name

cfg = dict(evaluation_mode='hybrid', heuristic_evaluator=None, heuristic_weight=0.5,
           num_simulations=400, late_simulations=400, simulation_switch_ply=20,
           fast_simulations=0, fast_sim_prob=0.0, c_puct=1.0, dirichlet_alpha=0.3,
           value_weight=1.0, max_depth=200, initial_temp=1.0, final_temp=0.1,
           annealing_steps=30, temp_clamp_fraction=0.6, use_batched_mcts=True,
           mcts_batch_size=32, use_enhanced_encoding=False, enable_subtree_reuse=True,
           fpu_reduction=0.25, epsilon_mix_start=0.25, epsilon_mix_end=0.0,
           epsilon_mix_taper_moves=20, root_policy_temp=1.0, use_cpp_engine=True)

pr = cProfile.Profile(); pr.enable()
play_game_worker(model_path=model_path, game_id=0, mcts_config=cfg)
pr.disable(); os.unlink(model_path)
pstats.Stats(pr).sort_stats('tottime').print_stats(30)
"
```

Save the top-30 self-time breakdown. Anything in `yinsh_ml/utils/encoding.py`,
`yinsh_ml/game_cpp/_convert.py`, or `yinsh_ml/network/wrapper.py` is in
scope for this plan. Anything in `torch.*` is not — that's the GPU NN
floor and a separate problem.

## Candidate 1: state encoding — direct bitboard→numpy

**Hypothesis.** `StateEncoder.encode_state` walks `board.pieces` once
per move per sim leaf, and the lazy materialization in
`CppBoard.pieces` iterates 85 cells × 4 bitboard checks each time. At
400 sims/move that's 400 × ~85 cells × 4 checks = 136K Python ops per
move just to fill the 4 piece-channel grid that the C++ side already
has in 4 bitboards.

**Fix.** Add a C++ binding `_engine.write_state_channels(state, out)`
that takes a pre-allocated `np.ndarray((4, 11, 11), dtype=float32)`
and writes the 4 piece channels directly via popcount-iter. Then
either:
  (a) Have `CppBoard.to_numpy_array` call it directly, OR
  (b) Add a fast path in `StateEncoder.encode_state` that recognizes
      `CppGameState` / `CppBoard` and bypasses the dict walk.

**Expected gain.** On a 290 ms/move budget, saving even 30 ms/move
of encoder time is a ~10% wall-clock improvement. Could be more if the
encoder is called per-leaf (vs per-move).

**Risk.** Low. Same numpy layout, same channel ordering — pure
constant-factor optimization. Verify with a parity test that the
fast-path output bit-equals the slow-path output across 1000 random
boards.

**Implementation steps.**
1. `yinsh_ml/game_cpp/src/bindings.cpp` — add
   `WriteStateChannels(state, py::array_t<float> out)` that asserts
   shape `(4, 11, 11)`, then for each of 4 bitboards iterates set
   bits via `__builtin_ctzll` and writes `1.0` to the right
   `(channel, row, col)`.
2. Bind it in `bindings.cpp::PYBIND11_MODULE` as
   `m.def("write_state_channels", ...)`.
3. Override `CppBoard.to_numpy_array()` to allocate the array and call
   the binding.
4. Add a parity test: random board → both Board.to_numpy_array() and
   CppBoard.to_numpy_array() produce equal arrays.
5. Re-run the paired stress benchmark; iter 2 ratio should improve.

## Candidate 2: move-list conversion — keep moves on the C++ side

**Hypothesis.** `CppGameState.get_valid_moves()` calls
`cpp_move_to_py(cm)` for every legal move. Profiling on Mac CPU
showed ~50K cpp_move_to_py calls per game (300 plies × ~80 valid
moves per ply / 2 — the `/2` is from MCTS reusing). Each call
allocates a Python `Move`, a `Position`, and possibly a `tuple` of
markers. That's ~50K-150K Python object allocations per game just for
the wrapper.

**Fix.** Change the contract: `CppGameState.get_valid_moves()` returns
the C++ `_engine.Move` list directly, not Python `Move` objects.
Callers that need a Python `Move` (e.g. for `state_encoder.move_to_index`)
convert lazily — but most callers only iterate, compare, and pick
one, which can be done on `_engine.Move` without conversion.

This is invasive — `state_encoder.move_to_index` accepts a Python
`Move` today. Either:
  (a) Add a parallel `move_to_index_cell(c_move)` on StateEncoder that
      takes the C++ move directly, OR
  (b) Add a `_engine.Move.to_index()` method on the C++ side that
      reproduces MoveEncoder's logic in C++ (bigger but faster).

(b) is the bigger win and the bigger lift.

**Expected gain.** Hard to estimate without profile data. If
cpp_move_to_py is >5% of self-time in the profile, this is worth it.
Could be 5-15% wall-clock.

**Risk.** Medium. Move encoding (the 7433-slot scheme) is rule-
correctness-sensitive; mistakes here corrupt training data silently.
A bit-exact parity test against MoveEncoder.move_to_index is mandatory.

**Implementation steps.**
1. Profile to confirm cpp_move_to_py is in top-10 self-time.
2. Pick (a) or (b) based on profile evidence.
3. For (b): port MoveEncoder.move_to_index to C++. ~150 lines, but
   the input is a small enum + a few ints. Validate via round-trip
   test: every `_engine.Move.to_index()` matches
   `MoveEncoder.move_to_index(cpp_move_to_py(cm))` for 100K random
   moves.
4. Update CppGameState.get_valid_moves to either return C++ moves
   directly or batch-convert.

## Candidate 3: move history — defer materialization

**Hypothesis.** Every `CppGameState.make_move(move)` appends to
`self._move_history`, which holds Python `Move` objects. These are
allocated per move, and on `copy.deepcopy(state)` (still happening for
`state_pool=None` MCTS runs) the list of Move objects gets copied
element-wise.

**Fix.** Store move history as a list of (move_type, source_cell,
dest_cell, marker_cells) tuples — primitives only. Materialize Python
`Move` objects only on `state.move_history` access (lazy).
Alternatively: don't store history at all in MCTS sim states. The
production MCTS doesn't read move_history during search, only at
game-end. We could expose `move_history` as `[]` during sims and
populate on the real GameState.

**Expected gain.** Smaller than the other two. Move history is ~100
items per game; the cost is dominated by deepcopy (which we already
killed for the bitboard portion). The wrapper's `copy()` does
`list(self._move_history)` which is shallow — already cheap. So
candidate 3 might be a 1-2% win.

**Risk.** Low. As long as `state.move_history` returns a list of
Python Move objects on access, callers don't notice.

**Implementation steps.**
1. Profile-confirm move_history is in top-15 self-time. If not, skip
   this candidate.
2. Convert `_move_history` to `list[tuple]` storing primitives.
3. Make `move_history` a property that materializes on access. Cache
   the materialization until next make_move.

## Recommended order

1. **Profile.** ~30 minutes. Top-30 self-time on cProfile. Decide
   based on data.
2. **Candidate 1 (encoder).** Highest confidence, lowest risk, biggest
   expected gain. ~3-4 hours.
3. **Re-profile.** Confirm encoder dropped out of top-15.
4. **Candidate 2 (move conversion).** Only if cpp_move_to_py / Move
   allocations are still a top-10 contributor. ~6-8 hours for option
   (b), ~2 hours for option (a).
5. **Re-profile.** Decide if Candidate 3 is worth it.
6. **Candidate 3 (move history).** Probably skip. ~2 hours if
   pursued.

Total budget: 8-15 hours of focused work for an additional ~1.3-1.6×
on top of the current 1.46× steady-state, taking the ratio toward
~2-2.5×. The brief's 5× target is unlikely without bigger structural
changes (batched MCTS in C++, lock-free tree, etc. — yngine territory
and out of scope).

## Stop conditions

- If Candidate 1 lands and the new ratio is ≥2× steady-state, ship it.
  Recipe cost reduction is now substantial enough to justify the work.
- If Candidate 1 is implemented and the speedup is <5%, the encoder
  was not the bottleneck — re-profile and reconsider candidates 2/3.
- If profiling shows the new dominant cost is `torch.*` GPU primitives,
  stop. The remaining gap is GPU floor + Amdahl's law on NN.

## Validation strategy

For every candidate landed:

1. **Parity tests pass.** `pytest yinsh_ml/game_cpp/tests/` stays
   green. Any change to C++ output shape / numpy layout / move encoding
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
