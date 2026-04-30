# Bitboard port prompt — replace Python YINSH engine with a fast C++ extension

> Self-contained brief for a fresh Claude Code session. Paste this whole
> file into the new session, or `cat` it as the first message.

---

## What you're being asked to do

Port the YINSH game engine in `yinsh_ml/game/` from pure Python to a C++
extension exposed via pybind11, modeled on the bitboard design from
[temhelk/yinsh](https://github.com/temhelk/yinsh) ("yngine"). The goal
is **10–100× speedup on the engine layer**, which translates directly
to faster self-play.

This is an *engineering performance* project, not a research project.
The game rules are settled. The reference Python implementation is
correct (recently audited and re-fixed). Your job is to produce a
faster implementation that's bit-exact equivalent to the Python
reference, validated against the existing test suite plus new
property-based parity tests.

## Why this matters

Phase D wrapped: a 12-iter warm-start training run with a new
gating-revert flag. The recipe works structurally (gates fire correctly,
no compounding regression) but plateaus sub-intermediate. The recipe's
peak (`iter_3`) loses 6/24 to `HeuristicAgent(depth=3)` at deployment
budget — well below the ≥65% intermediate bar.

To reach intermediate, we need to test other recipes (more games per
iter, larger network, MCTS knob tuning, growing replay window, frontier
shifts). Each recipe experiment costs ~$2–10 + hours of cloud time.
**A 10× speedup makes ~10× more recipes affordable.** That's why this
is the priority engineering project right now.

Strategic context: `WARMSTART_PHASE_LOG.md` §9. Optional reading;
glance at §9c if you want to know what comes after this lands.

## Profile baseline (read this — it determines your priorities)

Two profile harnesses already exist:
- `scripts/profile_engine.py` — random playouts, engine-only.
- `scripts/profile_mcts_selfplay.py` — one MCTS-driven game with
  `tottime` aggregated by module.

Headline numbers (CPU, untrained net, recent local run):

| Workload | Top hotspots (self time) |
|---|---|
| Random playouts | `find_marker_rows` (24%), `valid_move_positions` (11%), `Board.get_piece` (10%), `is_valid_position` (10%), `Position.__hash__` / `dict.get` (~12%) |
| MCTS self-play (16 sims) | `copy.deepcopy` of `GameState` — **~95%** of profiled time |

**Two motivators, not one.** The bitboard port wins via:

1. **Faster engine ops.** Replacing `find_marker_rows` and friends with
   bit-twiddled ray tables is the headline 10–100× win.
2. **Near-zero state cloning.** MCTS deep-copies `GameState` per
   simulation. The current `Board.pieces: Dict[Position, PieceType]`
   is brutally expensive to deepcopy. A 4-bitboard state is ~64 bytes
   of memcpy. **This is the bigger lever for MCTS-bound self-play** —
   and falls out for free if you adopt yngine's immutable-state
   pattern (`apply_move` returns a new state).

Re-run the harnesses on your dev box before committing to the
architecture. If something other than `find_marker_rows` /
`copy.deepcopy` dominates, refocus.

## Reference: temhelk/yinsh's "yngine" design

GitHub: https://github.com/temhelk/yinsh — the C++ engine lives at
`yngine/` inside that repo (verify the path before relying). Key
design points:

- **4 bitboards per state**, one per piece type per color:
  `white_rings`, `black_rings`, `white_markers`, `black_markers` — each
  a `__uint128_t`. 121 cells fit in 128 bits with room to spare.
- **Precomputed ray tables** `TABLE_RAYS[121][6]` — for each cell index
  and each of 6 directions, a bitboard of the cells reachable along
  that ray. Instant ray lookup; no loop.
- **Move generation** iterates set bits with `bit_scan_and_reset` and
  uses ray bitboards + popcount. O(1) amortized.
- **Immutable state.** `apply_move` returns a new state; no undo
  machinery. This is what gives us the deepcopy win — adopt it.
- **No hash inside yngine.** They have no Zobrist. We *do*, and we
  keep it because `HeuristicAgent` runs alpha-beta with a transposition
  table keyed on `ZobristHasher.hash_state(state)` (5–10× win on that
  agent). Port Zobrist to C++ alongside the engine.

Files in yngine to study:
- `bitboard.hpp` / `bitboard.cpp` — bitboard ops, bit-scan, ray shifts
- `board_state.hpp` / `board_state.cpp` — state representation, move
  generation, row detection (`length_of_row`, line enumeration over
  6/7-marker maximal runs)
- `generate_tables.cpp` — how they build `TABLE_RAYS[121][6]` and the
  pre-computed shift-in-direction tables

## The Python reference you're replacing

| File | ~Lines | What it does |
|---|---|---|
| `yinsh_ml/game/constants.py` | 154 | Board geometry, `VALID_POSITIONS`, `DIRECTIONS`, `HEX_DIRECTIONS`, `HEX_LINE_AXES` |
| `yinsh_ml/game/types.py` | 59 | `Player`, `PieceType`, `MoveType`, `GamePhase` enums |
| `yinsh_ml/game/board.py` | 565 | `Board` — piece placement, ring movement (with marker flips), `find_marker_rows`, `valid_move_positions` |
| `yinsh_ml/game/moves.py` | 246 | Move enumeration per game phase |
| `yinsh_ml/game/game_state.py` | 609 | `GameState` — phase tracking, win/draw, `make_move`, `is_terminal`, `_handle_marker_removal`, `_move_maker` |
| `yinsh_ml/game/zobrist.py` | 533 | Zobrist hash — includes side-to-move + game phase |

~2200 lines of Python. The C++ port should be smaller (bitboards are
dense) — yngine's equivalent is ~1500 lines.

## Recent rule-correctness fixes — non-negotiable

Your engine must encode all the fixes documented in
**`CLAUDE.md` § Foundational Rule Fixes (April 2026)**. Headlines:

- Only the 6 valid hex line directions (no pseudo-diagonals).
- Zobrist mixes in side-to-move AND game phase (with O(1) toggle helpers).
- `find_marker_rows` returns the full maximal run (5/6/7 positions);
  move-gen enumerates length-5 windows. Yngine does this exactly the
  same way.
- `_handle_marker_removal` is atomic (validate-all then remove-all).
- Stalemate in `MAIN_GAME` handled by `is_terminal` / `get_winner`.
- Move encoding fixed at 7433 slots. **Don't change it** — the network's
  policy head depends on it. The bitboard engine doesn't own move
  encoding; just enumerate moves and let the existing encoder map them.

Contract tests — your engine must pass every one of these unchanged:

- `yinsh_ml/tests/test_game_logic.py` — core rules
- `yinsh_ml/tests/test_hex_axes.py` — direction correctness (post-bugfix)
- `yinsh_ml/tests/test_ring_movement_validation.py` — move-gen
- `yinsh_ml/tests/test_zobrist_hasher.py`,
  `test_zobrist_incremental.py`, `test_zobrist_edge_cases.py` —
  hash invariants (especially side-to-move + phase)
- `yinsh_ml/tests/test_move_encoder.py` — round-trip (engine doesn't
  own this, but verify it still passes when the engine swaps in)

If any of these fails, the rules are wrong.

## Required surface area

The Python wrapper around the C++ engine MUST expose at least:

- `clone()` — fast O(64-byte) state copy. Replaces `__deepcopy__`.
- `make_move(move) -> bool`, OR immutable `apply_move(move) -> new_state`.
  Pick one consistently and document it.
- `get_valid_moves() -> List[Move]` — same `Move` dataclass as Python.
- `is_terminal() -> bool`, `get_winner() -> Optional[Player]`.
- `zobrist_hash() -> int` — includes side-to-move + phase. Required by
  `HeuristicAgent`'s transposition table.
- `to_numpy(out: np.ndarray)` — write the 4-channel piece layout
  directly from bitboards into a pre-allocated tensor. Recommended
  (not strictly required) — `StateEncoder.encode_state` currently
  walks `board.pieces`, and a bitboard-to-numpy path is a free
  additional ~5–10%.

The Python `Move` dataclass stays as-is (don't rev `MoveType` /
`MoveEncoder`).

## Architecture proposal (revise as needed)

**Don't replace the existing Python engine. Add the C++ engine
alongside, gated by a feature flag.** The Python engine remains the
reference; the C++ engine is validated against it.

```
yinsh_ml/
├── game/                       (existing Python engine — DO NOT touch)
└── game_cpp/                   (new — C++ extension)
    ├── __init__.py             (Python wrapper, same surface as game/)
    ├── _bitboard.cpp           (bitboard ops, ray shifts, table generation)
    ├── _board_state.cpp        (state representation, move gen, row detection)
    ├── _zobrist.cpp            (ported from yinsh_ml/game/zobrist.py)
    ├── _bindings.cpp           (pybind11 module)
    └── tests/
        └── test_engine_parity.py   (bit-exact equivalence)
```

Self-play config gets a flag:

```yaml
self_play:
  use_cpp_engine: false   # default off; opt in after parity is green
```

Plumb through `scripts/run_training.py` → `supervisor.py` →
`self_play.py`. Default off until parity is validated; flip when green.

### Build system

`setup.py` currently has no extension entry; `requirements.txt` has no
pybind11. Add:

- `pybind11>=2.10` to `[build-system].requires` in a new
  `pyproject.toml`.
- An `Extension` entry to `setup.py` (or migrate to PEP 517-only).
- For Apple Silicon dev: `MACOSX_DEPLOYMENT_TARGET=11.0`.
- For cloud (CUDA Docker images): build during instance setup. The
  bitboard engine is CPU-only — no runtime CUDA dependency.

## Validation strategy

Three layers:

1. **Existing test suite passes.** `pytest yinsh_ml/tests/` with
   `use_cpp_engine: true`. All contract tests above must pass.
2. **Property-based parity tests** in
   `yinsh_ml/tests/test_engine_parity.py`. Generate ≥10K random
   positions (play random moves until terminal). For each:
   - `cpp.get_valid_moves(state) == py.get_valid_moves(state)` (as sets)
   - `cpp.is_terminal(state) == py.is_terminal(state)`
   - `cpp.get_winner(state) == py.get_winner(state)`
   - For each valid move: applied-state board equality
   - Zobrist hashes match for the same state
3. **Performance benchmarks.** Re-run `scripts/profile_engine.py` and
   `scripts/profile_mcts_selfplay.py` with both engines; report
   ratios. Target ≥10× on random playouts; ≥5× on full MCTS self-play
   wall-clock (more headroom on the latter since deepcopy goes to
   zero).

## Plumbing gotchas

- **`HeuristicAgent` Zobrist coupling.** `yinsh_ml/agents/heuristic_agent.py`
  runs alpha-beta with a transposition table keyed on
  `ZobristHasher.hash_state(state)`. If your engine doesn't expose a
  matching `zobrist_hash()`, the agent's TT is dead and it slows
  5–10×. Non-negotiable.
- **MCTS subtree reuse** (`enable_subtree_reuse: true` in production)
  keeps `state` references in tree nodes (see `yinsh_ml/search/mcts.py`).
  Your wrapper must support storing handles in Python tree nodes and
  identity comparison. Immutable state makes this trivial — every
  detached subtree fork holds its own state object.
- **Memory pools.** `GameStatePool` and `TensorPool` are wired through
  self-play (`yinsh_ml/memory/`). With immutable C++ state, the
  GameState pool is bypassed. Confirm whether the pool was actually
  load-bearing (run a smoke with the pool disabled and compare); the
  bitboard win likely dwarfs any pool benefit.
- **State encoder.** `yinsh_ml/utils/encoding.py::StateEncoder.encode_state`
  walks `board.pieces`. Either keep it as-is and provide a `Board`-shaped
  view from C++, or add a fast `engine.to_numpy(out)` writing channels
  directly from bitboards (popcount-iter). Recommended.

## What "done" means

A PR that:
- Ships the C++ engine as `yinsh_ml/game_cpp/`.
- Has parity tests in `yinsh_ml/tests/test_engine_parity.py` that pass.
- Existing test suite passes (no regressions) with both engine flags.
- Benchmark numbers showing ≥10× on random playouts and ≥5× on
  MCTS self-play wall-clock (re-run the two profile harnesses).
- Feature flag (`self_play.use_cpp_engine: bool = False`) plumbed
  through training entry points.
- README updated with build instructions for Mac (Apple Silicon /
  Intel) and Linux/CUDA (cloud instances).

## Suggested first 1–2 days (validate the architecture before committing)

1. **Read these top to bottom**, in order:
   - `CLAUDE.md` (whole file)
   - `yinsh_ml/game/constants.py` (geometry — truth about the hex grid)
   - `yinsh_ml/game/board.py` (the move-gen logic you're replacing)
   - `yinsh_ml/game/zobrist.py` (the hashing you'll mirror)
2. **Verify the profile.** Run `scripts/profile_engine.py` and
   `scripts/profile_mcts_selfplay.py` on your dev box. Confirm
   `find_marker_rows` and `copy.deepcopy` are still the dominant
   costs. Refocus if not.
3. **Skim yngine.** Focus on `bitboard.cpp`, `board_state.cpp`,
   `generate_tables.cpp`. Pay attention to their 121-cell hex layout
   and how it maps to a 128-bit word.
4. **Read the contract tests** listed under "Recent rule-correctness
   fixes" above. Those are the rules your engine must encode.
5. **Build a minimal slice.** `_bitboard.cpp` + `_bindings.cpp`
   exposing one operation (`valid_move_positions` from a single
   bitboard) — bench it. You want a perf number in the first 1–2 days
   as a sanity check that pybind11 + the build system is sound on
   the user's setup.

## Repo state when you start

Branch off `main`. As of 2026-04-30 main contains all the recent
alphazero-comparison work (Phase D, gating revert, eval/play tooling,
profile harnesses). The feature branch `alphazero-comparison` exists
but is at the same SHA as main — either base is fine, but the PR
should target main.

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
git fetch origin --prune
git checkout -b bitboard-port main
```

Don't touch `yinsh_ml/game/`. Don't change network or training code.
Build infrastructure changes go in `setup.py` / `pyproject.toml`.

## Out of scope

- Don't port MCTS to C++. We have a working Python one. (Yngine's
  lock-free MCTS is a separate future project.)
- Don't change the move-encoding scheme (7433 slots fixed).
- Don't change network or trainer code.
- Don't add new game features.
- Don't optimize anything *outside* `yinsh_ml/game/` even if you find
  easy wins — separate PRs.

## FAQ

- **`std::bitset<128>` or `__uint128_t`?** `__uint128_t`. Yngine does
  it, it's fine on clang and gcc, and it compiles to native
  instructions. `std::bitset<128>` won't.
- **Should the Python wrapper subclass `GameState`?** No — compose.
  Hold a C++ engine handle and expose the same method names.
- **Pure-Python bitboards as a fallback?** Not worth it. Python
  bigint ops aren't free, and you don't get the deepcopy win without
  also restructuring to immutable state — at which point you've done
  70% of the structural work without the perf payoff. If pybind11 +
  the build system is genuinely 2 days of yak-shaving, ask the user
  before switching paths.
- **WebAssembly / browser?** Not a concern. CPU-side C++ is fine for
  Mac dev and Linux cloud.

## Final notes

This is a 1–2 week project for a focused engineer. Validate as you go
— a passing parity test on random positions catches most bugs before
they reach the trainer.

The user values:
- Concise commit messages with the "why," not the "what."
- Clear comments where C++ does something non-obvious (especially
  bit-twiddling in ray shifts).
- **No "Co-Authored-By" / "Generated by Claude" attribution** in commits.
- Direct edits to existing files when needed; don't create
  parallel-but-not-quite-compatible new files.

Good luck.
