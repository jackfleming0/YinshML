# Bitboard port prompt — replace Python YINSH engine with a fast C++ extension

> Self-contained prompt for a fresh Claude Code session. Paste this whole
> file into the new session, or `cat` it as the first message.

---

## What you're being asked to do

Port the YINSH game engine in `yinsh_ml/game/` from pure Python to a C++
extension exposed via pybind11, modeled on the bitboard design from
[temhelk/yinsh](https://github.com/temhelk/yinsh) ("yngine"). The goal
is **10–100× speedup on move generation** — currently the dominant
self-play bottleneck.

This is an *engineering performance* project, not a research project.
The game rules are settled. The reference Python implementation is
correct (recently audited and re-fixed; see "Recent rule fixes" below).
Your job is to produce a faster implementation that's bit-exact
equivalent to the Python reference, validated by the existing test
suite + new property-based tests.

## Why this matters (read once, then move on)

We just finished Phase D — a 12-iter warm-start training run with a new
gating-revert flag. The recipe works structurally (gates fire correctly,
no compounding regression) but plateaus sub-intermediate. The recipe's
peak (`iter_3`) loses 6/24 to `HeuristicAgent(depth=3)` at deployment
budget — well below the ≥65% intermediate bar.

To reach intermediate, we need to test other recipes (more games per
iter, larger network, MCTS knob tuning, growing replay window, frontier
shifts). Each recipe experiment costs ~$2–10 + hours of cloud time.
**A 10× speedup makes ~10× more recipes affordable.** That's why this
is the priority engineering project right now.

Full strategic context: `WARMSTART_PHASE_LOG.md` §9 (Phase D conclusion
+ Phase E ranking). You don't need to read it to do this task — but
glance at §9c if you want to know what comes after this lands.

## Reference: temhelk/yinsh's "yngine" design

GitHub: https://github.com/temhelk/yinsh — the C++ submodule lives at
`yngine/` inside that repo. Key design points (from the
alphazero-comparison study earlier this week):

- **4 bitboards per state**, one per piece type per color:
  `white_rings`, `black_rings`, `white_markers`, `black_markers` — each
  a `__uint128_t`. 121 cells fit in 128 bits with room to spare.
- **Precomputed ray tables** `TABLE_RAYS[121][6]` — for each cell index
  and each of 6 directions, a bitboard of the cells reachable along
  that ray. Instant ray lookup; no loop.
- **Move generation** iterates set bits with `bit_scan_and_reset` and
  uses ray bitboards + popcount. O(1) amortized.
- **Move/unmove** is immutable — `apply_move` returns a new state.
  No undo machinery needed; saves a lot of complexity.
- **No hash/transposition table inside yngine** — they don't have a
  Zobrist hash. We *do* (`yinsh_ml/game/zobrist.py`), and we want to
  keep it because it's load-bearing for the heuristic agent's
  alpha-beta. So your bitboard engine needs to expose enough for our
  Python-side Zobrist hasher to work, OR you can port Zobrist to C++
  too (probably cleaner).

Specific files in yngine to study (from the comparison study):
- `bitboard.hpp` / `bitboard.cpp` — bitboard ops, bit-scan, ray shifts
- `board_state.hpp` / `board_state.cpp` — state representation, move
  generation, row detection (`length_of_row`, line enumeration over
  6/7-marker maximal runs)
- `generate_tables.cpp` — how they build `TABLE_RAYS[121][6]` and the
  pre-computed "shift in direction" tables
- `mcts.cpp` — they parallelize MCTS with atomics and tree reuse. We
  don't need to port their MCTS (we already have a working one), but
  the move-gen interface is what your bitboard engine needs to support.

## The Python reference you're replacing

Look at these files (fully read each):

| File | Lines | What it does |
|---|---|---|
| `yinsh_ml/game/constants.py` | 153 | Board geometry, `VALID_POSITIONS`, `DIRECTIONS`, `HEX_DIRECTIONS`, `HEX_LINE_AXES` |
| `yinsh_ml/game/types.py` | 58 | `Player`, `PieceType`, `MoveType`, `GamePhase` enums |
| `yinsh_ml/game/board.py` | 564 | `Board` class — piece placement, ring movement (with marker flips), `find_marker_rows`, `valid_move_positions` |
| `yinsh_ml/game/moves.py` | 246 | Move enumeration per game phase |
| `yinsh_ml/game/game_state.py` | 608 | `GameState` — phase tracking, win/draw, `make_move`, `is_terminal`, `get_winner`, `_handle_marker_removal`, `_move_maker` |
| `yinsh_ml/game/zobrist.py` | 533 | Zobrist hash — must include side-to-move + game phase per recent fix |

Total ~2200 lines of Python. The C++ port should be smaller (bitboards
are dense) — yngine's equivalent is roughly 1500 lines of C++.

## CRITICAL — recent rule fixes that MUST be preserved

These are non-negotiable. If your bitboard engine doesn't pass the
existing tests, it's wrong. From `CLAUDE.md` (the "Foundational Rule
Fixes (April 2026)" section):

1. **Pseudo-diagonal row bug fix.** `find_marker_rows` /
   `is_valid_marker_sequence` no longer accept the non-hex `(-1, 1)` /
   `(1, -1)` diagonal. Single source of truth is `DIRECTIONS` (3
   forward axes) and `HEX_LINE_AXES` (6 signed directions) in
   `constants.py`. The yngine reference is correct here too — its
   `Direction` enum has only the 6 valid hex directions
   (SE/NE/N/NW/SW/S).
2. **Zobrist includes side-to-move AND game phase.**
   `ZobristHasher.hash_state(state)` mixes in `current_player` *and*
   `phase`. Two positions with identical board layout but different
   side-to-move or phase MUST hash differently. There are O(1)
   `toggle_side_to_move` / `toggle_phase` helpers for incremental
   updates.
3. **Encoder side-to-move sentinel.** `decode_state` doesn't flip
   colors for BLACK; current player is recovered from a sentinel at
   off-board cell A1 on channel 5. This isn't game-engine logic but
   may matter if you change how state is exposed — check
   `yinsh_ml/utils/encoding.py` consumers.
4. **Move encoding is 7433 slots, not 8390.** The REMOVE_MARKERS
   sub-layout uses a deterministic 123-entry lookup table over the
   123 valid 5-in-a-row hex lines (was 121 before — 6/7-marker
   maximal-run support added 2 entries). Policy-head checkpoints
   from before this batch are NOT load-compatible.
   `NetworkWrapper.load_model` hard-fails on size mismatch. **Your
   bitboard engine doesn't need to know the move encoding** — it just
   needs to enumerate moves; encoding happens elsewhere.
5. **Stalemate detection in MAIN_GAME.** `is_terminal` /
   `get_winner` handle no-legal-moves correctly. yngine handles this
   via a `PassMove` variant — pick whichever interface is cleanest;
   the Python tests will tell you if you've broken it.
6. **`_handle_marker_removal` is atomic.** Validate-all-then-remove-all
   semantics. Don't half-apply.
7. **6/7-marker maximal row support.** `find_marker_rows` returns the
   full maximal run (5/6/7 positions); the move generator enumerates
   every length-5 window over it. yngine's
   `number_of_rows = total_length - 4` enumeration is the exact
   pattern (`board_state.cpp:215–227`). `total_moves` stays at 7433
   because the 123-entry REMOVE_MARKERS table is already
   windows-of-5, agnostic to underlying run length.

The complete list of recent rule-correctness fixes is documented in
`CLAUDE.md`. Read that section before designing your bitboard layout —
some of these will affect your data structures.

## Architecture proposal (revise as needed)

**Don't replace the existing Python engine. Add the C++ engine
alongside, gated by a feature flag.** The Python engine remains the
reference; your C++ engine is validated against it.

Suggested layout:

```
yinsh_ml/
├── game/                       (existing Python engine — DO NOT touch)
│   ├── board.py
│   ├── game_state.py
│   ├── moves.py
│   └── ...
└── game_cpp/                   (new — C++ extension)
    ├── __init__.py             (Python wrapper exposing same interface as game.GameState)
    ├── _bitboard.cpp           (bitboard ops, ray shifts, table generation)
    ├── _board_state.cpp        (state representation, move gen, row detection)
    ├── _zobrist.cpp            (ported from yinsh_ml/game/zobrist.py)
    ├── _bindings.cpp           (pybind11 module)
    └── tests/
        └── test_engine_parity.py   (bit-exact equivalence tests)
```

Self-play config gets a flag like:

```yaml
self_play:
  use_cpp_engine: false   # default off; opt in for speedup
```

Default off until parity is fully validated. When parity is proven
across the existing test suite, flip the default.

**Build system:** add a `setup.py` extension entry. Use pybind11 (modern,
header-only). For development on macOS, set `MACOSX_DEPLOYMENT_TARGET`
appropriately. For cloud (CUDA), build during instance setup — no
runtime CUDA dependency, this is a CPU-only extension.

**Alternative if pybind11 + C++ proves painful:** pure-Python bitboards
using Python's `int` (arbitrary precision). 5–10× speedup vs the
current loop-heavy code, no build-system pain. Worth keeping as a
fallback — though the user's preference is the full C++ port for the
10–100× headline speedup.

## Validation strategy

Three layers:

1. **Existing test suite must pass.** `pytest yinsh_ml/tests/` —
   especially `test_game_logic.py`, anything touching `Board` or
   `GameState`. The C++ engine should expose the same Python interface
   (so a `use_cpp_engine` flag swaps which class is used) OR there
   should be a `to_python()` / `from_python()` bridge.
2. **Property-based parity tests.** For 10K random game positions
   (generated by playing random moves until terminal), assert:
   - `cpp_engine.get_valid_moves(state) == py_engine.get_valid_moves(state)`
   - `cpp_engine.is_terminal(state) == py_engine.is_terminal(state)`
   - `cpp_engine.get_winner(state) == py_engine.get_winner(state)`
   - For each valid move: `cpp_engine.apply(state, move).board == py_engine.apply(state, move).board`
   - Zobrist hashes match (same hash for same state).
3. **Performance benchmarks.** Add a `benchmarks/` directory or
   extend the existing `yinsh_ml/search/performance_profiler.py`.
   Measure: random-playout games per second, full MCTS runs per
   second, end-to-end self-play time. Target ≥10× over Python; report
   what you actually get.

## What "done" means

A PR that:
- Ships the C++ engine as `yinsh_ml/game_cpp/`.
- Has parity tests in `yinsh_ml/tests/test_engine_parity.py` that pass.
- Existing test suite still passes (no regressions).
- Has benchmark numbers showing ≥10× speedup on a representative
  workload (random playouts is the easy benchmark; self-play
  iteration time is the load-bearing one).
- Has a feature flag (`self_play.use_cpp_engine: bool = False`) plumbed
  through `scripts/run_training.py` → `supervisor.py` → `self_play.py`
  so an A/B is trivial.
- README updated with build instructions for both Mac (Apple Silicon /
  Intel) and Linux/CUDA (cloud instances). The dev primary platform is
  Mac M-series (MPS); cloud is x86_64 + CUDA.

## Suggested first steps (1–2 days, before committing to the full port)

1. **Read these files top to bottom**, in order:
   - `CLAUDE.md` (whole file — context for the codebase)
   - `WARMSTART_PHASE_LOG.md` §9 (why this project matters)
   - `yinsh_ml/game/constants.py` (geometry — the truth about the hex grid)
   - `yinsh_ml/game/board.py` (the move-gen logic you're replacing)
   - `yinsh_ml/game/zobrist.py` (the hashing you'll need to mirror)
2. **Skim yngine** (https://github.com/temhelk/yinsh) — focus on
   `bitboard.cpp`, `board_state.cpp`, `generate_tables.cpp`. Pay
   attention to how they map their 121-cell hex grid to a 128-bit
   layout (their cell ordering vs ours).
3. **Read the existing tests** in `yinsh_ml/tests/test_game_logic.py`
   and any `test_board*.py`, `test_zobrist*.py`. Those are the contract
   your bitboard engine must satisfy.
4. **Profile the current Python engine** to confirm the bottleneck.
   Run `python scripts/run_training.py --config configs/smoke.yaml` or
   similar with `cProfile` and identify which functions dominate. If
   it's `Board.valid_move_positions` and `Board.find_marker_rows`,
   that's expected; if it's elsewhere, refocus.
5. **Start with a small slice.** Implement the bitboard structure +
   one operation (e.g., `valid_move_positions`) and bench it before
   tackling the full move generator. You want a working perf number
   in the first 1–2 days as a sanity check that the architecture is
   sound.
6. **Decide between full C++ port and pure-Python bitboards.** If
   pybind11 build infra is fast on the user's setup, go C++. If it's
   2 days of yak-shaving before any code runs, switch to pure-Python
   bitboards as a stepping stone (still a 5–10× speedup), and revisit
   C++ once the core design is validated.

## Key constraints / gotchas

- **The user's primary dev platform is Mac M-series.** Build needs to
  work there. Apple Silicon C++ is fine, but watch for clang vs gcc
  differences in 128-bit int support (`__uint128_t` is fine on both,
  but bit ops can vary).
- **Cloud is x86_64 + CUDA.** The bitboard engine is CPU-only; the
  CUDA part is just the neural net. Build needs to work in a CUDA
  Docker container too.
- **Recent rule fixes (above) MUST be preserved.** The Python tests
  encode them; if your engine fails any of those tests, the rules are
  wrong.
- **Don't change the move encoding (7433 slots, REMOVE_MARKERS table).**
  That's set in `yinsh_ml/utils/encoding.py` and the network's policy
  head depends on it. Your bitboard engine just needs to enumerate
  moves; the encoding layer maps them to indices.
- **The heuristic agent uses the engine too.**
  `yinsh_ml/agents/heuristic_agent.py` runs alpha-beta on the same
  `GameState` interface. If your engine swaps in cleanly, both
  self-play and the heuristic agent benefit.
- **Don't break the move-encoding round-trip.** `state.get_valid_moves()`
  → `encoder.move_to_index(m)` → `encoder.index_to_move(i)` →
  `state.make_move(m)` must produce identical results between Python
  and C++ engines. This is the most likely source of subtle bugs.

## Repo state when you start

Branch: probably `clean-slate` (active dev) or a new branch off
`main`. Recent commits include the gating-revert PR (merged to `main`)
and several doc commits on `alphazero-comparison`.

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
git fetch origin --prune
git checkout -b bitboard-port main      # or branch off clean-slate
```

Don't touch the existing `yinsh_ml/game/` package. Don't change the
network or training code. Build infrastructure changes go in
`setup.py` and a new `pyproject.toml` extension entry if needed.

## Out of scope

- Don't port the MCTS to C++. We have a working Python one. yngine's
  lock-free MCTS is a separate future project (`TODO_baseline.md`
  Tier 5).
- Don't change the move-encoding scheme. The 7433-slot policy head is
  fixed and load-bearing.
- Don't change the network or trainer code.
- Don't add new game features (no AI tweaks, no scoring rule changes).
- Don't optimize anything *outside* the game engine package — even if
  you find easy wins, they belong in separate PRs.

## Questions you'll probably have, answered up front

- **"Should I use std::bitset or `__uint128_t`?"** `__uint128_t` is
  what yngine uses and it's fine on clang and gcc. `std::bitset<128>`
  is slower for ray operations because it doesn't compile to native
  instructions. Use `__uint128_t`.
- **"What about WebAssembly / browser deployment?"** Not a concern
  right now. CPU-side game engine in C++ is fine for both Mac dev and
  Linux cloud.
- **"Should the Python wrapper subclass the existing GameState?"**
  No. Compose, don't inherit. A Python-side `GameState` that holds a
  C++ engine handle and exposes the same method names is the cleanest
  bridge.
- **"What about the `__copy__` / `__deepcopy__` semantics?"** The
  Python `GameState` is copyable (used in MCTS simulations). The C++
  engine should expose a `clone()` method; the Python wrapper's
  `__copy__` calls it. yngine uses immutable state (every `apply_move`
  returns a new state) — that pattern works here too and avoids most
  copy issues.

## Final notes

This is a 1–2 week project for a focused engineer. Don't try to
finish in a day. Validate as you go — a passing parity test on
random positions catches most bugs before they reach the trainer.

The user values:
- Concise commit messages with the "why," not just the "what."
- Clear comments where the C++ does something non-obvious (especially
  bit-twiddling in ray shifts).
- No "Co-Authored-By" / "Generated by Claude" attribution in commits.
- Direct edits to existing files when needed; don't create
  parallel-but-not-quite-compatible new files.

Good luck.
