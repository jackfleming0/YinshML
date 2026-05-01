# yinsh_ml.game_cpp — C++ bitboard engine

A pybind11 port of `yinsh_ml.game` that's bit-exact equivalent to the
Python reference, written to take engine cost off the critical path
during training. Default-off; opt in per training config when ready.

## Why this exists

Profile baseline on the Python engine (random-playout workload):

| op | self-time share |
|---|---:|
| `Board.find_marker_rows` | 24% |
| `Board.valid_move_positions` | 11% |
| `copy.deepcopy(GameState)` (under MCTS) | ~95% of profiled time |

Recipe experiments cost real money per run, so cutting engine cost
buys more recipes per dollar. See `WARMSTART_PHASE_LOG.md` §9 for
strategic context, `BITBOARD_PORT_PROMPT.md` for the design brief.

Measured speedups, benched apples-to-apples vs the Python engine:

| comparison | speedup |
|---|---:|
| `valid_ring_destinations` | 771× |
| `find_marker_rows` | 557× |
| `State.clone` vs `copy.deepcopy(GameState)` | ~134,000× |
| Random-game playout (no NN) | **258×** |
| Standalone MCTS self-play (no encoder overhead) | ~21× |
| Full training worker w/ ResNet on CPU | ~1× (NN dominates ≥60%) |

The full-worker number isn't a regression — it's what happens once the
engine bottleneck is gone: `torch.conv2d` becomes the new largest
share. Bigger networks / more sims / GPU paths trade off differently;
A/B per config before flipping the flag in production.

## Layout

```
yinsh_ml/game_cpp/
├── README.md                    (this file)
├── __init__.py                  (re-exports CppGameState, CppBoard, _engine)
├── _convert.py                  (Move/Position <-> cell-index helpers)
├── game_state.py                (CppGameState, CppBoard — Python facade)
├── src/
│   ├── bitboard.hpp             (121-cell layout, kBoardMask)
│   ├── tables.hpp               (constexpr ray tables, valid_ring_destinations,
│   │                             find_marker_rows)
│   ├── moves.hpp                (Move POD)
│   ├── state.hpp                (State POD, trivially-copyable)
│   ├── apply.hpp                (apply_move + phase machinery)
│   ├── movegen.hpp              (get_valid_moves dispatcher)
│   ├── zobrist.hpp              (XOR-only hashing, table fed from Python)
│   └── bindings.cpp             (pybind11 module)
└── tests/
    ├── test_valid_ring_destinations.py
    ├── test_find_marker_rows.py
    ├── test_zobrist_parity.py
    ├── test_state_clone.py
    ├── test_rollout_parity.py     (load-bearing: 50 random rollouts in lockstep)
    └── test_wrapper_parity.py     (load-bearing: 20 lockstep rollouts via the
                                    Python facade, plus deepcopy bench)
```

Nothing in `yinsh_ml/game/` was touched. The C++ engine lives alongside
as a parallel implementation, gated on `self_play.use_cpp_engine`.

## Build

The extension is wired through `setup.py` as a `Pybind11Extension`.

### macOS (Apple Silicon or Intel)

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
# If the .so didn't pick up automatically:
python setup.py build_ext --inplace
```

`MACOSX_DEPLOYMENT_TARGET=11.0` is set inside `setup.py` so the build
produces a universal2 (arm64+x86_64) wheel that runs on both Apple
Silicon and Intel without re-compilation.

### Linux / CUDA cloud instances

The bitboard engine is **CPU-only** — no runtime CUDA dependency. On a
fresh cloud Linux box (Ubuntu / Debian / CentOS) you need a C++17
compiler:

```bash
sudo apt-get update && sudo apt-get install -y build-essential   # Debian/Ubuntu
# or: sudo yum install -y gcc-c++ make                            # RHEL/CentOS

python -m venv venv
source venv/bin/activate
pip install -e .
```

Compilation takes a handful of seconds; the resulting `.so` is small
(~80 KB on Apple Silicon, similar on Linux x86_64).

### Verifying the build

```bash
python -c "from yinsh_ml.game_cpp import _engine; \
           print('cells:', _engine.CELL_COUNT, 'valid:', _engine.VALID_CELL_COUNT)"
# expected: cells: 121 valid: 85

pytest yinsh_ml/game_cpp/tests/ -q
# expected: ~70 tests pass; takes ~30s
```

The `.so` is gitignored. Re-run `python setup.py build_ext --inplace`
after any header change.

## Using the engine

### From a training config

```yaml
self_play:
  use_cpp_engine: true
```

That's the entire opt-in surface. The flag plumbs through
`scripts/run_training.py` → `supervisor.py` → `SelfPlay` →
`play_game_worker`. When true, the worker instantiates `CppGameState`
directly and skips the `GameStatePool` (clone() is a struct memcpy,
no pool needed).

### From Python code

```python
from yinsh_ml.game_cpp import CppGameState

state = CppGameState()  # duck-typed drop-in for yinsh_ml.game.GameState
moves = state.get_valid_moves()
state.make_move(moves[0])
print(state.phase, state.current_player, state.white_score)

# Cheap clone — bottoms out in struct memcpy via __deepcopy__.
import copy
clone = copy.deepcopy(state)
```

`CppGameState` exposes the same surface MCTS / self-play / encoder /
heuristic-agent code reads from a `GameState`: `make_move`,
`get_valid_moves`, `is_terminal`, `get_winner`, `is_stalemate`,
`current_player`, `phase`, `white_score`, `black_score`,
`rings_placed`, `move_history`, `copy()`, plus a `board` attribute
that mirrors `Board.pieces`, `Board.get_piece`, `Board.is_empty`,
`Board.find_marker_rows`, `Board.valid_move_positions`,
`Board.to_numpy_array`. Moves are exchanged as the same
`yinsh_ml.game.types.Move` dataclass — no schema change for callers.

### Direct C++ primitives

For tools that want to drive the engine without going through the
GameState facade (e.g. high-throughput parity tests), call into
`_engine` directly:

```python
from yinsh_ml.game_cpp import _engine

state = _engine.State()
moves = _engine.get_valid_moves(state)
state = _engine.apply_move(state, moves[0])
print(_engine.is_terminal(state), _engine.winner(state))
```

The `_engine` module is documented inline in `src/bindings.cpp`.

## Validation strategy

Three layers; all must be green before the flag flips by default.

1. **Existing test suite passes.** `pytest yinsh_ml/tests/test_*.py`
   covering the foundational rule fixes (hex axes, Zobrist invariants,
   move encoder round-trip) — the C++ engine is never used by these
   tests, so they verify the Python reference stayed correct.

2. **Property-based parity tests.** Live in
   `yinsh_ml/game_cpp/tests/`. Highlights:
   - `test_rollout_parity.py` — 50 random rollouts, comparing every
     state field, every legal-move set, every phase transition.
   - `test_wrapper_parity.py` — 20 random rollouts driven through
     `CppGameState`, comparing `board.pieces`,
     `board.find_marker_rows`, `board.valid_move_positions`,
     `board.to_numpy_array`, `get_valid_moves`, `is_terminal`,
     `get_winner` per ply.
   - `test_zobrist_parity.py` — empty-board hash, 4000+
     `(board × phase × player)` combinations, incremental toggles.

3. **Performance benchmarks.** Run alongside parity to catch
   regressions:
   - `scripts/profile_engine.py` — Python random-playout baseline.
     The C++ comparison lives inline in `bindings.cpp`'s
     `bench_random_playouts` helper.
   - Wrapper deepcopy bench in `test_wrapper_parity.py::test_wrapper_deepcopy_uses_fast_path`
     asserts ≥10× over `copy.deepcopy(GameState)`.

## Caveats and out-of-scope

- **MCTS port.** Not done. The brief explicitly defers it. Yngine has
  a lock-free MCTS that may eventually be worth porting; for now MCTS
  stays in Python and works against `CppGameState` via duck typing.
- **Move encoding.** Unchanged at 7433 slots. The C++ engine doesn't
  own the encoding; it just enumerates `Move` objects that the
  existing `MoveEncoder` consumes.
- **HeuristicAgent transposition table.** Currently uses Python
  `ZobristHasher`. Migrating it to `_engine.Zobrist` is straightforward
  (the C++ side is bit-exact with the Python keys) but separate from
  this port.
- **`CppBoard` is read-only.** Mutation goes through
  `CppGameState.make_move`. Code that mutates a `Board` directly
  needs to migrate.

## Reference

- yngine, the bitboard engine that inspired this port:
  https://github.com/temhelk/yinsh
- Brief: `BITBOARD_PORT_PROMPT.md`
- Foundational rule fixes the engine encodes: `CLAUDE.md` §
  "Foundational Rule Fixes (April 2026)"
