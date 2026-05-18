# Live game viewer

YINSIM-style game viewer for YINSH self-play parquet output. Built to
audit heuristic-vs-heuristic data generation (e.g. checking for the
offense-only equilibrium that the project's heuristic feature set is
vulnerable to) and to watch games as they're produced.

## What's here

```
yinsh_ml/viz/
├── board_render.py     # hex-board renderer → matplotlib.Figure
├── game_replay.py      # parquet → reconstructed Board snapshots
└── __init__.py
scripts/
├── dashboard_games.py            # Streamlit viewer (live mode)
└── generate_heuristic_games.py   # bulk HA-vs-HA harness (CPU only)
yinsh_ml/tests/
└── test_viz.py         # 7 tests, ~1.2s
```

## Quickstart

```bash
# Terminal 1 — generate games live, one parquet per game
python scripts/generate_heuristic_games.py \
    --output-dir self_play_data/live \
    --num-games 200 \
    --depth-mix "2:30,3:60,4:10" \
    --workers 8 --batch-size 1 --epsilon 0.05

# Terminal 2 — watch them stream in
streamlit run scripts/dashboard_games.py
# In the sidebar:
#   - Point "Parquet directory" at self_play_data/live/parquet_data
#   - Toggle "Auto-refresh" + pick an interval (2-30s)
#   - "Compute heuristic features on-the-fly" should be on by default
```

## Running a real audit corpus

One-line wrapper:

```bash
scripts/run_heuristic_audit.sh                # 200 games, depth-mix 2:20,3:60,4:20
scripts/run_heuristic_audit.sh my_run_name    # custom name → self_play_data/my_run_name/
NUM_GAMES=50 scripts/run_heuristic_audit.sh smoke    # quick smoke
DEPTH_MIX=3:100 WORKERS=4 scripts/run_heuristic_audit.sh d3only
```

Output lands in `self_play_data/<run_name>/parquet_data/` (one parquet
per game in live mode) plus a `run.log` for the harness output. Open
the dashboard in another terminal and point the sidebar there with
Auto-refresh on — games appear as they complete.

### What the recommended defaults give you

| knob | default | why |
|---|---|---|
| `NUM_GAMES=200` | 200 | enough for the per-game metrics to stabilise statistically; ~1-2h serial on a modern laptop, faster with `WORKERS=8` |
| `DEPTH_MIX="2:20,3:60,4:20"` | 60% at depth 3 | depth 1-2 doesn't deliberately build runs (no case-(a) threats appear, audit produces nothing); depth 4+ is slow and adds little new information |
| `TIME_LIMIT_SEC=2.0` | 2.0 | caps iterative deepening; depth-3 search comfortably fits |
| `MAX_MOVES=200` | 200 | catches stalled games but lets real games play to completion |
| `EPSILON=0.05` | 0.05 | ε-greedy at root for trajectory diversity — 5% random moves per turn |
| `BATCH_SIZE=1` | 1 | one parquet per game → dashboard sees games live |

### Interpreting the results — decision tree

After the run completes, open the dashboard, pick a few games at random, and check the totals across all games (the sidebar's game list shows scores; the Trajectory tab's defensive-miss caption shows the count per game).

**Case 1 — Few captures (mean <2/game), long games.**
Heuristic is shuffling markers without building anything. Investigate first:
- Are games ending at `MAX_MOVES`? If yes, raise `MAX_MOVES` to 300 first; YINSH games at depth 3+ can run long.
- If games end early with 0-0 score, the heuristic may be stuck in a defensive equilibrium. Bump `EPSILON` to 0.10 for more exploration and re-run.

**Case 2 — Many captures (mean ≥3/game), low defensive-miss rate (<10% of threat-turns).**
Healthy game dynamics. Most captures are coming via case (b) path-flips (not preventable). Audit confirms the heuristic produces usable warm-start data. Move on to supervised pretraining.

**Case 3 — Many captures, high defensive-miss rate (≥30% of threat-turns).**
Smoking gun for offense-only collapse — the heuristic builds runs and the opponent fails to defend them. Three responses, in order of cost:
- Cheap: bump training depth from 2 to 3+ in the self-play config (`configs/*.yaml::self_play.num_simulations` for MCTS-mode, or `--depth` for pure HA).
- Medium: ship the Tier A proxy metrics in `TODO_baseline.md` viz section (threat-resolution latency, defense-to-capture ratio, case (a) vs (b) capture classification, per-game miss-rate distribution). All ~20 lines of pandas each over the existing trajectory data — no new game-engine APIs needed. Gives you actual numbers to compare across recipe changes.
- Deeper: Tier C tactical-quality defensive analysis (flip resilience, counter-attack setup, multi-step search). 2-3 days of engine extensions + scoring; opens the door to quantifying *how good* heuristic defense is, not just whether it happened.

**Case 4 — Very skewed W/B scores (one side captures 80%+).**
Should not happen in symmetric HA-vs-HA. Likely cause: a bug or first-move advantage that depth-3 search can't overcome. Investigate by:
- Playing a few of the skewed games manually through the dashboard (look for systematically bad moves)
- Re-running with `SEED_BASE` set to a different value to rule out unlucky seed selection

### Live audit workflow (recommended)

Two terminals:

```bash
# Terminal 1
scripts/run_heuristic_audit.sh audit_v1

# Terminal 2
streamlit run scripts/dashboard_games.py
# Sidebar:
#   - Parquet directory: self_play_data/audit_v1/parquet_data
#   - Live mode: Auto-refresh ON, interval 5s
#   - Compute heuristic features on-the-fly: ON
```

Watch the game count climb in the sidebar. Sample 5-10 games as they
come in — you'll quickly see whether the audit is fingerprinting one
of the cases above, often before the full 200-game run finishes.

## Output placement

- **Default**: `self_play_data/<run_name>/parquet_data/*.parquet`
- **Gitignored**: the repo's `.gitignore` is allowlist-style — all of
  `self_play_data/` is ignored implicitly. No risk of committing
  generated games.
- **Per-game file naming** (live mode): `games_batch_NNNN_TIMESTAMP.parquet`,
  one game per file when `--batch-size 1`.
- **Bulk file naming** (default): same pattern, multiple games per file
  controlled by `--batch-size` (default 100).

## Architecture: loaders + annotators

The viewer splits into three layers so future consumers don't need to
fork the existing audit code:

```
┌─────────────────────────────────────────────────────────────────┐
│ Source (parquet / BGA JSON / in-memory moves / replay buffer)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Loader → GameReplay                                             │
│   load_game(parquet_dir, game_id)        parquet                │
│   replay_from_dataframe(df)              DataFrame              │
│   replay_from_moves(moves, ...)          any List[Move]         │
│   (your_adapter_here)                    any new source         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Annotator(s) → populate GameReplay.annotations                  │
│   captures_and_threats_annotator()       scores, threats, miss  │
│   heuristic_features_annotator()         the 7 hf_* features    │
│   (network_annotator(wrapper))           network value + π      │
│   (mcts_annotator(net, num_sims))        re-run MCTS per turn   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Consumer (Streamlit dashboard, Jupyter, save-to-file, ...)      │
└─────────────────────────────────────────────────────────────────┘
```

**Each layer is independent.** A new source plugs in as a new loader.
A new metric plugs in as a new annotator. The dashboard reads
`replay.features` (loader-provided) and `replay.annotations`
(annotator-provided) generically.

## API

### Rendering

`render_board(board, *, last_move=None, highlight=None, title=None, ax=None, figsize=(8,8), show_coords=True) → Figure`

Renders a `Board` to a `matplotlib.Figure`. Drops into `st.pyplot(fig)`,
`fig.savefig("foo.png")`, or any matplotlib backend without further glue.
`last_move` highlights a move with from/to circles and an arrow;
`highlight` marks arbitrary positions with dashed circles.

Geometry: monotonic skew along the matching-sign diagonal hex axis. All
three hex axes render as 60°-separated unit-length screen lines —
verified by `test_hex_axes_are_unit_distance`.

### Loaders

```python
from yinsh_ml.viz import (
    list_games, load_game,            # parquet directory → GameReplay
    replay_from_dataframe,            # in-memory pandas DataFrame → GameReplay
    replay_from_moves,                # any List[Move] → GameReplay  ← most general
)

# Parquet (current self-play harness output)
summary = list_games(Path("self_play_data/run/parquet_data"))
replay  = load_game(Path("..."), "game_id")

# Source-agnostic — works for BGA scraper output, in-memory moves
# from a running worker, synthetic moves in tests, anything that can
# produce a List[Move]:
replay = replay_from_moves(moves, game_id="my_game", winner="WHITE")
```

### GameReplay shape

```python
replay.moves            # List[Move]
replay.states           # List[Board], length len(moves)+1
replay.board_after(i)   # Board after move i
replay.board_before(i)  # Board before move i
replay.iter_states()    # → Iterator[(turn_idx, GameState)]  O(N) single pass

replay.features         # List[Dict] — values inlined by the loader (e.g.
                        # FeatureExtractor columns from parquet). May be
                        # empty if the loader had nothing extra.
replay.annotations      # List[Dict] — values added by annotators after
                        # load. Empty until annotate() runs.

replay.winner           # "WHITE" / "BLACK" / None
replay.replay_truncated_at  # None unless a move was illegal
```

### Annotators

An annotator is `Callable[[GameReplay, turn_idx, GameState], Dict[str, Any]]` —
no subclassing required. Run one or more via `annotate()`:

```python
from yinsh_ml.viz import (
    annotate,
    captures_and_threats_annotator,
    heuristic_features_annotator,
)

annotate(replay, [
    captures_and_threats_annotator(),
    heuristic_features_annotator(player=Player.WHITE),
])
# Now replay.annotations[i] is a dict with capture, white_score,
# white_threats, defensive_miss, hf_completed_runs_differential, …
```

Multiple annotators run in a single forward pass through the game.
Their output dicts are merged per turn. A broken annotator logs a
warning and is skipped for that turn — doesn't take the whole run down.

### Writing a custom annotator

```python
def my_annotator(replay, turn_idx, state) -> Dict[str, Any]:
    # state is the GameState AFTER move turn_idx
    return {
        "my_metric": some_computation(state),
        "another": state.board.count_pieces(),
    }

annotate(replay, [my_annotator])
# replay.annotations[i] now has "my_metric" and "another" keys
```

For stateful annotators (e.g. tracking deltas across turns) use a
closure-with-mutable-dict or a class — see
`captures_and_threats_annotator` in `annotators.py` for the closure
pattern. The annotator framework doesn't care which.

### Sketches for use cases not yet implemented

**Neural self-play review** — currently the training pipeline writes
`replay_buffer.pkl.gz` with encoded state tensors but no moves. Two
paths: (a) thread a `GameRecorder` through `training/self_play.py` to
write a sidecar GameRecord parquet (small change, opens the door to
viewing neural games); (b) write a `network_annotator(wrapper)` that
runs the network on each replayed state and emits the value
prediction + top-k move probabilities.

**Expert game review** — load a BGA-scraped game into `replay_from_moves`
(BGA scraper already produces the same move-dict schema as parquet),
then annotate with `network_annotator` and/or `mcts_annotator` to see
what your current model would have played at each position. The
dashboard surfaces the model's recommendation alongside the human's
actual move.

**MCTS visit display** — `mcts_annotator(network, num_sims=400)` would
re-run MCTS at each replayed position and emit the visit-count
distribution. Expensive (linear in game length × num_sims) but very
informative for understanding why a particular move was chosen.

### Harness flags

Run `python scripts/generate_heuristic_games.py --help` for the full list.
Key ones for the audit workflow:

- `--depth-mix "2:30,3:60,4:10"` — sample search depth per game from
  this distribution. Mixed depth = more diverse trajectories.
- `--epsilon 0.05` — ε-greedy at the root for trajectory diversity.
- `--batch-size 1` — one parquet file per game (live mode).
- `--chunk-size N` — multiprocessing chunk size, independent of
  `--batch-size`. Default `max(batch_size, workers)`.
- `--max-moves 300` — same default as `training/self_play.py`.

## Known limitations

- **4-row threat metric only catches half the failure modes.** YINSH
  ring moves do two things atomically: leave a marker at the source AND
  flip every marker along the path. So a 5-row can form either via
  (a) gradual buildup — where a 4-row IS visible on the opponent's turn
  and can be defended by flipping one of its markers — or (b) a
  single ring move whose path-flip converts a 3-row or scattered
  markers directly into a 5-row, with no visible 4-row state. The
  `count(length == 4)` metric catches case (a) defensive misses but is
  blind to case (b). Pair with `white_potential`/`black_potential`
  (length ≥3) and capture-event tracking for the full picture.
- **HA-vs-HA throughput is low.** Iterative deepening + transposition
  table init dominates per-move time. At depth 1 / time-limit 0.3s,
  expect ~50 games/hour serial. Multiprocessing scales linearly.
- **No incremental parquet writer.** Live mode uses `--batch-size 1` as
  a workaround. If you need both small write latency *and* large
  parquet files for downstream training, the project's
  `ParquetDataStorage` would need an `append_to_batch` mode.
- **`tracking/yinsh_visualizer.py` has a geometry bug** (zig-zag
  `col_idx % 2` offset instead of monotonic skew). Untouched here;
  noted in `board_render.py`'s docstring for a future cleanup.
- **HeuristicAgent's import chain pulls in torch/coremltools/seaborn**
  via `search/__init__.py` and `network/__init__.py`. A pure-search
  agent shouldn't need any of those. The dashboard sidesteps it by
  calling `extract_all_features` directly.

## Tests

```bash
pytest yinsh_ml/tests/test_viz.py -v
```

Seven tests covering: hex-axis unit-distance invariant, render smoke,
parquet replay roundtrip with per-turn `Board` equality, capture
detection from score deltas, `iter_states` correctness, `list_games`
summary columns.
