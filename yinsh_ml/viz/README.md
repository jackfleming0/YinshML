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

## Audit runbook

For the **offense-only equilibrium audit** specifically:

1. **Generate ≥200 games at depth 3** with `--epsilon 0.05` for trajectory
   diversity. Depth 2 is too short-sighted to test the hypothesis;
   depth 4+ is slow.
2. Open the dashboard, switch to the **Trajectory** tab.
3. For each game (sample at least 20), look for:
   - **Score progression** climbing for one side without the other
     responding within 10-15 moves
   - **`completed_runs_differential`** monotonically growing for one
     player — the signature of one side building captures unchallenged
   - **`white_potential` / `black_potential`** (rows of length ≥3)
     diverging — earlier-stage early warning
4. **Capture count distribution**: count captures per game across the
   corpus. Healthy distribution: most games have 2-5 captures total,
   roughly balanced W/B. Sick: one side dominates or games end at the
   move cap with no captures (heuristic is just shuffling markers).

Audit-positive findings → consider adding defensive features or
relying more on search depth (see `TODO_baseline.md` viz section).

## Output placement

- **Default**: `self_play_data/<run_name>/parquet_data/*.parquet`
- **Gitignored**: the repo's `.gitignore` is allowlist-style — all of
  `self_play_data/` is ignored implicitly. No risk of committing
  generated games.
- **Per-game file naming** (live mode): `games_batch_NNNN_TIMESTAMP.parquet`,
  one game per file when `--batch-size 1`.
- **Bulk file naming** (default): same pattern, multiple games per file
  controlled by `--batch-size` (default 100).

## API

### `render_board(board, *, last_move=None, highlight=None, title=None, ax=None, figsize=(8,8), show_coords=True) → Figure`

Renders a `Board` to a `matplotlib.Figure`. Drops into `st.pyplot(fig)`,
`fig.savefig("foo.png")`, or any matplotlib backend without further glue.
`last_move` highlights a move with from/to circles and an arrow;
`highlight` marks arbitrary positions with dashed circles.

Geometry: monotonic skew along the matching-sign diagonal hex axis. All
three hex axes render as 60°-separated unit-length screen lines —
verified by `test_hex_axes_are_unit_distance`.

### `GameReplay`

```python
from yinsh_ml.viz.game_replay import load_game, list_games

# Summary of all games in a parquet directory
summary = list_games(Path("self_play_data/run/parquet_data"))

# Load and replay one game
replay = load_game(Path("..."), "game_id")
replay.moves            # List[Move]
replay.states           # List[Board], length len(moves)+1
replay.board_after(i)   # Board after move i
replay.board_before(i)  # Board before move i
replay.iter_states()    # → Iterator[(turn_idx, GameState)]  O(N) single pass
replay.features         # List[Dict] — parquet-recorded features per turn
replay.winner           # "WHITE" / "BLACK" / None
replay.replay_truncated_at  # None unless a move was illegal
```

`iter_states()` is the one to use when you need full `GameState` (for
phase- or score-dependent feature computation) at every turn — it does a
single forward pass.

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

- **4-row threat metric is too narrow for YINSH.** Markers go 4→5 in a
  single ring-move (the ring leaves a marker at its source), so 4-row
  threats live <1 turn. Use the broader `white_potential`/`black_potential`
  (length ≥3) and capture events as the real audit signals.
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
