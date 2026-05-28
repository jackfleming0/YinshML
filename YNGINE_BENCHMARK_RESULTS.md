# yngine benchmark — first measured win rates

**Date:** 2026-05-28
**Branch:** `vs-yngine-eval`
**Goal:** Close the "we've never actually measured a model vs yngine" gap
flagged in [`VOLUME_PRETRAIN_RESULTS.md`](VOLUME_PRETRAIN_RESULTS.md) § *Session
update — 2026-05-21*. yngine generated the 200K-game pretraining corpus but
had never been used as a benchmark opponent.

## TL;DR

The currently-deployed `iter1_ema` model **swept yngine-MCTS-1,000 at both
sim budgets, 17-0 each**. SPRT (p0=0.50, p1=0.60, α=β=0.05) crossed the
"STRONGER" boundary in the minimum possible 17 games at both settings:

| Our MCTS sims | yngine MCTS sims | mode | games | record | Wilson 95% CI | verdict |
|---:|---:|:---|---:|---:|---:|:---|
| 200 | 1,000 | SPRT | 17 | 17-0-0 | [0.816, 1.000] | **STRONGER** |
| 800 | 1,000 | SPRT | 17 | 17-0-0 | [0.816, 1.000] | **STRONGER** |

W/B balance is clean at both budgets (9-W / 8-B; SPRT alternates colors).

**Honest read:** the deployed model is meaningfully stronger than yngine at
the level that *generated its own training data*. This is the expected
outcome — pretraining on yngine games then doing self-play on top should
yield a model that beats its teacher — but it had never been measured.
The result confirms the supervised + self-play loop is working as
intended at this compute level. It does **not** tell us how the model
holds up against yngine at higher compute (MCTS-10K, the level used in
the WAVE3 fingerprint audit), which is the natural next test.

## Source + protocol

**yngine source:** [`github.com/temhelk/yngine`](https://github.com/temhelk/yngine),
the C++ MCTS library used by [`temhelk/yinsh`](https://github.com/temhelk/yinsh)
as its AI backend. There is no native CLI binary — only a library — so this
harness ships a small stdin/stdout driver
(`third_party/yngine_driver/yngine_driver.cpp`) that wraps `Yngine::MCTS`
in a line-oriented protocol the Python bridge speaks.

The protocol is intentionally tiny:

```
< ready                         (handshake)
> apply P x y                   place ring at yngine coords
> apply M fx fy tx ty dir       ring move
> apply R fx fy dir             remove 5-marker row
> apply X x y                   remove ring
> go sims <N>                   search N iterations, return chosen move
> go time <secs> threads <T>    time-budgeted search
< move <wire>                   chosen move in the same wire format
> state | new | quit
```

Coordinate convention matches `scripts/yngine_corpus_to_npz.py`:
`yngine (x, y) → our Position(chr('A' + x), 11 - y)`. The six yngine
directions map to our hex deltas via `(dx, dy) → (dx, -dy)` (we mirror y);
see `yinsh_ml/yngine/move_codec.py` for the full table.

**Apple Silicon build:** `build.sh` patches yngine's `allocators.cpp` to add
`__APPLE__` to the mmap branch (upstream gates it on `__linux__` only) and
runs cmake from `third_party/yngine_driver/`. Idempotent — safe to re-run
after `git submodule update`.

**yngine bugs worked around:**

1. `MCTS::~MCTS` unconditionally `.join()`s an internal `search_thread`,
   which throws `std::system_error: Invalid argument` if no search ever
   ran (the thread is default-constructed). The driver runs a 1-iteration
   warmup search after every `MCTS` instantiation to leave the thread in
   a joinable state. Cheap (~µs on the empty placement board).
2. yngine's MCTS prints `DEBUG: …` lines unconditionally to `std::cout`
   (`mcts.cpp:237, 304, 406`). The driver `freopen`s stdout to
   `/dev/null` and uses a separately-duped fd for protocol replies, so
   the Python pipe only carries protocol traffic.
3. yngine's `ArenaAllocator` mmaps its full memory pool up front. The
   original 512 MB default crashed sequential eval games via the macOS
   OOM killer (~85% system memory pressure was being hit per game,
   and one process died mid-search with empty stderr). Dropped the
   driver-side default to 128 MB — still far above the ~50 MB MCTS-10K
   peak from the WAVE3 V2a fingerprint run on the cloud box.

**Default yngine level:** MCTS-1,000 — same as the corpus generation level
documented in `VOLUME_PRETRAIN_RESULTS.md` (the engine that taught the
deployed model). The earlier fingerprint audit
(`WAVE3_EXPERIMENT_LOG.md § V2a`) ran yngine at MCTS-10K against itself;
1K is materially weaker and more representative of what produced our
training data.

## Smoke benchmark — `models/iter1_ema_2026-05-27/iter1_ema.pt`

This is the currently-deployed model on `https://yinsh.jackflemingux.com`.

Both runs used a sequential probability ratio test (SPRT) instead of a
fixed-n match. The original plan was 100 fixed-n games at each setting,
but the first run reached 26-0 by game 26 — burning compute past the point
the conclusion was settled — and switching to SPRT (p0=0.50, p1=0.60,
α=β=0.05) terminates as soon as the LLR crosses ±2.94. The two MCTS-200
runs (fixed-n killed at 26-0 + SPRT 17-0) cross-check each other; the
MCTS-800 SPRT run was the only one for that budget.

### MCTS-200 (model) vs MCTS-1,000 (yngine) — SPRT

Command:

```bash
python scripts/eval_vs_yngine.py \
    --model-path models/iter1_ema_2026-05-27/iter1_ema.pt \
    --num-sims 200 --yngine-sims 1000 \
    --sprt --sprt-p0 0.50 --sprt-p1 0.60 --sprt-alpha 0.05 --sprt-beta 0.05 \
    --output logs/iter1_ema_vs_yngine_sims200_sprt.json
```

| metric | value |
|---|---|
| games played | 17 (SPRT terminated on upper boundary) |
| model wins | 17 (W: 9, B: 8) |
| yngine wins | 0 |
| draws | 0 |
| model WR | 1.000 |
| Wilson 95% CI | [0.816, 1.000] |
| SPRT LLR | +3.10 (upper bound +2.94) |
| verdict | **STRONGER** |
| per-game wall time | mean 75.8s (min 61.6s, max 125.3s) |
| total wall clock | ~21 min |

Game lengths: min 43 / mean 50.4 / max 83 moves. Longest game was a
51-move ring-shuffle exchange the model played as White (game 12);
still ended in a model capture.

### MCTS-800 (model) vs MCTS-1,000 (yngine) — SPRT

Command:

```bash
python scripts/eval_vs_yngine.py \
    --model-path models/iter1_ema_2026-05-27/iter1_ema.pt \
    --num-sims 800 --yngine-sims 1000 \
    --sprt --sprt-p0 0.50 --sprt-p1 0.60 --sprt-alpha 0.05 --sprt-beta 0.05 \
    --output logs/iter1_ema_vs_yngine_sims800_sprt.json
```

| metric | value |
|---|---|
| games played | 17 (SPRT terminated on upper boundary) |
| model wins | 17 (W: 9, B: 8) |
| yngine wins | 0 |
| draws | 0 |
| model WR | 1.000 |
| Wilson 95% CI | [0.816, 1.000] |
| SPRT LLR | +3.10 (upper bound +2.94) |
| verdict | **STRONGER** |
| per-game wall time | mean 288s (min 245s, max 373s) |
| total wall clock | ~82 min |

Game lengths: min 43 / mean 55.2 / max 81 moves. Mostly comfortable
wins; longest game (81 moves) was an 8-move ring-jockeying phase before
the model assembled a 5-row.

### Notes on what we don't have (yet)

- **No model losses to inspect.** The original plan included "sample
  positions where model lost cleanly vs yngine"; both SPRT runs were
  sweeps, so there's nothing to show. The natural follow-up is yngine
  at MCTS-10K (or higher), where the gap should narrow.
- **First MCTS-800 SPRT crashed on game 6** with the 512 MB pool —
  yngine_driver was OOM-killed by macOS. That run was discarded; the
  17-0 above is the second attempt with the 128 MB pool. The crash
  validated the bridge's error handling (it logs `err: yngine_driver
  closed stdout unexpectedly` and counts the game as a draw, which is
  excluded from the SPRT LLR) but ate ~30 minutes of compute.

## Interpretation

- **Is the deployed model beating yngine?** Yes, unambiguously, at the
  yngine-MCTS-1,000 level. Both 200- and 800-sim configurations achieve
  17-0 with SPRT lower bound ≥0.816, well clear of 0.5. The eval is
  decisive even with the smallest sample SPRT permits.
- **Does more model compute help (MCTS-200 → MCTS-800)?** Indistinguishable
  at this level — both are saturating yngine-1K. To resolve "is 800 sims
  better than 200 sims," we'd need a harder opponent (yngine-10K) or a
  direct model-vs-model head-to-head — the `eval_vs_frozen_anchor.py`
  loop already handles the latter.
- **At what level / sim budget does the picture flip?** Not measured yet.
  The fingerprint audit (`WAVE3_EXPERIMENT_LOG.md § V2a`) used yngine at
  MCTS-10K and described it as "a serious engine, not hobby code" — 10×
  more compute than the corpus level. If a frontier check is wanted,
  bump `--yngine-sims 10000`.

## Suggested next steps

1. **Sweep yngine levels.** Run the same SPRT against `--yngine-sims
   5000` and `--yngine-sims 10000`. The corpus level (1K) saturates;
   the question is where the model breaks. yngine at 10K still runs
   in single-digit seconds per move on this box (cloud V2a measured
   34 s/game at MCTS-10K self-play with 16 threads; per-move single-
   threaded should be similar order). Budget ~3-4× wall clock per level.
2. **Measure `models/yngine_volume_15ch_pretrain` for comparison.** That
   checkpoint is the pretraining-only baseline (no self-play). If it
   gets to ~50% WR vs yngine-1K, then the self-play loop's contribution
   is the entire 50→100 gap visible here. See `D2_PREP.md`.
3. **Track yngine WR alongside frozen-anchor WR.** The "iterate and
   promote" ladder in `eval_vs_frozen_anchor.py` is the cheap relative
   loop; this harness is the slow absolute measurement. Run it after
   every anchor promotion (or every Nth, if SPRT terminates fast) so
   relative gains can be sanity-checked against an external reference.
4. **Drop the yngine pool further if running many parallel games.**
   128 MB / process holds for sequential games; parallel eval might
   want 64 MB to keep system memory headroom.

## Repro

Build the driver once:

```bash
bash third_party/yngine_driver/build.sh
```

Tests:

```bash
python -m pytest yinsh_ml/yngine/tests/test_move_codec.py   # fast, no binary
python -m pytest yinsh_ml/yngine/tests/test_bridge.py -m slow   # subprocess-level
```

Override the binary path with `YNGINE_DRIVER=/abs/path` if you want to
run against a different build.
