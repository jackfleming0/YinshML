# Branch D.2 — Enhanced 15-channel encoding (prep doc)

Written 2026-05-23 as scoping work during the D.1 v2 overnight run.
Captures everything needed to execute D.2 from a cold start tomorrow,
including the strategic corpus-regeneration decision that gates the
heavy compute.

## Goal

Test the hypothesis that enriching the input encoding from 6 to 15
channels lifts the ceiling. The new channels expose tactical and
strategic features the current 6-channel encoding requires the net to
re-derive from raw piece positions (row threats, partial rows, ring
mobility, center distance, ring influence, turn number, score diff).
The prior is "+~100 ELO from improved feature representation" per
the encoder's own docstring; treat as optimistic, the real Step-2-style
test of "small/no edge" remains plausible.

See `ARCHITECTURAL_IMPROVEMENTS_PLAN.md` Phase 1 for the original framing.

## What's already built

- ✅ **`yinsh_ml/utils/enhanced_encoding.py::EnhancedStateEncoder`** —
  Full 15-channel encoder inheriting from `StateEncoder` (move-encoding
  API shared, so policy slots are identical → no policy-head changes
  needed). 637 lines, fully implemented, decode round-trips cleanly.
- ✅ **Tests**: `yinsh_ml/tests/test_enhanced_encoding.py` — covers
  channel shapes, side-to-move normalization, decode round-trip, etc.
- ✅ **Network plumbing**: `YinshNetwork(input_channels=15)` works.
  `NetworkWrapper(use_enhanced_encoding=True)` auto-detects from
  checkpoint or accepts explicit flag. The encoder + wrapper + supervisor
  + self-play workers all propagate the flag (`use_enhanced_encoding`)
  through `mcts_config`, same pattern as `value_head_type`.
- ✅ **Supervised pretraining script**: `scripts/run_supervised_pretraining.py`
  has `--use-enhanced-encoding` flag and `GameConverter(encoder=...)`
  takes an optional encoder.
- ✅ **Self-play config knob**: `encoding.type: enhanced` in YAML
  (read by `scripts/run_training.py:267`).
- ✅ **NEW (2026-05-23)**: `yngine_corpus_to_npz.py` now has a
  `--use-enhanced-encoding` flag — switches the encoder at conversion
  time. **This is the clean path IF the raw yngine shard files are
  available.**

## What's NOT built (the only real gap)

A **15-channel volume corpus**. The existing `yngine_volume.npz` was
encoded with the 6-channel `StateEncoder()`. We cannot use it directly
to pretrain a 15-channel net — input shape mismatch.

`scripts/run_supervised_pretraining.py` does not auto-detect the
corpus's channel count; passing `--use-enhanced-encoding` against a
6-channel corpus will fail at the first batch.

## The strategic decision: how to get a 15-channel corpus

Pick ONE of:

### Path A: Re-run yngine_corpus_to_npz on raw yngine shards

The cleanest option. Requires the raw `shard_*.txt` files (the input
to `yngine_corpus_to_npz.py`). These were intermediate artifacts from
the original volume corpus generation.

**Do we have them?** Probably NOT locally (the original VOLUME_PRETRAIN_RESULTS
.md noted the vast.ai box where they were generated was torn down).
The gh release `mcts400-session-snapshot-2026-05-23` has `yngine_volume.npz`
but no raw shards (verify with `gh release view` before committing to
the other paths).

**If they're available**:
```bash
python scripts/yngine_corpus_to_npz.py \
    --corpus-dir /path/to/shards/ \
    --output expert_games/yngine_volume_enhanced.npz \
    --use-enhanced-encoding
```
Cost: ~30 min on the box (replay 200K games, encode each position).
Disk: ~10-15GB output (states are 2.5x bigger but otherwise same shape).

### Path B: Decode + re-encode from the existing 6-channel npz

A workaround when raw shards aren't available. Read each (state, policy,
value) tuple from `yngine_volume.npz`, decode the 6-channel state back
to a `GameState` via `StateEncoder.decode_state`, re-encode via
`EnhancedStateEncoder.encode_state`. Save as a new 15-channel npz.

**The catch — minor information loss on channels 13 & 14:**
- `decode_state` recovers `current_player`, `phase`, `board pieces`,
  `rings_placed`, `scores` (verified in encoding.py docstring).
- It does NOT recover `game_state.move_count` → channel 13
  (turn number, normalized 0-1 capped at 100) will default to 0
  for every position. Lose this signal.
- Score differential (channel 14) IS recoverable (scores survive
  decode). No loss.
- All threat/structural channels (4-7, 8, 10, 11) are deterministic
  functions of board state → fully recoverable.

So Path B yields a 14/15-channel-equivalent corpus. Whether the lost
turn-number signal matters depends on how heavily downstream layers
rely on it — probably small, since the curriculum the original net
followed (epsilon-mix taper by move number) was implicit not explicit.

**Implementation: DONE 2026-05-24.** `scripts/regenerate_npz_with_enhanced_encoder.py`:
- Validates input is 6-channel (rejects already-15-channel with clear error)
- Loads input states into RAM (~6 GB for 13.6M corpus — fine on ≥64 GB box)
- Multiprocessing pool with one encoder pair per worker (initializer)
- Writes output to pre-allocated `np.lib.format.open_memmap` so RAM stays
  flat during the write phase
- Output layout matches `convert_npz_to_mmap_shards.py`: directory of
  `states.npy`, `policy_indices.npy`, `values.npy`, `total_moves.npy`,
  plus a `NOTES.md` documenting the channel-13 limitation per-corpus
- Tested in `yinsh_ml/tests/test_regenerate_15ch.py` (6 tests, 18s):
  end-to-end run, mmap-compatibility, channel-by-channel round-trip
  fidelity, channel-13 zeroed invariant, metadata passthrough, wrong-
  channel-count rejection, `--max-positions` truncation

Usage:
```bash
python scripts/regenerate_npz_with_enhanced_encoder.py \
    --input  expert_games/yngine_volume.npz \
    --output expert_games/yngine_volume_15ch_mmap/ \
    --workers 16  # default: cpu_count
```

The output `--output` directory is directly consumable by
`run_supervised_pretraining.py --data-dir <path>` (mmap-loading path).

**Cost**: ~1-2h CPU on a decent box (13.6M positions, mostly
EnhancedStateEncoder per-state work). ~10 GB output disk for the
uncompressed mmap shards (states alone are 13.6M × 15 × 121 × 4 B ≈ 9.9 GB).

### Path C: Generate a NEW corpus via self-play with current best model

Use Branch C `best_iter_4.pt` (or D.1 v2 if it ended up STRONGER) to
play games against itself, encoding with `EnhancedStateEncoder` at
each turn. Comparable corpus size = ~100K games (vs original 200K
yngine games — the yngine targets had argmax policy, so per-game
information content is lower than self-play with full visit
distributions).

This avoids yngine setup entirely AND gives a self-play-derived
corpus (different distribution from yngine — may be cleaner since it
matches our self-play loop's target distribution).

**Cost**: ~10-15h on 5090 (100K games × MCTS-200 / 4 workers).
**Disk**: ~30-50GB raw, maybe 10-15GB after npz conversion.

Effectively this is "do supervised pretraining with self-play data
instead of yngine data," which is a legitimate methodology shift.

### Path D: Skip pretraining entirely

Train D.2 from a randomly-initialized 15-channel net (no warm-start).
**High risk** — Branch C without value-grounded warm-start collapsed
82.5% → 60% (the original Branch C failure that volume-pretrain fixed).
The 15-channel net would have even less prior structure than 6-channel
since the new channels need to be learned from scratch.

Only use this as a last resort if all other paths fail.

## My recommendation: Path B first

Reasoning: Path B is the cheapest experiment that tests the encoding
hypothesis. If it works, great — we can later upgrade to Path A or C
for the full effect with channel 13 restored. If it FAILS with Path
B's data, the encoding probably isn't the lever; we'd save the
~10-15h of Path A/C compute.

The "channel 13 zeroed out" caveat is real but small. The other 14
channels (especially threats, partial rows, mobility) carry the
hypothesized information lift.

**Tomorrow execution plan (Path B):**

```bash
# 1. New ≥100GB vast.ai instance (5090 if available — same speed budget as D.1)
# 2. Clone + venv + cu128 torch
# 3. Pull the 6-channel corpus
gh release download mcts400-session-snapshot-2026-05-23 \
    --repo jackfleming0/YinshML --pattern yngine_volume.npz \
    --dir expert_games/

# 4. (~1-2h CPU) Re-encode to 15 channels via decode + re-encode.
# Script outputs mmap-shard format directly (no separate npz→mmap step).
python scripts/regenerate_npz_with_enhanced_encoder.py \
    --input  expert_games/yngine_volume.npz \
    --output expert_games/yngine_volume_15ch_mmap/

# 5. (~3h on 5090) Pretrain a 15-channel net (use gap_v2 head if v2 was good)
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_15ch_mmap/ \
    --use-enhanced-encoding \
    --value-head-type {spatial | gap_v2 — pick from v2 result} \
    --output-dir models/yngine_volume_15ch_pretrain \
    --epochs 3 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12

# 6. (~7h on 5090) Run Branch D.2 self-play loop warm-started from that
python scripts/run_training.py \
    --config configs/branchD2_enhanced_mcts200.yaml \
    --init-checkpoint models/yngine_volume_15ch_pretrain/best_supervised.pt

# 7. SPRT screen vs frozen best_iter_4
python scripts/eval_vs_frozen_anchor.py \
    --candidate runs_branchD2/<TS>/iteration_4/checkpoint_iteration_4_ema.pt \
    --anchor models/branchC_volume_pretrain/best_iter_4.pt \
    --sprt --sprt-p1 0.60 --sprt-max-games 400 \
    --device cuda --quiet-mcts \
    --output-json logs/branchD2_iter4_vs_frozen.json
```

**Total wall-time estimate: ~12-15h on 5090.** Roughly:
- Setup + corpus download + re-encode: ~30-45 min
- Supervised pretrain: ~3h
- Self-play training: ~7h
- SPRT: ~10 min - 4h depending on decisiveness

## Risks / failure modes

1. **The encoded 15-channel corpus produces an unhealthy pretrain
   (high val loss, low val accuracy)**. Indicates either the
   regeneration corrupted something or the encoder itself has a bug
   on real-world states. Check val PAcc/VAcc on the same metrics the
   original 6-channel pretrain hit (val PAcc 0.286, VAcc 0.629). If
   substantially worse, re-investigate corpus regeneration before
   proceeding to self-play.
2. **Self-play loop appears to train (loss decreases, anchor at
   95-100%) but SPRT comes back NOT_STRONGER** — same failure mode as
   D.1 v1. Diagnostically informative: would mean the 15-channel
   features aren't useful enough to compose with the existing trunk
   capacity. Path forward: try D.3 (SE blocks) or revisit D.1.
3. **Anchor drops below 100% during training** — warm-start broke or
   the 15-channel net can't replicate the spatial pattern. Audit by:
   (a) checking input shape matches (6 → 15), (b) confirming the
   decode-reencode preserved enough information, (c) trying Path A
   instead.

## Test plan (before launch)

These should ALL pass on the cloud env before kicking off the heavy
training run:
- `pytest yinsh_ml/tests/test_enhanced_encoding.py` — encoder
  round-trip, shape, side-normalization
- Manual: instantiate `NetworkWrapper(use_enhanced_encoding=True,
  value_head_type=gap_v2)` (or spatial), feed a real game state, confirm
  forward shapes
- Manual: regenerate a TINY corpus subset (~1K positions) via Path B,
  confirm decode round-trips channels 0-12 + 14 cleanly
- Manual: 1-iter supervised pretrain smoke (epoch=1, batch=32, ~500
  positions) → confirm loss decreases

## Files added by this prep

- `scripts/yngine_corpus_to_npz.py`: added `--use-enhanced-encoding`
  flag (Path A enabler)
- `configs/branchD2_enhanced_mcts200.yaml`: the Branch C MCTS-200
  config with `encoding.type: enhanced` (Path B + others)
- `D2_PREP.md`: this doc
- **`scripts/regenerate_npz_with_enhanced_encoder.py`** (added after v2
  resolved): Path B implementation. Multiprocess decode→re-encode with
  mmap output. ~1-2h CPU on the 13.6M corpus.
- **`yinsh_ml/tests/test_regenerate_15ch.py`**: 6 tests pinning the
  regenerator's invariants — end-to-end runs, mmap-compatibility,
  channels 0-12+14 round-trip cleanly, channel 13 is all-zeros, metadata
  passthrough, wrong-channel-count rejection.
