# E1 — GAP-native pretrain from scratch (Path 2)

**Status:** QUEUED
**Cost:** pretrain ~3h ($5) + self-play ~6h ($10) + SPRT ~1-4h ($2-5) = **~12h, ~$20**.
**Stack-rank:** Likely+ 2 / Unblocks 4 / Info-gain 4 / Cost 2 / Impl-risk 3 / Sum 15
**Dependencies / blocks:** Sequence after A1, A3, A4 — they all narrow the search space better than E1. Unblocks: a confirmed answer on the GAP architecture question (which has been hanging since D.1 v2).

## Description
**Goal:** train a 15-ch (or 6-ch?) supervised pretrain from scratch with the GAP value head as the architecture from epoch 0, then self-play, then SPRT. Tests whether D.1's failure mode was *warm-start specialization* (theory A) or *GAP-is-fundamentally-wrong* (theory B).

**Mechanism:** D.1 v1 + v2 both warm-started a GAP head from a spatial-head checkpoint, then self-played. Both failed SPRT 1-15-0 (structural determinism verified). Hypothesis A: the trunk's 30M params were tuned for the spatial head's output shape and brief self-play couldn't unlearn that. Hypothesis B: GAP is fundamentally wrong for YINSH (position-discarding via average pooling kills value signal). E1 tests A directly — a from-scratch GAP-native trunk has no specialization to overcome.

## Outcome
Pending — SPRT verdict. Distinguishes theory A (warm-start specialization, fixable) from theory B (GAP fundamentally wrong for YINSH, architecture dead-end).

## Details

**Supporting evidence:**
- D.1 v1/v2 SPRTs were verified as structural determinism (same trajectory across seeds), so the failures aren't noise. They're real — but they don't distinguish theory A from theory B.
- Network code supports `--value-head-type gap_v2` end-to-end (D.1 v2 used it).

**Reasons to not believe:**
- **GAP-native pretrain may just confirm theory B.** If the architecture fundamentally can't learn YINSH value, no amount of "fresh training" fixes it. Then E1 is throwaway compute.
- **6-ch GAP might be more sensible than 15-ch GAP.** GAP is averaging spatial info; doing it with 15 input channels gives more raw signal to start with but may not help if the head's bottleneck is the avg-pooling.

**Methodology:**
```bash
# Option A: 15-ch corpus (existing)
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_15ch_mmap/ \
    --use-enhanced-encoding \
    --value-head-type gap_v2 \
    --output-dir models/yngine_volume_15ch_gap_pretrain \
    --epochs 6 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12

# Option B: 6-ch corpus
# Same but without --use-enhanced-encoding and pointed at 6-ch corpus
```

Then full D.2-style self-play loop and SPRT.

**Open questions:**
- Should we test both 6-ch and 15-ch GAP-native? The 6-ch version is a cleaner test of "GAP itself works" (no encoding confound). Pick one and defer the other.

## Provenance & links
- Related: D.1 v1/v2 (the warm-start GAP runs that failed SPRT 1-15-0 — E1 is the from-scratch follow-up). Sequence after [[a1]], [[a3]], [[a4]].
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section.
