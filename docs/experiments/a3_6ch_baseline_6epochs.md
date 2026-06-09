# A3 — Re-pretrain 6-ch baseline at 6 epochs

**Status:** QUEUED
**Cost:** ~3h on 5090, ~$5. SPRT after: 30 min - 4h.
**Stack-rank:** Likely+ 3 / Unblocks 5 / Info-gain 5 / Cost 4 / Impl-risk 5 / Sum 22
**Dependencies / blocks:** Need the original 6-ch yngine_volume mmap corpus locally or pull from gh release (regenerate the 6-ch mmap via `scripts/convert_npz_to_mmap_shards.py`, ~30 min — the npz survives in the gh release; the mmap was torn down). Blocks: any future "is encoding the lever?" question. Unblocks: A1's interpretation (if 6-ch-6-epoch beats `best_iter_4`, our whole baseline assumption was off).

## Description
**Goal:** train a fresh 6-channel supervised checkpoint to 6 epochs (same schedule as D.2's 15-ch pretrain) so we have an apples-to-apples comparison where only the *encoding* differs.

**Mechanism:** the current 6-ch baseline (`models/yngine_volume_pretrain/best_supervised.pt`) was trained 3 epochs. The doc notes it "climbed every epoch (VAcc 0.612 → 0.619 → 0.629), no overfit" — i.e. it was undertrained, not converged. Every D.2 comparison against this baseline carries a "more epochs OR more channels?" confound.

## Outcome
Pending — new checkpoint at `models/yngine_volume_6ch_pretrain_v2/best_supervised.pt`, plus a comparison-table entry in `VOLUME_PRETRAIN_RESULTS.md` showing per-epoch PAcc/VAcc trajectory next to 15-ch. Decisive read: does 6-ch-6-epoch alone already beat the 6-ch-3-epoch-plus-self-play champion (`best_iter_4`)? If so, the baseline assumption was off.

## Details

**Supporting evidence:**
- 6-ch baseline at epoch 3: VAcc 0.629, PAcc 0.286.
- 15-ch D.2 at epoch 3 (with T_max=6 schedule, so different LR curve): VAcc 0.631, PAcc 0.277. Slightly different despite "same epoch count" — the LR schedule shape differs.
- 15-ch D.2 at epoch 6: VAcc 0.636, PAcc 0.300. Of the +0.7 VAcc improvement vs 6-ch baseline, *some non-zero fraction* is purely the extra training budget, not the encoding.

**Reasons to not believe:**
- **Encoding might still be the dominant factor.** Even if a 6-ch-6-epoch beats 6-ch-3-epoch, it may still fall short of 15-ch-6-epoch. The comparison just lets us *quantify* the encoding contribution.
- **Reproducibility of the original 6-ch pretrain:** if we can't fully reproduce the 6-ch-3-epoch result (different random seed, different hardware), the baseline is shifting under our feet. Worth checking.

**Methodology:**
```bash
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_mmap/ \
    --output-dir models/yngine_volume_6ch_pretrain_v2 \
    --epochs 6 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12 \
    --value-head-type spatial
```

(Note: do NOT pass `--use-enhanced-encoding` — that's the whole point.)

Then evaluate:
- Val PAcc / VAcc per epoch (compare directly to D.2 15-ch trajectory)
- SPRT vs `best_iter_4` (does 6-ch-6-epoch alone already beat the 6-ch-3-epoch-plus-self-play champion?)

**Dependencies (detail):**
- Need the original 6-ch yngine_volume mmap corpus locally or pull from gh release. The 6-ch mmap was created on the original training box and torn down; the *npz* survives in the gh release. So step 0 is regenerating the 6-ch mmap (use `scripts/convert_npz_to_mmap_shards.py` — ~30 min).

**Open questions:**
- Do we ALSO want a 9-12 epoch run of the 6-ch baseline, in case it keeps climbing past 6? Cheap to extend via `--resume`. Defer until we see the 6-epoch result.

## Provenance & links
- Related: [[a1]] (interpretation this unblocks), [[a2]] (the same `--resume` extend question for 15-ch).
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section.
