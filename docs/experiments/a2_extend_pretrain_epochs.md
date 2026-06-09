# A2 — Extend D.2 pretrain to 9-12 epochs via `--resume`

**Status:** QUEUED
**Cost:** ~50min/epoch × 3-6 epochs = ~3-5h, ~$5-9.
**Stack-rank:** Likely+ 2 / Unblocks 2 / Info-gain 3 / Cost 5 / Impl-risk 5 / Sum 17
**Dependencies / blocks:** Blocked by nothing. The corpus + checkpoint + last_resume_state.pt are all on the box (or laptop). Unblocks: if PAcc climbs significantly (≥0.310) without VAcc moving, that reinforces the "value head is the bottleneck" reading and de-risks A4.

## Description
**Goal:** continue training the D.2 pretrain (`yngine_volume_15ch_pretrain`) from its 6-epoch state for another 3-6 epochs, using the resume support shipped in commit `d5e5151`.

**Mechanism:** PAcc was still climbing at epoch 6 (0.271 → 0.300 monotonic; +0.009 in the last epoch). The cosine LR schedule had bottomed out at eta_min ~1e-5, so the model was effectively done with the current schedule. A fresh cosine over 3-6 more epochs might extract additional signal.

## Outcome
Pending — new `best_supervised.pt` (overwritten in-place since output_dir is the same) with epoch 9+ metrics in the log. Optionally A1-style SPRT against `best_iter_4` to see if extra pretrain matters end-to-end. Promote signal: PAcc climbs significantly (≥0.310). Kill signal: extra epochs move PAcc but not VAcc (the more important head for self-play warm-start).

## Details

**Supporting evidence:**
- Epoch trajectory PAcc 0.271 → 0.275 → 0.277 → 0.284 → 0.291 → 0.300 — the delta from epoch 5→6 (+0.009) was larger than 4→5 (+0.007). Acceleration, not deceleration.
- Resume support is now tested (commit `d5e5151`); cosine schedule rebuilds fresh against the new `--epochs`, then advances to the resume point.

**Reasons to not believe:**
- **Val P-loss decrease was decelerating** (epoch deltas: -0.43, -0.03, -0.05, -0.06, -0.06, -0.09 — wait, that's actually accelerating again at the end). Mixed signal.
- **VAcc plateaued** (0.628 → 0.630 → 0.631 → 0.629 → 0.633 → 0.636 over 6 epochs — narrow band). Extra epochs may move PAcc but not VAcc, which is the more important head for self-play warm-start.
- **The D.2 self-play loop's behavior was driven mostly by iter_0's strength, not its precise PAcc value.** A 0.305 PAcc instead of 0.300 might not change the self-play outcome perceptibly.

**Methodology:**
```bash
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_15ch_mmap/ \
    --use-enhanced-encoding \
    --value-head-type spatial \
    --output-dir models/yngine_volume_15ch_pretrain \
    --epochs 9 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12 \
    --resume
```

Note: `--epochs 9` (or 12) is the NEW total epoch count. The resume logic will start from epoch 7 and run through 9. The cosine schedule rebuilds with T_max=9 and advances 6 steps to land at the right LR for "epoch 7 of 9."

**Open questions:**
- Is the in-place overwrite of `best_supervised.pt` correct, or do we want a separate output_dir for the extended run? Suggest: separate dir (`yngine_volume_15ch_pretrain_v2`) so we can compare both checkpoints.

## Provenance & links
- Related: [[a1]] (D.2 pretrain vs frozen anchor — same checkpoint), [[a4]] (value-head bottleneck hypothesis this de-risks), [[a3]] (fair-comparison pretrain epochs).
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section.
