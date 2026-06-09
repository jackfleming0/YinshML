# HF-1 — Cheap re-fit check (re-fit the original 6 weights)

**Status:** DONE: WORSE / NOT_STRONGER
**Date(s):** 2026-06-03
**Cost / hardware:** Cheap. Re-fit on 300 self-generated games (no parquet in container); offline + A/B.
**Artifacts:**
- Full writeup: `docs/experiments/completed/hf1_refit_results.md`
- Self-play data: `docs/experiments/refit6_selfplay.json`

## Description

The cheapest remaining untested heuristic-level axis: re-fit the original 6
production weights from game outcomes, rescaled to baseline's per-phase L1 budget
(pure reallocation — no extra magnitude), then A/B the re-fit agent vs the
hand-tuned baseline. Tests whether outcome-correlation-fit weights beat the
hand-tuned play-validated weights.

## Outcome

**WORSE — re-fitting hurts, significantly.**

- A/B vs baseline: **62–238, win-rate 0.207, −234 Elo, significant.**

**Lesson:** outcome-correlation ≠ good move-selection weight. In self-play data
the winner accumulates everything by the endgame, so the fit learns
reverse-causation and over-optimizes winner's-end-state correlates. The
hand-tuned baseline (validated for play) beats raw outcome-fits decisively.

## Details

- Re-fit done in-container on 300 self-generated games because no parquet dataset
  was available in the container.
- Weights rescaled to the baseline's per-phase L1 budget — a pure reallocation of
  weight across the 6 features, not a magnitude change, so the comparison isolates
  *which* features get weight, not *how much* total.
- Full writeup with per-phase weight deltas and methodology:
  `docs/experiments/completed/hf1_refit_results.md`.

## Provenance & links

- Closes the heuristic-*weight* axis negative, complementing Phase 1's (HF)
  heuristic-*feature* negative (`hf_feature_ablation.md`).
- Part of the "Heuristic evaluation-form investigation (2026-06-03) — forward queue".
- Next per sequencing: HF-2 (features-as-network-inputs, build-not-launch — gated
  on the in-flight value-head run).
