# HF-3 — Pivot to learned-value (AlphaZero direction)

**Status:** QUEUED (parked; references `TRAINING_REFACTOR_PLAN.md`)
**Date(s):** scoped 2026-06-03
**Cost / hardware:** Biggest effort of the three. Overlaps the current cloud value-head experiment.
**Artifacts:**
- Plan reference: `TRAINING_REFACTOR_PLAN.md`

## Description

The strategic redirect: demote the heuristic to a prior and trust the learned
value head. Rather than continuing to fix the heuristic's evaluation form (HF-1
weights, HF-2 channels), move the evaluation burden onto the network's learned
value — the AlphaZero direction.

## Outcome

Pending — parked. Revisit once the in-flight cloud value-head experiment lands
(this work overlaps it). This is where the evidence points for real strength
gains, but it is the biggest effort.

## Details

- Biggest effort of the three forward options.
- Where the evidence points for real strength gains.
- Overlaps the current cloud value-head experiment — revisit once that lands.

**Sequencing rationale (why this order):** HF-1 is cheap and closes the only
untested heuristic-level axis; HF-2 is the real test of the mechanism finding but
costs a training run; HF-3 is the strategic redirect. HF-1 and HF-2 are both
buildable now; HF-2/HF-3 launches are gated on the in-flight value-head run.

## Provenance & links

- Strategic successor to HF (feature ablation null) and HF-1 (weight re-fit
  worse) — both closed the heuristic axis negative, pointing toward learned value.
- See `completed/hf_feature_ablation.md`, `completed/hf1_refit_weights.md`,
  `hf2_features_as_net_inputs.md`.
- Part of the "Heuristic evaluation-form investigation (2026-06-03) — forward queue".
- Detailed direction in `TRAINING_REFACTOR_PLAN.md`.
