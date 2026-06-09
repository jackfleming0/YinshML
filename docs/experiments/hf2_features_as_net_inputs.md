# HF-2 — Features as network inputs (the functional-form test)

**Status:** QUEUED (build-not-launch; RESCOPED after reading the encoder + Branch D.2)
**Date(s):** scoped 2026-06-03
**Cost / hardware:** A training run (cloud). No new code to build for the first move. Adding a channel would require a fresh pretrain (breaking change).
**Artifacts:**
- 6-ch baseline: a `basic`/6-ch config (`encoding.type: basic`)
- 15-ch enhanced: `configs/branchD2_enhanced_mcts200.yaml` (= branchC config but `encoding.type: enhanced`)
- Flag plumbing: `run_training.py:264-281` → `use_enhanced_encoding`
- Encoder: `yinsh_ml/utils/enhanced_encoding.py::EnhancedStateEncoder`

## Description

The direct test of the Phase 1 (HF) conclusion: "the info is real but a linear
heuristic weight can't exploit it; let the net weight it nonlinearly." Instead of
adding a feature as a *linear* heuristic term, feed it to the network as an input
channel so the net can combine it nonlinearly. This is the real test of the
mechanism finding (linear/differential functional form is the bottleneck, not the
feature set).

## Outcome

Pending — gated on the in-flight value-head run.

**Crucial discovery: most of this is already built.** `EnhancedStateEncoder`
(15-ch, `use_enhanced_encoding`) already feeds the network: ring mobility (ch 8),
partial rows (ch 6/7), row threats (ch 4/5), ring influence, center distance,
turn/score. So `ring_mobility` / `near_completion` / `potential_runs` as *network
inputs* are ALREADY tested — Branch D.2 (15-ch) came back **NOT_STRONGER** (Done
entry 2026-05-25), though confounded by the since-fixed `decode_phase`
reading-wrong-channel bug, so it's "inconclusive-leaning-null", not a clean
refutation. Recent symmetry runs (`sym15-*`) are 15-ch, so the path is in active use.

## Details

**The genuinely NEW thing HF-2 could add** is `defensive_disruption` as a new
channel — the defensive term that is NOT in the enhanced encoder, and the one
palette feature with both independent signal (R²=0.47) and a clear strategic
story (denying opponent runs).

**Build cost / risk:** adding a channel is a *breaking* change (15→16,
NetworkWrapper hard-fails on channel mismatch → requires fresh pretrain).
Untestable in a torch-free container. So: do NOT commit untested encoder surgery
blind. The right "build" is:
1. this scoping,
2. a clean post-bugfix 15-ch vs 6-ch A/B to get an *unconfounded* read on the already-built channels before adding a 16th,
3. only then the `defensive_disruption` channel.

**Recommended first move (cheap-ish, cloud):** re-run the *existing* 15-ch vs 6-ch
comparison now that the phase bug is fixed — it answers "do strategic features as
network inputs help?" without any new code. If that's still null, adding one more
channel is unlikely to change the verdict and HF-2 closes.

**No new code to build.** The matched pair already exists: a `basic`/6-ch config
vs `configs/branchD2_enhanced_mcts200.yaml`. The `encoding.type: basic|enhanced`
flag is fully plumbed (`run_training.py:264-281` → `use_enhanced_encoding`). So
"build HF-2" = pick the 6-ch baseline config + the enhanced config, launch both
post-bugfix, H2H the finals. Gated on the in-flight value-head run; nothing to
write.

## Provenance & links

- Direct test of the HF mechanism finding ("linear/differential form is the
  bottleneck, not the feature set"). See `completed/hf_feature_ablation.md`.
- Branch D.2 (15-ch) NOT_STRONGER Done entry (2026-05-25) — confounded by the
  since-fixed `decode_phase` channel bug.
- Part of the "Heuristic evaluation-form investigation (2026-06-03) — forward queue".
- Followed by HF-3 (pivot to learned-value); both launches gated on the in-flight
  value-head run.
