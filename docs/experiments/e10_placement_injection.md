# E10 — Random + symmetric placement injection in corpus generation

**Status:** QUEUED
**Date(s):** proposed 2026-05-29
**Cost:** not stated (corpus-generation change)
**Branch / artifacts:** applies to corpus generation (E7-style iter1-vs-iter1 or future engine-corpus iterations). Production-recipe slot in the layer-3 "Data + exploration" bundle.

## Description

For corpus generation, mix placement sources to give the model broad state coverage and an explicit symmetry signal during training:

- **40%** placement from H-vs-H human replay (initialization signal)
- **20%** from BGA marginal sampling (variety on the human distribution)
- **20%** uniform random (state-coverage diversity)
- **20%** symmetric augmentation of human placements (explicit symmetry signal during training)

Each placement contributes the full **D2 augmentation (4× data per position)** to ensure symmetric coverage.

## Outcome

Not yet run — QUEUED. Gate/signal: part of the layer-3 "Data + exploration" bundle of the production recipe; stacks once the L1/L2/L3a foundation is validated and solid. Provides the broad state coverage and explicit symmetry signal that complement E9's exploration knobs and E16's weight regularizer.

## Details

E10 is the corpus-generation half of the production recipe's third layer (the other halves being E9 phase-aware exploration knobs and E8/L3a symmetric MCTS at inference). The framing: H-vs-H placement is an *initialization* signal, not a target — "use H-vs-H to initialize away from uniform; use random + symmetric variants for exploration; let self-play decide" (the new framing replacing "force it to match human play").

The 20% symmetric-augmentation slice and the per-position 4× D2 augmentation directly bake symmetry into the training data — complementary to the inference-time symmetric MCTS (E8/L3a) and the training-loss symmetric-weight regularizer (E16). Random + symmetric placement starts give the model broad state coverage so it doesn't lock into a path-dependent modal opening.

## Provenance & links

- Spun out 2026-05-29 from the plateau-diagnostics / placement-pathology session.
- Related: [[e8_symmetric_mcts]] (L3a inference symmetry), [[e9_phase_aware_exploration]] (L3c exploration knobs), [[e16_symmetric_weight_reg]] (L3d training-loss symmetry), E7 (split-corpus pretrain this builds on).
- Cross-doc: `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`.
