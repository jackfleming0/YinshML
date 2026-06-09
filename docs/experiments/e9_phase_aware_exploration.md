# E9 — Phase-aware exploration knobs (placement Dirichlet / temp / FPU)

**Status:** QUEUED
**Date(s):** proposed 2026-05-29 (also proposed as E1 in the opening_theory doc — relabeled E9 here for clarity)
**Cost:** ~30 LOC; no cloud
**Branch / artifacts:** implementation target `yinsh_ml/training/self_play.py::MCTS` (gate existing knobs on `state.phase == GamePhase.RING_PLACEMENT`). Production-recipe slot L3c.

## Description

Per-phase Dirichlet + temperature + FPU settings, applied specifically during the ring-placement phase to force broad exploration there (rather than the global tapering schedule that currently lets the FPU + uniform-policy stall lock in a path-dependent modal opening):

- `placement_dirichlet_alpha: 1.0` (vs 0.3 globally)
- `placement_epsilon_mix: 0.5` (vs tapering 0.25 → 0.14)
- `placement_temperature: 1.0` (vs tapering 1.0 → 0.55)
- `placement_fpu_reduction: 0.0` (vs 0.25 — defensive against re-emergence of the FPU stall if the policy ever drifts uniform)

**Mechanism:** gate the existing exploration knobs in `yinsh_ml/training/self_play.py::MCTS` on `state.phase == GamePhase.RING_PLACEMENT`.

## Outcome

Not yet run — QUEUED as production-recipe layer **L3c** ("More diversity at placement"). Gate/signal: stacks on top of the validated foundation (L1 Dropout(0) + L2 label smoothing + L3a symmetric MCTS) once that foundation is solid. Intended to give the model more placement diversity so self-play can discover what wins under symmetry-respecting search, rather than converging path-dependently.

## Details

E9 is one of the three "data + exploration" measures in the 2026-05-29 final production recipe (layer 3): split corpus AND aggressive exploration knobs (phase-aware Dirichlet + temperature + FPU at placement) AND symmetric-MCTS at inference time. The H-vs-H placement is an *initialization* signal, not a target; the exploration knobs are what let self-play decide the actual answer.

This directly addresses the FPU + uniform-policy MCTS-stall mechanism diagnosed in P3 (the "first-visited child wins all visits" stall). Setting `placement_fpu_reduction: 0.0` is explicitly defensive against re-emergence of that stall.

Sits alongside L3b (iter1-corpus pretrain / E7) and L3d (symmetric weight regularizer / E16) as a "Not yet tested" layer in the post-E11/E14 production-recipe tables.

## Provenance & links

- Originally proposed as E1 in `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`; relabeled E9 in the backlog for clarity.
- Spun out 2026-05-29 from the plateau-diagnostics session.
- Related: [[e8_symmetric_mcts]] (L3a, same recipe), [[e10_placement_injection]] (L-layer corpus exploration), [[l1_l2_dropout_labelsmoothing]] (L1/L2 foundation), [[e16_symmetric_weight_reg]] (L3d), E7 (L3b iter1-corpus pretrain).
- Cross-doc: `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`.
