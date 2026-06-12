# E17 — D2-equivariant network

**Status:** QUEUED · **tertiary / Lever D (promoted from Unscoped 2026-06-12)**
**Date(s):** deferred when [[e11]] confirmed H_W was fixable via [[e16]]; re-scoped 2026-06-12
**Cost:** ~15–30 GPU-h / **~$17–34** (full re-pretrain + loop + gate) + dev risk
**Branch / artifacts:** `yinsh_ml/network/model.py` (equivariant trunk); pre-check via the [[e11]] weight-symmetry diagnostic on `e26_lc_full`.

## Description
Bake D2 board symmetry into the **architecture** (an equivariant trunk) instead of
averaging it at **inference** ([[e8]]/E18) or regularizing toward it in the
**loss** ([[e16]]). The three approaches are not redundant — they sit at different
points on the cost/permanence curve, and E17 is the only one that makes symmetry
*free at runtime and exact*.

**Why now (the E26 corollary):** the distilled net **learned symmetry on its own**
— test-time D2 averaging went *null* on `e26_lc_full` (0.504, n=250), i.e. it had
nothing left to correct. That means the current architecture **spends capacity
relearning a symmetry you could bake in**. An equivariant trunk hands those
parameters back to actual position evaluation — which is exactly the representational
**headroom** A4/E25 say the value head needs (E25: from-scratch caps held-out AUC
at 0.677 while train→0.988 — a capacity/architecture ceiling).

## Outcome
Pending — SPRT vs the frozen champion. **Cheap pre-check first:** run the [[e11]]
weight-symmetry diagnostic on `e26_lc_full` (prediction: far less value-drift than
iter1's 2.8× range). If confirmed it also banks a free corollary — symmetric-MCTS
inference is redundant for distilled-family models (cheaper analysis-board
inference). Only **escalate to the full build if [[e27]] stalls and [[a4]] doesn't
unlock the value head** — i.e. when the loop has proven it needs more ceiling, not
just more turns.

## Details
- **Symmetry lever map:** [[e8]]/E18 = test-time averaging (real on *asymmetric*
  nets, deployed; **null** on the already-symmetric distilled net). [[e16]] =
  training regularizer (prototype, default-off, unvalidated). E17 = architecture
  (exact + runtime-free, highest impl cost). Post-E26, inference-time symmetry is
  "already captured" by distilled nets, which *raises* the relative appeal of
  baking it in structurally vs paying for it every forward pass.
- **Impl risk** is the main cost (a real architecture change + re-pretrain), which
  is why it's gated behind the cheap A4 capacity probe and the E27 stall condition
  — don't spend a re-pretrain on capacity until the loop proves capacity is the
  binding wall.
- Sibling deferred diagnostics E12/E13/E15 live in the [[e11]] detail file.

## Provenance & links
- [[e11]] (H_W confirmed: weights asymmetric — the original motivation) / [[e14]]
  (H_E ruled out) / [[e16]] (the regularizer alternative).
- [[e26]] Result 3 (D2 averaging null on the distilled net → the "capacity spent
  on relearning symmetry" argument).
- Capacity siblings: [[a4]] (cheap value-head probe — run first), [[e25]] (the
  intrinsic-ceiling finding this aims to raise).
- Gated by [[e27]] (only if the loop stalls).
