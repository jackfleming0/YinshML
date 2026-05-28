# Session Recap — 2026-05-21
## Non-saturated yardstick + MCTS engine consolidation

### The arc (one causal chain, not three separate tasks)

Set out to build **step 1** of the ceiling-raising plan in `VOLUME_PRETRAIN_RESULTS.md`:
a measurement that isn't saturated (the HA ladder and a human are both maxed out).
Building it **caught a serious latent bug**, and chasing that bug to its root
**exposed dead duplicate-engine debt** that we then excised. Each thread caused the
next.

---

### 1. The yardstick — DONE & validated

- **Decision: frozen-checkpoint anchor, not yngine.** The gating question is
  *relative* ("does lever X beat our current best?"), which a frozen `best_iter_4`
  answers cheaply. yngine (the only *absolute* ceiling-detector) is deferred behind
  an explicit trigger: build the bridge only when (a) a long scaled run needs a
  collusion tripwire, or (b) a candidate convincingly beats frozen-best and we want
  the absolute number.
- **`scripts/eval_vs_frozen_anchor.py`** — candidate(s) vs a fixed anchor on the
  validated batched MCTS, color-split, opening-sampled to defeat the
  deterministic-side artifact, with `gc`/MPS-cache hygiene for long runs.
- **Validated it discriminates:** abandoned seeds → **WEAKER (0/20, balanced
  colors)**; equal-strength checkpoints → **inconclusive**. No longer saturated.
- **SPRT mode (`--sprt`)** — sequential test that stops the moment the result is
  decisive (fishtest-style). Both boundaries validated: STRONGER in 12 games,
  NOT_STRONGER in 9. This is the load-bearing fix for measurement coarseness.
- **Internal Elo beefed:** `arena.games_per_match` 20 → 100 (cheap raw-policy
  tournament; shrinks the ±150 noise band ~2.2×).

### 2. The bug the yardstick caught

The first validation read a clean `0.500` — but the **positive control** revealed
the harness was measuring *nothing*: it reused the legacy `search/mcts.py::MCTS.search()`,
which **never expanded the root** and returned a uniform-random, net-blind policy.
Every checkpoint "played" identically. **The negative control passed on broken
code; only the positive control caught it.** Rebuilt the harness on the working
batched engine (`search_batch`).

### 3. MCTS engine consolidation — DONE

- **Audit first** (which corrected my own `TECH_DEBT.md` §3): the broken
  `search/mcts.py::MCTS` had **zero production use** and **never worked end-to-end**
  (broken at multiple layers — fixing root-expansion just exposed a numpy/tensor
  crash underneath). Self-play training and the HA-ladder anchor gates always used
  the working engine, so **the project's headline results were never contaminated.**
- **Excised it.** One MCTS engine now: `training/self_play.py::MCTS`.
  **−2,649 / +137 lines across 16 files.** Full suite green — the 41 remaining
  failures are all pre-existing, MCTS-unrelated (tensor-pool / zero-copy / memory
  macOS-backend, a `StubGameState` mismatch); **zero in any file touched.**

---

### What we learned (the durable bits)

- **Positive controls catch dead instruments; negative controls don't.** A
  same-strength control reading ~0.5 is consistent with both a working *and* a dead
  instrument. Always include a known-*different* control.
- **Audit before deleting in a divergent-duplicate codebase.** The audit overturned
  my own written assumptions twice (the §3 landmine claim; "it's a one-line fix").
- **Fixing dead code can be worse than excising it.** Repairing `search()` just
  surfaced the next layer of rot; resurrecting an unused engine also works *against*
  consolidation.

---

### What comes next

1. **Step 2 — the ceiling experiment.** MCTS-400 (then maybe 1000) self-play vs
   frozen `best_iter_4`, screened with `--sprt` (set `--sprt-p1` to the smallest
   edge worth promoting). De-risked now: one correct engine, one working instrument.
   Wants the GPU box — borderline cases still need many games and MPS is ~16–50 s/game.
2. **Optional yardstick refinement** — a shared on-distribution opening book
   (common-random-numbers pairing) for variance reduction, *only if* SPRT alone
   still burns too many games. Theory-free; sourced from real play, not opening theory.
3. **Deferred:** the yngine absolute-strength bridge (see the trigger in §1).

### Artifacts
- **New:** `scripts/eval_vs_frozen_anchor.py`, `TECH_DEBT.md` (resolved record),
  this recap.
- **Updated:** `VOLUME_PRETRAIN_RESULTS.md`, `CLAUDE.md`,
  `configs/wave3_branchC_mcts200.yaml`.
- **Removed:** `yinsh_ml/search/mcts.py`, `search/performance_profiler.py`, and
  three dead-engine scripts + one perf test.
