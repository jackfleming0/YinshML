# B4 — Disable promotion gate entirely

**Status:** QUEUED
**Cost:** ~6h, ~$10.
**Stack-rank:** Likely+ 2 / Unblocks 3 / Info-gain 4 / Cost 4 / Impl-risk 4 / Sum 17
**Dependencies / blocks:** Orthogonal to others.

## Description
**Goal:** run D.2 self-play with no gate — just take the latest checkpoint as the new "best" each iter. Tests whether the gate is helping or hurting.

**Mechanism:** as a negative control. If the no-gate run produces an iter_4 that's *worse* than the gated run's iter_4, the gate was net-positive (filtering out bad candidates). If the no-gate run produces an *equal or better* iter_4, the gate was net-zero or net-negative on this regime.

## Outcome
Pending — SPRT vs `best_iter_4` AND vs the gated D.2 result. Two-way comparison clarifies the gate's contribution. Gate is net-positive if no-gate iter_4 is worse than gated iter_4; net-zero/negative if equal or better.

## Details

**Supporting evidence:**
- D.2's loose gate promoted clearly-worse candidates. If a tight gate (B1) fixes that, no-gate would presumably be even worse — confirming the gate isn't useless, just mis-tuned. Useful as a calibration point.

**Reasons to not believe:**
- **Less informative than B1.** B1 directly tests the tuning hypothesis. B4 tests "does the gate matter at all" — coarser question.
- **Could waste compute.** If the gate is genuinely helping, no-gate run produces a worse model and the run is throwaway info-only.

**Methodology:**
```yaml
arena:
  promotion_threshold: 0.0  # promote any candidate
```

Or modify the supervisor to short-circuit the gate entirely.

**Open questions:**
- Is "always-promote" the same as "promote_threshold=0.0", or does the Wilson math handle 0.0 weirdly (e.g. divide-by-zero)? Check before running.

## Provenance & links
- Related: B1 (tighten Wilson gate to 0.50 — the tuning-hypothesis counterpart; B4 is the negative control). The B1+B2+B3 decision matrix routes to B4 when the gate froze the loop at iter_0 or proved ineffective.
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section.
