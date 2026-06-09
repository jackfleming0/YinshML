# C1 — Branch D.3: SE (squeeze-and-excitation) blocks

**Status:** QUEUED
**Cost:** ~12h pretrain + self-play, ~$20. Plus coding (~half-day).
**Stack-rank:** Likely+ 2 / Unblocks 3 / Info-gain 3 / Cost 2 / Impl-risk 4 / Sum 14
**Dependencies / blocks:** De-prioritized in light of D.2's findings; sequence behind A4, B1, D1.

## Description
**Goal:** add channel-attention SE blocks to the trunk, alongside the existing spatial-attention blocks. Tests whether selective per-channel feature emphasis helps.

**Mechanism:** spatial attention asks "which board cells matter" — already present. SE blocks ask "which feature channels matter" — orthogonal axis. ~5K extra params per block; cheap. Well-attested in Leela / KataGo.

## Outcome
Pending — SPRT verdict.

## Details

**Supporting evidence:**
- The doc has this teed up as the next architecture experiment after D.2.
- Mechanism is sound; the question is just whether YINSH benefits enough to justify the run.

**Reasons to not believe:**
- **D.2's near-zero ceiling movement weakens C1's prior.** If 15-ch encoding (which added 9 input channels of explicit features) barely moved metrics, channel attention adding implicit per-channel scaling probably also won't.
- **More likely to compound with other changes than stand alone.** SE + regression value head (A4) + lowered LR (B2) bundled might be much more than the sum of parts.

**Methodology:** add SE block module, integrate into trunk, run pretrain + self-play + SPRT.

## Provenance & links
- Related: [[a4]] (regression value head — candidate to compound with), B2 (lowered LR — candidate to compound with). Sequence behind A4, B1, [[d1]].
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section.
