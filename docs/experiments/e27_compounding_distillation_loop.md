# E27 — Compounding distillation loop (E26 iterated)

**Status:** QUEUED · **primary**
**Date(s):** scoped 2026-06-12
**Cost:** ~12–24 GPU-h / **~$13–27 per turn** (4090 @ $1.12/hr); compounding test 2–3 turns ≈ $30–65
**Branch / artifacts:** teacher = `e26_lc_full` (`models/e26_distilled_2026-06-10/e26_distilled.pt`); anchor = `iter1_frozen.pt` then `e26_lc_full`; `gen_distill_corpus.py --use-inference-server --inference-dtype bf16`; distill trainer; `scripts/eval_vs_frozen_anchor.py --sprt`.

## Description
Turn the AlphaZero crank again. E26 proved that distilling iter1_ema's
high-budget *search policy* into a fresh net beats the frozen champion at
tournament sims (~0.59–0.63 @ ≥400). That banked a **one-time** search premium
(~1.5× search-efficiency, ~60 Elo). E27 asks the load-bearing question for the
entire superhuman thesis: **does expert iteration compound, or does it saturate
after banking the search premium once?**

Mechanism: `e26_lc_full` is now the strongest engine at tournament sims, so it is
the strongest available *teacher*. Regenerate a corpus from **it** at 1600 sims →
distill again → `e26.2` → gate `e26.2` vs frozen `e26_lc_full` (the new anchor —
**not** iter1). Standard expert iteration, now with a teacher ~60 Elo stronger
than the last.

## Outcome
Pending — two sequential gates:
- **Gate 1 (promotion):** `e26_lc_full` beats frozen `iter1_ema` at ≥400 sims.
  This establishes the new fixed champion + anchor. (E26 already measured 0.589 @
  400 / 0.630 @ 1600; this is the confirmation/promotion action.)
- **Gate 2 (the real question):** `e26.2` beats frozen `e26_lc_full` at ≥400 sims.
  - **Compounds** (any well-powered CI clear of 0.50) → you have a flywheel;
    promote `e26.2`, iterate again, escalate search-infra (E29) since you now know
    you'll crank repeatedly.
  - **Saturates** (ties; CI brackets 0.50 at ≥400) → the gain was a one-shot
    absorption of the prior's search advantage, not self-improvement → redirect to
    the **capacity** lever (A4 cheap probe, then E17), because a saturated loop
    means you're hitting the architecture's representational ceiling (the E25
    finding).

Either outcome is forward motion — it's the result that re-ranks the whole
roadmap.

## Details
**Sequencing / gate discipline (learned from E26):**
- **Gate at ≥400 sims, never 96.** Strength-vs-search is U-shaped (Finding 7); the
  old ~96-sim gate sat in the trough and read E26's real plateau-break as a null.
- **Color-balance** every match (alternate which side is white; ~0.53–0.59
  first-player edge at this strength).
- **Distrust SPRT early-accepts** — they're biased toward the boundary (E26's 0.82
  @ 800 was 23-5 early-stop; true ≈0.62). Trust fixed-N / near-cap estimates.
- Corpus is **data-sufficient from ~150k positions** (E26 learning-curve); no need
  to over-generate. ~150–250k at 1600 sims.

**Free value-target rider (test the head E24 couldn't move):** the corpus stores
search *value* (1600-sim) alongside `policy_q`. E24 showed mirror@200 self-play
gives the value head no gradient — but 1600-sim search value is a genuinely
stronger target. Distill value too, as a one-line ablation, and read whether the
static head lifts off its 0.663 ceiling. (E25's 0.677 from-scratch cap predicts it
can't *without more capacity* — which is exactly the A4/E17 bridge — but it's
nearly free to check inside the loop.)

**Cost levers:** flip `inference_server_dtype: bf16` first (E20, ~1.3× free).
Corpus-gen is wide-parallel so the E20 inference server helps it; the ≥400-sim
gate is CPU/tree-bound (E26 infra finding) and does **not** benefit — parallelize
the gate across cores instead (E29).

## Provenance & links
- 2026-06-11 [[e26]] closeout (STRONGER; the lever this iterates); "Open threads /
  what's next" #1–2 name this run.
- 2026-06-09 [[e25]] (value head OUT → distil policy, not value).
- Gate discipline: Backlog Finding 7 (U-curve / gate at tournament sims).
- Downstream: [[e29]] (search-infra, escalate if this compounds), [[a4]]/[[e17]]
  (capacity, redirect if this saturates), [[e30]] (placement-value rider on this
  corpus).
