# Session prompt — break the YinshML plateau (post-E24)

Paste this into a fresh session. It hands off the current state and the next moves.
Full detail lives in `EXPERIMENT_BACKLOG.md` → **"Post-E24 lever board (2026-06-07)"**
and the E25/E26 entries; the E24 result is the Done entry "E24 Phase 1a".

---

## Mission
**Find and pull a lever that actually raises model strength.** We are NOT at a
stopping point — the goal is a model that beats the reigning champion
`iter1_ema_2026-05-27` head-to-head. Every experiment cashes out in **H2H vs a
frozen copy of iter1_ema** (color-balanced); nothing else is a verdict (R1).

## Where we are (and why the obvious read is incomplete)
The champion is undefeated across four plateau-break attempts:
- **E19** (deeper MCTS in the loop) — flat.
- **E22** (cross-teacher self-play) — failed (corrupted the policy).
- **E24** (real-LR sweep 3e-5/1e-4/3e-4) — NOT_STRONGER; value-head AUC never
  lifted off 0.737, hotter LR only eroded it (3e-4 H2H *collapsed* 53→32%).

**Critical:** all four **poked the mirror-continuation self-play loop while keeping
the same value head.** That's *one class* of lever, now exhausted — not the space.

The "value head is at a ceiling" story rests on two weak foundations:
1. It was measured on **noisy human games** (0.737 AUC) — enthusiast play has a
   blunder floor no evaluator can predict, so that number partly measures human
   unpredictability. We still don't know the head's discrimination on **clean,
   strong** positions (the one attempt — `gen_engine_labeled_corpus.py` — was OOD).
2. It was only ever trained with **targets no better than itself** (mirror
   self-play at 200 sims → MCTS target ≈ raw net → no gradient).

So the untested lever is: **inject a stronger, cleaner signal at scale.**

## What to do — in this order (cheap diagnostics AIM the expensive swing)

### 1. Ship free strength now (Lever B) — no training
- **E18:** deploy symmetric MCTS to the analysis board (`git push` + `yinsh-redeploy`;
  code already shipped, commit 09a6d86). +6–22 pp WR at inference, closes the
  friend-tester loop. Do it regardless.
- Also: simply play at **higher MCTS budget** — amplification is real (P1).

### 2. Run E25 — the binding-constraint diagnostic (Lever C) `[~1 day, mostly offline]`
This tells us where to point the big run. Two parts:
- **Clean value-eval:** build a *representative* held-out set from **strong
  self-play / high-budget-MCTS** positions (NOT heuristic-generated — that was
  E24's OOD mistake), labeled by outcome; re-measure value-head AUC with the
  existing `scripts/value_head_calibration.py`. Does iter1_ema read **well above
  0.737** on clean positions? If yes, the "ceiling" was a human-noise artifact.
- **Policy-vs-value ablation:** in MCTS, hold one head fixed and vary the other
  (e.g. flat policy + real value vs real policy + uniform value) and H2H the
  variants — *which head actually bounds played strength?* We've been assuming
  value; measure it.
- **Output:** a verdict on the real bottleneck (value / policy / capacity) that
  configures step 3.

### 3. Commit the big chips to E26 — high-budget-search distillation (Lever A) `[the top bet]`
*Only after E25 aims it.* Expert iteration done right: generate data from a
deliberately **stronger teacher** — iter1_ema (or an E21 ensemble) at **very high
MCTS budget (1600–3200+ sims)** — and **supervised-distill** those
(search-improved policy, search-value) targets into the net. The point: *the
target must EXCEED the student.* Search manufactures the better signal (it does
NOT require the value head to spontaneously improve); distillation banks it.
- Needs the **E20 throughput build** (high-sim self-play is CPU-bound) or
  patience/compute.
- If E25 says the bottleneck is **policy/capacity** instead, pivot to **A4**
  (scalar regression value head) / a bigger trunk rather than E26.

## Guardrails (learned the hard way)
- **H2H vs the FIXED champion `iter1_ema` is the only verdict.** In-loop tournaments
  green-checked a known loser; don't trust them.
- **Read the n≈8000 value-AUC over the n=60 H2H** unless H2H clears its ~±13 pp CI
  (E24: two arms slope ±3.3 in opposite directions — pure noise).
- **Do NOT measure the value head on heuristic-generated or human-only corpora** and
  call it the ceiling (the OOD trap; "best Brier" can be the head giving up).
- **Cheap-first:** a probe that could redirect a multi-day run is worth more than the
  run. Change one variable at a time.
- **Scope honesty:** E24 only tested *continuation + mirror + real LR*. Don't
  overclaim "self-play fails"; do claim "the mirror-continuation loop won't move
  this value head."

## Key pointers
- `EXPERIMENT_BACKLOG.md` → "Post-E24 lever board", E25, E26, E18, E20, E21, A4;
  Done entry "E24 Phase 1a".
- `docs/experiments/e24_phase1a_results.md` — the full E24 result + lessons.
- `scripts/value_head_calibration.py` — value-head AUC/Brier harness (use a
  **clean** `--data` corpus).
- `scripts/measure_h2h.py` + the lr-tag-aware summarizer in `scripts/e19_summarize.py`
  (generalized by PR #32) — H2H vs frozen.
- Champion: `models/iter1_ema_2026-05-27/iter1_ema.pt` (15ch). Keep a frozen copy
  as the yardstick.
