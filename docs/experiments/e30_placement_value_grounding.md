# E30 — Placement-value grounding (re-scoped to opening quality)

**Status:** QUEUED · **rider on the E27 corpus; gated by E28**
**Date(s):** scoped 2026-06-12 (originates as opening_theory-local "E2", built but never cleanly read)
**Cost:** ~4–8 GPU-h / **~$5–9 marginal** on top of [[e27]] (search_consistency adds ~+30–50% wall + its own gate)
**Branch / artifacts:** `search_consistency_placement_only` — **already in code**: `trainer.py:1512`, `supervisor.py:377`, `run_training.py:410`; config knob `trainer.search_consistency.placement_only`. Corpus = the [[e27]] 1600-sim corpus (already carries placement value targets).

## Description
Distill deep-search **value** into RING_PLACEMENT positions so the value head can
actually evaluate openings. The hypothesis (opening_theory **H1**): the value head
is **structurally blind during placement** — no markers ⇒ heuristic value ≈ 0, and
the NN value head only learns from terminal-outcome credit assigned back ~60 plies;
placement is ~15% of positions weighted 1.0 vs MAIN_GAME 2.0, so it sees ~7–8% of
effective training mass. With no per-position referee, the placement *policy*
drifts to a self-play attractor (the A5/K7 orbit imbalance; the "unheard-of" tight
cluster a friend flagged).

**Re-scoped (post-E25/E26).** This is **not** a strength-ceiling lever — E25 ruled
the value head out *in aggregate* (intrinsic ~0.66–0.68 AUC). It is an
**opening-quality / path-dependence** lever: does the model play sound, diverse
openings, or can a human opening-trap it? That matters specifically for "beat any
human," and it's a question E25 never touched (E25 measured value aggregate, never
split out placement — so H1 is *open, not refuted*).

## Outcome
Pending — **gated by [[e28]]** (only run if the cached-Q readout shows real
placement misvaluation; if placement Q is clean, skip). Re-scoped gate/metric (NOT
value-AUC, which E25 closed):
- **placement entropy** rises toward the human ~4.0 cluster-spread median (from the
  friend-anecdote ~2.0);
- the opening orbit **decorrelates** from the prior iteration / stops collapsing
  onto one attractor;
- **main-game strength non-regression** (H2H vs frozen champion ≥400 sims — the
  grounding must not cost main-game Elo).

## Details
- **Why it's nearly free here:** the [[e27]] 1600-sim corpus already contains
  deep-search value targets on placement positions. E30 is the one-line ablation of
  turning on `search_consistency` with `placement_only` and `value_weight=1.0,
  policy_weight=0.0` (value-grounding only). No new corpus.
- **History — untested, not refuted.** It only ever ran *bundled* inside the
  symmetry-fixes foundation run, which failed catastrophically (27% H2H; base
  already a 3:1 loss out of pretrain), so E30's effect was never isolated. Clean
  read still owed.
- **Suggested config** (from opening_theory E2): `search_consistency.enabled:
  true`, `policy_weight: 0.0`, `value_weight: 1.0`, `every_k_steps: 5`, `long_sims:
  128`, `warmup_iters: 0`, `placement_only: true`.

## Provenance & links
- Hypothesis + original design: `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`
  §H1 / §E2 (this is global-ID for that doc-local "E2"; cf. backlog [[e9]] which is
  that doc's local "E1").
- Bundled-and-confounded run: `SYMMETRY_FIXES_RUNBOOK.md` (E2 line; the 27% run).
- Aimed by [[e28]]; rides [[e27]]'s corpus; the value head it targets was ruled out
  *in aggregate* by [[e25]] but not on placement.
