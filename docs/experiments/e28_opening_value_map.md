# E28 — Opening-value map from cached search Q-values (Route A)

**Status:** QUEUED · **cheap, runs on existing data — do early, parallel to E27**
**Date(s):** scoped 2026-06-12
**Cost:** ~0–1 GPU-h / **~$0–1** (analysis pass; runs on a laptop/CPU)
**Branch / artifacts:** reads the **existing** E26 corpus (`policy_q`, `policy_prior`, `policy_idx`, Zobrist per position; commit 9e0775a stores them). New script: `analysis_board/opening_map/read_placement_values.py` (to write).

## Description
Read YINSH opening value off the search tree we already trust, instead of
generating fresh games. The E26 corpus already stores, per position, the
search-improved value/visit signals from `e26_lc_full` at high sims. So the
opening valuation is **already computed** — we just have to aggregate it:

1. Filter to placement-complete / placement-node positions.
2. **Zobrist-dedup** — canonicalizes the order-independent placement configs (a
   ring at A5 on move 1 ≡ A5 on move 5; the hash already encodes board + side +
   phase, so identical configs collapse).
3. Aggregate search **Q-values + visit counts** per config (value under strong
   play, with the opponent's deviations modelled by search — *not* a depth-0
   self-play win%).
4. Cluster placement-complete positions (hand features: centroid distance,
   wall-vs-center ratio, ring-pair spread, symmetry score) and **characterize the
   high/low-value clusters in human-readable terms**.

This is the cheap, correct version of the "map the openings" idea. It avoids the
two fatal flaws of the depth-0-rollout approach: it uses a **strong** evaluator
(search Q, not the biased raw prior) and it never enumerates the ~10¹⁴ config
space (it reads only the configs the strong teacher actually explored).

## Outcome
Pending — this is the **diagnostic that aims the expensive opening spend**
([[e30]], [[e31]]). Gate: does the readout reveal **systematic placement
misvaluation** or an exploitable opening attractor?
- **Clean** (placement Q-values look well-calibrated, no lopsided attractor) →
  opening work is a *robustness audit*, not a strength lever → park E31, skip E30
  grounding. Cost avoided: ~$30–40.
- **Off** (the model systematically mis-values a class of placements, or collapses
  onto one orbit) → promotes [[e30]] (placement-value grounding) + [[e31]] (the
  strength-gradient sweep): opening becomes a **live strength lever E25 never
  checked** — E25 measured value-AUC *in aggregate*, never split out the placement
  phase, which [[e30]]'s H1 says is the one structurally value-blind region.

## Details
**Why search Q, not depth-0 win%:** depth-0 = the raw policy prior — the *biased,
attractor-prone head under suspicion*. Win% under weak-vs-weak play ≠ value under
strong play. Search Q-values are computed with strength and model the opponent
deviating, so they answer the question the rollouts can't.

**On the win% method (validated this session, for [[e31]] not here):** a
fixed-opening self-play win% *is* legitimate signal — it estimates the joint
position's imbalance under the evaluating model. Caveats: (a) it's a property of
the full 10-ring joint config + side-to-move, not of one player's placement; (b)
read it relative to a **first-mover baseline** (uniform-random openings, same
model both sides) so the zero point and magnitude are right — the baseline also
launders any per-color skill imbalance since it's constant across openings; (c)
it's value *at the evaluating strength*. E28 sidesteps (a)/(b)/(c)'s sampling cost
by reading cached Q directly; [[e31]] pays for them when a real attractor warrants
strong rollouts.

**Opponent-relativity (the caveat that matters for the goal):** a self-play
opening map tells you what's *balanced against the model*. That is **not** the
same as what *beats a human* — humans make specific exploitable errors, and
"trap-a-human" openings need the BGA/Boardspace human corpora, a separate question
(noted in [[e31]]). E28 answers "is this opening sound," not "does it beat humans."

## Provenance & links
- Design lineage: `analysis_board/opening_map/DESIGN.md` (the rigorous sweep =
  [[e31]]); `analysis_board/multiplayer/EXPERIMENT_opening_theory.md` (H1/H4,
  opening attractor).
- Corpus signals: [[e26]] (commit 9e0775a stores `policy_q`/`policy_prior`/
  `policy_idx`/Zobrist per position; also the seed for the puzzle-mode idea).
- Aims: [[e30]] (grounding fix) and [[e31]] (strength-gradient sweep).
