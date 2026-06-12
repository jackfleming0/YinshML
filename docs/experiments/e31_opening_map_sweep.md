# E31 — Opening-map strength-gradient sweep (Route B)

**Status:** QUEUED · **conditional / parked — triggered by E28 or an E27 stall**
**Date(s):** scoped 2026-06-12 (formalizes the long-standing `opening_map/DESIGN.md`)
**Cost:** full ~24–36 GPU-h / **~$27–40**; small first pass (4K games) ~4–6 GPU-h / **~$5–7**
**Branch / artifacts:** `analysis_board/opening_map/DESIGN.md` (design captured); scripts to write: `cluster_openings.py`, `play_games.py`, `analyze_outcomes.py`, `report.py`; run output under `opening_map/runs/<ts>/`.

## Description
The rigorous empirical opening valuation that **isolates opening quality from play
quality** — the confound `DESIGN.md` calls *primary*: a "bad" opening might lose
only because no model plays it well, and be fine under stronger play. Three-axis
sweep:

1. **~20 clustered opening variants** (cluster placement-complete positions from
   replay buffers — don't enumerate the ~10¹⁴ config space; K-means on hand
   features / penultimate embeddings, modulo D2).
2. **≥3 model strengths** (random / heuristic-d3 / a strong net like
   `e26_lc_full`) — the **gradient across strength is the signal**: flat-low =
   structurally bad; climbs with strength = "needs skill"; flat-even = robust.
3. **500–1000 games/cell**, **same model both sides**, fixed openings played out
   **at strength (≥400 sims, NOT depth-0)**, reported as **Δ-from-first-mover
   baseline**.

The fixed-opening self-play win% is well-posed (validated this session): it
estimates the joint position's imbalance under the evaluating model. Read relative
to the uniform-random first-mover baseline so the zero point is right and any
per-color skill imbalance is laundered out (it's constant across openings).

## Outcome
Pending — **conditional.** Triggers:
- [[e28]] surfaces a real exploitable attractor / placement misvaluation worth
  characterizing under strong rollouts; **or**
- [[e27]] **saturates** (loop stops compounding) and "what's the opening landscape
  / next strategic direction" becomes the live question (DESIGN.md files this as
  "next strategic direction," not "next training cycle").

Gate: which opening clusters show a strength gradient, and whether any are
structurally imbalanced after baseline correction. Run the **4K-game first pass**
to validate methodology before committing the full 40K-game sweep.

## Details
- **Why not depth-0** (the rejected shortcut): depth-0 = the biased raw prior under
  suspicion; win% under weak play ≠ value under strong play, and it bakes in the
  exact play-vs-opening confound the design exists to remove.
- **Attribution caveat:** win% belongs to the full 10-ring joint config +
  continuation + side-to-move, not to one player's placement; "opening favors
  black" can mean black played well *or* white played badly — counterfactual
  (vary one side, hold the other) is needed to attribute to a single placement.
- **Opponent-relativity:** this measures openings that are *balanced against the
  model* (sound), **not** openings that *beat a human* — the latter needs human
  error-distribution data (BGA/Boardspace), a genuinely separate experiment. Don't
  let the self-play map silently answer the human-facing question.
- Three metrics/cell (not just win%): Wilson-CI win rate, mean game length,
  mean value-cost-of-misalignment (how *hard* the opening is to play).

## Provenance & links
- Full design: `analysis_board/opening_map/DESIGN.md` (clustering, baseline,
  metrics, scope, open questions).
- Cheap precursor: [[e28]] (read cached Q first — it aims/triggers this).
- Related: [[e30]] (the grounding fix if openings are mis-valued), [[e9]]
  (placement exploration knobs), opening_theory H4 (is the cluster opening real?).
