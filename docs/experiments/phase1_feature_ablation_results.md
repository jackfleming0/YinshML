# Phase 1 feature ablation — results (depth-1 breadth)

**Verdict: well-powered NULL.** None of the 5 experimental palette features,
added to the production heuristic as a fairly-weighted linear term, improves the
negamax HeuristicAgent at depth 1.

## Setup

- Tool: `scripts/experiments/ablation_phase1.py --normalize` (contribution-budget
  weighting so every feature contributes a comparable amount regardless of its
  raw magnitude).
- Each arm = the fixed baseline 6-feature weights **plus exactly one** palette
  feature → `base` vs `base+X` is a one-variable change.
- 300 games/cell, colors alternated, paired seed schedule, depth 1.
- Wilson 95% CIs; "significant" = CI excludes 0.5.
- Raw results: `docs/experiments/phase1_depth1_results.json`.

## Results (win-rate of base+feature vs base)

| Feature | budget | win-rate | 95% CI |
|---|---|---|---|
| defensive_disruption (placebo) | 0 | 0.483 | [0.43, 0.54] |
| defensive_disruption | 4 | 0.483 | [0.43, 0.54] |
| defensive_disruption | 6 | 0.493 | [0.44, 0.55] |
| defensive_disruption | 8 | 0.473 | [0.42, 0.53] |
| ring_mobility_differential | 4 | 0.513 | [0.46, 0.57] |
| ring_mobility_differential | 8 | 0.497 | [0.44, 0.55] |
| ring_confinement_pressure | 4 | 0.487 | [0.43, 0.54] |
| ring_confinement_pressure | 8 | 0.490 | [0.43, 0.55] |
| near_completion_threats | 4 | 0.480 | [0.42, 0.54] |
| near_completion_threats | 8 | 0.467 | [0.41, 0.52] |
| marker_tempo_differential | 4 | 0.483 | [0.43, 0.54] |
| marker_tempo_differential | 8 | 0.473 | [0.42, 0.53] |

Every arm clusters on the placebo (~0.48–0.51); every CI brackets 0.5.

## How we got here (the rigor that mattered)

1. A first run (80 games) showed `defensive_disruption` at 0.600 — a tempting
   "lead". Powering up to 300 games collapsed it to 0.483 (flat across doses).
   **The lead was small-sample noise.**
2. An early `ring_mobility @ raw-weight-4` arm was a *scaling artifact* (it
   contributed ~31 to the eval, dominating the tuned 6). Fixed with
   contribution-budget normalization; re-tested fairly → still null.

## Interpretation

- **Concluded:** the "new strategic features fix the evaluator" hypothesis is
  falsified *at the agent level* — as linear heuristic terms, they don't help.
- **Not ruled out:** the features as **network inputs** (nonlinear, learned
  weighting — the enhanced-encoder path), or **re-fitting the original 6**
  (a separate, untested axis).
- This null is consistent with the original review's deeper point: the static
  eval is myopic (multi-move sequencing blindness), and adding more *differential*
  terms to a depth-limited linear evaluator doesn't address that. Suggests
  heuristic-feature-tuning has low headroom; leverage is likely in the
  learned-value / network-input direction.

## Why the null? (mechanism diagnostics)

Three cheap diagnostics over real positions, to avoid concluding a null without
understanding it:

1. **Not tactical override.** Of evaluated positions, only ~5% are decided by
   the terminal/tactical detectors; **95% are decided by the weighted
   features.** So the features have full opportunity to act — the null isn't
   "tactics drown them out".
2. **Redundancy is only partial.** R² of each palette feature linearly predicted
   by the 6 production features: near_completion 0.61 (largely redundant),
   marker_tempo 0.57, ring_confinement 0.58, defensive_disruption 0.47,
   **ring_mobility 0.25 (mostly independent).**
3. **The decisive case:** `ring_mobility_differential` carries genuinely new
   information (R²=0.25) and **still doesn't help.** So "redundancy" can't be the
   whole story.

**Conclusion — it's the functional form, not the feature set.** The information
is real and largely available, but a single *linear, differential, single-
position* weight can't exploit signal whose value is contextual/nonlinear
(mobility helps in some positions, hurts in others; one scalar weight can't say
that). The bottleneck is that the heuristic is linear, not that it's missing
features. This is a direct argument for nonlinear/learned weighting (features as
network inputs; learned value head) over more hand-crafted linear terms.

## Decisions this drove

- **Did NOT run the depth-2/3 fade test.** Depth 1 is where shallow-search
  defense should help most; a null there means there is no depth-1 effect to
  fade. Running it would have cost ~13h to confirm a non-effect.
