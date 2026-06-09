# HF — Heuristic feature ablation (palette as linear terms)

**Status:** DONE: NULL / NOT_STRONGER
**Date(s):** 2026-06-03
**Cost / hardware:** CPU only (4 cores, parallel). Depth-1 ablation; staging avoided a ~13h depth-2/3 fade test.
**Artifacts:**
- Full writeup: `docs/experiments/completed/phase1_feature_ablation_results.md`
- Raw JSON: `docs/experiments/phase1_depth1_results.json`
- Palette features: `yinsh_ml/heuristics/experimental_features.py`
- Parallel harness: `match_runner.py`; liveness guard: `tests/test_feature_liveness.py`

## Description

Spurred by a coaching review of a strong human game (BGA 862307561, see
`docs/game_reviews/bga_862307561_review.md`). Fixed two dead heuristic features,
built a configurable-feature-set evaluator + a contribution-normalized
agent-ablation harness, and ran a well-powered depth-1 ablation of all 5
experimental palette features added one-at-a-time to the fixed baseline-6.

Dead-feature context fixed before the run:
- `potential_runs_count` was identically 0 — a latent bug.
- `completed_runs_diff` legitimately ~0 due to same-turn row removal.

## Outcome

**NULL** — no palette feature, at any fair weight, beats the baseline agent-vs-agent at depth 1.

- 300 games/cell; every arm 0.467–0.513 win-rate vs baseline; every Wilson 95% CI brackets 0.5. Placebo (`@0`) = 0.483.
- No feature, at any fair weight (contribution budget 4 or 8), beats baseline at depth 1.
- An earlier 80-game run showed `defensive_disruption` at 0.600; powering to 300 games collapsed it to 0.483 (flat dose curve). The lead was small-sample noise.

**The crucial detail:** `ring_mobility_differential` carries genuinely
independent information (R²=0.25 when regressed on the 6 production features) and
is **still null** — so the cause is NOT redundancy, and (separately) features
decide 95% of evals so it's NOT tactical override. The bottleneck is the
heuristic's **linear/differential functional form**, not its feature set.

## Details

**Confirmed/falsified findings:**
- ❌ "New strategic features fix the evaluator" (as linear heuristic terms) — falsified at the agent level, well-powered.
- ✅ The dead-feature bug was real and is fixed (guarded by `test_feature_liveness`).
- 🟡 Features as *network inputs* (nonlinear) — untested (→ HF-2).
- 🟡 Re-fitting the 6 weights — untested (→ HF-1).

**Operational lessons logged:**
- Raw weights are not comparable across features: `ring_mobility` (raw ~±7.8) at
  weight 4 contributed ~31 to the eval, dominating the tuned 6 — a scaling
  artifact that first read as "hurts". Fixed with contribution-budget
  normalization (`--normalize`). Always normalize feature magnitudes in ablation.
- Small-N agent A/B is dangerously noisy: an 80-game 0.60 evaporated to 0.483 at
  300 games. Power up before believing a single-cell lead.
- Depth cost on this box: depth1 ~6s/game, depth2 ~112s, depth3 >400s. Powered
  multi-depth sweeps need the parallel `match_runner.py` (≈3.4× on 4 cores) and
  staged scoping. Staging let us *not* run a ~13h depth-2/3 fade test once depth-1
  came back null (no effect to fade).

Full result tables, per-arm cells, and methodology in
`docs/experiments/completed/phase1_feature_ablation_results.md`.

## Provenance & links

- Closes the heuristic-*feature* axis negative. Complemented by HF-1's
  heuristic-*weight* negative (`hf1_refit_weights.md`).
- Mechanism finding ("linear/differential form is the bottleneck") motivates HF-2
  (features as network inputs) and HF-3 (pivot to learned-value).
- Part of the "Heuristic evaluation-form investigation (2026-06-03) — forward queue".
- Next per sequencing matrix: HF-1 (cheap re-fit) → HF-2 (features-as-network-inputs, gated on in-flight value-head run).
