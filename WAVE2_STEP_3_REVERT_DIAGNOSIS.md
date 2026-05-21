# Step 3 — Revert behavior diagnosis (conditional)

**Estimated effort**: 30 min pre-flight + 15-25h cloud wall time + 30 min post-flight.
**Cloud cost**: ~$50-80 on Vast.ai 4090 (longer than Step 2 because of extended iter count).
**Pre-conditions**:
- Step 2 complete.
- Step 2 still showed post-iter-1 regression (i.e., iters 2-4 didn't climb above iter 1 with n=40 anchors).

**Skip this step entirely if Step 2 produced clean monotonic improvement across iters.** Step 3 is a diagnostic for a specific failure mode.

---

## The question

In the 2026-05-12 run, the supervisor reverted every iter after iter 1. This means iters 2-4 all generated self-play data from iter 1's checkpoint, retrained, and produced a candidate that lost Wilson and got reverted.

Two competing hypotheses explain that pattern:

### Hypothesis H1 — "The gate is masking real learning"

The model **was** improving on iters 2-4 (or oscillating productively around iter 1's level), but Wilson's 95% CI with only 20 tournament games has half-width ≈ 22%, so any single iter that didn't beat iter 1 by a wide margin got rejected and reverted. The revert behavior throws away weight updates that, in aggregate, would have produced a better model.

This is the **AlphaZero-style continuous-training argument** (see `supervisor.py:191-196`):

> Why no reversion: Reversion creates a closed loop where the model can't escape local optima. AlphaZero's insight: more diverse training data beats careful curation.

If H1 is right: turn off gate-reverts and watch the model improve over more iters.

### Hypothesis H2 — "The gate is saving us from collapse"

The model **is** genuinely degrading after iter 1. Maybe the warm-start prior is strong enough that on-policy self-play *can't* improve it, so each iter pollutes the buffer with mediocre games and the trainer overfits on the degraded distribution. Reverts hold the line, preventing actual collapse.

If H2 is right: turn off gate-reverts and watch the model crash. Don't run experiments like this in production.

**Step 3 is the experiment that tells us which hypothesis is correct.**

---

## What changes from Step 2

Single variable change:

```diff
 arena:
   games_per_match: 20
   promotion_threshold: 0.55
   tournament_sliding_window: 3
   eval_seed: 20260511
-  revert_self_play_on_gate_failure: true
+  revert_self_play_on_gate_failure: false
```

Plus an extension of `num_iterations`:

```diff
-num_iterations: 5
+num_iterations: 12
```

Why 12: gives 11 measured iters after iter 0, enough to see whether the model recovers from a mid-run dip, hits a noise floor, or continuously degrades. Caps wall time around 25h on the current sequential-workers config.

Everything else stays the same as Step 2 (anchor n=40, all Step 1 wiring fixes, same init, same LR).

---

## What's being tested

### Primary signal — Elo trajectory across 12 iters

```
What "H1 wins" looks like (gate was masking learning):
  iter 1 Elo: 1516 (matches Step 2)
  iter 2 Elo: 1480 (dip — would have been reverted under gate)
  iter 3 Elo: 1510 (recovering)
  iter 4 Elo: 1540 (above baseline!)
  iter 5 Elo: 1565
  ...iters 6-11 continuing upward, possibly with bumps

What "H2 wins" looks like (gate was holding the line):
  iter 1 Elo: 1516
  iter 2 Elo: 1480
  iter 3 Elo: 1450
  iter 4 Elo: 1410
  ...iters 5-11 continue dropping, anchor win rates collapsing toward 0%
```

### Secondary signals — anchor win rate trajectory

At n=40, anchor win rates per iter are interpretable. Plot them:

```
H1 trajectory: 50% → 35% (dip) → 45% → 55% → 60% → 65% (climbing)
H2 trajectory: 50% → 35% → 25% → 15% → 5% (monotonic decay)
```

### Tertiary — value-outcome correlation across iters

If Step 1 located and fixed the value-sign bug, this should be uniformly positive across all 12 iters under H1, and grow erratic or negative under H2.

---

## How to evaluate

After the run completes, ask three questions in order:

1. **Did the model crash hard?**
   - If anchor win rate at iter ≥ 6 drops below 10% raw, H2 wins decisively. Stop. The gate revert exists for a reason; turn it back on.

2. **Did the model improve beyond iter 1's level?**
   - If best-iter raw > 60% with CI lower bound > 50% (n=40), H1 wins. Gate reverts were masking real learning.
   - Implication: change the default `revert_self_play_on_gate_failure` for warm-start runs to `false`, document the reasoning, and possibly relax the Wilson threshold.

3. **Did the model oscillate without improving or crashing?**
   - If anchor win rates bounce 40-55% without clear trend, neither hypothesis wins cleanly. The model has hit a noise floor. This becomes a recipe-design question: more iters won't help, but adjusting LR/dirichlet/buffer might.

---

## Possible paths forward depending on outcome

### Outcome A — H1 wins (model continues to improve without reverts)

- Update `configs/warm_start_combined_recipe.yaml`: set `revert_self_play_on_gate_failure: false`.
- Add a comment explaining the empirical evidence (link to this doc / the 2026 dated run results).
- Consider longer baseline runs (20-30 iters) to find where improvement plateaus.
- Wave 3 candidate: switch to parallel workers to compress wall time at this longer iter count.

### Outcome B — H2 wins (model crashes without reverts)

- Keep `revert_self_play_on_gate_failure: true`.
- The question becomes: **why** does on-policy self-play from a warm-start prior degrade? Likely candidates:
  - Buffer contamination — iter 1's checkpoint is strong enough that exploration noise hurts more than helps.
  - Value head over-confidence — the warm-start value head is highly tuned to the supervised distribution; self-play games are off-distribution.
  - Dirichlet noise too aggressive — try `dirichlet_alpha: 0.5` or `epsilon_mix_start: 0.10`.
- Wave 3 candidate: invariant-preserving curricula — train initial iters with mixed supervised + self-play data instead of pure on-policy.

### Outcome C — Noise floor (no improvement, no crash)

- The recipe is at a local optimum that current self-play can't escape.
- Hardest case to diagnose. Possible directions:
  - Switch evaluation mode from `hybrid` to `pure_neural` (forces network to commit to its own value estimates).
  - Cycle LR — drop by 5× for one iter, then resume.
  - Try a deeper architecture or wider context — possibly the network capacity is the bottleneck.
- This is the case that motivates re-reading `TRAINING_REFACTOR_PLAN.md` from scratch.

---

## Risks and how to mitigate

### Risk 1 — Run goes off the rails and produces garbage models for hours

**Mitigation**: monitor closely for the first 3 iters. If anchor win rate at iter 3 is < 20% raw, kill the run early.

```bash
# Cron-style monitoring, similar to the 2026-05-12 cloud run setup:
# Every 30 min, check the latest metrics/iteration_<N>.json
# If iter ≥ 3 and raw_anchor_wr < 0.20, send a kill signal.
```

### Risk 2 — Wall time exceeds budget

**Mitigation**: if `num_iterations: 12` projects to > 30h based on the first iter's elapsed time, kill and re-launch with `num_iterations: 8`. Plenty of signal in 8 iters.

### Risk 3 — Best model becomes ambiguous

Without gate reverts, "best" is whichever iter has the highest Elo or anchor win rate — not whatever's at `best_model.pt`. Make sure:

- The supervisor's `best_model_state.json` is still updated (it should be, even with reverts off).
- Post-run, identify best iter explicitly via the metrics sweep across all iters.

---

## When NOT to run Step 3

- Step 2 produced clean monotonic improvement → no need to diagnose a regression that isn't there.
- Step 2 found a value-correlation bug that's still unfixed → fixing the bug is higher priority than diagnosing the revert behavior.
- Total budget is tight → Step 3 is a "nice to know," not a "need to know." Step 1 + Step 2 produce the actionable Wave 2 result.

---

## Documentation outputs

If Step 3 runs:

- Write a post-mortem at `memory/project_wave3_revert_diagnosis.md` summarizing which hypothesis won and the supporting trajectory data.
- Update `configs/warm_start_combined_recipe.yaml` with the empirically-determined `revert_self_play_on_gate_failure` value.
- Update `supervisor.py`'s docstring at line 198-205 to reflect the empirical evidence (currently it's a hypothesis-style explanation).

---

## What's deliberately NOT in scope for Step 3

- Multi-variable experiments. The whole point is isolating the revert variable.
- Re-running Step 2 — Step 3 builds on Step 2's data, not replaces it.
- Investigating the value-sign bug if Step 1 didn't find it. That's a Step 1 redo, not a Step 3.

Step 3 is a clean A/B on one variable. If the answer is ambiguous, the next question is "what should we measure next?", not "what should we tune?".
