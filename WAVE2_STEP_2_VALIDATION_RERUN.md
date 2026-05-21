# Step 2 — Validation re-run with working telemetry and statistically meaningful anchors

**Estimated effort**: 30 min pre-flight + 12-15h cloud wall time + 30 min post-flight.
**Cloud cost**: ~$30-50 on Vast.ai 4090 or equivalent.
**Pre-conditions**:
- Step 1 complete; all 3 wiring bugs fixed; all 5 Wave 2 gates producing non-`None` values on smoke runs.
- Local `metrics_logger.py:469` patch committed (task #11).

The point of this step: produce **one defensible data point** that answers "did Wave 1 actually improve training quality, or did we waste the cloud run?" The current run can't answer that because anchor n=4 has CI half-widths ≈ 30-40%, swamping any real signal.

---

## What changes from the previous run

This is the **same recipe** (`warm_start_combined_recipe.yaml`) with three changes:

### Change 1 — Bigger anchor sample (the load-bearing change)

```diff
 anchor:
   enabled: true
-  num_games: 4
+  num_games: 40
   depth: 1
   seed: 1337
   max_moves_per_game: 150
   skip_first_n_iterations: 1
```

**Why 40 specifically**:
- At n=4, Wilson 95% CI for a 50% observed win rate is `[0.15, 0.85]` — half-width ≈ 35%. Useless.
- At n=40, the same observation gives CI ≈ `[0.35, 0.65]` — half-width ≈ 15%. **Interpretable**.
- At n=100+, half-width ≈ 10%. Better, but adds 15× anchor compute per iter (~30 min/iter at 48-sim eval pace), which would push total run time over 24h.
- 40 is the sweet spot: meaningful CIs without doubling wall time.

**Note on cost**: anchor runs at 64 sims (per `tournament.py` default), so 40 games × ~30s/game = 20 min/iter. Across 4 iters with anchor (iters 1-4): ~80 min added. Within budget.

### Change 2 — Step 1 fixes applied

All Wave 1 fixes plus:
- `train/effective_batch_size` actually logs to sidecar (B2 fix).
- `mcts/effective_child_visits` actually logs (B1 fix).
- `eval/value_outcome_correlation` either uses per-candidate metric names (B3 option b) or the gate-check script reads `entries[0]` (B3 option a).

### Change 3 — Init checkpoint

Decide what to warm-start from:

| Option | Description | Pro | Con |
|---|---|---|---|
| **A: `models/supervised_seed/best_supervised.pt`** | What we used last run. 130MB, same architecture, on laptop. | Apples-to-apples with the run we just did. | Different supervised data than the original `supervised_deep_256x18` 33% baseline. Not strictly apples-to-apples with the runbook's stated baseline. |
| **B: `runs_warm_start_combined_recipe_cloud/20260512_151604/iteration_1/checkpoint_iteration_1_ema.pt`** | Iter 1 from the run we just finished (the best stable model). | Starts from a more trained position; should accelerate convergence; tests whether continued training from iter 1 improves or degrades. | Mixes the question — you can't separate "did W1 help with warm start?" from "did continued training help?" |
| **C: Rebuild `supervised_deep_256x18` on cloud** | Per runbook §3 Option B. Takes 1-2h before main run. | Properly apples-to-apples vs the 33% baseline referenced in the original runbook. | Adds 1-2h cost; requires the original supervised training data on cloud. |

**Recommendation: Option A.** It matches what we just did, so the only variable changing is "anchor n=4 → n=40 + working gates." If you change init too, you can't attribute results cleanly. Save Option C for a Wave 3 baseline-validation pass if needed.

### What NOT to change (intentionally)

Resist the urge to also tune LR, value_lr_factor, epochs_per_iteration, dirichlet_alpha, etc. The whole point of Step 2 is to **isolate one variable** (working telemetry + meaningful anchor) and rerun. If you change five things and the result is different, you don't know why.

---

## What's being tested

Two questions. Both must be answered for Step 2 to be a useful spend:

### Q1: Are the Wave 1 fixes actually lifting training quality?

**Operational test**:
- Best-iter raw-policy win rate, n=40, vs depth-1 heuristic.
- **Baseline to beat**: the historical 33% from `warm_start_deep_lowlr` (the original Wave 2 baseline before Wave 1).
- **What "yes" looks like**: best-iter raw > 50% with CI lower bound > 40%.
- **What "no" looks like**: best-iter raw clusters around 33% with CI overlapping the baseline.
- **What "mixed" looks like**: one iter spikes high but others are flat — n=40 makes the spike either real (replicable) or random (won't repeat).

### Q2: Is the negative value-outcome correlation persistent across iters?

**Operational test**:
- With the corrected gate-check, look at `eval/value_outcome_correlation` for the candidate of each iter 1-4.
- **What "fixed" looks like** (if Step 1 located and fixed the value-sign bug): correlation positive and rising toward 1.0 across iters.
- **What "unchanged" looks like** (if Step 1 found no value-sign bug): correlations stay negative across iters — Wave 1 didn't address the real issue and Wave 3 needs to.
- **What "noisy" looks like**: correlations bounce around 0 ±0.2. Probably means the metric needs more data or the test was statistically weak.

---

## Pre-flight checklist

Before launching the cloud run:

- [ ] Step 1 wiring fixes committed on `training-pipeline-fixes`.
- [ ] Smoke run locally produces a sidecar JSON with all 5 gates non-`None`.
- [ ] `pytest yinsh_ml/tests/test_*.py` (all 10 Wave 1 tests + new wiring tests) green.
- [ ] Recipe diff applied: `configs/warm_start_combined_recipe.yaml` has `anchor.num_games: 40`.
- [ ] `models/supervised_seed/best_supervised.pt` exists locally (still on laptop from the last run).
- [ ] `analysis_output/heuristic_evaluator_model.pkl` exists locally (still on laptop).
- [ ] Vast.ai instance spun up — note the new IP/port and update `~/.ssh/config` `cloud` stanza (current pre-teardown IP was `23.158.136.85:26654`).
- [ ] `git push origin training-pipeline-fixes`.

Then follow `CLOUD_RERUN_RUNBOOK.md` §2 onwards. The runbook is still accurate apart from the gate-check script in §7 — apply the same B3 fix to the gate-check Python snippet there.

---

## What to evaluate when the run finishes

### Headline numbers (per-iter)

```
For each iter 1-4:
  raw_anchor_wr     (gate ≥ 50%, with CI lower bound > 40% = "real")
  mcts_anchor_wr    (gate same)
  value_correlation (gate > 0, rising toward 1.0)
  policy_loss, value_loss
  tournament_elo
  promotion/revert decision
```

### Section §8 final eval suite (60 games × 3 modes on best EMA)

Same as we ran this time. Probably automate it as the last step of the run script:

```bash
python scripts/eval_vs_heuristic.py --checkpoint <best_ema> \
    --num-games 60 --depth 1 --mcts-simulations 400 --label best_400
python scripts/eval_vs_heuristic.py --checkpoint <best_ema> \
    --num-games 60 --depth 1 --mcts-simulations 48 --label best_48
python scripts/eval_vs_heuristic.py --checkpoint <best_ema> \
    --num-games 60 --depth 1 --no-mcts --label best_raw
```

**Wave 2 success criteria from the runbook §8**:
- 400-sim: > 30% stochastic (baseline did this; should also do this).
- 48-sim: > 33% stochastic (the load-bearing claim — fix delta should show here).
- Raw: > baseline raw column (unknown baseline; iter 1 EMA from the 2026-05-12 run is the current data point).

---

## Predicted outcomes and what each means

| Outcome | What it means | Next step |
|---|---|---|
| All gates green, raw > 50% with CI lower bound > 40% | Wave 1 fixes work. Recipe is the new floor. | Move to Wave 3 (longer runs, parallel workers, alternative architectures). |
| Gates 1-3 green, value-correlation still negative | Telemetry's good but the value-sign bug Step 1 didn't catch is still there. | Return to Step 1 with sharper test; possibly fork to Wave 1.5 dedicated to value-head fix. |
| All gates green but raw lands at ~33% with CI overlapping baseline | Wave 1 fixes don't help meaningfully. The 2026-05-12 iter 2 75% spike was noise. | Hard question: is the recipe wrong, or are the fixes wrong? Probably triggers a deeper architectural review (TRAINING_REFACTOR_PLAN.md). |
| Some iters spike high, others crash | High variance — recipe is on the edge of stability. | Step 3 (revert diagnosis) becomes interesting; also consider lower LR or more conservative dirichlet. |
| Run crashes / OOMs / silently fails | Infra regression introduced by Step 1 fixes. | Revert the Step 1 changes that broke production, re-test in isolation. |

---

## Cost and time framing

- Cloud GPU: 4090 on Vast.ai ≈ $0.30-0.50/hr. 15h run ≈ $5-8. Negligible.
- Sequential mode (workers=0) is the bottleneck; if we want to halve wall time, set `num_workers: 4-8` in the recipe — but that's another variable change. Save for Wave 3.
- Total budget: **$30-50 covers including possibility of a re-launch if first attempt has a setup glitch**. Per the user "I've got some money to spend, don't worry."

---

## What's deliberately NOT in scope for Step 2

- Architecture changes (different ResNet depth, different value/policy head ratio).
- LR schedule changes.
- Adding new self-play policies (e.g., more aggressive exploration).
- Trying a different evaluation mode (`hybrid` → `pure_neural` etc.).
- Parallel workers.

These are all real questions, but each one needs to be isolated. Step 2 isolates "telemetry working + meaningful anchors." If that gives us a clean answer, the next variable choice is informed. If we batch changes, we learn nothing.
