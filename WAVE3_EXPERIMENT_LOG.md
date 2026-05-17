# Wave 3 — Running Experiment Log

Append-as-we-go log of every experiment. One block per experiment: hypothesis → setup → headline → lesson → link. Read top-to-bottom for the trajectory of what we've tried, what worked, what didn't, and why.

**How to use this file**:
- New experiment kicked off → append a `Status: IN FLIGHT` block with hypothesis + setup.
- Experiment finishes → flip status to `DONE`, fill in headline + lesson, link the deeper doc if one exists.
- Don't delete entries even if hypotheses are refuted — the refutation IS the data.

---

## Wave 2 (done)

### Step 1 — Telemetry wiring + `last_root_value` sign-flip fix
- **Started → finished**: 2026-05-13 (one session)
- **Hypothesis**: Three Wave 2 acceptance gates (B1 `mcts/effective_child_visits`, B2 `train/effective_batch_size`, B3 `eval/value_outcome_correlation`) were reading None or the wrong list index across the prior cloud run. With the gates broken, no future cloud run can be interpreted.
- **Setup**: Local; small smoke `configs/wave2_wiring_verify.yaml`.
- **Headline**:
  - B1 wired by threading `metrics_logger` through `play_game_thread`/`play_game_worker`.
  - B2 wired by adding `log_scalar('train/effective_batch_size', …)` in `train_iteration`.
  - B3 wired by per-candidate metric naming (`eval/value_outcome_correlation/<candidate_label>`) so the gate-check reads the right series.
  - **`last_root_value` sign-flip fix at `self_play.py:798/1022`** — root.value() was stored in opposite-of-root POV by legacy backprop convention. `tournament.py` consumed it as if it were current-player POV → systematic negative correlation. Fix: negate at the assignment site.
- **Lesson**: Telemetry can lie. The previous cloud run had -0.07 to -0.31 value_outcome_correlations across iters — looked like "value head learning wrong sign." Was actually a logging bug. Always treat unit-test-passing-but-production-failing as a sign of a missing assertion.
- **Commit**: `3cb047f`. **Deeper doc**: `WAVE2_STEP_1_WIRING_FIXES.md` (plan), inline notes in commit message.

### Step 2 — n=40 anchor validation re-run
- **Started → finished**: 2026-05-14 12:27 → 2026-05-15 08:48 (20.34h cloud)
- **Hypothesis**: With Step 1's fixes live, n=40 anchor evals will tell us whether the Wave 1 training-quality claim ("MCTS at 48 sims > 33% baseline; raw policy near 33%") holds.
- **Setup**: `configs/warm_start_combined_recipe.yaml` at `336dd98` (only change vs prior run: `anchor.num_games: 4 → 40`). Vast.ai 4090.
- **Headline**:
  - All 5 Wave 2 gates produced sensible values. **`eval/value_outcome_correlation` positive across all 5 iters** (+0.061 to +0.198) — sign-flip fix held at n=40 production scale.
  - **Wave 1's training-quality claim does NOT hold.** Best raw policy 30% (below 33% baseline). Best MCTS-48 60% at **iter 0** (seed, before any training).
  - Per-iter MCTS-48 EMA: 60 → 50 → 35 → 32.5 → 47.5. Monotonic decay (with rebound).
- **Lesson**: The supervised seed already plays better with MCTS than 5 iters of self-play training produces. The training pipeline was actively making things worse. Step 3 (revert diagnosis) was superseded — we found the revert mechanism *did* fire every iter (the SUMMARY log said "no reversion" due to a separate logging bug, fixed in `ae63e8c`).
- **Deeper doc**: `WAVE2_STEP_2_POSTMORTEM.md`.

---

## Wave 3 — branches and diagnostics

### Branch A — eval non-EMA candidates against the same anchor
- **Started → finished**: 2026-05-15 (~1 hour after launch — cheap, just inference)
- **Hypothesis (F4)**: EMA drift was eroding seed quality across iters. The trained candidates were close to seed; the EMA was averaging-in worse-and-worse weights.
- **Setup**: Inference-only; pull each iter's `checkpoint_iteration_N.pt` (non-EMA) through `eval_vs_heuristic` at MCTS-48 n=40. Compare to the EMA WRs we already had.
- **Headline**: **F4 refuted.** EMA was *stronger* than non-EMA candidate in every iter (gaps +2.5 to +32.5 points). EMA was the saving grace; the raw trained candidate was the weak one. Pattern is consistent with high-frequency training noise that the EMA filters out.

  | iter | EMA WR | non-EMA WR | gap |
  |---|---:|---:|---:|
  | 0 | 60.0% | 32.5% | +27.5 |
  | 4 | 47.5% | 15.0% | +32.5 |

- **Lesson**: When formulating a hypothesis, test the direction explicitly. F4 had the sign wrong — easy to fall into if you don't run the cheap counterfactual first.

### D1 — Pin the seed baseline
- **Date**: 2026-05-15
- **Hypothesis**: We were triangulating the seed's MCTS-48 WR from §8 (n=60, 63.3%) and prior-run iter 0 EMA (n=40, 60.0%) but never measured the raw seed checkpoint at the same anchor pipeline.
- **Setup**: `eval_vs_heuristic --checkpoint models/supervised_seed/best_supervised.pt --num-games 40 --mcts-simulations 48`.
- **Headline**: **Seed @ MCTS-48 n=40 = 67.5% (27/40).** CI ~[52%, 81%]. Consistent with §8's 63.3% within CI.
- **Lesson**: Pin the baseline before extrapolating. Once D1 landed, the "training damages 67.5% → 32.5%" claim was sharp instead of triangulated.

### D2 — Inspect Step 2's per-epoch training loss timeline
- **Date**: 2026-05-15 (local, free)
- **Hypothesis**: The "high-frequency LR overshoot" story (from chat speculation) implied loss spikes within iter 0's 4 epochs.
- **Setup**: Parse `metrics/iteration_*.json` per-epoch records.
- **Headline**: LR schedule was MUCH lower than the recipe nominally said. `lr: 0.0001` was the *peak* of warmup; actual epoch-1 LR was 1e-5 (10× smaller). Loss DECREASED monotonically across iter 0's epochs (4.46 → 3.96 ploss, 8.39 → 6.80 vloss). val_acc DECREASED (12.2% → 9.8%). Iter 4 trained at LR ≈ 2.5e-6 (near zero) — explains why iter 4's val_acc reverted to iter-0 levels.
- **Lesson**: Read the actual LR schedule, not the recipe headline. Loss decreasing while win-rate decreasing means the loss objective is misaligned with the task — not an LR overshoot story.

### D3 — 1-iter × 1-epoch micro-run
- **Started → finished**: 2026-05-15 20:30 → 23:51 (3.33h cloud, ~$2)
- **Hypothesis**: Is damage immediate (first-epoch) or cumulative (across 4 epochs)? `configs/wave3_d3_micro_1epoch.yaml` runs 1 iter × 1 epoch and measures the candidate.
- **Setup**: 1 iter × 200 games × 1 epoch, warmup_epochs: 1 (so LR hits peak by end of the one epoch).
- **Headline**: **Iter 0 candidate after 1 epoch = 62.5% MCTS-48** (n=40). Within CI of seed's 67.5%. Step 2's 4-epoch iter 0 was 32.5%. **Damage compounds; 1 epoch is fine, 4 epochs ruins.**
- **Lesson**: Per-iter overfitting on the 200-game buffer was the dominant damage source. Branch B's load-bearing diagnostic.

### Branch B — epochs_per_iteration 4 → 1
- **Started → finished**: 2026-05-16 12:01 → 2026-05-17 06:48 (18.78h cloud, ~$10)
- **Hypothesis**: D3 generalizes — running the full 5-iter recipe at 1 epoch per iter avoids the per-iter overfitting. Mean candidate WR should rise toward seed.
- **Setup**: `configs/wave3_branchB_epochs1.yaml`. Single knob change vs Step 2: `trainer.epochs_per_iteration: 4 → 1` plus `trainer.warmup_epochs: 10 → 1` (scaled proportionally).
- **Headline**: **Mean MCTS-48 EMA = 59.5% (Branch B) vs 45.0% (Step 2), +14.5 points absolute.** Every iter improved. 3 promotions vs Step 2's 1.

  | iter | Step 2 EMA WR | Branch B EMA WR | Δ |
  |---|---:|---:|---:|
  | 0 | 60.0% | 60.0% | 0 |
  | 1 | 50.0% | 70.0% | +20 |
  | 2 | 35.0% | 45.0% | +10 |
  | 3 | 32.5% | 60.0% | +27.5 |
  | 4 | 47.5% | 62.5% | +15 |

- **Lesson**: F_overfitting confirmed at the population level. **But iter 1 (70% MCTS-48, the peak) was Wilson-rejected** (17/40 head-to-head, LB=0.285 < 0.55 threshold) → pipeline reverted to iter 0 → strength didn't compound. The gate threw away the run's best model.
- **Deeper doc**: `WAVE3_BRANCH_B_RESULTS.md`.

### Branch B' — promotion_threshold 0.55 → 0.20 — `IN FLIGHT`
- **Started**: 2026-05-17 12:23 UTC
- **Hypothesis**: Iter 1's 70% MCTS-48 was a real strength gain. If the gate had promoted it, iter 2 self-play would have used iter 1's weights, and the gain might have compounded. Lower threshold to LB > 0.20 (just above iter 1's LB=0.285) promotes the marginal-but-strong candidates.
- **Setup**: `configs/wave3_branchB_prime_low_wilson.yaml`. Single knob change vs Branch B: `arena.promotion_threshold: 0.55 → 0.20`.
- **Expected outcomes**:
  - **Propagation**: iters 2+ stay near 70% → Wilson gate was the bottleneck. Branch C should change the gate logic to use anchor WR directly.
  - **Regression**: iter 1 still drops to ~60% after another iter of training → iter 1's gain was a fluke. Different architectural lever needed (Branch C: distillation source, head decoupling, or from-scratch).
- **Headline**: pending. Cron `7d947323` checks every 2h at :29.

---

## Heuristics built from this trajectory

These are general lessons that should outlast the specific experiments:

- **L1** — Pin baselines before extrapolating. We spent a chunk of chat reasoning about a 30-point seed-to-candidate drop before D1 measured the actual seed at the right pipeline.
- **L2** — Run the cheap counterfactual before trusting the headline hypothesis. Branch A (a $0 inference job on existing checkpoints) refuted F4. If we'd run a full cloud experiment to test F4, we'd have burned $10 to learn the same thing.
- **L3** — Read the actual numbers, not the recipe headlines. D2 found the LR schedule was 10× smaller than the recipe said.
- **L4** — Internal-metric pass and external-metric fail is a real category of bug. Step 1 (sign-flip in `last_root_value`) and Branch B (Wilson gate vs anchor WR disagreement) are both this pattern.
- **L5** — Log lines lie when there are multiple decision branches above the SUMMARY block. `supervisor.py:2028` claimed "no reversion" even when the revert path ran. Always trace the actual code path, not the summary.
- **L6** — A wide CI at n=40 is not "no signal." If the *direction* is consistent across 5 paired comparisons, the population-level effect is real even when individual-iter CIs overlap.

---

## File-of-files

- `WAVE2_NEXT_STEPS.md` — index, kept current at major phase boundaries.
- `WAVE2_STEP_2_POSTMORTEM.md` — Step 2 deep dive (5 findings + bug + EMA-drift hypothesis).
- `WAVE3_BRANCHES.md` — original branch scoping doc.
- `WAVE3_BRANCH_B_RESULTS.md` — Branch B deep dive.
- **`WAVE3_EXPERIMENT_LOG.md`** ← this file. Always append, never reorder.
- `configs/wave3_*.yaml` — the actual recipe files for each experiment.
- `runs_*` — per-experiment artifact dirs (some on laptop, some on cloud).
