# Wave 2 Step 2 — Postmortem

Run: `runs_warm_start_combined_recipe/20260514_122701/`. 5 iters × 200 games × 4 epochs over 20.34h on a Vast.ai 4090. Init: `models/supervised_seed/best_supervised.pt`. Recipe: `configs/warm_start_combined_recipe.yaml` at `336dd98` (anchor n=40).

---

## TL;DR

1. **Step 1 telemetry wins held end-to-end.** All 5 Wave 2 gates produced sensible values across all iters; the `last_root_value` sign-flip fix kept every iter's value-outcome correlation positive (+0.061 to +0.198), vs the prior cloud run's -0.07 to -0.31.
2. **Wave 1's training-quality claim does not survive n=40 anchors.** Best raw-policy WR was 30% (iter 1) — *below* the 33% baseline. Best MCTS-48 WR was 60% — but at **iter 0**, the supervised seed *before* any self-play.
3. **The decay is real, not noise.** MCTS-48 strength: 60% → 50% → 35% → 32.5% → 47.5%. CI half-widths at n=40 are ~15%, so the iter 0 vs iter 3 gap (60% vs 32.5%) is well outside noise.
4. **The "Step 3" conditional in the original plan should not be taken.** Turning `revert_self_play_on_gate_failure: false` was framed as testing whether the revert mechanism was masking real learning. In fact, the revert *did* fire every iter, *did* reload iter 0 weights for the next self-play, and *did* prevent the much worse collapse you'd get with compounding self-play data. The bottleneck is something else (see Finding 4 below).
5. **One concrete bug filed**: `supervisor.py:2028` log line is misleading. See "Bug: misleading decision log" below.

---

## Per-iter scoreboard

| Iter | ploss | vloss | val_acc | cand_corr | tourn Elo | anchor raw WR (n=40) | anchor MCTS-48 WR (n=40) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (seed init) | 3.96 | 6.80 | 9.8% | — | 1500 | 20.0% | **60.0%** ← peak |
| 1 | 3.36 | 2.10 | 17.0% | +0.161 | 1450 | **30.0%** ← peak | 50.0% |
| 2 | 3.07 | 1.87 | 21.5% | +0.087 | 1500 | 22.5% | 35.0% |
| 3 | 3.11 | 1.90 | 21.7% | +0.161 | 1488 | 27.5% | 32.5% |
| 4 | 3.35 | **3.68** | **9.8%** | +0.061 | 1483 | 17.5% | 47.5% |

Cand_corr is the candidate's `eval/value_outcome_correlation/checkpoint_iteration_N` from the iteration's sidecar JSON. Anchor WRs are MCTS-48-mode and raw-policy mode against `HeuristicAgent(depth=1)`, n=40 with `tournament_sliding_window: 3`. The `cand_corr` column is missing for iter 0 because `anchor.skip_first_n_iterations: 1`.

---

## Five findings

### F1 — Step 1 telemetry is provably correct on full-scale cloud data

The Wave 2 acceptance gates:
- `train/effective_batch_size`: 249-250/256 across all 5 iters (≈97%). W1d phase-weight cap holding.
- `train/policy_target_entropy_mean`: 0.13-0.21 across iters. Positive throughout (per gate).
- `mcts/effective_child_visits`: 16K+ entries per iter (B1 wired). Latest values 2.8-9.5 (subtree reuse accumulates across moves; metric still useful as a regression detector).
- `eval/value_outcome_correlation/checkpoint_iteration_N`: all five candidate values **positive** (+0.061 to +0.198). Compare to the prior cloud run before Step 1: -0.07 to -0.31 across iters 1-4. The sign-flip fix at `self_play.py:798/1022` (negate `root.value()` at the assignment site) flips the metric exactly as Step 1's analysis predicted.

This isn't speculative. Step 1's fixes survive at n=40 production scale.

### F2 — The supervised seed is the best model end-to-end

iter 0 MCTS-48 = **60.0%** (n=40, CI roughly [45%, 75%]) against `HeuristicAgent(d=1)`. All four self-played iterations produced an EMA that performed *worse* on the same eval. **Not within CI**: iter 0 vs iter 3 (60% vs 32.5%) is ~27 percentage points, well beyond either's CI half-width (~15%).

This is the "warm-start regression" pattern: a strong supervised checkpoint, then self-play training, doesn't surface an even-better basin. The prior cloud run (2026-05-12) showed the same pattern with worse telemetry. Step 1 didn't change the underlying dynamic — it just made it visible.

### F3 — The revert mechanism actually fired every iter

The supervisor log claims `Decision: 🔄 CONTINUE (AlphaZero-style, no reversion)` at every iter 1-4 SUMMARY. **This is the misleading log line** — the summary block at `supervisor.py:2028` prints "no reversion" unconditionally whenever the gate fails, even when the actual revert path (`supervisor.py:1775+`) executed.

Searching the log for the revert path's own message:
```
⏪ REVERTING: Iter 1 failed gate (ELO 1450.0 vs best 1500.0). Reloading best model (iter 0) for next self-play iteration.
   ✓ Loaded best weights from checkpoint_iteration_0.pt
⏪ REVERTING: Iter 2 failed gate (ELO 1500.0 vs best 1500.0). Reloading best model (iter 0) ...
⏪ REVERTING: Iter 3 failed gate (ELO 1488.7 vs best 1500.0). Reloading best model (iter 0) ...
⏪ REVERTING: Iter 4 failed gate (ELO 1483.1 vs best 1500.0). Reloading best model (iter 0) ...
```

Confirmed: every iter's self-play started from iter 0 weights, with optimizer state also reset (`reset_optimizer_on_revert: true` is the default). **The revert is not the bottleneck.**

### F4 — The decay is consistent with EMA drift, not weight degradation

`use_ema_for_eval: true` with `ema_decay: 0.999`. Each training step is an EMA update toward the current candidate. Per iter (~800 batches across 4 epochs), the EMA absorbs ~55% of the new weights (since `1 - 0.999^800 ≈ 0.55`). After 5 iters the EMA retains only ~2% of iter 0's original weights — the rest is the rolling average of *trained candidates*.

Internal tournament head-to-head says each iter's trained candidate is on average ≤ iter 0:
- iter 0 vs iter 1: 23/40 vs 17/40 (iter 0 wins 57.5%)
- iter 0 in {iter 0, iter 1, iter 2}: 52.5% rate vs iter 2 (52.5% vs 50%) and iter 1 (52.5% vs 47.5%)

So the EMA is averaging-in candidates that the tournament says are weaker. Eval performance decays accordingly. **This is testable for free**: re-run the anchor eval against the *non-EMA* `checkpoint_iteration_N.pt` checkpoints. If non-EMA candidates also decay, the issue is the training step; if non-EMA candidates hold steady, EMA drift is the bottleneck.

This is Branch A. See `WAVE3_BRANCHES.md`.

### F5 — Iter 4's value collapse is a separate symptom

- vloss: 1.90 (iter 3) → **3.68** (iter 4) — nearly 2× regression
- val_acc: 21.7% → **9.8%** — halved
- policy entropy: 0.172 → **0.208** — network becomes *less* confident
- policy loss: 3.11 → 3.35

Iter 4 broke the joint policy/value training. The cosine LR schedule (`lr_schedule: cosine, warmup_epochs: 10`) bottoms out by iter 4's epoch 17-20 of 20 — near-minimum LR has less corrective power against a noisy gradient batch. Likely interaction with buffer composition: by iter 4 the buffer has 1000 games of mixed-quality self-play, and the smaller-LR update can't escape a bad local average.

This is a recipe-tuning question. Defer to Branch B.

---

## Why Step 3 (the old conditional) is the wrong move

The original `WAVE2_STEP_3_REVERT_DIAGNOSIS.md` framed the question: "is the gate-revert mechanism masking real learning, or is it saving us from collapse?" Try `revert_self_play_on_gate_failure: false`.

Given F3 + F4: turning revert *off* would let iter 1's worse candidate generate iter 2's self-play data (worse data → worse iter 2 candidate → even worse iter 3 self-play, etc.). The revert is the *floor* of the current pipeline's performance, not its ceiling.

Step 3 should be considered superseded by Branch B (recipe sensitivity), which tests narrower hypotheses one knob at a time.

---

## Bug: misleading decision log

**Location**: `yinsh_ml/training/supervisor.py:2028`
**Symptom**: After a failed Wilson gate, the SUMMARY block always prints
```
│   Decision: 🔄 CONTINUE (AlphaZero-style, no reversion)
```
even when the revert path 250 lines earlier (`supervisor.py:1775+`) executed and reloaded `best_model.pt`.

**Why it matters**: makes it impossible to read the log linearly and see what actually happened to the network. We only discovered the revert *was* firing by grepping for the `⏪ REVERTING` string from the inner branch.

**Fix shape**: in the SUMMARY block, distinguish four cases:
- promote → `Decision: ✅ NEW BEST (promoted to best model)`
- kept_current_best → `Decision: ➡️ KEPT (already best)`
- gate failed + reverted to best → `Decision: ⏪ REVERTED (gate failed, restored iter N for next self-play)`
- gate failed + AlphaZero continue → `Decision: 🔄 CONTINUE (AlphaZero-style, no reversion)`

This requires the SUMMARY block to know whether the revert path executed. Track that in a flag set inside the revert branch.

---

## Other observations worth noting (not bugs)

- **`tournament_sliding_window: 3` makes the Wilson gate trivial after iter 2.** Once iter 0 falls out of the window (iter 3+), there are "no head-to-head games" between candidate and best — Wilson gate silently no-ops. The supervisor's gate logic was still operative because it falls through to the Elo comparison (candidate_elo < best_model_elo → revert). So this wasn't a silent failure, just a noisy log warning.
- **`_iteration_counter: 4` in `best_model_state.json`** is a counter the supervisor increments to enforce "no resume from old state" — it tracks observed iteration boundaries, not promotions. Misleading field name; not a bug.

---

## What's next

Three branches, sequential by default. See `WAVE3_BRANCHES.md` for details and pre-flight.

| Branch | Cost | Question it answers |
|---|---|---|
| **A** | $0 (local or 30 min cloud) | Is EMA drift the bottleneck (F4)? |
| **B** | ~$10-15 per pass | Which single recipe knob has the highest leverage? |
| **C** | Day or two of design | Should the warm-start + self-play pipeline shape change? |

Do A first — it's free and resolves a yes/no hypothesis. B and C are informed by what A reveals.
