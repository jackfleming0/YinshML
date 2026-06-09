# A4 — Regression value head pretrain

**Status:** QUEUED
**Cost:** Code changes ~2-3h (single dev-day); Pretrain ~3h, ~$5; Self-play ~6h, ~$10; SPRT 30 min - 4h, ~$1-5. **Total: ~12h on 5090, ~$20**, plus dev time.
**Stack-rank:** Likely+ 4 / Unblocks 4 / Info-gain 5 / Cost 4 / Impl-risk 4 / Sum 21
**Dependencies / blocks:** Blocks: nothing critical; A4 is a parallel-track experiment. Unblocks: if positive, it changes the recommended architecture for ALL future Branch D experiments. If negative, narrows the value-head plateau hypothesis to "head capacity" or "data ceiling" rather than "target discretization."

## Description
**Goal:** replace the 3-class cross-entropy value head with a scalar regression head (`tanh` output, MSE loss against `value ∈ [-1, 1]`), then re-run the D.2 pipeline. Tests whether the value-head plateau is caused by the discretization, not the architecture.

**Mechanism:** the current value head predicts {-1, 0, +1} discretized into 3 classes. YINSH outcomes have a real "decisiveness" gradient (2 captures ahead ≠ tied with momentum ≠ 1 capture behind) that the 3-class collapse throws away. A scalar head can encode that gradient. If the value-head plateau is caused by representation loss in the target structure, this fixes it directly.

## Outcome
Pending — SPRT verdict in `logs/d2_regr_iter4_vs_frozen.json`, regression-head supervised pretrain checkpoint saved, code changes committed (gated behind a flag so the classification path stays default). Positive → changes recommended architecture for all future Branch D experiments. Negative → narrows the plateau hypothesis to "head capacity" or "data ceiling" rather than "target discretization."

## Details

**Supporting evidence:**
- Val P-loss decreased monotonically through 6 epochs of D.2 pretrain (3.16 → 2.84) while VAcc was nearly flat (0.628 → 0.636). The argmax stayed similar but the *distribution* kept tightening — exactly the symptom of a model that's still learning but on a metric the loss-discretization can't resolve.
- The original AlphaZero paper used a scalar `tanh` value head; the 3-class variant here is an inherited convention, not a justified design choice.
- The codebase already supports a `value_mode='regression'` path in `NetworkWrapper.__init__` (see `wrapper.py` line 28). Not exercised in current configs but the plumbing exists.

**Reasons to not believe:**
- **Value-head plateau could be the data ceiling (theory III).** If yngine outcomes are noise-limited (many positions could go either way), no loss function gets you above the Bayes error from raw outcomes. Regression doesn't help in that case.
- **MSE on raw outcomes can be *worse* than CE on classes.** With only {-1, 0, +1} as raw targets, MSE pushes the network toward 0 (the mean), which is uninformative. We may need to use MCTS-rolled-out value estimates as targets, not raw outcomes — and those aren't in the yngine corpus. This is closely related to D1.
- **Self-play loss surface differs:** the self-play trainer uses CE on a discretized MCTS rollout value. If we pretrain with regression but self-play with classification, the warm-start washes out on iter 1. Probably need to *also* switch self-play to regression to be consistent. Doubles the code change.

**Methodology:**

Phase 1 — Add regression head support to `run_supervised_pretraining.py` **— prepped 2026-05-25 during B1B2B3 run:**
- ✅ `--value-mode {classification, regression}` CLI flag added (default classification; back-compat preserved).
- ✅ `--num-value-classes` override added (classification mode only).
- ✅ Training + eval loops branch on `model.value_mode`: regression uses `F.mse_loss(value_pred, values)` against the scalar tanh output; classification keeps the CE-on-discretized-classes path.
- ✅ Value-accuracy metric branches: regression logs *sign accuracy* (proxy — does the model predict the right side of zero?); classification keeps argmax-class accuracy.
- ✅ Smoke-tested both paths construct and forward correctly (regression returns `(B,)` value tensor; MSE loss computes cleanly).
- Save under a new path when actually run: `models/yngine_volume_15ch_pretrain_regr/`.

Phase 1.5 — Self-play side wiring (DEFERRED until we actually run A4):
- `scripts/run_training.py:452` constructs `NetworkWrapper` without passing `value_mode`. A regression-trained checkpoint loaded here will hard-fail at `load_model` because the value head's final-layer output dim is 1 (regression) vs `num_value_classes` (classification). The hard-fail is *desired* — louder than silent misload.
- When A4 actually runs, fix by either (a) adding `value_mode` as a config knob in the training YAML, or (b) auto-detecting from the checkpoint's last value-head layer output dim (parallel to existing encoding / capacity auto-detect in `wrapper.py:160`). Prefer (b).

Phase 2 — Decide on self-play matching:
- Option A: also switch self-play to regression. Requires changes in `yinsh_ml/training/trainer.py`'s value loss path. Bigger lift.
- Option B: pretrain regression, but cast back to classification at warm-start time (probably won't work cleanly — the network's last layer shape differs).
- Recommended: bite the bullet on Option A, since the whole point of A4 is to test the value-head structure.

Phase 3 — Run the same D.2 self-play loop with the regression-head pretrain init.

Phase 4 — SPRT vs `best_iter_4` (the 3-class champion).

**Open questions:**
- How do we handle the self-play replay buffer's pre-existing classification-target structure during transition? Probably need to invalidate (delete) any prior buffer when switching modes.
- Does the EMA path need adjusting? Same network, same parameters — should be fine, but worth a smoke test.

## Provenance & links
- Related: [[d1]] (MCTS-rolled-out value targets — the "better targets" alternative), [[a2]] (value-head bottleneck reading that de-risks A4), [[a1]] (interpretation gate).
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section. Code prepped during the B1+B2+B3 run (2026-05-25).
