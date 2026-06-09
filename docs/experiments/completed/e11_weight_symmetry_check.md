# E11 — Direct weight symmetry check (residual-25% diagnostic, H_W)

**Status:** DONE: H_W confirmed (asymmetric weights are the primary cause of the residual 25%)
**Date(s):** proposed 2026-05-30; run 2026-05-30 morning
**Cost:** ~5 min, ~30 LOC; no cloud
**Branch / artifacts:** standalone diagnostic script (~30 LOC) over a single position then a few mid-game positions, using the validated per-state `augmenter`. Branch `policy-symmetry-fixes`.

## Description

The prime diagnostic for the **residual 25%** — the orbit-internal asymmetry remaining after symmetric MCTS ([[e8_symmetric_mcts]]): A5 still ~3× over orbit average (20 vs 4/3/0 for K7/E1/G11). Four candidate causes were enumerated:

- **H_W: network weights are asymmetric** — D2 augmentation during training didn't fully enforce weight symmetry. *(Highest prior — the augmenter is opt-in per-sample and can skip transforms; even if applied perfectly, gradient noise across batches might not converge to D2-symmetric weights.)*
- **H_M: MCTS noise** — at 96 sims with 85 valid moves, the 4 transformed searches each have stochastic visit counts that don't fully cancel under averaging.
- **H_N: sampling noise** — 50 games × temp 0.5 isn't enough to resolve the true orbit distribution; with larger n it may converge to uniform.
- **H_E: encoding pipeline lossiness** — the augmenter's encode→transform→decode roundtrip might introduce small artifacts. (Tested by E14.)

**E11 method** — for a single position s (empty board, then a few mid-game positions) and each transform tid:
1. Compute `s_tid = augmenter.transform_state(s, tid)`
2. Run network forward on `s_tid` → raw `policy_tid, value_tid`
3. Inverse-transform `policy_tid` back to the original action space
4. Compare the 4 inverse-transformed policies against the original policy bytewise (and against each other)

If all 4 versions are equal (modulo numerical precision): network weights produce D2-symmetric outputs, residual asymmetry is purely MCTS-side (eliminates H_W; focus on H_M/H_N). If they differ measurably: H_W is real, and the magnitude tells us how much. **This is the single most informative test for the residual 25% — run E11 first.**

## Outcome

**H_W confirmed.** Weight asymmetry is the primary cause of the residual 25%. The fix is E16 (symmetric weight regularizer). E12/E13 (sim/sample sweeps) deferred — they were diagnostics for *if H_W was false*. E15 (training augmentation coverage audit) could be useful supporting evidence but isn't blocking.

## Details

**E11 — weight symmetry check across 6 states (empty + 5 mid-game):** for each state, ran the network on all 4 D2-transformed versions, inverse-transformed the policy back to original action space, and compared against the identity prediction.

| State | Value-head: 4 transforms | Value symmetric? | Policy top-1 stable across transforms? |
|---|---|---|---|
| Empty board | -0.0086 ×4 | YES (exact) | No (top-1 differs across 3 transforms) |
| After move 2 | -0.016, -0.013, -0.016, -0.013 | 2-way pairing | No |
| After move 4 | -0.027, -0.015, -0.027, -0.014 | Near-paired | 2 of 3 differ |
| After move 6 | -0.045, -0.026, -0.037, -0.031 | All differ (1.7× range) | 2 of 3 differ |
| After move 8 | -0.034, -0.012, -0.025, -0.027 | All differ (2.8× range) | 3 of 3 differ |

**Two strong signals confirming H_W:**

1. **Value head is exactly symmetric on the empty board** (all 4 transforms produce -0.008561 to 6-digit precision) but **becomes increasingly asymmetric as the position fills up**. By move 8, value varies 2.8× across orientations for *the same position*.
2. **Policy top-1 changes between transforms** in 5 of 6 states. KL divergences are small in absolute terms (~0.003-0.006 nats) but enough to flip the argmax on a near-uniform distribution.

**Mechanism:** the empty board is symmetric because the input has no spatial information — all rings/markers channels are zero; only uniform-broadcast channels (phase, sentinel) are nonzero, so asymmetric weights have nothing to act on differently. Once there's real game state, asymmetric weights produce different outputs for different D2 orientations. The asymmetry magnitude grows with the amount of spatial information.

**Why training didn't symmetrize the weights:** D2 augmentation ensures the *training data* is symmetric but does not *constrain* the learned weights to be symmetric. Gradient noise across batches + non-symmetric initialization + per-orientation BatchNorm running stats can produce weights that nearly-but-not-quite respect D2 symmetry.

**Verdict on the residual 25%:** H_W (asymmetric weights) is the primary cause. E12/E13 (sim sweeps, larger n) are deferred — they were diagnostics for if H_W was false. E15 (training augmentation coverage audit) could be useful supporting evidence but isn't blocking. The fix is E16.

**Recommended sequence for the residual-25% investigation (as planned):**
1. E11 (5 min): weight symmetry check — quickest disambiguation between H_W and H_M/H_N. *Do this today.*
2. E14 (15 min): pipeline integrity — rules out a stupid bug before spending more compute.
3. Conditional on E11: if symmetric weights → E12 (6h MPS) sim sweep, then E13 (12h MPS) if needed; if asymmetric weights → E15 (30 min) to find why training didn't symmetrize, then E16 (30 LOC + retrain) to fix.
4. E17 (D2-equivariant network, retrain from scratch) only if E16 falls short.

**Deferred siblings (the residual-25% candidate suite):**
- **E12 — MCTS sim-budget sweep** (~6h MPS): diagnostic for H_M vs H_N; re-run symmetric MCTS at 200 and 400 sims (50 games each). Shrinks with budget → H_M; stays same → H_N. **Deferred** (H_W confirmed; would measure noise, not signal).
- **E13 — Increase game count** (~12h MPS): 200 games of symmetric MCTS; orbit should converge to symmetric if path-dependence fully broken. **Deferred.**
- **E15 — Training-time augmentation coverage audit** (30 min): pull saved `AugmentationStats` from a recent run; check `total_augmentations / num_train_samples` (~4 with reflections), `invalid_transforms_skipped` (~0), per-position augmentation count distribution. Optional supporting evidence, not blocking.
- **E17 — Explicit D2 weight tying via equivariant network** (D2-equivariant Conv2d kernels, Cohen & Welling style): guarantees weight symmetry by construction; requires retraining from scratch + architecture verification. Defer unless E16 proves insufficient.

## Provenance & links

- Source snapshot: 2026-05-30 morning ("E11 + E14 results — H_W confirmed, H_E ruled out").
- Residual identified by [[e8_symmetric_mcts]]; companion integrity check [[e14_augmenter_integrity]]; fix [[e16_symmetric_weight_reg]].
- Related deferred diagnostics E12, E13, E15, E17 (captured above).
