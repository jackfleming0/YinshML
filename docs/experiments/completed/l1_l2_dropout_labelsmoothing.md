# L1 / L2 — Dropout(0.3→0) + label smoothing (policy-head plateau fix)

**Status:** DONE: both validated 2026-05-29 (L1 unblocks policy sharpening; L2 prevents over-confidence collapse)
**Date(s):** diagnosed + validated 2026-05-29; recovered onto `policy-symmetry-fixes` 2026-05-31 (L1 commit f8bdbcb)
**Cost:** local continued-pretrain (no cloud)
**Branch / artifacts:** L1 in `yinsh_ml/network/model.py` (policy head, `model.py:140`), commit f8bdbcb; L2 in `scripts/run_supervised_pretraining.py` (`--label-smoothing`, default 0.1, applied to the hard-target CE), `evaluate()` now takes the param; validated dry-run `dry_run_dropout_plus_ls.py`; branch `policy-dropout-fix` (original, from main), recovered onto `policy-symmetry-fixes`. Outputs: `models/supervised_2026-05-27/dropout_patch.pt`, `models/supervised_2026-05-27/dropout_plus_ls01.pt`. Production-recipe slots L1 and L2.

## Description

Two-part architecture+loss fix for the **post-iter1 plateau**, located via the P1/P2/P3 plateau diagnostics. Root cause: **Dropout(0.3) on the policy head** caps how sharp the policy can ever get; removing it (L1) unblocks sharpening but exposes an **over-confidence collapse** under one-hot CE, which **label smoothing** (L2) remedies.

**L1 — Dropout(0.3) → Dropout(0.0)** in the policy head:
```python
self.policy_head = nn.Sequential(
    nn.Conv2d(num_channels, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 11 * 11, 1024), nn.ReLU(),
    nn.Dropout(0.3),                       # ← culprit
    nn.Linear(1024, self.total_moves)
)
```
Dropout(0.3) on the 1024-feature bottleneck before a 1024→7,433 layer forces ensemble-averaging over dropout subnetworks during training. For 7,433-way classification with sharp MCTS targets, the minimum-loss solution becomes near-uniform output. Even at eval time (dropout off), the trained weights produce flat output because they were calibrated to the averaged-subnetwork minimum-loss point during training. AlphaZero-style policy heads typically don't use dropout; regularization can come from weight decay (already 1e-4 on the policy optimizer) or label smoothing if needed.

**L2 — Cross-entropy with label smoothing ε ≈ 0.05-0.1** (default 0.1) on the hard-target supervised CE: `F.cross_entropy(..., label_smoothing=0.1)`. Puts a floor on the loss so the now-flexible policy head can't drive loss arbitrarily close to 0 by over-concentrating.

## Outcome

**Both validated 2026-05-29.**
- **L1:** policy head sharpens 22.4× from uniform in 3 epochs, **past E6's epoch-5 ceiling in just epoch 2**. Decisively confirms Dropout(0.3) was the architectural cap.
- **L2:** prevents the over-confidence collapse that L1 alone causes. Without it, the patched model collapsed to F6 100% modal and white WR dropped to **6%**. Smoothing demonstrably slows premature over-concentration (epoch-1 F6 12.9% with smoothing vs 21.7% without; entropy 4.69 vs 3.70).

These are L1 and L2 of the production recipe; L3a (symmetric MCTS at inference) is [[e8_symmetric_mcts]] and L3d (symmetric weight regularizer) is [[e16_symmetric_weight_reg]].

## Details

### P1/P2/P3 plateau diagnostics (2026-05-29 afternoon)

Three diagnostics, each ~1-4h, no cloud, that gated the E7 commitment and located the root cause:

- **P1 — MCTS amplification quality:** ✓ **works.** 10 main-game positions, mean KL(visits_800 || visits_96) = **3.46 nats**, top-1 disagrees **50%** of the time between 96-sim and 800-sim search. Deep search finds different, sharper distributions than shallow search.
- **P2 — Value-head calibration:** ✓ **calibrated.** 5K H-vs-H positions, linear-fit slope **0.98** (1.0 = perfect), Brier score **0.66 vs baseline 0.78 (15% improvement)**. Value predictions track actual outcomes monotonically across the [-1,+1] range.
- **P3 — Self-play policy-target signal:** ✗ **broken — policy head can't reproduce MCTS targets.** KL(MCTS_visits || policy_pred) **median 9.0 nats** (max possible ~9 for 7,433-way classification), top-1 match rate **0%**, mean policy peak **0.000329** (uniform = 0.000134, so model is ~2.5× uniform).

Root cause located in `yinsh_ml/network/model.py` policy head: the **Dropout(0.3)** on the 1024-feature bottleneck. Consistent with all measured policy outputs across all training lineages (supervised, iter1, iter5) — they all converge to ~2-5× uniform peak, regardless of training data quality. **The fix is one line of code** (`model.py:140`).

### L1 validation — patch works at the policy-head level

Branch `policy-dropout-fix` (created from main, single change + diagnostic comment). Ran the same continued-pretrain recipe as the E6 dry-run (107K H-vs-H positions, 3 epochs, LR 5e-5, weight decay 1e-3, 4× placement oversample). Output: `models/supervised_2026-05-27/dropout_patch.pt`.

**Policy head learns dramatically faster:**

| Metric | Before training | E6 (5 epochs) | Dropout-patched (3 epochs) |
|---|---|---|---|
| Empty-board peak | 1.39% | 27.2% | 31.3% |
| Empty-board entropy | 4.44 | ~3.5 | 3.39 |
| Multiplier vs uniform | 1.04× | 20.3× | 22.4× |

The policy head sharpened past E6's epoch-5 ceiling in just epoch 2 (F6 28.9% at epoch 2 of the patched run vs F6 27.2% at epoch 5 of E6). Decisively confirms: Dropout(0.3) was the architectural cap on policy sharpness.

Test criteria used: empty-board policy peak > 0.05; empty-board policy entropy should DROP (sharper) vs E6; H2H vs iter1_ema same Wilson-LB ≥ 0.50 non-regression bar as E6.

### Over-confidence collapse — the new failure mode L1 alone exposes

50-game deployed_sampled self-play on the L1-patched model:
- Slot-1 modal: **F6 100%** (every single game)
- Aggregate cluster: F6/G6/G7/F7/G8 (mean_pw 1.26)
- Tight-cluster rate: **100%** (all 50 games)
- **White WR: 6% (3 of 50)** — catastrophic regression

Cause: cross-entropy on one-hot human-move targets with **no label smoothing** lets the now-flexible policy head drive loss arbitrarily close to 0 by over-concentrating. AlphaZero-style training uses MCTS visit distributions (naturally entropic) as policy targets, which the H-vs-H corpus doesn't provide. **Label smoothing is the cheap approximation.**

### L2 validation — label smoothing fixes the collapse

Same setup but `F.cross_entropy(..., label_smoothing=0.1)`. Output: `models/supervised_2026-05-27/dropout_plus_ls01.pt`. Epoch 1:
- F6 = **12.9%** (vs dropout-only 21.7% at same epoch)
- Entropy = **4.69** (vs dropout-only 3.70)

Smoothing working as designed — preventing premature over-concentration. (Production recipe: ε ≈ 0.05-0.1, OR train on MCTS visit distributions where available.)

### L2 placement note (recovery, 2026-05-31)

L2 was wired into **`scripts/run_supervised_pretraining.py`**, not `trainer.py`. The handoff said "trainer.py", but trainer.py's policy loss is a *soft*-target CE (MCTS visit distributions) where classic label smoothing doesn't apply. The validated dry-run (`dry_run_dropout_plus_ls.py`) used the *supervised* hard-target CE — that's where L2 belongs. Fixed a latent crash from the first L2 edit: a `replace_all` had put `args.label_smoothing` into `evaluate()`, which has no `args` — the real run would have NameError'd at first validation. Caught by a full end-to-end script smoke; `evaluate()` now takes the param.

### Reframing (Jack, 2026-05-29) — don't treat humans as ground truth

YINSH lacks anything like chess opening theory. BGA "top-10" is a few hundred enthusiasts, not centuries of accumulated wisdom. F6 might be the right opening, or it might just be cargo-culted as "the obvious first move." Treating the H-vs-H placement distribution as the *target* the model must converge to is an unsupported assumption.

**Decisive evidence the model's current convergence is path-dependence, not strategy — the D2 symmetry argument:** YINSH has D2 board symmetry (Klein 4-group: identity + 180° rotation + 2 reflections). iter1_ema's slot-1 distribution is A5 72.0% / A2 2.5%. A2 is the horizontal-reflection partner of A5. If A5 were genuinely optimal, the model should play A2 (and partners K7, K10) at ~equal rates (~18% each). The observed 72%/2.5% asymmetry is *physically impossible* for a real strategic principle — it can only arise via the FPU + uniform-policy MCTS-stall mechanism from P3. The symmetry breakage is the smoking gun for path-dependence. (This drove [[e8_symmetric_mcts]].)

**Implication — new framing:**

| Old framing | New framing |
|---|---|
| Fix placement so it matches human play | Fix architecture so the model can express what it learns, then let exploration discover |
| F6 modal is the target | Symmetric exploration is the target; whatever wins under self-play is the answer |
| Use H-vs-H data to teach placement | Use H-vs-H to initialize away from uniform; use random + symmetric variants for exploration; let self-play decide |

### Production recipe (the three layers)

1. **Architecture (L1):** `Dropout(0)` in policy head. Validated: sharpens 22.4× from uniform in 3 epochs. Without it: policy head architecturally cannot represent sharp distributions regardless of data quality.
2. **Loss (L2):** CE with label smoothing ε ≈ 0.05-0.1, OR MCTS visit distributions where available. Validated negatively: without smoothing, F6 collapses to 100% and white WR drops to 6%. Smoothing floors the loss.
3. **Data + exploration:** split corpus (H-vs-H placement initialization + iter1 main game) AND phase-aware exploration knobs ([[e9_phase_aware_exploration]]) AND symmetric MCTS at inference ([[e8_symmetric_mcts]]) AND random+symmetric placement injection ([[e10_placement_injection]]).

**Remaining (Task 3):** next cloud run stacks L1+L2+E16 — supervised pretrain `--use-enhanced-encoding --label-smoothing 0.1 --enable-symmetric-reg` on a Dropout=0 net, then self-play with `enable_symmetric_reg: true`. Config still needs writing (mirror `branchD2_enhanced` + symmetric_reg keys).

## Provenance & links

- Source snapshots: 2026-05-29 ~13:30 UTC (plateau diagnostics, dropout root cause), 2026-05-29 validation results, 2026-05-29 reframing + production recipe, 2026-05-31 ~14:00 UTC recovery snapshot.
- Diagnosed via P1/P2/P3 against [[e6_hvh_continued_pretrain]] (whose negative result was partially confounded by the dropout cap).
- L3a = [[e8_symmetric_mcts]] (symmetric MCTS at inference); L3d = [[e16_symmetric_weight_reg]].
- Cross-doc: `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`, `YNGINE_BENCHMARK_RESULTS.md`.
