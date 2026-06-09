# E16 — Symmetric-weight regularizer for training loss

**Status:** DONE: prototype built + wired into both training halves (default OFF)
**Date(s):** prototype designed 2026-05-30; implemented/wired 2026-05-31 (commit e1c6d55)
**Cost:** 4× forward per regularized step; at K=10 → +0.3× total training time (manageable on cloud)
**Branch / artifacts:** shared module `yinsh_ml/training/symmetric_reg.py`; wired into `trainer.py` (self-play; `enable_symmetric_reg` via supervisor `mode_settings`) AND `run_supervised_pretraining.py` (`--enable-symmetric-reg`); prototype `scripts/prototype_e16_symmetric_loss.py`; investigations `scripts/investigate_e16_value_weight.py`, `scripts/investigate_e16_dynamic.py`; tests `yinsh_ml/tests/test_symmetric_regularizer.py`; commit e1c6d55; branch `policy-symmetry-fixes`. Production-recipe slot L3d.

## Description

The **fix for H_W** (asymmetric network weights, confirmed by [[e11_weight_symmetry_check]]). Add a loss term that explicitly penalizes asymmetric output. For each training batch, compute output on D2-transformed versions, inverse-transform back, penalize KL divergence between the original and the average of the 4 inverse-transformed versions. Acts as continuous pressure to keep weights in the D2-symmetric subspace.

**Loss formulation (prototype):**
```
For each state s in batch:
    Forward on s → policy_0, value_0
    For tid in [1, 2, 3]:
        s_tid = T_tid(s)
        Forward → policy_tid_raw, value_tid_raw
        policy_tid = T_tid(policy_tid_raw)   # inverse via involution
    policy_sym = mean(policy_0, policy_1, policy_2, policy_3)
    value_sym = mean(value_0, value_1, value_2, value_3)
    loss_sym = KL(policy_0 || policy_sym) + α * mean((v_i - value_sym)^2)
```

## Outcome

**Prototype built, validated (loss math + penalty magnitude), and wired into both training halves; default OFF.** Two correctness fixes vs the original prototype and a data-driven default for `value_weight`.

The prototype validated the loss math and measured penalty magnitude on iter1_ema:

| State | KL policy asym | Value asym MSE | Value range | Total loss |
|---|---|---|---|---|
| Empty | 0.0020 | 0.000000 | (constant) | 0.0020 |
| After move 2 | 0.0062 | 0.000000 | 0.003 | 0.0062 |
| After move 4 | 0.0116 | 0.00004 | 0.013 | 0.0116 |
| After move 6 | 0.0173 | 0.00005 | 0.019 | 0.0173 |
| After move 8 | 0.0231 | 0.00006 | 0.022 | 0.0231 |

Penalty grows ~12× from empty to move 8, mirroring E11's finding that asymmetry scales with game complexity. The regularizer is nonzero and well-behaved — meaningful gradient signal to push weights toward the symmetric subspace.

**Expected effect:** after a few thousand training steps with the regularizer active, the network's per-position output asymmetry (re-running E11) should drop below the noise floor. Symmetric MCTS at inference should then produce orbit-symmetric visit distributions — fixing the residual 25%.

## Details

**As-built (2026-05-31, commit e1c6d55) differs from the 05-30 prototype in two important ways:**
1. **Policy KL is masked to the valid-move support.** The unmasked full-softmax KL was **~100× inflated** by never-trained invalid-move logits. Self-play masks the policy KL to MCTS visit support (`target_probs>0`); supervised has hard targets so it decodes the valid-move mask from the batch states (regularized steps only).
2. **`value_weight` set to 20** (was the prototype's guessed 0.5), from two investigations below.

**Wired into BOTH training halves via shared module `yinsh_ml/training/symmetric_reg.py`:**
- `trainer.py` (self-play): `enable_symmetric_reg` via supervisor `mode_settings`.
- `run_supervised_pretraining.py`: `--enable-symmetric-reg`.

Rationale for hitting supervised pretrain too: the supervised pretrain is where **most of the representation + its D2 asymmetry is learned** — the dynamic probe showed value_asym growing 6.7× under supervised task pressure — so enforcing weight symmetry there stops self-play from inheriting an already-asymmetric net.

**Full faithful policy-KL + value-asymmetry form.** The full 7433-move-index permutation is precomputed once and **verified to exactly match the validated per-state augmenter permutation**. Defaults: **α=0.1, K=10, value_weight=20**; off by default.

**Why value_weight=20 (two investigations):**
- **Static gradient pressure** (`scripts/investigate_e16_value_weight.py`): value_asym penalizes a scalar value in ~[-1,1] while policy-KL lives on a simplex, so at 0.5 the value term exerted only ~1/20th the gradient pressure of the policy term. ~10 equalizes ‖grad‖ on the trunk.
- **Dynamic probe** (`scripts/investigate_e16_dynamic.py`, 150 steps × {0.5, 10, 20, 50} on MPS): under task pressure, value_asym grew 6.7× at w=0.5, ~2.7-3.2× at w=10/20, and *shrank* (−17%) only at w=50; monotonic in w, with negligible (and largely illusory — it's memorization acc) policy cost. So equal pressure (10) only *slows* value drift; 20 holds it ~flat at no extra policy cost; 50 reverses it. **Default 20; push toward 50 if the per-K-step logs show value_asym climbing in the real run** (where task pressure is weaker than this stress test).

**MPS note:** E16 originally used `index_copy_` (unimplemented on MPS); now uses a gather (`index_select`) so it runs on Apple Silicon too.

**Trainer integration sketch (original 05-30 plan; as-built landed 2026-05-31, commit e1c6d55 — masked KL + value_weight=20):**
```python
# In yinsh_ml/training/trainer.py train_step / train_epoch:
# Every K=10 batches, compute the symmetric regularizer.
# For each sample in batch (small loop — only every K steps):
#     sym_loss = symmetric_loss_term(self.network.network, state, ...)
# Add to total loss with small weight (alpha=0.1):
#     total_loss = policy_loss + value_loss + alpha * sym_loss_batch.mean()
#     total_loss.backward()
```

**Compute cost:** 4× forward per regularized step. At K=10, that's +0.3× total training time. Manageable on cloud.

**Production-recipe role (L3d, "Fix residual 25% by pulling weights into D2-symmetric subspace"):** complements L3a symmetric MCTS at inference ([[e8_symmetric_mcts]]) — L3a guarantees symmetric *output* regardless of weights; E16/L3d makes the *weights themselves* symmetric so the residual 25% (orbit-internal asymmetry) goes away. Stays default-OFF pending the next cloud run.

**Remaining work (Task 3):** next cloud run stacking L1+L2+E16 end-to-end — supervised pretrain with `--use-enhanced-encoding --label-smoothing 0.1 --enable-symmetric-reg` on a Dropout=0 net, then self-play with `enable_symmetric_reg: true` in the config. **No config enables E16 yet** — one still needs writing (mirror `branchD2_enhanced` + the symmetric_reg keys). Symmetric MCTS (L3a) stays as a deploy-time noise-reducer even once the network is symmetric.

## Provenance & links

- Source snapshots: 2026-05-30 ("E16 prototype"), 2026-05-31 ~14:00 UTC recovery snapshot (as-built details, value_weight investigation).
- Fixes the residual 25% (H_W) confirmed by [[e11_weight_symmetry_check]]; H_E ruled out by [[e14_augmenter_integrity]].
- Complements [[e8_symmetric_mcts]] (L3a inference-time symmetry).
- Foundation partners: [[l1_l2_dropout_labelsmoothing]] (L1 Dropout(0), L2 label smoothing).
- Deferred alternative if E16 falls short: E17 (explicit D2 weight tying / equivariant network, Cohen & Welling).
