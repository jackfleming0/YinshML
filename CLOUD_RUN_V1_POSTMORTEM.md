# cloud_run_v1 (2026-05-06) — Training Quality Post-Mortem

> **Status**: post-mortem of the 25-iteration run launched off
> `cloud_run_v1.yaml`. Run completed cleanly (good — that's the GPU
> scaling thread). Trained model has a specific, identifiable
> failure: **the policy head learned nothing useful**. This doc is
> the diagnosis and the proposed next ablation.
>
> Companion to `GPU_SCALING_RESULTS.md` (infrastructure thread) and
> `INVESTIGATION_RESULTS.md` (the earlier B1/B2 white-wins
> investigation — *different* pathology, mostly resolved).

## Run summary

| metric | value |
|---|---|
| Iterations | 25 (completed) |
| Wall clock | 8.71h |
| Self-play throughput | 714 g/hr (predicted 702) |
| Promotions | 5 (iters 1, 2, 7, 11, 12); none after iter 12 |
| Buffer size at end | 100K / 100K (full) |
| Best model by ELO | iter 12 (1550.8) |

Iter 12 was the last promoted checkpoint. After that, every gate
failed for 12 straight iterations. The metric-level summary made it
look like ~73 ELO drift; the deeper picture is different.

## What we measured (iter 12, n=60 each)

```
iter 12 vs HeuristicAgent(depth=1):
  400-sim MCTS:   30/60  →  50%   CI95 [0.38, 0.62]
   48-sim MCTS:    0/60  →   0%   CI95 [0.00, 0.06]
  raw policy:      0/60  →   0%   CI95 [0.00, 0.06]
```

```
iter 1 vs iter 12 (head-to-head):
  temp=0:    iter 12 wins 19/30  →  63%   CI95 [0.22, 0.55]
  temp=0.5:  iter 12 wins 17/30  →  57%   CI95 [0.27, 0.61]
  per-color split: balanced both temps (no white-bias)
```

`anchor_win_rate` from the in-training tournament was a noisy n=4
measurement and gave 0.25 — that turned out to be statistical noise
relative to the actual 0.50 measured at n=60.

## What it means

The model's value head learned something. The policy head learned
nothing — *worse* than nothing, in fact: 0/60 vs depth-1 heuristic
when used alone. With 8× the search budget it was trained at
(400 sims vs 48), the value head + MCTS gets the model to a coin
flip. At self-play conditions (48 sims), the model loses 60-0.

This is the textbook "value-head shortcut" failure: gradient signal
flows preferentially through the value head, the policy head's
contribution to wins is small, and SGD finds the local minimum
where value does all the work.

The B1/B2 white-wins-100% pathology that motivated
`INVESTIGATION_RESULTS.md` is **not** present here. Per-color
split is balanced at both temperatures. The fixes that landed in
PR #9 (Phase D warm-start) appear to have stuck. This is a
*different* failure mode at the same destination.

## Why the policy head failed — three contributors

### 1. `value_head_lr_factor: 5.0` is too aggressive

The trainer multiplies the value head's gradients by 5×. With
`value_loss_weights: [0.5, 0.5]` the *loss* is balanced, but the
effective LR is 5× higher on value-head parameters. The value
target is a low-entropy scalar (`outcome ∈ {-1, 0, +1}`) and a
stable training signal — it converges fast. The policy target is a
high-entropy 7433-class softmax over noisy MCTS visit counts — it
needs slow, consistent updates to converge. Five-to-one favoring
the value head means the policy never gets a chance.

### 2. `num_simulations: 48` is too low for clean policy targets

MCTS visit counts at 48 sims are dominated by a handful of nodes
that the search visited the most. With 7433 possible move slots
and roughly 30-100 legal moves per state, 48 visits means the
visit-count distribution is essentially "1-3 modes plus zeros."
This is a noisy training target for the policy head. Worse: when
the value head's eval is also noisy (early in training), the noise
compounds.

The previous post-bitboard tuning runs (B1, B2) used 200 sims.
cloud_run_v1 inherited `num_simulations: 48` from
`cloud_smoke.yaml` without re-thinking whether that was right for
a sustained training run.

### 3. `annealing_steps: 30` cools to argmax mid-opening

With `temp_clamp_fraction: 0.6`, temperature reaches `final_temp`
around move 18. For a 90-move game that's mid-opening. The
remaining ~70 moves of self-play are essentially deterministic
play, which gives the policy targets *no exploration variance* to
learn from. Combined with the noisy MCTS targets above, the policy
head sees the same handful of move sequences with the same
peaky-near-deterministic targets game after game.

## Proposed next step: `configs/ablation_policy_recovery.yaml`

A short (5-iter, ~2h, ~$1) ablation that changes only those three
knobs:

| knob | old | new | reasoning |
|---|---|---|---|
| `value_head_lr_factor` | 5.0 | 1.0 | Equal LR for both heads. Stop drowning out the policy. |
| `num_simulations` | 48 | 100 | Cleaner MCTS visit counts; matches B-series tuning band. |
| `late_simulations` | 32 | 80 | Same reasoning, late-game. |
| `annealing_steps` | 30 | 60 | Keep self-play diverse longer. |
| `num_iterations` | 25 | 5 | Just a probe; want signal, not a finished model. |
| `games_per_iteration` | 200 | 50 | Faster iteration cadence; total run ~2h. |

Other knobs unchanged from `cloud_run_v1.yaml`.

### Pass criterion

After 5 iterations, run the same three-test eval on the latest
promoted checkpoint:

- 400-sim vs heuristic d=1 (n=60)
- training-config-sim vs heuristic d=1 (n=60) — at the new 100/80
  sim budget so it matches training conditions
- raw policy vs heuristic d=1 (n=60)

The win condition is **raw policy ≥ 0.20** (~12/60 wins). Anything
above zero proves the policy head is learning *something*. If we
hit 0.20+, run a longer training pass with the same knobs. If raw
stays at 0%, the bug is somewhere else — most likely in the
trainer's target construction or loss assembly, not in the
hyperparameters.

## What we won't do

- **Spin up another 25-iteration overnight with the same recipe.**
  We know it plateaus after iter 12. Burn rate without information.
- **Touch the GPU scaling work.** That thread is closed (PRs #11,
  #12, #13). Infrastructure is sound, throughput is predictable.
- **Revisit the white-wins protocol.** Not the bug we're looking
  at; `INVESTIGATION_RESULTS.md` already closed that one.

## Forward — what comes after the ablation

If the ablation lands a non-zero raw-policy number, the next
training run is:

- Same hyperparameter changes as the ablation
- 50 iterations (not 25 — the post-iter-12 plateau may have been
  premature convergence, longer horizon may help once the policy
  gradient flows)
- Larger buffer (200K — at 100 sims/move and 200 games/iter the
  buffer cycles less, so older diverse positions stay longer)
- Mid-run checkpoint to look at policy-head parameter norm growth
  (a cheap "policy is learning" signal worth tracking inline)

If the ablation produces 0% raw again, the bug is in the trainer.
Read `yinsh_ml/training/trainer.py` for:
- How policy targets are constructed from MCTS visit counts
- How the policy / value loss are combined into a single backward()
- Any conditional zeroing or masking on policy gradients
- Whether `value_head_lr_factor` is actually scaling LR or scaling
  gradients (those have different consequences with momentum)

---

## Ablation result (2026-05-06 evening)

The ablation completed. **Hyperparameter changes did nothing.** Same
exact pattern as cloud_run_v1's iter 12:

```
ablation iter 3 (best promoted) vs HeuristicAgent(depth=1), n=60:
  400-sim MCTS:   30/60  →  50%   CI95 [0.38, 0.62]
  100-sim MCTS:    0/60  →   0%   CI95 [0.00, 0.06]
  raw policy:      0/60  →   0%   CI95 [0.00, 0.06]

iter 1 vs iter 3 (n=30, temp=0):
  iter 1 wins 17-13  →  57%
  per-color split: 9W/8B (balanced)
```

Wall clock: 1.39h, 5 iterations, 3 promotions. Cost ~$0.70.

Two observations beyond "the fix didn't work":

1. **Identical numbers.** Three knob changes — `value_head_lr_factor
   5.0→1.0`, `num_simulations 48→100`, `annealing_steps 30→60` —
   produced *exactly* the same 50% / 0% / 0% pattern. None of those
   knobs are reachable from this hole.
2. **iter 1 beats iter 3.** Earlier checkpoint outperforms later
   one within the same run. The arena promotions are passing
   noise. Even the gating signal is unreliable here.

So the post-mortem's pass criterion (`raw policy ≥ 0.20`) failed.
The bug is not at the hyperparameter layer. **The trainer is the
next file to read.**

## Tomorrow's reading list — `yinsh_ml/training/trainer.py`

Five candidates worth grepping for, in priority order. First three
are smoking-gun shaped; last two are subtler.

| # | What to look for | Smoking gun if... |
|---|---|---|
| 1 | **Policy target normalization** — is `visit_counts / sum(visit_counts)` happening *after* invalid-move masking? | Targets put mass on illegal moves; policy trains on garbage. |
| 2 | **Loss summation into `.backward()`** — is `(policy_loss + value_loss).backward()` actually how it's combined, or are they backwarded separately? | If separate, only one backwarded value reaches the optimizer. |
| 3 | **Phase-weight application** — `RING_PLACEMENT=1.0, MAIN_GAME=2.0, RING_REMOVAL=0.5` — applied to value AND policy or only one? | Whole phases of training never reach the policy head. |
| 4 | **Augmentation transformation** — does the policy target rotate/flip when the state does? | Policy is trained on mismatched (state, target) pairs — looks like noise. |
| 5 | **`value_head_lr_factor` mechanism** — separate parameter group LR, or scaling gradients? With Adam, those are *very* different. | If implemented as gradient scaling on value only, the optimizer's running statistics get distorted in a way that interferes with policy learning across heads. |

Suggested first command:

```bash
grep -nE "policy_loss|value_loss|backward|policy_target|visit_counts|phase_weight" \
    yinsh_ml/training/trainer.py | head -40
```

That will scope the read to the relevant blocks. Then read those
sections in full, top to bottom, before changing anything.

### Validation pattern once a fix lands

The same three-test eval is the regression guard. Re-run on
whatever the new "iter 3" is after the trainer fix. Compare to the
two reference points already saved on the cloud box:

| reference | 400-sim | 100/48-sim | raw |
|---|---|---|---|
| cloud_run_v1 iter 12 | 50% | 0% | 0% |
| ablation iter 3      | 50% | 0% | 0% |

Anything that moves the **raw** column off zero is a real fix. The
400-sim column is likely already saturated against this depth-1
heuristic; don't read too much into it.

## How to reproduce the diagnosis

```bash
# On the cloud box, with the run dir from cloud_run_v1:
cd /workspace/YinshML
RUN=$(ls -dt runs_cloud_v1/*/ | head -1)

# Three-way comparison (n=60 each, ~5min/2min/40min respectively)
python scripts/eval_vs_heuristic.py \
    --checkpoint "$RUN/iteration_12/checkpoint_iteration_12.pt" \
    --num-games 60 --depth 1 --device cuda --label iter12_400
python scripts/eval_vs_heuristic.py \
    --checkpoint "$RUN/iteration_12/checkpoint_iteration_12.pt" \
    --num-games 60 --depth 1 --mcts-simulations 48 --device cuda --label iter12_48
python scripts/eval_vs_heuristic.py \
    --checkpoint "$RUN/iteration_12/checkpoint_iteration_12.pt" \
    --num-games 60 --depth 1 --no-mcts --device cuda --label iter12_raw
```

Expected on the existing iter 12 checkpoint: 50% / 0% / 0%.
Different numbers there mean the model has changed since this
post-mortem was written.

---

## Resolution (2026-05-06 evening)

The bug is **not** any of the five trainer candidates listed above.
None of them fire in the training path; the audit of each came back
clean (joint backward via `total_loss = policy_loss + value_loss`
landed in `6c77ee7` on 2026-04-27, well before cloud_run_v1).

The actual bug is in `TrainingSupervisor.clear_pytorch_memory`.
End-of-iteration cleanup walked every module's `_buffers` and set each
tensor to `None` "to free memory":

```python
for module in self.network.network.modules():
    if hasattr(module, '_buffers'):
        for key in list(module._buffers.keys()):
            buffer = module._buffers[key]
            if buffer is not None and torch.is_tensor(buffer):
                module._buffers[key] = None    # ← deregisters BN running stats
```

That deregistered every BatchNorm's `running_mean`, `running_var`, and
`num_batches_tracked`. The next saved checkpoint was missing 87 of 238
state-dict keys — every BN running stat across the 12 res blocks plus
the heads. On reload (`strict=False`), BN initialised to defaults
(`mean=0`, `var=1`), the conv stack normalised by the wrong statistics,
and the policy head's effective output collapsed.

### How it was found

Probing iter 0..5 of a local pre-cloud run (`runs/20260421_125023`,
same recipe as cloud_run_v1 modulo sims/workers) showed an unmistakable
trajectory:

| iter | entropy | unique top-1 / 128 | top-1 conf | signature |
|---:|---:|---:|---:|:---|
| 0 | 3.06 | 95 | 0.32 | normal early |
| 1 | 0.00 | 2  | 1.00 | mode collapse |
| 2 | 0.00 | 1  | 1.00 | mode collapse, worse |
| 3 | 3.66 | 104 | 0.22 | recovered (post-`_reset_network_objects()`) |
| 4 | 8.91 | 6  | 0.0002 | uniform reset |
| 5 | 4.05 | 105 | 0.18 | re-learning |

Checkpoint file sizes alternated by exactly 84 KB, matching the BN
running-stat tensor footprint. Iters 0, 3, 5 had the full state dict;
iters 1, 2, 4 didn't. `iteration_counter % 3 == 0` triggers
`_reset_network_objects()` — which builds a fresh `NetworkWrapper` and
loads the saved state — re-registering BN buffers from scratch with
default values. That's why the trajectory thrashes every three
iterations rather than monotonically degrading.

### Why the cloud_run_v1 / ablation 50% / 0% / 0% pattern matches

- 400-sim MCTS at eval can search past a broken policy: visit counts
  dominated by tree expansion overcome the wrong prior, recovering to a
  coin flip vs depth-1 heuristic.
- 48-sim or 100-sim MCTS doesn't have enough rollouts to escape the
  prior. Every move it picks is steered by a policy head whose features
  came out of a wrong-BN forward pass.
- Raw policy is the broken policy with no MCTS scaffold at all → 0/60
  vs the heuristic, deterministically.

The "value head learned, policy head didn't" framing in the original
diagnosis was wrong — both heads were broken at inference. The value
head only *appears* to work because expected value over a 7-class soft
distribution averages out errors, while the policy head's argmax over
7433 slots amplifies them.

### Fix

Remove the buffer-mutation block from `clear_pytorch_memory`. Replace
with a small loss-history clear (the only cache that actually holds
non-trivial tensor refs from training). Add
`yinsh_ml/tests/test_supervisor_bn_preservation.py` to lock the
invariant.

Commits: `91b7d22` (probe diagnostic), `cd5f0d5` (fix + regression
test) — branch `policy-collapse-hunt`.

### Validation

Post-fix smoke (`configs/smoke.yaml`, 2 iter): both checkpoints have
the full 238-key state dict including all 87 BN buffer keys, and
neither shows mode-collapse or uniform-reset signatures. The pre-fix
broken pattern is gone.

The right next move on cloud is **not** another ablation under the
same recipe — that just confirms the fix and burns budget. With BN
preserved, the cloud_run_v1 hyperparameters were probably already
fine; the model is worth retraining on the same recipe to find out.
The "Forward — what comes after the ablation" plan above (50 iter,
larger buffer, mid-run policy-norm checkpoint) is the right direction
once the fix is on cloud.
