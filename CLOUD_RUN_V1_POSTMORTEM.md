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
