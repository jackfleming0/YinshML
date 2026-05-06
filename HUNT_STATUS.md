# Policy collapse hunt — status

**Branch:** `policy-collapse-hunt` (off `ablation-result-followup`)
**TL;DR:** Bug found, fixed, regression-tested. Patch is committed.
The cloud_run_v1 50/0/0 was a single line in `clear_pytorch_memory`
nulling out every BatchNorm running stat at the end of each iteration.
Saved checkpoints had no BN keys; on reload they fell back to
mean=0/var=1 and the policy head's effective output collapsed.

## How it was found

Probe of `runs/20260421_125023` iter 0..5 (same recipe family as
cloud_run_v1, ran locally on Mac before cloud):

| iter | entropy | unique top-1 / 128 | top-1 conf |
|---:|---:|---:|---:|
| 0 | 3.06 | 95  | 0.32  (normal early) |
| 1 | **0.00** | **2**   | **1.00** (mode collapse) |
| 2 | **0.00** | **1**   | **1.00** (worse) |
| 3 | 3.66 | 104 | 0.22  (recovered — `_reset_network_objects` re-built the net) |
| 4 | **8.91** | 6   | **0.0002** (uniform reset) |
| 5 | 4.05 | 105 | 0.18  (relearning) |

Checkpoint sizes alternated by exactly 84 KB. The smaller files
(iters 1, 2, 4) were missing 87 of 238 state-dict keys — every BN
`running_mean` / `running_var` / `num_batches_tracked`. The pattern
matches the supervisor's `iteration_counter % 3 == 0` rebuild cadence.

## The bug

`yinsh_ml/training/supervisor.py::clear_pytorch_memory`, pre-fix:

```python
for module in self.network.network.modules():
    if hasattr(module, '_buffers'):
        for key in list(module._buffers.keys()):
            buffer = module._buffers[key]
            if buffer is not None and torch.is_tensor(buffer):
                # Re-register as zeros to clear cached tensors    ← comment lies
                module._buffers[key] = None                       ← deregisters BN
```

Setting `_buffers[key] = None` deregisters the buffer from the module.
Subsequent forward passes can't update running stats. The next saved
state_dict excludes the buffer entirely. On reload, BN reverts to
default init (zero/one stats) — the conv stack normalizes by wildly
wrong statistics — every downstream feature is corrupted.

The 84 KB delta between "good" and "bad" checkpoints is exactly the BN
running-stat tensor footprint (87 keys × ~256 channels × 4 bytes).

## Why it explains cloud_run_v1's 50% / 0% / 0%

- 400-sim MCTS overcomes a broken policy: tree search visits dominate
  the bad prior → coin flip vs depth-1 heuristic.
- 48-sim MCTS doesn't have rollouts to escape: every move steered by
  the wrong-BN policy head → 0/60.
- Raw policy with no scaffold → 0/60 deterministically.

The original framing "value learned, policy didn't" was wrong: both
heads were broken at inference. The value head only *appears* OK
because expected-value-over-classes averages out distortion, while the
policy head's argmax over 7433 slots amplifies it.

## What's committed (this branch)

| sha | what |
|---|---|
| `91b7d22` | `scripts/probe_policy.py` — diagnostic that found it |
| `cd5f0d5` | fix + regression test (`yinsh_ml/tests/test_supervisor_bn_preservation.py`) |
| `d86b28e` | `CLOUD_RUN_V1_POSTMORTEM.md` updated with resolution section |
| `fb6b328` | `save_model` save-time guard: refuses to write checkpoints with missing BN keys |

Tests pass: `pytest yinsh_ml/tests/test_supervisor_bn_preservation.py yinsh_ml/tests/test_augmentation.py` → 38 passed.

## In-progress validation

A 4-iter local smoke (`configs/smoke_post_fix.yaml`) is running to
confirm the policy actually learns post-fix instead of collapsing.
First-iter probe already shows healthy structure (entropy ≈ ln(N),
32/128 unique argmax — fresh-init learning, not collapse). Expect
entropy to *decrease* and unique-argmax to *grow* across the next
iterations as the policy peakies up onto learned moves.

## What to do next on cloud

1. Pull `policy-collapse-hunt` on the cloud box (rebase onto
   `ablation-result-followup` first if you'd like a clean trail).
2. Re-launch `configs/cloud_run_v1.yaml` as-is. The recipe was probably
   fine; the bug was clobbering everything else.
3. Use `scripts/probe_policy.py --run-dir <run_dir>` periodically
   during training to catch collapse early. Healthy trajectory:
   entropy starts near `ln(7433) ≈ 8.9`, decreases monotonically,
   stays well above 0; unique top-1 stays high (>50% of states).
4. If it still fails — but with healthy probe signatures — the failure
   is recipe-shaped (peaky targets, hybrid-eval shaping, value-head LR
   factor), not code-shaped. The original 5-knob postmortem candidates
   are still in `CLOUD_RUN_V1_POSTMORTEM.md` for that case.

## Things I considered and ruled out

- **Joint backward / optimizer split** — already correct since
  `6c77ee7` (2026-04-27). Verified by reading the patched code and a
  synthetic gradient-flow test.
- **Augmentation correctness** — D2 transforms preserve mass on legal
  moves for both colors and all phases. Verified empirically.
- **Phase weights leaking into loss** — they're sampling weights only.
- **Policy target normalization with illegal moves** — MCTS only
  writes mass into `root.children` slots (legal-only by construction).
- **`value_head_lr_factor` mechanism** — implemented as a separate
  optimizer LR (Adam policy + SGD value). Not gradient scaling.

The only one of the postmortem's five candidates that has any
remaining concern is whether `value_head_lr_factor=5.0` plus the
discrimination loss creates an *additional* problem on top of the now-
fixed BN bug. Worth watching in the next training run; not worth
preemptively changing.
