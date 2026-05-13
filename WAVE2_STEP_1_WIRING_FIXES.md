# Step 1 — Metric wiring fixes + value-outcome correlation investigation

**Estimated effort**: 1 day of focused work.
**Cloud cost**: $0 (all local).
**Pre-conditions**: Local repo on `training-pipeline-fixes`; task #11 (commit np.float_ patch) done; preflight test suite green.

This is the **gate** for any further cloud work. Step 2 can't produce trustworthy data until the Wave 2 acceptance gates are actually reporting the values they claim to report.

---

## The three wiring bugs, ranked by triviality

### Bug B2 — `train/effective_batch_size`: missing `log_scalar` call entirely

**Severity**: trivial fix.
**Diagnosis**: confirmed. The metric is computed at `yinsh_ml/training/trainer.py:1824` (`stats_accum['effective_batch_size'] = float(np.mean(effective_batch_sizes))`), included in the human-readable epoch summary log line (`Sampling: eff_bs=249.6/256 (97%)`), but never passed to `metrics_logger.log_scalar`. There is no call site for this metric anywhere in the trainer.

**Fix**: add one block to `trainer.py` after the epoch summary log. Suggested location is right after `log_training(epoch_metrics)` is called within `train_iteration`. Exact patch:

```python
# After the epoch summary block; mirrors the B3 telemetry pattern around line 1074.
if self.metrics_logger is not None and effective_batch_sizes:
    self.metrics_logger.log_scalar(
        'train/effective_batch_size',
        float(np.mean(effective_batch_sizes)),
        iteration=self.current_iteration,
    )
```

**Verification**:
1. Run `pytest yinsh_ml/tests/test_supervisor_metrics_wiring.py -v` — existing test should still pass.
2. Add a new test asserting `train/effective_batch_size` appears in the sidecar with a positive numeric value.
3. Run a 10-game × 1-iter smoke (`python scripts/run_training.py --config configs/warm_start_combined_recipe_smoke.yaml`) and inspect `metrics/iteration_0.json` for the metric.

---

### Bug B3 — `eval/value_outcome_correlation`: gate-check reads wrong list index

**Severity**: logging is correct; reader is wrong.
**Diagnosis**: the metric is logged once per `(checkpoint × mcts-mode)` pair inside the tournament's anchor-eval loop (`tournament.py:1079`). With `tournament_sliding_window: 3`, each iter logs ~3 correlations: one for the current candidate, plus historical comparisons against the prior 1-2 checkpoints. Sidecar entries are ordered: `[cand, hist1, hist2]`.

The runbook's gate-check snippet reads `entries[-1]['value']` — the **last** entry, which is the **oldest** historical comparison. Since historical comparisons use the same eval seed against the same anchor, they're byte-identical across iters (hence the "stale -0.186" we saw at iters 1 and 2).

**Inspect to confirm** (this is what we found in the current run):

```
iter 1 sidecar entries: [-0.1037, -0.1862]            (cand=iter1, hist=iter0)
iter 2 sidecar entries: [-0.3079, -0.1037, -0.1862]   (cand=iter2, hist=iter1, hist=iter0)
iter 3 sidecar entries: [-0.0711, -0.3079, -0.1037]   (cand=iter3, hist=iter2, hist=iter1)
iter 4 sidecar entries: [-0.0711, -0.3079]            (cand=iter4, hist=iter3)
```

The candidate's correlation is `entries[0]`, not `entries[-1]`.

**Two fix paths** — pick one:

#### Option B3-a: Fix the reader (1 line)
Change `entries[-1]` to `entries[0]` in the runbook §7 gate-check snippet, and in any dashboards/scripts that read this scalar. Cheap and immediate, but fragile — anyone reading the JSON later will hit the same trap.

#### Option B3-b: Fix the metric name (recommended)
Make the metric name unambiguous so each checkpoint's correlation has its own series:

```python
# tournament.py around line 1077-1081, replace:
metrics_logger.compute_and_log_value_outcome_correlation(step=iteration)

# with:
metrics_logger.compute_and_log_value_outcome_correlation(
    step=iteration,
    metric_name=f'eval/value_outcome_correlation/{candidate_label}',
)
# (requires adding the metric_name kwarg to metrics_logger.py:197 with a
# default of 'eval/value_outcome_correlation' for backwards-compat)
```

Then the gate-check fetches `eval/value_outcome_correlation/candidate_iteration_<N>` explicitly. Slightly more invasive but the data model becomes self-describing.

**Verification**:
1. Per-checkpoint correlation series visible in `metrics/iteration_<N>.json`.
2. New gate-check fetches the right series unambiguously.
3. The values match what the run log printed for each anchor-eval pair.

---

### Bug B1 — `mcts/effective_child_visits`: zero entries despite 4 call sites in the MCTS class

**Severity**: needs investigation — the producer code exists but never runs in batched-MCTS mode.

**What we know**:
- `_log_effective_child_visits` defined at `self_play.py:1026`.
- Called from 4 MCTS code paths: `self_play.py:755, 807, 981, 1017`.
- Sidecar shows **0 entries** for this metric across all 5 iters of the run.
- The recipe uses `use_batched_mcts: true` + `mcts_batch_size: 64`.

**Hypothesis** (probabilistic — verify before fixing):
The 4 call sites are all within the **MCTS class** (around `self_play.py:200`). In batched MCTS mode, MCTS instances may be created inside the `SelfPlay` runner class (around `self_play.py:1455+`) and dispatched via a different code path that doesn't go through any of those 4 call sites — or the `metrics_logger` isn't forwarded to those instances.

**Investigation path**:
1. Search for every `MCTS(` instantiation site: `grep -n "MCTS(" yinsh_ml/training/self_play.py`.
2. For each, check whether `metrics_logger=` is passed.
3. Trace the actual code path taken when `use_batched_mcts=True`. There's a constructor flag at `self_play.py:1455` (`use_batched_mcts: bool = True`). Find where it branches in the search loop.
4. If batched MCTS calls a different "search and pick policy" function: add a `self._log_effective_child_visits(root, budget)` call at the appropriate point there.

**Fix shape** (assuming the hypothesis holds):
- Add the `_log_effective_child_visits` call inside the batched-MCTS search-and-policy function, after sims are done and visit counts are read.
- OR: refactor so all MCTS code paths funnel through one "extract visits + log" helper.

**Verification**:
1. Smoke run with `use_batched_mcts: true` — sidecar should now show many `mcts/effective_child_visits` entries (one per search call, so thousands per iter).
2. Latest value should be `>= 0.7` per the Wave 2 gate.
3. Also verify with `use_batched_mcts: false` (sequential path) — that already worked, don't regress.

---

## The bigger question — negative value-outcome correlation at every measured iter

**This is the most important signal in the whole run** — and the §8 60-game eval suite corroborated it. Iter 1 EMA's raw-policy win rate against the depth-1 heuristic was **25%** at n=60 (well below the 33% baseline), while its 400-sim MCTS win rate was **76.7%**. The network's policy head alone is weak; MCTS at depth is compensating. The most likely upstream cause is a value-head sign issue — which is exactly what the value-outcome correlation has been telling us.

The runbook flags negative `eval/value_outcome_correlation` as the T1.3 regression signature:

> `eval/value_outcome_correlation < 0` — value head learning the WRONG sign (T1.3 regression — re-run `test_mcts_backprop_perspective.py`)

With the corrected gate-check (reading `entries[0]` instead of `entries[-1]`), the actual candidate value-correlation trajectory was:

```
iter 1: -0.1037
iter 2: -0.3079    ← worst (matches "iter 2 reverted" Elo signal)
iter 3: -0.0711    ← partial recovery
iter 4: -0.0711
```

All negative. The model's value head is producing predictions whose sign **anti-correlates** with terminal outcomes. That's not a noise issue — that's a directional bug in either:

- the training signal (value targets have wrong sign somewhere in the bootstrap), or
- the MCTS backprop perspective (perspective flips not handled correctly in YINSH's non-alternating turn structure), or
- the eval-time computation of `(root_value, terminal_outcome)` pairs (sign convention mismatch).

### Investigation plan

#### Step 1.1 — Run the existing regression tests with extra rigor

```bash
pytest yinsh_ml/tests/test_value_target_pipeline.py -v
pytest yinsh_ml/tests/test_mcts_backprop_perspective.py -v
pytest yinsh_ml/tests/test_value_head_diagnostics.py -v
```

If they all pass: the unit-test contract is correct in isolation, but production data doesn't match it. That means there's a missing assertion or a path the tests don't cover.

If any fail: that's your starting point — fix the failure and re-run training.

#### Step 1.2 — Manual inspection of a single training sample

Load the iter 1 EMA checkpoint and a known game from the replay buffer:

```python
from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.utils.encoding import StateEncoder

# Load checkpoint
nw = NetworkWrapper('runs_warm_start_combined_recipe_cloud/20260512_151604/iteration_1/checkpoint_iteration_1_ema.pt')

# Pick a known-terminal position from late-game in the replay buffer
# (positions where the candidate clearly wins or loses)
# Feed through the network and compare predicted_value sign to terminal_outcome sign
```

If predicted values systematically point the wrong way: the value head has the wrong sign convention.

#### Step 1.3 — Trace the value-target write site

The value head trains on targets stored in the experience buffer. The bootstrap is at `yinsh_ml/training/trainer.py:1369` (per the W1a fix and the assertion in `test_value_target_pipeline.py`). Verify:

```bash
grep -n "self.experience.values\[idx\]" yinsh_ml/training/trainer.py
```

Make sure the indexing matches what the test asserts.

#### Step 1.4 — Check the eval-time pair computation

The pairs are logged at `tournament.py:944-952` via `log_eval_value_pair(root_value, terminal_outcome)`. Read the surrounding code carefully — make sure:

- `root_value` is in the *candidate's* perspective (not the anchor's).
- `terminal_outcome` is also in the *candidate's* perspective (`+1` if candidate won, `-1` if lost).
- No double-negation or perspective flip is happening.

A common bug: outcome stored from white's perspective, root_value stored from current-player's perspective; if candidate is black, these don't align unless you flip one.

### Possible paths forward

Once the bug is located, possible fixes:

- **If it's the value-target bootstrap**: add an explicit perspective annotation to the experience buffer entries, fix the read site, re-train from iter 1 EMA and verify correlation goes positive.
- **If it's the MCTS backprop perspective**: the W1 T1.3 fix may not have covered every YINSH-specific path. YINSH has capture sequences where the player to move doesn't switch — those are the edge cases to instrument.
- **If it's the eval-time computation**: pure logging bug, easy fix in `tournament.py:944-952`.

In all cases: **add a regression test** that uses a deterministic mini-game with a known outcome and asserts the correlation is positive. The current tests don't cover end-to-end correlation sign.

---

## What to evaluate Step 1 on

When you think Step 1 is done, you should be able to answer **yes** to all of these:

1. The `train/effective_batch_size` series appears in `metrics/iteration_<N>.json` with values around `204-256` per iter.
2. The `mcts/effective_child_visits` series appears with thousands of entries per iter, latest values `>= 0.7`.
3. The `eval/value_outcome_correlation` series (or per-checkpoint series) is unambiguous about which value belongs to the current candidate.
4. The 10 Wave 1 preflight tests still pass (`pytest yinsh_ml/tests/test_mcts_serial_vs_batch_parity.py …` per runbook §5).
5. A 10-game × 1-iter smoke run produces a sidecar JSON where all 5 Wave 2 gates have non-`None` values.
6. You've located *where* the negative value-outcome correlation originates. (Fixing it is a separate question — but you should at least know whether it's targets, backprop, or eval-time computation.)

If item 6 is unresolved but items 1-5 are done, you can still proceed to Step 2 — but understand that Step 2's value-correlation gate will continue to fire RED and that signal will be the headline result of the next run.

---

## What's deliberately NOT in scope for Step 1

- Tuning hyperparameters
- Adding new training features
- Changing the recipe in any way except wiring fixes
- Running anything on cloud

Resist the urge. Step 1 is mechanical and diagnostic. The hypothesis-validating work is Step 2.
