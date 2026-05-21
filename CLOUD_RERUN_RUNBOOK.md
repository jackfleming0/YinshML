# Cloud Wave 2 runbook — warm_start_combined_recipe

**Branch**: `training-pipeline-fixes` (off `policy-collapse-hunt`).
**Goal**: run the canonical Wave 2 validation recipe with all Wave 1 fixes wired through end-to-end, validate the LR=0.0001 + deep_256x18 baseline against the new telemetry gates.

Total expected wall time on cloud GPU: ~4-6h for the canonical recipe (200 games × 5 iters). Smoke ran 5h on local MPS at 50×2 — cloud should compress that materially.

> **Context for whoever runs this next:** the previous runbook (this file at commit `b701fd5`) was for the BN-stat-trash fix on `policy-collapse-hunt`. Wave 1 (commits `2bf1721` through `f1ba916`) supersedes that work — see `memory/project_tier1_verification.md` and `memory/project_wave2_smoke.md` for what changed and what's already validated.

## Cloud box

```bash
ssh -p 26654 root@23.158.136.85 -L 8080:localhost:8080
```

`-L 8080:localhost:8080` forwards the dashboard port; visit `http://localhost:8080` after launch for live metrics.

For convenience, drop this into `~/.ssh/config`:

```ssh-config
Host cloud
    HostName 23.158.136.85
    Port 26654
    User root
    LocalForward 8080 localhost:8080
```

Then `ssh cloud` and `rsync … cloud:…` work directly.

> **If you reboot/reprovision/spin up a different instance**: update the SSH stanza here. Treat this as the canonical record of the current instance.

## 1. Push the branch (already pushed before you start)

```bash
# from laptop
git push origin training-pipeline-fixes
```

## 2. Bring the cloud box up to date

Repo lives at `/workspace/YinshML`. Python env is conda `main` at `/venv/main`; should auto-activate on login. If not:

```bash
conda activate main      # or: source /venv/main/bin/activate
```

One-time DNS fix if the box's resolv.conf is empty:

```bash
cat /etc/resolv.conf
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf
```

Check out the branch:

```bash
ssh cloud
cd /workspace/YinshML

# Fresh-clone case:
#   cd /workspace && git clone https://github.com/jackfleming0/YinshML.git && cd YinshML

git fetch origin
git checkout training-pipeline-fixes
git pull

# Verify CUDA + branch + package
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import yinsh_ml; print('yinsh_ml ok:', yinsh_ml.__file__)"
git log --oneline -12
# Should include (Wave 1 + Wave 2 work):
#   f1ba916  telemetry: wire MetricsLogger in supervisor + proxy delegate
#   ecc49e9  wire: B2 value-outcome correlation in run_anchor_eval (post-W1e)
#   64ba563  merge: W1e dual-mode anchor eval + det-collapse routing
#   f18daff  merge: W1b telemetry bundle (B1, B2, B3)
#   2dca238  merge: W1c iteration-aware Dirichlet tapering (T3.6)
#   8741a44  merge: W1-NEW batched MCTS dedup fix (T1.1)
#   dd21265  config: warm_start_combined_recipe — canonical Wave 2 recipe
#   5ad3a4f  feat(eval): T4.9 dual-mode anchor + T5.4 det-collapse routing
#   6a8bb60  infra: T4.10 + T4.11 — worker crash counter + gate-revert mtime guard
#   1354686  feat: cap phase-weight oversampling + effective-batch-size telemetry
#   096bbeb  fix: T1.2 — value target reads buffered terminal outcome
#   4c017a5  fix(mcts): T1.1 — stop wasting B-1 sims at the unexpanded root
```

If `import yinsh_ml` fails:

```bash
pip install -e .
# also if requirements.txt changed:
pip install -r requirements.txt
```

## 3. Get the init checkpoint onto the box

**Important context**: the canonical recipe header says `models/supervised_deep_256x18/best_supervised.pt`. That checkpoint was on an earlier cloud instance and is **probably lost**. Pick the path that fits your situation:

### Option A (fastest) — rsync `supervised_seed` from laptop

Same architecture (256-ch × deep), what the Wave 2 local smoke validated against. Different supervised data than the original `supervised_deep_256x18` so it's NOT apples-to-apples vs the original 33% baseline, but it's the same starting strength the smoke gave us.

```bash
# from laptop
ssh cloud "ls -la /workspace/YinshML/models/supervised_seed/best_supervised.pt 2>/dev/null || echo MISSING"

# if MISSING:
rsync -avz --progress \
  models/supervised_seed/best_supervised.pt \
  cloud:/workspace/YinshML/models/supervised_seed/
```

The recipe's invocation already references this path; nothing to edit.

### Option B (proper) — rebuild `supervised_deep_256x18` on cloud

If you have the original training data (`expert_games/training_data.npz` or similar) and want apples-to-apples with the 33% baseline, run the supervised pretrain to regenerate the deep checkpoint:

```bash
# on cloud
cd /workspace/YinshML
python scripts/run_supervised_pretraining.py \
    --data expert_games/training_data.npz \
    --num-channels 256 \
    --num-blocks 18 \
    --epochs 30 \
    --output-dir models/supervised_deep_256x18 \
    --device cuda
```

~1-2h depending on data size + epochs. Then update the launch command in step 5 to use this path instead.

### Option C — rsync `supervised_deep_256x18` from laptop

Only if you actually have it locally (check first):

```bash
ls models/supervised_deep_256x18/best_supervised.pt 2>/dev/null || echo "not on laptop either"
```

If present, rsync up just like Option A. If "not on laptop either", fall back to A or B.

## 4. Sync the heuristic weights (one-time, only if missing)

```bash
ssh cloud "ls -la /workspace/YinshML/analysis_output/heuristic_evaluator_model.pkl 2>/dev/null || echo MISSING"

# if MISSING:
rsync -avz --progress \
  analysis_output/heuristic_evaluator_model.pkl \
  cloud:/workspace/YinshML/analysis_output/
```

Required for hybrid evaluation mode (the recipe sets `evaluation_mode: hybrid`).

## 5. Pre-flight: run Wave 1 test suite (~60s)

These tests are the regression guards for the seven Wave 1 workstreams. If any fail, **stop** and triage before launching.

```bash
cd /workspace/YinshML
python -m pytest \
    yinsh_ml/tests/test_mcts_serial_vs_batch_parity.py \
    yinsh_ml/tests/test_value_target_pipeline.py \
    yinsh_ml/tests/test_mcts_backprop_perspective.py \
    yinsh_ml/tests/test_epsilon_mix.py \
    yinsh_ml/tests/test_replay_buffer_oversampling_cap.py \
    yinsh_ml/tests/test_self_play_worker_crash_counter.py \
    yinsh_ml/tests/test_supervisor_gate_revert_mtime_guard.py \
    yinsh_ml/tests/test_telemetry_safeguards.py \
    yinsh_ml/tests/test_anchor_eval_dual_and_routing.py \
    yinsh_ml/tests/test_supervisor_metrics_wiring.py \
    -v
```

Expect ~150 passed. If any fail, **stop** and check `git log` against the expected commit list in §2.

## 6. Launch the run

```bash
# pick the init path based on your §3 choice
INIT=models/supervised_seed/best_supervised.pt
# or:
# INIT=models/supervised_deep_256x18/best_supervised.pt

nohup python scripts/run_training.py \
    --config configs/warm_start_combined_recipe.yaml \
    --init-checkpoint "$INIT" \
    > /workspace/YinshML/wave2_cloud.log 2>&1 &
echo $! > /workspace/YinshML/wave2_cloud.pid
```

Run dir lands at `runs_warm_start_combined_recipe/<timestamp>/`.

## 7. Mid-run gut-check at iter 1 (~1-1.5h after launch on cloud GPU)

> `anchor.skip_first_n_iterations: 1` is set in the recipe, so iter 0 has no anchor eval. **Iter 1 is the first iteration with the Wave 2 acceptance gates measurable.**

After "ITERATION 1 SUMMARY" appears in the log:

```bash
RUN=$(ls -dt runs_warm_start_combined_recipe/*/ | head -1)

# Headline numbers
for i in 0 1; do
  echo "=== iter $i ==="
  cat "$RUN/iteration_$i/metrics.json" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); \
      print(f'  policy_loss={d.get(\"policy_loss\",-1):.3f}  value_loss={d.get(\"value_loss\",-1):.3f}  anchor_raw={d.get(\"anchor_eval\",{}) or \"empty\"}  anchor_mcts={d.get(\"anchor_eval_mcts\",{}) or \"empty\"}')"
done
```

Then the five Wave 2 acceptance gates from `metrics/iteration_<N>.json` (the new MetricsLogger sidecar wired in by commits `aff1346` + `f1ba916`):

```bash
python3 - <<'PY'
import json
from pathlib import Path

run = sorted(Path('runs_warm_start_combined_recipe').glob('*/'), key=lambda p: p.stat().st_mtime)[-1]
for iter_n in (0, 1):
    m_path = run / 'metrics' / f'iteration_{iter_n}.json'
    if not m_path.exists():
        print(f"iter {iter_n}: metrics/iteration_{iter_n}.json missing")
        continue
    m = json.load(open(m_path))
    scalars = m.get('metrics', m).get('scalars', {})
    def latest(name):
        entries = scalars.get(name, [])
        return entries[-1]['value'] if entries else None
    print(f"--- iter {iter_n} ---")
    print(f"  mcts/effective_child_visits      = {latest('mcts/effective_child_visits')}  (gate >= 0.7)")
    print(f"  train/policy_target_entropy_mean = {latest('train/policy_target_entropy_mean')}  (gate > 0)")
    print(f"  train/effective_batch_size       = {latest('train/effective_batch_size')}  (gate >= 0.8 * 256 = 204.8)")
    print(f"  eval/value_outcome_correlation   = {latest('eval/value_outcome_correlation')}  (gate > 0 by iter 2, rising toward 1.0)")
    print(f"  eval/deterministic_collapse_count= {latest('eval/deterministic_collapse_count')}  (gate == 0)")
PY
```

### Healthy signatures (let it cook)

- `mcts/effective_child_visits >= 0.7` — direct regression check for the T1.1 batched-MCTS fix
- `train/policy_target_entropy_mean > 0` and not collapsing toward 0 — model is exploring
- `train/effective_batch_size >= 204.8` (80% of 256) — W1d phase-weight cap is doing its job
- `eval/value_outcome_correlation > 0` and rising — T1.2 value-target fix is producing useful learning signal
- `eval/deterministic_collapse_count == 0` — no argmax mirages
- `anchor_eval` (raw policy) and `anchor_eval_mcts` (MCTS-assisted) both populated with non-empty dicts including `wilson_ci`, `elo_delta` (W1e dual-mode)

### Broken signatures (kill, investigate)

- `mcts/effective_child_visits < 0.5` — batched MCTS still wasting sims (re-run parity test on this branch; consider reverting `4c017a5`)
- `train/policy_target_entropy_mean -> 0` across phases — policy collapsing to argmax; check Dirichlet noise wiring (`epsilon_mix_iteration_start/end`) and reduce
- `eval/value_outcome_correlation < 0` — value head learning the WRONG sign (T1.3 regression — re-run `test_mcts_backprop_perspective.py`)
- `eval/value_outcome_correlation` stays at ~0 through iter 2 — value head not learning at all; check that W1a fix actually applied (`grep "self.experience.values\[idx\]" yinsh_ml/training/trainer.py` near line 1369)
- `eval/deterministic_collapse_count > 0` — model fell into a deterministic line; check eval temperature (recipe sets `raw_policy_temperature: 0.5`, should prevent this)
- `anchor_eval_mcts` is `{}` while `anchor_eval` is populated — MCTS-mode anchor crashed; check log for "Anchor eval failed (non-fatal)" — would mean the `_SupervisorMetricsProxy.log_eval_value_pair` wiring regressed (was fixed in `f1ba916`)

To kill:

```bash
kill $(cat /workspace/YinshML/wave2_cloud.pid)
```

## 8. Let it cook + final eval (after iter 5)

If iter 1 is healthy, leave the run. Recipe is 5 iters. Expected ~4-5 more hours.

When complete, identify the best iter by `anchor_eval_mcts` win rate:

```bash
RUN=$(ls -dt runs_warm_start_combined_recipe/*/ | head -1)
for i in $(seq 0 4); do
  if [ -f "$RUN/iteration_$i/metrics.json" ]; then
    python3 -c "import json; d=json.load(open('$RUN/iteration_$i/metrics.json')); \
      raw=d.get('anchor_eval',{}); mcts=d.get('anchor_eval_mcts',{}); \
      print(f'iter $i: raw_wr={raw.get(\"win_rate\",-1)}, mcts_wr={mcts.get(\"win_rate\",-1)}, elo={d.get(\"tournament_rating\",-1):.1f}')"
  fi
done
```

Pick the iter with highest `mcts_wr` — call it `BEST`.

Run the three-eval validation:

```bash
RUN=$(ls -dt runs_warm_start_combined_recipe/*/ | head -1)
BEST=4   # or whatever you picked
CK="$RUN/iteration_$BEST/checkpoint_iteration_${BEST}.pt"
EMA="$RUN/iteration_$BEST/checkpoint_iteration_${BEST}_ema.pt"

# 400-sim MCTS — what we know overcomes a broken policy
python scripts/eval_vs_heuristic.py --checkpoint "$EMA" \
    --num-games 60 --depth 1 --device cuda --label best_400

# 48-sim MCTS — training-config sim count (this is what the recipe actually uses)
python scripts/eval_vs_heuristic.py --checkpoint "$EMA" \
    --num-games 60 --depth 1 --mcts-simulations 48 --device cuda --label best_48

# Raw policy — does the policy head stand alone?
python scripts/eval_vs_heuristic.py --checkpoint "$EMA" \
    --num-games 60 --depth 1 --no-mcts --device cuda --label best_raw
```

Note: `eval_vs_heuristic.py` was updated in commit `5ad3a4f` (W1e) to default `--temperature 0.5` instead of `0.0` argmax. If you want the old argmax-determinism mode for diagnosis only, add `--temperature 0.0` explicitly.

### Wave 2 success criteria

The 33% baseline (warm_start_deep_lowlr) on cloud hit:
- 400-sim: validated above 30% stochastic
- 48-sim: ~33% stochastic
- Raw: needs measuring — this was the open question pre-Wave-1

**With Wave 1 fixes**, the hypothesis is:
- 48-sim: meaningfully above 33% (the load-bearing claim — fix delta should show here)
- Raw: above the 33% baseline's raw column (T1.2 value-target fix should improve raw policy quality)
- 400-sim: still strong

Any of those three above the baseline is real evidence the Wave 1 fixes improved training quality, not just plumbing.

## 9. After the run

```bash
# from laptop, pull artifacts back
rsync -avz cloud:/workspace/YinshML/runs_warm_start_combined_recipe/ \
    ./runs_warm_start_combined_recipe_cloud/
rsync -avz cloud:/workspace/YinshML/wave2_cloud.log ./

# Pull the metrics sidecar JSON specifically (it's small and useful for offline analysis)
rsync -avz cloud:/workspace/YinshML/runs_warm_start_combined_recipe/*/metrics/ \
    ./runs_warm_start_combined_recipe_cloud_metrics/

# Tear down the instance from the provider UI (Vast.ai / RunPod / etc.)
```

Then:
- Write a Wave 2 cloud post-mortem (companion to `memory/project_wave2_smoke.md`).
- If raw-policy column is meaningfully above baseline, the Tier 1 fix story is validated and we can move to Wave 3 (C++ engine enforcement, longer runs).
- If gates 1-3 (effective_child_visits, entropy, eff_batch) all green but the strength numbers are flat vs baseline — fixes are correct but recipe needs more iters or a different LR.

## What success looks like overall

1. All five Wave 2 acceptance gates pass at iter 1 and stay green through iter 5.
2. Best-iter raw-policy win rate ≥ baseline raw column (whatever it was — measure both for the comparison).
3. No `Anchor eval failed (non-fatal)` warnings in the log (those mean the metrics wiring regressed).
4. No reverted iterations on the Wilson gate (or at most 1/5).

If 1 + 2 hold, the Tier 1 fixes are validated and the recipe is the new floor.
