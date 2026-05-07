# Cloud re-run runbook — cloud_run_v1 with BN fix

Branch: `policy-collapse-hunt` (off `ablation-result-followup`).
Goal: re-run `configs/cloud_run_v1.yaml` with the BN-stat-trash fix in
place. Same recipe as the original 25-iter run, but every iteration is
now actually load-bearing instead of every third.

Total expected wall time: ~8h. Total spend: ~$4 at $0.40/hr.

## Cloud box

Already provisioned. SSH:

```bash
ssh -p 3310 root@63.143.59.21 -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` forwards the dashboard port to your
laptop (visit `http://localhost:8080` after launch if you want to
watch live metrics).

> **Note**: that IP/port pair is for *this* instance. If you reboot,
> reprovision, or spin up a different one, update the SSH command and
> the rsync `cloud:` references below.

For convenience, drop this into `~/.ssh/config` on your laptop so the
later `rsync cloud:` shortcuts work:

```ssh-config
Host cloud
    HostName 63.143.59.21
    Port 3310
    User root
    LocalForward 8080 localhost:8080
```

After that you can `ssh cloud` and `rsync … cloud:…` directly.

## 1. Push the branch (already done — for reference)

```bash
# from /Users/jackfleming/PycharmProjects/YinshML on laptop
git push origin policy-collapse-hunt
```

Already pushed: `origin/policy-collapse-hunt` is current as of the
last laptop session.

## 2. Pull the branch on the cloud box

The repo should already exist from previous runs. If not, clone it
first; either way end up at the right SHA.

```bash
ssh cloud
cd /workspace/YinshML  # or wherever you cloned it before

# If the repo isn't there yet:
#   cd /workspace && git clone https://github.com/jackfleming0/YinshML.git && cd YinshML

git fetch origin
git checkout policy-collapse-hunt
git pull

# Activate the env you set up before.
source venv/bin/activate  # or however you named it

# Verify CUDA + branch
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
git log --oneline -8
# should include: 5441552 cdf71b0 260165d 7fcd9da fb6b328 d86b28e cd5f0d5 91b7d22
```

If `pip install -r requirements.txt` is needed (new deps since last
visit), run it. The branch hasn't added any.

## 3. Sync the heuristic weights (only if not already on the box)

```bash
# from laptop — check first whether it's already there:
ssh cloud "ls -la /workspace/YinshML/analysis_output/heuristic_evaluator_model.pkl"

# If missing, copy it up:
rsync -avz --progress \
  analysis_output/heuristic_evaluator_model.pkl \
  cloud:/workspace/YinshML/analysis_output/
```

## 4. Sanity check the fix landed (30 sec)

```bash
cd /workspace/YinshML
python -m pytest yinsh_ml/tests/test_supervisor_bn_preservation.py \
                 yinsh_ml/tests/test_wrapper_save_guard.py -v
```

Should report `3 passed`. If anything fails, **stop** and figure out
why before launching the run.

## 5. Launch the run

```bash
# Background it so an SSH disconnect doesn't kill it.
nohup python scripts/run_training.py --config configs/cloud_run_v1.yaml \
    > /workspace/YinshML/cloud_v1_rerun.log 2>&1 &
echo $! > /workspace/YinshML/cloud_v1_rerun.pid
```

Run dir lands at `runs/<timestamp>/` (config has `save_dir: runs_cloud_v1`
overridden — actually it's set to `runs_cloud_v1`, so look there).

## 6. Mid-run gut-check at iter 3 (~1h after launch)

This is the load-bearing decision point. **Don't skip it.**

```bash
# On cloud, after iter 3 finishes (watch the log for "ITERATION 3 SUMMARY"):
RUN=$(ls -dt runs_cloud_v1/*/ | head -1)

python scripts/probe_policy.py --run-dir "$RUN" --num-states 256 --device cuda
```

Then look at the per-iteration metrics:

```bash
for i in 0 1 2 3; do
  echo "=== iter $i ==="
  cat "$RUN/iteration_$i/metrics.json" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); \
      print(f'  policy_loss={d.get(\"policy_loss\",-1):.3f}  value_loss={d.get(\"value_loss\",-1):.3f}  anchor={d.get(\"anchor_win_rate\",-1)}')"
done
```

**Healthy signatures (continue the run):**
- Probe entropy at iter 3 is well below `ln(7433) ≈ 8.91` AND well
  above `0`. Anywhere in 4-7 range is fine.
- Unique top-1 across 256 states is >100. (Pre-fix this was 1-3.)
- `anchor_win_rate` is climbing iter-over-iter (or at least non-zero by
  iter 2-3). Anchor is reliable now that BN works.
- Policy loss is *decreasing* iteration-over-iteration. Value loss too.

**Broken signatures (kill it; spin up the hyperparameter sweep):**
- Probe entropy = 0.00 with confidence ~1.0 → still collapsing somehow.
  Different bug from the BN one we fixed.
- Anchor win rate is 0/N at iter 3 with no upward trend.
- Policy loss flat or rising.

If broken, kill with:
```bash
kill $(cat /workspace/YinshML/cloud_v1_rerun.pid)
```
…and we go to the hyperparameter sweep (Step 8).

## 7. Let it cook + final eval (after iter 25)

If iter 3 looks healthy, leave the run alone. ~7 more hours.

When complete, find the best-by-anchor checkpoint:

```bash
RUN=$(ls -dt runs_cloud_v1/*/ | head -1)
for i in $(seq 0 24); do
  if [ -f "$RUN/iteration_$i/metrics.json" ]; then
    python3 -c "import json; d=json.load(open('$RUN/iteration_$i/metrics.json')); \
      print(f'iter $i: anchor={d.get(\"anchor_win_rate\",-1)}, elo={d.get(\"tournament_rating\",-1):.1f}')"
  fi
done
```

Pick the iter with the highest `anchor_win_rate` — call it `BEST`.

Run the three-test eval from the original postmortem:

```bash
CK="$RUN/iteration_$BEST/checkpoint_iteration_${BEST}.pt"
EMA="$RUN/iteration_$BEST/checkpoint_iteration_${BEST}_ema.pt"

# 400-sim MCTS — what we know overcomes a broken policy
python scripts/eval_vs_heuristic.py --checkpoint "$EMA" \
    --num-games 60 --depth 1 --device cuda --label best_400
# 48-sim MCTS — training-config sim count
python scripts/eval_vs_heuristic.py --checkpoint "$EMA" \
    --num-games 60 --depth 1 --mcts-simulations 48 --device cuda --label best_48
# Raw policy — the failure mode we were hunting
python scripts/eval_vs_heuristic.py --checkpoint "$EMA" \
    --num-games 60 --depth 1 --no-mcts --device cuda --label best_raw
```

**Pre-fix baseline was 50% / 0% / 0%.** What we want to see now:
- 400-sim: ≥55% (anything above 50% is real strength; the original hit
  this via search alone)
- 48-sim: meaningfully above 0% — even 20-30% means the policy head is
  contributing something
- Raw: ≥10% is a real win. The "raw policy can stand alone" threshold.

Anything that moves the **raw** column off zero is the headline result
— that's the column the postmortem said was the regression guard.

## 8. Hyperparameter sweep (only if Step 6 said abort)

`CLOUD_TRAINING_PLAN.md §4` has the template — but with the BN fix in
place, the three knobs most worth varying are:

```yaml
# Config A: lower value head LR factor
trainer:
  value_head_lr_factor: 1.0   # was 5.0

# Config B: zero discrimination loss
trainer:
  discrimination_weight: 0.0  # was 0.5

# Config C: less peaky targets
self_play:
  final_temp: 0.5             # was 0.1
```

Each at 5 iter, ~$1-2 each. Pick winner by anchor win rate. Run that
recipe at full 25-iter scale.

## 9. After the run — regardless of outcome

```bash
# from laptop, pull artifacts back
rsync -avz cloud:/workspace/YinshML/runs_cloud_v1/ ./runs_cloud_v1_rerun/
rsync -avz cloud:/workspace/YinshML/cloud_v1_rerun.log ./

# Tear down the instance to stop the meter (do this from the
# provider's web UI — Vast.ai/RunPod/etc.)
```

Then:
- Update `CLOUD_RUN_V1_POSTMORTEM.md` with the new numbers.
- If raw column >0%, also write a follow-up note for the next agent
  (which knobs you'd push next, what surprised you).

## What success looks like overall

- Probe at iter 3: healthy entropy + diverse argmax + climbing anchor.
- Best iter's three-test: 400-sim well above 50%, 48-sim above 0%,
  raw above 0%.
- Promotions firing on consecutive iterations, not the iter%3 cadence
  (that pattern was the BN bug — should disappear now).

If all three hold, the recipe is validated and the next conversation
is "scale up to 50 iter + bigger buffer per the postmortem's Forward
section." If only the first two hold (raw still 0%), recipe is
*almost* there but needs Step 8.
