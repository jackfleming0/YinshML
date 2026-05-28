# Step 2 — MCTS-400 Ceiling Experiment: GPU Runbook

Turnkey instructions for running the MCTS-400 controlled-lever experiment on a
fresh rented GPU box (vast.ai RTX 4090 or better), then screening the result on
the non-saturated yardstick.

**What this tests:** whether a *stronger self-play teacher* (MCTS-400 vs the
MCTS-200 used in Branch C) moves the network off the plateau. One variable
changes vs `wave3_branchC_mcts200.yaml`: search depth. See
`configs/wave3_branchC_mcts400.yaml` header and `VOLUME_PRETRAIN_RESULTS.md`.

**The decisions baked in (2026-05-21):**
- **Init = `best_supervised.pt`** (same as Branch C MCTS-200) → controlled comparison.
- **Anchor = `best_iter_4.pt`** (the MCTS-200 output from that same init).
- **SPRT `--sprt-p1 0.60`** (~70 Elo) — the smallest edge worth promoting.
- **5 iters × 100 games** (~30-32h, ~$30-40 on a 4090).

---

## 0. Before you touch the box (run on your laptop)

The box gets code via `git` and the two checkpoints via `scp`. Push the new
config so a plain `git clone` picks it up:

```bash
# from the repo root, on your laptop
git add configs/wave3_branchC_mcts400.yaml STEP2_MCTS400_RUNBOOK.md
git commit -m "Add MCTS-400 ceiling experiment config + GPU runbook"
git push origin training-pipeline-fixes
```

Confirm the two checkpoints exist locally (they are NOT in git — 131M each):
```bash
ls -lh models/yngine_volume_pretrain/best_supervised.pt   # the INIT
ls -lh models/branchC_volume_pretrain/best_iter_4.pt      # the ANCHOR (for screening)
```

---

## 1. Provision the box

- **vast.ai**: RTX 4090 (the Branch C runs used one; ~3.2h/iter at MCTS-200, so
  budget ~6h/iter here). Any single modern CUDA GPU works.
- **Image**: a PyTorch CUDA image (e.g. `pytorch/pytorch:*-cuda*-cudnn*-runtime`).
- **Disk**: ≥40 GB. The run dir (`runs_wave3_branchC_mcts400/`) holds 5 iters of
  checkpoints (~130M each ×3 variants/iter) + buffers; budget ~10-15 GB plus the
  two transferred checkpoints.
- Note the SSH host/port vast.ai gives you; export for convenience:
  ```bash
  export BOX="root@<host>" PORT=<port>      # e.g. root@1.2.3.4  -p 12345
  ```

## 2. Code + deps (on the box)

```bash
git clone https://github.com/jackfleming0/YinshML.git
cd YinshML
git checkout training-pipeline-fixes        # the branch with the config + fixed engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Transfer the two checkpoints (from your laptop)

```bash
# from laptop, repo root. Adjust -P <PORT> to your box.
scp -P $PORT models/yngine_volume_pretrain/best_supervised.pt \
    $BOX:~/YinshML/models/yngine_volume_pretrain/best_supervised.pt
scp -P $PORT models/branchC_volume_pretrain/best_iter_4.pt \
    $BOX:~/YinshML/models/branchC_volume_pretrain/best_iter_4.pt
```
(Create the dirs on the box first if `scp` complains:
`ssh -p $PORT $BOX 'mkdir -p ~/YinshML/models/yngine_volume_pretrain ~/YinshML/models/branchC_volume_pretrain'`.)

## 4. Smoke test (~30s — DON'T skip; it de-risks a 32h run)

Confirm the init checkpoint loads on CUDA and MCTS-400 self-play actually runs,
*before* committing a day of compute. This reuses the validated batched engine:

```bash
python scripts/eval_vs_frozen_anchor.py \
    --candidate models/yngine_volume_pretrain/best_supervised.pt \
    --anchor models/branchC_volume_pretrain/best_iter_4.pt \
    --num-games 2 --num-simulations 400 --device cuda
```
Expect: it loads both checkpoints (238/238 tensors), plays 2 games on `cuda`,
prints a verdict. If it crashes here, fix it now — not 6 hours into the run.

## 5. Launch the run

```bash
mkdir -p logs
nohup python scripts/run_training.py \
    --config configs/wave3_branchC_mcts400.yaml \
    --init-checkpoint models/yngine_volume_pretrain/best_supervised.pt \
    > logs/mcts400.log 2>&1 &
echo "PID $!"
```
Run inside `tmux`/`screen` so an SSH drop doesn't kill it. Warm-start should log
`Warm-starting from init checkpoint` and load 238/238 tensors clean.

## 6. Monitor

```bash
tail -f logs/mcts400.log
nvidia-smi -l 5          # GPU util should be high during self-play
```

**Healthy signs (this is a STABILITY-first run):**
- Anchor vs HA(d=1) stays **100%** (40/40) each iter from iter 2 on — the
  collapse tripwire. The original Branch C *failed* here (82.5%→60%); the
  value-grounded init fixed that, and MCTS-400 should hold it.
- Promotions happen (loose Wilson-0.20 gate); internal Elo prints per iter.

**Parsing gotcha (from `VOLUME_PRETRAIN_RESULTS.md`):** anchor WRs are NOT in
`metrics.json` (`anchor_eval` stays `{}`) — they're only in the run **log**,
lines `Anchor vs HeuristicAgent(d=1):` and `ANCHOR (prev,raw/mcts):`. Internal
Elo (`tournament_rating`) IS in each iter's `metrics.json`.

## 7. Screen the candidates — the actual measurement (on the box)

Run the SPRT eval on the GPU (far faster than MPS). The eval uses **64 sims**
(the validation-gate budget) — that's the *measurement* budget and is correctly
independent of the 400 sims used to *generate* the targets. Don't conflate them.

Screen the **final** iter first (most-trained candidate):
```bash
python scripts/eval_vs_frozen_anchor.py --sprt --sprt-p1 0.60 --device cuda \
    --anchor models/branchC_volume_pretrain/best_iter_4.pt \
    --candidate runs_wave3_branchC_mcts400/*/iteration_4/checkpoint_iteration_4_ema.pt \
    --output-json logs/mcts400_iter4_vs_frozen.json
```
(Per-iter EMA checkpoints live in `<run>/iteration_N/checkpoint_iteration_N_ema.pt`
— the `use_ema_for_eval` weights, which is what the gate promoted on.)

If iter 4 reads **STRONGER**, screen the earlier iters to see *when* it crossed
(cheap, and tells you whether it's still climbing). The `[0-3]` bracket glob
matches only real files, so no phantom cross-product paths:
```bash
python scripts/eval_vs_frozen_anchor.py --sprt --sprt-p1 0.60 --device cuda \
    --anchor models/branchC_volume_pretrain/best_iter_4.pt \
    --candidate runs_wave3_branchC_mcts400/*/iteration_[0-3]/checkpoint_iteration_[0-3]_ema.pt \
    --output-json logs/mcts400_trajectory_vs_frozen.json
```

## 8. Interpreting the verdict

- **STRONGER** (LLR crosses upper bound): MCTS-400 self-play beat the MCTS-200
  output — **search depth was the lever.** Re-freeze the new best as the anchor
  and climb again (MCTS-1000 next, or scale iters). The plateau was measurement-
  /teacher-limited, not a true ceiling.
- **NOT_STRONGER** (LLR crosses lower bound): doubling sims didn't move it past
  the ~70-Elo bar. Either the ceiling is real for this architecture, or the
  lever is elsewhere (→ Branch D architecture, audit Gap 1). Don't scale MCTS
  further on this arch.
- **INCONCLUSIVE** (hit `--sprt-max-games`): the true edge sits inside
  [0.50, 0.60). Decide if a sub-70-Elo gain is worth scaling for; if so, re-run
  that candidate with `--sprt-p1 0.55` to resolve the smaller gap (more games).

## 9. Pull results back + teardown

```bash
# from laptop: grab the small JSONs + the winning checkpoint (if STRONGER)
scp -P $PORT $BOX:~/YinshML/logs/mcts400_*.json ./logs/
scp -P $PORT $BOX:~/YinshML/logs/mcts400.log ./logs/
scp -P $PORT "$BOX:~/YinshML/runs_wave3_branchC_mcts400/*/iteration_4/checkpoint_iteration_4_ema.pt" \
    ./models/branchC_mcts400/        # only if it won — this becomes the new anchor
```
**TERMINATE the box via the vast.ai web UI** once results are pulled (the
previous box silently kept billing through a maintenance window — don't repeat
that). Everything needed is the JSONs + the one winning checkpoint.

Then update `VOLUME_PRETRAIN_RESULTS.md` §"Next steps" with the verdict.
