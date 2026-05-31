# Symmetry-Fixes Foundation Run — GPU Runbook

Turnkey, step-by-step instructions for the **L1 + L2 + E16** foundation run on a
rented GPU box (vast.ai RTX 4090 or better). Covers the full pipeline:
continued-pretrain → self-play, with a mandatory smoke test before the spend.

**What this tests:** whether the three policy/symmetry fixes break the
post-iter-1 plateau and fix the lopsided-opening pathology.
- **L1** — Dropout(0.3)→0 in the policy head. *Already in `model.py`* (it's the
  architecture), so it applies to every net built below. **Nothing to set.**
- **L2** — Label smoothing ε=0.1 on the hard-target CE. Applied in **Step A
  (continued-pretrain)** via `--label-smoothing 0.1`.
- **E16** — D2 symmetric-weight regularizer. Applied in **BOTH** Step A
  (`--enable-symmetric-reg`) and Step B self-play (the `trainer.symmetric_reg`
  block in the config). `value_weight=20` is measured.

**The decisions baked in (2026-05-31):**
- **Pretrain init = `models/supervised_2026-05-27/best_supervised.pt`** — the
  existing 15-channel, spatial-head, classification checkpoint (engine-pretrained).
  We *continue-pretrain* it with the fixes rather than re-pretraining from
  scratch. This is the validated dry-run recipe (`dry_run_dropout_plus_ls.py`
  recovered F6 modal 28% / white-WR 46% from exactly this setup), it's a
  controlled comparison vs Branch C/D, and it's cheap.
- **NOT regenerating the engine corpus for this run.** A fresh 15ch engine
  corpus (E7) is explicitly "stacked-on-top, not foundation — apply only after
  the fixes land" (handoff + `EXPERIMENT_BACKLOG.md`). It's a stronger base but
  an *unvalidated extra variable*; deferred to Appendix A.
- **Continued-pretrain corpus = `expert_games/hvh_full_game_15ch.npz`** (human
  BGA games, 107K positions, already 15-channel — no regen).
- **Self-play = `configs/symmetry_fixes_mcts200.yaml`** (mirrors Branch D.2 +
  the `symmetric_reg` block). 5 iters × 100 games at MCTS-200.

**Go/no-go:** the plateau was *post-iter-1*. Watch whether value discrimination /
value-outcome correlation / WR-vs-anchor keep climbing through iters 2–3 instead
of flattening. **You'll know by iter 2–3** — don't wait for iter 5.

---

## 0. Before you touch the box (run on your LAPTOP)

The box gets code via `git`, and the checkpoint + corpus via `scp` (neither is
in git — too big). Push the branch so a plain `git clone` picks up the config,
the fixes, and this runbook:

```bash
# repo root, on your laptop
git status                                   # confirm you're on policy-symmetry-fixes
git push origin policy-symmetry-fixes
```

Confirm the two large files exist locally (sizes approximate):
```bash
ls -lh models/supervised_2026-05-27/best_supervised.pt    # the INIT (~130 MB)
ls -lh expert_games/hvh_full_game_15ch.npz                # the CORPUS (~11 MB)
```

> If `best_supervised.pt` isn't local, pull it from the gh release snapshot
> (`gh release list` → the model snapshot), or use any other 15ch/spatial/
> classification checkpoint you trust as the init.

---

## 1. Provision the box

- **vast.ai**: RTX 4090 (Branch C/D used one; ~3.2h/iter at MCTS-200 → budget
  ~half-a-day for 5 iters + the short pretrain). Any single modern CUDA GPU works.
- **Image**: a PyTorch CUDA image (e.g. `pytorch/pytorch:*-cuda*-cudnn*-runtime`).
- **Disk**: ≥40 GB. Run dir (`runs_symmetry/`) holds 5 iters × ~130 MB × ~3
  variants + buffers (~10–15 GB) plus the transferred init checkpoint + corpus.
- Note the SSH host/port vast.ai gives you:
  ```bash
  export BOX="root@<host>" PORT=<port>        # e.g. root@1.2.3.4   port 12345
  ```

## 2. Code + deps (on the BOX)

```bash
git clone https://github.com/jackfleming0/YinshML.git
cd YinshML
git checkout policy-symmetry-fixes
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Transfer the init checkpoint + corpus (from your LAPTOP)

```bash
# from laptop, repo root. Adjust the port.
ssh -p $PORT $BOX 'mkdir -p ~/YinshML/models/supervised_2026-05-27 ~/YinshML/expert_games'
scp -P $PORT models/supervised_2026-05-27/best_supervised.pt \
    $BOX:~/YinshML/models/supervised_2026-05-27/best_supervised.pt
scp -P $PORT expert_games/hvh_full_game_15ch.npz \
    $BOX:~/YinshML/expert_games/hvh_full_game_15ch.npz
```

## 4. Smoke test (~3–5 min — DO NOT SKIP; it de-risks a multi-hour run)

Exercises the WHOLE chain at tiny scale on the box, before the real spend:
continued-pretrain with E16 → a self-play iteration driven by the config (so the
`symmetric_reg` YAML→mode_settings→trainer wiring actually fires). Run it:

```bash
# (on the box, in the YinshML venv)
bash scripts/smoke_symmetry_pipeline.sh
```

**Expect, in order:**
1. `E16 symmetric regularizer ON: ... value_weight=20.0 ...` during pretrain,
   then `Training complete` + a saved checkpoint under `/tmp/smoke_pretrain/`.
2. The self-play iteration: `E16 sym-reg: kl=... value_asym=...` lines from the
   trainer (proves the config wiring reached the self-play loop), and the
   iteration finishing without a traceback.
3. Final line: `SMOKE OK`.

If it crashes, fix it now — not 4 hours into the run. (This same script is what
was run locally before launch; see §"Smoke validated" at the bottom.)

## 5. Step A — Continued-pretrain WITH the fixes (on the BOX)

Produces the warm-start checkpoint. Dropout=0 is automatic; you add L2 + E16.

```bash
mkdir -p logs models/supervised_symmetry
nohup python scripts/run_supervised_pretraining.py \
    --data expert_games/hvh_full_game_15ch.npz \
    --checkpoint models/supervised_2026-05-27/best_supervised.pt \
    --use-enhanced-encoding --value-mode classification --value-head-type spatial \
    --label-smoothing 0.1 --enable-symmetric-reg \
    --epochs 5 --batch-size 128 --lr 5e-5 \
    --output-dir models/supervised_symmetry \
    > logs/pretrain.log 2>&1 &
echo "PID $!"
```
Run inside `tmux`/`screen` so an SSH drop doesn't kill it. ~10–30 min on a 4090
(107K positions × 5 epochs).

**Healthy signs (tail `logs/pretrain.log`):**
- `E16 symmetric regularizer ON: weight=0.1, value_weight=20.0, every_k_steps=10`.
- Empty-board policy peak **lands ~20–30%** by the last epoch — sharp but not
  collapsed. If ~1% → dropout somehow still on (shouldn't be). If ~100% → label
  smoothing not applied. **Either extreme = stop before Step B.**
- `E16 sym-reg: kl=... value_asym=...` lines; `value_asym` should hold/decline.
- A `best_supervised.pt` saved under `models/supervised_symmetry/`.

## 6. Step B — Self-play loop (on the BOX)

Warm-start from the Step A checkpoint:

```bash
nohup python scripts/run_training.py \
    --config configs/symmetry_fixes_mcts200.yaml \
    --init-checkpoint models/supervised_symmetry/best_supervised.pt \
    > logs/selfplay.log 2>&1 &
echo "PID $!"
```
Warm-start should log `Warm-starting from init checkpoint` / loaded NNN/NNN
tensors clean (the head/encoder must match — spatial + 15ch, which Step A
produced).

## 7. Monitor (on the BOX)

```bash
tail -f logs/selfplay.log
nvidia-smi -l 5            # GPU util high during self-play
```

**Watch for:**
- **E16 is alive:** `E16 sym-reg: kl=... value_asym=...` every ~10 train steps.
  If you NEVER see these, the regularizer didn't activate — kill and check the
  config wiring (`enable_symmetric_reg` in `mode_settings`).
- **value_asym trend:** should hold flat or decline across iters. If it climbs,
  bump `trainer.symmetric_reg.value_weight` toward 50 and relaunch.
- **The plateau test (the point of the run):** value discrimination /
  value-outcome correlation / WR-vs-anchor should **keep improving through iters
  2–3**, not flatten. Internal Elo (`tournament_rating`) is in each iter's
  `metrics.json`; anchor WRs are only in the **log** (lines `Anchor vs
  HeuristicAgent(d=1):`) — `anchor_eval` in `metrics.json` stays `{}` (known
  parsing gotcha from `VOLUME_PRETRAIN_RESULTS.md`).
- **Openings symmetrized:** after iter 1, optionally pull a checkpoint and run
  `scripts/measure_model_openings.py` — the A5 orbit should be balanced, not
  72/0/0/0.

## 8. Pull results + teardown

```bash
# from laptop — sync the run dir back
rsync -avz -e "ssh -p $PORT" $BOX:~/YinshML/runs_symmetry/ ./runs_symmetry/
rsync -avz -e "ssh -p $PORT" $BOX:~/YinshML/logs/ ./logs_symmetry/
```
Then **destroy the vast.ai instance** (stop billing). Commit the EMA checkpoint
you want to keep / deploy.

**Post-run validation (on laptop):**
- Re-run E11 (value-head symmetry) on the final checkpoint — `value_asym` should
  be well below the iter1_ema baseline.
- H2H the new EMA vs `iter1_ema` (`scripts/measure_h2h.py`).
- If it's stronger: this is also the new model to deploy to the analysis board
  (and Task 1 symmetric MCTS keeps it varied).

---

## Appendix A — (DEFERRED) stronger base via a 15-channel engine corpus

Only after the fixes are confirmed to help. Regenerate the engine corpus at 15ch
(the existing `yngine_volume.npz` is 6-channel) and pretrain from it instead of
the human corpus. Path B is built + tested (`regenerate_npz_with_enhanced_encoder.py`):

```bash
python scripts/regenerate_npz_with_enhanced_encoder.py \
    --input expert_games/yngine_volume.npz \
    --output expert_games/yngine_volume_15ch_mmap/ --workers 16   # ~1-2h CPU, ~10GB
# then Step A with: --data-dir expert_games/yngine_volume_15ch_mmap/  (no --checkpoint,
# i.e. pretrain from scratch) and more epochs.
```
Caveat: channel 13 (turn number) is zeroed (not recoverable from a 6ch decode) —
a 14.5/15-channel corpus. Minor, documented in D2_PREP.md.

## Appendix B — local (Apple Silicon) dry run

The short human-corpus pretrain (Step A) is small enough to run on a Mac's MPS in
~5–15 min/epoch if you want to produce the checkpoint locally and only rent the
box for self-play. Pass `--device mps`. E16 runs on MPS (gather, not index_copy_).
Self-play (Step B) wants the GPU box for reasonable wall-clock.
