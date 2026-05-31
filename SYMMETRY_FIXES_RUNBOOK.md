# Symmetry-Fixes Foundation Run — GPU Runbook

Turnkey, step-by-step instructions for the **opening-fix foundation run** on a
rented GPU box (vast.ai RTX 4090/5090 or better). Full pipeline:
**build E10 corpus → from-scratch pretrain → self-play.**

**What this fixes (five levers, all landed in code):**
- **L1** — Dropout(0.3)→0 in the policy head. *In `model.py`* (architecture) → every
  net here gets it. The plateau's root cause (30% dropout forced near-uniform policy).
- **L2** — Label smoothing ε=0.1 on the supervised hard-target CE. Stops the
  *opposite* failure (collapse to a single modal opening). `--label-smoothing 0.1`.
- **E16** — D2 symmetric-weight regularizer, in BOTH pretrain and self-play. Fixes
  the lopsided-orbit asymmetry (A5 72% / K7,E1,G11 ~0%). value_weight=20 (measured).
- **E2** — placement value-grounding. Distills deep-search VALUE into RING_PLACEMENT
  positions so the value head can actually evaluate openings — the *referee* that
  lets self-play fairly arbitrate opening styles. In the self-play config.
- **E10** — placement-diversified corpus. Mixes human / yngine-engine / random
  placements (+ 4× D2 aug) so the corpus *exposes* every opening style instead of
  pre-judging by omission. The engine slice is capped (35%) so its wall-clustering
  is *represented, not dominant* — it might be a real edge; let outcomes decide.

**Decisions baked in (2026-05-31):**
- **From-scratch pretrain** on the E10 corpus — NOT continue-from-`best_supervised.pt`.
  The pathology is in that checkpoint's weights (dropout-baked policy + asymmetric
  value head); starting from a symmetric random init lets E16 *maintain* symmetry
  instead of *undoing* it. (Continue-pretrain is the cheap fallback — Appendix A.)
- **Engine slice = yngine only** (the real external engine's placements), not the
  current model's. **No BGA** (dropped for v1).
- **Budget ~$100.** The whole run is realistically ~$15–30; $100 is comfortable
  false-start room.

**Go/no-go:** the plateau was *post-iter-1*. Watch whether value discrimination /
value-outcome correlation / WR-vs-anchor keep climbing through **iters 2–3**
instead of flattening — and whether the opening orbit is balanced (A5≈K7≈E1≈G11).

---

## 0. Before you touch the box (on your LAPTOP)

```bash
git status                         # on policy-symmetry-fixes
git push origin policy-symmetry-fixes
ls -lh expert_games/yngine_volume.npz            # 6ch engine corpus (the big input)
ls -lh expert_games/hvh_placement_only_15ch.npz  # human placements (E10 source)
```
> If `yngine_volume.npz` isn't local, pull it from the gh release snapshot
> (`gh release list` → the `yngine_volume.npz` asset). No init checkpoint needed —
> we pretrain from scratch.

## 1. Provision the box

- **RTX 4090/5090** (or better). **≥64 GB RAM** — the 15ch regen + E10 build hold
  large arrays in memory. **≥60 GB disk** — the 15ch corpus is ~10–15 GB, the E10
  corpus + run dir another ~20 GB.
  ```bash
  export BOX="root@<host>" PORT=<port>
  ```

## 2. Code + deps (on the BOX)

```bash
git clone https://github.com/jackfleming0/YinshML.git && cd YinshML
git checkout policy-symmetry-fixes
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Transfer the inputs (from your LAPTOP)

```bash
ssh -p $PORT $BOX 'mkdir -p ~/YinshML/expert_games'
scp -P $PORT expert_games/yngine_volume.npz           $BOX:~/YinshML/expert_games/
scp -P $PORT expert_games/hvh_placement_only_15ch.npz $BOX:~/YinshML/expert_games/
```

## 4. Smoke test (~3–5 min — DO NOT SKIP; it de-risks the whole run)

Runs the *entire* path at tiny scale: build a tiny E10 corpus → from-scratch
pretrain (L2+E16) → a self-play iteration via the real config (E16 fires).
```bash
bash scripts/smoke_symmetry_pipeline.sh
```
Expect it to end with `SMOKE OK`. If it crashes, fix it now — not 10 hours in.

## 5. Step A0 — Regenerate the engine corpus at 15 channels (on the BOX)

`yngine_volume.npz` is 6-channel; the run is 15-channel. Path B re-encodes it
(built + tested):
```bash
tmux new -s build   # so it survives an SSH drop
python scripts/regenerate_npz_with_enhanced_encoder.py \
    --input expert_games/yngine_volume.npz \
    --output expert_games/yngine_volume_15ch_mmap/ --workers 16
```
~1–2 h CPU, ~10 GB out. (Channel 13 / turn-number is zeroed — not recoverable from
a 6ch decode; minor, documented in D2_PREP.md.)

## 6. Step A1 — Build the E10 placement-diversified corpus (on the BOX)

```bash
python scripts/build_e10_corpus.py \
    --engine-corpus expert_games/yngine_volume_15ch_mmap/ \
    --human-placements expert_games/hvh_placement_only_15ch.npz \
    --output expert_games/e10_corpus_15ch/ \
    --human-frac 0.40 --engine-frac 0.35 --random-frac 0.25 \
    --max-main-game 4000000          # caps main-game to fit RAM (~30 GB); raise if you have more
```
Prints the placement split + per-source counts + the final size. **RAM note:** the
builder holds the assembled corpus in memory before the mmap write; `--max-main-game
4000000` keeps it ~30 GB. The guard will refuse + tell you the cap if you'd OOM.

## 7. Step A2 — From-scratch pretrain with the fixes (on the BOX)

No `--checkpoint` (from scratch). Dropout=0 is automatic; you add L2 + E16.
```bash
mkdir -p logs models/supervised_symmetry
nohup python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/e10_corpus_15ch/ \
    --use-enhanced-encoding --value-mode classification --value-head-type spatial \
    --label-smoothing 0.1 --enable-symmetric-reg \
    --epochs 12 --batch-size 256 --lr 1e-3 \
    --output-dir models/supervised_symmetry \
    > logs/pretrain.log 2>&1 &
echo "PID $!"   # run inside tmux
```
~40–70 min/epoch on a 4090 at full corpus (faster on a 5090) → budget a few hours
for 12 epochs (~$5–9).

**Healthy signs (`tail -f logs/pretrain.log`):**
- `E16 symmetric regularizer ON: ... value_weight=20.0`.
- **Empty-board policy peak lands ~20–30%** by the last epochs — sharp but not
  collapsed (NOT ~1% = dropout still on; NOT ~100% = label smoothing missing).
- `E16 sym-reg:` lines with `value_asym` holding low (it starts symmetric, so it
  should *stay* low, not climb).
- `best_supervised.pt` saved under `models/supervised_symmetry/`.

## 8. Step B — Self-play loop (on the BOX)

Warm-start from the Step A2 checkpoint. The config carries E16 + E2.
```bash
nohup python scripts/run_training.py \
    --config configs/symmetry_fixes_mcts200.yaml \
    --init-checkpoint models/supervised_symmetry/best_supervised.pt \
    > logs/selfplay.log 2>&1 &
echo "PID $!"   # tmux
```

## 9. Monitor (on the BOX)

```bash
tail -f logs/selfplay.log ; nvidia-smi -l 5
```
- **E16 alive:** `E16 sym-reg: kl=... value_asym=...` every ~10 train steps. Never
  seeing it ⇒ regularizer off — kill and check `enable_symmetric_reg` in the config.
- **E2 alive:** search-consistency lines on placement positions; the value head's
  placement predictions should sharpen across iterations.
- **value_asym** holds low / declines. If it climbs, bump
  `trainer.symmetric_reg.value_weight` toward 50 and relaunch.
- **The plateau test:** value discrimination / value-outcome correlation /
  WR-vs-anchor keep improving through **iters 2–3**. (Internal Elo is in each iter's
  `metrics.json`; anchor WRs are only in the log — `anchor_eval` in `metrics.json`
  stays `{}`, a known parsing gotcha.)
- **Openings:** after iter 1, `scripts/measure_model_openings.py` on a checkpoint —
  the A5 orbit should be balanced, not 72/0/0/0. If the engine's wall-clustering is
  genuinely strong, you'll see it played *symmetrically* across all 4 orbit cells.

## 10. Pull results + teardown

```bash
# laptop:
rsync -avz -e "ssh -p $PORT" $BOX:~/YinshML/runs_symmetry/ ./runs_symmetry/
rsync -avz -e "ssh -p $PORT" $BOX:~/YinshML/logs/ ./logs_symmetry/
```
Then **destroy the instance.** Post-run (laptop): re-run E11 on the final
checkpoint (`value_asym` should be well below the iter1_ema baseline), H2H the new
EMA vs `iter1_ema` (`scripts/measure_h2h.py`), and if it's stronger, deploy it to
the analysis board (Task 1 symmetric MCTS keeps it varied). Don't forget to push
Task 1 live separately (`git push` + `yinsh-redeploy`).

---

## Appendix A — cheap fallback: continue-pretrain (skip the corpus build)

If you want a same-day signal before the bigger from-scratch compute, continue-
pretrain the existing 15ch checkpoint instead of Steps A0–A2 (it's the validated
dry-run recipe, but it *patches* an already-pathological checkpoint rather than
fixing the root):
```bash
python scripts/run_supervised_pretraining.py \
    --data expert_games/hvh_full_game_15ch.npz \
    --checkpoint models/supervised_2026-05-27/best_supervised.pt \
    --use-enhanced-encoding --value-mode classification --value-head-type spatial \
    --label-smoothing 0.1 --enable-symmetric-reg \
    --epochs 5 --batch-size 128 --lr 5e-5 --output-dir models/supervised_symmetry
```

## Appendix B — engine slice = current model's placements (instead of yngine)

If you specifically want the *neural model's* emergent wall-clustering in the
corpus (not the engine's), generate self-play placements from the current model
and feed them as the engine slice — but D2-symmetrize them first (keep the style,
drop the lopsided A5-only execution). Not built; only if v1's yngine slice proves
the wrong representation.

## Appendix C — local (Apple Silicon) dry run

E16 + E2 run on MPS (gather, not `index_copy_`). The corpus build + a short from-
scratch pretrain are feasible locally with `--device mps`; self-play wants the box.
