# Cloud Training Plan — Handoff Doc

**Owner:** this is a handoff document for a new agent to execute a stabilize-then-scale training plan for YinshML. The user has committed ~$100 of cloud compute budget, with the goal of producing a model **competitive with good (non-expert) human YINSH players**.

**Scope of budget:** ~$100 total cloud spend. Not superhuman. Not multi-week. A reproducible, stable, documented training pipeline that produces a middle-to-strong player, on which future investment can compound.

**Status as of handoff:** rule-correctness fixes and training-polish features shipped (see `git log`). Training runs on Mac have validated the engine is functionally correct but the training recipe is unstable (loss improves, playing strength regresses by iter 3-4). Crashes every 5-7 iterations on MPS. Resume loses state. Need to fix these *before* spending on cloud.

---

## 0. Quick orientation

Read before doing anything else:

- `CLAUDE.md` — project overview, architecture, commands. Authoritative.
- `RESEARCH_LOG.md` — decisions and prior-run observations. Especially the note about value head stuck at `[-0.06, +0.10]` pre-fixes.
- `YINSH_RULES.md` — engine rules. Post-April-2026 correctness fixes documented in CLAUDE.md's "Foundational Rule Fixes" section.
- `configs/training.yaml` — current baseline config. Reduced from more ambitious settings after hitting Mac-perf limits (see inline comments).
- `logs/training_*.log` — prior run logs; the most recent gives pace + loss trajectory.

Do not assume the in-conversation context exists. Everything you need is in the repo.

---

## 1. Current known-bad state (to fix in Phase A)

These are already-identified issues that will bite on any longer run. Fix them before cloud launch.

### 1.1 Replay buffer persistence

**Status:** Partially fixed at time of handoff (see uncommitted changes in `yinsh_ml/training/trainer.py` and `yinsh_ml/training/supervisor.py`). Verify, finish, test, commit.

**What's fixed:** trainer now loads `replay_buffer.pkl` or `.pkl.gz` (fallback); supervisor saves buffer after each iteration's checkpoint.

**What to verify:**
- `grep -n "save_buffer\|load_buffer\|replay_buffer" yinsh_ml/training/supervisor.py yinsh_ml/training/trainer.py`
- Round-trip test — kill training mid-run, restart, confirm `[Replay Buffer] Loaded from ...` appears and buffer size matches what was saved.

**Acceptance criteria:**
- After a crash at iter N, resume picks up with the buffer populated at iter N-1's size (not empty).
- Training time for iter N+1 post-resume roughly matches pre-crash iters at similar buffer fill.

### 1.2 Best-model / Elo state loss on resume

**Status:** Broken. When training was resumed via `--resume`, the supervisor reset `_iteration_counter` to 0 and the next candidate was auto-promoted as "first model" — wiping the prior Elo 1546.7 baseline.

**Root cause (suspected):** `runs/<dir>/best_model_state.json` gets rewritten during promotion logic on the resumed run before the persisted state is properly respected. See `supervisor.py::_load_best_model_state` (line ~1690) and the promotion flow around `supervisor.py:1081`.

**Task:**
1. Trace the resume code path in `scripts/run_training.py::_load_resume_checkpoint` (line ~134) into `TrainingSupervisor.__init__` (line ~339). Figure out why `_iteration_counter` ends up at 0.
2. On `--resume`, the `_iteration_counter` must be set to the iteration AFTER the resumed checkpoint. E.g., resuming from `iteration_5/checkpoint_iteration_5.pt` should set counter to 6.
3. `tournament_history.json` must also survive the resume. Check `yinsh_ml/utils/tournament.py` persistence code.
4. Add a regression test: manually write a `best_model_state.json` with Elo=1600, iter=3, then construct a supervisor on that dir and assert `best_model_elo == 1600`.

**Acceptance criteria:**
- Kill training after iter 3 promotes. `--resume`. Iter 4 runs, and its tournament compares against iter 3's checkpoint (not against itself as "first model").
- `best_model_state.json` does not get clobbered on launch.

### 1.3 Absolute evaluation anchor

**Status:** Missing. The current tournament is relative — iter N vs iter N-1 and iter N-2. There's no fixed opponent, so we can't tell if absolute playing strength is improving over long horizons.

**Task:**
1. In `yinsh_ml/utils/tournament.py`, add a fixed baseline: the pure `HeuristicAgent` at depth 3 (or whatever budget is comparable to the training MCTS sim count).
2. Every iteration, after the round-robin tournament, run N=40 games of `candidate_checkpoint vs HeuristicAgent(depth=3)` with deterministic seed.
3. Log `anchor_win_rate` per iteration to `metrics.json` and stdout: `ANCHOR: iter N, {won}/{total} = {rate:.1%}`.
4. Also log `anchor_win_rate` for all 3 tournament models (current candidate, prev best, prev-prev), not just the candidate.

**Acceptance criteria:**
- Every iteration's log has a clear `ANCHOR: ...` line.
- If you plot `anchor_win_rate` over iterations, the trend is interpretable without cross-referencing Elo tables.
- Baseline doesn't change across iterations (same heuristic weights, same depth, same seed).

### 1.4 Per-run manifest

**Status:** Missing. Run dirs have configs but no single pinned summary.

**Task:**
Write `runs/<timestamp>/manifest.json` at launch with:
```json
{
  "git_sha": "...",
  "git_branch": "...",
  "git_dirty": false,
  "config": {...full training.yaml...},
  "encoder": "basic|enhanced",
  "total_moves": 7433,
  "device": "cuda|mps|cpu",
  "hardware": "NVIDIA RTX 4090|Apple M4|...",
  "start_time_iso": "...",
  "cloud_instance_id": "..."
}
```
Also write `runs/<timestamp>/manifest_final.json` at successful completion with aggregated results.

**Acceptance criteria:** `cat manifest.json` on any run dir shows everything needed to reproduce that experiment, including git SHA. No ambiguity when comparing two runs.

### 1.5 Cloud-safe log/artifact sync

**Status:** Missing. If a cloud instance dies mid-run, we lose everything since the last rsync.

**Task:**
1. Add `scripts/sync_run.sh` that rsyncs `runs/<timestamp>/` to a configurable remote (S3 bucket, GCS bucket, or a DigitalOcean Spaces / R2 bucket). Bucket URL configurable via env var.
2. Call it from the supervisor at every iteration end, right after `_save_best_model_state()`.
3. Exclude: `replay_buffer.pkl.gz` if size > 500 MB (it's huge; prioritize checkpoints and metrics).
4. Fallback: if sync fails, log warning and continue. Never crash on sync failure.

**Acceptance criteria:** Kill the cloud instance. Download the bucket contents. Resume locally from that download. Training picks up at iter N correctly.

### 1.6 Verify CUDA portability

**Status:** Unknown. Code was developed on MPS. May have device-string assumptions.

**Task:**
1. `grep -rn "\"mps\"\|'mps'\|mps:0" yinsh_ml/ scripts/` — find any hardcoded MPS references.
2. Check `yinsh_ml/network/wrapper.py::__init__` device handling. Confirm `device='auto'` picks CUDA when available.
3. Check `yinsh_ml/memory/tensor_pool.py` and `yinsh_ml/memory/zero_copy.py` — device-specific code paths.
4. Trial-launch the training on a cloud CUDA box for 1-2 iterations (cost: ~$2). Confirm no crashes. This is the "smoke dry-run".

**Acceptance criteria:**
- `device: auto` → CUDA → 2 clean iterations with checkpoints written.
- Loss trajectory in first 2 iters on CUDA looks sane (policy loss decreasing 7→4-ish, same as Mac baseline).

---

## 2. Phase A — Stabilize on Mac (1-2 days)

Complete all 1.x items above. Do **not** skip to cloud without finishing these. Each one has caused or will cause problems that waste cloud time.

**Order of implementation:**
1. 1.1 (buffer) — already in progress, just verify
2. 1.3 (anchor eval) — most informative; touch this next
3. 1.4 (manifest) — smallest, ~30 min
4. 1.2 (Elo persistence) — harder, debug session required
5. 1.6 (CUDA port check) — grep + trial
6. 1.5 (remote sync) — can be stubbed until cloud phase

**Verification run on Mac:** with all of these landed, run 3 iterations locally. Kill after iter 2. Resume. Iter 3 must pick up cleanly with correct Elo, buffer, and iteration counter. Ship this as the local-portable baseline.

**Commit cadence:** one commit per 1.x item. Don't land a giant merge.

---

## 3. Phase B — Cloud smoke (1 evening, ~$5-10)

Goal: prove the code runs end-to-end on a cloud CUDA box before burning real budget.

### 3.1 Provider choice

**Recommended:** Vast.ai 4090 at ~$0.40/hr. Good cost/perf, flexible. Alternatives:
- **Lambda Labs A10** ($0.60-1/hr): more reliable, less setup friction.
- **RunPod 4090** ($0.40-0.50/hr): similar to Vast, sometimes more stock.
- **Paperspace Gradient** ($0.50-1/hr): bundled Jupyter environment if preferred.

Do NOT use A100 ($2-3/hr) — overkill for this model size. A single 4090 or A10 has enough VRAM (model is ~130 MB; batches don't exceed a few GB).

### 3.2 Instance setup

```bash
# On cloud box (Ubuntu 22.04 + CUDA 12.x assumed)
git clone <your repo URL>
cd YinshML
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Verify CUDA available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.3 Data transfer

The heuristic weights file is non-negotiable:
```bash
# From local
rsync -avz --progress \
  analysis_output/heuristic_evaluator_model.pkl \
  user@cloud-box:/root/YinshML/analysis_output/
```

Do NOT copy `runs/` or `large_scale_selfplay_data/parquet_data/` unless you explicitly need to resume a specific run. Fresh start is cleaner.

### 3.4 Smoke run

```bash
# On cloud, from YinshML/
source venv/bin/activate
python scripts/run_training.py -c configs/training.yaml --iterations 2
```

Monitor for 1-2 hours. Success means:
- 2 iterations complete cleanly.
- Final iter-2 policy loss in the 4-5 range (matching Mac baseline).
- Anchor win rate is logged.
- Checkpoints written to `runs/<timestamp>/iteration_*`.

If this works, tear down the instance. Move to Phase C. Cost so far: ~$1-5.

### 3.5 Pitfalls to watch for on first cloud run

- **Multiprocessing / MPS paths** — `yinsh_ml/memory/tensor_pool.py` has Apple-Silicon branches. Verify they no-op cleanly on CUDA.
- **Autocast** — bf16 autocast is enabled in config. Some CUDA drivers prefer fp16. If loss NaN's immediately, try `enable_autocast: false` as a quick bisect.
- **Num workers** — Mac cap is 3. On a real cloud box with 16+ cores, bump `num_workers: 6` or higher. Measure before committing — self-play throughput doesn't scale linearly past 8 workers for this model.

---

## 4. Phase C — A/B sweep ($20-30, ~40 hours)

Goal: identify which training recipe change fixes the iter-3 regression we saw on Mac. Four hypotheses, each gets 10 iterations on a 4090.

**Budget per config:** 10 iterations × ~1.2 hrs/iter on 4090 ≈ 12 hours. Four configs = 48 hours. Can run sequentially on one box, or 2× parallel on two boxes.

### 4.1 Hypotheses and configs

Each config is a clone of `configs/training.yaml` with one targeted change. Save as `configs/ablation_<name>.yaml`.

**Config A — `ab_buffer100k_epochs2`** (reduce overfit pressure):
```yaml
trainer:
  epochs_per_iteration: 2    # was 4
  max_buffer_size: 100000    # was 50000
```

**Config B — `ab_games150`** (more data per iter):
```yaml
self_play:
  games_per_iteration: 150   # was 50
# Everything else unchanged from baseline.
```

**Config C — `ab_curriculum_slow`** (heuristic anchor longer):
```yaml
self_play:
  heuristic_weight_start: 0.5
  heuristic_weight_end: 0.0
  heuristic_weight_anneal_iterations: 20   # was 10
```

**Config D — `ab_lr_conservative`** (reduce fit-then-drift):
```yaml
trainer:
  lr: 0.0005               # was 0.001
  warmup_epochs: 20        # was 10
  epochs_per_iteration: 2  # also reduce epochs
```

**Baseline:** the current `configs/training.yaml` unchanged. Run it too, so you have a 5-way comparison. Call it config `ab_baseline`.

### 4.2 Evaluation

For each config:
- Run 10 iterations.
- Collect `anchor_win_rate` per iter (from 1.3 above).
- Collect `promotion_count` (iterations that passed the gate).
- Collect final `policy_loss` and `value_loss` at iter 10.
- Collect `discrimination` (mean_abs_value) at iter 10.

Rank configs by this composite:
1. `anchor_win_rate` at iter 10 (primary — absolute playing strength).
2. `promotion_count` / 10 (stability — fraction of iterations that improved).
3. `discrimination` (secondary — value head expressiveness).

Ignore loss directly — we've learned loss can improve while strength regresses.

### 4.3 Launch pattern

Sequential on one box:
```bash
for cfg in ab_baseline ab_buffer100k_epochs2 ab_games150 ab_curriculum_slow ab_lr_conservative; do
  python scripts/run_training.py -c configs/${cfg}.yaml --iterations 10 --save-dir runs_ablation/${cfg}
  bash scripts/sync_run.sh runs_ablation/${cfg}
done
```

Run the sync between configs — if the instance dies, you don't lose the prior ablation.

### 4.4 Decision criteria

The **winner** is the config with the best composite ranking above. If two are close, prefer the simpler change (fewer knobs moved).

If **all five configs regress** (none reach >55% anchor win rate by iter 10), don't proceed to Phase D. Instead:
- Increase iteration count to 15-20 (maybe regression bottoms out and recovers).
- Check for a code-level bug in training, not a recipe issue.
- Consult prior RESEARCH_LOG entries — there may be a known pattern.

If **the baseline wins**, that's still informative: means the recipe isn't the issue and the iter-3 regression we saw on Mac was MPS/Mac-specific or small-sample noise. Proceed to Phase D with baseline.

### 4.5 Output artifact

Write `ablation_report.md` summarizing:
- Final anchor win rate per config.
- Promotion count per config.
- Loss trajectory plots (use matplotlib; save PNGs).
- Chosen winner + rationale.

This document is the deliverable of Phase C. Commit it to the repo.

---

## 5. Phase D — Winner run ($30-50, ~50-80 hours)

Goal: produce the actual best-model checkpoint for the $100 budget. Scale up the winning config.

### 5.1 Config

Start from the Phase C winner's config. Make these adjustments:
```yaml
# num_iterations: 40-50   # full run
# games_per_iteration: 150   # if not already
# num_workers: 6-8           # if the cloud box has the cores
```

Leave everything else from the winner untouched. The point is to scale the winning recipe, not re-experiment.

### 5.2 Compute

- **4090 ($0.40/hr) for 50-80 hours = $20-32.**
- Alternative: Lambda A10 ($0.60-1/hr) for reliability, same time → $30-80.

Pick based on whether you want lower cost (4090, less reliable spot pricing) or lower risk (A10, dedicated).

### 5.3 Monitoring

- Start the run with `scripts/sync_run.sh` wired to an S3/R2/Spaces bucket.
- From your local machine, `rsync -avz cloud:runs_winner/ runs_cloud/` every hour to a local mirror. The bucket is primary, local mirror is belt-and-suspenders.
- Check `tail -f` logs periodically. Flag any:
  - `error`, `traceback`, `exception` (filter out `coremltools`, `scikit-learn`, `Torch` version warnings).
  - Iterations where `anchor_win_rate` regresses more than 10 points iter-to-iter.
  - Buffer size going backward (indicates persistence bug resurgence).

### 5.4 Stopping conditions

- **Expected:** run completes 50 iterations, anchor win rate plateaus. Take best checkpoint by anchor win rate (not necessarily last).
- **Good stopping signal:** anchor win rate has not improved in 10 consecutive iterations and is >70%. Stop early, save money.
- **Bad stopping signal:** anchor win rate declining over 10 consecutive iterations. Pause and investigate — may have recipe issue Phase C didn't catch.

### 5.5 Deliverable

- `best_checkpoint.pt` + `best_checkpoint_ema.pt` copied off the cloud box.
- Final `manifest_final.json` with all metrics.
- Bakeoff result of winner vs pre-winner-iter baseline (use `scripts/run_bakeoff.py`).
- Brief writeup in `TRAINING_RESULT.md`: config, metrics, limitations, suggestions for next run.

---

## 6. Success criteria for the $100 budget

At completion:

1. **Code is stable.** Crash-free for 20+ consecutive iterations on cloud.
2. **State persists.** Resumes don't lose buffer/Elo. Proven by a deliberate mid-run kill + resume test.
3. **Absolute signal.** Every iter has an anchor win rate. Trajectory is plottable and interpretable.
4. **Model exists.** A trained checkpoint that achieves ≥65% anchor win rate against `HeuristicAgent(depth=3)`. This is the "competitive with good players" threshold for this budget.
5. **Reproducible.** `manifest.json` pins the git SHA. A new operator can re-train with the same numbers.
6. **Documented.** `ablation_report.md` and `TRAINING_RESULT.md` both exist.

### Non-goals for this budget

- **Superhuman play.** Would require 10× the compute. Out of scope.
- **Distributed self-play architecture.** Nice to have, not needed at this scale.
- **Bitboard / Cython optimization.** Worth doing later; not a prerequisite.
- **Beating top BGA ranked players.** Possible but not assured; it's the stretch outcome if everything goes right.

---

## 7. Gotchas and lessons learned

From the ~20 hours of Mac training we did before handoff, things to not re-learn:

- **`logger.debug()` in hot paths kills perf on Python.** We stripped 297 of these from game-logic files. Don't reintroduce. If you need trace logging, use `if logger.isEnabledFor(logging.DEBUG):` gating or a separate trace flag.
- **`is_terminal()` is O(1) now, intentionally.** Don't add the `get_valid_moves()` check back onto the hot path. Stalemate detection lives in `is_stalemate()` and is only called at game boundaries or `get_winner()`.
- **Heuristic_weight=1.0 disables batched MCTS speedup.** Hybrid at 0.5 lets the NN calls batch. Baseline settled on 0.5 → 0.0 curriculum.
- **`total_moves = 7433`** (post-REMOVE_MARKERS encoding fix). Older checkpoints with `8390` are incompatible — `NetworkWrapper.load_model` will fail loudly. Don't try to warm-start from pre-April-2026 runs.
- **MPS tensor device leaks** caused the EMA crash. On CUDA, same class of bug could exist — keep an eye on cross-device ops after CoreML export (if we still do that).
- **Batch-sync idle workers.** Between batches of 10 games, 2 of 3 workers sit idle waiting for the slowest. ~5-10 min/batch throughput leak. Not urgent but a future optimization.
- **Loss-vs-Elo divergence is real.** Don't use `policy_loss` or `value_loss` as success signals. The anchor win rate from 1.3 is the trusted metric.

---

## 8. File pointers for the new agent

Key files this plan touches or references, with why:

| Path | Why |
| --- | --- |
| `configs/training.yaml` | Baseline training config. Copy/modify for ablations. |
| `scripts/run_training.py` | Entry point. Resume logic is here. |
| `yinsh_ml/training/supervisor.py` | Orchestrator. Best-model persistence, buffer save, tournament. |
| `yinsh_ml/training/trainer.py` | Network training loop. `GameExperience` replay buffer. |
| `yinsh_ml/training/self_play.py` | Self-play data generation + MCTS config parsing. |
| `yinsh_ml/search/mcts.py` | MCTS implementation. |
| `yinsh_ml/utils/tournament.py` | Round-robin tournament. Add anchor eval here. |
| `yinsh_ml/agents/heuristic_agent.py` | Fixed opponent for anchor eval. |
| `yinsh_ml/training/ema.py` | EMA shadow. Device-alignment fix in commit `59a867a`. |
| `logs/training_*.log` | Prior run logs. Source of truth for pace and crash patterns. |
| `runs/<timestamp>/` | Per-run artifacts: checkpoints, config snapshot, tournament history. |
| `CLAUDE.md` | Project-wide instructions and conventions. READ FIRST. |
| `RESEARCH_LOG.md` | Historical findings. Includes the discrimination plateau at 0.104. |
| `YINSH_RULES.md` | Canonical rules. Reflects the April 2026 correctness fixes. |

---

## 9. Open questions the new agent may hit

Don't block on these — proceed with best judgment, document the call:

- **Tournament anchor opponent strength.** I recommended `HeuristicAgent(depth=3)`. Depth 5 would be a stronger ceiling but slower. Pick one and stick with it across all configs; don't change mid-ablation.
- **Buffer size in Phase C Config A.** 100K is a guess. If memory allows, try 150K. The question is "enough diversity without going stale" — more buffer is generally safer.
- **Autocast on CUDA.** Default config has it on. If it destabilizes on CUDA (NaN losses), turn off. Don't let it block forward progress.
- **Whether to A/B on CUDA-with-bigger-games.** E.g., `num_simulations: 200, games_per_iteration: 200`. This would be "the real thing" but might exceed the ablation budget. If curious and budget allows, run it as a 5th ablation.
- **Whether the `total_moves=7433` layout is the best possible.** The encoder fix is correct but the 123-line REMOVE_MARKERS table isn't necessarily optimal for learning. Out of scope for this budget. Note for future.

---

## 10. Final checklist before handing the cloud box back

- [ ] All artifacts synced to bucket AND local mirror.
- [ ] `ablation_report.md` committed to repo.
- [ ] `TRAINING_RESULT.md` committed to repo.
- [ ] Best checkpoint + EMA sibling copied locally.
- [ ] Cloud instance terminated (no lingering charges).
- [ ] Any recipe adjustments that proved necessary folded into `configs/training.yaml` with clear inline comments.
- [ ] A follow-up "what would $200-500 more buy us" note added to `NEXT_UP.md`.

Good luck.
