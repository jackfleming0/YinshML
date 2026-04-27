# Remote training investigation — handoff for peer review

**Date:** 2026-04-23 to 2026-04-26 (3-day session)
**Branch:** `clean-slate`
**Driver:** Jack Fleming
**Goal:** train a YINSH model on cloud GPU within ~$100 budget, per `CLOUD_TRAINING_PLAN.md`. Target: ≥65% win rate vs `HeuristicAgent(depth=3)` after a Phase D winner run.

---

## TL;DR

We landed all the Phase A stabilization fixes the plan called for and ran the Phase B–D pipeline end-to-end on a cloud RTX 4090. Once a 260× heuristic perf bug was fixed, throughput became excellent. **But every recipe we ran shows the same iter‑3‑4 regression pattern from `CLAUDE.md`'s warning: anchor win rate peaks at iter 3-4 (~50%), then collapses to 0-15% by iter 15-25 and never recovers.** We tried two opposite recipes (smaller LR + smaller updates AND deeper MCTS + more games + pinned heuristic) and both regressed. The pattern is recipe-independent. We have a working iter-4 checkpoint at ~50% anchor but it's not the deliverable we wanted, and the underlying cause is unidentified.

We've spent **~$10 of cloud compute** so far. Plenty of budget remains; we lack a hypothesis worth spending it on without outside eyes on the training-loop architecture.

---

## Setup

- **Cloud provider**: Vast.ai
- **Hardware**: 1× RTX 4090 (24 GB VRAM) at ~$0.365/hr on-demand
- **Image**: `vastai/pytorch_2.10` (PyTorch 2.10 + CUDA 13)
- **Working dir on cloud**: `/workspace/YinshML`
- **Repo state**: `clean-slate` branch (public on GitHub: `jackfleming0/YinshML`)
- **Storage backup**: none used in the end (R2 was prepared but the runs were short enough to not need spot-instance recovery; on-demand instances were used throughout)

---

## What we did

### Phase A — local stabilization (Mac, then verified on cloud)

Per `CLOUD_TRAINING_PLAN.md` §1. All landed and verified with a 3-iter smoke + kill + resume on Mac.

| Item | Commit | Notes |
|---|---|---|
| 1.1 Replay buffer persistence | `4c22ba5` | Already shipped; added 5-test regression suite. |
| 1.6 CUDA portability | `a3e6522` | Fixed hardcoded `'mps'` device in CoreML export, mirrored `empty_cache` for CUDA, fixed device-selection ordering. |
| 1.4 Per-run manifest | `857457b` | `manifest.json` at launch, `manifest_final.json` at completion. Git SHA, full config, encoder, hardware. |
| 1.3 Anchor evaluation | `4a7db91` | Fixed-opponent eval vs `HeuristicAgent(depth=3)` after every iter, with a per-iter `ANCHOR: iter N, W/T = X%` log line and metrics.json payload. |
| 1.2 Resume Elo persistence | `460f717` | Root cause: path reconstruction stripped `iteration_N/` subdir, triggered `_reset_best_model_state`, zeroed `_iteration_counter`. Fix: multi-candidate path resolution + explicit `set_resume_iteration()` + gated "first model" auto-promote on `candidate_iteration == 0`. 12-test regression suite. |
| 1.5 Cloud sync hook | `3fe5445` | `scripts/sync_run.sh` honoring `SYNC_RUN_DEST` (s3 / gs / rclone / rsync). Never used in production; kept for spot-instance future use. |

### Phase B — cloud smoke (~$1.30, 3.6 hrs)

First cloud run on the 4090. Goal: prove the pipeline runs end-to-end. Found and fixed several CUDA-specific issues:

| Bug | Fix commit | Notes |
|---|---|---|
| Linux `multiprocessing` default = `fork`, can't reinit CUDA in forked child | `b44949e` | Force `spawn` start method in both `self_play.py` and `agents/tournament.py`. |
| Self-play workers were hardcoded to `device('cpu')` | `9796fe7` | Was a defensive Mac+fork+MPS workaround. Now auto-detects CUDA when available. Workers got 100% CPU but 0% GPU until this fix. |
| `requirements.txt` pinned `numpy==1.24.3`, fails to build on Py3.12 | `ed0bc1c` | Cloud images use Py3.12 venvs. Relaxed to `numpy>=1.24.3,<3.0`. |
| Anchor `num_games` / `depth` not configurable from YAML | `6bf1657` | Added `anchor:` block plumbing to `mode_settings`. |

Phase B completed: 2 iters, both checkpoints saved correctly, `manifest_final.json` written, anchor eval fired. **Confirmed everything works on cloud.**

### Phase C — ablation sweep (~$2.55, 7 hrs)

Ran the 5 ablations from plan §4.1 at trimmed budgets (`num_simulations: 48`, `games_per_iteration: 30`, anchor `num_games: 6`, 10 iters per config). The trims were forced by the *first* attempt timing out — see "perf wins" below. Results across 5 configs (anchor win rate %, iters 3–9; mean over those 7 iters):

```
                       3     4     5     6     7     8     9   |  Mean
ablation_baseline     0.0  16.7   0.0   0.0  16.7  33.3  16.7  | 11.9%
buffer100k_epochs2    0.0   0.0  16.7  16.7  16.7  33.3  16.7  | 14.3%
games150              0.0  16.7  33.3  33.3  33.3  16.7   0.0  | 19.0%
curriculum_slow       0.0  16.7   0.0   0.0  33.3  33.3  16.7  | 14.3%
lr_conservative       0.0  66.7   0.0  50.0   0.0  16.7  33.3  | 23.8%   (winner)
```

`lr_conservative` (lr 5e-4, epochs 2, warmup 20) won by mean and peak. `games150` showed the iter-3-4 regression most clearly (peaked at iter 5-7, declined to 0% by iter 9). **All trajectories are noisy** — at 6 anchor games per iter, 95% CI half-width is ~±40 pts. Whether the configs were really differentiable is questionable.

### Phase D — winner scaled up (~$2.55, 7 hrs, partial)

Took `lr_conservative` recipe, scaled knobs back up (`num_simulations: 96`, `games_per_iteration: 50`, `max_buffer_size: 100K`, `anchor.num_games: 20`). 50 iters planned. Killed at iter 26.

**Anchor trajectory at clean 20-game samples (CI ±~22pt):**

```
iter 3:  10/20 = 50.0%   ← peak
iter 4:  10/20 = 50.0%   ← peak
iter 5:   3/20 = 15.0%   ← collapse
iter 15:  0/20 =  0.0%
iter 17:  0/20 =  0.0%
iter 22:  2/20 = 10.0%
iter 25:  3/20 = 15.0%
```

**The iter-3-4 regression `CLAUDE.md` warned about, in clean detail.** Tournament Elo also hovered around 1500 (i.e., no clear winner among iters); promotion gate fell back to "iteration 1 still best" for most of the run.

### Path 2 — radical recipe attempt (~$1.10, 3 hrs, killed)

Hypothesis: maybe the regression is the AlphaZero "bootstrap death spiral" — bad MCTS visits → bad training targets → worse network → even worse visits. Counter with deeper MCTS so search-quality dominates the prior.

| Knob | Phase D winner | Path 2 |
|---|---|---|
| `num_simulations` | 96 | **256** |
| `games_per_iteration` | 50 | **100** |
| `heuristic_weight_anneal` | 0.5 → 0.0 over 10 iters | **pinned at 0.5 forever** |
| `max_buffer_size` | 100K | **300K** |

```
iter 2:  2/20 = 10.0%
iter 3:  0/20 =  0.0%   ← worse than Phase D winner at the same iter!
iter 4:  (running, killed)
```

Killed after iter 3 because anchor was *lower* than Phase D winner at the equivalent point. Counter to expectation.

---

## Perf wins along the way

Two surprising bottlenecks worth flagging:

### 260× heuristic speedup (commit `d74d6e9`)

Profiling (driver: Jack, harness `profile_self_play_worker.py`) showed `detect_forced_sequences()` in `yinsh_ml/heuristics/forced_sequences.py` consumed **99.3%** of `YinshHeuristics.evaluate_position()` wall-clock on a 16-ply hybrid MCTS slice. The function does a recursive ~10×10×10 multi-ply lookahead with `state.copy()` + `get_valid_moves()` at every leaf — essentially mini-MCTS inside MCTS. Outer MCTS already does this lookahead via simulation.

Fix: added `enable_forced_sequence_detection` flag (default `True`), gated the call in `evaluate_position`, set `False` at MCTS-side instantiation sites in `mcts.py` and `self_play.py`'s embedded MCTS class. Standalone `HeuristicAgent` keeps the default.

**Result: 158.76 ms/eval → 0.61 ms/eval (260×).** Self-play iter wall-clock dropped from ~3 hrs to ~1 minute on Mac sanity scale, ~30 min to a few minutes on cloud Phase C scale.

### 100× anchor opponent speedup (commit `ca11f92`)

After the heuristic fix, `run_anchor_eval` was still slow (~7 min/game, ~40 min per model). The anchor opponent uses `HeuristicAgent` which constructs its *own* `YinshHeuristics()` with default args (forced-sequence ON). At depth=1, evaluating ~60 child positions per move at 159ms each = ~9-10 sec per anchor move.

Fix: pass an explicit `YinshHeuristics(enable_forced_sequence_detection=False)` to the anchor's `HeuristicAgent` via the existing `evaluator=` constructor kwarg. Anchor opponent now ~56 ms/move on Mac MPS (likely faster on CUDA). Whole anchor phase went from ~2 hrs to ~2 min on the Phase C runs.

**Caveat**: this changes the anchor's playing strength. It still does its own alpha-beta search via the `HeuristicAgent` wrapper, but no longer does forced-sequence lookahead in the heuristic. Anchor stays consistent across iterations and configs in the sweep, so relative comparisons are valid; absolute win rates skew up slightly.

---

## Investigation findings (no root cause found)

### What's confirmed working

- Per-position value targets are constructed correctly (game outcome flipped to side-to-move perspective in `self_play.py:1567+`).
- Spawn multiprocessing + CUDA contexts work cleanly per worker (verified via `[WORKER N] param.device=cuda:0` diagnostic).
- Manifest writes, replay buffer persistence, resume-from-checkpoint all verified end-to-end.

### What's broken or suspect (not yet root cause)

1. **D2 augmentation silently fails on 3 of 4 transforms** (`yinsh_ml/training/augmentation.py`). Every iter logs:
   ```
   WARNING - Failed to apply transform 1: No valid-move mapping for transform 1
   WARNING - Failed to apply transform 2: No valid-move mapping for transform 2
   WARNING - Failed to apply transform 3: No valid-move mapping for transform 3
   ```
   Result: configured 4× expansion, getting 1× (identity only). Reduces effective training data 4×. Probably a hex-symmetry coordinate bug in `_transform_move()`. **Not a regression cause** but compounds the data-starvation problem.

2. **Pseudo W/B/D logging is wrong** (`supervisor.py:837-839`). It computes `outcome = np.mean(values)` for a game, but `values` is a list of per-position outcomes flipped to side-to-move. For any decisive game the mean is ≈ 0 (half the positions flip sign), so every game looks like a draw. **Just a logging bug**, doesn't affect training. But misleads diagnostics.

3. **EMA decay 0.999 may be too slow for short runs**. With `batch_size: 256` and `buffer ~3000-50000`, we get ~12-25 gradient steps per iter × 2-4 epochs = ~25-100 steps per iter. After 25 iters that's ~1500 steps. EMA at 0.999 means `0.999^1500 ≈ 22% original weights still`. The tournament + anchor both `use_ema_for_eval: true`, so they're evaluating a heavily-smoothed-toward-random-init shadow. **Plausibly explains why early-iter eval looks decent (mostly random noise that occasionally beats heuristic) and late-iter eval looks like the live weights are actively diverging.** Not yet tested.

4. **Value head plateau noted in `CLAUDE.md`**: "value head stuck at `[-0.06, +0.10]` pre-fixes". The "fixes" listed in CLAUDE.md were rule-correctness fixes (April 2026), not value-head fixes. The value head may still struggle to learn from random-init self-play outcomes — those games are essentially random, so outcome targets are too noisy to be informative. Anchor data is consistent with this: discrimination metric in training logs hovers at 0.18-0.22 across all our runs (network is barely confident in its values).

5. **Anchor uses raw network argmax, not MCTS** (`tournament.py::run_anchor_eval`). The candidate plays via `predict() + select_move(temperature=0)`. Tests the raw policy head, not the trained model in its intended use (with MCTS). Doesn't explain regression but means anchor numbers are pessimistic vs how the model would play in deployment.

### Suspected root cause (unconfirmed)

Most likely: **the value head can't learn useful predictions from random self-play outcomes, and the policy head trains on MCTS visits that are themselves dominated by the bad policy prior.** Bootstrap death spiral.

This is consistent with:
- Multi-recipe regression (knob changes don't escape the spiral)
- Discrimination metric stuck at ~0.2 across all configs
- AlphaZero's well-documented dependence on extreme compute or warm-start to escape this
- `CLAUDE.md`'s pre-existing note about value head plateau

But it's a hypothesis. **We have not done the diagnostic experiment that would prove or disprove it.**

---

## Open experiments worth a peer's time

### Cheap, fast (<$2 each)

1. **Disable EMA for eval**: set `use_ema_for_eval: false`, rerun any config 10 iters, see if anchor numbers change shape. If LIVE network performs much differently than EMA, EMA was masking real progress.

2. **Faster EMA decay**: try `ema_decay: 0.95` so the shadow actually tracks. Same hypothesis as (1).

3. **Disable augmentation**: `augmentation.enabled: false`. Eliminates the silently-failing transforms as a confound.

4. **Use MCTS during anchor eval**: currently anchor uses raw policy argmax; the trained model is meant to be used with MCTS. Worth measuring strength under MCTS.

5. **Print live network anchor + EMA anchor side-by-side** to definitively localize whether training is helping.

### Bigger structural experiments (~$5-10 each)

6. **Supervised warm-start**: there is `scripts/run_supervised_pretraining.py` and `analysis_data/` (100K games). Pretrain value+policy on supervised data first, then warm-start self-play from that checkpoint. If self-play still regresses, the regression is genuinely in the self-play loop. If it doesn't regress, we've found the missing ingredient.

7. **Pure-heuristic value targets**: replace the game-outcome value with the heuristic's evaluation at each position. Bootstraps the value head from a known-good signal until self-play results are meaningful.

8. **Smaller network**: 12 ResNet blocks × 256 channels is large for the data we have. Try 6×128 to see if the value head can fit a usable prediction faster.

### Investigation only (no compute)

9. **Audit `_transform_move()` in `augmentation.py`** to find why 3/4 hex transforms reject all moves. If we recover this, training data 4× without spending more compute.

10. **Audit value-target sign flow end-to-end**: `self_play.py:1560+` → buffer → `trainer.py:980+` (the discretization). Even though I didn't find a bug, fresh eyes are valuable.

11. **Trace the policy target**: MCTS visit counts → normalize → store in buffer → train. Specifically: are the visit counts BEFORE or AFTER applying root Dirichlet noise? Training on noised visits is a known footgun.

---

## Reproducing what we have

The Phase D iter-4 checkpoint (50% anchor, our highest-confidence model) lives on the cloud box at:
```
runs_phase_d/<timestamp>/iteration_4/checkpoint_iteration_4.pt
runs_phase_d/<timestamp>/iteration_4/checkpoint_iteration_4_ema.pt
runs_phase_d/<timestamp>/iteration_4/metrics.json
```

To reproduce:
```bash
git clone -b clean-slate https://github.com/jackfleming0/YinshML.git
cd YinshML
pip install -r requirements.txt && pip install -e .
# On a cloud GPU box:
python scripts/run_training.py -c configs/phase_d_winner.yaml --iterations 5
```

Iter 4 should have ~50% anchor on a 4090. Per-iter wall-clock ~15 min. Total: ~75 min.

---

## Spend summary

| Phase | Cost (USD) | Wall-clock | Notes |
|---|---|---|---|
| Cloud smoke (B) | ~$1.33 | 3.6 hrs | Found CUDA + spawn + numpy + worker-device fixes |
| Sanity rerun | ~$0.76 | 2.1 hrs | Verified fixes, saw anchor jump 0% → 50% |
| Phase C sweep | ~$2.55 | 7.0 hrs | 5 configs × 10 iters at trimmed budget |
| Phase D winner | ~$2.55 | 7.0 hrs (killed at 26/50) | Regression confirmed at 20-game anchor |
| Path 2 attempt | ~$1.10 | 3.0 hrs (killed at 4/30) | Deeper MCTS didn't help |
| Stuck Phase C attempt (pre anchor-perf-fix) | ~$1.80 | 5 hrs (killed) | Wasted; informed `d92aa34` perf trim |
| **Total so far** | **~$10** | **~28 hrs** | Of ~$48 deposited |

---

## What we'd recommend the peer look at first

1. **EMA + augmentation experiment** (above #1, #3) — both are cheap and could explain a lot. ~$1.50, 3 hrs.

2. **Audit augmentation `_transform_move`** — silent failure of 3/4 transforms is suspicious and might be a clue to coordinate-system bugs elsewhere.

3. **Supervised warm-start** (above #6) — the highest-leverage experiment, but requires confidence that the supervised pipeline works.

If those all flatline: the iter-3-4 regression really is "you need pretraining or way more compute" and that becomes a $200-500 follow-up project, not a $100 round.

---

## Files / commits worth grepping

- `CLOUD_TRAINING_PLAN.md` — original handoff doc
- `CLAUDE.md` — project conventions and a prescient note about the value-head plateau
- `configs/ablation_*.yaml` — Phase C sweep configs
- `configs/phase_d_winner.yaml` — Phase D config (lr_conservative scaled up)
- `configs/phase_d_path2.yaml` — Path 2 config (deep MCTS attempt)
- `scripts/run_phase_c.sh` — sweep launcher
- `cloud_logs/phase_d_*.log` (on cloud box) — training logs

Key recent commits on `clean-slate`:
```
d74d6e9 perf(heuristic): gate forced-sequence detection for MCTS callers (260× speedup)
ca11f92 perf(anchor): use fast heuristic evaluator for anchor opponent (~100× speedup)
9796fe7 fix(mp): self-play worker picks CUDA when available
b44949e fix(mp): force 'spawn' start method so self-play workers survive on Linux
460f717 fix(training): preserve Elo and iteration counter across --resume
4a7db91 feat(training): absolute evaluation anchor vs HeuristicAgent
857457b feat(training): per-run reproducibility manifest
```
