# Warm-start phase research log

**Started:** 2026-04-26 (after `REMOTE_TRAINING_INVESTIGATION.md` peer review handoff)
**Branch:** `clean-slate`
**Driver:** Jack Fleming
**Goal:** spend <$200 on cloud compute to produce a YINSH model that plays competitively with an intermediate human (proxy: ≥65% win rate vs `HeuristicAgent(depth=3)` via MCTS at deployment search budget).

This file captures the chain of decisions, evidence, and recipe choices for the warm-start training phase so that context isn't lost between cloud instance restarts, terminal disconnects, or model swaps.

---

## 1. Context coming in

After 3 days of cloud experimentation (`REMOTE_TRAINING_INVESTIGATION.md`, ~$10 spent), every recipe regressed at iter 3-4 from a peak of ~50% anchor down to 0-15%. Two opposite recipes (smaller LR + smaller updates *and* deeper MCTS + more games + pinned heuristic) both regressed the same way. Pattern was recipe-independent and the underlying cause was unidentified.

Peer review proposed a 4-part plan:

1. **Joint update path.** `trainer.py:604` — the value loss was only stepping the value head; the shared trunk only learned from policy loss. Fix this first.
2. **`--init-checkpoint` flag.** `run_training.py:346` — current `--resume` couples weight loading with iteration counter and run dir. A clean warm-start path was missing.
3. **MCTS-vs-anchor as primary metric.** `tournament.py:683`, `tournament.py:375` — anchor + tournament both used raw policy argmax. Trained model is meant to be used *with* MCTS; eval should reflect that.
4. **Rebuild supervised seed against current encoder.** Move encoding went 8390→7433 in April 2026 (the foundational rule fixes). Old supervised checkpoints are dead.

**Peer's prediction:** if we land 1-3 and warm-start from a fresh supervised seed, we should see one of three patterns:
- **MCTS improves but raw stays weak** → original problem was eval mismatch.
- **Both improve through iter 6+** → recipe is sound, keep scaling.
- **Both still collapse** → flip the existing search-consistency probe; if that fails, the problem is offline-signal quality.

---

## 2. Implementation (2026-04-26)

All four landed as a single coherent changeset. Verified end-to-end with a Mac smoke run before any cloud spend.

### 2a. Joint optimizer fix — `yinsh_ml/training/trainer.py`

**Bug confirmed in code reading:** `policy_optimizer = Adam(trunk + policy_head)`, `value_optimizer = SGD(value_head only)`. The training step did:
1. `policy_loss.backward()` → populates trunk + policy_head .grad
2. `policy_optimizer.step()` → updates trunk + policy_head from policy gradient
3. (second forward pass)
4. `value_loss.backward()` → populates trunk .grad with value-loss gradient
5. `value_optimizer.step()` → only steps value_head

**Net effect:** trunk's value-loss gradient was computed (wasteful) but never applied. The trunk only ever saw the policy signal. That fully explains the value-head plateau and the discrimination metric stuck at ~0.2 across all prior runs — value head sat on a trunk with no incentive to encode value-relevant features.

**Fix:** single forward, `total_loss = policy_loss + value_loss`, single backward, both optimizers step. No double-counting because policy_head and value_head are disjoint subgraphs and the trunk only appears in the policy_optimizer's param set. Per-head clip max_norms preserved (1.0 trunk+policy, 0.5 value).

**Verification:** unit probe confirmed (a) trunk weight delta after one step is 0.77 in L2 (was 0 for value-loss contribution before), (b) value-only loss produces trunk `.grad` norm ~630 — autograd graph routes value-head gradient through the trunk as expected. All 68 trainer-adjacent tests pass.

### 2b. `--init-checkpoint` flag — `scripts/run_training.py`

New CLI arg, mutually exclusive with `--resume`. Creates a fresh timestamped run dir, loads only model weights (skips optimizer/scheduler state), starts at iteration 0. Internal helper `_extract_model_state` shared with the resume path.

### 2c. MCTS-vs-anchor — `yinsh_ml/utils/tournament.py`, `yinsh_ml/training/supervisor.py`

`run_anchor_eval` and `run_anchor_eval_batch` accept `use_mcts: bool` and `mcts_simulations: int`. When `use_mcts=True`, the candidate plays via pure-neural MCTS with subtree reuse on, root noise off, deterministic temp.

Supervisor's anchor block now runs both raw + MCTS when `anchor.mcts_enabled: true`, prints `ANCHOR[raw]` immediately after raw call returns (so we don't wait through MCTS to see the diagnostic), then `ANCHOR[mcts/N]` after. Both result payloads persist in `metrics.json` under `anchor_eval` and `anchor_eval_mcts` keys; the primary `anchor_win_rate` metric is the MCTS variant when enabled, raw otherwise.

**Important post-launch fix:** initial implementation used `mcts.search()` (single-leaf evaluation, ~10× slower than `mcts.search_batch()`). Switched to `search_batch(batch_size=32)` to match self-play's MCTS path.

### 2d. Configs

- `configs/smoke_warmstart.yaml` — 1-iter, 20 games, no EMA, anchor enabled at iter 0.
- `configs/diag_warmstart_3iter.yaml` — 3-iter diagnostic with full peer recipe.
- `configs/phase_d_warmstart.yaml` — the 12-iter winner-attempt; 100 games/iter, heuristic curriculum 1.0→0.0 over 10 iters, no EMA, anchor at depth=3 (later dropped to 2; see §4b).

---

## 3. Validation — Mac smoke

Three smoke attempts on Mac MPS:

| | Result |
|---|---|
| **v1** | Hung in MCTS anchor for 90+ min; killed. Diagnosed as single-leaf MCTS perf bug. |
| **v2** | Same hang pattern; killed. Diagnosed as supervisor printing both ANCHOR lines after both calls return; fixed to print raw immediately. |
| **v3** | Clean run: **ANCHOR[raw] = 50% (2/4), ANCHOR[mcts/32] = 100% (4/4)**, total wall-clock 35 min. |

**Smoke verdict:** the MCTS-vs-raw split the peer predicted shows up *immediately* even with only one epoch of supervised pretraining as the warm-start. Joint-update fix + warm-start path verified end-to-end.

Mac MPS forced a config tweak: anchor depth=3 took >90s/game on positions produced by the warm-started network, so smoke was set to `depth: 2, num_games: 4, mcts_simulations: 32`. Cloud configs unaffected (4090 was assumed to handle depth=3 fine).

---

## 4. Cloud Phase 1 run — 2026-04-27

### 4a. Setup

- Vast.ai 4090, image `vastai/pytorch_2.10`.
- Supervised pretraining on cloud: 10 epochs in 559s, final val PAcc=0.291 / VAcc=0.892. Strong warm-start.
- Launched `phase_d_warmstart.yaml` with `--init-checkpoint models/supervised_seed/best_supervised.pt`. 238/238 tensors loaded, 0 mismatches.

### 4b. First failure — anchor depth=3 hang

After iter 1 completed cleanly (30 min, expected), tournament eval landed:

```
checkpoint_iteration_0 vs checkpoint_iteration_1: 188/200 (p=0.940, CI=[0.898,0.965])
```

Iter_0 (warm-start prior) crushed iter_1 (post-self-play) **94-6** in 200 games. The supervised prior was being damaged by one iteration of self-play — a sharp regression.

Then anchor eval started and **hung for 2h 50m** with 0% GPU utilization. CPU was 99% in `list_contains`/`set_contains` Python frames — the heuristic alpha-beta search at depth=3 was blowing up on positions produced by the regressed iter_1 network.

Hypothesis: regressed network produces erratic-but-confident moves → unusual mid-game positions with high branching → alpha-beta tree explodes (poor move ordering kills the pruning). Random-init networks in prior runs didn't trigger this; well-trained networks won't either; but a freshly-regressed warm-start network is the worst case.

**Tactical decision:** drop `anchor.depth: 3 → 2`, `num_games: 40 → 10`, `mcts_simulations: 96 → 32`. Tournament still does the heavy lifting on relative strength; anchor just needs to not hang.

### 4c. Second launch — clean signal through iter 5

Relaunched 19:59 UTC with the depth=2 anchor patch. Got through 5 iterations before the Vast instance was preempted/destroyed at ~02:39 UTC.

**Anchor:**

| Iter | raw | mcts/32 |
|---|---|---|
| 0 (warm-start) | 100% | 100% |
| 1 | **50%** | 100% |
| 2 | 100% | 100% |
| 3 | **50%** | 100% |
| 4 | 100% | 100% |
| 5 | (lost) | (lost) |

**Tournament Glicko at iter 5:**

| Model | Glicko | W-L-D | Notable |
|---|---|---|---|
| iter_3 | 1584.7 | 275-125-0 (68.8%) | best so far |
| iter_2 | 1458.2 | 163-237-0 (40.8%) | |
| iter_4 | 1457.1 | 162-238-0 (40.5%) | tied with iter_2 |

Per-pair CI:
- iter_3 vs iter_4: 133/200 = 66.5% (iter_3 better)
- iter_2 vs iter_4: 105/200 = 52.5% (statistically tied)
- iter_0 vs iter_2: 174/200 = 87.0% (iter_0 still beats iter_2)
- iter_0 vs iter_1: 376/400 = 94.0% (warm-start still strongest)

### 4d. What we learned

1. **Joint-update fix is working.** No catastrophic divergence — every iter that "wobbles" recovers within the next iter. The death-spiral from prior runs is gone.
2. **Peer's eval-mismatch hypothesis is supported.** MCTS pegged at 100% throughout; raw oscillates 50% / 100% / 50% / 100%. Under deployment-style play (MCTS), the network is consistently strong even when the policy head wobbles.
3. **No monotonic improvement yet.** Best model is iter_3 (Glicko 1585), iter_4 is back near iter_2's level (~1457). Could be plateauing at the supervised prior's level.
4. **Anchor at depth=2 is saturated.** 100% / 100% / 100% across iters means we can't differentiate models. Need depth=3 for usable signal — and depth=3 only blew up on the iter_1 position distribution; once past that, depth=3 should be fine.
5. **Per-iter cost ballooning.** Iter 1 = 29 min, iter 4 = 107 min. Tournament's sliding window plus anchor's multi-checkpoint loop scale O(N) with iteration count. A 30+ iter run at this rate would be 50+ hours.
6. **Vast preemption risk is real.** Lost the run at iter 5 with no warning; only the terminal scrollback survived. R2 sync was prepared but not used; should be wired in for any long run.

---

## 5. Strategic plan (from 2026-04-28 onward)

The 5-iter run gave evidence the recipe is fundamentally sound but raised three preconditions that must be met before scaling to a 30-50 iter "winner" run:

1. **Per-iter cost stays flat as iter count grows.**
2. **Anchor differentiates models** (depth=2 saturated; need depth=3 with reasonable budget).
3. **Trend confirmed monotonic** (currently wobble + recovery, not improvement).

Plan:

| Phase | Goal | Recipe | Cost (est) | Cumulative |
|---|---|---|---|---|
| 1. **De-risk run** | Verify iter 6-12 trends; confirm depth=3 anchor works on later iters; cost-flat per iter | `configs/phase_d_warmstart_derisk.yaml` | ~$4 | ~$20 |
| 2. **Decision gate** | If iter 8-10 trend ↑ → continue. If oscillating → investigate | n/a | $0 | ~$20 |
| 3. **Long run** | 30-40 iters, full self-play scale, harder anchor | `configs/phase_d_winner_long.yaml` (TBD post-Phase 1) | ~$15-20 | ~$40 |
| 4. **Final eval** | 200-game eval vs depth=3 at deployment MCTS budget; confirm ≥65% target | `configs/eval_intermediate.yaml` | ~$2 | ~$42 |

**Buffer:** ~$160 of $200 budget for tweaks, larger network experiments, or a second long run if needed.

### De-risk recipe (Phase 1) — diff from `phase_d_warmstart.yaml`

```yaml
# Self-play unchanged — we want to test what we'll ultimately scale.
self_play:
  games_per_iteration: 100

# Tournament: cut scope so per-iter cost stays flat.
arena:
  games_per_match: 50              # was 100; Wilson CI still ±10pt
  tournament_sliding_window: 2     # was 3; pairs go from 3 to 1

# Anchor: re-enable depth=3, tighter scope on candidate only.
anchor:
  enabled: true
  num_games: 20                    # was 10; depth=3 at n=20 has ±22pt Wilson CI
  depth: 3                         # was 2; iter≥2 won't blow up alpha-beta
  mcts_enabled: true
  mcts_simulations: 64             # was 32; matches deployment budget
  skip_first_n_iterations: 1

num_iterations: 12
```

**Expected per-iter cost: ~25-30 min flat.** Total: ~5-6 hours, ~$2-3.

### Phase 2 decision gate

After Phase 1 finishes, look at `ANCHOR[mcts/64]` across iter 1-12:

- **Monotonic improvement** from ~50% (iter 1) toward 70-80%+ (iter 8-12) → green light, scale to Phase 3.
- **Continued oscillation 50/100/50/100…** → don't scale yet. Investigate candidates: heuristic curriculum schedule too aggressive, EMA, batch size, or the regression hypothesis.
- **Plateau at one level (e.g. all 100% or all 50%)** → anchor is still saturated even at depth=3, or model has plateaued at warm-start strength. Need stronger benchmark.

### Long-run recipe (Phase 3) — sketch, finalize after Phase 2

- 30-40 iterations.
- Self-play scale held at 100 games/iter (unchanged from de-risk).
- `tournament_sliding_window: 2` (cost control) + `games_per_match: 100` (sharper CI for promotion gate).
- Anchor: depth=3, 40 games, MCTS sims 96.
- R2 sync **wired in** (`SYNC_RUN_DEST` env var) so per-iter checkpoints persist if instance is preempted.
- Strongly consider: reduce `epochs_per_iteration` from 4 if late iters show overfitting; or bump to 6 if tournament shows too-slow improvement.

### Phase 4 — final eval

200-game match vs `HeuristicAgent(depth=3)` at deployment MCTS budget (the "hard" preset: 400 sims, c_puct=1.5). Strongest checkpoint from Phase 3 wins if ≥65% (target) or ≥55% (stretch — still beats the heuristic but not "intermediate" yet).

---

## 6. Open questions / parking lot

- **Why does iter 4 win 65% vs iter 3 in tournament but iter_3 has higher Glicko?** Glicko aggregates pair outcomes and uncertainty. iter_3 also beat iter_2 strongly while iter_4 only tied with iter_2 — asymmetric pair results land iter_3 above iter_4. Probably correct behavior; flag if pattern repeats.
- **CoreML export error during iter 4** (`Expected more than 1 value per channel when training, got input size torch.Size([1, 512])`). Probably a BatchNorm-in-eval-mode issue with batch=1 export. Non-fatal (training continued), but should fix for deployment.
- **Heuristic curriculum** is set to 1.0 → 0.0 over 10 iters. Is the falloff correlated with the iter 1, 3 wobbles? Worth checking after Phase 1.
- **Replay buffer growth.** Iter 5's buffer was 64,380 / 100,000. By iter 12 it'll be at cap. Behavior at cap is FIFO eviction — should be fine but worth confirming.
- **Why does raw oscillate 50%/100%?** Suggests training is bouncing between two local minima of policy distribution. Could be batch effect, LR schedule, or temperature scheduling artifact. Phase 1 with depth=3 anchor will tell us if it's a real two-state thing or just noise at depth=2.

---

## 7. Reproducibility

**Code:** branch `clean-slate` at the commit landing the joint-update fix + `--init-checkpoint` + MCTS-vs-anchor. Verify with `python scripts/run_training.py --help | grep init-checkpoint`.

**Supervised seed:** regenerable in ~10 min on a 4090:
```bash
mkdir -p expert_games/validated models/supervised_seed
# (rsync expert_games/validated/all_games.json from local Mac)
python scripts/run_supervised_pretraining.py \
  --games-dir expert_games/validated --epochs 10 --batch-size 256 \
  --output-dir models/supervised_seed --device cuda
```

**Phase 1 launch:**
```bash
nohup python scripts/run_training.py \
  -c configs/phase_d_warmstart_derisk.yaml \
  --init-checkpoint models/supervised_seed/best_supervised.pt \
  > cloud_logs/derisk_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Lost data:** Phase 1 first attempt iter 1-5 log lines are in `cloud_logs/lost_run/iter1-5_summary.txt` (terminal scrollback paste). Run dir + checkpoints are gone (instance preempted).

---

## 8. Spend tracker

| Phase | Cost | Cumulative | Notes |
|---|---|---|---|
| Pre-warm-start cloud experiments | ~$10 | $10 | Per `REMOTE_TRAINING_INVESTIGATION.md` |
| Mac smoke v1-v3 | $0 | $10 | Mac compute |
| Cloud supervised seed (10 epochs) | ~$0.10 | $10.10 | 9 min on 4090 |
| Cloud Phase 1 first attempt (iter 1, hung) | ~$1.50 | $11.60 | Killed |
| Cloud Phase 1 second attempt (5 iters, preempted) | ~$2.50 | $14.10 | Lost run dir |
| **Phase 1 de-risk (planned)** | ~$3-4 | ~$18 | Next |
| **Phase 3 long run (planned)** | ~$15-20 | ~$38 | Gated on Phase 2 verdict |
| **Phase 4 eval (planned)** | ~$2 | ~$40 | After Phase 3 |
| **Reserve** | — | $200 cap | $160 buffer |
