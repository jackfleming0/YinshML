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

## 5b. De-risk run attempt 1 — depth=3 anchor hangs again (2026-04-28)

Launched on a fresh Vast 4090 ~13:18 UTC. Iter 1 self-play + training: 21 min ✓. Iter 1 tournament: 20 min ✓ (96-4 vs warm-start, same regression as before).

**Anchor at depth=3 hung again** — same pathology as 2026-04-27 (§4b): GPU 0% util, CPU 66% in alpha-beta. 62 min of zero log progress.

**Root cause** (clearer now): the supervisor's anchor block evaluates `iter_N + iter_(N-1) + iter_(N-2)` regardless of `tournament_sliding_window`. So during *iter_1's* anchor phase, it loads iter_1 (the regressed network) and has it play the heuristic at depth=3. iter_1's pathological positions blow up alpha-beta. depth=3 cannot survive iter_1's position distribution, period.

**Decision:** drop `anchor.depth: 3 → 2` and `mcts_simulations: 64 → 32` for this run. Tournament + Glicko give monotonic-improvement signal regardless; anchor was the saturation tiebreaker, and saturating at 100% across iters is *better* than hanging.

**Three viable fixes for future depth=3 anchor (any of these unlocks the long run with usable depth=3 signal):**
1. **Code change**: modify supervisor to anchor *only* the candidate (skip the `prev_iter` loop). One-line change in `supervisor.py:1106-1118`. Cleanest.
2. **`anchor.skip_first_n_iterations: 4`**: with `current_iteration - 1, current_iteration - 2` lookback and `skip=4`, first anchor fires at iter 4 with batch `[iter_4, iter_3, iter_2]` — iter_1 is out of the batch. Coarse but config-only.
3. **Two-stage recipe**: depth=2 anchor for iters 1-3 (covers the wobble window), depth=3 from iter 4 onward. Needs config-level transitions, more involved.

**For long run (Phase 3) plan to use option 1.** Documented in §5 under "Long-run recipe."

---

## 5c. Decision: anchor is dropped from training loop (2026-04-28)

After two depth=3 hangs (§4b, §5b) and the depth=2 saturation seen in the iter-1-5 run, the cost/value of anchor-during-training is upside-down. Decision: **drop anchor from per-iter training, replace with a one-shot end-of-run eval.**

**Rationale.** Anchor was meant to give:
1. Absolute strength vs a fixed baseline.
2. Cross-run comparability.
3. The MCTS-vs-raw diagnostic from peer's plan.

Of these, #3 was the load-bearing reason. We already validated the MCTS-vs-raw split in the iter-1-5 run (raw oscillates 50%/100%, MCTS pegged at 100%) — peer's hypothesis confirmed; no further per-iter anchor evidence needed. #1 and #2 can be served by a single end-of-run eval against `HeuristicAgent(depth=3)` at the deployment MCTS budget — ~30 min, one-shot, decoupled from training.

**What replaces it:**
- **During training:** tournament round-robin alone. Per-pair Wilson CI on 50 games is ±10pt — tight enough to detect iteration-over-iteration trends. Glicko aggregates pair outcomes into a single rating per checkpoint.
- **End of training:** `scripts/eval_vs_heuristic.py` (TBD — write before Phase 4) plays the strongest-by-Glicko checkpoint against `HeuristicAgent(depth=3)` for 200 games at the "hard" deployment preset (400 MCTS sims). Reports win rate + Wilson CI. Single-pass, ~30 min, ~$0.20.

**Per-iter cost win:** ~10-30 min saved per iter. Long run (30 iters) saves 5-15 hours.

**Phase 3 long-run config will set `anchor.enabled: false`.** Tournament is the load-bearing per-iter metric.

---

## 5d. The big finding: gating doesn't roll back, recipe damages warm-start (2026-04-29)

After de-risk run completed (12 iters, 4.36h, ~$2), head-to-head between iter_0, iter_3, iter_5, iter_9 (`scripts/eval_head_to_head.py`):

```
iter_0 vs iter_3: 40-0  ★★★
iter_0 vs iter_5: 40-0  ★★★
iter_0 vs iter_9: 40-0  ★★★
iter_3 vs iter_5: 20-20 (tied)
iter_3 vs iter_9:  0-40 ★★★
iter_5 vs iter_9:  0-40 ★★★
```

**The supervised warm-start (iter_0) crushes every post-self-play checkpoint.** iter_9 IS the best post-self-play model — gradual improvement from iter_3 is real — but the *whole post-warm-start regime* is in a strictly weaker basin.

**Root cause** (the deeper architectural issue surfaced by Jack's gating question): YinshML supervisor's gating loop is incomplete vs. canonical AlphaZero.

In `yinsh_ml/training/supervisor.py`:
- `self.network` is shared between `self_play` and `trainer`. Training overwrites it each iter.
- Self-play always uses the latest-trained network, regardless of whether it cleared the promotion gate.
- `best_model_state` is a tag for inference only — it doesn't feed back into self-play data generation.

Canonical AlphaZero gating requires a fourth step that's missing:
1. Train candidate.
2. Match candidate vs best_model.
3. If candidate > best_model: promote, use candidate for next self-play.
4. **If candidate ≤ best_model: discard candidate weights, keep using best_model for self-play.** ← MISSING

Without step 4, once iter_1's training destroys the supervised prior, every subsequent self-play game is generated by a progressively-weaker network. iter_0's strength is never used as a data generator after iter_1. Self-play cannot rescue it back — there's no path for the warm-start-strength data to re-enter the buffer.

**Why the iter-0-vs-iter-1 sliding-window result was misleading.** The tournament does evaluate iter_0 vs iter_1 (during iter_1's tournament phase). iter_1 fails to promote. But that promotion failure has *no effect on training* — the trainer just keeps the iter_1 weights and self-plays with them in iter_2.

### Implications for warm-start training

The current recipe is fundamentally mismatched to a warm-start scenario. Running more iters of it produces more iter_9-class results (still 40-0 below iter_0). The two real fixes are:

1. **Implement rollback gating** (proper AZ). When promotion fails, restore the network weights from `best_model_state.pt` before the next self-play iteration. ~1-day code change in supervisor.py, plus tests. Right answer; non-trivial.
2. **External anchor via heuristic_weight pinned**. Set `heuristic_weight_start = heuristic_weight_end = 0.3`, never decay. The heuristic provides a non-network anchor that limits how far the network's self-play data can drift. Cheap to test (1 config change); speculative whether it actually preserves warm-start strength.

### Decision: eval iter_0 first

Before more compute spend: run `scripts/eval_vs_heuristic.py` against iter_0 (the warm-start) at deployment MCTS budget. If iter_0 + 400-sim MCTS already beats `HeuristicAgent(depth=3)` at ≥65%, **the supervised model alone is intermediate-level and we're done.** No long run needed.

If iter_0 is below the bar, two paths:
- Build option 1 (rollback gating) before any more long runs. Proper fix.
- Try option 2 (pinned heuristic) overnight as a cheap speculative test.

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

---

## 9. Phase D conclusion + Phase E plan (2026-04-30)

### 9a. Phase D verdict: gating revert works structurally; recipe plateaus sub-intermediate

After §5d's diagnosis, the gating-revert flag landed in PR #9 (merged to `main` 2026-04-29) and the 12-iter `phase_d_warmstart_derisk_revert.yaml` run completed cleanly.

**Mechanism evidence:** revert fired correctly on 9/9 failed gates (iters 1, 4, 5, 6, 7, 8, 9, 10, 11). Each rejected iteration reloaded best-model weights and reset optimizer momentum before the next self-play round, exactly as designed. No more compounding regression across iterations — the structural fix is correct.

**Strength evidence:** see `MODEL_PLAY_OBSERVATIONS.md` "Update 2." iter_3 (the recipe's peak) vs `HeuristicAgent(depth=3)` at deployment-realistic config (400 MCTS sims, 30s/move time-limit, 30 games):
- 6 wins / 24 losses / 0 draws = **20% win rate, CI95=[0.095, 0.373]**.
- Verdict from the script: `FAILS`.
- Below the 65% intermediate bar by a wide margin; below 50% even-match line; possibly weaker than the supervised seed itself (50% at 6 games, wide CI).

**Recipe plateau evidence:** within-run head-to-head shows iter_3 has the highest aggregate score (0.750) but a non-transitive cycle exists with iters 6 and 9. The post-iter_3 plateau is real — 8 attempts, 0 promotions, no monotonic improvement.

**Conclusion:** the gating revert solves the *compounding-regression* problem from §5d but does not, on its own, get us past the supervised seed's skill ceiling. iter_3 is at *roughly* iter_0's level; possibly slightly weaker. The recipe's ceiling on this seed is sub-intermediate.

### 9b. What we know now that we didn't before

- The §5d hypothesis ("missing rollback step is the bug") was correct *and necessary* but not *sufficient* — fixing it lets the recipe stabilize, but the local optimum it stabilizes to isn't intermediate-level.
- The recipe's iteration cost is ~20-25 min/iter at our scale. Plateau detection (8 reverts in a row) is now a fast, reliable signal that further iterations of the same recipe won't help.
- Eval costs are growing: depth=3 with time-limit is ~14 min/game, ~7h for a 30-game eval. Need to budget for this in any future experiment.
- The deferred Action B (growing replay window) wasn't needed to make the revert work — but might still help once we have a recipe that's actually pushing past the seed.

### 9c. Phase E options (ranked by EV against current data)

The path forward isn't "more iterations of the same recipe." Options, ordered by my best guess at expected value per dollar:

**Important context — supervised seed data quality is poor (commit `f2d899a`, 2026-04-30 morning).** Analysis via `scripts/analyze_and_filter_expert_data.py` found 41% of supervised training data is bot/anonymous (Dumbot 1576, guest 1099, WeakBot 306, SmartBot 166, BestBot 46). Filtering to humans_only retains 1312 games / 82,837 positions (35% of input). Likely explains why iter_0 plays "chain-shuffler" style — it's literally learned from Dumbot. **This re-orders the Phase E ranking below: Option 5 (stronger seed) is now the highest-EV change, ahead of Option 1 (bitboards), because the seed itself looks like the bottleneck.** Bitboards make all subsequent experiments cheaper, but a better seed makes them *productive*.

**Option 0 — Diagnose first (~$2, ~7h).** Rerun iter_0 (the supervised seed) at 30 games vs depth=3. The 6-game smoke result was 50%, but with CI=[0.188, 0.812]. We need a clean baseline to know whether iter_3 is *as bad as the seed* or *worse than the seed*. The two cases imply different next moves. Cheapest, most informative single experiment available.

```bash
python scripts/eval_vs_heuristic.py \
    --checkpoint models/supervised_seed/best_supervised.pt \
    --num-games 30 --depth 3 --mcts-simulations 400 \
    --time-limit-per-move 30 --device cuda \
    --label iter_0_seed --output-json eval_iter0_d3_full.json \
  2>&1 | tee cloud_logs/eval_iter0_d3_full_$(date +%Y%m%d_%H%M%S).log
```

**Option 1 — Bitboard port (action H from the alphazero-general comparison; engineering project, ~1-2 weeks).** Replace the Python game engine with a C++ extension via pybind11 using yngine's `__uint128_t` bitboards + precomputed `TABLE_RAYS[121][6]`. Expected 10-100× speedup on move generation, which is the current self-play bottleneck. Once this lands, *every other option becomes cheaper to test* — 10× more games per dollar means 10× more experiments. **Recommendation: do this first.** See `~/.claude/plans/steady-enchanting-nebula.md` action H for design notes.

**Option 2 — More games per iteration (~2-4× cost per iter).** Currently 100 games/iter. Doubling to 200 or 400 might break the plateau if the recipe is signal-starved. Cheap to A/B once bitboards land. Without bitboards, this 2× the run cost.

**Option 3 — Larger network.** Current network might be capacity-limited at the supervised-seed level. Bigger ResNet trunk + more attention. Risky on Mac (longer per-iter time); cheap on GPU. See `TODO_frontier.md` §2 for the broader Transformer direction.

**Option 4 — Tune new MCTS knobs.** `root_policy_temp` (currently 1.0) at 1.1–1.2, `fast_simulations` for cheaper exploration. These are the new flags from PR #9 — never been A/B tested. Low cost, low expected upside on its own, but a quick win if it works.

**Option 5 — Stronger supervised seed (PROMOTED to top priority 2026-04-30 given data-quality finding).** Two sub-options:
  - **5a. Filter and retrain.** Run `scripts/analyze_and_filter_expert_data.py` to produce a humans_only dataset, retrain seed at the same epoch count, re-evaluate vs depth=3 with full N=30. ~10 min retrain on 4090 + ~7h eval = ~$2 + half a day. Highest expected value.
  - **5b. Filter, retrain larger, retrain longer.** If 5a moves the needle, scale up: more epochs, larger network, full filtered-data sweep. ~$5-10 + ~half a day.

**Option 6 — Implement Action B (growing replay window).** Replace fixed-position FIFO with iteration-based window. Cleaner training-data dynamics. Modest expected impact alone, but synergistic with options 2-3.

**Option 7 — Frontier shift.** `TODO_frontier.md` §1 (MuZero), §5 (search-consistency loss). The "do now" probes (§4 IQL, §5 SC, §8 SAE) might tell us if this is necessary. We haven't run any of them yet.

### 9d. Recommended sequencing (revised 2026-04-30 PM, after data-quality finding)

1. **Option 5a (retrain seed on humans_only filtered data)** — ~10 min retrain + ~7h eval = ~$2, half a day. **Highest priority.** The supervised seed analysis found 41% of training data is bot games; filtering should produce a meaningfully different (and presumably stronger) seed. Eval at full N=30 vs depth=3 to compare to iter_3's 20% and the original seed's 50%-at-N=6 / TBD-at-N=30.
2. **Option 0 (rerun iter_0 / original seed at N=30)** — can run in parallel with Option 5a since they're independent checkpoints. Establishes the canonical "is the dirty-data seed actually intermediate?" baseline.
3. **Option 1 (bitboard port)** — engineering project. Worth starting in parallel with the above evals since it's compute-independent. Once it lands, all subsequent recipe experiments become 10× cheaper.
4. **Decision gate after 5a + 0:**
   - If filtered-data seed clears ≥50% vs depth=3 (CI lower bound > 0.4): seed quality WAS the bottleneck. Run a fresh `phase_d_warmstart_derisk_revert.yaml` with the new seed, expect different recipe behavior.
   - If filtered-data seed is still ~20-30%: seed quality is one factor but not the load-bearing one. Consider Options 3 (larger network) or 7 (frontier shift).
   - If filtered-data seed is markedly *worse* than dirty-data seed: surprising; investigate (maybe the bots played reasonable enough moves that filtering them removed coverage).
5. **After bitboards land:** Options 2 (more games), 4 (MCTS knobs), 6 (growing window) become cheap parallel A/Bs.

### 9e. What this means for the budget

Phase D total spend: ~$15 actual (vs ~$40 planned through Phase 3+4). The early plateau detection saved us a 30-iter long run that wouldn't have helped. Phase E planning starts with $185 of the $200 budget remaining. Bitboard port is engineering effort, not cloud spend; recipe experiments after bitboards become 10× more efficient.

---
