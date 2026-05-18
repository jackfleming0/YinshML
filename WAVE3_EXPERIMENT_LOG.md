# Wave 3 — Running Experiment Log

Append-as-we-go log of every experiment. One block per experiment: hypothesis → setup → headline → lesson → link. Read top-to-bottom for the trajectory of what we've tried, what worked, what didn't, and why.

**How to use this file**:
- New experiment kicked off → append a `Status: IN FLIGHT` block with hypothesis + setup.
- Experiment finishes → flip status to `DONE`, fill in headline + lesson, link the deeper doc if one exists.
- Don't delete entries even if hypotheses are refuted — the refutation IS the data.

---

## Wave 2 (done)

### Step 1 — Telemetry wiring + `last_root_value` sign-flip fix
- **Started → finished**: 2026-05-13 (one session)
- **Hypothesis**: Three Wave 2 acceptance gates (B1 `mcts/effective_child_visits`, B2 `train/effective_batch_size`, B3 `eval/value_outcome_correlation`) were reading None or the wrong list index across the prior cloud run. With the gates broken, no future cloud run can be interpreted.
- **Setup**: Local; small smoke `configs/wave2_wiring_verify.yaml`.
- **Headline**:
  - B1 wired by threading `metrics_logger` through `play_game_thread`/`play_game_worker`.
  - B2 wired by adding `log_scalar('train/effective_batch_size', …)` in `train_iteration`.
  - B3 wired by per-candidate metric naming (`eval/value_outcome_correlation/<candidate_label>`) so the gate-check reads the right series.
  - **`last_root_value` sign-flip fix at `self_play.py:798/1022`** — root.value() was stored in opposite-of-root POV by legacy backprop convention. `tournament.py` consumed it as if it were current-player POV → systematic negative correlation. Fix: negate at the assignment site.
- **Lesson**: Telemetry can lie. The previous cloud run had -0.07 to -0.31 value_outcome_correlations across iters — looked like "value head learning wrong sign." Was actually a logging bug. Always treat unit-test-passing-but-production-failing as a sign of a missing assertion.
- **Commit**: `3cb047f`. **Deeper doc**: `WAVE2_STEP_1_WIRING_FIXES.md` (plan), inline notes in commit message.

### Step 2 — n=40 anchor validation re-run
- **Started → finished**: 2026-05-14 12:27 → 2026-05-15 08:48 (20.34h cloud)
- **Hypothesis**: With Step 1's fixes live, n=40 anchor evals will tell us whether the Wave 1 training-quality claim ("MCTS at 48 sims > 33% baseline; raw policy near 33%") holds.
- **Setup**: `configs/warm_start_combined_recipe.yaml` at `336dd98` (only change vs prior run: `anchor.num_games: 4 → 40`). Vast.ai 4090.
- **Headline**:
  - All 5 Wave 2 gates produced sensible values. **`eval/value_outcome_correlation` positive across all 5 iters** (+0.061 to +0.198) — sign-flip fix held at n=40 production scale.
  - **Wave 1's training-quality claim does NOT hold.** Best raw policy 30% (below 33% baseline). Best MCTS-48 60% at **iter 0** (seed, before any training).
  - Per-iter MCTS-48 EMA: 60 → 50 → 35 → 32.5 → 47.5. Monotonic decay (with rebound).
- **Lesson**: The supervised seed already plays better with MCTS than 5 iters of self-play training produces. The training pipeline was actively making things worse. Step 3 (revert diagnosis) was superseded — we found the revert mechanism *did* fire every iter (the SUMMARY log said "no reversion" due to a separate logging bug, fixed in `ae63e8c`).
- **Deeper doc**: `WAVE2_STEP_2_POSTMORTEM.md`.

---

## Wave 3 — branches and diagnostics

### Branch A — eval non-EMA candidates against the same anchor
- **Started → finished**: 2026-05-15 (~1 hour after launch — cheap, just inference)
- **Hypothesis (F4)**: EMA drift was eroding seed quality across iters. The trained candidates were close to seed; the EMA was averaging-in worse-and-worse weights.
- **Setup**: Inference-only; pull each iter's `checkpoint_iteration_N.pt` (non-EMA) through `eval_vs_heuristic` at MCTS-48 n=40. Compare to the EMA WRs we already had.
- **Headline**: **F4 refuted.** EMA was *stronger* than non-EMA candidate in every iter (gaps +2.5 to +32.5 points). EMA was the saving grace; the raw trained candidate was the weak one. Pattern is consistent with high-frequency training noise that the EMA filters out.

  | iter | EMA WR | non-EMA WR | gap |
  |---|---:|---:|---:|
  | 0 | 60.0% | 32.5% | +27.5 |
  | 4 | 47.5% | 15.0% | +32.5 |

- **Lesson**: When formulating a hypothesis, test the direction explicitly. F4 had the sign wrong — easy to fall into if you don't run the cheap counterfactual first.

### D1 — Pin the seed baseline
- **Date**: 2026-05-15
- **Hypothesis**: We were triangulating the seed's MCTS-48 WR from §8 (n=60, 63.3%) and prior-run iter 0 EMA (n=40, 60.0%) but never measured the raw seed checkpoint at the same anchor pipeline.
- **Setup**: `eval_vs_heuristic --checkpoint models/supervised_seed/best_supervised.pt --num-games 40 --mcts-simulations 48`.
- **Headline**: **Seed @ MCTS-48 n=40 = 67.5% (27/40).** CI ~[52%, 81%]. Consistent with §8's 63.3% within CI.
- **Lesson**: Pin the baseline before extrapolating. Once D1 landed, the "training damages 67.5% → 32.5%" claim was sharp instead of triangulated.

### D2 — Inspect Step 2's per-epoch training loss timeline
- **Date**: 2026-05-15 (local, free)
- **Hypothesis**: The "high-frequency LR overshoot" story (from chat speculation) implied loss spikes within iter 0's 4 epochs.
- **Setup**: Parse `metrics/iteration_*.json` per-epoch records.
- **Headline**: LR schedule was MUCH lower than the recipe nominally said. `lr: 0.0001` was the *peak* of warmup; actual epoch-1 LR was 1e-5 (10× smaller). Loss DECREASED monotonically across iter 0's epochs (4.46 → 3.96 ploss, 8.39 → 6.80 vloss). val_acc DECREASED (12.2% → 9.8%). Iter 4 trained at LR ≈ 2.5e-6 (near zero) — explains why iter 4's val_acc reverted to iter-0 levels.
- **Lesson**: Read the actual LR schedule, not the recipe headline. Loss decreasing while win-rate decreasing means the loss objective is misaligned with the task — not an LR overshoot story.

### D3 — 1-iter × 1-epoch micro-run
- **Started → finished**: 2026-05-15 20:30 → 23:51 (3.33h cloud, ~$2)
- **Hypothesis**: Is damage immediate (first-epoch) or cumulative (across 4 epochs)? `configs/wave3_d3_micro_1epoch.yaml` runs 1 iter × 1 epoch and measures the candidate.
- **Setup**: 1 iter × 200 games × 1 epoch, warmup_epochs: 1 (so LR hits peak by end of the one epoch).
- **Headline**: **Iter 0 candidate after 1 epoch = 62.5% MCTS-48** (n=40). Within CI of seed's 67.5%. Step 2's 4-epoch iter 0 was 32.5%. **Damage compounds; 1 epoch is fine, 4 epochs ruins.**
- **Lesson**: Per-iter overfitting on the 200-game buffer was the dominant damage source. Branch B's load-bearing diagnostic.

### Branch B — epochs_per_iteration 4 → 1
- **Started → finished**: 2026-05-16 12:01 → 2026-05-17 06:48 (18.78h cloud, ~$10)
- **Hypothesis**: D3 generalizes — running the full 5-iter recipe at 1 epoch per iter avoids the per-iter overfitting. Mean candidate WR should rise toward seed.
- **Setup**: `configs/wave3_branchB_epochs1.yaml`. Single knob change vs Step 2: `trainer.epochs_per_iteration: 4 → 1` plus `trainer.warmup_epochs: 10 → 1` (scaled proportionally).
- **Headline**: **Mean MCTS-48 EMA = 59.5% (Branch B) vs 45.0% (Step 2), +14.5 points absolute.** Every iter improved. 3 promotions vs Step 2's 1.

  | iter | Step 2 EMA WR | Branch B EMA WR | Δ |
  |---|---:|---:|---:|
  | 0 | 60.0% | 60.0% | 0 |
  | 1 | 50.0% | 70.0% | +20 |
  | 2 | 35.0% | 45.0% | +10 |
  | 3 | 32.5% | 60.0% | +27.5 |
  | 4 | 47.5% | 62.5% | +15 |

- **Lesson**: F_overfitting confirmed at the population level. **But iter 1 (70% MCTS-48, the peak) was Wilson-rejected** (17/40 head-to-head, LB=0.285 < 0.55 threshold) → pipeline reverted to iter 0 → strength didn't compound. The gate threw away the run's best model.
- **Deeper doc**: `WAVE3_BRANCH_B_RESULTS.md`.

### Branch B' — promotion_threshold 0.55 → 0.20
- **Started → finished**: 2026-05-17 12:23 → 2026-05-18 07:15 (18.87h cloud, ~$10)
- **Hypothesis**: Iter 1's 70% MCTS-48 was a real strength gain. If the gate had promoted it, iter 2 self-play would have used iter 1's weights, and the gain might have compounded. Lower threshold to LB > 0.20 (just above iter 1's LB=0.285) promotes the marginal-but-strong candidates.
- **Setup**: `configs/wave3_branchB_prime_low_wilson.yaml`. Single knob change vs Branch B: `arena.promotion_threshold: 0.55 → 0.20`.
- **Headline**: **5/5 promotions** (every iter became new best). **Best model = iter 4 at 67.5% MCTS-48 — equal to the seed**. Mean WR 57.0% (B was 59.5%; -2.5 within CI noise). Branch B's iter 1 at 70% **did not reproduce in B'** — B' iter 1 was 52.5%. That confirmed B's 70% was a high-variance outlier, not a reproducible state.

  | iter | B' anchor MCTS-48 | promoted? |
  |---|---:|---|
  | 0 | 62.5% | ✅ |
  | 1 | 52.5% | ✅ (would have been ⏪ in B) |
  | 2 | 47.5% | ✅ |
  | 3 | 55.0% | ✅ |
  | 4 | 67.5% | ✅ (final best) |

- **Lesson**: The "Wilson gate threw away the run's best model" framing from Branch B was real but the underlying gain was variance, not signal. **The pipeline is now stable but not constructive — it matches seed quality, doesn't exceed it.** Branch C's question shifts from "stop the damage" to "make it gain ground past seed."
- **Deeper doc**: `WAVE3_BRANCH_B_PRIME_RESULTS.md` (to be written).

---

### Gap 2 measurement — 300-move cap hit rate (Eric Jang AutoGo audit)
- **Date**: 2026-05-18 (free, ~10 min local Python)
- **Hypothesis**: The audit's Gap 2 claim — `self_play.py:1751-1754` applies `value = clip(score_diff/3, -1, 1)` to games that hit `max_game_moves=300` as if they were terminal. If a meaningful fraction of self-play games hit the cap, the value head has been learning partial-credit non-terminal labels. The audit's threshold for "matters" was ≥15%.
- **Setup**: Parsed Step 2 and Branch B training logs (`runs_*/training.log` and `branchB.log`), counted self-play games (`Worker-N - INFO - Game M finished in K moves.`) per run, computed % at cap.
- **Headline**: **0/1000 games hit the 300-move cap in both runs.** Mean game length ~87 moves (Step 2: 87.9, Branch B: 86.0). Max observed: 119 moves (Step 2) / 119 (Branch B). YINSH self-play games end naturally well before the cap.
- **Lesson**: Gap 2 is **NOT** the bottleneck for our recipe. The 300-move cap is so generous for YINSH that it never fires. One audit hypothesis ruled out without spending compute. Audit's framing was generic to AlphaZero-style games (where Go can hit caps); YINSH has shorter games. Cheap measurement, decisive result.

### V2a — yngine self-play vs HA self-play "fingerprint" comparison
- **Date**: 2026-05-18 (~30 min on cloud, ~$0.30)
- **Hypothesis**: Before investing 1-2 days in a yngine ↔ Python protocol bridge for direct head-to-head (V2b), measure each engine's self-play "fingerprint" — game length distribution, outcome balance, decisive-vs-draw ratio. If the fingerprints diverge wildly, one engine is much weaker; if they overlap, V2b is needed for the strength decision.
- **Setup**: Cloned + built `temhelk/yngine` on the cloud box (cmake-min downgraded 3.30→3.20; otherwise clean build). Wrote a small C++ self-play driver (`yngine_selfplay.cpp`) that runs N games of yngine-MCTS-vs-itself. Ran 30 games at MCTS-10K. In parallel, ran 30 games of `scripts/generate_heuristic_games.py` at depth=3 (16 completed before the worker pool exited early — enough for fingerprint).
- **Headline**:

  | Engine | n | Mean moves | W/B/Draws | Wall/game |
  |---|---:|---:|---:|---:|
  | yngine MCTS-10K | 30 | 69.6 | 53% / 47% / 0 | 34s |
  | HA depth=3 | 16 | 82.3 | 56% / 44% / 0 | ~180s |

  Both engines are decisive (zero draws). Both produce roughly 50/50 W/B balance. yngine games are ~15% shorter. **yngine is ~5× faster per game on the same hardware.**
- **Lesson**: The fingerprints overlap — no engine looks broken or much weaker. The 5× throughput gap matters for volume scaling: yngine @ 1K sims would generate 100K games in ~10h on 8 cores (yngine self-play, no neural net needed); HA @ d=3 would take ~3 days wall time for the same volume. Direct strength comparison (V2b) is still needed to decide which engine is the better corpus generator, but V2a confirms yngine is a real opponent worth a bridge.
- **Date**: 2026-05-18 (free, ~30 min)
- **Hypothesis**: Volume corpus generation (audit's core thesis: 10K-100K cheap value-head pretraining games) only helps if the teacher is reasonably strong. If our HeuristicAgent has tactical gaps, volume scaling on its games bakes those gaps into the value head. yngine (C++ MCTS engine for YINSH) is a candidate external strength reference.
- **Setup**: Read `temhelk/yngine/yngine/{mcts.cpp,mcts.hpp,board_state.cpp}` (~1700 lines C++).
- **Headline**: yngine is a **serious engine, not hobby code**. Key features:
  - Pure-UCT MCTS with random rollouts. No neural net, no hand-crafted heuristic.
  - **128-bit bitboards** for state representation (efficient).
  - **Parallel lock-free MCTS** with atomic ops + pool allocator + cross-move tree reuse — sophisticated implementation.
  - **Exploration constant: 0.5** (vs the standard sqrt(2) ≈ 1.414). The commented-out `std::numbers::sqrt2_v<float>` suggests it was tuned down deliberately, but 0.5 is unusually low — exploits more aggressively.
  - **Random rollout to natural game-end** every playout (no move cap, no biasing). Standard for vanilla MCTS but produces noisy game-end signals; quality scales with sim count.
- **Verdict**: V2 head-to-head match is worth building. yngine at ~10K sims should be a meaningful opponent against our HeuristicAgent at depth=3. C++ speed advantage means high sim counts are cheap.
- **Lesson**: Reading 1700 lines of well-written C++ is a fast way to assess engine quality vs running a full benchmark — saves the dev cost of V2 if the code looks like junk. yngine is the opposite case: code looks like junk's opposite, so V2 is justified.

### V2a-qual — turn-by-turn review of yngine and HA games (verifying that quant fingerprints aren't hiding obvious blunders)
- **Date**: 2026-05-18 (~1h)
- **Hypothesis**: V2a's quantitative fingerprints (game length, outcome balance, draws) were similar across engines. Before committing to yngine for a 100K-game volume corpus, look at actual games turn-by-turn — does either engine miss obvious 5-row captures, do dumb ring shuffles, fail to defend 4-row threats?
- **Setup**:
  - **yngine**: extended `yngine_selfplay_dump.cpp` to print each move + the full BoardState ASCII after every move. Ran 2 games at MCTS-10K.
  - **HA**: used the merged-from-GRV viz module on the 16 HA d=3 games from V2a. Rendered key board states + dumped the move sequence for ha_000000 (87 turns, W wins 3-2).
- **Headline findings**:
  - **yngine plays multi-move plans**. Game 0 move-21 was a long ring jump (8,6 → 8,2) that completed a 5-marker diagonal (4,10)–(5,9)–(6,8)–(7,7)–(8,6) built up over 5 prior moves. Then captured the row (move 22) and removed a ring (move 23). That's sophisticated tactical play — not random rollout noise.
  - **HA plays snake-style positional buildup**. ha_000000's white played `B1→B2→...→B7→C8→D9` (sequential single-step ring moves) over turns 10-20 — methodically building markers along a column. Eventually completed a `B5,C6,D7,E8,F9` diagonal capture at turn 85.
  - **Neither engine** missed visible 5-row captures, did obvious random shuffles, or placed rings in pathological spots. Openings spread rings across the board (E3/G11/B1/J11/B4 for HA; (0,9)/(4,10)/(6,9)/(5,9)/(1,10) for yngine).
  - **Style difference**: yngine completes captures faster (game 0 = 59 moves). HA games run longer (87 turns) — more cautious / more positional. This is a style difference, not a quality difference.
- **Lesson**: The 5× yngine speed advantage holds even after looking at actual games. Neither engine is broken or obviously weak. **For the volume corpus, yngine wins on throughput; quality looks at least competitive in this sample.**
- **Mitigation for "yngine might have subtle blind spots not visible in 2 games"**: after generating the volume corpus and pretraining the value head on it, validate the resulting checkpoint against the standard HA(d=1) anchor pipeline at n=40 BEFORE committing to a full neural self-play training run on top.

### Volume corpus generation — `IN FLIGHT`
- **Started**: 2026-05-18 15:08 UTC
- **Hypothesis**: Per Eric Jang AutoGo audit, the value head needs 10-100K cheap games for proper grounding. V2a confirmed yngine plays competently and is ~5× faster per game than our HA. Pursue Option B: yngine for the volume corpus.
- **Setup**: Cloud (192-core EPYC, 1TB RAM). 64 parallel `yngine_volume` workers, each playing ~3125 games at MCTS-1K → 200K total games. Each game ~70 moves. Output: text format `G/P/M/R/X/S/E` per shard, ~150-200KB per shard.
- **Speed observation**: Initial 184-worker run was only 2.4 games/sec (heavy thread-spawn contention — yngine spawns 2 threads per `MCTS::search()` call). Dropped to 64 workers → 19.3 games/sec, ETA ~3 hours. The user picked this over patching yngine source to avoid risk of corrupting a known-good corpus.
- **Translator**: `scripts/yngine_corpus_to_npz.py` — already written and smoke-tested on 3 yngine games (201 positions, 0 replay failures). yngine (x, y) → our (col=chr('A'+x), row=11-y). All 6 yngine hex directions land on our 3 hex axes.
- **Headline**: pending. Cron `56d92b17` checks every 2h at :41.

### Branch C — MCTS-200 self-play targets — `IN FLIGHT`
- **Started**: 2026-05-18 11:15 UTC
- **Hypothesis**: Self-play target quality is the plateau bottleneck. §8 showed iter 0 EMA was 81.7% at MCTS-400 vs 63.3% at MCTS-48 vs 16.7% raw — the deeper-search teacher is much stronger than the training-time teacher. Training against MCTS-200 targets (4× deeper than MCTS-48) should produce candidates that exceed seed.
- **Setup**: `configs/wave3_branchC_mcts200.yaml`. Same as B' plus three coupled knobs: `num_simulations 48→200`, `late_simulations 32→100`, `games_per_iteration 200→100` (the last to keep wall time under ~30h since per-game cost is ~4× higher).
- **Decision doc**: `WAVE3_BRANCH_C_DECISION.md`.
- **Success criteria** (any one suffices): best iter MCTS-48 anchor ≥ 75%, OR raw policy ≥ 30%, OR mean ≥ 65% with peak ≥ 70%.
- **Failure criteria**: best iter MCTS-48 ≤ 67.5% AND mean ≤ 57% — target depth wasn't the bottleneck; fall back to C3 (head decoupling).
- **Headline**: pending. Cron `dd59c71a` checks every 2h at :37.

---

## Heuristics added after Branch B'

- **L7** — Single-iter "outliers" with strong-looking results should be assumed to be variance until they reproduce. The +20-point iter 1 in Branch B looked like a real lift; the matched-recipe Branch B' iter 1 was -17.5 points off it. Two samples at n=40 anchor are nowhere near enough to call a single-iter result.
- **L8** — Stability and progress are different gates. Branch B' achieved **5/5 promotions** (pipeline stable, no Wilson-thrashing) and **finishes at seed quality** (no degradation across iters). Neither outcome includes "model gets meaningfully better than seed." Recipe tuning closed the destruction problem; it didn't open the improvement problem.

---

## Heuristics built from this trajectory

These are general lessons that should outlast the specific experiments:

- **L1** — Pin baselines before extrapolating. We spent a chunk of chat reasoning about a 30-point seed-to-candidate drop before D1 measured the actual seed at the right pipeline.
- **L2** — Run the cheap counterfactual before trusting the headline hypothesis. Branch A (a $0 inference job on existing checkpoints) refuted F4. If we'd run a full cloud experiment to test F4, we'd have burned $10 to learn the same thing.
- **L3** — Read the actual numbers, not the recipe headlines. D2 found the LR schedule was 10× smaller than the recipe said.
- **L4** — Internal-metric pass and external-metric fail is a real category of bug. Step 1 (sign-flip in `last_root_value`) and Branch B (Wilson gate vs anchor WR disagreement) are both this pattern.
- **L5** — Log lines lie when there are multiple decision branches above the SUMMARY block. `supervisor.py:2028` claimed "no reversion" even when the revert path ran. Always trace the actual code path, not the summary.
- **L6** — A wide CI at n=40 is not "no signal." If the *direction* is consistent across 5 paired comparisons, the population-level effect is real even when individual-iter CIs overlap.

---

## File-of-files

- `WAVE2_NEXT_STEPS.md` — index, kept current at major phase boundaries.
- `WAVE2_STEP_2_POSTMORTEM.md` — Step 2 deep dive (5 findings + bug + EMA-drift hypothesis).
- `WAVE3_BRANCHES.md` — original branch scoping doc.
- `WAVE3_BRANCH_B_RESULTS.md` — Branch B deep dive.
- **`WAVE3_EXPERIMENT_LOG.md`** ← this file. Always append, never reorder.
- `configs/wave3_*.yaml` — the actual recipe files for each experiment.
- `runs_*` — per-experiment artifact dirs (some on laptop, some on cloud).
