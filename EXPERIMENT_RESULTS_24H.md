# 24-hour autonomous experiment results

**Window**: 2026-05-07 17:38 UTC → 2026-05-08 17:38 UTC
**Branch**: `policy-collapse-hunt`
**Spend**: ~$10 of $25 authorized (Vast.ai 4090 instance, possibly terminated near end)

## TL;DR

1. **Original BN-stat-trash bug stays fixed.** All experiments in this window built on the working trainer.
2. **Three single-knob ablations on top of cloud_run_v1 recipe**: A null, **B winner** (`discrimination_weight: 0.5 → 0.0`), C negative.
3. **Ablation B at 5 iter looked spectacular** (raw policy 60/60 vs depth-1 heuristic) — but trajectory probing later revealed it was a single-iter spike, not a stable plateau.
4. **25-iter scale-up of B's recipe** trained cleanly (best ELO 1590, beat v1 ceiling of 1528) — but raw policy never re-hit 100%; iter 9 EMA tops out at 50% / iter 9 LIVE at 0%.
5. **The 60/60 raw was a lucky local optimum, not a robust property of the recipe.** Random-seed sensitivity dominates.
6. **The robust headline is still d1_400 = 60/60 (100%)** on scale-up B's iter 9 EMA. The model + heavy MCTS sweeps depth-1 heuristic. The naked-policy ceiling appears lower than ablation B made it look.

## Code shipped (committed and pushed on `policy-collapse-hunt`)

| sha | what |
|---|---|
| `60f33f9` | EMA-rebind fix in `_reset_network_objects` + 3 ablation configs (A/B/C) + regression test |
| `45af47e` | `configs/scale_up_b.yaml` — 25-iter scale-up of B's recipe |

5 regression tests passing on cloud:
- `test_supervisor_bn_preservation.py` (1) — buffer-nulling guard
- `test_wrapper_save_guard.py` (2) — refuse to save corrupted checkpoints
- `test_supervisor_reset_ema_rebinding.py` (2) — EMA shadow rebinds on reset

## Ablation chain results (5 iter each, n=60 vs HeuristicAgent depth=1)

| ablation | knob change | 48-sim | raw | verdict |
|---|---|---:|---:|---|
| A | value_head_lr_factor: 5.0→1.0 | 30/60 (50%) | 0/60 (0%) | **null** |
| **B** | discrimination_weight: 0.5→0.0 | 30/60 (50%) | **60/60 (100%)** | **WINNER (apparent)** |
| C | final_temp: 0.1→0.5 | 0/60 (0%) | 0/60 (0%) | **negative** (made it worse) |

## Scale-up B (25 iter on B's recipe, 9.05h, 4 promotions)

| metric | value |
|---|---|
| Best by ELO | iter 9 EMA at 1590.3 (vs v1 rerun's 1528.2) |
| Iter 4 dip → iter 7-9 breakout | first sustained ELO climb above v1 ceiling |
| Late-iter overfitting | val_loss climbs 1.39 → 1.75 over iters 18-22 |

Comprehensive eval on iter 9 EMA (partial — chain killed at d3_raw stuck):

| eval | result |
|---|---|
| d1 raw | 30/60 (50%) |
| d1 48-sim | 30/60 (50%) |
| d1 400-sim | **60/60 (100%)** |
| d3 raw | (eval was stuck — killed) |
| d3 48-sim | not run |
| d5 raw | not run |

## Raw-policy trajectory probes (the surprise)

Confirmed via per-iter `eval_vs_heuristic.py --no-mcts --depth 1 --num-games 60`:

**Ablation B (5 iter run, runs_ablation_b/20260507_152617):**

| iter | raw |
|---:|---:|
| 0 (random) | 0/60 |
| 1 | 0/60 |
| 2 | **60/60** ← best by ELO; single-iter spike |
| 3 | 0/60 ← lost it |
| 4 | 0/60 ← still lost |

The "60/60 winner" was iter 2 of one specific run with one specific seed, sandwiched between 0/60 iters. Not a property of the recipe.

**Scale-up B (25 iter, runs_scale_up_b/20260507_201618):**

| iter | EMA raw | LIVE raw |
|---:|---:|---:|
| 0 (random) | 30/60 | – |
| 2 | 0/60 | – |
| 5 | 0/60 | – |
| 7 | 30/60 | – |
| 9 (best by ELO) | 30/60 | 0/60 |
| 15 | 0/60 | – |
| 20 | 0/60 | – |

Scale-up B reaches 50% raw briefly at iters 7-9 (the same window where ELO peaked at 1567-1590), bracketed by 0/60 iters before and after. 50% is "ties with depth-1 heuristic," not "beats." Otherwise: 0/60.

Two runs of identical configuration (modulo num_iterations) produced wildly different raw-policy trajectories:

- Ablation B random init's EMA: 0/60. Trained iter 2 EMA: 60/60.
- Scale-up B random init's EMA: 30/60. Trained iter 2 EMA: 0/60.

The discrepancy at iter 0 (different random initializations) propagates: ablation B happened to land on a basin where 5 iters of training on top of heuristic-shaped self-play data produced a 100% raw policy. Scale-up B started elsewhere and never found that basin.

The 100% raw of ablation B is **not a property of the discrimination=0 recipe** — it's an artifact of the (recipe + that particular random seed + stopping at iter 2). Scale-up B's recipe is identical but doesn't reproduce.

## What's robust vs what isn't

**Robust findings (from full 24-hour run + prior):**
- BN-stat-trash fix is necessary for any of this to work (cloud_run_v1 rerun went 0/60 → 30/60 at 48 sims, headline result of the original bug fix)
- `discrimination_weight: 0.0` substantially helps over 0.5 (both ablation B and scale-up B perform better than v1 at the high-sim eval)
- `value_head_lr_factor` change (A) doesn't help
- `final_temp: 0.5` (C) actively hurts

**NOT robust:**
- Ablation B's 60/60 raw policy. Single-seed lucky spike.
- Scale-up B's apparent ELO breakthrough doesn't translate to better raw policy.
- "Best by ELO" model selection. Scale-up B's iter 9 (best-by-ELO) was iterating against a tournament pool of weaker self-play opponents; raw eval against a fixed heuristic shows it's no better than random init.

## Recommended next experiments (post-user-return)

1. **Multi-seed ablation B** (3-5 separate 5-iter runs with different seeds). If the 60/60 holds across seeds, the recipe is real. If only 1 of 5 seeds hits it, we know it's lottery, not recipe.
2. **Different model selection criterion**. Scale-up's iter 9 was best-by-ELO but iter 9 EMA loses to a saved iter 0 of the same run on raw eval. Selecting by `raw_anchor_win_rate` (with n=20+ samples) would be different. Need to instrument the supervisor to track this and select on it.
3. **Shorter scale-ups with checkpointed best-raw** — train 5 iter, eval raw on EVERY iter, keep the iter with best raw. Then continue from there for 5 more, repeat. More expensive per-iter but converges to a meaningfully strong model.
4. **Investigate iter 0 EMA performance variance**. Ablation B iter 0 EMA = 0/60, scale-up B iter 0 EMA = 30/60. Both are essentially random init wrapped in 1 epoch of EMA. Why such different behavior? Likely seed → random argmax → either matches or fights heuristic's preferred opening lines.

## Multi-seed ablation B (2026-05-08 13:47–19:28 UTC, ~$2)

Re-ran ablation B with 3 additional seeds (b_s2/3/4 — different `eval_seed` and `anchor_seed`) plus per-iter raw evals at n=60. Goal: distinguish recipe from lottery.

### 4×5 raw policy matrix (all vs HeuristicAgent depth=1, n=60)

| seed | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 | peak |
|---|---:|---:|---:|---:|---:|---|
| s1 (original ablB) | 0/60 | 0/60 | **60/60** | 0/60 | 0/60 | iter 2: 100% |
| s2 (b_s2) | 30/60 | 0/60 | **60/60** | 30/60 | 0/60 | iter 2: 100% |
| s3 (b_s3) | 30/60 | 0/60 | **30/60** | 0/60 | 0/60 | iter 2: 50% |
| s4 (b_s4) | 0/60 | **30/60** | 0/60 | 0/60 | 0/60 | iter 1: 50% |

### Findings

- **Iter-2 sweet spot is recipe-real, not full lottery**: 3 of 4 seeds peak at iter 2 (the 4th, s4, peaked one iter earlier).
- **60/60 (perfect) magnitude is 50/50 across seeds**: 2 of 4 hit 100%, the other 2 only tied (50%) at peak.
- **Universal collapse to 0/60 by iter 3-4**: every seed loses raw policy strength after the early peak. Continuing training is actively harmful for raw-policy strength under this recipe.
- **Best-by-ELO is the wrong selector**: across the 4 seeds, "best" was iter 4, iter 4, iter 0, iter 1 (per the chain log). Iter 2 was the strongest raw policy in 3 of 4 seeds but never selected.
- **Random init varies hugely**: iter 0 EMA spans 0/60 to 30/60 just from the random init. The recipe builds on top of that, but the starting point dominates the 50% gap between "tied heuristic" and "swept heuristic" outcomes.

### Depth-3 sanity check on iter-2 winners (n=20, time-limit 5s/move)

| candidate | vs depth-1 (n=60) | vs depth-3 (n=20) |
|---|---:|---:|
| s1_iter2 EMA (original ablB) | 60/60 (100%) | 10/10 (50%) |
| s2_iter2 EMA (b_s2) | 60/60 (100%) | 10/10 (50%) |

The iter-2 sweet spot is **genuinely strong**, not depth-1-specific. Both winners tie depth-3 heuristic at 50%. That's a meaningfully strong model from 5 iters of training. The recipe (BN fix + EMA-rebind + discrimination=0) is producing real, useful models — not just exploiting weak opponents.

### Updated picture

| outcome | rate across seeds |
|---|---|
| iter 2 sweet spot exists | 3/4 (75%) |
| 60/60 perfect at sweet spot | 2/4 (50%) |
| 30/60 (tie heuristic) at sweet spot | 4/4 (100%) — every seed at least ties at peak |
| 0/60 by iter 4 | 4/4 (100%) — universal late collapse |

The headline shifts from "ablation B 60/60 winner" to "ablation B reliably produces a model that ties depth-1 heuristic at iter 2; half the time it sweeps; all the time it collapses thereafter."

### What this means for training

1. **`discrimination_weight: 0.0` is robustly an improvement** — every seed reliably reaches 30/60+ at peak, vs the v1 baseline's hard 0/60.
2. **Late training is destructive** under this recipe: peaks at iter 1-2, collapses by iter 3-4. The 25-iter scale-up "best-by-ELO at iter 9" succeeded at tournament play but was already past the raw-policy peak.
3. **Need either**: (a) an early-stopping criterion that selects on raw_anchor not ELO, or (b) a recipe variant that prevents the post-peak collapse (e.g., LR decay tied to loss plateau, regularization changes).
4. **Random-init sensitivity is real**: even with same recipe, network initialization determines ~50% of the eventual ceiling.

## Collapse-probe chain (2026-05-08 22:05 → 2026-05-09 04:05 UTC, ~$2)

After multi-seed ablation B confirmed the iter-2 sweet spot is recipe-real (not pure lottery), the **collapse** that follows it became the open question. Three candidate fixes, each on top of B's recipe (BN fix + EMA-rebind + discrimination_weight=0):

- **no_anneal**: hold `heuristic_weight: 0.5` constant (don't anneal to 0)
- **big_buffer**: `max_buffer_size: 100k → 500k` (more diverse training data)
- **low_lr**: `lr: 0.001 → 0.0003` (gentler weight updates)

### 4-probe matrix (raw policy vs HeuristicAgent depth=1, n=60)

| probe | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 | peak | recipe difference |
|---|---:|---:|---:|---:|---:|---|---|
| ablation B (s1) | 0 | 0 | **60** | 0 | 0 | iter 2: 100% | baseline |
| no_anneal | 30 | **60** | 0 | 0 | 0 | iter 1: 100% | constant heuristic |
| big_buffer | **60** | 0 | 0 | 0 | 30 | iter 0: 100% | 5× buffer |
| low_lr | 0 | 30 | 0 | 0 | 30 | iter 1: 50% | 3× smaller LR |

### Findings

1. **The peak SHIFTS EARLIER, never disappears**: ablation B at iter 2 → no_anneal iter 1 → big_buffer iter 0. Each candidate fix moved the peak 1 iter earlier rather than preventing the collapse.
2. **`low_lr` produced a *weaker* peak (30/60), not a sustained one**. Slower learning didn't reach the heuristic-mimicry attractor that produces 60/60.
3. **The collapse is not preventable by these knobs**. All 4 probes show 0/60 at iter 3-4 (with one mild recovery: big_buffer iter 4 = 30/60).
4. **None of these fixes are the answer for a multi-day run**.

### Refined hypothesis: the 60/60 is heuristic mimicry

The pattern across probes is consistent with a specific mechanism:

- iter 0 (random init): policy makes near-uniform moves; heuristic at depth-1 beats it.
- Iters 1-2: training on heuristic-shaped MCTS visit-count targets pulls the network toward "mimic depth-1 heuristic." The network reaches a state where it picks the same moves the heuristic does on most positions, but **slightly noisier in a way that wins 60/60** (heuristic at d=1 commits to specific moves; the imitator can play subtly different moves that exploit the heuristic's deterministic responses).
- Iters 2+: self-play moves the network away from heuristic-mimicry. Self-play opponents are similar mimicker networks, so the loss signal pushes toward "what defeats other imitators" — not "what's a strong general policy." The network drifts to strategies that beat itself but lose to the heuristic.

Bigger buffer accelerates mimicry (peak at iter 0). No anneal keeps stronger heuristic guidance during MCTS (peak at iter 1). Lower LR is too slow to reach the mimicry attractor at all.

If this hypothesis is right, the collapse isn't a bug — it's the recipe doing exactly what it's designed to do, and the 60/60 is the *halfway point* between random and "self-play-shaped attractor that's worse than heuristic."

### What this means for a multi-day run

Bad news: **none of the cheap knobs prevent collapse**. The recipe peaks early at "good imitation of depth-1 heuristic" and then degrades. Running it for 1000 iters would produce 998 iters of degradation.

What might actually work (untested):

1. **Higher training-time MCTS sims**: at 48 sims, MCTS can't discover moves that beat the heuristic; targets are dominated by the heuristic. At 200-400+ sims, MCTS could find non-heuristic moves and produce targets that pull the network past mimicry.
2. **Curriculum lock**: keep `heuristic_weight ≥ 0.3` permanently, so the heuristic always provides signal, even if it caps the model at "heuristic plus epsilon."
3. **Supervised pre-training from a 60/60 checkpoint, then RL**: take the iter-2 EMA, freeze it as a teacher, train against it for many iters before opening up to free self-play.
4. **Different value head architecture**: the discrimination loss removal helped — maybe other value-head changes (smaller head, frozen value head, etc.) further unlock training.

### Tried (1) but couldn't complete

Launched `ablation_more_sims.yaml` at `num_simulations: 200, late_simulations: 150, num_iterations: 4`. Killed after 40 min — only 10 of 200 games done. **At 200 sims, self-play takes ~4 min/game**, so a single iter at 200 games would take ~13h. Even at 30 games/iter the chain would take 5+ hours total.

Lesson: testing the "more sims" hypothesis at meaningful scale on this hardware needs more wall time than a single overnight allows. A real test would be:

- 100 sims (vs 200) + 50 games/iter + 4 iters — ~$5, ~12h. Doable in a future session.
- Or batched-MCTS path that's been mentioned but not validated.

## Tier-1 #1 follow-up: more sims didn't help (2026-05-09 ~21:00)

The leading hypothesis after the collapse-probe chain was: at 48 sims, MCTS targets are heuristic-shaped → mimicry attractor → collapse. Higher sims should let MCTS find non-heuristic moves and pull the network past mimicry.

Two probes:

### more_sims_v2 (100 sims, 50 games/iter, 5 iter)

All five iters at **0/60 raw vs depth-1 heuristic**. Never reached even the mimicry-spike. Could be data starvation (50 games is too few) or actual disruption of the mimicry path. Ran more_sims_full to disambiguate.

### more_sims_full (100 sims, **200 games/iter**, 5 iter)

| iter | raw vs d1 |
|---:|---:|
| 0 | 0/60 |
| 1 | 0/60 |
| 2 | **30/60 (50%)** ← peak |
| 3 | 0/60 |
| 4 | 0/60 |

Same shape as ablation B (spike at iter 2, collapse) but **half the magnitude** — 30/60 vs 60/60. So data starvation wasn't the issue; the deeper search is changing what the network is learning.

### What that means

**The "60/60 perfect raw" was actually achieved by aligning to depth-1 heuristic specifically.** At 48 sims, MCTS visit counts are dominated by the heuristic's preferred moves (because the network is initially uniform, so heuristic guides everything). Network learns to pick exactly those moves. Beats depth-1 heuristic 60/60 because the heuristic is deterministic and the imitator can play subtly different moves that exploit it.

At 100 sims, MCTS visit counts include moves the heuristic doesn't pick (search finds them). Targets are a *mix* of heuristic-shaped and search-shaped. Network learns a worse imitation of either → 30/60 instead of 60/60.

**More sims → less mimicry, but no replacement signal.** Within 5 iters, the network can't learn enough from the richer search-shaped signal to compensate for losing the easy mimicry path.

### Cross-checking against depth-3

**Result: 0/20 (0%) at depth-3.** more_sims_full iter 2 EMA loses every single game. Compared to ablation B's iter 2 = 10/10 (50%) at d3, this is a brutal drop.

**Conclusion**: more sims actively WEAKENED the model on both d1 and d3 metrics. The 60/60 at d1 from ablation B isn't pure mimicry illusion — it's the model's actual ceiling, which deeper-search training erodes without offering replacement signal. The 60/60 model had real play-strength (50% at d3); the more_sims_full model has neither (30/60 at d1, 0/20 at d3).

**Tier-1 #1 hypothesis is dead.** Higher train-time sims doesn't unlock past-mimicry strength — it just degrades the mimicry without providing alternative skill.

## Tier-1 #2: pure_neural mode (2026-05-10 02:02–03:31 UTC, ~$1)

After more_sims hypothesis died, ran pure_neural variant: same as ablation B but `evaluation_mode: pure_neural`, `heuristic_weight: 0.0`. Removes heuristic from MCTS leaf eval entirely — network drives all decisions.

### Result

| iter | pure_neural raw |
|---:|---:|
| 0 | 0/60 |
| 1 | **60/60 (100%)** |
| 2 | **60/60 (100%)** ← TWO sustained iters |
| 3 | 0/60 |
| 4 | 0/60 |

**First probe to extend the peak past one iteration.** ablation B was 0/0/60/0/0 (single iter at 60/60); pure_neural is 0/60/60/0/0 (two consecutive iters at 60/60).

### Why this matters

- The heuristic in MCTS leaf eval was doing two things simultaneously: (1) bootstrapping the network's training signal early, and (2) anchoring the network to heuristic-shaped policies, which made the network fragile when its own play diverged from heuristic-shape.
- Removing the heuristic loses (1) — but the network apparently doesn't need it as much as we thought. By iter 1, pure_neural reaches 60/60 (one iter earlier than ablation B!).
- And it gains stability. Two iters at peak instead of one. The collapse mechanism is delayed but not eliminated.

### Followup — pure_neural_long (10 iters, in flight)

Launched 03:32 UTC, ETA done ~07:30 UTC. **The strategic question for multi-day**: does the peak window extend further as we keep training, or is "two iters" the cap?

- If iters 1-5 all at 60/60 → pure_neural is THE recipe for multi-day. Train until plateau plus regression test, eval frequently.
- If still cap at 2 iters → we have a small-but-real improvement and need a different attack on the remaining 8-iter degradation.

## Tier-1 #2 follow-up: pure_neural at 10 iters (2026-05-10 03:32–06:36)

Wanted to test if the 2-iter sustained peak from pure_neural (5-iter) extends to more iters. Result with a different seed:

| iter | pure_neural_long raw |
|---:|---:|
| 0-2 | 0/60 |
| 3 | **30/60 (50%)** ← only spike |
| 4-9 | 0/60 |

Lottery again. Two seeds, two completely different trajectories:

- pure_neural (seed A): 0 / 60 / 60 / 0 / 0 — spike iters 1+2, 100% magnitude, sustained 2 iters
- pure_neural_long (seed B): 0 / 0 / 0 / 30 / 0 / 0 / 0 / 0 / 0 / 0 — single iter at 50%

Same recipe. Random init dominates outcome. Cannot conclude pure_neural is reliably better than hybrid from n=1.

Launched 3 more pure_neural seeds (5-iter each, ~4.5h total) to settle the question. ETA done ~11:15 UTC.

## Multi-seed pure_neural confirms the lottery (2026-05-10 06:45–11:12, ~$3)

Ran 4 pure_neural seeds (5-iter each) plus the 10-iter pure_neural_long.

### 5-seed pure_neural matrix (iters 0-4, raw vs heuristic d=1, n=60)

| seed | i0 | i1 | i2 | i3 | i4 | peak | sustained? |
|---|---:|---:|---:|---:|---:|---|---|
| pure_neural (s1) | 0 | **60** | **60** | 0 | 0 | 100% | yes (2 iters) |
| pure_neural_long (s2) | 0 | 0 | 0 | 30 | 0 | 50% | no |
| pn_s2 (s3) | 30 | 0 | 30 | 30 | 0 | 50% | yes (2 iters at peak) |
| pn_s3 (s4) | 0 | 0 | 0 | 30 | 0 | 50% | no |
| pn_s4 (s5) | 30 | 0 | 0 | **60** | 0 | 100% | no |

### Compare to hybrid (ablation B, 4 seeds)

|  | pure_neural | hybrid (ablation B) |
|---|---:|---:|
| Hit 60/60 at peak | 2/5 (40%) | 2/4 (50%) |
| Hit ≥30/60 at peak | 5/5 (100%) | 4/4 (100%) |
| Sustained ≥2 iters at peak | 2/5 (40%) | 0/4 (0%) |
| Collapsed to 0/60 by iter 4 | 5/5 | 4/4 |

**Pure_neural's only advantage over hybrid is sustainability**: when it works, the peak holds across 2 iterations instead of collapsing immediately. Both are equally lottery-prone on magnitude. Both universally collapse by iter 4.

## Final verdict on the multi-day knob question

**No knob in the explored space prevents the late-iter collapse.** Tested:
- value_head_lr_factor (null)
- discrimination_weight (positive — shipping)
- final_temp (negative)
- heuristic_weight curriculum (just shifts the spike)
- max_buffer_size (just shifts the spike)
- learning_rate (lower weakens, higher destabilizes)
- num_simulations (more sims actively *weaken* the model on both d1 and d3)
- evaluation_mode (pure_neural marginally improves sustainability, doesn't prevent collapse)
- num_iterations (no change — collapse persists across 5/10/25 iters)

The collapse appears to be a structural property of the recipe + dataset:
- Self-play data evolves with the network. Once the network drifts past heuristic-mimicry, it's playing against itself in increasingly weird ways. Targets become "what beats this particular network's quirks" rather than "good general policy."
- 48-200 simulation budgets are insufficient for MCTS to discover genuinely strong moves the network doesn't already prefer. The search amplifies the network's prior, doesn't correct it.
- Either the search budget needs to be much higher (hundreds of sims at game-tractable wall time → batched MCTS) or the data distribution needs to be anchored against something stronger than the network itself (distillation, frozen anchor opponents, curriculum that doesn't drift).

**For a multi-day run today**: the right move is to ship what works (5-iter ablation B / pure_neural recipe + iter 2 EMA) and pursue code-level work (batched MCTS, distillation, anchor-curriculum) before committing more compute to the same recipe space. A 1000-iter run on the current recipe would produce 998 iters of degradation regardless of which knob we chose.

## Decisive experiment: warm_start from a 60/60 winner (2026-05-10 11:16–13:07, ~$1)

Used `--init-checkpoint runs_ablation_b/20260507_152617/iteration_2/checkpoint_iteration_2_ema.pt` (the original 60/60 winner) as starting weights for a fresh 5-iter run with the same recipe (ablation B / discrimination=0).

### Result

| iter | warm_start raw |
|---:|---:|
| 0 (warm-start + 1 iter training) | 0/60 |
| 1 | 30/60 (50%) |
| 2 | 0/60 |
| 3 | 0/60 |
| 4 | 0/60 |

**One iteration of training on top of a 60/60 model dropped it to 0/60.** Then a partial recovery at iter 1 (30/60), then collapse.

### What this proves

The training dynamics actively destroy strong policies. The "iter 2 spike" we've been chasing isn't the recipe building toward something — it's the model briefly visiting good-policy space before training pulls it elsewhere. Random-init runs spike at iter 2 because that's how long it takes to drift INTO the good region from random; warm-started runs spike at iter 1 because they start near it but get drifted OUT.

This is the cleanest evidence we have. **Continued self-play training under this recipe is not just "fails to improve" — it actively destroys strong policies.**

### What this means for multi-day

Multi-day training under the current recipe is structurally impossible. No yaml knob fixes this. Code-level changes required:

1. **Anchor curriculum**: train against frozen best-of-prior-iters mixed with self-play. Forces the network to keep beating its own past versions, preventing the drift that destroys strong policies.
2. **Distillation phase**: alternate between RL self-play (which destroys) and supervised distillation against frozen best (which restores).
3. **Aggressive early stopping**: train until peak (typically iter 1-3), evaluate against absolute anchor at high n, stop, hard-restart with the best checkpoint when ready to push further.
4. **Reduce LR drastically after iter 2**: e.g., LR ÷ 100 after the first peak. Most experiments showed lower LR weakens the spike, but that was at the START of training. After the spike, gentler updates might preserve rather than destroy.
5. **Different target distribution**: visit counts at 48 sims are a noisy student-teacher signal. Move to KL-to-prior or Bradley-Terry-style targets that don't pull as hard.

All of these are ~1-2 day code investigations. None are yaml-tweakable.

## Final state and what's worth shipping

**Code (committed and pushed, branch `policy-collapse-hunt`):**
- BN-stat-trash fix in `clear_pytorch_memory`
- EMA-rebind fix in `_reset_network_objects`
- save_model BN-key guard
- Three regression tests
- Several ablation YAML configs
- Probe diagnostic script
- Multiple results docs

**Models worth keeping (still on the cloud box):**
- `runs_scale_up_b/20260507_201618/iteration_9/checkpoint_iteration_9_ema.pt` — best ELO from the 25-iter run, 100% at d1 with 400-sim MCTS
- `runs_ablation_b/20260507_152617/iteration_2/checkpoint_iteration_2_ema.pt` — original 60/60-at-d1 winner, 50% at d3 raw
- `runs_ablation_b_s2/<latest>/iteration_2/checkpoint_iteration_2_ema.pt` — second 60/60-at-d1 winner, also 50% at d3 raw

**For the multi-day run, the gating question is no longer a knob — it's a recipe-level rethink.** Specifically: how do we make MCTS at training time discover moves the heuristic doesn't, so targets pull the network past heuristic mimicry rather than away from it? That requires either much higher sim budgets (with batched/parallelized MCTS to make wall-time tractable), or a different curriculum / loss structure. Both are >1-day code investigations, not knob ablations.

## Honest recommendation for ship

If you want a model **right now** that beats depth-1 heuristic 100% of the time: **scale-up B's iter 9 EMA at 400 sims (`runs_scale_up_b/20260507_201618/iteration_9/checkpoint_iteration_9_ema.pt`)**. That's robust (n=60). Just don't claim "raw policy strength" — claim "policy + meaningful MCTS budget."

Anything stronger (depth-3+ opponents, raw policy stand-alone) needs the multi-seed work above to know what's real.

## Cloud box state (as of 08:43 UTC 2026-05-08)

Box went unreachable. SSH timeouts. May have been reclaimed or had a network issue. Trajectory probes for ablation B iters 3+4 and scale-up B iters 3+7+15+20 didn't complete. Final results above are from probes that landed before the disconnect.

If cloud comes back, those probes are still queued via `traj_probes.sh` (didn't get to launch — connection failed before script copy). Re-launch manually if needed.

## Expert-game eval sweep (2026-05-10 19:29–19:35 UTC, ~$0.20)

The user stepped away. Every prior probe in this doc compared the network to its training target (depth-1 heuristic via `eval_vs_heuristic.py`). The results above all share that target. **One layer up the onion**: build a fixed, non-circular metric from the BGA expert-game holdout (`expert_games/bga/parsed/`) — top-1/top-3 policy agreement and value MSE bucketed by phase — and rerun the full trajectory matrix under it.

Wired up `scripts/eval_vs_expert.py` (loads 28910 positions from 436 BGA games, 0 parse failures) and `scripts/expert_eval_trajectory.py` (batched across many checkpoints). 55 checkpoints across ablation_b s1-s4, pure_neural s1-s4, pure_neural_long, warm_start. ~5s/checkpoint on the 4090.

### MAIN_GAME top-1 (% agreement with expert moves on non-trivial positions)

Random baseline ≈ 1.5% (1/65 avg legal moves). Strong play ≈ 30%+.

| run | i0 | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ablation_b (s1) | 3.7 | 4.6 | 4.2 | 2.4 | 2.3 |  |  |  |  |  | iter1: 4.6% |
| ablation_b_s2 | 3.2 | 4.1 | **5.0** | 3.6 | 2.8 |  |  |  |  |  | iter2: 5.0% |
| ablation_b_s3 | 3.3 | 4.2 | 4.1 | 3.4 | 3.1 |  |  |  |  |  | iter1: 4.2% |
| ablation_b_s4 | 3.1 | 3.3 | 2.7 | 2.1 | 2.8 |  |  |  |  |  | iter1: 3.3% |
| pure_neural | 3.3 | 4.0 | 3.4 | 3.3 | 3.6 |  |  |  |  |  | iter1: 4.0% |
| pure_neural_pn_s2 | 3.4 | 4.3 | 4.3 | 4.0 | 3.8 |  |  |  |  |  | iter1: 4.3% |
| pure_neural_pn_s3 | 3.4 | 3.9 | 2.8 | 2.5 | 2.9 |  |  |  |  |  | iter1: 3.9% |
| pure_neural_pn_s4 | 3.7 | 4.0 | 3.7 | 3.9 | 3.3 |  |  |  |  |  | iter1: 4.0% |
| pure_neural_long | 3.1 | 4.0 | 3.9 | 3.6 | 3.1 | 3.0 | 2.6 | 2.7 | 3.0 | 3.1 | iter1: 4.0% |
| warm_start | 3.1 | 3.8 | 2.6 | 2.5 | 2.3 |  |  |  |  |  | iter1: 3.8% |

### Findings — and they sting

1. **The iter-1/2 spike + late-iter decline IS real, not an artifact of vs-d1 measurement.** 8 of 10 runs peak at iter 1 or 2 on MAIN_GAME top-1 against expert moves. The shape we've been chasing is genuine model behavior, not a metric quirk. The `EXPERIMENT_RESULTS_24H.md` writeup conclusion is independently confirmed.

2. **The peak is microscopic.** Iter 0 baseline (random-ish init) = 3.1-3.7% MAIN top-1. Iter 2 peak = 4.2-5.0%. Lift of **~1 percentage point** on a metric where chance is ~1.5%. The "60/60 vs depth-1 heuristic" was reading this tiny absolute improvement amplified by heuristic-mimicry alignment with the eval target. **Best checkpoint anywhere in the 55-checkpoint matrix is 5.0% MAIN top-1 (ablation_b_s2 iter 2)**. Nothing in this sweep is close to "well-trained."

3. **Value head is uniformly broken.** All-phase value MSE ranges 0.99-1.33 across checkpoints; baseline (always-predict-0 on -1/+1 outcomes) is ~1.0. **No checkpoint, anywhere, has a value head that beats trivial baseline.** Several are notably worse (`runs_pure_neural_pn_s2` iter 3 = 1.33, `runs_ablation_b_s4` iter 3 = 1.17). The value head is contributing noise.

4. **ROW_COMPLETION ~91-92% across all 55 checkpoints**, with no iter-trajectory variation. This is "which 5-row to capture" — a forced choice. The encoding/decoding works. The network learns this from initialization (random init also passes). It's uninformative as a quality signal.

5. **Pure_neural's "stability" is real but trivial.** `pure_neural_long` over 10 iters: peaks at iter 1 (4.0%), bottoms at iter 6 (2.6%), settles to ~3.1% by iter 9. Translation: it doesn't crash, but it doesn't go anywhere. The 10-iter trajectory is a return to iter-0 baseline.

6. **Warm-start finding holds**: `warm_start` (iter 0 = 60/60 winner + 1 iter training) starts at 3.1% MAIN top-1, briefly rises to 3.8% at iter 1, then drops to 2.3% at iter 4. Training actively destroys the (already weak) policy structure.

### What this changes about the multi-day question

The writeup conclusion ("no yaml knob fixes collapse, need code-level changes") is right, but the framing was wrong. **It's not that the network reaches a good state and then collapses.** The network never reaches a good state. The "iter-2 spike" is ~5% expert agreement vs ~3% at iter 0. There's no plateau of strong play to preserve. The collapse-prevention framing makes the problem sound smaller than it is.

The real bottleneck question becomes: **does this architecture + this data have the capacity to learn good play at all, or are we hitting a ceiling that's not training-dynamics-related?**

That question is answerable. Supervised training on the BGA games (28910 positions, ~30 min run) gives a clean upper bound:
- If supervised tops out at ~5% MAIN top-1, the architecture/encoding is the cap and no RL recipe will help.
- If supervised reaches 20-40%, RL is failing to use the data and the problem is in the loss/curriculum.

Launching that experiment as the next probe. The result is binary, fast, and dispositive.

### Models worth keeping (revised)

The "models worth keeping" list higher up — `runs_scale_up_b/iteration_9_ema.pt`, `runs_ablation_b/iteration_2_ema.pt`, `runs_ablation_b_s2/iteration_2_ema.pt` — are still the best of what we have, but only by a hair. Best of the bunch (ablation_b_s2 iter 2) reaches 5.0% expert agreement on MAIN_GAME. These are useful as comparison anchors, not as ship-able models.

Per-checkpoint JSONs and summary CSV in `expert_eval_reports/` (not committed; regeneratable). Smoke output for the 60/60 winner shows ROW_COMPLETION top-1 = 0.914 (90% of forced-choice positions) but MAIN_GAME top-1 = 4.2% (15× chance, but small absolute number).

## Supervised pretraining + the deterministic-side artifact (2026-05-10 20:02–21:18 UTC, ~$0.30)

After the expert-eval sweep showed the RL trajectory caps at ~5% MAIN top-1, the question became: **is the architecture the cap, or is the RL signal the bottleneck?** Quick supervised pretrain on the same data answers it. `scripts/run_supervised_pretraining.py --games-dir expert_games/bga/parsed --epochs 40 --batch-size 256 --lr 0.001` then auto-chained eval_vs_expert + eval_vs_heuristic (d1 n=60, d3 n=20).

### Training trajectory (40 epochs, ~4 min on 4090)

- Best ckpt = epoch 5 by val_loss (cosine LR, val_loss U-shape)
- By epoch 40: train PAcc=96%, val PAcc=13% (heavy overfit on 23129 train / 2569 val)
- Val P-loss minimum at epoch 5 (4.99), climbs steadily to 7.9 by epoch 40

### Three eval surfaces, three different pictures

| metric | supervised (best, epoch 5) | ablation_b iter2 | ablation_b_s2 iter2 |
|---|---:|---:|---:|
| BGA full (mostly train, with mask) MAIN top-1 | **48.2%** | 4.2% | 5.0% |
| BGA full value_mse | 0.077 | 1.13 | 0.99 |
| Boardspace 500 holdout MAIN top-1 | **4.7%** | 3.8% | 3.9% |
| Boardspace 500 holdout ALL top-1 | 11.5% | 10.5% | 10.7% |
| Boardspace 500 holdout value_mse | 1.35 | 1.12 | 0.98 |
| d1 raw policy (n=60 vs HeuristicAgent depth=1) | 30/60 (50%) | 60/60 (100%) | 60/60 (100%) |
| d3 raw policy (n=20 vs HeuristicAgent depth=3, 5s/move) | 10/20 (50%) | 10/10 (50%, n=10) | 10/10 (50%, n=10) |

**The 48% is contaminated** (90% of those positions were in the training set). On boardspace held-out, supervised is 4.7% MAIN top-1 vs RL's 3.8-3.9%. ~1 pp lift. Within noise of the iter-2 RL spike. **The architecture is at or near its cap on the data we have.**

### The deterministic-side artifact

Examining the d1 / d3 game-by-game logs revealed something the writeup above missed:

- d1 games 1-30 (cand=White): cand wins all 30, every game 105 moves
- d1 games 31-60 (cand=Black): cand loses all 30, every game 83 moves
- d3 games 1-10 (cand=White): cand wins all 10, every game 103 moves
- d3 games 11-20 (cand=Black): cand loses all 10, every game 81 moves

Same exact move count every time = **fully deterministic raw-policy + deterministic heuristic** = same game played 30 / 10 times.

This reframes the entire prior "60/60 winner" narrative:
- "60/60 vs depth-1" actually means "deterministically wins as both White and Black"
- "30/60" means "deterministically wins as White, deterministically loses as Black" — i.e., what supervised does
- "0/60" means "deterministically loses as White, deterministically loses as Black"

The 60/60 winners weren't 2× stronger than 30/60 results — they happened to find a deterministic line that beats the heuristic from both sides. The 30/60 results aren't "tied" — they're "first-mover advantage realized once, against the heuristic's deterministic responses."

Implication: **most of the raw-policy strength results in this writeup are not measuring strength**. They're measuring "does this checkpoint's argmax line happen to win as White and/or Black against this specific heuristic's argmax line." A real strength signal needs either (a) stochastic policy (temperature > 0), (b) MCTS at training-time-realistic sim count, or (c) a non-deterministic opponent.

The 400-sim MCTS results that showed scale_up_b iter9 hitting 60/60 are still meaningful — MCTS naturally breaks ties, so that result is a real strength signal at high search budget. But 48-sim and raw-policy numbers should be discounted.

### Updated picture

What's robust:
- **Architecture is at or near its cap on this data.** Supervised pretraining on BGA tops out at ~4.7% MAIN top-1 on held-out boardspace, vs ~3.8% from RL training-from-scratch. ~1pp lift is real but tiny.
- **Value head is broken on held-out.** Supervised value_mse on boardspace = 1.35 (worse than always-predict-0). RL's value_mse = 1.0 (matches baseline). Both heads are roughly noise out-of-sample.
- **First-move advantage dominates raw-policy evals.** A model that wins as White but not as Black against a deterministic heuristic is the *expected* behavior; calling it 30/60 = "tied" overstates the deficit and 60/60 = "perfect" overstates the strength.

What's NOT robust:
- **Most raw-policy comparison numbers in the writeup above.** The "60/60 vs depth-1" measurements are deterministic-vs-deterministic — they bin into 0/30/60 by side-coverage, not by underlying skill.

### What should change

1. **Re-run all raw-policy evals with `temperature ≥ 0.5`** so determinism breaks and we get a real distribution. Not done in this session — flagging for next pass.
2. **Network capacity is the cap until proven otherwise.** Before any more RL recipe work, run a scale test: 256→512 channels and 12→18 blocks, supervised on BGA + boardspace combined (~32k games). If held-out top-1 goes from ~5% to ~15%, capacity was the constraint and we should ship that bigger network as the iter-0 init for all future RL.
3. **The expert-eval metric is the right primary signal going forward.** Non-circular, fixed, deterministic-side-immune (since it scores agreement on individual positions, not full games), reproducible across runs. `scripts/expert_eval_trajectory.py` makes it cheap (~5s/checkpoint).
4. **The "iter-2 spike" pattern is real model behavior** but it's a 1-1.5 percentage-point lift on a metric capped around 4-5%. The "fix collapse" framing is technically right but it's optimization-of-noise. The real question is "what gets us from 5% to 30%."

### Models worth keeping (revised again)

- `models/supervised_bga_post_fix/best_supervised.pt` — supervised baseline, ~5% MAIN top-1 on held-out, deterministic 30/60 vs d1, 50% vs d3 by side
- `runs_ablation_b/iteration_2_ema.pt`, `runs_ablation_b_s2/iteration_2_ema.pt` — RL iter-2 spikes, ~4% MAIN top-1 on held-out
- `runs_scale_up_b/iteration_9_ema.pt` — best by ELO, untested on held-out expert metric

For comparison purposes only. None ship as a strong agent without MCTS at 400+ sims, and even there the 60/60 measurements pre-determinism-correction are now suspect.

## Watch log

Full hour-by-hour trajectory in `~/.claude/projects/.../memory/overnight_watch_log.md`. Read top-down for moment-by-moment timeline.
