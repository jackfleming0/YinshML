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
| 2 | **60/60** ← best by ELO; the 100% headline |
| 3, 4 | (cloud went down before measured) |

**Scale-up B (25 iter, runs_scale_up_b/20260507_201618):**

| iter | EMA raw | LIVE raw |
|---:|---:|---:|
| 0 (random) | 30/60 | – |
| 2 | 0/60 | – |
| 5 | 0/60 | – |
| 9 (best by ELO) | 30/60 | 0/60 |

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

## Honest recommendation for ship

If you want a model **right now** that beats depth-1 heuristic 100% of the time: **scale-up B's iter 9 EMA at 400 sims (`runs_scale_up_b/20260507_201618/iteration_9/checkpoint_iteration_9_ema.pt`)**. That's robust (n=60). Just don't claim "raw policy strength" — claim "policy + meaningful MCTS budget."

Anything stronger (depth-3+ opponents, raw policy stand-alone) needs the multi-seed work above to know what's real.

## Cloud box state (as of 08:43 UTC 2026-05-08)

Box went unreachable. SSH timeouts. May have been reclaimed or had a network issue. Trajectory probes for ablation B iters 3+4 and scale-up B iters 3+7+15+20 didn't complete. Final results above are from probes that landed before the disconnect.

If cloud comes back, those probes are still queued via `traj_probes.sh` (didn't get to launch — connection failed before script copy). Re-launch manually if needed.

## Watch log

Full hour-by-hour trajectory in `~/.claude/projects/.../memory/overnight_watch_log.md`. Read top-down for moment-by-moment timeline.
