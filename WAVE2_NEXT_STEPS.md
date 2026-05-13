# Wave 2 — Next Steps

Pick-up document for the session that follows the 2026-05-12 Wave 2 cloud run. This is the index; deep dives live in `WAVE2_STEP_{1,2,3}_*.md`.

---

## State of work (as of 2026-05-13)

### What we just did

- Ran `configs/warm_start_combined_recipe.yaml` end-to-end on cloud (Vast.ai 4090).
- 5 iters × 200 games × 4 epochs. Total wall time **13.05h**. No infra failures.
- Initialized from `models/supervised_seed/best_supervised.pt` (Option A in the runbook — the `supervised_deep_256x18` checkpoint was lost on a prior instance).
- Wave 1 preflight: **123/123** tests passed (after a one-line NumPy 2.x compat fix in `metrics_logger.py:469`).
- Section §8 final eval suite (60 games × 3 modes on iter 1 EMA vs depth-1 heuristic) complete at 2026-05-13 16:23 UTC. Results in `runs_warm_start_combined_recipe_cloud/20260512_151604/section8_evals/`:

  | Mode | Win rate | W/L | Per-side W/B | vs runbook §8 baseline |
  |---|---:|---:|---|---|
  | **400-sim MCTS** | **76.7%** | 46/14 | 23/30, 23/30 | **way above** 30% baseline; CI ~[66%, 87%] |
  | **48-sim MCTS** (training config) | 41.7% | 25/35 | 13/30, 12/30 | above 33% baseline but CI overlaps it; **inconclusive** |
  | **Raw policy** | **25.0%** | 15/45 | 7/30, 8/30 | **below** 33% baseline — Wave 1 hypothesis on raw policy **fails** |

  Per-side balance is excellent — no white/black side bias in any mode. The 400-sim number is the headline strength signal. The 48-sim and raw numbers tell the more important story: **MCTS at depth is compensating for a network whose policy head alone is weak and whose value head is off**. This is consistent with the negative value-outcome correlation we found across all 4 measured iters.
- All artifacts pulled to laptop: `runs_warm_start_combined_recipe_cloud/20260512_151604/` (1.5 GB, all 5 iters incl checkpoints + EMA + metrics sidecar + manifest + logs).

### Key signals from the run

```
                 ploss  vloss   Elo      raw-anchor    mcts-anchor    decision
iter 0           4.226  7.503   1500.0   (no anchor)   (no anchor)    promoted (first)
iter 1           3.546  3.505   1516.7   50%           50%            promoted (Wilson fail, Elo up)
iter 2           2.944  1.844   1471.8   75%  ← peak   25%            REVERTED (Elo down)
iter 3           2.938  1.862   1466.1   50%           25%            REVERTED
iter 4           3.042  1.872   1460.5   0%   ← floor  0%             REVERTED
                                         ─── n=4 each — noisy ───
```

- **Iter 1** is the best stable checkpoint (only promotion that stuck).
- **Iter 2** peaked at 75% raw vs heuristic but Elo regressed → reverted.
- **Iters 3 and 4** degraded despite using iter 1's weights for self-play.
- All anchor samples are **n=4 per iter** — too noisy to draw firm conclusions per-iter.

### Two real bugs surfaced

1. **Metric wiring gaps**: 3 of 5 Wave 2 acceptance gates either weren't logged (`mcts/effective_child_visits`, `train/effective_batch_size`) or were logged but read with the wrong index (`eval/value_outcome_correlation`). See [Step 1](WAVE2_STEP_1_WIRING_FIXES.md).
2. **Negative value-outcome correlation** at *every measured iter* (-0.10, -0.31, -0.07, -0.07). Runbook flags this as the T1.3 regression signature ("value head learning the wrong sign") — but the broken gate-check read entry[-1] instead of entry[0], so we missed it live. The signal is real and unresolved.

### What we know vs what we don't

| Question | Status |
|---|---|
| Did the run complete cleanly? | ✅ Yes |
| Did Wave 1 fixes prevent infra failure? | ✅ Yes (no crashes, no metrics_logger crashes after np.float_ patch) |
| Does the model improve over iters? | Mixed. Losses drop monotonically; anchor signal noisy; tournament Elo peaked at iter 1 then declined. |
| Is the value head learning the wrong sign? | Likely yes — all 4 measured iters showed negative value-outcome correlation. Needs code-level investigation. |
| Did Wave 1's raw-policy fix help? | **No — measured at n=60. Raw policy = 25% vs heuristic, below the 33% baseline.** The iter 1 anchor's 50% raw was noise; the real number is lower. T1.2 value-target fix did not deliver on raw policy strength. |
| Is the iter 2 → iter 4 regression real or noise? | Unknown. Could be n=4 noise, could be real value-head degradation. |

---

## The three-step plan

### [Step 1 — Wiring fixes + value-correlation investigation](WAVE2_STEP_1_WIRING_FIXES.md)

**Local dev, no cloud spend.** Fix the 3 metric-wiring bugs and investigate the negative value-outcome correlation. Without working gates, any future cloud run produces the same blind interpretation. Estimated effort: **1 day of focused work**.

### [Step 2 — Validation re-run with working telemetry](WAVE2_STEP_2_VALIDATION_RERUN.md)

**Cloud, ~$30-50, ~12-15h wall time.** Same recipe, but `anchor.num_games: 4 → 40` so we get statistically meaningful per-iter readings. All 3 wiring bugs fixed. The point: one clean data point that actually answers "did Wave 1 help?" with confidence intervals you can stand behind. Pre-condition: Step 1 complete and tests green.

### [Step 3 — Revert diagnosis (conditional)](WAVE2_STEP_3_REVERT_DIAGNOSIS.md)

**Only runs if Step 2 still shows post-iter-1 regression.** Try `revert_self_play_on_gate_failure: false` to see whether the gate-revert is masking real learning or saving us from collapse. May extend `num_iterations` to 10-15 to see if continuous training climbs higher (AlphaZero-style). Cloud cost similar to Step 2.

---

## Open follow-up tasks (carry forward from the session that produced this doc)

- **#11**: Commit the local `metrics_logger.py:469` `np.float_ → np.float64` patch (currently uncommitted on `training-pipeline-fixes`). One-liner. Required for any future run that lands on a NumPy 2.x machine. _Hit this first — easy._
- **#12**: Investigate `[SearchConsistency] step fired 20× but every attempt produced 0 distillation samples` warnings that appeared 4× near the end of iter 4. Re-enable DEBUG logging on a smoke run to inspect per-position skip cause.
- **§8 final eval suite results**: ✅ done. Files pulled to `runs_warm_start_combined_recipe_cloud/20260512_151604/section8_evals/eval_{400sim,48sim,raw}.json`. Numbers above. The raw 25% result elevates Step 1's value-sign investigation from "nice to do" to **critical** — Wave 1 didn't deliver on the raw policy improvement claim.
- **Vast.ai instance teardown** — currently still alive at `23.158.136.85:26654` running the §8 evals. Tear down via Vast web UI once evals finish + results pulled.

---

## Important file/path references

- **Branch**: `training-pipeline-fixes` (off `policy-collapse-hunt`)
- **Cloud run artifacts (local)**: `runs_warm_start_combined_recipe_cloud/20260512_151604/`
- **Recipe**: `configs/warm_start_combined_recipe.yaml`
- **Training log**: `wave2_cloud.log` (laptop root)
- **Runbook**: `CLOUD_RERUN_RUNBOOK.md`
- **Wave 1 preflight tests**: 10 files under `yinsh_ml/tests/test_*.py` (see runbook §5 for the exact list)
- **Value pipeline tests** (focus for Step 1): `test_value_target_pipeline.py`, `test_mcts_backprop_perspective.py`, `test_value_head_diagnostics.py`
- **SSH alias**: `cloud` points to `23.158.136.85:26654` (in `~/.ssh/config`). Update if reprovisioning.

---

## How to start the next session

1. Open this file and re-orient.
2. Decide which step to tackle. Step 1 is the gate — Step 2 can't produce trustworthy data until Step 1 is done.
3. Open the corresponding `WAVE2_STEP_{N}_*.md` and follow the pre-flight checklist.
4. Touch base with the open tasks above (especially #11 — commit the np.float_ patch before anything else).
