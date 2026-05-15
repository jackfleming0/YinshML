# Wave 2 — Next Steps

Index document. Wave 2 is **DONE**. The three-step plan ran to completion. Wave 3 begins at `WAVE3_BRANCHES.md`.

---

## Wave 2 outcome summary (as of 2026-05-15)

### Step 1 — DONE (commit `3cb047f`)
- B1/B2/B3 telemetry wiring fixes shipped.
- `last_root_value` sign-flip fix at `self_play.py:798/1022` shipped (+ 3 new regression tests).
- Wave 1 preflight: **156 passed, 1 skipped** locally; **126 passed** on the cloud python 3.12 environment.
- Local smoke at `configs/wave2_wiring_verify.yaml` produced iter 1 sidecar with `eval/value_outcome_correlation = +0.0814` (was negative on the prior cloud run).
- See [WAVE2_STEP_1_WIRING_FIXES.md](WAVE2_STEP_1_WIRING_FIXES.md) for the deep dive.

### Step 2 — DONE (commit `336dd98`, run `runs_warm_start_combined_recipe/20260514_122701/`)
- 5 iters × 200 games × 4 epochs over 20.34h on Vast.ai 4090. Cost ~$10.
- All 5 Wave 2 gates produced sensible values across all 5 iters.
- Candidate `eval/value_outcome_correlation` was **positive in every iter** (+0.061 to +0.198) — Step 1's sign-flip fix held end-to-end on n=40 production data.
- **Wave 1's training-quality claim does not survive n=40 anchors.** Best raw policy 30% (below 33% baseline). Best MCTS-48 60% at **iter 0** (the supervised seed, before any training).
- See [WAVE2_STEP_2_POSTMORTEM.md](WAVE2_STEP_2_POSTMORTEM.md) for the deep review (5 findings + 1 bug + EMA-drift hypothesis).

### Step 3 — SUPERSEDED
- Original conditional: try `revert_self_play_on_gate_failure: false` if Step 2 still showed post-iter-1 regression.
- **Won't be run.** Step 2's deep review found the revert path *did* fire every iter (the SUMMARY log line is misleading — supervisor.py:2028 bug). Turning the revert off would let weaker self-play data compound across iters, making things worse, not better.
- `WAVE2_STEP_3_REVERT_DIAGNOSIS.md` remains in the repo for historical context only.

### Per-iter scoreboard (Step 2)

| Iter | ploss | vloss | val_acc | cand_corr | tourn Elo | anchor raw WR (n=40) | anchor MCTS-48 WR (n=40) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (seed init) | 3.96 | 6.80 | 9.8% | — | 1500 | 20.0% | **60.0%** ← peak |
| 1 | 3.36 | 2.10 | 17.0% | +0.161 | 1450 | **30.0%** ← peak | 50.0% |
| 2 | 3.07 | 1.87 | 21.5% | +0.087 | 1500 | 22.5% | 35.0% |
| 3 | 3.11 | 1.90 | 21.7% | +0.161 | 1488 | 27.5% | 32.5% |
| 4 | 3.35 | **3.68** | **9.8%** | +0.061 | 1483 | 17.5% | 47.5% |

---

## Wave 3 — Branches A / B / C

See [WAVE3_BRANCHES.md](WAVE3_BRANCHES.md) for full plans.

| Branch | Cost | Question |
|---|---|---|
| **A** | $0 | Is EMA drift the bottleneck? Settled by evaluating each iter's non-EMA candidate. |
| **B** | ~$10-15 per pass | Which single recipe knob has the highest leverage? Decided by Branch A. |
| **C** | Day or two | Should the warm-start + self-play pipeline shape change? Decided by A+B. |

Do them in order. A is free.

---

## Open follow-ups (deferred during Step 2)

- **#12 (from Step 1 doc)**: `[SearchConsistency] step fired 20× but every attempt produced 0 distillation samples` warnings near the end of iter 4 in the prior run. Not seen in Step 2 yet; revisit if it reappears.
- **§8 final eval suite for iter 0 EMA on Step 2 run**: in progress on cloud as of 2026-05-15 (3 × 60 games at 400/48/raw modes). Results land at `runs_warm_start_combined_recipe/20260514_122701/section8_evals/`.
- **Vast.ai instance teardown**: still alive at `cloud` alias (`69.176.92.111:52868`). Tear down after Branch A finishes.

---

## File / path references

- **Branch**: `training-pipeline-fixes` at `336dd98` (Step 2 recipe).
- **Step 2 cloud run artifacts (local)**: `runs_warm_start_combined_recipe/20260514_122701/` (metrics, manifest, suggestions, training.log). Checkpoints are still on cloud.
- **Recipe**: `configs/warm_start_combined_recipe.yaml` (anchor n=40).
- **Smoke recipe**: `configs/wave2_wiring_verify.yaml` (added in Step 1).
- **Runbook**: `CLOUD_RERUN_RUNBOOK.md`.
- **Wave 1 preflight tests**: 10 files in `yinsh_ml/tests/test_*.py`.
- **Step 1 contract regression tests**: `test_mcts_backprop_perspective.py` (the 3 `last_root_value` contract tests added in `3cb047f`).
- **SSH alias**: `cloud` → `69.176.92.111:52868` (updated in `~/.ssh/config` 2026-05-14).

---

## How to start the next session

1. Open `WAVE2_STEP_2_POSTMORTEM.md` and read findings F1-F5 + the bug section.
2. Open `WAVE3_BRANCHES.md` and check the pre-flight at the bottom.
3. Confirm §8 EMA evals finished on cloud.
4. Confirm the supervisor.py:2028 bug fix landed.
5. Kick off Branch A (cheap, settles the EMA-drift hypothesis).
