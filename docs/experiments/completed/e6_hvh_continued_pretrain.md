# E6 — Continued pretrain on H-vs-H full-game data

**Status:** RETIRED (policy fix works, strength regresses)
**Date(s):** 2026-05-28 (kicked off late evening) → 2026-05-29 (completed)
**Cost:** local continued-pretrain (no cloud); E7 successor estimated ~$200 / ~16h cloud
**Branch / artifacts:** corpus `hvh_full_game_15ch.npz` (107K H-vs-H positions); base `models/supervised_2026-05-27/best_supervised.pt`; dry-run output compared directly against the later `dropout-patch` A/B; implementation reference `dry_run_dropout_plus_ls.py` (the validated dry-run used the supervised hard-target CE)

## Description

Continued pretrain on human-vs-human (H-vs-H) full-game data to fix the **data side** of the placement pathology. The deployed engine played an anomalous 4-of-5 ring-cluster opening (friend-tester feedback 2026-05-28). Root cause traced: the yngine pretrain corpus has **uniform-random placement targets** (F6 at 1.3% = exactly uniform), so the supervised model faithfully learned a uniform placement policy; MCTS + FPU + uniform policy → a "first-visited child wins all visits" stall; the modal opening (A5 for iter1_ema, D6 for supervised) is **path-dependence, not strategic preference**.

E6's hypothesis: replacing uniform-random placement targets with real human placement targets (continued pretrain on H-vs-H) will teach a non-uniform placement policy and recover the human-modal F6 opening.

## Outcome

**RETIRED as a production path.** The policy fix works but strength regresses.

- F6 modal **achieved**: 27% on the policy head, 80% in self-play.
- H2H vs iter1_ema = **8-22 (27% WR, Wilson 95% CI [0.14, 0.44])** — a decisive loss. (Reported elsewhere in the timeline as 22-8 / white WR regressed to 38% in self-play.)
- Continued pretrain on H-vs-H main-game **forgot yngine-learned tactics**. The data fix alone was not enough.

**Result is partially confounded by the Dropout(0.3) architecture cap** (see [[l1_l2_dropout_labelsmoothing]]). E6 did fix the weak-policy issue at the empty board (F6 27% policy output), but the underlying architectural limit means the policy head could still only produce ~25% peak — not the sharp 60-90% distributions humans actually play. Some of E6's main-game strength regression may also be the dropout limiting how well the model can sharpen on new training signal.

## Details

E6 was the first attempt at the placement pathology and seeded the whole 2026-05-29 plateau investigation. The dry-run setup (107K H-vs-H positions, 5 epochs in the original; the dropout-patch A/B re-ran the identical recipe at 3 epochs, LR 5e-5, weight decay 1e-3, 4× placement oversample) became the direct A/B baseline for the L1 dropout patch.

**E6 vs the dropout-patched re-run (direct A/B):**

| Metric | Before training | E6 (5 epochs) | Dropout-patched (3 epochs) |
|---|---|---|---|
| Empty-board peak | 1.39% | 27.2% | 31.3% |
| Empty-board entropy | 4.44 | ~3.5 | 3.39 |
| Multiplier vs uniform | 1.04× | 20.3× | 22.4× |

The dropout-patched policy head sharpened **past E6's epoch-5 ceiling in just epoch 2** (F6 28.9% at epoch 2 of the patched run vs F6 27.2% at epoch 5 of E6) — decisively confirming that Dropout(0.3) was the architectural cap on policy sharpness, and that E6's "weak policy" symptom was largely architectural, not purely a data problem.

**Successor experiments spun out (E6 retired in favor of these):**
- **E7 (split-corpus pretrain)** — the leading next candidate. Per-game placement sampled from human-replay + random + BGA-marginal; main-game phase plays out with iter1_ema for both sides. Cleanly separates the two fixes: human placement targets where wanted, iter1-quality tactics where wanted. Cost ~$200 / ~16h cloud for v1. **Note:** the dropout-fix finding fully eclipsed E7's claimed value — no amount of better corpus quality matters if the policy head architecturally cannot learn from it. E7 stays on the backlog as a *future* quality lift after the architecture fix lands, not the next priority.
- **E7b (cross-teacher: iter1 + supervised + heuristic on different sides)** — on standby. If P3 shows self-play signal is near zero, cross-teacher is the only version that can produce genuinely disagreeing training targets.

**Load-bearing benchmark for E6/E7's premise:** the 2026-05-28 yngine benchmark — iter1_ema 17-0-0 SPRT vs yngine at both MCTS-200 and MCTS-800 — confirms iter1 > yngine teacher quality (see `YNGINE_BENCHMARK_RESULTS.md`).

## Provenance & links

- Full diagnosis in `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`.
- Triggered by friend-tester feedback 2026-05-28 (anomalous opening in deployed model).
- Source snapshots: 2026-05-29 ~13:30 UTC plateau-diagnostics snapshot.
- Related: [[l1_l2_dropout_labelsmoothing]] (the architecture cap that confounds E6), [[e8]] (symmetric MCTS), E7 / E7b (successors, on backlog).
- The plateau diagnostics P1/P2/P3 gated the E7 commitment — see [[l1_l2_dropout_labelsmoothing]] for full P1/P2/P3 numbers.
- Cross-doc: `YNGINE_BENCHMARK_RESULTS.md`, `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`.
