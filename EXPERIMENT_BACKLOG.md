# Experiment Backlog

The single index for YINSH training experiments: **what we should run next** and
**what we already learned**. Every experiment is one entry here — a short
description, a short outcome — with the full methodology, evidence, tables, and
operational lessons in a referenced detail file under `docs/experiments/`.

- **Queued / active** experiments → detail files in `docs/experiments/`.
- **Completed** experiments (done / retired / dropped) → detail files in
  `docs/experiments/completed/`.

Complements `VOLUME_PRETRAIN_RESULTS.md` (chronological session log) and
`TRAINING_REFACTOR_PLAN.md` (the ceiling-raising roadmap). When a session ends
without a clear next step, read the **Current state** section below first.

## How to maintain this doc

- **One entry per experiment, here.** Each entry is: an `### ID — Name` header
  with a status tag, a one-line **What**, a one-line **Outcome**, and a `→`
  link to its detail file. Keep entries tight — depth lives in the detail file.
- **New hypothesis** → write the detail file in `docs/experiments/` and add a
  `Pending —` entry under *Active / queued* BEFORE running anything. Forces
  explicit cost-vs-info-gain thinking.
- **Experiment runs to completion** (positive, negative, or inconclusive) →
  fill in the detail file's Outcome, `git mv` it to `docs/experiments/completed/`,
  and move its entry to *Completed* with the verdict. Don't delete the rationale
  — the thinking that led to a run is durable calibration knowledge.
- **Detail-file template** lives at the bottom of this doc.
- **Re-rank / re-sequence** when a new finding promotes or demotes prior entries.

---

## Current state (2026-06-09) — the binding constraint is the value head, and it's OUT

Four swings at the post-iter1 plateau — **E19** (search depth), **E22**
(cross-teacher decisiveness), **E24** (LR sweep) — all came back NOT_STRONGER.
Every one poked the *mirror-continuation self-play loop while keeping the same
value head*. So we ran the cheap diagnostic (**E25**) to find the real binding
constraint, and it resolved decisively:

- **The value head is at an intrinsic ceiling, not a data/noise artifact.** On
  iter1's OWN strong self-play positions the value-AUC reads **0.663** — *below*
  the 0.737 human-corpus number, killing the "human-noise inflated the ceiling"
  hypothesis. Strong play makes outcomes less position-determined; ~0.74 is
  plausibly near the Bayes ceiling for static YINSH value.
- **No stronger teacher exists.** Pecking order **iter1_ema ≫ yngine > sharkdp >
  HeuristicAgent** — our NN dominates the whole classical field, so a
  cross-engine pretraining corpus would be a *downgrade*. The "better teacher
  corpus" lever is dead.
- **Policy is the load-bearing head.** MCTS ablation: flatpolicy loses 0.90,
  blindvalue loses 1.00 — both heads necessary, and search *manufactures* a
  better policy (visit counts ≫ raw prior) even though it can't exceed the
  value eval.

**Implication — stop chasing value discrimination and corpus diversity.** The
surviving levers are:
- **Lever B — test-time compute (free, ship now):** **E18** (deploy symmetric
  MCTS, +6–22 pp, no training) + simply playing at higher MCTS budget.
- **Lever A — search-improved *policy* distillation:** **E26** (reaimed) —
  distill iter1 searched at very high budget into the net; the one surviving
  target-improvement lever.
- **Lever D — architecture / capacity:** **A4** (scalar regression value head),
  a bigger trunk, or **E17** (D2-equivariant net).
- **Lever E — ensemble-teacher:** **E21** — viable as E26's teacher *if* we can
  find decorrelated members; the ~1h mode-b probe is the cheap test.

**Recommended sequence:** ship **E18 + higher play-time sims** (free) → run
**E26** policy-distillation → escalate to **architecture/capacity** (A4 / E17)
only if E26 stalls. **Discipline is unchanged: the only verdict is H2H vs the
FIXED champion `iter1_ema`; change one variable at a time; cheap diagnostics aim
the expensive swing.**

> **Meta-finding that frames everything:** nothing built since has beaten
> `iter1_ema_2026-05-27`. Not E6, not the dropout-fix lineage, not the symmetry
> run, not E19/E22/E24. The plateau is not "iter1→iter2 won't climb"; it is
> "every targeted fix so far failed to produce a model stronger than iter1_ema."

---

## Active / queued

### Plateau-break levers (current focus)

#### E18 — Deploy symmetric MCTS (L3a/E8) to the analysis board  `QUEUED · do-first`
**What:** Ship the validated symmetric-MCTS inference path (averages one net over the 4 D2 transforms) to the analysis board — zero new training, free strength bump.
**Outcome:** Pending — gate is just the deploy action (`git push` + `yinsh-redeploy`). Validated: D6 opening concentration 0.857→0.214, iter1 WR 48%→54%.
→ [details](docs/experiments/e18_symmetric_mcts_deploy.md)

#### E26 — High-budget-search distillation campaign  `RUNNING · reaimed to policy-distill`
**What:** Distill targets from a stronger teacher (iter1_ema at 1600–3200+ sims) into the net — search manufactures the better signal, distillation banks it.
**Outcome:** Pending — reaimed 2026-06-09 to distill the search-improved POLICY (value teacher undercut by E25). Verdict gate: distilled net beating frozen iter1_ema in H2H.
→ [details](docs/experiments/e26_distillation_campaign.md) · [box runbook](docs/experiments/e26_box_runbook.md)

#### E21 — Model ensembling (averaged P/V), primarily as a teacher  `QUEUED · parked`
**What:** Average decorrelated evaluators (N models × 4 transforms via the E8 harness); teacher-mode distillation is the real lever, player-mode a cheap probe.
**Outcome:** Pending — gate is the ~1h mode-b H2H probe (does ensemble{iter1, sym15} beat iter1?). Diversity is the whole risk; obvious candidates are correlated.
→ [details](docs/experiments/e21_ensemble_teacher.md)

#### E20 — Self-play throughput build (process-based inference server)  `BUILT · MEASURED`
**What:** Process-based GPU inference server (coalesce across worker processes) + real virtual loss (fill in-search batches). Built process-side, not threaded, because the C++ bindings only release the GIL in Bench* fns.
**Outcome (2026-06-09, 4090):** **21.9× serial / 9.5× the old process-pool ceiling** (peak 7,140 g/hr @48 workers, pure-neural); **8.65× serial on the real hybrid recipe**. At 64% of the GPU roofline; remaining ~1.6× is CPU/IPC/server-bound (batch plateaus ~127, more workers regress). Reinvest the win as more sims/move (E25 value-target quality).
→ [details](docs/experiments/e20_throughput_build.md)

#### HF-2 — Heuristic features as network inputs (functional-form test)  `QUEUED`
**What:** Feed strategic features to the net as channels so it weights them nonlinearly — the direct test of the linear/differential-form bottleneck HF found. Mostly already built (15-ch EnhancedStateEncoder).
**Outcome:** Pending — first move is an unconfounded post-bugfix 15-ch vs 6-ch A/B; only then add a `defensive_disruption` channel (R²=0.47 independent info).
→ [details](docs/experiments/hf2_features_as_net_inputs.md)

#### HF-3 — Pivot to learned-value (AlphaZero direction)  `QUEUED · parked/strategic`
**What:** Demote the heuristic to a prior and trust the learned value head — the strategic redirect where the evidence points for real strength gains.
**Outcome:** Pending — biggest effort, overlaps the current value-head work. Revisit once that lands. See `TRAINING_REFACTOR_PLAN.md`.
→ [details](docs/experiments/hf3_pivot_to_learned_value.md)

### Exploration / corpus knobs (production-recipe layers)

#### E9 — Phase-aware exploration knobs  `QUEUED`
**What:** Per-phase Dirichlet/temperature/FPU at ring placement (alpha 1.0, epsilon 0.5, temp 1.0, fpu 0.0) to force placement diversity. ~30 LOC in `self_play.py::MCTS`.
**Outcome:** Pending — production-recipe layer L3c; stacks once the L1/L2/L3a foundation is solid.
→ [details](docs/experiments/e9_phase_aware_exploration.md)

#### E10 — Random + symmetric placement injection in corpus generation  `QUEUED`
**What:** Mix placement sources in corpus gen (40% human-replay / 20% BGA-marginal / 20% uniform / 20% symmetric-augmented), each with full 4× D2 augmentation.
**Outcome:** Pending — layer-3 "data + exploration" bundle; gives state coverage + an explicit symmetry signal once the foundation is validated.
→ [details](docs/experiments/e10_placement_injection.md)

### Branch-D candidate family (older queue — pretrain/architecture)

These were scored before the plateau diagnostics narrowed the field; A1/B1/B2/B3
have since run (see *Completed*). Kept for the priors they encode. The
**stack-rank table** below scores them on five axes (1–5, higher = better; the
Sum is a tie-breaker — read the axes for the decision you're actually making).

| ID | Experiment | Likely + | Unblocks | Info gain | Cost ($, h) | Impl risk | **Sum** | Status |
|---|---|---|---|---|---|---|---|---|
| **F1** | Audit + fix bare-`NetworkWrapper` construction sites | 5 | 5 | 3 | 5 | 5 | **23** | queued (partial) |
| **A3** | Re-pretrain 6-ch baseline at 6 epochs | 3 | 5 | 5 | 4 | 5 | **22** | queued |
| **A4** | Regression value head pretrain | 4 | 4 | 5 | 4 | 4 | **21** | queued |
| **A2** | Extend D.2 pretrain to 9-12 epochs via `--resume` | 2 | 2 | 3 | 5 | 5 | **17** | queued |
| **B4** | Disable promotion gate entirely | 2 | 3 | 4 | 4 | 4 | **17** | queued |
| **D1** | Self-play data corpus for pretrain | 3 | 4 | 4 | 2 | 3 | **16** | queued |
| **E1** | GAP-native pretrain from scratch (Path 2) | 2 | 4 | 4 | 2 | 3 | **15** | queued |
| **C1** | Branch D.3 — SE blocks | 2 | 3 | 3 | 2 | 4 | **14** | queued |

*Scoring rubric — Likely+:* 1 = expected flatline, 5 = strong mechanism prior.
*Unblocks:* 1 = one-off, 5 = gates multiple downstream. *Info gain:* 1 =
ambiguous regardless, 5 = decisively settles an open question. *Cost:* 1 = >$50
/ >20h, 5 = <$2 / <2h. *Impl risk:* 1 = high bug/waste chance, 5 = config-knob
on battle-tested paths.

#### F1 — Audit + fix bare `NetworkWrapper(device=...)` construction sites  `QUEUED · partial`
**What:** Convert the 8 remaining bare-construct-then-`load_model` sites to `NetworkWrapper(model_path=...)` so encoding auto-detection engages; add a cross-arch load test.
**Outcome:** Pending — critical path (`tournament.py`, `eval_vs_frozen_anchor.py`) already fixed; 8 enumerated scripts + the regression test still outstanding.
→ [details](docs/experiments/f1_networkwrapper_audit.md)

#### A3 — Re-pretrain 6-ch baseline at 6 epochs  `QUEUED`
**What:** Train a fresh 6-ch checkpoint to 6 epochs (matching D.2's schedule) so only encoding differs — removes the epochs-vs-channels confound in every D.2 comparison.
**Outcome:** Pending — does 6-ch-at-6-epochs alone beat `best_iter_4`? If so the baseline assumption was wrong.
→ [details](docs/experiments/a3_6ch_baseline_6epochs.md)

#### A4 — Regression value head pretrain  `QUEUED`
**What:** Swap the 3-class CE value head for a scalar tanh+MSE head; test whether the VAcc plateau is target-discretization rather than architecture. (Now also Lever D for the plateau.)
**Outcome:** Pending — SPRT vs `best_iter_4`. Positive rewrites the Branch-D architecture; negative narrows the plateau to a capacity/data ceiling.
→ [details](docs/experiments/a4_regression_value_head.md)

#### A2 — Extend D.2 pretrain to 9-12 epochs via `--resume`  `QUEUED`
**What:** Resume the 15-ch pretrain from epoch 6 for 3-6 more epochs; PAcc was still climbing (+0.009/epoch) when stopped.
**Outcome:** Pending — promote if PAcc climbs ≥0.310; kill if PAcc moves but VAcc stays flat (~0.636).
→ [details](docs/experiments/a2_extend_pretrain_epochs.md)

#### B4 — Disable promotion gate entirely  `QUEUED`
**What:** Run D.2 self-play with no gate (always promote latest) as a negative control on whether the gate helps or hurts.
**Outcome:** Pending — two-way SPRT vs `best_iter_4` and gated D.2; a worse iter_4 ⇒ the gate is net-positive.
→ [details](docs/experiments/b4_disable_promotion_gate.md)

#### D1 — Self-play data corpus for pretrain  `QUEUED`
**What:** Generate ~100K self-play games with the best D.2 teacher, capture MCTS visit/value targets, build a 15-ch corpus, re-pretrain a fresh init.
**Outcome:** Pending — SPRT vs anchor; a dramatic win ⇒ yngine raw-outcome targets were the data ceiling. Overlaps E26.
→ [details](docs/experiments/d1_selfplay_corpus_pretrain.md)

#### E1 — GAP-native pretrain from scratch (Path 2)  `QUEUED`
**What:** Train a from-scratch GAP-value-head pretrain (no spatial-head warm-start) to separate "warm-start specialization" from "GAP is fundamentally wrong."
**Outcome:** Pending — SPRT verdict distinguishes a fixable warm-start artifact from an architecture dead-end.
→ [details](docs/experiments/e1_gap_native_pretrain.md)

#### C1 — Branch D.3: SE (squeeze-and-excitation) blocks  `QUEUED`
**What:** Add channel-attention SE blocks alongside the existing spatial attention; cheap (~5K params/block), well-attested in Leela/KataGo.
**Outcome:** Pending — SPRT verdict.
→ [details](docs/experiments/c1_se_blocks.md)

---

## Completed

Reverse-chronological. Detail files in `docs/experiments/completed/`.

#### E25 — Binding-constraint diagnostic  `DONE · value head OUT`
**What:** sharkdp benchmark + clean on-distribution value-eval + policy-vs-value MCTS ablation, to find the real binding constraint before committing big compute.
**Outcome:** Value head OUT — on-distribution AUC **0.663** (intrinsic ceiling, not a human-noise artifact); ablation shows policy is load-bearing; no engine exceeds iter1. Intrinsic-ceiling check (§7.1/§10) closed it: from-scratch on 1.6M positions caps held-out AUC at **0.677** while train→0.988 → not data, not encoding. Redirect to search/policy.
→ [details](docs/experiments/completed/e25_sharkdp_value_ceiling.md)

#### E24 — Real self-play campaign — Phase 1a LR sweep  `DONE · NOT_STRONGER (LR is not the lever)`
**What:** Test whether the plateau is an artifact of cautious lr=1e-5; Phase 1a = 3-arm LR sweep {3e-5, 1e-4, 3e-4}, one variable = `trainer.lr`, tight 0.55 gate + revert.
**Outcome:** AUC never lifts off 0.737 at any LR; 3e-4 H2H collapsed 53→32%. The bottleneck is the training TARGET, not the optimizer. Ladder stops here.
→ [forward rationale](docs/experiments/completed/e24_self_play_campaign.md) · [results](docs/experiments/completed/e24_phase1a_results.md) · [runbook](docs/experiments/completed/e24_phase1a.md)

#### HF — Heuristic feature ablation (palette as linear terms)  `DONE · NULL`
**What:** Well-powered depth-1 agent ablation of all 5 experimental palette features, added one at a time to the fixed baseline-6 (CPU, 300 games/cell).
**Outcome:** NULL — every arm 0.467–0.513 WR, all CIs bracket 0.5. `ring_mobility` carries independent info (R²=0.25) yet is still null → the bottleneck is the linear/differential FORM, not the feature set.
→ [details](docs/experiments/completed/hf_feature_ablation.md) · [full results](docs/experiments/completed/phase1_feature_ablation_results.md)

#### HF-1 — Re-fit the 6 production weights  `DONE · WORSE`
**What:** Re-fit the 6 production heuristic weights from 300 self-play outcomes, rescaled to baseline's per-phase L1 budget, then A/B vs the hand-tuned baseline.
**Outcome:** WORSE — 62–238, WR 0.207, −234 Elo, significant. Outcome-correlation ≠ good move-selection weight; closes the heuristic-weight axis negative.
→ [details](docs/experiments/completed/hf1_refit_weights.md) · [results](docs/experiments/completed/hf1_refit_results.md)

#### E23 — Gap-controlled opponent league (E22 scale-up)  `DROPPED`
**What:** A pool/ladder of opponents keeping the strength gap in the decisive-but-not-saturated band as the learner climbs, each rung frozen-H2H gated.
**Outcome:** Dropped 2026-06-03 — gated on E22 climbing; E22 declined, so the league premise is moot.
→ [details](docs/experiments/completed/e23_opponent_league.md)

#### E22 — Cross-teacher self-play  `DONE · FAILED`
**What:** Make games decisive by pitting iter1_ema against weaker sym15 to sharpen the value head; dual-arm (mirror control vs cross treatment), opponent the only variable.
**Outcome:** Cross arm DEGRADES −4.5 pp/iter (51.7→33.3) vs mirror's +1.2; corrupted the policy (overfit to beating sym15), value head unchanged.
→ [details](docs/experiments/completed/e22_cross_teacher.md)

#### E19 — Plateau-break ablation ladder from iter1_ema (dual-arm depth)  `DONE · NOT_STRONGER`
**What:** Change one variable off known-good iter1_ema, H2H-gated; rung 1 = depth-only 200→800 sims, dual-arm (iter1_ema vs sym15), measuring slope not just level.
**Outcome:** Depth treads water — Arm A flat; Arm B's dropout-off head also declined → the limiter is the value TARGET, not depth or head architecture.
→ [details](docs/experiments/completed/e19_plateau_ladder.md)

#### E16 — Symmetric-weight regularizer for training loss  `DONE · prototype built (default off)`
**What:** Training-loss term penalizing KL/value divergence between a state's output and the mean of its 4 inverse-D2-transformed outputs, pulling weights into the D2-symmetric subspace.
**Outcome:** Wired into both `trainer.py` and supervised pretrain (default off); masked-KL fix + `value_weight=20` from gradient-pressure/dynamic probes. Awaits a Task-3 cloud run to validate.
→ [details](docs/experiments/completed/e16_symmetric_weight_reg.md)

#### E11 — Direct weight symmetry check (residual-25% diagnostic)  `DONE · H_W confirmed`
**What:** Run the net on all 4 D2 transforms of fixed positions, inverse-transform policy back, compare — the prime diagnostic for the residual post-symmetric-MCTS asymmetry.
**Outcome:** H_W confirmed — weights are asymmetric (value drifts to 2.8× range by move 8; policy top-1 flips in 5/6 states). The fix is E16. (Deferred siblings E12/E13/E15/E17 documented in the detail file.)
→ [details](docs/experiments/completed/e11_weight_symmetry_check.md)

#### E14 — Augmenter pipeline integrity check (residual-25% diagnostic)  `DONE · H_E ruled out`
**What:** Bytewise round-trip test of the D2 augmenter (state involution + policy forward/back) over 100 replay states, to rule out encoding-pipeline lossiness.
**Outcome:** Passed bytewise-exact — H_E ruled out, no pipeline bug. With E11 this localizes the residual 25% to asymmetric weights (H_W).
→ [details](docs/experiments/completed/e14_augmenter_integrity.md)

#### E8 — Symmetric MCTS at inference (D2 averaging)  `DONE · validated & deployed`
**What:** Average policy+value over the 4 D2 board symmetries at each MCTS leaf, forcing symmetric inference output regardless of weight asymmetry.
**Outcome:** Breaks ~75% of opening path-dependence (A5 72%→40%) plus a +6 to +22 pp WR bonus from leaf-noise reduction; shipped to the analysis board (commit 09a6d86). Deploy = E18.
→ [details](docs/experiments/completed/e8_symmetric_mcts.md)

#### L1/L2 — Dropout(0.3→0) + label smoothing  `DONE · both validated`
**What:** Remove Dropout(0.3) from the policy head (the plateau cause located via P1/P2/P3 diagnostics) and add label-smoothing CE to prevent the resulting over-confidence collapse.
**Outcome:** L1 sharpens policy 22.4× from uniform in 3 epochs (past E6's epoch-5 ceiling); without L2 it collapses to F6 100% / 6% WR — smoothing ε=0.1 fixes it.
→ [details](docs/experiments/completed/l1_l2_dropout_labelsmoothing.md)

#### E6 — Continued pretrain on H-vs-H full-game data  `RETIRED`
**What:** Continued pretrain on human-vs-human full-game data to fix the uniform-placement pathology and recover the human F6 opening.
**Outcome:** Policy fix worked (F6 modal recovered) but strength regressed — lost H2H vs iter1_ema 8-22; retired. Partly confounded by the Dropout(0.3) cap (→ L1).
→ [details](docs/experiments/completed/e6_hvh_continued_pretrain.md)

#### vs-yngine — iter1_ema sweeps yngine-MCTS-1K  `DONE · STRONGER`
**What:** First absolute-strength benchmark — deployed iter1_ema vs the external engine yngine at MCTS-1K, SPRT'd at our 200 and 800 sims.
**Outcome:** STRONGER — 17-0-0 at both settings (WR 1.000, CI95 [0.816, 1.000]); saturates yngine-1K, SPRT terminated at the 17-game minimum. Load-bearing for "iter1 > teacher."
→ [details](docs/experiments/completed/vs_yngine_benchmark.md)

#### B1+B2+B3 — stop-the-leak bundled run  `DONE · INVALIDATED → NOT_STRONGER (re-run)`
**What:** Bundled three self-play knobs — tighten gate to 0.50 (B1), lower self-play LR 5-10× (B2), 200-400 games/iter (B3) — to salvage the loop on a strong warm-start.
**Outcome:** Original run INVALIDATED (ran under a phase-weight bug); RE-RUN #2 NOT_STRONGER (WR 0.476, CI95 [0.382, 0.571]). Ceiling ≈ warm-start strength; leading explanation now ~60–70% noise.
→ [details](docs/experiments/completed/b1b2b3_stop_the_leak.md)

#### A1 — D.2 pretrain (iter_0) vs best_iter_4  `DONE · STRONGER`
**What:** Direct SPRT of the 15-ch D.2 pretrained warm-start (no self-play) against the frozen 6-ch champion, to isolate the encoding axis from the loop.
**Outcome:** STRONGER in 21 games — WR 0.905, CI95 [0.711, 0.973]; the self-play loop had destroyed ~250-300 Elo of the warm-start.
→ [details](docs/experiments/completed/a1_d2_pretrain_vs_iter4.md)

#### Branch-D.2 — 15-channel enhanced encoding  `DONE · NOT_STRONGER`
**What:** Full pipeline (6→15-ch re-encode → 6-epoch pretrain → 5-iter MCTS-200 self-play) SPRT'd vs frozen best_iter_4.
**Outcome:** NOT_STRONGER — WR 0.526, CI95 [0.470, 0.582] (lower bound below 0.50); the loop diluted ~90 Glicko. Later qualified by the phase-weight bug (see B1+B2+B3).
→ [details](docs/experiments/completed/branchD2_15ch_encoding.md)

---

## Unscoped ideas

Candidate experiments that exist only as a line item — not yet scoped into a
detail file. Promote one by writing its detail file and a `Pending —` entry above.

- **E17 — D2-equivariant network.** Bake symmetry into the architecture instead
  of averaging it at inference (E8) or regularizing toward it (E16). A real
  Lever-D candidate; deferred when E11 confirmed H_W was fixable via E16.
- **E12 / E13 / E15 — residual-25% diagnostics** (sim sweep, more games,
  augmentation-coverage audit). Deferred once E11 confirmed H_W — see the
  [E11 detail file](docs/experiments/completed/e11_weight_symmetry_check.md).
- **C2 — Deeper trunk (more blocks).** Re-pretrain needed, ~$30. Stack-rank sum 13.
- **C3 — Skip-connection value head.** ~$30. Stack-rank sum 12.
- **D2 — Yngine + self-play data hybrid pretrain.** ~20h, ~$30. Stack-rank sum 12.
- **D3 — Filter yngine corpus by decisiveness.** ~4h re-pretrain, ~$7. Stack-rank sum 14.

---

## Findings driving this backlog

The priors that shape the ranking. If any is wrong, the ranking changes.

1. **Pretrain is where the strength lives.** D.2's iter_0 (the pretrained init)
   is stronger than any later self-play iter; the MCTS-200 loop dilutes the
   warm-start rather than improving it (A1 confirmed: iter_0 beat iter_4 at
   WR 0.905).
2. **15-ch encoding moved pretrain metrics only marginally** (Val PAcc +1.4 pts,
   VAcc +0.7 pts over 6-ch) — but the 6-ch baseline was undertrained, so the
   comparison is confounded (→ A3).
3. **Value head VAcc plateaued early** (0.628→0.636 over epochs 1→6) and E25 now
   shows it's an **intrinsic ceiling** (0.663 on-distribution AUC), not a metric
   or data artifact.
4. **The loose Wilson 0.20 gate enshrined degraded models** when the warm-start
   is strong; tight 0.55 gate + revert is now the default guard (E24 confirmed
   it catches every bad iter but cannot manufacture a gain).
5. **Search depth isn't the dominant lever** (Step 2: MCTS-400 vs 200 ≈ 30-40
   Elo; E19: depth treads water vs iter1_ema). Depth amplifies the evaluator, it
   can't exceed it.
6. **Cross-architecture eval infra is fragile** — the `use_enhanced_encoding`
   flag must be plumbed through every eval path; several bare-`NetworkWrapper`
   sites remain (→ F1).

---

## Cross-references

- `VOLUME_PRETRAIN_RESULTS.md` — chronological log of branch-D sessions.
- `TRAINING_REFACTOR_PLAN.md` — the ceiling-raising roadmap (HF-3 direction).
- `YNGINE_BENCHMARK_RESULTS.md` — raw win-rates vs the external yngine engine.
- `analysis_board/multiplayer/EXPERIMENT_opening_theory.md` — the opening
  path-dependence investigation that spawned E6/E8/E9/E10.
- `TECH_DEBT.md` — known bugs / instrument-correctness issues.
- `docs/NEXT_SESSION_post_e24.md` — the post-E24 session handoff.

---

## Detail-file template

Every experiment's detail file follows this shape. Front-load the verdict; the
durable value is **what we should now believe differently**, not just what happened.

```markdown
# <ID> — <Name>

**Status:** <QUEUED | RUNNING | DONE: VERDICT | RETIRED | DROPPED>
**Date(s):** <when>
**Cost / hardware:** <if known>
**Branch / artifacts:** <branches, configs, scripts, run dirs, model paths>

## Description
<What it is, the hypothesis, the mechanism — why it might work.>

## Outcome
<DONE: verdict + key numbers + the single crucial detail that gates downstream
interpretation. QUEUED/RUNNING: "Pending —" then the decision gate and the
signal that would promote or kill it.>

## Details
<Everything else, lossless: full methodology + commands, supporting evidence,
reasons-to-not-believe, trajectory/comparison tables, confirmed/falsified
findings (✅/❌/🟡), operational lessons (bugs, infra surprises, calibration shifts).>

## Provenance & links
<Dates, related experiments ([[e19]]-style links), source snapshots, cross-doc refs.>
```
