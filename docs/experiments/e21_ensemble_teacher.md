# E21 — Model ensembling (averaged P/V), primarily as a TEACHER

**Status:** QUEUED / PARKED (behind the E19 verdict)
**Date(s):** spun out of a 2026-06-02 discussion (Jack), while E19 Arm A's depth result came in flat
**Cost:** Cheap probe (mode b) is ~1h of H2H; the teacher (mode a) is a full training run, gated like everything else on a real learning rate.
**Branch / artifacts:** reuses the E8 harness — `server.py::_symmetric_search_batch` (already averages one net's policy+value over the 4 D2 transforms at each leaf). All candidates 15-ch / 7433-move-space / value∈[-1,1].

## Description
**Why now:** E19's read is that depth couldn't beat iter1_ema because the binding constraint is *value-head resolution* — the value head is only ~15% better than baseline (P2: Brier 0.66 vs 0.78), so 4× search just walks a wider tree to the same blurry verdict (depth amplifies evaluation, can't exceed it). Averaging several **decorrelated** evaluators is the textbook fix for a noisy estimator (variance reduction + calibration — deep-ensembles). So ensembling aims at the exact wall depth bounced off, where depth could not.

**Mechanics are nearly free — reuse the E8 harness.** Symmetric MCTS (deployed) already averages ONE net's policy+value over the 4 D2 transforms at each leaf. Ensembling is the same averaging on a different axis: average over N models (optionally × the 4 transforms). All current candidates are 15-ch / 7433-move-space / value∈[-1,1], so policies average cleanly (mean the softmax probs over the valid-move support, renormalize — exactly what the symmetric path does) and values just average. Generalizing the harness from "4 transforms of 1 model" to "N models × 4 transforms" is a small change; trivial to prototype at inference on the analysis board.

**Two modes — the teacher is the prize, the player is a probe:**
- **(a) Ensemble-as-TEACHER / distillation (the real lever).** Use the ensemble's averaged P/V as the self-play training *target* and distill it into a single net (ensemble/policy distillation, "born-again" nets). This manufactures the "target genuinely exceeds the student" that deeper search was a proxy for but couldn't deliver — because the evaluator, not the search, was the limit. The distilled single net is what we'd then H2H vs iter1_ema. **This is the version that could actually break the plateau.**
- **(b) Ensemble-as-PLAYER (test-time augmentation, cheap probe).** Average at inference and play the result. As a strength play this is weak: averaging a champion with a *materially weaker* sibling (sym15, 27%) usually regresses toward the mean (ensembles beat the best member only when members are co-equal with decorrelated errors). But it's a cheap, informative experiment: **does ensemble{iter1_ema, sym15} beat iter1_ema H2H?** If YES despite sym15 being weaker → hard evidence the 15ch-symmetric substrate makes *decorrelated* errors (captures something iter1_ema misses), which argues for the pivot regardless of whether the ensemble ships. If NO → ~1h of H2H confirms they're correlated. Either way you learn something.

## Outcome
Not yet concluded — QUEUED / PARKED. **Sequence:** do NOT act until the [[e19]] Arm-A-vs-Arm-B slope verdict is in and digested — if Arm B reveals the substrate lever, that reshapes what's worth ensembling/distilling. The decision gate: the cheap mode-b probe (~1h) tells you whether iter1_ema and sym15 make decorrelated errors; the mode-a teacher run is gated like everything else on a real learning rate. On the post-E24 lever board (2026-06-07) E21 is **Lever E — ensemble-teacher:** still viable as [[e26]]'s teacher, but needs *decorrelated* members we may lack; the ~1h mode-b probe is the cheap test. [[e24]]'s falsification redirected the search toward the value-TARGET path, explicitly naming E21 ensemble-teacher distillation as a candidate.

## Details
**Reasons to not believe / watch:**
- **Diversity is the whole game, and the obvious candidates are correlated.** armA = iter1_ema + 3 barely-moving iters; armB = sym15 + 3. So {iter1,armA} and {sym15,armB} are near-duplicates — the "4-model average" collapses to **~2 independent opinions** (iter1_ema family vs sym15 family). Averaging a model with its own light continuation buys ~no variance reduction. Real decorrelation must come from a different *axis*: the **6ch↔15ch** encoding split (different features → different mistakes — the cheapest source we have lying around), different seeds/corpora/value-head-types, or the equivariant net (E17).
- Player-mode likely lands *below* the champion if members are unequal (see above).
- Distillation adds a training stage with its own failure modes (and must obey the NaN-target guard from the [[e19]] incident — averaging would actually *mask* a single model's NaN, a mild robustness bonus).
- Ensemble inference is N× (or N×4×) forward cost per leaf — fine at the deployed budget, heavier in training.

**Discipline:** still cashes out only in H2H vs the fixed champion iter1_ema (R1); change one thing at a time.

## Provenance & links
- Source snapshots: 2026-06-02 discussion (Jack) scope; 2026-06-07 post-E24 lever board (Lever E); 2026-06-09 ([[e26]] reaim notes E21 as a still-viable teacher).
- Related: [[e19]] (its flat depth verdict is the parking-gate and the motivation), [[e08]] (E8 harness reused), [[e24]] (redirected toward value-target / ensemble-teacher path), [[e26]] (could use the ensemble as its high-budget teacher).
