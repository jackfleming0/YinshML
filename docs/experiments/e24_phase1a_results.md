# E24 Phase 1a — LR sweep results — **LR IS NOT THE LEVER** (2026-06-04)

Three-arm LR sweep continuing self-play from `iter1_ema` (warm-start), one
variable = `trainer.lr` ∈ {3e-5, 1e-4, 3e-4}; everything else the champion recipe
(15ch, 200 sims, 400 games/iter, 3 iters, cosine+warmup, **0.55 gate + revert**).
~6 h/iter (CPU-bound self-play), ~3 days total. Value head tracked per-iter with
`value_head_calibration.py`; H2H vs a frozen iter1_ema with `e24_summarize_h2h.py`.

## Verdict
The value head's AUC **never lifts off 0.737 at any LR**. Higher LR only erodes it
faster (AUC down, v_std compressed) and, at 3e-4, **collapses gameplay**. E24's
"the value head is frozen because our LR was too timid" hypothesis is **falsified**:
gentle LR treads water, hot LR collapses, **none improve**. LR is not the lever —
the bottleneck is the training TARGET (mirror self-play → ~50/50 noisy value labels
→ no discrimination gradient).

## Value head — human corpus (PRIMARY; baseline reproduces the known 0.737)
```
arm        iter   Brier↓   sign-acc↑   corr↑    AUC↑    v_std
baseline    —     0.6704     0.646     0.378    0.737   0.391
lr3e-5      0      0.6700     0.648     0.378    0.737   0.391
lr3e-5      1      0.6698     0.646     0.376    0.737   0.376
lr3e-5      2      0.6720     0.645     0.367    0.736   0.319
lr1e-4      0      0.6706     0.648     0.378    0.737   0.393
lr1e-4      1      0.6722     0.649     0.367    0.736   0.316
lr1e-4      2      0.6787     0.653     0.356    0.732   0.291
lr3e-4      0      0.6704     0.646     0.376    0.737   0.374
lr3e-4      1      0.6784     0.648     0.359    0.729   0.277
lr3e-4      2      0.6899     0.651     0.342    0.724   0.239
```
Monotonic in LR: AUC end 0.736 / 0.732 / **0.724**; v_std end 0.319 / 0.291 /
**0.239**. More LR → more erosion, more compression toward the mean.

## H2H vs frozen iter1_ema (color-balanced, n=60/iter)
```
lr3e-5:  45.0 -> 48.3 -> 51.7 %   slope +3.3 pp/iter   [noise — CIs bracket 50%]
lr1e-4:  53.3 -> 50.0 -> 46.7 %   slope -3.3 pp/iter   [noise — opposite direction]
lr3e-4:  53.3 -> 38.3 -> 31.7 %   slope -10.8 pp/iter  [COLLAPSE — 19-41 by iter-2]
```
The gentle arms' ±3.3 are noise (4-game swings). **3e-4's −10.8 is the one
significant gameplay signal in the sweep — and it's a collapse.** Anchor MCTS
stayed 40/40 vs depth-1 heuristic throughout (no loss of basic competence; the
damage is specifically in value discrimination / fine play).

## The engine-corpus footnote (OOD — do not use for absolute numbers)
The `gen_engine_labeled_corpus.py` corpus gave baseline iter1_ema **AUC 0.575 /
Brier 1.086** (worse than blind), because its *positions* come from myopic
HeuristicAgent play the net never trained on. The independent *labeler* idea was
sound; generating the *positions* from heuristic play was the flaw. **Trap to
avoid:** on this corpus, 3e-4 had the *lowest* Brier (1.014) yet the *worst*
gameplay — because v_std compressed (predictions shrink toward 0), which
mechanically lowers Brier on a near-blind corpus. "Best Brier" there = the
evaluator giving up, not improving. A correct engine corpus relabels
REPRESENTATIVE positions with strong-engine outcomes (TODO).

## In-loop gate behavior
Every degrading iteration was rejected by the 0.55 Wilson gate and reverted
(e.g. 3e-5 iter-2 ELO 1486 vs best 1531; 1e-4 0.458; 3e-4 0.480). Gating prevented
*runaway* collapse and held each gentle arm near iter-1 ≈ champion — but it cannot
*manufacture* a gain the loop has no signal to produce.

## Decision
The E24 LR ladder **terminates here**. Do NOT run Phase 1b (extend) or Phase 2
(anti-forgetting) — neither LR nor forgetting is the lever; anti-forgetting would
*preserve* 0.737 but can't push it up. Redirect to the **value-TARGET** path:
**E21** ensemble-teacher distillation (a target that exceeds the student) and/or a
value-head/architecture change (A4 regression head, decisive-signal targets).
**E18** (deploy symmetric MCTS) remains the cheap free win.

**Scope guard (avoid the prior overclaim):** this tested *continuation from
iter1_ema + mirror self-play + real LR, gated*. It does NOT test from-scratch
self-play or a non-mirror/decisive target — those remain open.
