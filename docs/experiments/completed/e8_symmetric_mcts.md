# E8 — Symmetric MCTS at inference (D2 averaging) [= L3a]

**Status:** DONE: VALIDATED & deployed (code-complete, shipped to analysis board)
**Date(s):** proposed 2026-05-29; built/validated overnight 2026-05-29 → 2026-05-30; shipped 2026-05-31
**Cost:** 4× network forward at each MCTS leaf (well within deployed_sampled budget); ~50 LOC + ~1h wall to build
**Branch / artifacts:** reference impl `scripts/measure_symmetric_openings.py::symmetric_search()` (also `symmetric_search_batch`); deploy impl `analysis_board/server.py::_symmetric_search_batch` (sync + async paths), default-on via env `YNS_SYMMETRIC_MCTS`, UI toggle, effective-4×-budget async routing; commit 09a6d86 (ship); branch `policy-symmetry-fixes`. L3a in the production-recipe layering.

## Description

At each MCTS root expansion, evaluate the network on all **4 symmetric variants** of the position (D2 Klein 4-group: identity + 180° rotation + 2 reflections), then **average policy + value across the variants** before MCTS uses them. Guarantees the inference-time output respects board symmetry regardless of policy-head weight asymmetries.

**Why this matters:** iter1_ema's A5 72% / A2 2.5% slot-1 asymmetry is the smoking gun for MCTS path-dependence (A2 is the horizontal-reflection partner of A5; in a D2-symmetric game a genuinely optimal A5 should be played at equal rates with its 3 symmetry partners — roughly 18% each). Even after fixing the policy head (dropout patch), any residual asymmetry in the trained weights could still produce symmetry-breaking convergence. Symmetric MCTS makes the inference-time output mathematically symmetric.

Standard in AlphaZero implementations (Leela, KataGo) — not a novel invention, just not previously wired in this codebase.

## Outcome

**VALIDATED overnight 2026-05-29 → 2026-05-30. Two clean benefits → SHIP IMMEDIATELY.**

1. Breaks ~75% of MCTS path-dependence (orbit partners activate).
2. Improves main-game WR via noise reduction in leaf evaluations (+6 pp for iter1, **+22 pp for dropout+LS**).

**iter1_ema A5 orbit ({A5, K7, E1, G11}):**

| Position | Vanilla iter1 (n=200) | Symmetric iter1 (n=50) |
|---|---|---|
| A5 | 72.0% | 40.0% |
| K7 | 0% | 8.0% |
| E1 | 0% | 6.0% |
| G11 | 1.0% | 0% |
| A5 share of orbit | 99% | 74% |

~75% of path-dependence broken; A5 dropped from 72% → 40%. White WR 48% → 54%.

**Dropout+LS model: white WR 24% → 46% under symmetric MCTS** — a massive +22 pp bonus from noise reduction in MCTS leaf evaluations even though the policy is over-concentrated. This bonus was **not predicted**.

The 4× evaluation cost per move is well within the deployed_sampled budget. Even without retraining, gives users a less weird opening AND a stronger main game. Wired into the deploy inference path; **code-complete, not yet deployed** (needs `git push` + `yinsh-redeploy`). Validated on deployed iter1_ema: opening D6 concentration **0.857 → 0.214** (commit 09a6d86).

## Details

E8 and L3a are the same intervention (E8 = the backlog entry, L3a = its slot in the layered production recipe).

**The symmetry / path-dependence argument (Jack's reframing, 2026-05-29 ~17:50 UTC):** YINSH has D2 board symmetry. iter1_ema's slot-1 distribution shows A5 72.0% vs A2 2.5%. A2 is the horizontal-reflection partner of A5. In a D2-symmetric game, if A5 is genuinely strategically optimal, the model should play A2 (and the other two symmetry partners K7, K10) at ~equal rates — roughly 18% each. The observed 72%/2.5% asymmetry is *physically impossible* if the model had learned any real strategic principle. It can only happen via the FPU + uniform-policy MCTS-stall mechanism diagnosed in P3. **The symmetry breakage is the smoking gun for path-dependence.** This drove the "don't force humans, set up for exploration" recipe and motivated symmetric MCTS as the inference-time guarantee against another A5-style asymmetric collapse.

**Implementation notes:**
- Overnight build (2026-05-29 → 2026-05-30, ~8h wall): averaged policy + value across 4 D2 transforms per move; 50 games each of iter1_ema and dropout+LS via `scripts/measure_symmetric_openings.py`.
- Deploy adapter: `analysis_board/server.py::_symmetric_search_batch`, both sync and async paths, default-on via `YNS_SYMMETRIC_MCTS`, with a UI toggle and effective-4×-budget async routing.

**Residual 25% (orbit-internal asymmetry after symmetric MCTS):** A5 still ~3× over orbit average (20 vs 4/3/0 for K7/E1/G11). This residual was investigated by E11/E14 (and the deferred E12/E13/E15/E17) and traced to **H_W: asymmetric network weights**. The fix is the E16 symmetric-weight regularizer. See [[e11_weight_symmetry_check]], [[e14_augmenter_integrity]], [[e16_symmetric_weight_reg]].

**Role going forward:** Symmetric MCTS (L3a) stays as a deploy-time noise-reducer even once the network is made symmetric by E16. It is a pure win with no retraining.

## Provenance & links

- Spun out as a new experiment from the 2026-05-29 session; validated overnight into 2026-05-30 morning.
- Source snapshots: 2026-05-29 ~13:30 UTC, 2026-05-29 final production recipe, 2026-05-30 morning results, 2026-05-31 ~14:00 UTC recovery snapshot.
- Related: [[e6_hvh_continued_pretrain]], [[l1_l2_dropout_labelsmoothing]] (L1/L2 partners in the recipe), [[e11_weight_symmetry_check]], [[e14_augmenter_integrity]], [[e16_symmetric_weight_reg]] (residual-25% chain), [[e18_symmetric_mcts_deploy]] (downstream deploy work), E9/E10 (exploration/corpus layers).
- Cross-doc: `analysis_board/multiplayer/EXPERIMENT_opening_theory.md`, `YNGINE_BENCHMARK_RESULTS.md`.
