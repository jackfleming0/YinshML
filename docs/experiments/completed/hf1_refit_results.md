# HF-1 — re-fit the 6 production weights — **WORSE / NOT_STRONGER** (2026-06-03)

**Verdict:** re-fitting the 6 production weights on game outcomes makes the
negamax agent **significantly worse**, not better.

## What ran

- Generated 300 epsilon-greedy (ε=0.15) baseline self-play games on this box
  (no parquet corpus present), 18,500 labeled main-game positions, labeled by
  whether the side-to-move ultimately won.
- Fit the 6 production weights with numpy logistic regression, then **rescaled
  each phase to the baseline's L1 budget** so the re-fit differs only in
  *allocation*, not absolute eval magnitude (a pure reallocation test).
- A/B vs baseline, 300 games, depth 1, parallel.
- Artifacts: `docs/experiments/refit6_selfplay.json`.

## Result

```
refit6 vs baseline (depth1, 300g): 62-238-0  winrate=0.207  CI[0.16,0.26]  elo=-234  sig=YES
```

The re-fit allocation is wildly different from baseline — it zeroes out
`connected_marker_chains` and `board_control` and pours the budget into
`potential_runs_count` / `ring_spread` / `ring_positioning`. That allocation
**loses 79% of games**.

## Why (the methodological lesson)

**Outcome-correlation ≠ good move-selection weight.** Logistic regression finds
features that *predict who wins*. But in self-play data the eventual winner has
accumulated more of almost everything by the endgame, so "high potential_runs →
winning" is largely **reverse causation** (winning produces the high feature
value, not vice-versa). Using such coefficients as a *greedy* evaluation to
maximize each move over-optimizes winner's-end-state correlates and plays
worse. The hand-tuned baseline — validated for play, not for outcome
prediction — beats it decisively.

This is the same reverse-causation that AlphaZero avoids with MCTS-improved
policy targets and bootstrapped value, and it reinforces:

- The Phase 1 finding (heuristic-feature *form* is the bottleneck).
- That naive "fit weights to 100k-game outcome correlations" is itself a
  suspect recipe — the production defaults play well *despite* their stated
  correlation origin, suggesting they were tuned beyond raw correlations.

## Caveats

- Fit data is self-generated weak (depth-1, ε-greedy) play → mildly circular and
  low-quality labels. A fit on **strong/external** games (champion, yngine, human)
  might allocate differently. But the reverse-causation problem is intrinsic to
  outcome-fitting and would persist; a better objective (TD/bootstrapped value,
  or MCTS-visit targets) is the real fix — i.e. the learned-value direction.

## Decision

HF-1 closes negative. Combined with Phase 1, the heuristic-tuning path (new
features *and* re-weighting) is unproductive. Leverage is in learned/nonlinear
evaluation (HF-2 unconfounded 15ch-vs-6ch read; HF-3 learned value).
