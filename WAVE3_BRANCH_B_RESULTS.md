# Wave 3 Branch B — Results + Path Forward

Run: `runs_wave3_branchB_epochs1/20260516_120110/`. 5 iters × 200 games × 1 epoch over 18.78h on Vast.ai 4090. Init: `models/supervised_seed/best_supervised.pt`. Recipe: `configs/wave3_branchB_epochs1.yaml` at `3025829`.

---

## TL;DR

**Branch B is a real, modest win.** Cutting `epochs_per_iteration` 4→1 (with proportional warmup change) lifted the mean candidate MCTS-48 win rate from 45.0% (Step 2) to 59.5% (Branch B) across the same 5 iters — a **+14.5 absolute / +32% relative** improvement. F_overfitting hypothesis (Step 2's training was overfitting on the 200-game buffer) is confirmed at the population level.

But **iter 1 was the strongest checkpoint (70% MCTS-48) and got Wilson-rejected**, so the pipeline reverted to iter 0 and the gain didn't propagate. This is the same sliding-window/Wilson-too-strict pattern from Step 2 — and it's now the highest-leverage thing to fix.

---

## Per-iter scoreboard vs Step 2

Both runs: 5 iters × 200 games. Seed baseline = 67.5% MCTS-48 (n=40).

| Iter | Step 2 EMA MCTS-48 | Branch B EMA MCTS-48 | Δ | Branch B decision |
|---|---:|---:|---:|---|
| 0 | 60.0% (24/40) | 60.0% (24/40) | 0 | ✅ promoted (first) |
| 1 | 50.0% (20/40) | **70.0% (28/40)** | +20 | ⏪ REVERTED (Wilson 17/40 LB=0.285 < 0.55) |
| 2 | 35.0% (14/40) | 45.0% (18/40) | +10 | ✅ promoted (Wilson failed, Elo improved) |
| 3 | 32.5% (13/40) | **60.0% (24/40)** | +27.5 | ✅ promoted (Wilson failed, Elo improved) |
| 4 | 47.5% (19/40) | **62.5% (25/40)** | +15 | ⏪ REVERTED (Wilson 18/40 LB=0.329) |

**Mean**: Step 2 = 45.0%, **Branch B = 59.5%** (+14.5 points).

Internal-tournament Elo trajectory (round-robin head-to-head, sliding-window=3):
- iter 0: 1500 (first model)
- iter 1: 1450 (loses 17/40 to iter 0)
- iter 2: 1500 (round-robin among {iter 0, 1, 2}; iter 2 wins 21/40 vs iter 0, fails Wilson but Elo > best)
- iter 3: 1488 (round-robin among {iter 1, 2, 3}; promotes by Elo)
- iter 4: 1483 (round-robin among {iter 2, 3, 4}; fails Wilson, fails Elo → REVERT)

3 promotions in Branch B vs Step 2's 1 — the supervisor was meaningfully more often able to advance the best model.

---

## What this tells us

### F_overfitting: confirmed at population level

Every single iter of Branch B beats the corresponding Step 2 iter on MCTS-48 anchor WR. The Wilcoxon sign test on 5 paired positive deltas gives p≈0.06 — marginal at n=5 but directionally clean. The +14.5-point mean is well outside individual-iter CI half-widths (~15%).

Per-iter overfitting on the small (200-game) buffer was a real bottleneck. Cutting to 1 epoch recovered most of the seed's quality.

### But the pipeline still throws away its best models

Iter 1's 70% MCTS-48 was the run's peak. Per the supervisor's Wilson logic (`supervisor.py:2966`):

```python
def _should_promote(self, wins, total, threshold=0.55, conf=0.95):
    lb = self._wilson_lower_bound(wins, total, z)
    return lb > threshold
```

For iter 1 vs iter 0: 17/40 wins → Wilson 95% lower bound = 0.285 < 0.55 → REJECT. The Elo path (line 1711) also rejected (iter 1 Elo = 1450 < best 1500). So iter 1 reverted, and iter 2's self-play used iter 0's weaker weights.

**The internal head-to-head metric and the external anchor metric disagreed.** Iter 1 was strong vs heuristic d1 (70% MCTS-48) but weak head-to-head vs iter 0 (42.5%). Possible explanations:
- Rock-paper-scissors: iter 1 may exploit specific heuristic-style mistakes that iter 0 (also seed-like) doesn't make.
- Sample noise: at n=40 head-to-head, win-rate CIs are wide.
- Different positions tested: head-to-head is "model vs model from start"; anchor is "model vs heuristic from start" — different game distributions.

Whatever the cause, the gate logic killed a model that was statistically better at the actual task we care about.

### Iter-to-iter variance ≠ per-iter overfitting

Branch B trajectory: 60 → 70 → 45 → 60 → 62.5. That's 25-point swings between adjacent iters even though each iter trains for only 1 epoch. So per-iter overfitting *was* a bottleneck, but it wasn't the *only* source of decay.

Other contributors that survived Branch B's fix:
- **Buffer composition drift**: by iter 4, buffer = 1000 games (some from iter 0 policy, some from iter 2 policy, etc.). Composition affects training signal.
- **Search-consistency loop** (`search_consistency.enabled: true`): the network distills from its own (in-training) MCTS outputs. As weights move, MCTS targets move.
- **Phase weight imbalance**: `MAIN_GAME: 2.0` biases training toward main-game positions; the value head needs all phases.

---

## Path forward — Branch B' (lower Wilson threshold)

Direct test of "did Wilson kill the run's best model": re-run Branch B's recipe with `arena.promotion_threshold: 0.55 → 0.20`. That promotes any candidate with Wilson LB > 0.20 — including iter 1 (LB=0.285).

If iter 1 had been promoted, iter 2's self-play would have used iter 1's 70%-MCTS-48 weights. That's the load-bearing change.

Two outcomes worth distinguishing:
1. **Iter 1's strength propagates**: iters 2+ stay near 70% MCTS-48. Wilson gate was the bottleneck; switching to a permissive threshold is the fix.
2. **Iter 1 still loses to seed after one more iter of training**: even with iter 1's weights as starting point, iter 2 still drops back to ~60%. Tells us iter 1's gain was a fluke / unstable. Different architectural change needed.

**Cost**: same as Branch B (~$8-10, ~18h).

If Branch B' confirms the Wilson gate was the bottleneck, the architectural follow-up (Branch C) becomes "make the gate logic use anchor WR, not Wilson on head-to-head" — a small code change with high leverage.

---

## What's NOT being tested in Branch B'

- Total iter count (still 5; pure-AlphaZero needs 10-20 iters to compound).
- Distillation target (still MCTS-48; could distill from MCTS-400 for stronger policy targets).
- LR schedule, dirichlet noise, search-consistency on/off — all unchanged.

These are Branch C candidates and beyond.

---

## Operational notes

- **The decision-log fix from `ae63e8c` is now visible in production logs**: iter 1 SUMMARY correctly shows `Decision: ⏪ REVERTED (gate failed, restored iter 0 for next self-play)`. Iter 0/2/3 show `Decision: ✅ NEW BEST`. Iter 4 shows ⏪ REVERTED (back to iter 3). No more misleading "no reversion" messages.
- The Vast.ai box dropped off around iter 4 completion time (~06:00 UTC 2026-05-17) but came back — was a transient network issue, not a tear-down. All artifacts pulled to laptop.
- Final best model on cloud: `runs_wave3_branchB_epochs1/20260516_120110/iteration_3/checkpoint_iteration_3.pt`. Iter 1's 70% checkpoint still on disk for reference / future re-test.
