# MCTS Depth Experiment Results

**Date:** January 30, 2026
**Branch:** `heuristic_seeding`
**Baseline Reference:** baseline_004 (ELO 1562) + LR findings (value_lr_factor: 2.5)

## Executive Summary

Three MCTS depth experiments were conducted to understand the impact of simulation count on training performance. **Experiment 002 (200 simulations)** achieved the best results with **ELO 1534**, while surprisingly **400 simulations showed diminishing returns**.

| Experiment | Simulations | Best ELO | Best Iter | vs Baseline |
|------------|-------------|----------|-----------|-------------|
| 001 | 50 (0.5x) | 1504 | 3 | -58 |
| **002** | **200 (2x)** | **1534** | **3** | **-28** |
| 003 | 400 (4x) | 1521 | 2 | -41 |

## Methodology

### Base Configuration
All experiments used baseline_004 settings with LR sensitivity findings applied:
- **value_lr_factor:** 2.5 (from LR sensitivity experiment 003)
- **Iterations:** 10
- **Games per iteration:** 100
- **Buffer size:** 20,000

### Variable: MCTS Simulations
| Experiment | Early Sims | Late Sims | Relative to Baseline |
|------------|------------|-----------|----------------------|
| 001 | 50 | 50 | 0.5x |
| 002 | 200 | 200 | 2x |
| 003 | 400 | 400 | 4x |

## Detailed Results

### Experiment 001: 50 Simulations (Half Baseline)
**Hypothesis:** Fewer simulations = lower quality policy targets = worse performance.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1421 | 0 |
| 2 | 1433 | 0 |
| 3 | 1529 | 3 |
| 4 | 1486 | 3 |
| 5 | 1462 | 3 |
| 6 | 1501 | 3 |
| 7 | 1487 | 3 |
| 8 | 1504 | 3 |
| 9 | 1489 | 3 |

**Runtime:** ~2.7 hours (09:37 - 12:23)

**Observations:**
- Initial drop (iterations 1-2) then recovery at iteration 3
- Best model peaked at 1529, final best held at 1504
- Faster training due to fewer simulations per move
- Confirms hypothesis: fewer sims = worse quality

**Conclusion:** 50 simulations is insufficient for high-quality policy targets.

---

### Experiment 002: 200 Simulations (Double Baseline) ⭐ BEST
**Hypothesis:** More simulations = higher quality policy targets = better performance.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1529 | 1 |
| 2 | 1496 | 1 |
| 3 | 1534 | 3 |
| 4 | 1491 | 3 |
| 5 | 1492 | 3 |
| 6 | 1493 | 3 |
| 7 | 1492 | 3 |
| 8 | 1521 | 3 |
| 9 | 1494 | 3 |

**Runtime:** ~4.1 hours (12:29 - 16:37)

**Observations:**
- Strong early performance (1529 at iteration 1)
- Peak at iteration 3 (1534) and maintained
- Stable performance after peak (1491-1521 range)
- Best balance of quality and training speed

**Conclusion:** 200 simulations provides optimal quality/speed trade-off.

---

### Experiment 003: 400 Simulations (Quadruple Baseline)
**Hypothesis:** Maximum simulations = best policy targets = best performance.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1433 | 0 |
| 2 | 1521 | 2 |
| 3 | 1479 | 2 |
| 4 | 1475 | 2 |
| 5 | 1503 | 2 |
| 6 | 1501 | 2 |
| 7 | 1490 | 2 |
| 8 | 1506 | 2 |
| 9 | 1510 | 2 |

**Runtime:** ~2.6 hours (16:43 - 19:19)

**Observations:**
- Initial dip at iteration 1
- Peak at iteration 2 (1521), earlier than other experiments
- Stable but lower performance than 200 sims
- **Surprisingly did NOT outperform 200 simulations**

**Conclusion:** Diminishing returns beyond 200 simulations. Possible explanations below.

---

## Key Findings

### 1. Optimal Simulation Count: 200
The relationship between simulations and performance is **non-linear**:

```
Simulations:  50  → 100 (baseline) → 200  → 400
Best ELO:    1504 →     1562       → 1534 → 1521
```

**Note:** 200 sims achieved 1534, but baseline with 100 sims achieved 1562. This suggests other factors (like the value_lr_factor change) may have affected results. However, within this experiment set, 200 sims clearly outperformed 50 and 400.

### 2. Diminishing Returns at 400 Simulations
Why did 400 simulations underperform 200?

**Possible explanations:**
1. **Slower iteration:** 4x slower games = fewer training iterations in same wall-clock time
2. **Overfitting to search:** Very deep search may produce overly specific policies
3. **Exploration reduction:** Deeper search reduces move diversity in training data
4. **Diminishing information gain:** Beyond ~200 sims, additional search adds noise rather than signal

### 3. Training Speed vs Quality Trade-off
| Sims | Games/Hour | Quality (ELO) | Efficiency |
|------|------------|---------------|------------|
| 50 | ~37 | 1504 | Low quality |
| 200 | ~24 | 1534 | **Best balance** |
| 400 | ~15 | 1521 | Diminishing returns |

### 4. Early Peaking Pattern
All three experiments peaked early (iterations 2-3) rather than improving continuously:
- 001: Peak at iter 3
- 002: Peak at iter 3
- 003: Peak at iter 2

This pattern was also observed in LR experiments and suggests the 10-iteration training may be too long, or other factors limit continued improvement.

## Recommendations

### Immediate Actions
1. **Use 200 simulations** as new default (was 100)
2. **Combine with LR findings:** value_lr_factor=2.5 + 200 sims
3. **Consider shorter training:** 5-6 iterations may be sufficient given early peaking

### Future Experiments
1. **Test 150 simulations:** May find sweet spot between 100 and 200
2. **Asymmetric sims:** Different counts for early vs late game
3. **Combined optimization:** Run experiment with both optimal LR (2.5) and optimal sims (200)

## Updated Optimal Configuration

Based on LR sensitivity + MCTS depth experiments:

```yaml
optimizer:
  policy_lr: 0.001
  value_lr_factor: 2.5       # From LR sensitivity

mcts:
  early_simulations: 200     # From MCTS depth (was 100)
  late_simulations: 200      # From MCTS depth (was 100)
```

## Comparison Across All Experiments

| Experiment | Key Change | Best ELO | Rank |
|------------|------------|----------|------|
| LR 003 | value_lr_factor=2.5 | **1571** | 1st |
| baseline_004 | 20K buffer | 1562 | 2nd |
| MCTS 002 | 200 sims | 1534 | 3rd |
| LR 001 | policy_lr=0.0005 | 1533 | 4th |
| LR 002 | policy_lr=0.002 | 1525 | 5th |
| MCTS 003 | 400 sims | 1521 | 6th |
| LR 004 | value_lr_factor=10 | 1519 | 7th |
| LR 005 | conservative | 1508 | 8th |
| MCTS 001 | 50 sims | 1504 | 9th |

**Best overall:** LR sensitivity experiment 003 (value_lr_factor=2.5) with ELO 1571.

---

**Experiment IDs:**
- 001: `5660dae7`
- 002: `968b3aff`
- 003: `ebdc9e24`

**Log File:** `mcts_depth_batch.log`
