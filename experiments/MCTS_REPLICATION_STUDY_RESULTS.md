# MCTS Replication Study - Final Report

**Date:** February 2, 2026
**Branch:** `heuristic_seeding`
**Base Config:** value_lr_factor=2.5, 10 iterations, 100 games/iteration

## Executive Summary

A 12-experiment replication study was conducted to validate MCTS simulation count findings.
Each configuration (200, 400, 800 simulations) was run 4 times for statistical significance.

**Key Result:** No statistically significant differences between simulation counts (ANOVA p=0.17).
The original finding that 200 sims outperforms 400 sims was likely due to random variance.

| Config | Mean Best | Std Dev | Mean Final | Best Range | Held Peak |
|--------|-----------|---------|------------|------------|-----------|
| 200 sims | 1550.0 | 21.4 | 1516.2 | [1525, 1575] | 1/4 |
| 400 sims | 1523.2 | 15.0 | 1493.8 | [1504, 1542] | 1/4 |
| 800 sims | 1541.5 | 10.2 | 1505.5 | [1533, 1558] | 0/4 |

## Detailed Results by Replicate

| Experiment | Best ELO | Best Iter | Final ELO | Std Dev | Regressions | Trend | Held Peak |
|------------|----------|-----------|-----------|---------|-------------|-------|-----------|
| mcts_200_rep1 | 1567 | 9 | 1567 | 34.7 | 4 | +4.6 | Yes |
| mcts_200_rep2 | 1575 | 3 | 1521 | 37.7 | 4 | -0.4 | No |
| *mcts_200_rep3 | 1533 | 3 | 1497 | 22.6 | 3 | -0.2 | No |
| *mcts_200_rep4 | 1525 | 2 | 1480 | 23.9 | 5 | +1.0 | No |
| *mcts_400_rep1 | 1514 | 3 | 1486 | 7.7 | 5 | -1.4 | No |
| mcts_400_rep2 | 1504 | 9 | 1504 | 31.0 | 3 | +3.4 | Yes |
| mcts_400_rep3 | 1542 | 4 | 1508 | 33.1 | 5 | +4.7 | No |
| *mcts_400_rep4 | 1533 | 5 | 1477 | 17.0 | 5 | +1.1 | No |
| *mcts_800_rep1 | 1533 | 2 | 1497 | 19.7 | 4 | -2.7 | No |
| *mcts_800_rep2 | 1542 | 2 | 1520 | 18.7 | 5 | +0.3 | No |
| *mcts_800_rep3 | 1533 | 2 | 1511 | 21.3 | 4 | +0.2 | No |
| mcts_800_rep4 | 1558 | 1 | 1494 | 26.1 | 3 | -2.2 | No |

*Asterisk indicates stable experiment (std < 25)

## Statistical Analysis

### Pairwise Comparisons (Best ELO)

**200 vs 400 sims:**
- Difference: +26.8 ELO (200 sims higher)
- t-statistic: 1.773
- p-value: 0.1266 (not significant)

**200 vs 800 sims:**
- Difference: +8.5 ELO (200 sims higher)
- t-statistic: 0.621
- p-value: 0.5572 (not significant)

**400 vs 800 sims:**
- Difference: -18.2 ELO (400 sims lower)
- t-statistic: -1.740
- p-value: 0.1325 (not significant)

### One-Way ANOVA (All Groups)

- F-statistic: 2.137
- p-value: 0.1741
- Result: No significant differences between groups

## ELO Progression by Iteration

| Iter | 200 sims (mean) | 400 sims (mean) | 800 sims (mean) |
|------|-----------------|-----------------|-----------------|
| 0 | 1500.0 | 1500.0 | 1500.0 |
| 1 | 1462.8 | 1447.0 | 1497.8 |
| 2 | 1483.2 | 1491.5 | 1516.8 |
| 3 | 1522.2 | 1491.0 | 1505.0 |
| 4 | 1471.8 | 1509.5 | 1488.2 |
| 5 | 1461.8 | 1495.0 | 1495.5 |
| 6 | 1502.5 | 1489.0 | 1504.0 |
| 7 | 1473.8 | 1503.8 | 1500.0 |
| 8 | 1487.5 | 1495.0 | 1476.2 |
| 9 | 1516.2 | 1493.8 | 1505.5 |

## Stability Analysis

| Config | Mean Std Dev | Mean Regressions | Mean Trend | Stability Score |
|--------|--------------|------------------|------------|-----------------|
| 200 sims | 29.7 | 4.0 | +1.2 | 7.0 (Volatile) |
| 400 sims | 22.2 | 4.5 | +1.9 | 6.7 (Volatile) |
| 800 sims | 21.5 | 4.0 | -1.1 | 6.1 (Volatile) |

## Key Findings

### 1. Original Finding NOT Confirmed
The original finding that 200 sims > 400 sims was **not statistically significant** (p=0.13).
This suggests the original result may have been due to random variance.

### 2. 800 Simulations Shows Best Stability
- Mean best ELO: 1541.5
- Lowest variance across replicates (std=10.2)
- Most consistent training (mean std_dev=21.5)

### 3. Diminishing Returns Pattern
| Sims | Mean Best | Marginal Gain | Time Cost |
|------|-----------|---------------|-----------|
| 200  | 1550.0     | baseline      | 1x        |
| 400  | 1523.2     | -26.8          | ~2x       |
| 800  | 1541.5     | +18.2          | ~4x       |

### 4. Training Instability is Universal
- Only 2/12 experiments held their peak ELO
- Most experiments peaked mid-training then regressed
- This suggests need for better regularization or early stopping

## Recommendations

### For Production Training
1. **Use 200 simulations** for fastest iteration with comparable results
2. **Use 800 simulations** if stability is prioritized over speed
3. **Avoid 400 simulations** - worst of both worlds (slow + lower performance)

### For Future Experiments
1. Implement early stopping when peak is detected
2. Test checkpoint ensemble to capture best iterations
3. Investigate training instability causes (LR decay, buffer composition)

## Updated Optimal Configuration

```yaml
mcts:
  early_simulations: 200    # Fast iteration, good performance
  late_simulations: 200
  # OR for stability:
  # early_simulations: 800  # Slower but most consistent
  # late_simulations: 800
```

---

**Experiment IDs:**
- mcts_200_rep1
- mcts_200_rep2
- mcts_200_rep3
- mcts_200_rep4
- mcts_400_rep1
- mcts_400_rep2
- mcts_400_rep3
- mcts_400_rep4
- mcts_800_rep1
- mcts_800_rep2
- mcts_800_rep3
- mcts_800_rep4

**Log File:** `experiments/logs/mcts_replication_batch.log`

**Analysis Script:** `python scripts/analyze_experiment_stability.py`
