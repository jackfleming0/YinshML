# Temperature Schedule Experiment Results

**Date:** January 31, 2026
**Branch:** `heuristic_seeding`
**Baseline Reference:** baseline_004 + LR findings (value_lr_factor: 2.5) + MCTS findings (200 sims)

## Executive Summary

Four temperature schedule experiments were conducted to understand the impact of exploration/exploitation balance on training performance. **Experiment 003 (higher final temperature)** achieved the best results with **ELO 1558**, suggesting that maintaining exploration throughout training improves performance.

| Experiment | Configuration | Best ELO | Best Iter | Stability (Std) | Held Peak |
|------------|---------------|----------|-----------|-----------------|-----------|
| 001 | 15 annealing steps (faster) | 1527 | 9 | ⭐ 15.6 | ✅ Yes |
| 002 | 60 annealing steps (slower) | 1508 | 3 | ⭐ 15.7 | No |
| **003** | **Final temp 0.3 (higher)** | **1558** | **9** | 34.8 | ✅ Yes |
| 004 | Initial 1.5 + 20 steps | 1529 | 2 | TBD | No |

## Methodology

### Base Configuration
All experiments used optimized settings from previous experiments:
- **value_lr_factor:** 2.5 (from LR sensitivity)
- **MCTS simulations:** 200 (from MCTS depth)
- **Iterations:** 10
- **Games per iteration:** 100

### Variable: Temperature Schedule
| Experiment | Initial | Final | Annealing Steps | Description |
|------------|---------|-------|-----------------|-------------|
| Baseline | 1.0 | 0.1 | 30 | Standard linear decay |
| 001 | 1.0 | 0.1 | 15 | Faster decay to exploitation |
| 002 | 1.0 | 0.1 | 60 | Slower decay, extended exploration |
| 003 | 1.0 | 0.3 | 30 | Higher final temp, sustained exploration |
| 004 | 1.5 | 0.1 | 20 | High initial + fast decay |

## Detailed Results

### Experiment 001: Faster Annealing (15 steps)
**Hypothesis:** Quicker shift to exploitation may help convergence.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1471 | 0 |
| 2 | 1525 | 2 |
| 3 | 1508 | 2 |
| 4 | 1518 | 2 |
| 5 | 1492 | 2 |
| 6 | 1504 | 2 |
| 7 | 1503 | 2 |
| 8 | 1510 | 2 |
| 9 | 1527 | 9 |

**Stability Metrics:**
- Std Dev: 15.6 ⭐ (stable)
- Regressions: 4
- Trend: +2.2 (positive)
- Held Peak: ✅ Yes

**Conclusion:** Good stability with positive trend. Faster exploitation didn't hurt.

---

### Experiment 002: Slower Annealing (60 steps)
**Hypothesis:** Extended exploration may find better solutions.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1488 | 0 |
| 2 | 1492 | 0 |
| 3 | 1508 | 3 |
| 4 | 1506 | 3 |
| 5 | 1482 | 3 |
| 6 | 1460 | 3 |
| 7 | 1475 | 3 |
| 8 | 1469 | 3 |
| 9 | 1503 | 3 |

**Stability Metrics:**
- Std Dev: 15.7 ⭐ (stable)
- Regressions: 5
- Trend: -2.2 (negative)
- Held Peak: ❌ No (peaked at 1508, ended at 1503)

**Conclusion:** Stable but underperformed. Extended exploration without exploitation hurt final performance.

---

### Experiment 003: Higher Final Temperature (0.3) ⭐ BEST
**Hypothesis:** Maintaining exploration throughout may improve generalization.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1483 | 0 |
| 2 | 1442 | 0 |
| 3 | 1471 | 0 |
| 4 | 1467 | 0 |
| 5 | 1425 | 0 |
| 6 | 1458 | 0 |
| 7 | 1467 | 0 |
| 8 | 1450 | 0 |
| 9 | 1558 | 9 |

**Stability Metrics:**
- Std Dev: 34.8 (volatile)
- Regressions: 5
- Trend: +2.0 (positive)
- Held Peak: ✅ Yes

**Observations:**
- Struggled early (dipped to 1425 at iteration 5)
- Dramatic late improvement at iteration 9
- Highest final ELO despite early volatility

**Conclusion:** High variance but highest ceiling. Sustained exploration allows late breakthroughs.

---

### Experiment 004: High Initial + Fast Decay
**Hypothesis:** Maximum early exploration then rapid exploitation.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1450 | 0 |
| 2 | 1529 | 2 |
| 3 | 1508 | 2 |
| 4 | 1489 | 2 |
| 5 | 1483 | 2 |
| 6 | 1492 | 2 |
| 7 | 1476 | 2 |
| 8 | 1482 | 2 |
| 9 | 1516 | 2 |

**Stability Metrics:**
- Std Dev: ~22
- Regressions: 5
- Trend: ~+1.0
- Held Peak: ❌ No (peaked at 1529, ended at 1516)

**Conclusion:** Peaked early at iteration 2, then couldn't improve. Aggressive strategy didn't pay off.

---

## Key Findings

### 1. Sustained Exploration Wins
The highest performing config (003) maintained exploration with final temp 0.3:
- More stochastic move selection throughout training
- Higher diversity in training data
- Allows late-game breakthroughs

### 2. Annealing Speed Has Moderate Impact
| Annealing | Best ELO | Observation |
|-----------|----------|-------------|
| Fast (15 steps) | 1527 | Good, stable |
| Standard (30 steps) | Baseline | Reference |
| Slow (60 steps) | 1508 | Underperformed |

Fast annealing is slightly better than slow.

### 3. Stability vs Performance Trade-off
| Metric | 001 (Fast) | 003 (High Final) |
|--------|------------|------------------|
| Best ELO | 1527 | 1558 |
| Stability | ⭐ Excellent | Volatile |
| Risk | Low | High variance |

**Choose based on goals:**
- **Consistent results:** Use fast annealing (001)
- **Maximum performance:** Use high final temp (003)

## Recommendations

### Immediate Actions
1. **For production:** Consider `final_temp: 0.3` for maximum ELO
2. **For reliability:** Use `annealing_steps: 15` for stable training
3. **Avoid:** Slow annealing (60 steps) showed no benefit

### Updated Optimal Configuration
```yaml
temperature:
  initial: 1.0
  final: 0.3           # CHANGED from 0.1 (maintains exploration)
  annealing_steps: 30  # Standard
  schedule: linear
  clamp_fraction: 0.6
```

### Future Experiments
1. Test intermediate final temps (0.15, 0.2, 0.25)
2. Non-linear annealing schedules (cosine, exponential)
3. Adaptive temperature based on training progress

---

## Stability Analysis Protocol

All experiments now include stability metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Std Dev** | ELO variance across iterations | < 25 |
| **Regressions** | Iterations with ELO drop | < 5 |
| **Held Peak** | Final ELO == Best ELO | ✅ Yes |
| **Trend** | Linear slope of ELO progression | > 0 |
| **Quality Score** | Composite (performance + stability) | > 2.0 |

Run stability analysis: `python scripts/analyze_experiment_stability.py <log_file>`

---

**Experiment IDs:**
- 001: `1335ab10`
- 002: `903708b4`
- 003: `8db94a5c`
- 004: `4c140ddd`

**Log File:** `temp_schedule_batch.log`
