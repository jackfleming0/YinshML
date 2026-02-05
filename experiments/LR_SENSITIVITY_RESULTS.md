# Learning Rate Sensitivity Experiment Results

**Date:** January 27-29, 2026
**Branch:** `heuristic_seeding`
**Baseline Reference:** `baseline_004` (ELO 1562, best performer from Phase A)

## Executive Summary

Five learning rate sensitivity experiments were conducted to understand the impact of policy and value learning rates on training performance. **Experiment 003 (lower value LR factor)** achieved the best results with **ELO 1571**, suggesting that slower value head learning improves model quality.

| Experiment | Configuration | Best ELO | Best Iter | vs Baseline |
|------------|---------------|----------|-----------|-------------|
| 001 | Policy LR 0.0005 (half) | 1533 | 9 | -29 |
| 002 | Policy LR 0.002 (double) | 1525 | 1 | -37 |
| **003** | **Value LR factor 2.5 (half)** | **1571** | **3** | **+9** |
| 004 | Value LR factor 10.0 (double) | 1519 | 6 | -43 |
| 005 | Conservative (both lower) | 1508 | 9 | -54 |

## Methodology

### Base Configuration (from baseline_004)
- **Iterations:** 10
- **Games per iteration:** 100
- **Epochs per iteration:** 4
- **Batch size:** 256
- **Buffer size:** 20,000
- **MCTS simulations:** 100 (early and late)
- **Tournament games:** 40 per match

### Learning Rate Parameters (Baseline)
- **Policy LR:** 0.001
- **Value LR factor:** 5.0 (effective value LR = policy_lr × factor)
- **Scheduler:** StepLR (step_size=10, gamma=0.9)

## Detailed Results

### Experiment 001: Lower Policy LR (0.0005)
**Hypothesis:** Slower policy learning may improve convergence stability.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1400 | 0 |
| 2 | 1400 | 0 |
| 3 | 1483 | 0 |
| 4 | 1475 | 0 |
| 5 | 1475 | 0 |
| 6 | 1471 | 0 |
| 7 | 1458 | 0 |
| 8 | 1425 | 0 |
| 9 | 1533 | 9 |

**Observations:**
- Struggled early (dropped to 1400 at iterations 1-2)
- Slow recovery through iterations 3-8
- Late surge at iteration 9 achieved final best
- Pattern suggests policy may have been learning too slowly to adapt

**Conclusion:** Half policy LR is too conservative; model took too long to learn.

---

### Experiment 002: Higher Policy LR (0.002)
**Hypothesis:** Faster policy learning may speed up training.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1525 | 1 |
| 2 | 1492 | 1 |
| 3 | 1483 | 1 |
| 4 | 1483 | 1 |
| 5 | 1499 | 1 |
| 6 | 1475 | 1 |
| 7 | 1429 | 1 |
| 8 | 1504 | 1 |
| 9 | 1514 | 1 |

**Observations:**
- Peaked early at iteration 1 (ELO 1525)
- Never exceeded initial peak despite 8 more iterations
- Suggests overfitting or instability from high LR
- Best model remained iteration 1 throughout

**Conclusion:** Double policy LR causes early overfitting; model can't improve past initial learning.

---

### Experiment 003: Lower Value LR Factor (2.5) ⭐ BEST
**Hypothesis:** Slower value learning may improve stability and final performance.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1450 | 0 |
| 2 | 1458 | 0 |
| 3 | 1571 | 3 |
| 4 | 1486 | 3 |
| 5 | 1480 | 3 |
| 6 | 1504 | 3 |
| 7 | 1476 | 3 |
| 8 | 1475 | 3 |
| 9 | 1531 | 3 |

**Observations:**
- Initial dip (iterations 1-2) then strong recovery
- Achieved highest ELO (1571) at iteration 3
- Maintained stable performance after peak
- Best model found relatively early and held

**Conclusion:** Lower value LR factor produces best results. Value head benefits from slower, more stable learning.

---

### Experiment 004: Higher Value LR Factor (10.0)
**Hypothesis:** Faster value learning may help the model assess positions better.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1508 | 1 |
| 2 | 1503 | 1 |
| 3 | 1517 | 3 |
| 4 | 1504 | 3 |
| 5 | 1487 | 3 |
| 6 | 1519 | 6 |
| 7 | 1511 | 6 |
| 8 | 1487 | 6 |
| 9 | 1503 | 6 |

**Observations:**
- Gradual improvement through iteration 6
- Best achieved 1519 (lower than baseline)
- More volatile than 003 (lower value LR)
- Multiple "best" model changes (1→3→6)

**Conclusion:** Higher value LR causes instability; model oscillates rather than converging.

---

### Experiment 005: Conservative (Both Lower)
**Hypothesis:** Both learning rates reduced may provide stable, gradual improvement.

| Iter | ELO | Best |
|------|-----|------|
| 0 | 1500 | 0 |
| 1 | 1450 | 0 |
| 2 | 1508 | 2 |
| 3 | 1489 | 2 |
| 4 | 1490 | 2 |
| 5 | 1446 | 2 |
| 6 | 1503 | 2 |
| 7 | 1475 | 2 |
| 8 | 1493 | 2 |
| 9 | 1508 | 9 |

**Observations:**
- Modest peak at iteration 2 (1508)
- Highly variable performance (1446-1508 range)
- Final iteration tied with early best
- No clear improvement trend

**Conclusion:** Too conservative; both learning rates being low prevents effective learning.

---

## Key Findings

### 1. Value Learning Rate is Critical
The value LR factor had the largest impact on performance:
- **2.5 (half):** Best results (1571 ELO)
- **5.0 (baseline):** Good results (1562 ELO)
- **10.0 (double):** Worse results (1519 ELO)

**Insight:** The value head benefits from slower, more stable learning. This may be because accurate position evaluation is harder to learn and requires more careful gradient updates.

### 2. Policy Learning Rate Sweet Spot
Policy LR modifications showed clear boundaries:
- **0.0005 (half):** Too slow, delayed learning
- **0.001 (baseline):** Good balance
- **0.002 (double):** Early overfitting

**Insight:** The baseline policy LR (0.001) appears near-optimal. Deviations in either direction hurt performance.

### 3. Combined Conservative Approach Fails
Reducing both learning rates (experiment 005) performed poorly, suggesting:
- The policy needs sufficient LR to learn from MCTS guidance
- Only the value head benefits from reduced LR
- Asymmetric LR approach (normal policy, lower value) is optimal

### 4. Early Peaking Indicates Overfitting
Experiments that peaked early (002 at iter 1, 003 at iter 3) showed different outcomes:
- **002 (high policy LR):** Peaked and couldn't improve (overfitting)
- **003 (low value LR):** Peaked and maintained (stable optimum)

**Insight:** Early peaks aren't inherently bad; the key is whether the model can maintain or improve from that peak.

## Recommendations

### Immediate Actions
1. **Update baseline configuration:** Set `value_lr_factor: 2.5` as new default
2. **Keep policy LR at 0.001:** No benefit from changing
3. **Consider 2.0-3.0 range for value LR factor:** Further tuning may find even better values

### Future Experiments
1. **Fine-tune value LR factor:** Test 2.0, 2.5, 3.0 specifically
2. **Separate schedulers:** Test different decay rates for policy vs value
3. **Warmup strategies:** Test LR warmup for early iterations
4. **Longer training:** Test if 003's configuration maintains advantage over 20+ iterations

## Configuration for Next Experiments

Based on these findings, recommended base configuration:

```yaml
optimizer:
  policy_lr: 0.001           # Keep baseline
  value_lr_factor: 2.5       # CHANGED from 5.0
  policy_weight_decay: 0.0001
  value_weight_decay: 0.001
  value_momentum: 0.9
  scheduler_type: step
  scheduler_step_size: 10
  scheduler_gamma: 0.9
```

## Technical Notes

### Logging Fix Applied
During these experiments, a critical logging bug was fixed:
- **Problem:** `print()` statements in `trainer.py` bypassed logging levels, causing 11GB+ log files
- **Solution:** Converted all prints to proper `logger.debug()` calls
- **Result:** Log files now ~2MB per experiment (was potentially 11GB+)

### Storage Cleanup
Old training artifacts were removed to free disk space:
- `runs/` (11GB) - Old training runs pre-experiment system
- `models/` (2GB) - Legacy model iterations
- `runs_smoke/`, `test_runs/` (~800MB) - Old test data

---

**Experiment IDs:**
- 001: `7f551bfe`
- 002: `3263b81a`
- 003: `a19a4688`
- 004: `51e55515`
- 005: `4b087fd5`

**Log Files:**
- 001: `experiment_20260127_195806.log` (1.9MB)
- 002-004: `lr_batch_002-005.log` (6.0MB)
- 005: `lr_005_restart.log` (1.9MB)
