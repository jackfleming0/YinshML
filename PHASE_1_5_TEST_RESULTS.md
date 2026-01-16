# Phase 1.5 Test Results: Value Head Fixes

**Date**: January 15, 2026
**Status**: ❌ **FAILED - Fixes Did Not Work**
**Run Directory**: `runs/20260115_155403/`
**Duration**: 5.13 hours

---

## 🚨 Critical Finding

**Phase 1.5 fixes (removing double tanh + pure MSE loss) DID NOT improve tournament performance.**

Models still rejected despite dramatic training loss improvements.

---

## Results Summary

### Training Loss Progression

| Iteration | Policy Loss | Value Loss | Change from Baseline | Tournament Result |
|-----------|-------------|------------|---------------------|-------------------|
| **Iteration 0** | 6.5675 | 0.1830 | - | ✅ PROMOTED (baseline) |
| **Iteration 1** | 3.9839 | 0.0711 | **-39% / -61%** ⬇️ | 🚫 **REJECTED (48.5%)** |
| **Iteration 2** | 4.6962 | 0.0706 | **-29% / -61%** ⬇️ | 🚫 **REJECTED (45.0%)** |

### Tournament Performance

| Iteration | ELO Rating | Win Rate | Decision | Notes |
|-----------|-----------|----------|----------|-------|
| **Iteration 0** | 1500.0 | - | ✅ PROMOTED | Baseline established |
| **Iteration 1** | 1492.5 | 48.5% | 🚫 REJECTED | Same as pre-Phase 1.5! |
| **Iteration 2** | 1468.3 | 45.0% | 🚫 REJECTED | **WORSE** than Iter 1 |

---

## Comparison to Pre-Phase 1.5 Test

### Before Phase 1.5 Fixes (Previous Test)

| Iteration | Policy Loss | Value Loss | Win Rate | Decision |
|-----------|-------------|------------|----------|----------|
| Iteration 0 | 6.1476 | 0.4595 | - | ✅ PROMOTED |
| Iteration 1 | 3.9788 | 0.2558 | 48.5% | 🚫 REJECTED |
| Iteration 2 | 4.6108 | 0.2752 | 48.2% | 🚫 REJECTED |

### After Phase 1.5 Fixes (This Test)

| Iteration | Policy Loss | Value Loss | Win Rate | Decision |
|-----------|-------------|------------|----------|----------|
| Iteration 0 | 6.5675 | 0.1830 | - | ✅ PROMOTED |
| Iteration 1 | 3.9839 | 0.0711 | 48.5% | 🚫 REJECTED |
| Iteration 2 | 4.6962 | 0.0706 | 45.0% | 🚫 REJECTED |

### Key Observations:

1. **Value Loss is MUCH Lower** (0.071 vs 0.256) - Phase 1.5 changes affected training
2. **Tournament Win Rates IDENTICAL** (48.5% both tests) - No improvement in play
3. **Iteration 2 Actually WORSE** (45.0% vs 48.2%) - Concerning trend

---

## What Phase 1.5 Fixed

### Fix 1: Removed Double Tanh ✅ Applied

**Location**: `yinsh_ml/network/wrapper.py:151`

**Change**: Removed second `torch.tanh()` application

**Impact on Training**: Value loss decreased from ~0.26 to ~0.07 (73% reduction!)

**Impact on Tournaments**: **NONE** - Win rates unchanged

---

### Fix 2: Pure MSE Loss ✅ Applied

**Location**: `yinsh_ml/training/trainer.py:518-533`

**Change**: Replaced hybrid MSE+BCE loss with pure MSE

**Impact on Training**: Value loss much lower, better convergence

**Impact on Tournaments**: **NONE** - Win rates unchanged

---

## Why Didn't Phase 1.5 Fixes Work?

### Evidence That Fixes Were Applied:

1. ✅ Value loss is MUCH lower (0.071 vs 0.256 in pre-fix test)
2. ✅ Training converges faster and to lower loss
3. ✅ Changes are clearly having an effect on the training process

### Evidence That Fixes Didn't Help:

1. ❌ Win rates identical to pre-fix test (48.5% vs 48.5%)
2. ❌ Models still rejected at same threshold
3. ❌ No ELO improvement across iterations
4. ❌ Iteration 2 actually performed WORSE (45% vs 48%)

---

## Possible Root Causes

### Hypothesis 1: Hybrid Evaluation Mode Masking Improvements

**Current Setting**: 70% heuristic + 30% neural

**Theory**: Neural improvements are being drowned out by the heuristic evaluator

**Evidence**:
- Network is clearly learning (losses decrease)
- But play strength doesn't improve
- 70% weight means heuristic dominates evaluation

**Test**: Run with `heuristic_weight=0.0` (pure neural mode)

---

### Hypothesis 2: Statistical Insignificance

**Current Setting**: 200 games per tournament match

**Theory**: Sample size too small to detect real improvement with MCTS noise

**Evidence**:
- Win rates very close to 50% (48.5%, 45.0%)
- High variance in MCTS evaluation
- Confidence intervals may overlap

**Test**: Increase to 400-800 games per match

---

### Hypothesis 3: Value Network Not Actually Improving

**Theory**: Lower loss ≠ better value predictions for actual game positions

**Evidence**:
- Value loss decreased dramatically
- But win rates didn't improve
- Network may be overfitting to training distribution
- Training data may not represent tournament play

**Diagnostic**: Analyze value predictions on held-out tournament positions

---

### Hypothesis 4: Policy Head Dominates, Value Ignored

**Theory**: MCTS may be relying more on policy than value

**Evidence**:
- Policy loss also decreased significantly
- Value improvements might not matter if MCTS trusts policy more
- Hybrid mode adds heuristic values on top

**Test**: Check MCTS value weights and backup formulas

---

### Hypothesis 5: Training Data Quality Issues

**Theory**: Self-play isn't generating good training signal

**Evidence**:
- Network learns patterns (loss decreases)
- But patterns don't transfer to stronger play
- May be learning to predict its own (weak) play

**Diagnostic**: Analyze self-play game quality and diversity

---

### Hypothesis 6: Baseline Contamination

**Theory**: Iteration 0 baseline is too similar to trained models

**Evidence**:
- All iterations using same network architecture
- Baseline might already be "good enough" with heuristics
- Small neural improvements can't overcome 70% heuristic weight

**Test**: Compare pure neural (0% heuristic) across iterations

---

## Training Metrics Analysis

### Loss Convergence

Both iterations showed strong learning:
- Policy loss: 6.5 → 4.0 → 4.7 (learns then rebounds after rejection)
- Value loss: 0.18 → 0.07 → 0.07 (dramatic improvement, stays low)

### Value Head Diagnostics

From final epoch output:
```
Pre-tanh Activations:
  Range: [0.263, 0.804]
  Distribution: 0.575 ± 0.089
  Saturated: 0.0%

Predictions:
  Confidence: 0.575
  High Confidence: 0.4%
  Distribution: 0.575 ± 0.089

Alignment with Targets:
  Sign Match: 100.0%
  MSE: 0.065
  MAE: 0.219
```

**Key Observations**:
- No saturation (0.0%) - Network using full range
- Perfect sign matching (100%) - Predicting win/loss correctly
- Low MSE (0.065) - Accurate value predictions
- But only 0.4% high confidence predictions

**Issue**: Network is predicting values around 0.5-0.6, very low confidence!

This might be the smoking gun - the network isn't expressing strong win/loss confidence.

---

## Memory Performance: ✅ Excellent

All memory optimizations from previous phase working perfectly:

- **Peak RSS**: 1.7 GB (safe)
- **Buffer Size**: Hit 10,000 cap exactly
- **System Memory**: 3+ GB free throughout
- **No OOM crashes**: All 3 iterations completed successfully
- **Worker Count**: 3 workers (capped correctly)

---

## Timeline

- **Start**: 15:54:03
- **Iteration 0 Complete**: 16:42:15 (48 min)
- **Iteration 1 Complete**: 18:57:19 (2.25 hrs including tournament)
- **Iteration 2 Complete**: 21:02:09 (2.1 hrs including tournament)
- **Total Duration**: 5.13 hours

---

## Next Steps: Diagnostic Testing

### Priority 1: Pure Neural Mode Test 🔴 HIGH

**Test**: Run 3 iterations with `heuristic_weight=0.0`

**Goal**: See if neural improvements are masked by heuristics

**Expected**: If neural is improving, pure mode should show it

**Command**:
```bash
# Modify run_training.py or config to set heuristic_weight=0.0
python scripts/run_training.py --iterations 3
```

---

### Priority 2: Increase Tournament Sample Size 🟡 MEDIUM

**Test**: Run with 400-800 games per match

**Goal**: Reduce statistical noise, increase confidence

**Expected**: Clearer signal if improvement is real but small

**Command**:
```bash
# Modify supervisor.py: GAMES_PER_MATCH = 400
python scripts/run_training.py --iterations 3
```

---

### Priority 3: Value Prediction Analysis 🟡 MEDIUM

**Test**: Analyze value predictions on tournament positions

**Goal**: Understand if value network is actually improving

**Method**:
1. Collect positions from tournament games
2. Evaluate with Iter 0 vs Iter 1 vs Iter 2 models
3. Compare value predictions to actual outcomes
4. Check if later iterations have better calibration

---

### Priority 4: Training Data Quality Analysis 🟢 LOW

**Test**: Analyze self-play games for quality

**Goal**: Ensure training data has good learning signal

**Metrics**:
- Game length distribution
- Move diversity
- Position complexity
- Outcome distribution (wins vs losses)

---

### Priority 5: MCTS Configuration Review 🟢 LOW

**Test**: Review MCTS value weighting and backup

**Goal**: Ensure value predictions are being used properly

**Check**:
- Value weight in UCB formula
- Backup calculations
- Temperature schedules
- Exploration parameters

---

## Conclusions

### What We Learned ✅

1. **Phase 1.5 fixes were applied correctly** - Training metrics changed dramatically
2. **Memory optimizations are rock solid** - No OOM issues across full test
3. **Network is learning patterns** - Losses decrease consistently
4. **Problem is NOT double-tanh or hybrid loss** - Those fixes didn't help

### What's Still Broken ❌

1. **Training improvements don't translate to play** - Core issue persists
2. **Models rejected despite lower loss** - Same ~48% win rate
3. **Value predictions lack confidence** - Only 0.4% high-confidence predictions
4. **Root cause still unknown** - Need more diagnostics

### Critical Next Action

**Run pure neural mode test** (`heuristic_weight=0.0`) to determine if:
- Neural network IS improving but masked by heuristics → Test will show improvement
- Neural network NOT improving meaningfully → Test will show similar ~50% results

This single test will tell us whether the issue is:
- **Evaluation masking** (fixable by adjusting hybrid weight)
- **Training fundamentals** (requires deeper investigation)

---

## Recommendation

**DO NOT commit Phase 1.5 changes yet.** While they may be correct in principle, they didn't solve the core problem and may have introduced unintended effects (lower value confidence).

**Next**: Run pure neural mode test as Priority 1 diagnostic.

If pure neural shows improvement → Phase 1.5 fixes worked, just masked by heuristics
If pure neural shows no improvement → Deeper issues with training/evaluation

---

## Files Generated

- **Run Directory**: `runs/20260115_155403/`
- **Checkpoints**: `checkpoint_iteration_0.pt` (promoted), others deleted
- **Best Model**: `best_model.pt` (Iteration 0)
- **Suggestions**: `suggestions_iter_2.yaml`, `suggestions_iter_3.yaml`
- **This Report**: `PHASE_1_5_TEST_RESULTS.md`

---

**Status**: Phase 1.5 test complete, fixes did not improve tournament performance. Awaiting further diagnostics.
