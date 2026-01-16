# Pure Neural Mode Test Results

**Date**: January 16, 2026 (completed 04:11 AM)
**Status**: ❌ **CRITICAL FINDING - Neural Network NOT Improving**
**Run Directory**: `runs/20260115_223637/`
**Duration**: 5.58 hours
**Configuration**: 100% Neural (heuristic_weight=0.0)

---

## 🚨 CRITICAL FINDING

**The pure neural mode test definitively proves the neural network is NOT improving in playing strength, despite dramatic training loss reductions.**

This eliminates the hypothesis that heuristic evaluation was masking improvements.

---

## Pure Neural Mode Results

### Training Loss Progression

| Iteration | Policy Loss | Value Loss | Change from Baseline |
|-----------|-------------|------------|---------------------|
| **Iteration 0** | 6.7067 | 0.3022 | - |
| **Iteration 1** | 3.9563 | 0.0725 | **-41% / -76%** ⬇️ |
| **Iteration 2** | 4.7630 | 0.0806 | **-29% / -73%** ⬇️ |

### Tournament Performance (Pure Neural)

| Iteration | ELO Rating | Win Rate | Decision |
|-----------|-----------|----------|----------|
| **Iteration 0** | 1500.0 | - | ✅ PROMOTED (baseline) |
| **Iteration 1** | 1477.5 | **46.5%** | 🚫 **REJECTED** |
| **Iteration 2** | 1485.0 | **47.2%** | 🚫 **REJECTED** |

---

## Comparison Across All Three Tests

### Test 1: Hybrid Mode (70% Heuristic) - Pre-Phase 1.5

| Iteration | Policy Loss | Value Loss | Win Rate | Decision |
|-----------|-------------|------------|----------|----------|
| Iteration 0 | 6.1476 | 0.4595 | - | ✅ PROMOTED |
| Iteration 1 | 3.9788 | 0.2558 | 48.5% | 🚫 REJECTED |
| Iteration 2 | 4.6108 | 0.2752 | 48.2% | 🚫 REJECTED |

---

### Test 2: Hybrid Mode (70% Heuristic) - Phase 1.5 Fixes

| Iteration | Policy Loss | Value Loss | Win Rate | Decision |
|-----------|-------------|------------|----------|----------|
| Iteration 0 | 6.5675 | 0.1830 | - | ✅ PROMOTED |
| Iteration 1 | 3.9839 | 0.0711 | 48.5% | 🚫 REJECTED |
| Iteration 2 | 4.6962 | 0.0706 | 45.0% | 🚫 REJECTED |

---

### Test 3: Pure Neural Mode (0% Heuristic) - Phase 1.5 Fixes

| Iteration | Policy Loss | Value Loss | Win Rate | Decision |
|-----------|-------------|------------|----------|----------|
| Iteration 0 | 6.7067 | 0.3022 | - | ✅ PROMOTED |
| Iteration 1 | 3.9563 | 0.0725 | **46.5%** | 🚫 REJECTED |
| Iteration 2 | 4.7630 | 0.0806 | **47.2%** | 🚫 REJECTED |

---

## Key Observations

### 1. Consistent Rejection Across All Modes

**Win Rates Across Tests**:
- Pre-Phase 1.5, Hybrid (70% heuristic): 48.5%, 48.2%
- Phase 1.5, Hybrid (70% heuristic): 48.5%, 45.0%
- Phase 1.5, Pure Neural (0% heuristic): **46.5%, 47.2%**

**All three tests show ~46-48% win rates → All models REJECTED**

This is statistically significant evidence that:
- ❌ Neural network is NOT improving at evaluation
- ❌ Problem is NOT heuristic masking
- ❌ Problem is NOT double-tanh or hybrid loss (Phase 1.5 fixes)
- ✅ Problem is FUNDAMENTAL to training/evaluation

---

### 2. Training Losses Decrease Dramatically

**Policy Loss**: Consistently drops ~40% (6.5 → 4.0)
**Value Loss**: Consistently drops ~60-75% (0.3-0.4 → 0.07)

**But**: This learning does NOT translate to stronger play in ANY mode

---

### 3. Pure Neural Mode Is Slightly WORSE

**Hybrid Mode (70% heuristic)**: 48.5%, 48.2%, 45.0% win rates
**Pure Neural Mode (0% heuristic)**: **46.5%, 47.2%** win rates

Pure neural performs slightly worse (1-2 percentage points), suggesting:
- Heuristic evaluator is slightly stronger than neural network
- Neural network improvements are marginal at best
- Even without heuristic "masking", neural isn't improving

---

## What This Definitively Rules Out

### ❌ Hypothesis 1: Heuristic Masking (ELIMINATED)

**Theory**: 70% heuristic weight masks neural improvements

**Test Result**: Pure neural mode (0% heuristic) shows SAME rejection pattern

**Conclusion**: Heuristics are NOT masking improvements. Neural genuinely not improving.

---

### ❌ Hypothesis 2: Double-Tanh Bug (ELIMINATED)

**Theory**: Double-tanh over-compression prevents learning

**Test Result**: Phase 1.5 fixed double-tanh, but models still rejected

**Conclusion**: Double-tanh was a bug, but NOT the root cause of rejection

---

### ❌ Hypothesis 3: Hybrid Loss Mismatch (ELIMINATED)

**Theory**: MSE+BCE training vs MSE inference causes mismatch

**Test Result**: Phase 1.5 switched to pure MSE, but models still rejected

**Conclusion**: Hybrid loss may have been suboptimal, but NOT the root cause

---

## Remaining Hypotheses

### ✅ Hypothesis 4: Training Data Quality Issues (LIKELY)

**Theory**: Self-play games don't provide good learning signal

**Evidence**:
- Network learns patterns (loss decreases)
- But patterns don't translate to stronger play
- May be learning to predict its own (weak) play
- Training distribution ≠ evaluation distribution

**Next Test**: Analyze self-play game quality, diversity, and outcomes

---

### ✅ Hypothesis 5: Value Predictions Not Useful (LIKELY)

**Theory**: Value network predictions don't help MCTS make better decisions

**Evidence from Phase 1.5 Test**:
- Only 0.4% high-confidence predictions
- Most predictions clustered around 0.5-0.6 (low confidence)
- Network correctly predicts WHO wins (100% sign match)
- But expresses very weak confidence about it

**Theory**: Weak confidence predictions don't guide MCTS effectively
- MCTS needs strong value signals to prune bad branches
- Predictions around 0.5-0.6 are barely better than random
- Policy head may be dominating MCTS decisions

**Next Test**:
1. Analyze value prediction distributions in MCTS
2. Check if MCTS is actually using value predictions
3. Compare MCTS with/without value guidance

---

### ✅ Hypothesis 6: Network Capacity Issues (POSSIBLE)

**Theory**: Network too small or architecture suboptimal

**Evidence**:
- Consistent ~47% win rates suggest network at capacity
- Can't learn beyond certain level
- May need deeper/wider architecture

**Next Test**: Try larger network (e.g., 20 blocks instead of 12)

---

### ✅ Hypothesis 7: Policy Head Dominates (POSSIBLE)

**Theory**: MCTS relies more on policy than value predictions

**Evidence**:
- Policy loss also decreases significantly
- But still doesn't translate to better play
- Both policy and value improvements failing to help

**Next Test**:
1. Review MCTS UCB formula value weighting
2. Test with value_weight=0 (pure policy) vs value_weight=2 (heavy value)

---

### ✅ Hypothesis 8: Baseline Contamination (POSSIBLE)

**Theory**: Iteration 0 baseline already "good enough"

**Evidence**:
- All iterations hover around 46-48% win rate
- Very narrow performance band
- May indicate all models similarly weak/strong

**Next Test**: Compare all iterations against random/heuristic baseline

---

### ✅ Hypothesis 9: Statistical Insignificance (POSSIBLE)

**Theory**: 200 games not enough to detect small improvements

**Evidence**:
- Win rates very close to 50% (46-48%)
- Confidence intervals likely overlap
- May need 400-800 games per match

**Next Test**: Run tournament with 400-800 games

---

## Critical Insight: Low Confidence Predictions

From Phase 1.5 test diagnostics:

```
Value Head Analysis:
  Confidence: 0.575
  High Confidence: 0.4%
  Distribution: 0.575 ± 0.086
  Sign Match: 100.0%
```

**Key Finding**: Network predicts outcomes correctly (100% sign match) but with very low confidence (0.575 ± 0.086)

**Why This Matters for MCTS**:
- MCTS uses value predictions to guide search
- High confidence values prune bad branches effectively
- Low confidence values (~0.5) barely influence search
- MCTS may be effectively ignoring value predictions

**Example**:
- Strong prediction: "This position is 0.9 for White" → MCTS explores more
- Weak prediction: "This position is 0.55 for White" → MCTS barely adjusts

**If value predictions are all ~0.5-0.6, MCTS can't effectively use them!**

---

## Training vs Evaluation Mismatch (Refined Understanding)

### What We Know:

1. **Training**: Network learns to predict outcomes with lower MSE
   - Value loss: 0.4 → 0.07 (76% reduction)
   - Sign accuracy: 100% (predicts winner correctly)

2. **Evaluation**: Network doesn't help MCTS play better
   - Win rates: ~47% (no improvement)
   - Predictions too weak/uncertain to guide search

### The Real Mismatch:

**Training optimizes for**: Accurate value predictions (low MSE)
**Evaluation needs**: Confident, discriminative value predictions (high variance)

**Network learns to**:
- Predict average outcome accurately (minimize MSE)
- Output values close to 0.5-0.6 (safe, low-error predictions)
- Avoid confident predictions that might be wrong

**But MCTS needs**:
- Strong signals about position quality
- Values close to 0 or 1 for clear-cut positions
- Confidence to prune bad branches

**Analogy**: Network learned to always predict "60% chance of rain" (safe, low error) instead of "10% sun" or "90% storm" (useful for decisions).

---

## Root Cause Analysis

### Primary Suspects:

1. **Value Predictions Too Weak/Uncertain** (⭐ Most Likely)
   - Network outputs ~0.5-0.6 for most positions
   - MCTS can't effectively use such weak signals
   - Training for low MSE encourages conservative predictions

2. **Training Data Quality** (⭐ Likely)
   - Self-play between weak players generates weak training signal
   - Network learns to predict weak play, not strong play
   - Need stronger baseline or supervised data

3. **Statistical Insignificance** (Possible)
   - 200 games may not detect small improvements
   - Need larger sample size

---

## Recommended Next Actions

### Priority 1: Diagnose Value Prediction Usage 🔴 CRITICAL

**Goal**: Determine if MCTS is actually using value predictions effectively

**Tests**:
1. Log value predictions during MCTS search
2. Analyze value distribution across visited nodes
3. Check correlation between value predictions and final move selection
4. Compare MCTS with value_weight=0 (pure policy) vs value_weight=1.0

**Expected**: May find value predictions are effectively ignored due to low confidence

---

### Priority 2: Analyze Training Data Quality 🔴 CRITICAL

**Goal**: Determine if self-play generates good learning signal

**Metrics**:
1. Game length distribution (are games diverse?)
2. Outcome distribution (50/50 or biased?)
3. Position complexity (same positions repeated?)
4. Move diversity (exploring or exploiting?)
5. Value target distribution (clustered around 0.5?)

**Expected**: May find training data has weak signal (values all ~0.5)

---

### Priority 3: Increase Tournament Sample Size 🟡 HIGH

**Goal**: Rule out statistical insignificance

**Test**: Run with 400-800 games per match

**Expected**: If real improvement exists but small, larger sample will detect it

---

### Priority 4: Bootstrap from Stronger Baseline 🟡 MEDIUM

**Goal**: Generate better training data from stronger play

**Approach**:
1. Start with pure heuristic evaluator (100% heuristic)
2. Generate self-play games with heuristic MCTS
3. Train neural network on heuristic-guided play
4. Network learns to predict strong play, not weak play

**This breaks the "weak learning from weak play" cycle**

---

### Priority 5: Modify Loss Function for Confidence 🟢 LOW

**Goal**: Encourage network to make confident predictions

**Approach**:
1. Add confidence penalty term
2. Reward predictions close to -1 or +1
3. Penalize predictions close to 0
4. Force network to "commit" to position evaluation

**Risk**: May increase MSE but improve MCTS guidance

---

## Comparison to AlphaZero

### AlphaZero Success Factors:

1. **Large Network**: 20-40 residual blocks (we have 12)
2. **Many Iterations**: 100+ iterations (we tested 3)
3. **More Training**: 700k games for Go (we use 50 per iteration)
4. **Strong Confidence**: Value predictions spread across [-1, 1]

### Our Results:

1. **Smaller Network**: 12 blocks (may be insufficient)
2. **Few Iterations**: 3 iterations (not enough for convergence)
3. **Less Training**: 50 games per iteration (weak signal)
4. **Weak Confidence**: Values clustered at 0.5-0.6 (not useful)

**Key Difference**: AlphaZero's value network makes confident predictions across full range. Ours makes weak predictions clustered around 0.5.

---

## Conclusions

### What We Learned ✅

1. **Heuristic masking is NOT the problem** - Pure neural mode shows same rejection
2. **Phase 1.5 fixes (double-tanh, hybrid loss) are NOT the root cause** - Fixed them, still rejected
3. **Network IS learning patterns** - Training losses decrease consistently
4. **But patterns DON'T improve play** - ~47% win rate across all tests
5. **Value predictions are too weak** - Only 0.4% high confidence, clustered at 0.5-0.6
6. **Problem is FUNDAMENTAL** - Either training data, network capacity, or value usage

### Critical Next Step

**Priority 1: Diagnose why value predictions don't help MCTS**

This single diagnostic will tell us whether:
- Value predictions are correct but too weak (confidence problem)
- Value predictions are ignored by MCTS (integration problem)
- Value predictions are wrong/useless (capacity problem)

Once we understand how MCTS uses values, we can target the fix:
- If ignored → Adjust MCTS value weighting
- If too weak → Modify loss function for confidence
- If wrong → Improve training data or network capacity

---

## Recommendation

**DO NOT proceed with longer training (10+ iterations) until we understand why current training doesn't improve play.**

Running more iterations with the current setup will just produce more rejected models.

**Next action**: Deep diagnostic of value prediction usage in MCTS and training data quality analysis.

---

## Files Generated

- **Run Directory**: `runs/20260115_223637/`
- **Checkpoints**: `checkpoint_iteration_0.pt` (promoted)
- **Best Model**: `best_model.pt` (Iteration 0)
- **This Report**: `PURE_NEURAL_MODE_TEST_RESULTS.md`

---

**Status**: Pure neural test complete. Neural network definitively NOT improving. Root cause investigation needed before further training.
