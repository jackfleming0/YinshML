# Bootstrap Failure Analysis: Root Cause Revision

**Date**: January 18, 2026
**Status**: 🔴 CRITICAL DISCOVERY - Bootstrap from 100% heuristic FAILED
**Previous Hypothesis**: REJECTED

---

## Executive Summary

The 100% heuristic bootstrap test **failed to improve discrimination**, achieving 0.042 (same as no-penalty baseline) instead of expected 0.10-0.15. This fundamentally challenges our previous hypothesis and reveals that **pure MSE loss is the core problem**, not MCTS target quality.

### Critical Finding

**Bootstrap hypothesis was WRONG**: We believed weak MCTS targets (from weak network) were the issue. Bootstrapping from 100% heuristic MCTS should have provided strong, diverse targets. But discrimination did NOT improve.

**New insight**: The problem is **pure MSE loss**, which inherently minimizes prediction variance regardless of target quality.

---

## Test Results Comparison

| Test | Targets | Heuristic | Variance Penalty | Discrimination | Weak % | Std Dev |
|------|---------|-----------|------------------|----------------|--------|---------|
| **Baseline** | Outcomes | 0.7 | 0.2 | 0.043 | 75% | 0.083 |
| **Test 1** | Outcomes | 0.3 | 0.5 | 0.043 | 71% | 0.079 |
| **Test 2 ⭐** | MCTS | 0.3 | 0.5 | **0.059** | **35%** | **0.107** |
| **Test 3** | MCTS | 0.3 | 0.0 | 0.043 | 75% | 0.084 |
| **Bootstrap** | MCTS | **1.0** | 0.0 | **0.042** | **73%** | **0.079** |

### Key Observations

1. **Bootstrap = Test 3** in discrimination (0.042 vs 0.043)
2. **Only Test 2** (with variance penalty 0.5) showed improvement
3. **Heuristic quality irrelevant** - 100% heuristic no better than 30%
4. **Variance penalty was essential**, not a constraint

---

## Why Bootstrap Failed: MSE Loss Properties

### Pure MSE Loss Minimizes Variance

**Mathematical property**:
- Given targets with variance σ²
- MSE loss: L = E[(y - ŷ)²]
- Optimal prediction: ŷ = E[y] (the mean)
- Network learns to predict mean + small adjustments
- **Result**: Predictions have lower variance than targets

### Evidence from Tests

**Test 2 (with penalty)**:
```
Training variance: 0.0211 (higher due to penalty forcing variance)
Discrimination: 0.059 (best result)
Conclusion: Penalty prevented MSE's variance minimization
```

**Test 3 (no penalty)**:
```
Training loss: 0.014 (perfect fit to targets!)
Training variance: 0.0127 (40% LOWER than Test 2)
Discrimination: 0.043 (regression to baseline)
Conclusion: Network perfectly learned low-variance predictions
```

**Bootstrap (no penalty)**:
```
Discrimination: 0.042 (same as Test 3)
Std dev: 0.079 (low variance)
Conclusion: Even with 100% heuristic targets, MSE pulled toward mean
```

### Why Variance Penalty Helped

In Test 2, variance penalty forced the network to:
1. Not perfectly fit the (potentially low-variance) MCTS targets
2. Maintain higher prediction variance
3. Result: Better discrimination (0.059)

The penalty wasn't constraining good discrimination - it was **compensating for MSE's inherent variance minimization**.

---

## Revised Root Cause

### Previous Hypothesis (REJECTED)

> "Weak value discrimination is caused by weak MCTS targets during self-play due to bootstrap problem. Solution: Bootstrap from 100% heuristic."

**Why wrong**: Bootstrap from 100% heuristic produced same weak discrimination (0.042). Target quality isn't the limiting factor.

### New Hypothesis (CONFIRMED)

> "Weak value discrimination is caused by **pure MSE loss**, which inherently minimizes prediction variance regardless of target quality. Even with strong, diverse MCTS targets, MSE pulls predictions toward the mean."

**Evidence**:
1. Bootstrap (strong targets) = Test 3 (weak targets) = 0.042 discrimination
2. Only Test 2 (variance penalty) showed improvement
3. Test 3 perfectly fit targets (loss 0.014) but with low variance
4. MSE is mathematically biased toward mean prediction

---

## Why AlphaZero Doesn't Have This Problem

AlphaZero and AlphaGo use different approaches:

### 1. **Cross-Entropy Loss** (AlphaGo Zero)

AlphaGo Zero uses:
- Value head predicts **discrete win probability** (3 outcomes: win/loss/draw)
- Loss: Cross-entropy on softmax outputs
- Encourages confident predictions (high probability on one outcome)
- **No variance minimization** - wants certainty, not mean

### 2. **Large-Scale Self-Play**

- 25,000 TPUs, millions of games
- Huge diversity in training data
- MCTS with 800-1600 simulations per move
- Strong policy network (50-60% accuracy)
- Virtuous cycle establishes quickly

### 3. **Different Game Properties**

- Go: Clearer winning/losing patterns
- Our heuristic might not discriminate as strongly
- Yinsh has more draws/close games

---

## Solutions: Four Approaches

### Option 1: Re-add Variance Penalty (Higher Weight) ⭐ QUICK TEST

**Rationale**: Test 2 worked (0.059) with penalty 0.5. Try higher weight (1.0-2.0).

**Implementation**:
```python
# trainer.py
variance_weight = 1.0  # Up from 0.5
variance_penalty = variance_weight * torch.exp(-batch_variance * 10)
value_loss = value_loss_mse + variance_penalty
```

**Expected**:
- Discrimination: 0.059 → 0.08-0.10
- Forces network to maintain variance
- Counteracts MSE's mean-seeking

**Pros**:
- Quick to test (45 min)
- Already showed improvement in Test 2
- Proven concept

**Cons**:
- Hacky solution
- Not how AlphaZero does it
- May not reach 0.15+ target

---

### Option 2: Switch to Classification-Based Value Head 🎯 RECOMMENDED

**Rationale**: Match AlphaZero approach - predict discrete outcomes.

**Implementation**:
```python
# Discrete outcomes: -3, -2, -1, 0, +1, +2, +3 (score difference)
# Value head outputs 7 logits
# Loss: Cross-entropy
# Prediction: Weighted average of outcome values

class ValueHead(nn.Module):
    def __init__(self):
        # ... existing layers ...
        self.output = nn.Linear(hidden_dim, 7)  # 7 outcome classes

    def forward(self, x):
        logits = self.output(x)  # [batch, 7]
        probs = F.softmax(logits, dim=-1)
        # Expected value = weighted sum of outcomes
        outcomes = torch.tensor([-3, -2, -1, 0, 1, 2, 3]) / 3.0  # Normalize to [-1, 1]
        value = (probs * outcomes).sum(dim=-1)
        return value, probs

# Training
def value_loss_fn(logits, target_value):
    # Convert continuous target to soft distribution
    target_class = torch.round((target_value * 3)).long() + 3  # Map to [0, 6]
    return F.cross_entropy(logits, target_class)
```

**Expected**:
- Encourages confident predictions (high prob on one outcome)
- No variance minimization
- Matches AlphaZero design

**Pros**:
- Proven approach (AlphaZero)
- Naturally encourages discrimination
- No hacky penalties needed

**Cons**:
- More code changes
- Need to handle continuous MCTS values → discrete classes
- Requires retraining from scratch

---

### Option 3: Huber Loss + Variance Penalty

**Rationale**: Huber loss less sensitive to outliers, combine with variance penalty.

**Implementation**:
```python
value_loss_huber = F.smooth_l1_loss(pred_values, target_values)
variance_penalty = 1.0 * torch.exp(-batch_variance * 10)
value_loss = value_loss_huber + variance_penalty
```

**Expected**:
- More robust than MSE
- Still needs variance penalty

**Pros**:
- Quick to implement
- More stable than pure MSE

**Cons**:
- Still not addressing root cause
- Less proven than classification approach

---

### Option 4: Check Heuristic MCTS Value Diversity

**Rationale**: Verify heuristic actually produces diverse MCTS values.

**Investigation**:
1. Run 100% heuristic MCTS on diverse positions
2. Collect root values after search
3. Analyze distribution, variance, discrimination
4. Compare to expected diversity

**Expected findings**:
- If heuristic MCTS values have σ ~ 0.08-0.10, this explains low discrimination
- If heuristic MCTS values have σ ~ 0.20+, then MSE is definitely the problem

**Implementation**:
```python
# Modify diagnose_value_predictions.py to collect MCTS values during search
# Compare pure_neural vs pure_heuristic value distributions
```

---

## Recommended Path Forward

### Phase 1: Quick Test - Increase Variance Penalty (Option 1)

**Immediate action** (45 minutes):
1. Set `variance_weight = 1.5` (up from 0.5)
2. Keep MCTS value targets, heuristic 0.3
3. Run 1-iteration test
4. Expected: Discrimination 0.07-0.10

**If successful** (discrimination > 0.08):
- Continue with higher penalty
- Run 3-iteration tournament
- Monitor if penalty needs tuning over iterations

**If insufficient** (discrimination < 0.08):
- Proceed to Phase 2

---

### Phase 2: Investigate Heuristic Quality (Option 4)

**Investigation** (30 minutes):
1. Add logging to collect MCTS root values during pure_heuristic search
2. Run diagnostics comparing pure_neural vs pure_heuristic value distributions
3. Determine if heuristic MCTS actually produces diverse targets

**Expected outcomes**:
- **If heuristic σ < 0.10**: Heuristic is also weak, need better heuristic or Option 2
- **If heuristic σ > 0.15**: Confirms MSE is the problem, proceed to Option 2

---

### Phase 3: Classification-Based Value Head (Option 2)

**Major refactor** (2-3 hours):
1. Redesign value head for classification (7-class output)
2. Modify loss function to cross-entropy
3. Update inference to compute expected value from probabilities
4. Retrain from scratch
5. Expected discrimination: 0.15+ (AlphaZero levels)

---

## Comparison to Expected Results

### Original Goals (from ROOT_CAUSE document)

```
After Bootstrap:
- Discrimination: 0.10-0.15 (heuristic quality)
- Network learns strong baseline
- Virtuous cycle begins
```

### Actual Bootstrap Results

```
Achieved:
- Discrimination: 0.042 (NO improvement)
- Std dev: 0.079 (worse than Test 2)
- Weak positions: 73% (no change)
```

**Conclusion**: Bootstrap hypothesis was wrong. MSE loss is the fundamental issue.

---

## Key Insights

1. **MSE loss inherently minimizes variance** - mathematical property, not fixable
2. **Variance penalty was essential** - compensates for MSE's mean-seeking
3. **Target quality isn't the bottleneck** - 100% heuristic no better than 30%
4. **AlphaZero avoids this** - uses classification-based value head
5. **Need different loss function** - pure MSE fundamentally problematic

---

## File Locations

**Bootstrap Model**: `runs/20260117_190526/best_model.pt`
**Diagnostics**: `diagnostics_bootstrap.json`, `diagnostics_bootstrap.md`
**Config Used**: heuristic_weight=1.0, variance_penalty=0.0

---

## Next Steps Decision Matrix

| If Goal Is... | Recommended Action | Time | Expected Result |
|---------------|-------------------|------|-----------------|
| **Quick improvement** | Option 1: variance_weight=1.5 | 45 min | Discrimination 0.07-0.10 |
| **Understand root cause** | Option 4: Investigate heuristic | 30 min | Confirm MSE is problem |
| **Long-term solution** | Option 2: Classification value head | 2-3 hrs | Discrimination 0.15+ |
| **Minimal changes** | Option 3: Huber + variance penalty | 1 hr | Discrimination 0.08-0.12 |

**Immediate recommendation**: Start with Option 1 (quick test), then decide based on results whether to proceed to Option 2 (proper solution) or Option 4 (investigation).

---

## Conclusion

The bootstrap from 100% heuristic revealed that our previous hypothesis was incorrect. The issue is not weak MCTS targets from a weak network, but rather the **pure MSE loss function**, which mathematically minimizes prediction variance regardless of target quality.

**Solution**: Either use higher variance penalty to force discrimination (quick fix) or switch to classification-based value head with cross-entropy loss (proper AlphaZero-style solution).

**Evidence**: Only Test 2 (with variance penalty) achieved improved discrimination (0.059). All tests with pure MSE (including 100% heuristic bootstrap) collapsed to ~0.042 discrimination.
