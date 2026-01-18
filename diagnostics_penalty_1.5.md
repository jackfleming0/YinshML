# Value Prediction Diagnostics Report

## Executive Summary

### Value Prediction Distribution

- **Range**: [-0.367, 0.414]
- **Mean**: 0.007 ± 0.109
- **High Confidence** (|v| > 0.7): 0.0%
- **Low Confidence** (|v| < 0.3): 99.6%
- **Median**: 0.007

### Value Discrimination Between Moves

- **Mean Std Dev**: 0.056
- **Weak Discrimination** (<0.05 std): 43.0%

## Interpretation

⚠️ **CRITICAL**: Very few high-confidence predictions (<5%).
The network rarely makes strong predictions, which limits MCTS guidance.

⚠️ **CRITICAL**: Very low standard deviation (<0.15).
Predictions are clustered around the mean, providing weak discrimination.

## Recommendations

1. **Modify loss function** to encourage confident predictions
   - Add confidence penalty term
   - Reward predictions near -1 or +1
   - Penalize predictions near 0

3. **Consider architectural changes**
   - Increase network capacity (20 blocks instead of 12)
   - Add value head normalization layers

