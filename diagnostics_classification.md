# Value Prediction Diagnostics Report

## Executive Summary

### Value Prediction Distribution

- **Range**: [-0.724, 0.657]
- **Mean**: 0.009 ± 0.186
- **High Confidence** (|v| > 0.7): 0.0%
- **Low Confidence** (|v| < 0.3): 90.3%
- **Median**: 0.027

### Value Discrimination Between Moves

- **Mean Std Dev**: 0.082
- **Weak Discrimination** (<0.05 std): 23.0%

## Interpretation

⚠️ **CRITICAL**: Very few high-confidence predictions (<5%).
The network rarely makes strong predictions, which limits MCTS guidance.

## Recommendations

3. **Consider architectural changes**
   - Increase network capacity (20 blocks instead of 12)
   - Add value head normalization layers

