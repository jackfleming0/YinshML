# Value Prediction Diagnostics Report

## Executive Summary

### Value Prediction Distribution

- **Range**: [-0.259, 0.324]
- **Mean**: 0.028 ± 0.079
- **High Confidence** (|v| > 0.7): 0.0%
- **Low Confidence** (|v| < 0.3): 100.0%
- **Median**: 0.024

### Value Discrimination Between Moves

- **Mean Std Dev**: 0.042
- **Weak Discrimination** (<0.05 std): 73.0%

## Interpretation

⚠️ **CRITICAL**: Very few high-confidence predictions (<5%).
The network rarely makes strong predictions, which limits MCTS guidance.

⚠️ **CRITICAL**: Very low standard deviation (<0.15).
Predictions are clustered around the mean, providing weak discrimination.

⚠️ **CRITICAL**: Weak discrimination between moves (>50%).
Values don't vary much between different moves at the same position.
This means MCTS can't effectively use values to prune bad branches.

## Recommendations

1. **Modify loss function** to encourage confident predictions
   - Add confidence penalty term
   - Reward predictions near -1 or +1
   - Penalize predictions near 0

2. **Improve training data quality**
   - Bootstrap from stronger baseline (heuristic MCTS)
   - Ensure diverse, informative positions
   - Check if value targets are too clustered

3. **Consider architectural changes**
   - Increase network capacity (20 blocks instead of 12)
   - Add value head normalization layers

