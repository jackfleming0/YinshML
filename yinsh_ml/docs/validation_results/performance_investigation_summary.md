# Performance Investigation Summary

## Issue Identified

Initial tournament runs showed extremely slow performance:
- 10 games took over 5 hours (should be minutes)
- Average move time showing as 6542 seconds (clearly incorrect)
- Some individual moves taking 256+ seconds

## Root Causes Found

### 1. Calculation Bug (Fixed)
**Problem**: `average_move_time` was calculated incorrectly - dividing total duration by number of games instead of number of moves.

**Fix**: Updated `_aggregate_results()` in `tournament.py` to:
- Track actual number of moves made by heuristic agent (`num_moves` field)
- Calculate `avg_move_time = total_duration / total_moves` instead of `/ total_games`

### 2. Performance Bottleneck: Forced Sequence Detection (Fixed)
**Problem**: `detect_forced_sequences()` was causing exponential slowdowns:
- Called on every position evaluation during search
- Recursive search through all valid moves (up to 77 moves in ring placement phase)
- With depth 3 search, this resulted in 443,862+ recursive calls
- Turn 8 took 256 seconds for just 26 nodes evaluated

**Fixes Applied**:
1. Added phase check in `evaluator.py` to skip forced sequence detection during RING_PLACEMENT phase
2. Added early exit in `_analyze_forced_sequences()` for non-MAIN_GAME phases
3. Limited branching factor to first 10 moves in forced sequence analysis
4. Updated tournament script to use faster config (depth 2, no iterative deepening, 0.5s time limit)

### 3. Configuration Too Slow for Tournaments (Fixed)
**Problem**: Default config (depth 3 + iterative deepening + 1.0s time limit) was too slow for large-scale tournaments.

**Fix**: Tournament script now uses optimized config:
- `max_depth=2` (reduced from 3)
- `time_limit_seconds=0.5` (reduced from 1.0)
- `max_branching_factor=20` (reduced from 24)
- `use_iterative_deepening=False` (disabled for speed)

## Performance Improvements

### Before Optimizations:
- Turn 8: 256 seconds for 26 nodes
- Average move time: 28+ seconds
- Estimated game time: 2456 seconds (~41 minutes)

### After Optimizations:
- Turn 8: 1.094 seconds for 2 nodes
- Average move time: 0.612 seconds
- Estimated game time: 53 seconds (~1 minute)
- **Improvement: ~46x faster**

### Heuristic Evaluation Performance:
- **Average**: 0.026 ms (well under 1ms target)
- **Maximum**: 0.229 ms (well under 10ms target)
- **Throughput**: 38,319 evaluations/second

## Success Criteria Verification

✅ **Win Rate**: 100% (exceeds 60% target)
✅ **Heuristic Evaluation Time**: 0.026 ms (well under 1ms target)
✅ **Max Evaluation Time**: 0.229 ms (well under 10ms target)

## Files Modified

1. `yinsh_ml/agents/tournament.py`:
   - Fixed `average_move_time` calculation to use total moves instead of total games
   - Added `num_moves` tracking in worker function

2. `yinsh_ml/heuristics/evaluator.py`:
   - Added phase check to skip forced sequence detection during RING_PLACEMENT
   - Added `GamePhase` import

3. `yinsh_ml/heuristics/forced_sequences.py`:
   - Added early exit for non-MAIN_GAME phases in `_analyze_forced_sequences()`
   - Limited branching factor to first 10 moves for performance

4. `scripts/run_final_validation.py`:
   - Updated to use faster tournament config (depth 2, no iterative deepening)

5. `scripts/generate_validation_report.py`:
   - Added heuristic evaluation benchmarking
   - Updated success criteria to check actual evaluation time (not move selection time)
   - Clarified distinction between evaluation time and move selection time

## Recommendations

1. **For Tournament Validation**: Use the optimized config (depth 2, no iterative deepening)
2. **For Strong Play**: Use default config (depth 3, iterative deepening) but expect slower performance
3. **For Fast Evaluation**: Use depth 1 or fast mode in `HeuristicPolicy`

## Next Steps

The heuristic system now meets all performance requirements and is ready for:
- Integration into AlphaZero training pipeline
- Use as baseline opponent
- Large-scale tournament validation (1000+ games)

