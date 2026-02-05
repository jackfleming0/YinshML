# OOM Prevention Plan

## Root Cause Analysis

### Identified Memory Accumulation Points

1. **Tournament Round-Robin Loading** (PRIMARY CAUSE)
   - Location: `yinsh_ml/utils/tournament.py:428-432`
   - Issue: ALL model checkpoints loaded into memory simultaneously
   - Impact: 10 iterations × ~130MB = 1.3GB just for tournament models
   - This happens AFTER training, when memory is already elevated

2. **Cumulative Iteration Memory**
   - Each iteration adds to replay buffer (up to 20K samples)
   - MCTS tree structures not fully released between games
   - MPS (Apple Silicon GPU) virtual memory allocations grow

3. **Training Phase Memory**
   - More epochs = longer training = more gradient accumulation tracking
   - 8 epochs vs 4 epochs extends the window for memory leaks

### Memory Profile During Experiment

```
Self-Play Phase:     ~2-3 GB
Training Phase:      ~4-5 GB (8 epochs pushes to ~6GB)
Tournament Phase:    ~6-9 GB (loads ALL models)
                     ↑ OOM THRESHOLD ~8-9GB
```

## Solutions

### 1. Tournament Memory Optimization (HIGH PRIORITY)

**Option A: Lazy Model Loading**
```python
# In tournament.py, modify run_full_round_robin_tournament()
# Instead of loading all models upfront:

def _get_model(self, model_id, path):
    """Load model on-demand, cache only current pair."""
    if model_id not in self._active_models:
        # Clear previous models
        self._active_models.clear()
        gc.collect()
        torch.mps.empty_cache()
        self._active_models[model_id] = self._load_model(path)
    return self._active_models[model_id]
```

**Option B: Limit Tournament Participants**
- Only include last N iterations (e.g., N=5) in round-robin
- Compare new model only against current best + baseline

**Option C: Stream-Based Tournament**
- Load model pair → play match → unload → repeat
- Higher I/O but prevents memory accumulation

### 2. Training Phase Memory (MEDIUM PRIORITY)

Add explicit memory cleanup after each epoch:
```python
# In trainer.py, after each epoch:
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

### 3. Self-Play Memory (LOW PRIORITY)

Already implemented via batch processing (10 games at a time).
Consider reducing batch size if needed: `--games-per-batch 5`

### 4. Configuration Options

Add to experiment config:
```yaml
memory:
  tournament_max_models: 5  # Limit round-robin participants
  clear_cache_after_epoch: true
  aggressive_gc: true  # Multiple gc.collect() calls
```

## Implementation Priority

1. **Immediate**: Implement lazy model loading in tournament
2. **Short-term**: Add tournament participant limit config
3. **Medium-term**: Memory monitoring/alerting system

## Monitoring

Add memory checkpoint logging:
```python
def log_memory_checkpoint(phase: str):
    import psutil
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[MEMORY] {phase}: {mem_mb:.1f} MB")
```

Call at: iteration start, post-selfplay, post-training, pre-tournament, post-tournament

## Testing

To test memory-constrained scenarios:
```bash
# Limit Python memory (Linux/Mac)
ulimit -v 8000000  # 8GB limit
python scripts/run_experiment.py ...
```
