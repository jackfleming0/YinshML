# Checkpoint Retention Policy

## Current State

- **Total checkpoints**: 18 files
- **Total size**: 2.3 GB
- **Per checkpoint**: ~131 MB
- **Per experiment**: 262-523 MB

### Current Storage Pattern
```
experiments/<exp_id>/
├── best_model.pt           # 131 MB (KEEP - best performing)
├── iteration_0/
│   └── checkpoint_iteration_0.pt  # 131 MB (often redundant)
├── iteration_3/
│   └── checkpoint_iteration_3.pt  # 131 MB (if promoted)
├── iteration_8/
│   └── checkpoint_iteration_8.pt  # 131 MB (if promoted)
└── replay_buffer_snapshot.pkl.gz  # ~1 MB
```

## Retention Policy

### What to KEEP

1. **best_model.pt** - Always keep the best performing model
2. **Final iteration checkpoint** - Keep for potential resume
3. **Replay buffer snapshot** - Small, useful for analysis

### What to DELETE (Safe to Remove)

1. **Intermediate checkpoints** - Iterations that weren't promoted
2. **Duplicate of best_model** - If checkpoint_iteration_X.pt == best_model.pt

### Recommended Retention Rules

| File Type | Keep | Delete After |
|-----------|------|--------------|
| best_model.pt | Always | Never |
| Final checkpoint | Always | Never |
| Promoted checkpoints | 30 days | After analysis |
| Non-promoted checkpoints | Immediately | After iteration completes |
| Replay buffer | Always | Never (small) |

## Implementation

### Manual Cleanup Script

```bash
#!/bin/bash
# cleanup_checkpoints.sh - Remove non-essential checkpoints

EXPERIMENTS_DIR="experiments"

for exp_dir in "$EXPERIMENTS_DIR"/*/; do
    if [ ! -d "$exp_dir" ]; then continue; fi

    # Get the best model iteration from best_model_state.json
    best_iter=$(jq -r '.best_iteration' "$exp_dir/best_model_state.json" 2>/dev/null)

    # Get total iterations
    total_iter=$(ls -d "$exp_dir"/iteration_*/ 2>/dev/null | wc -l)
    last_iter=$((total_iter - 1))

    for iter_dir in "$exp_dir"/iteration_*/; do
        iter_num=$(basename "$iter_dir" | sed 's/iteration_//')

        # Keep best and last iteration
        if [ "$iter_num" != "$best_iter" ] && [ "$iter_num" != "$last_iter" ]; then
            ckpt="$iter_dir/checkpoint_iteration_${iter_num}.pt"
            if [ -f "$ckpt" ]; then
                echo "Removing: $ckpt"
                rm "$ckpt"
            fi
        fi
    done
done
```

### Automatic Cleanup in Code

Add to `experiment_runner.py` after each iteration:
```python
def cleanup_old_checkpoints(self, current_iteration: int, keep_best: bool = True):
    """Remove checkpoints that are no longer needed."""
    best_iter = self.supervisor.best_iteration

    for i in range(current_iteration):
        if i == best_iter and keep_best:
            continue
        if i == current_iteration - 1:  # Keep previous for safety
            continue

        ckpt_path = self.output_dir / f"iteration_{i}" / f"checkpoint_iteration_{i}.pt"
        if ckpt_path.exists():
            ckpt_path.unlink()
            logger.info(f"Cleaned up checkpoint: {ckpt_path}")
```

### Configuration Option

Add to experiment config:
```yaml
checkpointing:
  keep_all: false  # Set true for debugging
  keep_best: true
  keep_last_n: 2   # Keep last N iterations
  auto_cleanup: true
```

## Estimated Savings

| Scenario | Current | After Cleanup | Savings |
|----------|---------|---------------|---------|
| 10-iter experiment | 4 checkpoints (524 MB) | 2 checkpoints (262 MB) | 50% |
| 5 experiments | 2.3 GB | ~1.0 GB | 1.3 GB |

## Quick Cleanup Commands

```bash
# List all checkpoints with sizes
find experiments -name "*.pt" -exec ls -lh {} \;

# Count checkpoints per experiment
for d in experiments/*/; do echo "$d: $(find "$d" -name "*.pt" | wc -l) checkpoints"; done

# Remove ALL non-best checkpoints (DANGEROUS - backup first!)
# find experiments -path "*/iteration_*/checkpoint_*.pt" -delete

# Safer: List what would be deleted
find experiments -path "*/iteration_*/checkpoint_*.pt" -newer experiments/*/best_model.pt
```

## Archive Strategy

For completed experiments that need long-term storage:
```bash
# Archive an experiment (compress checkpoints)
tar -czvf archive/baseline_001.tar.gz experiments/e23c46af/

# Keep only best_model.pt in active directory
rm experiments/e23c46af/iteration_*/checkpoint_*.pt
```
