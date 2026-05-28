# Reference Model System

## Overview

The Reference Model System provides consistent Elo baselines across different experiments by preserving specific trained models as fixed tournament opponents.

Instead of abstract "control agents," this system uses real trained models (like `value_recovery_20250630` iteration 2) that all future experiments automatically play against in tournaments.

## Key Benefits

✅ **Consistent Elo Comparisons**: All experiments play against the same reference models  
✅ **Real Model Baselines**: Use actual trained models, not artificial opponents  
✅ **Minimal Integration**: Just one line of code to enable  
✅ **Automatic Tournament Inclusion**: No manual setup required per experiment  
✅ **Preserved Model Files**: Models are copied and preserved for reproducibility  

## Quick Start

### 1. Set Up Baseline Reference

```bash
python scripts/setup_reference_baseline.py
```

This preserves `value_recovery_20250630` iteration 2 as "baseline_reference" (Elo 1516.9).

### 2. Integrate with Training Script

Add this single line to your training script:

```python
from yinsh_ml.utils.reference_models import ReferenceModelManager, integrate_reference_models_with_tournament

# Initialize reference manager
reference_manager = ReferenceModelManager(project_root)

# Integrate with existing tournament system
integrate_reference_models_with_tournament(tournament_manager, reference_manager)

# Now all tournaments automatically include reference models!
tournament_manager.run_round_robin_tournament(current_iteration)
```

### 3. Run Experiments

Your tournaments now automatically include the reference baseline. No other changes needed!

## What Happens During Tournaments

1. **Normal Round-Robin**: Current models play each other (unchanged)
2. **Reference Matches**: Current models automatically play against reference models
3. **Elo Updates**: All ratings updated including reference model ratings
4. **Enhanced Logging**: Results show both current and reference performance

Example output:
```
Running round-robin tournament...
iteration_5 vs iteration_4: iteration_5 wins 65%

Running matches against 1 reference models...
iteration_5 vs baseline_reference: 58.3% win rate

Updated ratings (including references):
  iteration_5: 1547.8
Reference models:
  baseline_reference: 1516.9
```

## Managing Reference Models

### View Current Reference Models

```python
from yinsh_ml.utils.reference_models import ReferenceModelManager

reference_manager = ReferenceModelManager(project_root)
models = reference_manager.get_reference_models()

for name, info in models.items():
    print(f"{name}: {info.description} (Elo {info.elo_rating})")
```

### Add New Reference Model

```python
# Preserve the best model from a new experiment
reference_manager.preserve_model(
    experiment_name="new_experiment_20250701",
    iteration=8,
    model_path=Path("results/new_experiment_20250701/checkpoint_iteration_8.pt"),
    elo_rating=1623.4,  # Known Elo from tournament results
    reference_name="strong_baseline",
    description="Strong model from new_experiment_20250701 iter 8",
    games_per_tournament=100
)
```

### Remove Reference Model

```python
reference_manager.remove_reference_model("baseline_reference")
```

## File Structure

```
reference_models/
├── reference_models.json      # Configuration file
└── baseline_reference.pt      # Preserved model checkpoint
```

The configuration tracks:
- Original experiment and iteration
- Known Elo rating
- Preservation date
- Model file path
- Description
- Games per tournament

## Integration Details

The `integrate_reference_models_with_tournament()` function:

1. **Wraps** your existing tournament method
2. **Preserves** normal tournament behavior
3. **Adds** reference model matches automatically
4. **Updates** Elo ratings including reference models
5. **Logs** enhanced results

No changes to your existing tournament code required!

## Use Cases

### Cross-Experiment Comparison
```
Experiment A iteration 10: 1523 Elo vs baseline_reference (1516)
Experiment B iteration 8:  1578 Elo vs baseline_reference (1516)
```
Direct comparison: Experiment B is stronger than Experiment A.

### Progress Tracking
Track improvement against a consistent baseline across experiments.

### Regression Detection
If a new model performs worse against the baseline than previous models, you've potentially regressed.

### Ablation Studies
Compare architectural changes against the same reference point.

## Example Scripts

- `scripts/setup_reference_baseline.py`: Set up the default baseline
- `scripts/example_tournament_integration.py`: Integration examples

## Best Practices

1. **Choose Stable References**: Use models from completed experiments with known good performance
2. **Document Elo Ratings**: Include the known Elo when preserving models
3. **Update Descriptions**: Add meaningful descriptions for future reference
4. **Multiple References**: Consider preserving multiple reference models for different strength levels
5. **Clean Up**: Remove outdated reference models that are no longer useful

## API Reference

### ReferenceModelManager

#### `__init__(project_root: Path)`
Initialize the reference model manager.

#### `preserve_model(experiment_name, iteration, model_path, elo_rating, reference_name=None, description="", games_per_tournament=100)`
Preserve a model as a reference baseline.

#### `get_reference_models() -> Dict[str, ReferenceModelInfo]`
Get all available reference models.

#### `load_reference_model(name: str, device='cpu') -> NetworkWrapper`
Load a reference model for tournament use.

#### `remove_reference_model(name: str) -> bool`
Remove a reference model.

### Integration Function

#### `integrate_reference_models_with_tournament(tournament_manager, reference_manager)`
Integrate reference models with existing tournament system.

## Troubleshooting

### "No reference models found"
Run `python scripts/setup_reference_baseline.py` first.

### "Reference model file not found"
The preserved model file was moved or deleted. Re-preserve the model.

### Integration not working
Ensure you call `integrate_reference_models_with_tournament()` before running tournaments.

### Memory issues
Reduce `games_per_tournament` in reference model configurations if running out of memory. 