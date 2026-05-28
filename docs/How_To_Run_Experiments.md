# How to Run Experiments in YinshML

This guide provides step-by-step instructions for running training experiments, monitoring progress, and analyzing results using YinshML's experiment tracking system.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Running Training Experiments](#running-training-experiments)
- [One-Command Launch](#one-command-launch)
- [Monitoring Training Progress](#monitoring-training-progress)
- [Analyzing Results](#analyzing-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Environment Setup

Before running any commands, ensure your environment is properly set up:

```bash
# Always set the Python path for YinshML
export PYTHONPATH=/Users/jackfleming/PycharmProjects/YinshML

# If using a virtual environment, activate it first
# source venv/bin/activate
```

## Running Training Experiments

YinshML uses configuration-based training. Select the appropriate config for your experiment.

### List Available Configurations

```bash
python -c "from experiments.config import COMBINED_EXPERIMENTS; print(list(COMBINED_EXPERIMENTS.keys()))"
```

### Run a Training Experiment

```bash
python experiments/runner.py --config <CONFIG_NAME> --device <DEVICE> [--debug]
```

Where:
- `<CONFIG_NAME>`: Name of the configuration (e.g., `smoke`, `short_baseline`, `separate_value_head_smoke`)
- `<DEVICE>`: Computing device to use (`mps` for Mac GPUs, `cuda` for NVIDIA GPUs, or `cpu`)
- `--debug`: Optional flag to enable detailed debug logging

#### Example Commands

```bash
# Quick smoke test (fast, minimal resources)
python experiments/runner.py --config smoke --device mps --debug

# Short baseline run
python experiments/runner.py --config short_baseline --device mps

# Value head architecture test
python experiments/runner.py --config separate_value_head_smoke --device mps
```

## One-Command Launch

If you prefer a single entry-point that **handles training, TensorBoard, and live experiment monitoring** in one go, use the new `yinsh-track launch` command (added in v0.1.1).

### One-Time Setup (after pulling new code)

1. Activate your virtual-env as usual.
2. Install / re-install the project in *editable* mode so the refreshed console scripts are linked:

   ```bash
   pip install -e .
   ```

3. Verify the CLI is on your path (should print a path inside your venv):

   ```bash
   which yinsh-track
   # → /path/to/venv/bin/yinsh-track
   ```

   • If that still returns nothing, re-activate the venv (`source venv/bin/activate`) or fall back to the module form shown below.

4. (Optional) Test the CLI loads:

   ```bash
   yinsh-track --help | head
   ```

5. You are ready to launch.

**Fallback:** If your shell still cannot locate the script, you can always invoke the launcher via the module entry-point:

```bash
python -m yinsh_ml.cli.main launch <CONFIG_NAME> --device <DEVICE>
```

### Quick Start

```bash
# start a smoke test on Apple GPU, TensorBoard & monitor enabled by default
yinsh-track launch smoke --device mps

# full run on CUDA, verbose logs, but skip TensorBoard
yinsh-track launch full --device cuda --debug --no-tensorboard
```

#### Command Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--device [cuda|mps|cpu]` | Compute device to use | `mps` if available, else `cpu` |
| `--debug` | Forward debug flag to the underlying runner | off |
| `--no-tensorboard` | Don't auto-start TensorBoard | off |
| `--no-monitor` | Don't show the live *running experiment* list | off |

Under the hood the launcher:

1. Ensures the repo root is on `PYTHONPATH`.
2. Starts TensorBoard at `http://localhost:6006` (unless disabled).
3. Spawns a background thread that prints the list of *running* experiments every 30 s.
4. Invokes `experiments/runner.py` with the supplied config/device/debug flags.
5. Cleans up all background processes when the training run finishes or you hit **Ctrl-C**.

This is the fastest way to kick off a fully-instrumented training session with minimal typing.

## Monitoring Training Progress

YinshML provides multiple ways to monitor your training experiments.

### Real-time Training Logs

Training logs are displayed in the terminal where you run the training command. These show:
- Memory usage and optimization statistics
- Game generation progress
- Training loss values
- Tournament evaluation results

### Experiment Tracking CLI

The experiment tracking CLI provides commands to monitor and manage experiments.

#### List All Experiments

```bash
python -m yinsh_ml.cli.main list
```

#### List Running Experiments

```bash
python -m yinsh_ml.cli.main list --status running
```

#### Search Experiments

```bash
python -m yinsh_ml.cli.main search --tags "training" --date-after "2025-05-01"
```

### TensorBoard Dashboard

YinshML integrates with TensorBoard to provide detailed visualizations of training metrics.

#### Launch TensorBoard

```bash
tensorboard --logdir logs/
```

Then open [http://localhost:6006](http://localhost:6006) in your browser to view:
- Loss curves
- Accuracy metrics
- Model performance visualizations
- Memory usage statistics

## Analyzing Results

After experiments complete, analyze the results using the following tools.

### Compare Experiments

Compare metrics from multiple experiments:

```bash
# Compare specific experiments by ID
python -m yinsh_ml.cli.main compare 1 2 3

# Include configuration differences
python -m yinsh_ml.cli.main compare 1 2 --include-config

# Focus on specific metrics
python -m yinsh_ml.cli.main compare 1 2 --metrics policy_loss value_loss elo_rating

# Add statistical significance testing
python -m yinsh_ml.cli.main compare 1 2 --statistical

# Export comparison data
python -m yinsh_ml.cli.main compare 1 2 --format json > comparison.json
python -m yinsh_ml.cli.main compare 1 2 --format csv > comparison.csv
```

### Examine Result Files

Training results and model checkpoints are saved in the `results` directory:

```bash
# List result directories
ls -la results/

# View metrics for a specific experiment
cat results/<experiment_name>/final_summary_metrics.json

# Check specific iteration metrics
cat results/<experiment_name>/iteration_<N>/metrics.json
```

### Reproduce an Experiment

To reproduce a specific experiment:

```bash
python -m yinsh_ml.cli.main reproduce <experiment_id>
```

## Advanced Usage

### Running in Multiple Terminals

For improved monitoring, run commands in separate terminals:

1. **Terminal 1**: Run training
   ```bash
   export PYTHONPATH=/Users/jackfleming/PycharmProjects/YinshML
   python experiments/runner.py --config separate_value_head_smoke --device mps
   ```

2. **Terminal 2**: Monitor experiments
   ```bash
   export PYTHONPATH=/Users/jackfleming/PycharmProjects/YinshML 
   python -m yinsh_ml.cli.main list --status running
   ```

3. **Terminal 3**: View TensorBoard
   ```bash
   export PYTHONPATH=/Users/jackfleming/PycharmProjects/YinshML
   tensorboard --logdir logs/
   ```

### Configuration Options

Common configuration options to try:

- `smoke`: Ultra-quick test run (3 iterations, minimal games)
- `short_baseline`: Medium-length baseline run
- `separate_value_head_smoke`: Test run with separate value head architecture
- `value_head_config`: Full training with value head optimization

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure `PYTHONPATH` is set correctly: `export PYTHONPATH=/Users/jackfleming/PycharmProjects/YinshML`

2. **CUDA not available**:
   - On Mac, use `--device mps` instead of `cuda`
   - On systems without GPU, use `--device cpu`

3. **Memory errors during training**:
   - Try a configuration with smaller batch size or fewer games
   - Use the `smoke` configuration to test your setup

4. **Experiment tracking errors**:
   - Check database at `experiments/tracking.db`
   - Ensure TensorBoard logs directory exists and is writable

### Getting Help

If you encounter issues not covered here, check:
- The code documentation in the `yinsh_ml` module
- The TensorBoard logs for detailed metrics
- The terminal output for specific error messages

## References

- [YinshML Memory System Documentation](./memory_system.md)
- [Experiment Tracking System Design](./experiment_tracking.md)
- [Model Architecture Overview](./model_architecture.md) 