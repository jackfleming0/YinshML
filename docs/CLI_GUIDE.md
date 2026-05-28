# YinshML CLI Guide

The `yinsh-track` command-line interface provides a comprehensive set of tools for managing machine learning experiments. This guide covers all commands, options, and best practices.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Global Options](#global-options)
- [Commands](#commands)
  - [start](#start---create-experiments)
  - [list](#list---view-experiments)
  - [compare](#compare---compare-experiments)
  - [reproduce](#reproduce---reproduce-experiments)
  - [search](#search---search-experiments)
  - [config](#config---view-configuration)
- [Configuration](#configuration)
- [Shell Completion](#shell-completion)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

Install YinshML with pip:

```bash
pip install yinsh-ml
```

The CLI tool `yinsh-track` will be available after installation.

## Quick Start

1. **Create your first experiment:**
   ```bash
   yinsh-track start "My First Experiment" --description "Testing the CLI"
   ```

2. **List all experiments:**
   ```bash
   yinsh-track list
   ```

3. **Compare experiments:**
   ```bash
   yinsh-track compare 1 2 --metrics accuracy loss
   ```

## Global Options

These options are available for all commands:

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to configuration file |
| `--database`, `--db` | Path to experiment database |
| `--verbose`, `-v` | Enable verbose output |
| `--quiet`, `-q` | Suppress non-essential output |
| `--no-color` | Disable colored output |
| `--install-completion` | Generate shell completion script |
| `--version` | Show version information |
| `--help` | Show help message |

### Examples:
```bash
# Use custom database
yinsh-track --database /path/to/experiments.db list

# Verbose mode for debugging
yinsh-track --verbose start "Debug Experiment"

# Quiet mode for scripts
yinsh-track --quiet list --format json > experiments.json
```

## Commands

### `start` - Create Experiments

Create and start tracking a new experiment.

**Syntax:**
```bash
yinsh-track start NAME [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--description`, `-d` | Description of the experiment |
| `--tags`, `-t` | Tags for organization (multiple allowed) |
| `--config-file`, `--config` | JSON configuration file |
| `--parameter`, `-p` | Parameters in key=value format (multiple allowed) |

**Examples:**
```bash
# Basic experiment
yinsh-track start "Model v2.1" --description "Testing new architecture"

# With tags and parameters
yinsh-track start "Hyperparameter Search" \
    --tags ml --tags tuning \
    --parameter lr=0.001 --parameter batch_size=32

# Using configuration file
yinsh-track start "Production Run" \
    --config config/model.json \
    --description "Final model training"

# Complex parameter types
yinsh-track start "A/B Test" \
    --parameter model_type=transformer \
    --parameter dropout=0.1 \
    --parameter use_attention=true
```

**Parameter Types:**
- Strings: `--parameter name=value`
- Numbers: `--parameter lr=0.001` (auto-detected)
- Booleans: `--parameter debug=true` or `--parameter debug=false`

### `list` - View Experiments

List and filter experiments with detailed information.

**Syntax:**
```bash
yinsh-track list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--status`, `-s` | Filter by status (running, done, failed, etc.) |
| `--tags`, `-t` | Filter by tags (comma-separated) |
| `--limit`, `-l` | Maximum number of results (default: 20) |
| `--format`, `-f` | Output format: table, json, csv |
| `--sort` | Sort by: id, name, status, created, updated |
| `--reverse` | Reverse sort order |
| `--date-from` | Filter from date (YYYY-MM-DD) |
| `--date-to` | Filter to date (YYYY-MM-DD) |

**Examples:**
```bash
# List all experiments
yinsh-track list

# Filter by status
yinsh-track list --status running
yinsh-track list --status done --limit 5

# Filter by tags
yinsh-track list --tags "ml,training"
yinsh-track list --tags "production"

# Date range filtering
yinsh-track list --date-from 2024-01-01 --date-to 2024-12-31
yinsh-track list --date-from 2024-06-01

# Sorting and ordering
yinsh-track list --sort name
yinsh-track list --sort created --reverse

# Different output formats
yinsh-track list --format json --limit 10
yinsh-track list --format csv > experiments.csv

# Combined filters
yinsh-track list --status running --tags ml --sort updated --reverse
```

### `compare` - Compare Experiments

Compare multiple experiments side-by-side.

**Syntax:**
```bash
yinsh-track compare EXPERIMENT_ID... [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--metrics`, `-m` | Specific metrics to compare (multiple allowed) |
| `--format`, `-f` | Output format: table, json, csv |
| `--include-config` | Include configuration differences |
| `--statistical` | Include statistical significance tests |

**Examples:**
```bash
# Basic comparison
yinsh-track compare 1 2 3

# Compare specific metrics
yinsh-track compare 1 2 --metrics accuracy loss

# Include configuration differences
yinsh-track compare 1 2 --include-config

# Full comparison with statistics
yinsh-track compare 1 2 3 --include-config --statistical

# JSON output for processing
yinsh-track compare 1 2 --format json > comparison.json
```

### `reproduce` - Reproduce Experiments

Generate reproduction scripts and instructions for experiments.

**Syntax:**
```bash
yinsh-track reproduce EXPERIMENT_ID [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--output-dir` | Output directory for reproduction files |
| `--dry-run` | Show reproduction plan without creating files |
| `--script-only` | Generate only the reproduction script |
| `--force` | Override environment warnings |
| `--skip-env` | Skip environment setup |
| `--skip-data` | Skip data preparation |

**Examples:**
```bash
# Basic reproduction
yinsh-track reproduce 123

# Custom output directory
yinsh-track reproduce 123 --output-dir ./reproductions/exp123

# Dry run to see what would be generated
yinsh-track reproduce 123 --dry-run

# Generate only the script
yinsh-track reproduce 123 --script-only

# Skip environment checks
yinsh-track reproduce 123 --force --skip-env
```

**Generated Files:**
- `reproduce_experiment_123.py` - Executable reproduction script
- `requirements.txt` - Package dependencies
- `experiment_metadata.json` - Full experiment data
- `REPRODUCTION_INSTRUCTIONS.md` - Step-by-step guide

### `search` - Search Experiments

Search experiments with advanced filtering capabilities.

**Syntax:**
```bash
yinsh-track search [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--query` | Text search in names and descriptions |
| `--status` | Filter by status |
| `--tags` | Filter by tags (comma-separated) |
| `--metric` | Filter by metric name |
| `--metric-min` | Minimum metric value |
| `--metric-max` | Maximum metric value |
| `--date-from` | Filter from date (YYYY-MM-DD) |
| `--date-to` | Filter to date (YYYY-MM-DD) |
| `--limit` | Maximum number of results |
| `--format` | Output format: table, json, csv |

**Examples:**
```bash
# Text search
yinsh-track search --query "neural network"

# Search by status and tags
yinsh-track search --status running --tags ml

# Metric-based search
yinsh-track search --metric accuracy --metric-min 0.9

# Complex search
yinsh-track search --query "production" \
    --status done \
    --metric accuracy --metric-min 0.85 \
    --date-from 2024-01-01

# JSON output
yinsh-track search --query "model" --format json
```

### `config` - View Configuration

Display current configuration settings.

**Syntax:**
```bash
yinsh-track config
```

**Example:**
```bash
yinsh-track config
```

## Configuration

YinshML CLI can be configured through configuration files and environment variables.

### Configuration File Locations

The CLI looks for configuration files in this order:
1. `--config` option path
2. `./.yinsh-track.json` (current directory)
3. `./yinsh-track.json` (current directory)
4. `~/.yinsh-track.json` (home directory)
5. `~/.config/yinsh-track.json` (user config directory)
6. `/etc/yinsh-track.json` (system-wide)

### Configuration Options

```json
{
  "database_path": "/path/to/experiments.db",
  "output_format": "table",
  "color_output": true,
  "show_timestamps": true,
  "max_rows": 50,
  "default_status": "running",
  "auto_capture_git": true,
  "auto_capture_system": true,
  "auto_capture_environment": true,
  "confirm_destructive": true,
  "verbose": false,
  "quiet": false
}
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `YINSH_TRACK_DB_PATH` | Database path |
| `YINSH_TRACK_OUTPUT_FORMAT` | Default output format |
| `YINSH_TRACK_COLOR` | Enable/disable colors (true/false) |
| `YINSH_TRACK_VERBOSE` | Enable verbose mode (true/false) |
| `YINSH_TRACK_QUIET` | Enable quiet mode (true/false) |

## Shell Completion

Enable tab completion for commands and options.

### Bash Completion

```bash
# Generate and install completion
yinsh-track --install-completion bash > ~/.yinsh-track-completion.bash
echo "source ~/.yinsh-track-completion.bash" >> ~/.bashrc
source ~/.bashrc
```

### Zsh Completion

```bash
# Create completion directory
mkdir -p ~/.zsh/completions

# Generate and install completion
yinsh-track --install-completion zsh > ~/.zsh/completions/_yinsh-track

# Add to ~/.zshrc if not present
echo "fpath=(~/.zsh/completions \$fpath)" >> ~/.zshrc
echo "autoload -U compinit && compinit" >> ~/.zshrc
source ~/.zshrc
```

## Examples

### Typical Workflow

```bash
# 1. Start a new experiment
yinsh-track start "CNN Classifier" \
    --description "Image classification with CNN" \
    --tags vision --tags classification \
    --parameter lr=0.001 --parameter epochs=50

# 2. List running experiments
yinsh-track list --status running

# 3. Search for similar experiments
yinsh-track search --query "CNN" --tags vision

# 4. Compare with previous experiments
yinsh-track compare 1 2 3 --metrics accuracy loss --include-config

# 5. Reproduce a successful experiment
yinsh-track reproduce 2 --output-dir ./reproduce_exp2
```

### Batch Operations

```bash
# Export all experiments to CSV
yinsh-track list --format csv --limit 1000 > all_experiments.csv

# Find high-performing models
yinsh-track search --metric accuracy --metric-min 0.95 --format json

# Compare all experiments from last month
yinsh-track list --date-from 2024-05-01 --date-to 2024-05-31 --format json | \
    jq -r '.[].id' | xargs yinsh-track compare
```

## Troubleshooting

### Common Issues

**Database not found:**
```bash
# Check current configuration
yinsh-track config

# Specify database path
yinsh-track --database /path/to/experiments.db list
```

**Permission errors:**
```bash
# Check file permissions
ls -la ~/.yinsh-track.json

# Use verbose mode for debugging
yinsh-track --verbose start "Debug Test"
```

**Import errors:**
```bash
# Verify installation
pip show yinsh-ml

# Reinstall if necessary
pip install --upgrade yinsh-ml
```

### Getting Help

- Use `--help` with any command for detailed information
- Use `--verbose` for debugging information
- Check the configuration with `yinsh-track config`
- Review error messages for specific suggestions

### Performance Tips

- Use `--limit` to control result sizes
- Use `--quiet` mode in scripts
- Consider JSON format for programmatic processing
- Use specific filters to reduce query time

## Advanced Usage

### Custom Scripts

```bash
#!/bin/bash
# Automated experiment comparison script

# Get all experiments from last week
EXPERIMENTS=$(yinsh-track list --date-from $(date -d '7 days ago' +%Y-%m-%d) --format json | jq -r '.[].id')

# Compare them if we have multiple
if [ $(echo $EXPERIMENTS | wc -w) -gt 1 ]; then
    yinsh-track compare $EXPERIMENTS --include-config --format json > weekly_comparison.json
    echo "Weekly comparison saved to weekly_comparison.json"
fi
```

### Integration with Other Tools

```bash
# Use with jq for JSON processing
yinsh-track list --format json | jq '.[] | select(.status == "running")'

# Use with awk for CSV processing
yinsh-track list --format csv | awk -F',' '$3 == "done" {print $1, $2}'

# Pipe to other analysis tools
yinsh-track search --metric accuracy --format json | python analyze_results.py
``` 