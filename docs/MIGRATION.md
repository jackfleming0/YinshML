# YinshML Experiment Data Migration Guide

This guide explains how to migrate historical experiment data into the new YinshML experiment tracking system.

## Overview

The migration tool scans existing experiment result directories and imports their data into the new ExperimentTracker database, preserving:

- Experiment configurations
- Final summary metrics
- Training history (when available)
- Experiment metadata and context

## Quick Start

### Using the CLI Command (Recommended)

```bash
# Preview what would be migrated (dry run)
yinsh-track migrate results/ --dry-run

# Migrate all experiments from the results directory
yinsh-track migrate results/

# Save a detailed report
yinsh-track migrate results/ --report-file migration_report.txt

# Enable debug logging
yinsh-track migrate results/ --log-level DEBUG
```

### Using the Python Module Directly

```python
from pathlib import Path
from yinsh_ml.tracking.migration_tool import MigrationTool

# Initialize migration tool
migration_tool = MigrationTool()

# Migrate all experiments
results = migration_tool.migrate_directory(Path("results"))

# Print summary
successful = sum(1 for r in results if r.success)
print(f"Migrated {successful}/{len(results)} experiments")
```

## Supported Data Formats

The migration tool can import data from several historical formats:

### 1. Final Summary Metrics (`final_summary_metrics.json`)

Standard format containing:
- Experiment configuration (as used in `experiments/config.py`)
- Final aggregated metrics for each training iteration
- Completion timestamp
- Experiment metadata

**Example structure:**
```json
{
  "config": {
    "num_iterations": 10,
    "games_per_iteration": 100,
    "lr": 0.001,
    ...
  },
  "final_metrics": {
    "policy_loss": [0.5, 0.4, 0.3, ...],
    "value_loss": [0.8, 0.7, 0.6, ...],
    "tournament_rating": [1200, 1250, 1300, ...]
  },
  "completed_timestamp": 1672531200.0
}
```

### 2. Training Logs (`run_*.log`)

Log files containing timestamped training events. The migration tool extracts:
- Iteration-level metrics using regex patterns
- Training times and durations
- Policy/value losses and ELO ratings
- Tournament results

### 3. Tournament History (`tournament_history.json`)

Historical tournament results and ELO progressions.

### 4. Other Data Files

- `experiment_id.txt`: Links to existing tracking records (if already migrated)
- Checkpoint and model files (metadata preserved)

## Migration Process

### 1. Directory Scanning

The tool scans the specified directory for subdirectories containing experiment data. It identifies experiment directories by looking for characteristic files:

- `final_summary_metrics.json`
- `run_*.log` files
- `tournament_history.json`

### 2. Data Parsing and Validation

For each experiment directory:
1. Parse configuration and metrics from available files
2. Validate data integrity and format
3. Handle missing or corrupted data gracefully

### 3. Database Import

1. Create new experiment record in tracking database
2. Import configuration snapshot
3. Import metrics with proper timestamps and iterations
4. Link experiment to any existing checkpoints
5. Add migration metadata and tags

### 4. Error Handling

- Graceful degradation for missing or corrupted files
- Detailed error and warning reporting
- Continue processing other experiments if one fails
- Skip experiments already migrated (by checking `experiment_id.txt`)

## Command Line Options

```bash
yinsh-track migrate [OPTIONS] RESULTS_DIR
```

**Arguments:**
- `RESULTS_DIR`: Directory containing historical experiment results

**Options:**
- `--dry-run`: Preview migration without importing data
- `--log-level [DEBUG|INFO|WARNING|ERROR]`: Set logging verbosity
- `--report-file PATH`: Save detailed migration report to file

## Migration Report

The tool generates a comprehensive report including:

- **Summary Statistics**: Total experiments, success rate, metrics imported
- **Per-Experiment Details**: Status, warnings, errors, metrics count
- **Error Analysis**: Detailed failure reasons and recovery suggestions

**Example report:**
```
============================================================
YINSHML EXPERIMENT DATA MIGRATION REPORT  
============================================================
Generated: 2024-01-15T10:30:00
Total experiments: 5

SUMMARY
--------------------
Successful imports: 4
Failed imports: 1
Total metrics imported: 1,247
Success rate: 80.0%

DETAILED RESULTS
--------------------
[SUCCESS] Migrated_smoke
  Source: /path/to/results/smoke
  Experiment ID: 1001
  Metrics imported: 15

[SUCCESS] Migrated_full_training
  Source: /path/to/results/full_training  
  Experiment ID: 1002
  Metrics imported: 500
  
[FAILED] Migrated_corrupted_exp
  Source: /path/to/results/corrupted_exp
  Errors:
    - Failed to parse final_summary_metrics.json: Invalid JSON
```

## Best Practices

### Before Migration

1. **Backup Data**: Create backups of historical experiment data
2. **Review Structure**: Ensure experiment directories follow expected structure
3. **Test with Dry Run**: Use `--dry-run` to preview migration results
4. **Check Dependencies**: Verify ExperimentTracker database is accessible

### During Migration

1. **Monitor Progress**: Use appropriate log level for visibility
2. **Save Reports**: Always use `--report-file` for audit trails
3. **Handle Failures**: Review failures and re-run if needed

### After Migration

1. **Verify Results**: Check imported experiments in tracking system
2. **Update Links**: Ensure checkpoint files are properly linked
3. **Clean Up**: Archive original data as needed
4. **Document**: Update project documentation with migration details

## Troubleshooting

### Common Issues

**"No experiment directories found"**
- Check the results directory path
- Verify directory contains expected file structure
- Use `--log-level DEBUG` for detailed scanning information

**"Failed to parse configuration"**
- Review JSON format in `final_summary_metrics.json`
- Check for missing required configuration fields
- Consider manual data cleaning for corrupted files

**"Database connection failed"**
- Ensure ExperimentTracker is properly initialized
- Check database permissions and file paths
- Verify tracking system dependencies are installed

**"Partial metric import"**
- Review warnings in migration report
- Check log file formats and regex patterns
- Consider manual import for complex historical formats

### Recovery Procedures

**For Failed Experiments:**
1. Review detailed error messages in report
2. Fix underlying data issues (JSON format, file permissions, etc.)
3. Re-run migration for specific directories
4. Use `--log-level DEBUG` for detailed diagnostics

**For Partial Imports:**
1. Check which metrics were successfully imported
2. Manually import missing data if critical
3. Update experiment records with additional context
4. Document any data limitations in experiment descriptions

## Integration with Existing Workflows

### Continuous Migration

For ongoing development, consider:
- Running periodic migrations for new historical data
- Setting up automated migration pipelines
- Integrating migration checks into experiment workflows

### Data Validation

After migration:
- Compare imported metrics with original files
- Verify experiment relationships and dependencies
- Test analysis and visualization tools with migrated data

### Team Coordination

- Share migration reports with team members
- Document any data transformations or limitations
- Establish conventions for future experiment data formats

## API Reference

### MigrationTool Class

Main class for performing migrations:

```python
class MigrationTool:
    def __init__(self, tracker: Optional[ExperimentTracker] = None)
    def scan_experiment_directory(self, results_dir: Path) -> List[Path]
    def import_experiment(self, experiment_dir: Path) -> MigrationResult
    def migrate_directory(self, results_dir: Path) -> List[MigrationResult]
```

### MigrationResult Class

Container for migration results:

```python
@dataclass
class MigrationResult:
    experiment_name: str
    source_path: str
    success: bool
    experiment_id: Optional[int] = None
    metrics_imported: int = 0
    errors: List[str] = None
    warnings: List[str] = None
```

## Advanced Usage

### Custom Data Parsers

For specialized historical data formats:

```python
from yinsh_ml.tracking.migration_tool import MigrationTool

class CustomMigrationTool(MigrationTool):
    def import_experiment(self, experiment_dir: Path) -> MigrationResult:
        # Custom parsing logic for specialized formats
        result = super().import_experiment(experiment_dir)
        
        # Add custom metric extraction
        custom_metrics = self.parse_custom_format(experiment_dir)
        # ... additional processing
        
        return result
```

### Selective Migration

For migrating specific experiments:

```python
migration_tool = MigrationTool()

# Migrate only specific directories
specific_dirs = [Path("results/experiment1"), Path("results/experiment2")]
for exp_dir in specific_dirs:
    result = migration_tool.import_experiment(exp_dir)
    print(f"Migrated {exp_dir.name}: {'✓' if result.success else '✗'}")
```

### Batch Processing

For large-scale migrations:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def migrate_large_dataset(results_dirs: List[Path]):
    migration_tool = MigrationTool()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        
        futures = [
            loop.run_in_executor(executor, migration_tool.migrate_directory, results_dir)
            for results_dir in results_dirs
        ]
        
        all_results = await asyncio.gather(*futures)
    
    return all_results
```

---

For additional support or questions about the migration process, please refer to the project documentation or contact the development team. 