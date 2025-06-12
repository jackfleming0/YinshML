"""
Compare command for comparing experiments.
"""

import click
import json
from typing import List

from ..utils import (
    get_experiment_tracker, format_comparison_table, 
    validate_experiments_exist, get_config
)


@click.command()
@click.argument('experiment_ids', nargs=-1, required=True, type=int)
@click.option('--metrics', '-m', multiple=True,
              help='Specific metrics to compare (can be used multiple times)')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['table', 'json', 'csv']),
              help='Output format')
@click.option('--include-config', is_flag=True,
              help='Include configuration differences')
@click.option('--statistical', is_flag=True,
              help='Include statistical significance tests')
@click.pass_context
def compare(ctx, experiment_ids: List[int], metrics: List[str], 
           output_format: str, include_config: bool, statistical: bool):
    """
    Compare multiple experiments side-by-side with detailed analysis.
    
    Analyzes differences between 2-10 experiments, showing configuration
    parameters, metrics, and statistical comparisons. Helps identify what
    changes led to performance improvements or regressions.
    
    \b
    The comparison includes:
    • Basic experiment information (status, creation dates, tags)
    • Configuration parameter differences (highlighted)
    • Metric comparisons with statistical analysis
    • Visual indicators for significant differences
    
    \b
    Examples:
        # Basic comparison of three experiments
        yinsh-track compare 1 2 3
        
        # Compare specific metrics only
        yinsh-track compare 1 2 3 --metrics accuracy loss f1_score
        
        # Include configuration differences
        yinsh-track compare 1 2 --include-config
        
        # Full analysis with statistical tests
        yinsh-track compare 1 2 3 --include-config --statistical
        
        # Export comparison for further analysis
        yinsh-track compare 1 2 --format json > comparison.json
        yinsh-track compare 1 2 --format csv > comparison.csv
        
        # Quick comparison of recent experiments
        yinsh-track list --limit 3 --format json | jq -r '.[].id' | xargs yinsh-track compare
    
    \b
    Tips:
        • Use --include-config to spot configuration differences
        • Statistical analysis helps determine significant improvements
        • JSON/CSV output is ideal for automated analysis
        • Compare experiments with similar setups for meaningful insights
        • Look for patterns in successful experiment configurations
    """
    config = get_config()
    use_color = config.get('color_output', True) and not output_format
    
    try:
        # Validate input
        if len(experiment_ids) < 2:
            raise click.ClickException("Need at least 2 experiments to compare")
        
        if len(experiment_ids) > 10:
            raise click.ClickException("Cannot compare more than 10 experiments at once")
        
        # Get tracker and validate experiments exist
        tracker = get_experiment_tracker()
        valid_ids = validate_experiments_exist(tracker, list(experiment_ids))
        
        if len(valid_ids) < 2:
            raise click.ClickException("Need at least 2 valid experiments to compare")
        
        with click.progressbar(length=3, label='Comparing experiments') as bar:
            # Get experiment data
            experiments = []
            for exp_id in valid_ids:
                exp_data = tracker.get_experiment(exp_id)
                if exp_data:
                    experiments.append(exp_data)
            bar.update(1)
            
            # Get comparison data using tracker's compare method
            try:
                comparison_data = tracker.compare_experiments(
                    valid_ids, 
                    metric_names=list(metrics) if metrics else None
                )
                comparison_data['experiments'] = experiments  # Add full experiment data
            except Exception as e:
                # Fallback to manual comparison if tracker method fails
                click.echo(f"Warning: Using fallback comparison method: {e}", err=True)
                comparison_data = _manual_compare_experiments(experiments, metrics)
            bar.update(1)
            
            # Add configuration comparison if requested
            if include_config:
                config_diff = _compare_configurations(experiments)
                comparison_data['config_diff'] = config_diff
            bar.update(1)
        
        # Output results
        if output_format == 'json':
            # Convert datetime objects to strings for JSON serialization
            json_data = _prepare_json_output(comparison_data)
            click.echo(json.dumps(json_data, indent=2))
            
        elif output_format == 'csv':
            csv_output = _format_comparison_csv(comparison_data)
            click.echo(csv_output)
            
        else:  # table format (default)
            table_output = format_comparison_table(comparison_data, use_color=use_color)
            click.echo(table_output)
            
            # Additional statistical information if requested
            if statistical and comparison_data.get('metrics'):
                click.echo("\n=== Statistical Analysis ===")
                _display_statistical_analysis(comparison_data['metrics'], use_color)
    
    except Exception as e:
        raise click.ClickException(f"Comparison failed: {e}")


def _manual_compare_experiments(experiments: List[dict], metrics: List[str] = None) -> dict:
    """Manual comparison fallback when tracker method fails."""
    comparison_data = {
        'experiments': experiments,
        'metrics': {}
    }
    
    # Collect all metrics from all experiments
    all_metrics = set()
    for exp in experiments:
        # Note: This is a simplified version - in reality we'd need to query metrics separately
        # For now, we'll just show basic experiment info
        pass
    
    return comparison_data


def _compare_configurations(experiments: List[dict]) -> dict:
    """Compare experiment configurations."""
    if not experiments:
        return {}
    
    config_diff = {
        'common': {},
        'differences': {}
    }
    
    # Extract configurations
    configs = []
    for exp in experiments:
        config = exp.get('config', {})
        user_config = config.get('user_config', {}) if isinstance(config, dict) else {}
        configs.append(user_config)
    
    if not any(configs):
        return config_diff
    
    # Find all configuration keys
    all_keys = set()
    for config in configs:
        all_keys.update(config.keys())
    
    # Compare each key across experiments
    for key in all_keys:
        values = {}
        for i, exp in enumerate(experiments):
            config = configs[i]
            values[str(exp['id'])] = config.get(key, 'N/A')
        
        # Check if all values are the same
        unique_values = set(v for v in values.values() if v != 'N/A')
        if len(unique_values) <= 1 and 'N/A' not in values.values():
            config_diff['common'][key] = list(unique_values)[0] if unique_values else 'N/A'
        else:
            config_diff['differences'][key] = values
    
    return config_diff


def _prepare_json_output(data: dict) -> dict:
    """Prepare data for JSON output by converting non-serializable objects."""
    import copy
    from datetime import datetime
    
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetime(item) for item in obj]
        else:
            return obj
    
    return convert_datetime(copy.deepcopy(data))


def _format_comparison_csv(data: dict) -> str:
    """Format comparison data as CSV."""
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    experiments = data.get('experiments', [])
    if not experiments:
        return "No experiments to compare"
    
    # Write basic info
    writer.writerow(['Attribute'] + [f"Experiment_{exp['id']}" for exp in experiments])
    
    # Status row
    writer.writerow(['Status'] + [exp.get('status', 'unknown') for exp in experiments])
    
    # Created row
    writer.writerow(['Created'] + [exp.get('timestamp', 'unknown') for exp in experiments])
    
    # Tags row
    writer.writerow(['Tags'] + [','.join(exp.get('tags', [])) for exp in experiments])
    
    # Configuration differences
    if data.get('config_diff', {}).get('differences'):
        writer.writerow([])  # Empty row
        writer.writerow(['Parameter'] + [f"Experiment_{exp['id']}" for exp in experiments])
        
        for param, values in data['config_diff']['differences'].items():
            row = [param]
            for exp in experiments:
                row.append(values.get(str(exp['id']), 'N/A'))
            writer.writerow(row)
    
    return output.getvalue()


def _display_statistical_analysis(metrics_data: dict, use_color: bool = True):
    """Display statistical analysis of metrics."""
    # This would implement statistical significance tests
    # For now, just show a placeholder
    click.echo("Statistical significance testing not yet implemented.")
    click.echo("Future version will include:")
    click.echo("  - T-test for continuous metrics")
    click.echo("  - Effect size calculations") 
    click.echo("  - Confidence intervals")
    click.echo("  - Distribution comparisons") 