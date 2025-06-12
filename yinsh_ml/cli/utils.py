"""
Utility functions for YinshML CLI.

Provides formatting, table display, color coding, progress bars, and other helper functions.
"""

import sys
import json
import csv
import io
import re
import difflib
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Callable
import click

from .config import get_config

# Progress bar utilities
def show_progress(iterable, length=None, label="Processing", show_eta=True):
    """Show a progress bar for long-running operations."""
    config = get_config()
    if config.get('quiet', False):
        # In quiet mode, don't show progress bars
        return iterable
    
    return click.progressbar(
        iterable, 
        length=length,
        label=label,
        show_eta=show_eta,
        show_percent=True,
        show_pos=True
    )

def progress_callback(label: str = "Processing"):
    """Create a progress callback for operations without iterables."""
    config = get_config()
    if config.get('quiet', False):
        return lambda: None
    
    def callback():
        click.echo(f"{label}...", nl=False)
        return lambda: click.echo(" done")
    
    return callback

def format_datetime(dt: Union[str, datetime, None]) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    
    if isinstance(dt, str):
        try:
            # Try parsing ISO format
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            return dt  # Return as-is if can't parse
    
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return str(dt)


def format_duration(start_time: Union[str, datetime, None], 
                   end_time: Union[str, datetime, None] = None) -> str:
    """Format duration between two times."""
    if start_time is None:
        return "N/A"
    
    if isinstance(start_time, str):
        try:
            # Handle different datetime formats from database
            if 'T' in start_time:
                # ISO format with T separator
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            else:
                # Format like '2025-06-06 19:00:41' - parse as local time
                start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                # Try alternative parsing
                start_time = datetime.fromisoformat(start_time)
            except ValueError:
                return "N/A"
    
    if end_time is None:
        end_time = datetime.now()
    elif isinstance(end_time, str):
        try:
            if 'T' in end_time:
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            else:
                end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                end_time = datetime.fromisoformat(end_time)
            except ValueError:
                end_time = datetime.now()
    
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        # Make sure both times are timezone-aware or naive
        if start_time.tzinfo is None and end_time.tzinfo is not None:
            # If start_time is naive and end_time has timezone, make end_time naive
            end_time = end_time.replace(tzinfo=None)
        elif start_time.tzinfo is not None and end_time.tzinfo is None:
            # If start_time has timezone and end_time is naive, make start_time naive
            start_time = start_time.replace(tzinfo=None)
        
        delta = end_time - start_time
        total_seconds = int(delta.total_seconds())
        
        # Handle negative durations (shouldn't happen but just in case)
        if total_seconds < 0:
            return "N/A"
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    return "N/A"


def colorize_status(status: str, use_color: bool = True) -> str:
    """Apply color coding to experiment status."""
    if not use_color:
        return status
    
    status_colors = {
        'running': 'yellow',
        'done': 'green', 
        'completed': 'green',
        'failed': 'red',
        'error': 'red',
        'cancelled': 'red',
        'paused': 'cyan',
        'pending': 'blue',
        'queued': 'blue',
    }
    
    color = status_colors.get(status.lower(), 'white')
    return click.style(status, fg=color)


def truncate_text(text: str, max_length: int = 30) -> str:
    """Truncate text with ellipsis if too long."""
    if text is None:
        return "N/A"
    
    text = str(text)
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."


def parse_parameters(param_list: List[str]) -> Dict[str, Any]:
    """Parse parameter list in key=value format."""
    params = {}
    
    for param_str in param_list:
        if '=' not in param_str:
            raise click.BadParameter(f"Parameter '{param_str}' must be in key=value format")
        
        key, value = param_str.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        if not key:
            raise click.BadParameter(f"Empty parameter key in '{param_str}'")
        
        # Try to parse as number or boolean
        if value.lower() in ('true', 'false'):
            params[key] = value.lower() == 'true'
        else:
            try:
                # Try integer first
                params[key] = int(value)
            except ValueError:
                try:
                    # Try float
                    params[key] = float(value)
                except ValueError:
                    # Keep as string
                    params[key] = value
    
    return params


def format_table(data: List[Dict[str, Any]], 
                headers: List[str], 
                max_width: Optional[Dict[str, int]] = None,
                use_color: bool = True) -> str:
    """Format data as a table."""
    if not data:
        return "No data to display."
    
    # Apply max width constraints
    if max_width:
        for row in data:
            for col, max_w in max_width.items():
                if col in row:
                    row[col] = truncate_text(str(row[col]), max_w)
    
    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header)
        for row in data:
            if header in row:
                # Handle colored text by measuring actual content length
                content = str(row[header])
                if use_color and 'status' in header.lower():
                    # For status columns, measure the original text length
                    content = content
                display_length = len(content)
                col_widths[header] = max(col_widths[header], display_length)
    
    # Build table
    lines = []
    
    # Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    
    # Separator
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    lines.append(separator)
    
    # Data rows
    for row in data:
        row_values = []
        for header in headers:
            value = str(row.get(header, "N/A"))
            
            # Apply color coding for status columns
            if use_color and 'status' in header.lower():
                value = colorize_status(value, use_color)
            
            # Pad to column width (accounting for ANSI color codes)
            if use_color and '\x1b[' in value:
                # For colored text, pad based on visible length
                visible_length = len(click.unstyle(value))
                padding = col_widths[header] - visible_length
                padded_value = value + " " * padding
            else:
                padded_value = value.ljust(col_widths[header])
            
            row_values.append(padded_value)
        
        lines.append(" | ".join(row_values))
    
    return "\n".join(lines)


def output_experiments(experiments: List[Dict[str, Any]], 
                      output_format: str,
                      use_color: bool = True,
                      headers: Optional[List[str]] = None) -> str:
    """Output experiments in the specified format."""
    if not experiments:
        return "No experiments found."
    
    if output_format == 'json':
        return json.dumps(experiments, indent=2, default=str)
    
    elif output_format == 'csv':
        if not headers:
            # Use all keys from first experiment
            headers = list(experiments[0].keys())
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        
        # Prepare data for table display first to get the right format
        table_data = []
        for exp in experiments:
            row = {}
            for header in headers:
                # Handle case variations and custom headers
                header_lower = header.lower()
                
                if header_lower == 'duration':
                    row[header] = format_duration(exp.get('timestamp'), None)
                elif header_lower in ['created', 'created_at']:
                    row[header] = format_datetime(exp.get('timestamp'))
                elif header_lower == 'updated_at':
                    row[header] = format_datetime(exp.get('updated_at') or exp.get('timestamp'))
                elif header_lower in ['name']:
                    row[header] = exp.get('name', '')
                elif header_lower in ['description']:
                    row[header] = exp.get('notes', '')
                elif header_lower in ['id']:
                    row[header] = exp.get('id', '')
                elif header_lower in ['status']:
                    row[header] = exp.get('status', 'unknown')
                elif header_lower in ['tags']:
                    tags = exp.get('tags', [])
                    if isinstance(tags, list):
                        row[header] = ', '.join(tags) if tags else 'None'
                    else:
                        row[header] = str(tags) if tags else 'None'
                elif '(latest)' in header:
                    # Handle metric columns like 'accuracy (latest)'
                    metric_name = header.split(' ')[0].lower()
                    metric_key = f'{metric_name}_latest'
                    row[header] = exp.get(metric_key, 'N/A')
                else:
                    # Direct mapping for other headers
                    row[header] = exp.get(header, exp.get(header_lower, 'N/A'))
            table_data.append(row)
        
        # Write the processed data
        for row in table_data:
            # Remove color codes for CSV
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, str) and '\x1b[' in v:
                    clean_row[k] = click.unstyle(v)
                else:
                    clean_row[k] = v
            writer.writerow(clean_row)
        return output.getvalue()
    
    else:  # table format
        if not headers:
            headers = ['ID', 'Name', 'Status', 'Tags', 'Created']
        
        # Prepare data for table display
        table_data = []
        for exp in experiments:
            row = {}
            for header in headers:
                # Handle case variations and custom headers
                header_lower = header.lower()
                
                if header_lower == 'duration':
                    # Use timestamp field for duration calculation
                    row[header] = format_duration(exp.get('timestamp'), None)
                elif header_lower in ['created', 'created_at']:
                    # Map created_at to timestamp field from database
                    row[header] = format_datetime(exp.get('timestamp'))
                elif header_lower == 'updated_at':
                    # Check if there's an updated_at field, otherwise use timestamp
                    row[header] = format_datetime(exp.get('updated_at') or exp.get('timestamp'))
                elif header_lower in ['name']:
                    row[header] = truncate_text(exp.get('name', ''), 25)
                elif header_lower in ['description']:
                    # Map description to notes field from database
                    row[header] = truncate_text(exp.get('notes', ''), 40)
                elif header_lower in ['id']:
                    row[header] = exp.get('id', 'N/A')
                elif header_lower in ['status']:
                    status = exp.get('status', 'unknown')
                    row[header] = colorize_status(status, use_color)
                elif header_lower in ['tags']:
                    tags = exp.get('tags', [])
                    if isinstance(tags, list):
                        row[header] = ', '.join(tags) if tags else 'None'
                    else:
                        row[header] = str(tags) if tags else 'None'
                elif '(latest)' in header:
                    # Handle metric columns like 'accuracy (latest)'
                    metric_name = header.split(' ')[0].lower()
                    metric_key = f'{metric_name}_latest'
                    row[header] = exp.get(metric_key, 'N/A')
                else:
                    # Direct mapping for other headers
                    row[header] = exp.get(header, exp.get(header_lower, 'N/A'))
            table_data.append(row)
        
        max_widths = {
            'name': 25,
            'description': 40,
            'status': 12
        }
        
        return format_table(table_data, headers, max_widths, use_color)


def handle_error(error: Exception, verbose: bool = False, context: str = None) -> None:
    """Handle and display errors appropriately with helpful suggestions."""
    error_msg = str(error).lower()
    suggestions = []
    
    # Common error patterns and suggestions
    if "no such file or directory" in error_msg or "file not found" in error_msg:
        suggestions.append("• Check if the file path is correct")
        suggestions.append("• Ensure the file exists and you have read permissions")
    elif "permission denied" in error_msg:
        suggestions.append("• Check file/directory permissions")
        suggestions.append("• You may need to run with appropriate privileges")
    elif "database" in error_msg:
        suggestions.append("• Verify the database path is correct")
        suggestions.append("• Check if the database file is accessible")
        suggestions.append("• Try initializing a new experiment database")
    elif "experiment" in error_msg and "not found" in error_msg:
        suggestions.append("• Use 'yinsh-track list' to see available experiments")
        suggestions.append("• Check if the experiment ID is correct")
    elif "connection" in error_msg or "network" in error_msg:
        suggestions.append("• Check your network connection")
        suggestions.append("• Verify any API endpoints are accessible")
    elif "invalid" in error_msg or "format" in error_msg:
        suggestions.append("• Check the input format and syntax")
        suggestions.append("• Refer to help text with --help for examples")
    elif "dependency" in error_msg or "import" in error_msg:
        suggestions.append("• Ensure all required packages are installed")
        suggestions.append("• Try: pip install -r requirements.txt")
    
    # Display error
    click.echo(click.style(f"✗ Error: {error}", fg='red'), err=True)
    
    # Show context if provided
    if context:
        click.echo(f"  Context: {context}", err=True)
    
    # Show suggestions
    if suggestions:
        click.echo(click.style("\nSuggestions:", fg='yellow'), err=True)
        for suggestion in suggestions:
            click.echo(f"  {suggestion}", err=True)
    
    # Show verbose traceback
    if verbose:
        click.echo(click.style("\nDetailed traceback:", fg='cyan'), err=True)
        import traceback
        click.echo(traceback.format_exc(), err=True)
    else:
        click.echo("\nUse --verbose for detailed error information", err=True)

def suggest_command(invalid_cmd: str, available_commands: List[str]) -> Optional[str]:
    """Suggest the closest matching command for typos."""
    matches = difflib.get_close_matches(invalid_cmd, available_commands, n=1, cutoff=0.6)
    return matches[0] if matches else None

def format_success_message(message: str, details: Dict[str, Any] = None) -> str:
    """Format a consistent success message with optional details."""
    lines = [click.style(f"✓ {message}", fg='green')]
    
    if details:
        for key, value in details.items():
            if value is not None:
                lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    config = get_config()
    if not config.get('confirm_destructive', True):
        return True
    
    return click.confirm(message, default=default)

def verbose_echo(message: str, **kwargs):
    """Echo message only in verbose mode."""
    config = get_config()
    if config.get('verbose', False):
        click.echo(message, **kwargs)

def quiet_echo(message: str, **kwargs):
    """Echo message unless in quiet mode."""
    config = get_config()
    if not config.get('quiet', False):
        click.echo(message, **kwargs)

def debug_echo(message: str, **kwargs):
    """Echo debug message only in verbose mode with debug styling."""
    config = get_config()
    if config.get('verbose', False):
        click.echo(click.style(f"DEBUG: {message}", fg='cyan'), **kwargs)


def get_experiment_tracker():
    """Get ExperimentTracker instance with proper configuration."""
    try:
        from ..tracking import ExperimentTracker
        
        config = get_config()
        db_path = config.get('database_path')
        
        # Create tracker configuration
        tracker_config = {
            'auto_capture_git': config.get('auto_capture_git', True),
            'auto_capture_system': config.get('auto_capture_system', True),
            'auto_capture_environment': config.get('auto_capture_environment', True),
        }
        
        return ExperimentTracker.get_instance(db_path=db_path, config=tracker_config)
    
    except ImportError as e:
        raise click.ClickException(f"Failed to import ExperimentTracker: {e}")
    except Exception as e:
        raise click.ClickException(f"Failed to initialize ExperimentTracker: {e}")


def format_comparison_table(comparison_data: Dict[str, Any], 
                          use_color: bool = True) -> str:
    """Format experiment comparison data as a table."""
    if not comparison_data:
        return "No comparison data available."
    
    experiments = comparison_data.get('experiments', [])
    if not experiments:
        return "No experiments to compare."
    
    lines = []
    
    # Header with experiment names
    exp_names = [f"Exp {exp['id']}: {truncate_text(exp.get('name', 'Unknown'), 20)}" 
                 for exp in experiments]
    headers = ['Attribute'] + exp_names
    
    # Experiment basic info
    lines.append("=== Experiment Information ===")
    
    # Status comparison
    status_row = ['Status'] + [colorize_status(exp.get('status', 'unknown'), use_color) 
                               for exp in experiments]
    
    # Created dates
    created_row = ['Created'] + [format_datetime(exp.get('timestamp')) 
                                 for exp in experiments]
    
    # Tags
    tags_row = ['Tags'] + [', '.join(exp.get('tags', [])) if exp.get('tags') else 'None' 
                           for exp in experiments]
    
    # Create basic info table
    basic_data = [
        dict(zip(headers, status_row)),
        dict(zip(headers, created_row)),
        dict(zip(headers, tags_row))
    ]
    
    basic_table = format_table(basic_data, headers, use_color=use_color)
    lines.append(basic_table)
    lines.append("")
    
    # Configuration comparison
    if comparison_data.get('config_diff'):
        lines.append("=== Configuration Differences ===")
        config_diff = comparison_data['config_diff']
        
        if config_diff.get('common'):
            lines.append("Common Parameters:")
            for key, value in config_diff['common'].items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if config_diff.get('differences'):
            lines.append("Different Parameters:")
            diff_data = []
            for key, values in config_diff['differences'].items():
                row = {'Parameter': key}
                for i, exp in enumerate(experiments):
                    exp_key = f"Exp {exp['id']}"
                    value = values.get(str(exp['id']), 'N/A')
                    if use_color and len(set(values.values())) > 1:
                        # Highlight differences
                        row[exp_key] = click.style(str(value), fg='red')
                    else:
                        row[exp_key] = str(value)
                diff_data.append(row)
            
            diff_headers = ['Parameter'] + [f"Exp {exp['id']}" for exp in experiments]
            diff_table = format_table(diff_data, diff_headers, use_color=use_color)
            lines.append(diff_table)
            lines.append("")
    
    # Metrics comparison
    if comparison_data.get('metrics'):
        lines.append("=== Metrics Comparison ===")
        metrics = comparison_data['metrics']
        
        for metric_name, metric_data in metrics.items():
            lines.append(f"\n{metric_name}:")
            
            metric_rows = []
            for stat_name, values in metric_data.items():
                if stat_name == 'experiments':
                    continue
                
                row = {'Statistic': stat_name.replace('_', ' ').title()}
                for exp in experiments:
                    exp_id = str(exp['id'])
                    value = values.get(exp_id, 'N/A')
                    if isinstance(value, (int, float)):
                        row[f"Exp {exp['id']}"] = f"{value:.4f}"
                    else:
                        row[f"Exp {exp['id']}"] = str(value)
                metric_rows.append(row)
            
            metric_headers = ['Statistic'] + [f"Exp {exp['id']}" for exp in experiments]
            metric_table = format_table(metric_rows, metric_headers, use_color=use_color)
            lines.append(metric_table)
    
    return "\n".join(lines)


def create_reproduction_script(experiment_data: Dict[str, Any], 
                             output_dir: str = None) -> str:
    """Create a reproduction script for an experiment."""
    if not experiment_data:
        return "No experiment data available for reproduction."
    
    script_lines = [
        "#!/usr/bin/env python",
        '"""',
        f"Reproduction script for experiment {experiment_data.get('id')}: {experiment_data.get('name', 'Unknown')}",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        '"""',
        "",
        "import os",
        "import sys",
        "import json",
        "from pathlib import Path",
        "",
        "# Experiment metadata",
        f"EXPERIMENT_ID = {experiment_data.get('id')}",
        f"EXPERIMENT_NAME = {repr(experiment_data.get('name', 'Unknown'))}",
        f"EXPERIMENT_DESCRIPTION = {repr(experiment_data.get('notes', ''))}",
        "",
    ]
    
    # Add configuration
    config = experiment_data.get('config', {})
    if config:
        user_config = config.get('user_config', {})
        if user_config:
            script_lines.extend([
                "# Experiment configuration",
                "EXPERIMENT_CONFIG = {",
            ])
            for key, value in user_config.items():
                script_lines.append(f"    {repr(key)}: {repr(value)},")
            script_lines.extend([
                "}",
                "",
            ])
    
    # Add environment information
    environment = experiment_data.get('environment', {})
    if environment:
        system_info = environment.get('system', {})
        if system_info:
            script_lines.extend([
                "# Original environment information",
                "ORIGINAL_ENVIRONMENT = {",
                f"    'python_version': {repr(system_info.get('python_version', 'Unknown'))},"
                f"    'platform': {repr(system_info.get('platform', 'Unknown'))},"
                f"    'working_directory': {repr(system_info.get('working_directory', 'Unknown'))},"
                "}",
                "",
            ])
        
        # Add package requirements
        packages = environment.get('environment', {}).get('installed_packages', {})
        if packages:
            script_lines.extend([
                "# Required packages (install with: pip install <package>==<version>)",
                "REQUIRED_PACKAGES = {",
            ])
            for pkg, version in packages.items():
                script_lines.append(f"    {repr(pkg)}: {repr(version)},")
            script_lines.extend([
                "}",
                "",
            ])
    
    # Add Git information
    git_info = experiment_data.get('environment', {}).get('git', {})
    if git_info:
        script_lines.extend([
            "# Git information",
            f"GIT_COMMIT = {repr(git_info.get('commit', 'Unknown'))}",
            f"GIT_BRANCH = {repr(git_info.get('branch', 'Unknown'))}",
            f"GIT_REMOTE_URL = {repr(git_info.get('remote_url', 'Unknown'))}",
            "",
        ])
    
    # Add reproduction instructions
    script_lines.extend([
        "def check_environment():",
        '    """Check if current environment matches original experiment."""',
        "    import platform",
        "    issues = []",
        "    ",
        "    # Check Python version",
        "    current_python = sys.version",
        "    original_python = ORIGINAL_ENVIRONMENT.get('python_version', '')",
        "    if current_python != original_python:",
        '        issues.append(f"Python version mismatch: current={current_python}, original={original_python}")',
        "    ",
        "    # Check platform",
        "    current_platform = platform.platform()",
        "    original_platform = ORIGINAL_ENVIRONMENT.get('platform', '')",
        "    if current_platform != original_platform:",
        '        issues.append(f"Platform mismatch: current={current_platform}, original={original_platform}")',
        "    ",
        "    return issues",
        "",
        "def install_requirements():",
        '    """Install required packages."""',
        "    import subprocess",
        "    ",
        "    for package, version in REQUIRED_PACKAGES.items():",
        "        try:",
        '            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])',
        '            print(f"✓ Installed {package}=={version}")',
        "        except subprocess.CalledProcessError:",
        '            print(f"✗ Failed to install {package}=={version}")',
        "",
        "def reproduce_experiment():",
        '    """Main reproduction function."""',
        '    print(f"Reproducing experiment: {EXPERIMENT_NAME}")',
        '    print(f"Original ID: {EXPERIMENT_ID}")',
        "    print()",
        "    ",
        "    # Check environment compatibility",
        "    issues = check_environment()",
        "    if issues:",
        '        print("Environment compatibility issues:")',
        "        for issue in issues:",
        '            print(f"  ⚠️  {issue}")',
        "        print()",
        "    ",
        "    # Print configuration",
        '    print("Experiment configuration:")',
        "    for key, value in EXPERIMENT_CONFIG.items():",
        '        print(f"  {key}: {value}")',
        "    print()",
        "    ",
        '    print("To reproduce this experiment:")',
        '    print("1. Ensure you\'re in the correct Git commit:", GIT_COMMIT)',
        '    print("2. Install required packages using install_requirements()")',
        '    print("3. Use the configuration parameters shown above")',
        '    print("4. Run your training/experiment code with these parameters")',
        "",
        "if __name__ == '__main__':",
        "    reproduce_experiment()",
    ])
    
    return "\n".join(script_lines)


def validate_experiments_exist(tracker, experiment_ids: List[int]) -> List[int]:
    """Validate that experiments exist and return valid IDs."""
    valid_ids = []
    for exp_id in experiment_ids:
        try:
            experiment = tracker.get_experiment(exp_id)
            if experiment:
                valid_ids.append(exp_id)
            else:
                click.echo(f"Warning: Experiment {exp_id} not found", err=True)
        except Exception as e:
            click.echo(f"Warning: Could not check experiment {exp_id}: {e}", err=True)
    
    return valid_ids


def search_experiments_by_text(tracker, query: str, existing_experiments: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search experiments by text query in name and description fields."""
    if not query:
        return existing_experiments or []
    
    # If we have existing experiments, filter them
    if existing_experiments:
        filtered = []
        for exp in existing_experiments:
            name = exp.get('name', '').lower()
            description = exp.get('notes', '').lower()
            query_lower = query.lower()
            
            if query_lower in name or query_lower in description:
                filtered.append(exp)
        return filtered
    
    # Otherwise, we need to do a more complex database query
    # For now, we'll get all experiments and filter them
    # In a production system, this would be optimized with SQL LIKE queries
    try:
        all_experiments = tracker.query_experiments(include_tags=True)
        filtered = []
        
        query_lower = query.lower()
        for exp in all_experiments:
            name = exp.get('name', '').lower()
            description = exp.get('notes', '').lower()
            
            if query_lower in name or query_lower in description:
                filtered.append(exp)
        
        return filtered
    except Exception as e:
        click.echo(f"Error searching experiments: {e}", err=True)
        return []


def filter_experiments_by_metrics(tracker, experiments: List[Dict[str, Any]], 
                                metric_name: str, min_value: float = None, 
                                max_value: float = None) -> List[Dict[str, Any]]:
    """Filter experiments by metric values."""
    if not metric_name:
        return experiments
    
    filtered = []
    
    for exp in experiments:
        try:
            # Get metrics for this experiment using the correct method
            metrics = tracker.get_metric_history(exp['id'], metric_name)
            
            if not metrics:
                continue  # Skip experiments without this metric
            
            # Get the latest value for this metric
            latest_metric = metrics[-1]  # Assuming metrics are ordered by timestamp
            metric_value = latest_metric['metric_value']
            
            # Apply min/max filters
            if min_value is not None and metric_value < min_value:
                continue
            
            if max_value is not None and metric_value > max_value:
                continue
            
            # Add metric info to experiment for display
            exp['latest_metric_value'] = metric_value
            filtered.append(exp)
            
        except Exception as e:
            # Skip experiments where we can't get metrics
            continue
    
    return filtered


def parse_search_tags(tags_str: str) -> List[str]:
    """Parse comma-separated tags string into list."""
    if not tags_str:
        return []
    
    return [tag.strip() for tag in tags_str.split(',') if tag.strip()] 