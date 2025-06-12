"""
List command for viewing experiments.
"""

from datetime import datetime, date
import click
from typing import Optional

from ..utils import (
    handle_error, 
    get_experiment_tracker,
    output_experiments,
    verbose_echo,
    show_progress
)
from ..config import get_config


@click.command(name='list')
@click.option('--status', '-s', 
              help='Filter by experiment status')
@click.option('--tags', '-t',
              help='Filter by tags (comma-separated)')
@click.option('--limit', '-l', type=int, default=20,
              help='Maximum number of experiments to show (default: 20)')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['table', 'json', 'csv']),
              help='Output format')
@click.option('--sort', 
              type=click.Choice(['id', 'name', 'status', 'created', 'updated']),
              default='created',
              help='Sort experiments by field (default: created)')
@click.option('--reverse', is_flag=True,
              help='Reverse sort order')
@click.option('--date-from',
              help='Filter experiments from date (YYYY-MM-DD)')
@click.option('--date-to',
              help='Filter experiments to date (YYYY-MM-DD)')
@click.pass_context
def list_experiments(ctx, status: Optional[str], tags: Optional[str], 
                    limit: int, output_format: Optional[str], 
                    sort: str, reverse: bool, date_from: Optional[str],
                    date_to: Optional[str]):
    """
    List and filter experiments with detailed information.
    
    Shows a table, JSON, or CSV view of experiments with filtering and sorting
    options. Results can be filtered by status, tags, date ranges, and more.
    
    \b
    Available status values: running, done, failed, paused, cancelled, pending
    
    \b
    Examples:
        # List all experiments (default table format)
        yinsh-track list
        
        # Filter by status
        yinsh-track list --status running
        yinsh-track list --status done --limit 5
        
        # Filter by tags (comma-separated)
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
    
    \b
    Tips:
        • Use --limit to control number of results (default: 20)
        • JSON format is ideal for programmatic processing
        • CSV format can be imported into spreadsheets
        • Combine multiple filters for precise results
    """
    config = get_config()
    verbose = config.get('verbose', False)
    use_color = config.get('color_output', True)
    
    # Use format from config if not specified
    if not output_format:
        output_format = config.get('output_format', 'table')
    
    try:
        # Get experiment tracker
        tracker = get_experiment_tracker()
        
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
        
        # Parse date filters
        start_date = None
        end_date = None
        
        if date_from:
            try:
                start_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            except ValueError:
                raise click.ClickException(f"Invalid date format for --date-from: {date_from}. Use YYYY-MM-DD")
        
        if date_to:
            try:
                end_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            except ValueError:
                raise click.ClickException(f"Invalid date format for --date-to: {date_to}. Use YYYY-MM-DD")
        
        verbose_echo("Fetching experiments from database...")
        
        # Build query description for verbose mode
        query_parts = []
        if status:
            query_parts.append(f"status={status}")
        if tag_list:
            query_parts.append(f"tags={','.join(tag_list)}")
        if start_date:
            query_parts.append(f"from={start_date}")
        if end_date:
            query_parts.append(f"to={end_date}")
        
        if query_parts and verbose:
            verbose_echo(f"Applying filters: {', '.join(query_parts)}")
        
        # Query experiments
        experiments = tracker.query_experiments(
            status=status,
            tags=tag_list,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=0,
            include_metrics=False,  # Don't include metrics for list view
            include_tags=True
        )
        
        verbose_echo(f"Retrieved {len(experiments)} experiments")
        
        if not experiments:
            click.echo("No experiments found.")
            return
        
        # Sort experiments
        sort_key_map = {
            'id': 'id',
            'name': 'name', 
            'status': 'status',
            'created': 'created_at',
            'updated': 'updated_at'
        }
        
        sort_key = sort_key_map.get(sort, 'created_at')
        
        try:
            experiments.sort(
                key=lambda x: x.get(sort_key, ''),
                reverse=reverse
            )
        except Exception as e:
            if verbose:
                click.echo(f"Warning: Could not sort by {sort}: {e}", err=True)
        
        # Display experiments
        if output_format == 'table':
            # Show count and filters
            count_msg = f"Found {len(experiments)} experiment(s)"
            
            filters = []
            if status:
                filters.append(f"status={status}")
            if tags:
                filters.append(f"tags={tags}")
            if date_from:
                filters.append(f"from={date_from}")
            if date_to:
                filters.append(f"to={date_to}")
            
            if filters:
                count_msg += f" (filtered by: {', '.join(filters)})"
            
            click.echo(count_msg)
            click.echo()
        
        # Format and display output
        result = output_experiments(
            experiments, 
            output_format, 
            use_color=use_color and output_format == 'table'
        )
        click.echo(result)
        
        if output_format == 'table' and len(experiments) == limit:
            click.echo(f"\nShowing {limit} experiments. Use --limit to see more.")
        
    except Exception as e:
        handle_error(e, verbose)
        ctx.exit(1) 