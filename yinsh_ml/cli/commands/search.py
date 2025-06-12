"""
Search command for finding experiments.
"""

import click
from datetime import datetime
from typing import Optional

from ..utils import (
    get_experiment_tracker, 
    output_experiments, 
    handle_error,
    search_experiments_by_text,
    filter_experiments_by_metrics,
    parse_search_tags
)


@click.command()
@click.option('--query', '-q', 
              help='Search query (searches name and description)')
@click.option('--status', '-s',
              help='Filter by experiment status')
@click.option('--tags', '-t',
              help='Filter by tags (comma-separated)')
@click.option('--metric', '-m',
              help='Filter by metric name')
@click.option('--metric-min', type=float,
              help='Minimum value for specified metric')
@click.option('--metric-max', type=float,
              help='Maximum value for specified metric')
@click.option('--date-from',
              help='Filter experiments from date (YYYY-MM-DD)')
@click.option('--date-to',
              help='Filter experiments to date (YYYY-MM-DD)')
@click.option('--limit', '-l', type=int, default=20,
              help='Maximum number of results (default: 20)')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['table', 'json', 'csv']),
              help='Output format')
@click.pass_context
def search(ctx, query: Optional[str], status: Optional[str], tags: Optional[str],
          metric: Optional[str], metric_min: Optional[float], metric_max: Optional[float],
          date_from: Optional[str], date_to: Optional[str], 
          limit: int, output_format: Optional[str]):
    """
    Search experiments with advanced filtering and text queries.
    
    Powerful search tool to find experiments based on multiple criteria including
    text search across names and descriptions, status filtering, tag matching,
    metric value ranges, and date ranges. Combine multiple filters for precise
    results and export in various formats for further analysis.
    
    \b
    Search capabilities:
    • Text search in experiment names and descriptions (case-insensitive)
    • Status filtering (running, done, failed, paused, cancelled, pending)
    • Tag-based filtering with comma-separated values (AND logic)
    • Metric value filtering with min/max constraints
    • Date range filtering with YYYY-MM-DD format
    • Result limiting and multiple output formats
    
    \b
    Examples:
        # Text search across experiment names and descriptions
        yinsh-track search --query "neural network"
        yinsh-track search --query "CNN classification"
        
        # Filter by experiment status
        yinsh-track search --status running
        yinsh-track search --status done --limit 10
        
        # Tag-based filtering
        yinsh-track search --tags "ml,production"
        yinsh-track search --tags "vision" --status done
        
        # Metric-based filtering (find high-performing models)
        yinsh-track search --metric accuracy --metric-min 0.9
        yinsh-track search --metric loss --metric-max 0.1
        yinsh-track search --metric f1_score --metric-min 0.85 --metric-max 0.95
        
        # Date range filtering
        yinsh-track search --date-from 2024-01-01 --date-to 2024-01-31
        yinsh-track search --date-from 2024-06-01  # From date to now
        
        # Combined advanced searches
        yinsh-track search --query "transformer" \\
            --status done \\
            --metric accuracy --metric-min 0.9 \\
            --tags "nlp,production"
            
        # Export results for analysis
        yinsh-track search --metric accuracy --metric-min 0.85 --format json
        yinsh-track search --tags "production" --format csv > production_models.csv
        
        # Find recent high-performing experiments
        yinsh-track search --date-from 2024-06-01 \\
            --metric accuracy --metric-min 0.9 \\
            --status done --format json
    
    \b
    Search Logic:
        1. Apply basic filters (status, tags, dates) to database query
        2. Filter results by text search in names/descriptions
        3. Apply metric filtering with value constraints
        4. Limit results and format output
        5. Display search summary with applied criteria
    
    \b
    Tips:
        • Combine multiple filters for precise targeting
        • Use metric filtering to find high-performing models
        • Text search is case-insensitive and searches both names and descriptions
        • JSON/CSV output is ideal for programmatic analysis
        • Use --limit to control large result sets
        • Date filtering helps focus on recent experiments
    """
    # Get global context
    verbose = ctx.obj.get('verbose', False) if ctx.obj else False
    use_color = not (ctx.obj.get('no_color', False) if ctx.obj else False)
    
    # Validate metric filtering inputs
    if (metric_min is not None or metric_max is not None) and not metric:
        raise click.ClickException("--metric must be specified when using --metric-min or --metric-max")
    
    try:
        # Get experiment tracker
        tracker = get_experiment_tracker()
        
        # Parse tags if provided
        tag_list = parse_search_tags(tags)
        
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
        
        if verbose:
            click.echo("Searching experiments...")
            if query:
                click.echo(f"  Text query: '{query}'")
            if status:
                click.echo(f"  Status filter: {status}")
            if tag_list:
                click.echo(f"  Tags filter: {', '.join(tag_list)}")
            if metric:
                click.echo(f"  Metric filter: {metric}")
                if metric_min is not None:
                    click.echo(f"    Min value: {metric_min}")
                if metric_max is not None:
                    click.echo(f"    Max value: {metric_max}")
            if start_date or end_date:
                click.echo(f"  Date range: {start_date or 'beginning'} to {end_date or 'now'}")
        
        # Step 1: Apply basic filters using ExperimentTracker
        experiments = tracker.query_experiments(
            status=status,
            tags=tag_list if tag_list else None,
            start_date=start_date,
            end_date=end_date,
            limit=None,  # We'll apply limit after other filters
            offset=0,
            include_metrics=False,
            include_tags=True
        )
        
        if verbose:
            click.echo(f"  Found {len(experiments)} experiments after basic filtering")
        
        # Step 2: Apply text search filter
        if query:
            experiments = search_experiments_by_text(tracker, query, experiments)
            if verbose:
                click.echo(f"  Found {len(experiments)} experiments after text search")
        
        # Step 3: Apply metric filtering
        if metric:
            experiments = filter_experiments_by_metrics(
                tracker, experiments, metric, metric_min, metric_max
            )
            if verbose:
                click.echo(f"  Found {len(experiments)} experiments after metric filtering")
        
        # Step 4: Apply limit
        if limit and len(experiments) > limit:
            experiments = experiments[:limit]
            if verbose:
                click.echo(f"  Limited results to {limit} experiments")
        
        # Display results
        if not experiments:
            click.echo("No experiments found matching your criteria.")
            return
        
        # Add search result info
        search_info = []
        if query:
            search_info.append(f"text='{query}'")
        if status:
            search_info.append(f"status={status}")
        if tag_list:
            search_info.append(f"tags={','.join(tag_list)}")
        if metric:
            metric_filter = f"metric={metric}"
            if metric_min is not None:
                metric_filter += f" (≥{metric_min})"
            if metric_max is not None:
                metric_filter += f" (≤{metric_max})"
            search_info.append(metric_filter)
        if start_date or end_date:
            date_filter = f"dates={start_date or 'beginning'} to {end_date or 'now'}"
            search_info.append(date_filter)
        
        search_summary = f"Found {len(experiments)} experiments"
        if search_info:
            search_summary += f" matching: {', '.join(search_info)}"
        
        if output_format != 'json' and output_format != 'csv':
            click.echo(search_summary)
            click.echo()
        
        # Prepare headers for output
        headers = ['ID', 'Name', 'Status', 'Tags', 'Created']
        
        # Add metric column if we're filtering by metrics
        if metric:
            headers.insert(-1, f'{metric} (latest)')  # Insert before 'Created'
            
            # Add metric values to experiment data for display
            for exp in experiments:
                if 'latest_metric_value' in exp:
                    exp[f'{metric}_latest'] = f"{exp['latest_metric_value']:.4f}"
                else:
                    exp[f'{metric}_latest'] = 'N/A'
        
        # Format and output results
        result = output_experiments(
            experiments, 
            output_format or 'table',
            use_color=use_color,
            headers=headers
        )
        
        click.echo(result)
        
    except click.ClickException:
        raise
    except Exception as e:
        handle_error(e, verbose)
        raise click.ClickException(f"Search failed: {e}") 