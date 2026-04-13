"""CLI interface for data inspection and validation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd
from datetime import datetime

from .data_loader import DataLoader, FilterConfig, PerformanceTracker
from .data_storage import StorageConfig, SelfPlayDataManager

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_list_games(args):
    """List games with optional filtering."""
    config = StorageConfig(
        output_dir=args.data_dir,
        parquet_dir=args.parquet_dir,
        validation_enabled=not args.skip_validation
    )
    
    loader = DataLoader(config)
    
    # Create filter config
    filter_config = None
    if any([args.min_length, args.max_length, args.winner, args.agent_version, args.training_run]):
        filter_config = FilterConfig(
            min_game_length=args.min_length,
            max_game_length=args.max_length,
            winner_filter=args.winner,
            agent_version_filter=args.agent_version,
            training_run_filter=args.training_run,
            exclude_invalid=not args.include_invalid
        )
    
    # Load games
    df, metrics = loader.load_games(filter_config, use_cache=not args.no_cache)
    
    if df.empty:
        print("No games found matching criteria.")
        return
    
    # Display results
    print(f"Found {len(df)} games")
    print(f"Loading time: {metrics.loading_time_seconds:.2f}s")
    print(f"Games per second: {metrics.games_per_second:.1f}")
    print(f"Memory usage: {metrics.memory_usage_mb:.1f}MB")
    
    if args.format == 'table':
        # Show summary table
        if 'total_turns' in df.columns:
            print(f"\nGame Length Statistics:")
            print(f"  Average: {df['total_turns'].mean():.1f}")
            print(f"  Min: {df['total_turns'].min()}")
            print(f"  Max: {df['total_turns'].max()}")
        
        # Calculate outcome statistics from metadata
        white_wins = 0
        black_wins = 0
        draws = 0
        
        for _, row in df.iterrows():
            if 'metadata' in row and isinstance(row['metadata'], dict):
                outcome = row['metadata'].get('outcome', 0)
                if outcome > 0.1:
                    white_wins += 1
                elif outcome < -0.1:
                    black_wins += 1
                else:
                    draws += 1
            else:
                draws += 1
        
        total_outcomes = white_wins + black_wins + draws
        if total_outcomes > 0:
            print(f"\nOutcome Statistics:")
            print(f"  White wins: {white_wins} ({white_wins/total_outcomes*100:.1f}%)")
            print(f"  Black wins: {black_wins} ({black_wins/total_outcomes*100:.1f}%)")
            print(f"  Draws: {draws} ({draws/total_outcomes*100:.1f}%)")
        
        # Show sample of games
        if args.limit:
            sample_df = df.head(args.limit)
            print(f"\nSample games (first {len(sample_df)}):")
            for idx, row in sample_df.iterrows():
                game_id = row.get('game_id', idx)
                turns = row.get('total_turns', 'N/A')
                
                # Extract outcome from metadata
                outcome = 'N/A'
                timestamp = 'N/A'
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    outcome = row['metadata'].get('outcome', 'N/A')
                    timestamp = row['metadata'].get('timestamp', 'N/A')
                
                print(f"  Game {game_id}: {turns} turns, outcome={outcome}, time={timestamp}")
    
    elif args.format == 'json':
        # Export to JSON
        output_file = args.output or f"games_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        df.to_json(output_file, orient='records', indent=2)
        print(f"Exported to {output_file}")
    
    elif args.format == 'csv':
        # Export to CSV
        output_file = args.output or f"games_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Exported to {output_file}")


def cmd_stats(args):
    """Show comprehensive dataset statistics."""
    config = StorageConfig(
        output_dir=args.data_dir,
        parquet_dir=args.parquet_dir,
        validation_enabled=not args.skip_validation
    )
    
    loader = DataLoader(config)
    
    # Create filter config
    filter_config = None
    if any([args.min_length, args.max_length, args.winner, args.agent_version, args.training_run]):
        filter_config = FilterConfig(
            min_game_length=args.min_length,
            max_game_length=args.max_length,
            winner_filter=args.winner,
            agent_version_filter=args.agent_version,
            training_run_filter=args.training_run,
            exclude_invalid=not args.include_invalid
        )
    
    # Get statistics
    stats = loader.get_dataset_stats(filter_config, use_cache=not args.no_cache)
    
    # Display statistics
    print("=== Dataset Statistics ===")
    print(f"Total games: {stats.total_games}")
    print(f"Total turns: {stats.total_turns}")
    print(f"Average game length: {stats.avg_game_length:.1f} turns")
    print(f"Game length range: {stats.min_game_length} - {stats.max_game_length} turns")
    
    print(f"\n=== Outcome Statistics ===")
    print(f"White wins: {stats.white_wins} ({stats.win_rate_white*100:.1f}%)")
    print(f"Black wins: {stats.black_wins} ({stats.win_rate_black*100:.1f}%)")
    print(f"Draws: {stats.draws} ({stats.draw_rate*100:.1f}%)")
    
    print(f"\n=== Storage Statistics ===")
    print(f"Total size: {stats.total_size_mb:.1f}MB")
    print(f"Average size per game: {stats.avg_size_per_game_mb:.3f}MB")
    
    if stats.date_range_start and stats.date_range_end:
        print(f"\n=== Date Range ===")
        print(f"Start: {stats.date_range_start}")
        print(f"End: {stats.date_range_end}")
    
    if stats.agent_versions:
        print(f"\n=== Agent Versions ===")
        for version in stats.agent_versions:
            print(f"  {version}")
    
    if stats.training_runs:
        print(f"\n=== Training Runs ===")
        for run in stats.training_runs:
            print(f"  {run}")
    
    print(f"\n=== Validation Results ===")
    print(f"Validation errors: {stats.validation_errors}")
    print(f"Validation warnings: {stats.validation_warnings}")
    
    # Export to JSON if requested
    if args.output:
        stats_dict = {
            'timestamp': datetime.now().isoformat(),
            'filter_config': filter_config.__dict__ if filter_config else None,
            'statistics': stats.__dict__
        }
        with open(args.output, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        print(f"\nStatistics exported to {args.output}")


def cmd_validate(args):
    """Validate stored data."""
    config = StorageConfig(
        output_dir=args.data_dir,
        parquet_dir=args.parquet_dir,
        validation_enabled=True
    )
    
    data_manager = SelfPlayDataManager(config)
    
    print("Validating stored data...")
    validation_result = data_manager.validate_stored_data()
    
    print(f"\n=== Validation Results ===")
    print(f"Valid: {validation_result.valid}")
    print(f"Errors: {len(validation_result.errors)}")
    print(f"Warnings: {len(validation_result.warnings)}")
    
    if validation_result.errors:
        print(f"\n=== Errors ===")
        for i, error in enumerate(validation_result.errors, 1):
            print(f"{i}. {error}")
    
    if validation_result.warnings:
        print(f"\n=== Warnings ===")
        for i, warning in enumerate(validation_result.warnings, 1):
            print(f"{i}. {warning}")
    
    if validation_result.stats:
        print(f"\n=== Validation Statistics ===")
        for key, value in validation_result.stats.items():
            print(f"{key}: {value}")
    
    # Export results if requested
    if args.output:
        results = {
            'timestamp': datetime.now().isoformat(),
            'valid': validation_result.valid,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'stats': validation_result.stats
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nValidation results exported to {args.output}")


def cmd_export(args):
    """Export data for training."""
    config = StorageConfig(
        output_dir=args.data_dir,
        parquet_dir=args.parquet_dir,
        validation_enabled=not args.skip_validation
    )
    
    loader = DataLoader(config)
    
    # Create filter config
    filter_config = None
    if any([args.min_length, args.max_length, args.winner, args.agent_version, args.training_run]):
        filter_config = FilterConfig(
            min_game_length=args.min_length,
            max_game_length=args.max_length,
            winner_filter=args.winner,
            agent_version_filter=args.agent_version,
            training_run_filter=args.training_run,
            exclude_invalid=not args.include_invalid
        )
    
    # Export data
    output_dir = args.output_dir or "training_export"
    result = loader.export_for_training(
        output_dir=output_dir,
        filter_config=filter_config,
        format=args.format
    )
    
    print(f"Export completed:")
    print(f"  Games exported: {result['games_exported']}")
    print(f"  Files created: {len(result['files_created'])}")
    print(f"  Total size: {result['total_size_mb']:.1f}MB")
    print(f"  Export time: {result['export_time']:.2f}s")
    
    print(f"\nFiles created:")
    for file_path in result['files_created']:
        print(f"  {file_path}")


def cmd_performance(args):
    """Show performance metrics."""
    config = StorageConfig(
        output_dir=args.data_dir,
        parquet_dir=args.parquet_dir
    )
    
    loader = DataLoader(config)
    tracker = PerformanceTracker()
    
    # Load data to generate metrics
    df, metrics = loader.load_games(use_cache=not args.no_cache)
    tracker.record_loading_metrics(metrics)
    
    print("=== Performance Metrics ===")
    print(f"Total games: {metrics.total_games}")
    print(f"Loading time: {metrics.loading_time_seconds:.2f}s")
    print(f"Games per second: {metrics.games_per_second:.1f}")
    print(f"MB per second: {metrics.mb_per_second:.1f}")
    print(f"Memory usage: {metrics.memory_usage_mb:.1f}MB")
    print(f"Files processed: {metrics.files_processed}")
    print(f"Validation time: {metrics.validation_time_seconds:.2f}s")
    
    # Get performance summary
    summary = tracker.get_performance_summary(args.duration_hours)
    if summary['samples'] > 0:
        print(f"\n=== Performance Summary ({args.duration_hours}h) ===")
        print(f"Samples: {summary['samples']}")
        print(f"Average games/sec: {summary['avg_games_per_second']:.1f}")
        print(f"Average MB/sec: {summary['avg_mb_per_second']:.1f}")
        print(f"Peak games/sec: {summary['peak_games_per_second']:.1f}")
        print(f"Peak MB/sec: {summary['peak_mb_per_second']:.1f}")
        print(f"Average memory usage: {summary['avg_memory_usage_mb']:.1f}MB")


def cmd_cleanup(args):
    """Clean up old data files."""
    config = StorageConfig(
        output_dir=args.data_dir,
        parquet_dir=args.parquet_dir
    )
    
    data_manager = SelfPlayDataManager(config)
    
    print(f"Cleaning up old batch files, keeping last {args.keep_last_n}...")
    data_manager.cleanup_old_batches(args.keep_last_n)
    print("Cleanup completed.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Yinsh Self-Play Data Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all games
  python -m yinsh_ml.self_play.cli list
  
  # Show statistics for games with at least 20 turns
  python -m yinsh_ml.self_play.cli stats --min-length 20
  
  # Export white wins to CSV
  python -m yinsh_ml.self_play.cli export --winner white --format csv
  
  # Validate all stored data
  python -m yinsh_ml.self_play.cli validate
  
  # Show performance metrics
  python -m yinsh_ml.self_play.cli performance
        """
    )
    
    parser.add_argument('--data-dir', default='self_play_data',
                       help='Data directory (default: self_play_data)')
    parser.add_argument('--parquet-dir', default='parquet_data',
                       help='Parquet subdirectory (default: parquet_data)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip data validation')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List games command
    list_parser = subparsers.add_parser('list', help='List games')
    list_parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                           help='Output format (default: table)')
    list_parser.add_argument('--output', '-o', help='Output file for json/csv format')
    list_parser.add_argument('--limit', type=int, help='Limit number of games shown')
    list_parser.add_argument('--min-length', type=int, help='Minimum game length')
    list_parser.add_argument('--max-length', type=int, help='Maximum game length')
    list_parser.add_argument('--winner', choices=['white', 'black', 'draw'], help='Filter by winner')
    list_parser.add_argument('--agent-version', help='Filter by agent version')
    list_parser.add_argument('--training-run', help='Filter by training run ID')
    list_parser.add_argument('--include-invalid', action='store_true',
                           help='Include invalid games')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dataset statistics')
    stats_parser.add_argument('--output', '-o', help='Export statistics to JSON file')
    stats_parser.add_argument('--min-length', type=int, help='Minimum game length')
    stats_parser.add_argument('--max-length', type=int, help='Maximum game length')
    stats_parser.add_argument('--winner', choices=['white', 'black', 'draw'], help='Filter by winner')
    stats_parser.add_argument('--agent-version', help='Filter by agent version')
    stats_parser.add_argument('--training-run', help='Filter by training run ID')
    stats_parser.add_argument('--include-invalid', action='store_true',
                            help='Include invalid games')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate stored data')
    validate_parser.add_argument('--output', '-o', help='Export validation results to JSON file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data for training')
    export_parser.add_argument('--output-dir', default='training_export',
                              help='Output directory (default: training_export)')
    export_parser.add_argument('--format', choices=['parquet', 'csv', 'json'], default='parquet',
                              help='Export format (default: parquet)')
    export_parser.add_argument('--min-length', type=int, help='Minimum game length')
    export_parser.add_argument('--max-length', type=int, help='Maximum game length')
    export_parser.add_argument('--winner', choices=['white', 'black', 'draw'], help='Filter by winner')
    export_parser.add_argument('--agent-version', help='Filter by agent version')
    export_parser.add_argument('--training-run', help='Filter by training run ID')
    export_parser.add_argument('--include-invalid', action='store_true',
                              help='Include invalid games')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Show performance metrics')
    perf_parser.add_argument('--duration-hours', type=float, default=24,
                            help='Time period for summary (default: 24)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data files')
    cleanup_parser.add_argument('--keep-last-n', type=int, default=10,
                               help='Number of recent files to keep (default: 10)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'list':
            cmd_list_games(args)
        elif args.command == 'stats':
            cmd_stats(args)
        elif args.command == 'validate':
            cmd_validate(args)
        elif args.command == 'export':
            cmd_export(args)
        elif args.command == 'performance':
            cmd_performance(args)
        elif args.command == 'cleanup':
            cmd_cleanup(args)
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
