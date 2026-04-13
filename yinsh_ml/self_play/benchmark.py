"""Benchmark script for data loading performance and statistics accuracy."""

import time
import logging
import random
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .data_loader import DataLoader, FilterConfig, PerformanceTracker
from .data_storage import StorageConfig, SelfPlayDataManager
from .game_recorder import GameRecord, GameTurn

logger = logging.getLogger(__name__)


class DataBenchmark:
    """Benchmark suite for data loading and statistics."""
    
    def __init__(self, temp_dir: Path = None):
        """Initialize benchmark suite.
        
        Args:
            temp_dir: Temporary directory for test data
        """
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="yinsh_benchmark_"))
        self.config = StorageConfig(
            output_dir=str(self.temp_dir),
            parquet_dir="test_data",
            batch_size=50,  # Smaller batches for testing
            validation_enabled=False  # Disable validation for synthetic test data
        )
        self.data_manager = SelfPlayDataManager(self.config)
        self.loader = DataLoader(self.config)
        self.performance_tracker = PerformanceTracker()
        
        logger.info(f"Initialized benchmark in {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up {self.temp_dir}")
    
    def generate_test_data(self, num_games: int = 1000) -> List[GameRecord]:
        """Generate test game data.
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            List of GameRecord objects
        """
        logger.info(f"Generating {num_games} test games...")
        
        games = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(num_games):
            # Generate random game length (10-100 turns)
            game_length = random.randint(10, 100)
            
            # Generate random outcome (-1 to 1)
            outcome = random.uniform(-1, 1)
            
            # Generate random timestamp within last 30 days
            random_days = random.randint(0, 30)
            random_hours = random.randint(0, 23)
            timestamp = base_time + timedelta(days=random_days, hours=random_hours)
            
            # Generate random agent version
            agent_version = f"v{random.randint(1, 5)}.{random.randint(0, 9)}"
            
            # Generate random training run ID
            training_run_id = f"run_{random.randint(1, 10)}"
            
            # Create game record
            start_time_float = timestamp.timestamp()
            end_time_float = start_time_float + random.uniform(30, 300)  # 30s to 5min game
            
            game_record = GameRecord(
                game_id=f"test_game_{i}",
                start_time=start_time_float,
                end_time=end_time_float,
                duration=end_time_float - start_time_float,
                total_turns=game_length,
                winner="white" if outcome > 0.1 else "black" if outcome < -0.1 else None,
                final_score={"white": 1 if outcome > 0.1 else 0, "black": 1 if outcome < -0.1 else 0},
                turns=[],
                metadata={
                    "agent_version": agent_version,
                    "training_run_id": training_run_id,
                    "timestamp": timestamp.isoformat(),
                    "outcome": outcome,
                    "valid": True
                }
            )
            
            # Add some random turns
            for turn_idx in range(game_length):
                turn = GameTurn(
                    turn_number=turn_idx + 1,
                    current_player=random.choice(['white', 'black']),
                    move={
                        'type': random.choice(['place_ring', 'move_ring', 'remove_run']),
                        'position': (random.randint(0, 10), random.randint(0, 10))
                    },
                    features={
                        'state_hash': f"state_{i}_{turn_idx}",
                        'policy_distribution': np.random.random(121).tolist(),
                        'value_estimate': random.uniform(-1, 1)
                    },
                    timestamp=start_time_float + turn_idx * 2.0  # 2 seconds per turn
                )
                game_record.turns.append(turn)
            
            games.append(game_record)
        
        logger.info(f"Generated {len(games)} test games")
        return games
    
    def store_test_data(self, games: List[GameRecord]) -> None:
        """Store test data using data manager.
        
        Args:
            games: List of GameRecord objects to store
        """
        logger.info(f"Storing {len(games)} games...")
        
        start_time = time.time()
        for game in games:
            self.data_manager.store_game(game)
        
        # Flush remaining data
        self.data_manager.flush_storage()
        
        storage_time = time.time() - start_time
        logger.info(f"Stored {len(games)} games in {storage_time:.2f}s "
                   f"({len(games)/storage_time:.1f} games/sec)")
    
    def benchmark_loading_performance(self, 
                                     filter_configs: List[FilterConfig] = None) -> Dict[str, Any]:
        """Benchmark data loading performance with different configurations.
        
        Args:
            filter_configs: List of filter configurations to test
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting loading performance benchmark...")
        
        if filter_configs is None:
            filter_configs = [
                None,  # No filters
                FilterConfig(min_game_length=20),
                FilterConfig(max_game_length=50),
                FilterConfig(winner_filter='white'),
                FilterConfig(exclude_invalid=True)
            ]
        
        results = {}
        
        for i, filter_config in enumerate(filter_configs):
            config_name = f"config_{i}" if filter_config is None else f"filter_{i}"
            logger.info(f"Testing configuration: {config_name}")
            
            # Clear cache for fair comparison
            self.loader.clear_cache()
            
            # Benchmark loading
            start_time = time.time()
            df, metrics = self.loader.load_games(
                filter_config=filter_config,
                use_cache=False,
                validate=True
            )
            end_time = time.time()
            
            # Record metrics
            self.performance_tracker.record_loading_metrics(metrics)
            
            results[config_name] = {
                'filter_config': filter_config.__dict__ if filter_config else None,
                'games_loaded': len(df),
                'loading_time': end_time - start_time,
                'games_per_second': metrics.games_per_second,
                'mb_per_second': metrics.mb_per_second,
                'memory_usage_mb': metrics.memory_usage_mb,
                'validation_time': metrics.validation_time_seconds,
                'files_processed': metrics.files_processed
            }
            
            logger.info(f"  Loaded {len(df)} games in {end_time - start_time:.2f}s "
                       f"({metrics.games_per_second:.1f} games/sec)")
        
        return results
    
    def benchmark_incremental_loading(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark incremental loading with different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting incremental loading benchmark...")
        
        if batch_sizes is None:
            batch_sizes = [100, 500, 1000, 2000]
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            start_time = time.time()
            total_games = 0
            total_batches = 0
            
            for batch_df, metrics in self.loader.load_games_incremental(
                batch_size=batch_size,
                filter_config=None
            ):
                total_games += len(batch_df)
                total_batches += 1
            
            end_time = time.time()
            
            results[f"batch_{batch_size}"] = {
                'batch_size': batch_size,
                'total_games': total_games,
                'total_batches': total_batches,
                'loading_time': end_time - start_time,
                'games_per_second': total_games / (end_time - start_time) if end_time > start_time else 0,
                'avg_batch_time': (end_time - start_time) / total_batches if total_batches > 0 else 0
            }
            
            logger.info(f"  Processed {total_games} games in {total_batches} batches "
                       f"({total_games/(end_time - start_time):.1f} games/sec)")
        
        return results
    
    def benchmark_statistics_accuracy(self) -> Dict[str, Any]:
        """Benchmark statistics generation and verify accuracy.
        
        Returns:
            Dictionary with accuracy verification results
        """
        logger.info("Starting statistics accuracy benchmark...")
        
        # Load all data
        df, _ = self.loader.load_games(use_cache=False, validate=False)
        
        if df.empty:
            logger.warning("No data available for statistics benchmark")
            return {'error': 'No data available'}
        
        # Generate statistics using DataLoader
        start_time = time.time()
        stats = self.loader.get_dataset_stats(use_cache=False)
        stats_time = time.time() - start_time
        
        # Manual calculation for verification
        manual_stats = self._calculate_manual_stats(df)
        
        # Compare results
        accuracy_results = {
            'total_games': {
                'loader': stats.total_games,
                'manual': manual_stats['total_games'],
                'match': stats.total_games == manual_stats['total_games']
            },
            'avg_game_length': {
                'loader': stats.avg_game_length,
                'manual': manual_stats['avg_game_length'],
                'match': abs(stats.avg_game_length - manual_stats['avg_game_length']) < 0.001
            },
            'white_wins': {
                'loader': stats.white_wins,
                'manual': manual_stats['white_wins'],
                'match': stats.white_wins == manual_stats['white_wins']
            },
            'black_wins': {
                'loader': stats.black_wins,
                'manual': manual_stats['black_wins'],
                'match': stats.black_wins == manual_stats['black_wins']
            },
            'draws': {
                'loader': stats.draws,
                'manual': manual_stats['draws'],
                'match': stats.draws == manual_stats['draws']
            }
        }
        
        # Check if all statistics match
        all_match = all(result['match'] for result in accuracy_results.values())
        
        results = {
            'statistics_time': stats_time,
            'all_statistics_match': all_match,
            'accuracy_results': accuracy_results,
            'loader_stats': stats.__dict__,
            'manual_stats': manual_stats
        }
        
        logger.info(f"Statistics generation took {stats_time:.3f}s")
        logger.info(f"All statistics match: {all_match}")
        
        return results
    
    def benchmark_export_performance(self, formats: List[str] = None) -> Dict[str, Any]:
        """Benchmark export performance with different formats.
        
        Args:
            formats: List of export formats to test
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting export performance benchmark...")
        
        if formats is None:
            formats = ['parquet', 'csv', 'json']
        
        results = {}
        export_dir = self.temp_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        for format_type in formats:
            logger.info(f"Testing export format: {format_type}")
            
            start_time = time.time()
            result = self.loader.export_for_training(
                output_dir=export_dir,
                format=format_type
            )
            end_time = time.time()
            
            results[f"export_{format_type}"] = {
                'format': format_type,
                'games_exported': result['games_exported'],
                'files_created': result['files_created'] if isinstance(result['files_created'], list) else [result['files_created']],
                'export_time': end_time - start_time,
                'size_mb': result['total_size_mb'],
                'games_per_second': result['games_exported'] / (end_time - start_time) if end_time > start_time else 0
            }
            
            logger.info(f"  Exported {result['games_exported']} games in {end_time - start_time:.2f}s "
                       f"({result['total_size_mb']:.1f}MB)")
        
        return results
    
    def run_full_benchmark(self, num_games: int = 1000) -> Dict[str, Any]:
        """Run complete benchmark suite.
        
        Args:
            num_games: Number of test games to generate
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info(f"Starting full benchmark suite with {num_games} games...")
        
        try:
            # Generate and store test data
            games = self.generate_test_data(num_games)
            self.store_test_data(games)
            
            # Run all benchmarks
            results = {
                'benchmark_info': {
                    'num_games': num_games,
                    'temp_dir': str(self.temp_dir),
                    'timestamp': datetime.now().isoformat()
                },
                'loading_performance': self.benchmark_loading_performance(),
                'incremental_loading': self.benchmark_incremental_loading(),
                'statistics_accuracy': self.benchmark_statistics_accuracy(),
                'export_performance': self.benchmark_export_performance(),
                'performance_summary': self.performance_tracker.get_performance_summary(24)
            }
            
            logger.info("Benchmark suite completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        finally:
            # Cleanup
            self.cleanup()
    
    def _calculate_manual_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Manually calculate statistics for verification."""
        if df.empty:
            return {}
        
        total_games = len(df)
        total_turns = df['total_turns'].sum() if 'total_turns' in df.columns else 0
        avg_game_length = total_turns / total_games if total_games > 0 else 0
        
        # Outcome statistics - check metadata for outcome
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
                draws += 1  # Default to draw if no outcome info
        
        return {
            'total_games': total_games,
            'total_turns': total_turns,
            'avg_game_length': avg_game_length,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws
        }


def main():
    """Main benchmark entry point."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Yinsh Data Loading Benchmark")
    parser.add_argument('--num-games', type=int, default=1000,
                       help='Number of test games to generate (default: 1000)')
    parser.add_argument('--output', '-o', help='Output file for benchmark results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmark
    benchmark = DataBenchmark()
    
    try:
        results = benchmark.run_full_benchmark(args.num_games)
        
        # Print summary
        print("=== Benchmark Summary ===")
        print(f"Test games: {results['benchmark_info']['num_games']}")
        
        # Loading performance summary
        loading_results = results['loading_performance']
        if loading_results:
            best_config = max(loading_results.items(), 
                            key=lambda x: x[1]['games_per_second'])
            print(f"Best loading performance: {best_config[1]['games_per_second']:.1f} games/sec "
                  f"({best_config[0]})")
        
        # Statistics accuracy
        stats_results = results['statistics_accuracy']
        if 'all_statistics_match' in stats_results:
            print(f"Statistics accuracy: {'PASS' if stats_results['all_statistics_match'] else 'FAIL'}")
        
        # Export performance summary
        export_results = results['export_performance']
        if export_results:
            best_export = max(export_results.items(),
                            key=lambda x: x[1]['games_per_second'])
            print(f"Best export performance: {best_export[1]['games_per_second']:.1f} games/sec "
                  f"({best_export[0]})")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
