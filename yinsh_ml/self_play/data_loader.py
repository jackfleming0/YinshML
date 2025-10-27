"""Data loading utilities and performance tracking for Yinsh self-play data."""

import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from .data_storage import SelfPlayDataManager, StorageConfig, ValidationResult
from .game_recorder import GameRecord

logger = logging.getLogger(__name__)


@dataclass
class LoadingMetrics:
    """Metrics for data loading performance."""
    total_games: int = 0
    total_turns: int = 0
    total_size_bytes: int = 0
    loading_time_seconds: float = 0.0
    games_per_second: float = 0.0
    mb_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    files_processed: int = 0
    validation_time_seconds: float = 0.0


@dataclass
class FilterConfig:
    """Configuration for data filtering."""
    min_game_length: Optional[int] = None
    max_game_length: Optional[int] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    winner_filter: Optional[str] = None  # 'white', 'black', 'draw'
    agent_version_filter: Optional[str] = None
    training_run_filter: Optional[str] = None
    exclude_invalid: bool = True


@dataclass
class DatasetStats:
    """Statistics for a dataset."""
    total_games: int = 0
    total_turns: int = 0
    avg_game_length: float = 0.0
    min_game_length: int = 0
    max_game_length: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    win_rate_white: float = 0.0
    win_rate_black: float = 0.0
    draw_rate: float = 0.0
    total_size_mb: float = 0.0
    avg_size_per_game_mb: float = 0.0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    agent_versions: List[str] = None
    training_runs: List[str] = None
    validation_errors: int = 0
    validation_warnings: int = 0
    
    def __post_init__(self):
        if self.agent_versions is None:
            self.agent_versions = []
        if self.training_runs is None:
            self.training_runs = []


class DataLoader:
    """Efficient data loading and filtering for self-play data."""
    
    def __init__(self, config: StorageConfig = None):
        """Initialize data loader.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.data_manager = SelfPlayDataManager(self.config)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        logger.info("Initialized DataLoader")
    
    def load_games(self, 
                   filter_config: FilterConfig = None,
                   use_cache: bool = True,
                   validate: bool = True) -> Tuple[pd.DataFrame, LoadingMetrics]:
        """Load games with optional filtering and performance tracking.
        
        Args:
            filter_config: Optional filtering configuration
            use_cache: Whether to use cached data if available
            validate: Whether to validate loaded data
            
        Returns:
            Tuple of (DataFrame, LoadingMetrics)
        """
        start_time = time.time()
        cache_key = self._get_cache_key(filter_config)
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug("Using cached data")
            df = self._cache[cache_key]
            metrics = LoadingMetrics(
                total_games=len(df),
                loading_time_seconds=0.0,
                games_per_second=float('inf'),
                files_processed=0
            )
            return df, metrics
        
        # Load raw data
        logger.info("Loading games from storage...")
        df = self.data_manager.load_all_games()
        
        if df.empty:
            logger.warning("No games found in storage")
            metrics = LoadingMetrics(loading_time_seconds=time.time() - start_time)
            return df, metrics
        
        # Apply filters
        if filter_config:
            df = self._apply_filters(df, filter_config)
            logger.info(f"Applied filters, {len(df)} games remaining")
        
        # Validate if requested
        validation_time = 0.0
        if validate:
            validation_start = time.time()
            validation_result = self.data_manager.validate_stored_data()
            validation_time = time.time() - validation_start
            
            if not validation_result.valid:
                logger.warning(f"Data validation found issues: {validation_result.errors}")
        
        # Calculate metrics
        loading_time = time.time() - start_time
        total_size = self._estimate_dataframe_size(df)
        
        metrics = LoadingMetrics(
            total_games=len(df),
            total_turns=df['turn_count'].sum() if 'turn_count' in df.columns else 0,
            total_size_bytes=total_size,
            loading_time_seconds=loading_time,
            games_per_second=len(df) / loading_time if loading_time > 0 else 0,
            mb_per_second=(total_size / (1024 * 1024)) / loading_time if loading_time > 0 else 0,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            files_processed=self._count_processed_files(),
            validation_time_seconds=validation_time
        )
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = df.copy()
            self._cache_timestamps[cache_key] = time.time()
        
        logger.info(f"Loaded {len(df)} games in {loading_time:.2f}s "
                   f"({metrics.games_per_second:.1f} games/sec)")
        
        return df, metrics
    
    def load_games_incremental(self, 
                              batch_size: int = 1000,
                              filter_config: FilterConfig = None) -> Iterator[Tuple[pd.DataFrame, LoadingMetrics]]:
        """Load games in batches for memory-efficient processing of large datasets.
        
        Args:
            batch_size: Number of games per batch
            filter_config: Optional filtering configuration
            
        Yields:
            Iterator of (DataFrame, LoadingMetrics) tuples
        """
        logger.info(f"Starting incremental loading with batch size {batch_size}")
        
        # Get all file paths
        storage_files = self._get_storage_files()
        if not storage_files:
            logger.warning("No storage files found")
            return
        
        total_games_processed = 0
        total_time = 0.0
        
        for file_path in storage_files:
            start_time = time.time()
            
            try:
                # Load single file
                if self.data_manager.storage.use_json_fallback:
                    df = pd.read_json(file_path)
                else:
                    df = pd.read_parquet(file_path)
                
                # Apply filters
                if filter_config:
                    df = self._apply_filters(df, filter_config)
                
                # Process in batches
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size].copy()
                    
                    if batch_df.empty:
                        continue
                    
                    batch_time = time.time() - start_time
                    total_games_processed += len(batch_df)
                    total_time += batch_time
                    
                    metrics = LoadingMetrics(
                        total_games=len(batch_df),
                        total_turns=batch_df['turn_count'].sum() if 'turn_count' in batch_df.columns else 0,
                        loading_time_seconds=batch_time,
                        games_per_second=len(batch_df) / batch_time if batch_time > 0 else 0,
                        files_processed=1
                    )
                    
                    yield batch_df, metrics
                    
                    start_time = time.time()  # Reset timer for next batch
                
            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {e}")
                continue
        
        logger.info(f"Incremental loading completed: {total_games_processed} games in {total_time:.2f}s")
    
    def get_dataset_stats(self, 
                         filter_config: FilterConfig = None,
                         use_cache: bool = True) -> DatasetStats:
        """Generate comprehensive statistics for the dataset.
        
        Args:
            filter_config: Optional filtering configuration
            use_cache: Whether to use cached data
            
        Returns:
            DatasetStats object
        """
        df, metrics = self.load_games(filter_config, use_cache, validate=False)
        
        if df.empty:
            return DatasetStats()
        
        # Basic game statistics
        total_games = len(df)
        total_turns = df['total_turns'].sum() if 'total_turns' in df.columns else 0
        
        # Game length statistics
        game_lengths = df['total_turns'] if 'total_turns' in df.columns else []
        avg_game_length = game_lengths.mean() if len(game_lengths) > 0 else 0.0
        min_game_length = game_lengths.min() if len(game_lengths) > 0 else 0
        max_game_length = game_lengths.max() if len(game_lengths) > 0 else 0
        
        # Outcome statistics - extract from metadata
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
        
        win_rate_white = white_wins / total_games if total_games > 0 else 0.0
        win_rate_black = black_wins / total_games if total_games > 0 else 0.0
        draw_rate = draws / total_games if total_games > 0 else 0.0
        
        # Size statistics
        total_size_mb = metrics.total_size_bytes / (1024 * 1024)
        avg_size_per_game_mb = total_size_mb / total_games if total_games > 0 else 0.0
        
        # Date range - extract from metadata
        date_range_start = None
        date_range_end = None
        agent_versions = []
        training_runs = []
        
        for _, row in df.iterrows():
            if 'metadata' in row and isinstance(row['metadata'], dict):
                metadata = row['metadata']
                
                # Extract timestamp
                if 'timestamp' in metadata:
                    try:
                        timestamp = pd.to_datetime(metadata['timestamp'])
                        if date_range_start is None or timestamp < date_range_start:
                            date_range_start = timestamp
                        if date_range_end is None or timestamp > date_range_end:
                            date_range_end = timestamp
                    except:
                        pass
                
                # Extract agent version
                if 'agent_version' in metadata:
                    agent_versions.append(metadata['agent_version'])
                
                # Extract training run
                if 'training_run_id' in metadata:
                    training_runs.append(metadata['training_run_id'])
        
        # Remove duplicates
        agent_versions = list(set(agent_versions))
        training_runs = list(set(training_runs))
        
        # Validation statistics
        validation_result = self.data_manager.validate_stored_data()
        validation_errors = len(validation_result.errors)
        validation_warnings = len(validation_result.warnings)
        
        return DatasetStats(
            total_games=total_games,
            total_turns=total_turns,
            avg_game_length=avg_game_length,
            min_game_length=min_game_length,
            max_game_length=max_game_length,
            white_wins=white_wins,
            black_wins=black_wins,
            draws=draws,
            win_rate_white=win_rate_white,
            win_rate_black=win_rate_black,
            draw_rate=draw_rate,
            total_size_mb=total_size_mb,
            avg_size_per_game_mb=avg_size_per_game_mb,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            agent_versions=agent_versions,
            training_runs=training_runs,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings
        )
    
    def export_for_training(self, 
                           output_dir: Union[str, Path],
                           filter_config: FilterConfig = None,
                           format: str = 'parquet') -> Dict[str, Any]:
        """Export filtered data for training pipelines.
        
        Args:
            output_dir: Directory to save exported data
            filter_config: Optional filtering configuration
            format: Export format ('parquet', 'csv', 'json')
            
        Returns:
            Dictionary with export information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting data to {output_path} in {format} format")
        
        df, metrics = self.load_games(filter_config, use_cache=False, validate=True)
        
        if df.empty:
            logger.warning("No data to export")
            return {'games_exported': 0, 'files_created': [], 'total_size_mb': 0.0, 'export_time': 0.0}
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"training_data_{timestamp}"
        
        files_created = []
        
        if format == 'parquet':
            output_file = output_path / f"{base_filename}.parquet"
            df.to_parquet(output_file, compression='snappy')
            files_created.append(str(output_file))
            
        elif format == 'csv':
            output_file = output_path / f"{base_filename}.csv"
            df.to_csv(output_file, index=False)
            files_created.append(str(output_file))
            
        elif format == 'json':
            output_file = output_path / f"{base_filename}.json"
            df.to_json(output_file, orient='records', indent=2)
            files_created.append(str(output_file))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Export metadata
        metadata_file = output_path / f"{base_filename}_metadata.json"
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'total_games': len(df),
            'total_turns': df['turn_count'].sum() if 'turn_count' in df.columns else 0,
            'filter_config': asdict(filter_config) if filter_config else None,
            'format': format,
            'files_created': files_created,
            'loading_metrics': asdict(metrics)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        files_created.append(str(metadata_file))
        
        logger.info(f"Exported {len(df)} games to {len(files_created)} files")
        
        return {
            'games_exported': len(df),
            'files_created': files_created,
            'total_size_mb': metrics.total_size_bytes / (1024 * 1024),
            'export_time': metrics.loading_time_seconds
        }
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Data cache cleared")
    
    def _apply_filters(self, df: pd.DataFrame, filter_config: FilterConfig) -> pd.DataFrame:
        """Apply filtering configuration to DataFrame."""
        filtered_df = df.copy()
        
        # Game length filters
        if filter_config.min_game_length is not None:
            filtered_df = filtered_df[filtered_df['total_turns'] >= filter_config.min_game_length]
        
        if filter_config.max_game_length is not None:
            filtered_df = filtered_df[filtered_df['total_turns'] <= filter_config.max_game_length]
        
        # Date range filters - extract from metadata
        if filter_config.date_range_start is not None:
            mask = []
            for _, row in filtered_df.iterrows():
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    try:
                        timestamp = pd.to_datetime(row['metadata'].get('timestamp'))
                        mask.append(timestamp >= filter_config.date_range_start)
                    except:
                        mask.append(False)
                else:
                    mask.append(False)
            filtered_df = filtered_df[mask]
        
        if filter_config.date_range_end is not None:
            mask = []
            for _, row in filtered_df.iterrows():
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    try:
                        timestamp = pd.to_datetime(row['metadata'].get('timestamp'))
                        mask.append(timestamp <= filter_config.date_range_end)
                    except:
                        mask.append(False)
                else:
                    mask.append(False)
            filtered_df = filtered_df[mask]
        
        # Winner filter - extract from metadata
        if filter_config.winner_filter is not None:
            mask = []
            for _, row in filtered_df.iterrows():
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    outcome = row['metadata'].get('outcome', 0)
                    if filter_config.winner_filter == 'white':
                        mask.append(outcome > 0.1)
                    elif filter_config.winner_filter == 'black':
                        mask.append(outcome < -0.1)
                    elif filter_config.winner_filter == 'draw':
                        mask.append(abs(outcome) <= 0.1)
                    else:
                        mask.append(True)
                else:
                    mask.append(False)
            filtered_df = filtered_df[mask]
        
        # Agent version filter - extract from metadata
        if filter_config.agent_version_filter is not None:
            mask = []
            for _, row in filtered_df.iterrows():
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    agent_version = row['metadata'].get('agent_version')
                    mask.append(agent_version == filter_config.agent_version_filter)
                else:
                    mask.append(False)
            filtered_df = filtered_df[mask]
        
        # Training run filter - extract from metadata
        if filter_config.training_run_filter is not None:
            mask = []
            for _, row in filtered_df.iterrows():
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    training_run_id = row['metadata'].get('training_run_id')
                    mask.append(training_run_id == filter_config.training_run_filter)
                else:
                    mask.append(False)
            filtered_df = filtered_df[mask]
        
        # Exclude invalid games - extract from metadata
        if filter_config.exclude_invalid:
            mask = []
            for _, row in filtered_df.iterrows():
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    valid = row['metadata'].get('valid', True)
                    mask.append(valid)
                else:
                    mask.append(True)  # Include if no validity info
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def _get_cache_key(self, filter_config: FilterConfig) -> str:
        """Generate cache key for filter configuration."""
        if filter_config is None:
            return "no_filters"
        
        key_parts = []
        if filter_config.min_game_length is not None:
            key_parts.append(f"min_len_{filter_config.min_game_length}")
        if filter_config.max_game_length is not None:
            key_parts.append(f"max_len_{filter_config.max_game_length}")
        if filter_config.winner_filter is not None:
            key_parts.append(f"winner_{filter_config.winner_filter}")
        if filter_config.agent_version_filter is not None:
            key_parts.append(f"agent_{filter_config.agent_version_filter}")
        if filter_config.training_run_filter is not None:
            key_parts.append(f"run_{filter_config.training_run_filter}")
        if filter_config.exclude_invalid:
            key_parts.append("exclude_invalid")
        
        return "_".join(key_parts) if key_parts else "no_filters"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cache_age = time.time() - self._cache_timestamps[cache_key]
        return cache_age < self.cache_ttl_seconds
    
    def _estimate_dataframe_size(self, df: pd.DataFrame) -> int:
        """Estimate DataFrame size in bytes."""
        return df.memory_usage(deep=True).sum()
    
    def _count_processed_files(self) -> int:
        """Count number of files processed."""
        storage_files = self._get_storage_files()
        return len(storage_files)
    
    def _get_storage_files(self) -> List[Path]:
        """Get list of storage files."""
        parquet_dir = self.data_manager.storage.parquet_dir
        
        if self.data_manager.storage.use_json_fallback:
            return list(parquet_dir.glob("*.json"))
        else:
            return list(parquet_dir.glob("*.parquet"))


class PerformanceTracker:
    """Tracks performance metrics for data operations."""
    
    def __init__(self):
        self.metrics_history: List[LoadingMetrics] = []
        self.start_time = time.time()
    
    def record_loading_metrics(self, metrics: LoadingMetrics) -> None:
        """Record loading metrics."""
        self.metrics_history.append(metrics)
        logger.debug(f"Recorded metrics: {metrics.games_per_second:.1f} games/sec, "
                    f"{metrics.mb_per_second:.1f} MB/sec")
    
    def get_performance_summary(self, duration_hours: float = 24) -> Dict[str, Any]:
        """Get performance summary over time period."""
        cutoff_time = time.time() - (duration_hours * 3600)
        recent_metrics = [m for m in self.metrics_history 
                         if time.time() - m.loading_time_seconds >= cutoff_time]
        
        if not recent_metrics:
            return {'period_hours': duration_hours, 'samples': 0}
        
        total_games = sum(m.total_games for m in recent_metrics)
        total_time = sum(m.loading_time_seconds for m in recent_metrics)
        total_size_mb = sum(m.total_size_bytes for m in recent_metrics) / (1024 * 1024)
        
        return {
            'period_hours': duration_hours,
            'samples': len(recent_metrics),
            'total_games': total_games,
            'total_time_seconds': total_time,
            'total_size_mb': total_size_mb,
            'avg_games_per_second': total_games / total_time if total_time > 0 else 0,
            'avg_mb_per_second': total_size_mb / total_time if total_time > 0 else 0,
            'avg_memory_usage_mb': np.mean([m.memory_usage_mb for m in recent_metrics]),
            'peak_games_per_second': max(m.games_per_second for m in recent_metrics),
            'peak_mb_per_second': max(m.mb_per_second for m in recent_metrics)
        }
    
    def export_metrics(self, output_file: Union[str, Path]) -> None:
        """Export metrics to JSON file."""
        metrics_data = [asdict(m) for m in self.metrics_history]
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(metrics_data)} metrics records to {output_file}")
