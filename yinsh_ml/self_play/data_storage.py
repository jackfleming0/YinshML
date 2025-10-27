"""Efficient data storage and validation system for Yinsh self-play."""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import parquet support, fall back to JSON if not available
try:
    import pyarrow as pa
    PARQUET_AVAILABLE = True
except ImportError:
    try:
        import fastparquet
        PARQUET_AVAILABLE = True
    except ImportError:
        PARQUET_AVAILABLE = False

from .game_recorder import GameRecord, GameTurn

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for data storage system."""
    output_dir: str = "self_play_data"
    parquet_dir: str = "parquet_data"
    batch_size: int = 100  # Games per parquet file
    compression: str = "snappy"  # Parquet compression
    validation_enabled: bool = True
    checksum_enabled: bool = True
    backup_enabled: bool = True


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class ParquetDataStorage:
    """Efficient parquet-based storage for self-play data."""
    
    def __init__(self, config: StorageConfig = None):
        """Initialize parquet data storage.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.output_dir = Path(self.config.output_dir)
        self.parquet_dir = self.output_dir / self.config.parquet_dir
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Track batches
        self.current_batch: List[GameRecord] = []
        self.batch_count = 0
        
        # Check if parquet is available
        if not PARQUET_AVAILABLE:
            logger.warning("Parquet support not available, falling back to JSON storage")
            self.use_json_fallback = True
        else:
            self.use_json_fallback = False
        
        logger.info(f"Initialized ParquetDataStorage with output: {self.output_dir}")
    
    def store_game_record(self, game_record: GameRecord) -> None:
        """Store a single game record.
        
        Args:
            game_record: Game record to store
        """
        self.current_batch.append(game_record)
        
        # Write batch when it reaches batch_size
        logger.debug(f"Current batch size: {len(self.current_batch)}, target: {self.config.batch_size}")
        if len(self.current_batch) >= self.config.batch_size:
            logger.info(f"Batch size reached, writing batch")
            self._write_batch()
    
    def _write_batch(self) -> None:
        """Write current batch to parquet file."""
        if not self.current_batch:
            logger.warning("_write_batch called with empty batch")
            return
        
        logger.info(f"Writing batch {self.batch_count} with {len(self.current_batch)} games")
        
        # Convert batch to DataFrame
        df = self._batch_to_dataframe(self.current_batch)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_json_fallback:
            # Use JSON fallback
            filename = f"games_batch_{self.batch_count:04d}_{timestamp}.json"
            filepath = self.parquet_dir / filename
            
            try:
                # Write to JSON
                df.to_json(filepath, orient='records', indent=2)
                logger.info(f"Wrote batch {self.batch_count} with {len(self.current_batch)} games to {filepath} (JSON)")
            except Exception as e:
                logger.error(f"Failed to write batch {self.batch_count}: {e}")
                raise
        else:
            # Use parquet
            filename = f"games_batch_{self.batch_count:04d}_{timestamp}.parquet"
            filepath = self.parquet_dir / filename
            
            try:
                # Write to parquet
                df.to_parquet(
                    filepath,
                    compression=self.config.compression,
                    index=False
                )
                logger.info(f"Wrote batch {self.batch_count} with {len(self.current_batch)} games to {filepath}")
            except Exception as e:
                logger.error(f"Failed to write batch {self.batch_count}: {e}")
                raise
        
        # Clear batch and increment counter
        self.current_batch.clear()
        self.batch_count += 1
    
    def _batch_to_dataframe(self, game_records: List[GameRecord]) -> pd.DataFrame:
        """Convert batch of game records to DataFrame.
        
        Args:
            game_records: List of game records
            
        Returns:
            DataFrame with flattened game data
        """
        rows = []
        
        for game_record in game_records:
            # Game-level metadata
            game_metadata = {
                'game_id': game_record.game_id,
                'start_time': game_record.start_time,
                'end_time': game_record.end_time,
                'duration': game_record.duration,
                'total_turns': game_record.total_turns,
                'winner': game_record.winner,
                'white_score': game_record.final_score.get('white', 0),
                'black_score': game_record.final_score.get('black', 0),
                'final_phase': game_record.metadata.get('final_phase'),
                'total_moves': game_record.metadata.get('total_moves', 0),
                'feature_count': game_record.metadata.get('feature_count', 0)
            }
            
            # Add turn-level data
            for turn in game_record.turns:
                row = game_metadata.copy()
                row.update({
                    'turn_number': turn.turn_number,
                    'current_player': turn.current_player,
                    'move_type': turn.move.get('type'),
                    'move_source': turn.move.get('source'),
                    'move_destination': turn.move.get('destination'),
                    'move_markers': json.dumps(turn.move.get('markers', [])),
                    'turn_timestamp': turn.timestamp
                })
                
                # Add feature columns
                row.update(turn.features)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def flush(self) -> None:
        """Flush remaining games in current batch."""
        if self.current_batch:
            self._write_batch()
    
    def load_games(self, batch_file: Optional[str] = None) -> pd.DataFrame:
        """Load games from parquet/JSON files.
        
        Args:
            batch_file: Specific batch file to load, or None for all
            
        Returns:
            DataFrame with game data
        """
        if batch_file:
            filepath = self.parquet_dir / batch_file
            if not filepath.exists():
                raise FileNotFoundError(f"Batch file not found: {filepath}")
            
            if self.use_json_fallback:
                return pd.read_json(filepath)
            else:
                return pd.read_parquet(filepath)
        
        # Load all files
        if self.use_json_fallback:
            data_files = list(self.parquet_dir.glob("*.json"))
        else:
            data_files = list(self.parquet_dir.glob("*.parquet"))
            
        if not data_files:
            logger.warning("No data files found")
            return pd.DataFrame()
        
        dfs = []
        for filepath in data_files:
            try:
                if self.use_json_fallback:
                    df = pd.read_json(filepath)
                else:
                    df = pd.read_parquet(filepath)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        if self.use_json_fallback:
            data_files = list(self.parquet_dir.glob("*.json"))
        else:
            data_files = list(self.parquet_dir.glob("*.parquet"))
        
        total_size = sum(f.stat().st_size for f in data_files)
        
        # Load sample to get data stats
        if data_files:
            try:
                if self.use_json_fallback:
                    sample_df = pd.read_json(data_files[0])
                else:
                    sample_df = pd.read_parquet(data_files[0])
                sample_stats = {
                    'columns': len(sample_df.columns),
                    'sample_rows': len(sample_df),
                    'memory_usage': sample_df.memory_usage(deep=True).sum()
                }
            except Exception as e:
                logger.error(f"Failed to analyze sample file: {e}")
                sample_stats = {}
        else:
            sample_stats = {}
        
        return {
            'total_files': len(data_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'compression': self.config.compression if not self.use_json_fallback else 'none',
            'batch_size': self.config.batch_size,
            'current_batch_size': len(self.current_batch),
            'batch_count': self.batch_count,
            'storage_format': 'json' if self.use_json_fallback else 'parquet',
            **sample_stats
        }


class DataValidator:
    """Validates self-play data for completeness and quality."""
    
    def __init__(self, config: StorageConfig = None):
        """Initialize data validator.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.logger = logging.getLogger(f"{__name__}.DataValidator")
    
    def validate_game_record(self, game_record: GameRecord) -> ValidationResult:
        """Validate a single game record.
        
        Args:
            game_record: Game record to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check basic structure
        if not game_record.game_id:
            errors.append("Missing game ID")
        
        if game_record.total_turns <= 0:
            errors.append("Invalid total turns")
        
        if not game_record.turns:
            errors.append("No turns recorded")
        
        # Check turn consistency
        expected_turn_numbers = list(range(1, len(game_record.turns) + 1))
        actual_turn_numbers = [turn.turn_number for turn in game_record.turns]
        
        if expected_turn_numbers != actual_turn_numbers:
            errors.append("Turn numbers are not sequential")
        
        # Check feature consistency
        if game_record.turns:
            first_turn_features = set(game_record.turns[0].features.keys())
            for i, turn in enumerate(game_record.turns[1:], 1):
                turn_features = set(turn.features.keys())
                if first_turn_features != turn_features:
                    errors.append(f"Turn {i+1} has inconsistent features")
                    break
        
        # Check feature value ranges and distributions
        feature_ranges = self._validate_feature_ranges(game_record.turns)
        stats['feature_ranges'] = feature_ranges
        
        # Check feature distributions and detect anomalies
        distribution_validation = self._validate_feature_distributions(game_record.turns)
        errors.extend(distribution_validation['errors'])
        warnings.extend(distribution_validation['warnings'])
        stats['feature_distributions'] = distribution_validation['stats']
        
        # Check feature consistency across turns
        consistency_validation = self._validate_feature_consistency(game_record.turns)
        errors.extend(consistency_validation['errors'])
        warnings.extend(consistency_validation['warnings'])
        stats['feature_consistency'] = consistency_validation['stats']
        
        # Check for missing values
        missing_values = self._check_missing_values(game_record.turns)
        if missing_values:
            warnings.append(f"Missing values in features: {missing_values}")
        
        # Check game outcome consistency
        if game_record.winner and game_record.final_score:
            white_score = game_record.final_score.get('white', 0)
            black_score = game_record.final_score.get('black', 0)
            
            if game_record.winner == '1' and white_score <= black_score:
                warnings.append("Winner doesn't match score (white won but score suggests otherwise)")
            elif game_record.winner == '-1' and black_score <= white_score:
                warnings.append("Winner doesn't match score (black won but score suggests otherwise)")
        
        # Enhanced completeness checks
        completeness_checks = self._check_completeness(game_record)
        errors.extend(completeness_checks['errors'])
        warnings.extend(completeness_checks['warnings'])
        stats['completeness'] = completeness_checks['stats']
        
        # Check move sequence validity
        move_validity = self._validate_move_sequence(game_record.turns)
        errors.extend(move_validity['errors'])
        warnings.extend(move_validity['warnings'])
        stats['move_validity'] = move_validity['stats']
        
        # Check game outcome completeness
        outcome_completeness = self._check_outcome_completeness(game_record)
        errors.extend(outcome_completeness['errors'])
        warnings.extend(outcome_completeness['warnings'])
        stats['outcome_completeness'] = outcome_completeness['stats']
        
        # Check game state transitions
        state_transitions = self._validate_game_state_transitions(game_record.turns)
        errors.extend(state_transitions['errors'])
        warnings.extend(state_transitions['warnings'])
        stats['state_transitions'] = state_transitions['stats']
        
        # Check board state consistency
        board_consistency = self._validate_board_state_consistency(game_record.turns)
        errors.extend(board_consistency['errors'])
        warnings.extend(board_consistency['warnings'])
        stats['board_consistency'] = board_consistency['stats']
        
        # Check move legality
        move_legality = self._validate_move_legality(game_record.turns)
        errors.extend(move_legality['errors'])
        warnings.extend(move_legality['warnings'])
        stats['move_legality'] = move_legality['stats']
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a DataFrame of game data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        stats = {}
        
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, stats=stats)
        
        # Check required columns
        required_columns = ['game_id', 'turn_number', 'current_player', 'move_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate game_id + turn_number combinations
        duplicates = df.duplicated(subset=['game_id', 'turn_number'])
        if duplicates.any():
            errors.append(f"Found {duplicates.sum()} duplicate game_id + turn_number combinations")
        
        # Check feature value ranges
        feature_columns = [col for col in df.columns if col not in required_columns + ['start_time', 'end_time', 'duration', 'timestamp', 'turn_timestamp']]
        feature_stats = {}
        
        for col in feature_columns:
            if df[col].dtype in ['float64', 'int64']:
                col_stats = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'null_count': df[col].isnull().sum()
                }
                feature_stats[col] = col_stats
                
                # Check for reasonable ranges
                if col.startswith('ring_') or col.startswith('marker_'):
                    if col_stats['min'] < 0 or col_stats['max'] > 1:
                        warnings.append(f"Feature {col} has values outside [0,1] range")
        
        stats['feature_statistics'] = feature_stats
        stats['total_rows'] = len(df)
        stats['total_games'] = df['game_id'].nunique()
        stats['total_turns'] = len(df)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def _validate_feature_ranges(self, turns: List[GameTurn]) -> Dict[str, Dict[str, float]]:
        """Validate feature value ranges and distributions.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with feature range statistics
        """
        if not turns:
            return {}
        
        # Collect all feature values
        feature_values = {}
        for turn in turns:
            for feature_name, value in turn.features.items():
                if isinstance(value, (int, float)):
                    if feature_name not in feature_values:
                        feature_values[feature_name] = []
                    feature_values[feature_name].append(value)
        
        # Calculate comprehensive statistics
        feature_stats = {}
        for feature_name, values in feature_values.items():
            if values:
                values_sorted = sorted(values)
                n = len(values)
                
                # Basic statistics
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / n
                
                # Variance and standard deviation
                variance = sum((x - mean_val) ** 2 for x in values) / n
                std_dev = variance ** 0.5
                
                # Percentiles for distribution analysis
                p25 = values_sorted[int(0.25 * n)] if n > 0 else 0
                p50 = values_sorted[int(0.5 * n)] if n > 0 else 0
                p75 = values_sorted[int(0.75 * n)] if n > 0 else 0
                
                # Range and interquartile range
                range_val = max_val - min_val
                iqr = p75 - p25
                
                # Outlier detection (using IQR method)
                outlier_threshold_low = p25 - 1.5 * iqr
                outlier_threshold_high = p75 + 1.5 * iqr
                outliers = [x for x in values if x < outlier_threshold_low or x > outlier_threshold_high]
                
                feature_stats[feature_name] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_dev,
                    'variance': variance,
                    'p25': p25,
                    'p50': p50,
                    'p75': p75,
                    'range': range_val,
                    'iqr': iqr,
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / n) * 100 if n > 0 else 0,
                    'count': n
                }
        
        return feature_stats
    
    def _check_missing_values(self, turns: List[GameTurn]) -> List[str]:
        """Check for missing values in turns.
        
        Args:
            turns: List of game turns
            
        Returns:
            List of features with missing values
        """
        if not turns:
            return []
        
        # Get all feature names
        all_features = set()
        for turn in turns:
            all_features.update(turn.features.keys())
        
        missing_features = []
        for feature_name in all_features:
            for turn in turns:
                if feature_name not in turn.features or turn.features[feature_name] is None:
                    missing_features.append(feature_name)
                    break
        
        return missing_features
    
    def _validate_feature_distributions(self, turns: List[GameTurn]) -> Dict[str, Any]:
        """Validate feature distributions and detect anomalies.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with distribution validation results
        """
        errors = []
        warnings = []
        stats = {}
        
        if not turns:
            return {'errors': errors, 'warnings': warnings, 'stats': stats}
        
        # Collect feature values
        feature_values = {}
        for turn in turns:
            for feature_name, value in turn.features.items():
                if isinstance(value, (int, float)):
                    if feature_name not in feature_values:
                        feature_values[feature_name] = []
                    feature_values[feature_name].append(value)
        
        # Analyze each feature's distribution
        distribution_stats = {}
        for feature_name, values in feature_values.items():
            if not values:
                continue
                
            n = len(values)
            values_sorted = sorted(values)
            
            # Basic statistics
            min_val = min(values)
            max_val = max(values)
            mean_val = sum(values) / n
            
            # Distribution analysis
            std_dev = (sum((x - mean_val) ** 2 for x in values) / n) ** 0.5
            
            # Percentiles
            p25 = values_sorted[int(0.25 * n)] if n > 0 else 0
            p50 = values_sorted[int(0.5 * n)] if n > 0 else 0
            p75 = values_sorted[int(0.75 * n)] if n > 0 else 0
            
            # Skewness (measure of asymmetry)
            skewness = 0
            if std_dev > 0:
                skewness = sum(((x - mean_val) / std_dev) ** 3 for x in values) / n
            
            # Kurtosis (measure of tail heaviness)
            kurtosis = 0
            if std_dev > 0:
                kurtosis = sum(((x - mean_val) / std_dev) ** 4 for x in values) / n - 3
            
            # Outlier detection
            iqr = p75 - p25
            outlier_threshold_low = p25 - 1.5 * iqr
            outlier_threshold_high = p75 + 1.5 * iqr
            outliers = [x for x in values if x < outlier_threshold_low or x > outlier_threshold_high]
            
            # Distribution quality checks
            feature_issues = []
            
            # Check for excessive outliers (>10% of data)
            outlier_percentage = (len(outliers) / n) * 100
            if outlier_percentage > 10:
                warnings.append(f"Feature {feature_name} has {outlier_percentage:.1f}% outliers")
            
            # Check for extreme skewness (>2 or <-2)
            if abs(skewness) > 2:
                warnings.append(f"Feature {feature_name} has extreme skewness: {skewness:.2f}")
            
            # Check for extreme kurtosis (>3 or <-1)
            if kurtosis > 3 or kurtosis < -1:
                warnings.append(f"Feature {feature_name} has extreme kurtosis: {kurtosis:.2f}")
            
            # Check for constant values (no variation)
            if std_dev == 0:
                warnings.append(f"Feature {feature_name} has no variation (constant value)")
            
            # Check for reasonable ranges based on feature type
            if feature_name.startswith('ring_') or feature_name.startswith('marker_'):
                if min_val < 0 or max_val > 1:
                    errors.append(f"Feature {feature_name} has values outside [0,1] range: [{min_val:.3f}, {max_val:.3f}]")
            
            # Check for NaN or infinite values
            import math
            if any(not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x) for x in values):
                errors.append(f"Feature {feature_name} contains NaN or invalid values")
            
            distribution_stats[feature_name] = {
                'count': n,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_dev,
                'p25': p25,
                'p50': p50,
                'p75': p75,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'outlier_count': len(outliers),
                'outlier_percentage': outlier_percentage,
                'iqr': iqr,
                'range': max_val - min_val
            }
        
        stats['distribution_analysis'] = distribution_stats
        stats['total_features'] = len(feature_values)
        stats['features_with_issues'] = len([f for f in distribution_stats.values() if f['outlier_percentage'] > 10 or abs(f['skewness']) > 2])
        
        return {'errors': errors, 'warnings': warnings, 'stats': stats}
    
    def _validate_feature_consistency(self, turns: List[GameTurn]) -> Dict[str, Any]:
        """Validate consistency of features across turns.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with consistency validation results
        """
        errors = []
        warnings = []
        stats = {}
        
        if not turns:
            return {'errors': errors, 'warnings': warnings, 'stats': stats}
        
        # Check feature consistency across turns
        first_turn_features = set(turns[0].features.keys())
        inconsistent_turns = []
        
        for i, turn in enumerate(turns[1:], 1):
            turn_features = set(turn.features.keys())
            if first_turn_features != turn_features:
                inconsistent_turns.append(i + 1)
        
        if inconsistent_turns:
            errors.append(f"Feature inconsistency in turns: {inconsistent_turns}")
        
        # Check for feature value consistency (no sudden jumps)
        feature_consistency_stats = {}
        for feature_name in first_turn_features:
            values = []
            for turn in turns:
                if feature_name in turn.features and isinstance(turn.features[feature_name], (int, float)):
                    values.append(turn.features[feature_name])
            
            if len(values) > 1:
                # Calculate differences between consecutive values
                differences = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
                max_diff = max(differences) if differences else 0
                mean_diff = sum(differences) / len(differences) if differences else 0
                
                # Check for sudden jumps (>1.5x mean difference)
                if differences:
                    jump_threshold = mean_diff * 1.5
                    jumps = [i for i, d in enumerate(differences) if d > jump_threshold]
                    if jumps:
                        warnings.append(f"Feature {feature_name} has sudden jumps in turns: {[j+2 for j in jumps]}")
                
                feature_consistency_stats[feature_name] = {
                    'max_consecutive_diff': max_diff,
                    'mean_consecutive_diff': mean_diff,
                    'total_values': len(values)
                }
        
        stats['feature_consistency'] = feature_consistency_stats
        stats['inconsistent_turns'] = len(inconsistent_turns)
        stats['total_turns_checked'] = len(turns)
        
        return {'errors': errors, 'warnings': warnings, 'stats': stats}
    
    def _validate_game_state_transitions(self, turns: List[GameTurn]) -> Dict[str, Any]:
        """Validate game state transitions for logical consistency.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with state transition validation results
        """
        errors = []
        warnings = []
        stats = {}
        
        if not turns:
            return {'errors': errors, 'warnings': warnings, 'stats': stats}
        
        # Track game state progression
        ring_count_white = 0
        ring_count_black = 0
        marker_count_white = 0
        marker_count_black = 0
        game_phase = "opening"  # opening, midgame, endgame
        
        transition_stats = {
            'ring_placements_white': 0,
            'ring_placements_black': 0,
            'ring_moves_white': 0,
            'ring_moves_black': 0,
            'invalid_transitions': [],
            'phase_transitions': [],
            'state_inconsistencies': []
        }
        
        for i, turn in enumerate(turns):
            if not turn.move:
                continue
                
            move_type = turn.move.get('type')
            player = turn.current_player
            
            # Validate ring placement phase
            if move_type == 'ring_placement':
                if player == 'white':
                    ring_count_white += 1
                    transition_stats['ring_placements_white'] += 1
                else:
                    ring_count_black += 1
                    transition_stats['ring_placements_black'] += 1
                
                # Check ring placement limits (max 5 rings per player)
                if ring_count_white > 5:
                    errors.append(f"Turn {i+1}: White has placed too many rings ({ring_count_white})")
                if ring_count_black > 5:
                    errors.append(f"Turn {i+1}: Black has placed too many rings ({ring_count_black})")
                
                # Check if ring placement is allowed (only in opening phase)
                if i > 10:  # After turn 10, ring placements should be rare
                    warnings.append(f"Turn {i+1}: Ring placement in late game phase")
                
            # Validate marker move phase
            elif move_type == 'MOVE_RING':
                if player == 'white':
                    marker_count_white += 1
                    transition_stats['ring_moves_white'] += 1
                else:
                    marker_count_black += 1
                    transition_stats['ring_moves_black'] += 1
                
                # Check if marker move is allowed (requires rings to be placed)
                if player == 'white' and ring_count_white == 0:
                    errors.append(f"Turn {i+1}: White attempted marker move without placing rings")
                elif player == 'black' and ring_count_black == 0:
                    errors.append(f"Turn {i+1}: Black attempted marker move without placing rings")
            
            # Track game phase transitions
            if i == 0:
                game_phase = "opening"
            elif i < 10:
                if game_phase != "opening":
                    transition_stats['phase_transitions'].append(f"Turn {i+1}: Unexpected phase transition to opening")
            elif i < 30:
                if game_phase == "opening":
                    game_phase = "midgame"
                    transition_stats['phase_transitions'].append(f"Turn {i+1}: Transition to midgame")
            else:
                if game_phase == "midgame":
                    game_phase = "endgame"
                    transition_stats['phase_transitions'].append(f"Turn {i+1}: Transition to endgame")
            
            # Validate move destination consistency
            if 'destination' in turn.move:
                destination = turn.move['destination']
                # Check for duplicate destinations (basic check)
                previous_destinations = [t.move.get('destination') for t in turns[:i] if t.move and 'destination' in t.move]
                if destination in previous_destinations:
                    warnings.append(f"Turn {i+1}: Duplicate move destination: {destination}")
            
            # Validate move source consistency (for marker moves)
            if move_type == 'marker_move' and 'source' in turn.move:
                source = turn.move['source']
                if not source:
                    errors.append(f"Turn {i+1}: Marker move missing source")
                elif source == turn.move.get('destination'):
                    errors.append(f"Turn {i+1}: Marker move source equals destination")
        
        # Final state validation
        if ring_count_white != ring_count_black and abs(ring_count_white - ring_count_black) > 1:
            warnings.append(f"Uneven ring placement: White={ring_count_white}, Black={ring_count_black}")
        
        # Check for reasonable game progression
        total_moves = len(turns)
        if total_moves > 0:
            ring_ratio = (ring_count_white + ring_count_black) / total_moves
            if ring_ratio > 0.5:  # More than 50% ring placements suggests early game
                if total_moves > 20:
                    warnings.append("High ring placement ratio in late game")
        
        stats['state_transitions'] = transition_stats
        stats['final_ring_counts'] = {'white': ring_count_white, 'black': ring_count_black}
        stats['final_marker_counts'] = {'white': marker_count_white, 'black': marker_count_black}
        stats['final_game_phase'] = game_phase
        
        return {'errors': errors, 'warnings': warnings, 'stats': stats}
    
    def _validate_board_state_consistency(self, turns: List[GameTurn]) -> Dict[str, Any]:
        """Validate board state consistency across turns.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with board state validation results
        """
        errors = []
        warnings = []
        stats = {}
        
        if not turns:
            return {'errors': errors, 'warnings': warnings, 'stats': stats}
        
        # Track board state
        occupied_positions = set()
        ring_positions = {'white': set(), 'black': set(), 1: set(), -1: set()}
        marker_positions = {'white': set(), 'black': set(), 1: set(), -1: set()}
        
        board_stats = {
            'total_positions_used': 0,
            'position_conflicts': [],
            'invalid_moves': [],
            'board_evolution': []
        }
        
        for i, turn in enumerate(turns):
            if not turn.move:
                continue
                
            move_type = turn.move.get('type')
            player = turn.current_player
            destination = turn.move.get('destination')
            source = turn.move.get('source')
            
            if not destination:
                continue
            
            # Check for position conflicts
            if destination in occupied_positions:
                errors.append(f"Turn {i+1}: Move to occupied position {destination}")
                board_stats['position_conflicts'].append(f"Turn {i+1}: {destination}")
            
            # Update board state based on move type
            if move_type == 'ring_placement':
                if destination not in occupied_positions:
                    occupied_positions.add(destination)
                    ring_positions[player].add(destination)
                    board_stats['total_positions_used'] += 1
                else:
                    errors.append(f"Turn {i+1}: Ring placement on occupied position {destination}")
            
            elif move_type == 'MOVE_RING':
                if source and source in ring_positions[player]:
                    # Remove ring from source
                    ring_positions[player].discard(source)
                    occupied_positions.discard(source)
                    
                    # Place marker at destination
                    if destination not in occupied_positions:
                        occupied_positions.add(destination)
                        marker_positions[player].add(destination)
                        board_stats['total_positions_used'] += 1
                    else:
                        errors.append(f"Turn {i+1}: Marker move to occupied position {destination}")
                else:
                    if source:
                        errors.append(f"Turn {i+1}: Marker move from invalid source {source}")
                    else:
                        errors.append(f"Turn {i+1}: Marker move missing source")
            
            # Track board evolution
            board_stats['board_evolution'].append({
                'turn': i + 1,
                'occupied_count': len(occupied_positions),
                'white_rings': len(ring_positions['white']),
                'black_rings': len(ring_positions['black']),
                'white_markers': len(marker_positions['white']),
                'black_markers': len(marker_positions['black'])
            })
        
        # Final board state validation
        total_rings = len(ring_positions['white']) + len(ring_positions['black'])
        total_markers = len(marker_positions['white']) + len(marker_positions['black'])
        
        if total_rings > 10:  # Max 5 rings per player
            errors.append(f"Too many rings on board: {total_rings}")
        
        if total_markers > 50:  # Reasonable upper bound
            warnings.append(f"High marker count: {total_markers}")
        
        stats['board_state'] = board_stats
        stats['final_positions'] = {
            'total_occupied': len(occupied_positions),
            'white_rings': len(ring_positions['white']),
            'black_rings': len(ring_positions['black']),
            'white_markers': len(marker_positions['white']),
            'black_markers': len(marker_positions['black'])
        }
        
        return {'errors': errors, 'warnings': warnings, 'stats': stats}
    
    def _validate_move_legality(self, turns: List[GameTurn]) -> Dict[str, Any]:
        """Validate move legality based on game rules.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with move legality validation results
        """
        errors = []
        warnings = []
        stats = {}
        
        if not turns:
            return {'errors': errors, 'warnings': warnings, 'stats': stats}
        
        legality_stats = {
            'illegal_moves': [],
            'suspicious_moves': [],
            'move_patterns': {},
            'rule_violations': []
        }
        
        for i, turn in enumerate(turns):
            if not turn.move:
                continue
                
            move_type = turn.move.get('type')
            player = turn.current_player
            destination = turn.move.get('destination')
            source = turn.move.get('source')
            
            # Basic move validation - check for required fields based on move type
            if move_type == 'PLACE_RING':
                # Ring placement uses 'source' for the placement position
                if not source:
                    errors.append(f"Turn {i+1}: Ring placement missing source position")
                    legality_stats['illegal_moves'].append(f"Turn {i+1}: No source for ring placement")
                    continue
            elif move_type == 'MOVE_RING':
                # Ring movement requires both source and destination
                if not source:
                    errors.append(f"Turn {i+1}: Ring movement missing source position")
                    legality_stats['illegal_moves'].append(f"Turn {i+1}: No source for ring movement")
                    continue
                if not destination:
                    errors.append(f"Turn {i+1}: Ring movement missing destination position")
                    legality_stats['illegal_moves'].append(f"Turn {i+1}: No destination for ring movement")
                    continue
            elif move_type == 'REMOVE_MARKERS':
                # Marker removal uses 'markers' field
                markers = turn.move.get('markers')
                if not markers:
                    errors.append(f"Turn {i+1}: Marker removal missing markers")
                    legality_stats['illegal_moves'].append(f"Turn {i+1}: No markers for removal")
                    continue
            elif move_type == 'REMOVE_RING':
                # Ring removal uses 'source' for the ring position
                if not source:
                    errors.append(f"Turn {i+1}: Ring removal missing source position")
                    legality_stats['illegal_moves'].append(f"Turn {i+1}: No source for ring removal")
                    continue
            else:
                # Unknown move type
                errors.append(f"Turn {i+1}: Unknown move type: {move_type}")
                legality_stats['illegal_moves'].append(f"Turn {i+1}: Unknown move type")
                continue
            
            # Validate destination format (basic coordinate check) - only if destination exists
            if destination and isinstance(destination, str) and len(destination) < 2:
                errors.append(f"Turn {i+1}: Invalid destination format: {destination}")
                legality_stats['illegal_moves'].append(f"Turn {i+1}: Bad destination format")
            
            # Validate source format (basic coordinate check) - only if source exists
            if source and isinstance(source, str) and len(source) < 2:
                errors.append(f"Turn {i+1}: Invalid source format: {source}")
                legality_stats['illegal_moves'].append(f"Turn {i+1}: Bad source format")
            
            # Validate move type specific rules
            if move_type == 'PLACE_RING':
                # Ring placement validation
                if i > 0:  # Not first move
                    # Check if player has already placed maximum rings
                    player_ring_count = sum(1 for t in turns[:i] 
                                          if t.move and t.move.get('type') == 'PLACE_RING' 
                                          and t.current_player == player)
                    if player_ring_count >= 5:
                        errors.append(f"Turn {i+1}: {player} attempting to place more than 5 rings")
                        legality_stats['rule_violations'].append(f"Turn {i+1}: Too many rings")
            
            elif move_type == 'MOVE_RING':
                # Marker move validation
                if not source:
                    errors.append(f"Turn {i+1}: Marker move missing source")
                    legality_stats['illegal_moves'].append(f"Turn {i+1}: No source")
                else:
                    # Check if source is valid (has a ring)
                    source_valid = False
                    for t in turns[:i]:
                        if (t.move and t.move.get('type') == 'ring_placement' 
                            and t.move.get('destination') == source):
                            source_valid = True
                            break
                    
                    if not source_valid:
                        errors.append(f"Turn {i+1}: Marker move from invalid source {source}")
                        legality_stats['illegal_moves'].append(f"Turn {i+1}: Invalid source")
            
            # Track move patterns
            if move_type not in legality_stats['move_patterns']:
                legality_stats['move_patterns'][move_type] = 0
            legality_stats['move_patterns'][move_type] += 1
            
            # Check for suspicious patterns
            if i > 2:
                # Check for repeated moves
                recent_moves = [t.move.get('destination') for t in turns[i-2:i] if t.move]
                if destination in recent_moves:
                    warnings.append(f"Turn {i+1}: Repeated move to {destination}")
                    legality_stats['suspicious_moves'].append(f"Turn {i+1}: Repeated destination")
        
        # Overall game legality checks
        total_moves = len(turns)
        if total_moves > 0:
            ring_placements = legality_stats['move_patterns'].get('PLACE_RING', 0)
            ring_moves = legality_stats['move_patterns'].get('MOVE_RING', 0)
            
            # Check for reasonable move distribution
            if ring_placements > 0 and ring_moves > 0:
                ratio = ring_moves / ring_placements
                if ratio < 0.1:  # Very few marker moves relative to ring placements
                    warnings.append("Unusually low marker move ratio")
                elif ratio > 10:  # Very many marker moves relative to ring placements
                    warnings.append("Unusually high marker move ratio")
        
        stats['move_legality'] = legality_stats
        
        return {'errors': errors, 'warnings': warnings, 'stats': stats}
    
    def _check_completeness(self, game_record: GameRecord) -> Dict[str, Any]:
        """Check completeness of game record.
        
        Args:
            game_record: Game record to check
            
        Returns:
            Dictionary with completeness check results
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check required fields
        required_fields = ['game_id', 'start_time', 'end_time', 'duration', 'total_turns']
        missing_fields = []
        for field in required_fields:
            if not hasattr(game_record, field) or getattr(game_record, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check metadata completeness
        if not game_record.metadata:
            warnings.append("No metadata provided")
        else:
            expected_metadata = ['feature_count', 'total_moves', 'final_phase']
            missing_metadata = [key for key in expected_metadata if key not in game_record.metadata]
            if missing_metadata:
                warnings.append(f"Missing expected metadata: {missing_metadata}")
        
        # Check turn completeness
        if game_record.turns:
            # Check if all turns have required fields
            incomplete_turns = []
            for i, turn in enumerate(game_record.turns):
                turn_issues = []
                if not turn.turn_number:
                    turn_issues.append("missing turn_number")
                if not turn.current_player:
                    turn_issues.append("missing current_player")
                if not turn.move:
                    turn_issues.append("missing move")
                if not turn.features:
                    turn_issues.append("missing features")
                if not turn.timestamp:
                    turn_issues.append("missing timestamp")
                
                if turn_issues:
                    incomplete_turns.append(f"Turn {i+1}: {', '.join(turn_issues)}")
            
            if incomplete_turns:
                errors.append(f"Incomplete turns: {incomplete_turns}")
            
            # Check if turn count matches total_turns
            if len(game_record.turns) != game_record.total_turns:
                errors.append(f"Turn count mismatch: {len(game_record.turns)} turns but total_turns={game_record.total_turns}")
        
        # Check duration consistency
        if game_record.start_time and game_record.end_time and game_record.duration:
            calculated_duration = game_record.end_time - game_record.start_time
            duration_diff = abs(calculated_duration - game_record.duration)
            if duration_diff > 1.0:  # Allow 1 second tolerance
                warnings.append(f"Duration inconsistency: calculated={calculated_duration:.2f}s, recorded={game_record.duration:.2f}s")
        
        stats = {
            'has_metadata': bool(game_record.metadata),
            'turn_count': len(game_record.turns) if game_record.turns else 0,
            'expected_turns': game_record.total_turns,
            'missing_fields': missing_fields,
            'duration_consistent': duration_diff <= 1.0 if 'duration_diff' in locals() else True
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _validate_move_sequence(self, turns: List[GameTurn]) -> Dict[str, Any]:
        """Validate move sequence for logical consistency.
        
        Args:
            turns: List of game turns
            
        Returns:
            Dictionary with move validation results
        """
        errors = []
        warnings = []
        stats = {}
        
        if not turns:
            return {'errors': errors, 'warnings': warnings, 'stats': stats}
        
        # Check player alternation
        player_sequence = [turn.current_player for turn in turns]
        expected_players = ['white', 'black'] * ((len(turns) + 1) // 2)
        expected_players = expected_players[:len(turns)]
        
        if player_sequence != expected_players:
            errors.append("Player sequence is not alternating correctly")
        
        # Check move types
        move_types = [turn.move.get('type') for turn in turns if turn.move]
        move_type_counts = {}
        for move_type in move_types:
            move_type_counts[move_type] = move_type_counts.get(move_type, 0) + 1
        
        # Check for reasonable move distribution
        if len(turns) > 10:  # Only check for longer games
            ring_placements = move_type_counts.get('PLACE_RING', 0)
            ring_moves = move_type_counts.get('MOVE_RING', 0)
            
            if ring_placements > 0 and ring_moves > 0:
                ratio = ring_moves / ring_placements
                if ratio < 0.5:  # Expect at least some marker moves
                    warnings.append(f"Low marker move ratio: {ratio:.2f}")
        
        # Check move destinations for duplicates (basic check)
        destinations = []
        for turn in turns:
            if turn.move and 'destination' in turn.move:
                destinations.append(turn.move['destination'])
        
        duplicate_destinations = len(destinations) - len(set(destinations))
        if duplicate_destinations > 0:
            warnings.append(f"Found {duplicate_destinations} duplicate move destinations")
        
        stats = {
            'total_moves': len(turns),
            'move_types': move_type_counts,
            'player_alternation_correct': player_sequence == expected_players,
            'duplicate_destinations': duplicate_destinations,
            'ring_placements': move_type_counts.get('PLACE_RING', 0),
            'ring_moves': move_type_counts.get('MOVE_RING', 0)
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _check_outcome_completeness(self, game_record: GameRecord) -> Dict[str, Any]:
        """Check completeness of game outcome.
        
        Args:
            game_record: Game record to check
            
        Returns:
            Dictionary with outcome completeness results
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check if game has a winner
        if not game_record.winner:
            warnings.append("Game has no winner specified")
        
        # Check if final score is provided
        if not game_record.final_score:
            warnings.append("Game has no final score")
        else:
            # Check score completeness
            white_score = game_record.final_score.get('white', 0)
            black_score = game_record.final_score.get('black', 0)
            
            if white_score == 0 and black_score == 0:
                warnings.append("Both players have zero score")
            
            # Check if winner matches score
            if game_record.winner:
                if game_record.winner == '1' and white_score <= black_score:
                    warnings.append("White marked as winner but score doesn't reflect victory")
                elif game_record.winner == '-1' and black_score <= white_score:
                    warnings.append("Black marked as winner but score doesn't reflect victory")
        
        # Check if game duration is reasonable
        if game_record.duration:
            if game_record.duration < 1.0:
                warnings.append("Game duration seems too short (< 1 second)")
            elif game_record.duration > 3600:  # 1 hour
                warnings.append("Game duration seems too long (> 1 hour)")
        
        # Check if turn count is reasonable
        if game_record.total_turns > 0:
            if game_record.total_turns < 2:
                errors.append("Game has too few turns (< 2)")
            elif game_record.total_turns > 200:  # Reasonable upper bound
                warnings.append("Game has unusually many turns (> 200)")
        
        stats = {
            'has_winner': bool(game_record.winner),
            'has_final_score': bool(game_record.final_score),
            'white_score': game_record.final_score.get('white', 0) if game_record.final_score else 0,
            'black_score': game_record.final_score.get('black', 0) if game_record.final_score else 0,
            'duration_seconds': game_record.duration,
            'total_turns': game_record.total_turns,
            'outcome_consistent': self._check_outcome_consistency(game_record)
        }
        
        return {
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }
    
    def _check_outcome_consistency(self, game_record: GameRecord) -> bool:
        """Check if game outcome is internally consistent.
        
        Args:
            game_record: Game record to check
            
        Returns:
            True if outcome is consistent, False otherwise
        """
        if not game_record.winner or not game_record.final_score:
            return False
        
        white_score = game_record.final_score.get('white', 0)
        black_score = game_record.final_score.get('black', 0)
        
        if game_record.winner == '1':  # White won
            return white_score > black_score
        elif game_record.winner == '-1':  # Black won
            return black_score > white_score
        elif game_record.winner == '0':  # Draw
            return white_score == black_score
        
        return False


class SelfPlayDataManager:
    """Main interface for self-play data storage and validation."""
    
    def __init__(self, config: StorageConfig = None):
        """Initialize data manager.
        
        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.storage = ParquetDataStorage(self.config)
        self.validator = DataValidator(self.config)
        
        logger.info("Initialized SelfPlayDataManager")
    
    def store_game(self, game_record: GameRecord) -> ValidationResult:
        """Store a game record with validation.
        
        Args:
            game_record: Game record to store
            
        Returns:
            Validation result
        """
        # Validate before storing
        if self.config.validation_enabled:
            logger.info(f"Validating game {game_record.game_id}")
            validation_result = self.validator.validate_game_record(game_record)
            if not validation_result.valid:
                logger.error(f"Game {game_record.game_id} failed validation: {validation_result.errors}")
                return validation_result
            logger.info(f"Game {game_record.game_id} validation passed")
        
        # Store the game
        logger.info(f"Storing game {game_record.game_id} in parquet storage")
        self.storage.store_game_record(game_record)
        
        logger.info(f"Stored game {game_record.game_id}")
        return ValidationResult(valid=True, errors=[], warnings=[], stats={})
    
    def flush_storage(self) -> None:
        """Flush any pending data to storage."""
        self.storage.flush()
    
    def load_all_games(self) -> pd.DataFrame:
        """Load all stored games.
        
        Returns:
            DataFrame with all game data
        """
        return self.storage.load_games()
    
    def validate_stored_data(self) -> ValidationResult:
        """Validate all stored data.
        
        Returns:
            Validation result
        """
        df = self.load_all_games()
        return self.validator.validate_dataframe(df)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information.
        
        Returns:
            Dictionary with storage information
        """
        return self.storage.get_storage_stats()
    
    def export_to_csv(self, output_file: Union[str, Path]) -> None:
        """Export all data to CSV.
        
        Args:
            output_file: Path to output CSV file
        """
        df = self.load_all_games()
        df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(df)} rows to {output_file}")
    
    def cleanup_old_batches(self, keep_last_n: int = 10) -> None:
        """Clean up old batch files, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent batches to keep
        """
        parquet_files = sorted(self.storage.parquet_dir.glob("*.parquet"))
        
        if len(parquet_files) > keep_last_n:
            files_to_remove = parquet_files[:-keep_last_n]
            for filepath in files_to_remove:
                try:
                    filepath.unlink()
                    logger.info(f"Removed old batch file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to remove {filepath}: {e}")
