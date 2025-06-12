"""
Migration tool for importing historical experiment data into the new tracking system.

This module provides utilities to scan existing experiment directories,
parse various historical data formats, and import them into ExperimentTracker
while maintaining data integrity and providing detailed migration reports.
"""

import json
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import traceback

try:
    from .experiment_tracker import ExperimentTracker
    from .config_serializer import ConfigurationSerializer
except ImportError:
    from experiment_tracker import ExperimentTracker
    from config_serializer import ConfigurationSerializer

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Container for migration operation results."""
    experiment_name: str
    source_path: str
    success: bool
    experiment_id: Optional[int] = None
    metrics_imported: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class MigrationReport:
    """Comprehensive migration report."""
    total_experiments: int
    successful_imports: int
    failed_imports: int
    total_metrics_imported: int
    results: List[MigrationResult]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class HistoricalDataParser:
    """Parser for different historical data formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HistoricalDataParser")
    
    def parse_final_summary_metrics(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse final_summary_metrics.json file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract experiment metadata
            experiment_data = {
                'config': data.get('config', {}),
                'final_metrics': data.get('final_metrics', {}),
                'completed_timestamp': data.get('completed_timestamp'),
                'experiment_id': data.get('experiment_id')  # May be None for historical data
            }
            
            return experiment_data
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def parse_training_log(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse training log file to extract metrics over time."""
        metrics = []
        warnings = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract iteration-level metrics using regex patterns
            patterns = {
                'iteration': r'--- Starting Iteration (\d+)/\d+',
                'iteration_complete': r'--- Iteration (\d+) completed in ([\d.]+)s ---',
                'policy_loss': r'Policy Loss=([\d.]+|N/A)',
                'value_loss': r'Value Loss=([\d.]+|N/A)',
                'elo': r'Elo=([\d.]+|N/A)',
                'tournament_rating': r'Tournament.*?rating.*?([\d.]+)',
                'training_time': r'training_time.*?([\d.]+)',
                'game_length': r'avg_game_length.*?([\d.]+)',
            }
            
            # Process log line by line to maintain temporal ordering
            lines = content.split('\n')
            current_iteration = None
            iteration_metrics = {}
            
            for line in lines:
                # Check for iteration start
                iteration_match = re.search(patterns['iteration'], line)
                if iteration_match:
                    current_iteration = int(iteration_match.group(1))
                    iteration_metrics = {'iteration': current_iteration}
                    continue
                
                # Extract metrics for current iteration
                if current_iteration is not None:
                    for metric_name, pattern in patterns.items():
                        if metric_name in ['iteration']:
                            continue
                            
                        match = re.search(pattern, line)
                        if match:
                            value_str = match.group(1) if metric_name != 'iteration_complete' else match.group(2)
                            
                            # Handle special cases
                            if value_str == 'N/A':
                                continue
                            
                            try:
                                value = float(value_str)
                                iteration_metrics[metric_name] = value
                            except ValueError:
                                warnings.append(f"Could not parse {metric_name} value: {value_str}")
                
                # Check for iteration completion to finalize metrics
                completion_match = re.search(patterns['iteration_complete'], line)
                if completion_match and current_iteration is not None:
                    iteration_num = int(completion_match.group(1))
                    iteration_time = float(completion_match.group(2))
                    
                    if iteration_num == current_iteration:
                        iteration_metrics['iteration_time'] = iteration_time
                        
                        # Add timestamp (approximate from log structure)
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            try:
                                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                iteration_metrics['timestamp'] = timestamp.isoformat()
                            except ValueError:
                                pass
                        
                        if len(iteration_metrics) > 1:  # More than just iteration number
                            metrics.append(iteration_metrics.copy())
                        
                        current_iteration = None
                        iteration_metrics = {}
            
        except Exception as e:
            warnings.append(f"Error parsing training log {file_path}: {e}")
        
        return metrics, warnings
    
    def parse_tournament_history(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse tournament_history.json file."""
        metrics = []
        warnings = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract tournament metrics
            for entry in data:
                if isinstance(entry, dict):
                    tournament_metric = {
                        'metric_name': 'tournament_result',
                        'value': entry.get('rating', 0),
                        'metadata': {
                            'tournament_id': entry.get('tournament_id'),
                            'win_rate': entry.get('win_rate'),
                            'games_played': entry.get('games_played'),
                            'opponent': entry.get('opponent')
                        }
                    }
                    
                    if 'timestamp' in entry:
                        tournament_metric['timestamp'] = entry['timestamp']
                    
                    metrics.append(tournament_metric)
            
        except Exception as e:
            warnings.append(f"Error parsing tournament history {file_path}: {e}")
        
        return metrics, warnings
    
    def parse_elo_ratings(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse elo_ratings.json file."""
        metrics = []
        warnings = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract current ratings
            current_ratings = data.get('current_ratings', {})
            for model_name, rating in current_ratings.items():
                elo_metric = {
                    'metric_name': 'elo_rating',
                    'value': float(rating),
                    'metadata': {'model_name': model_name}
                }
                metrics.append(elo_metric)
            
            # Extract rating history if available
            history = data.get('rating_history', [])
            for entry in history:
                if isinstance(entry, dict):
                    history_metric = {
                        'metric_name': 'elo_history',
                        'value': entry.get('rating', 0),
                        'metadata': {
                            'model_name': entry.get('model_name'),
                            'change': entry.get('change', 0)
                        }
                    }
                    
                    if 'timestamp' in entry:
                        history_metric['timestamp'] = entry['timestamp']
                    
                    metrics.append(history_metric)
            
        except Exception as e:
            warnings.append(f"Error parsing ELO ratings {file_path}: {e}")
        
        return metrics, warnings


class MigrationTool:
    """Main migration tool for importing historical experiment data."""
    
    def __init__(self, tracker: Optional[ExperimentTracker] = None):
        self.logger = logging.getLogger(f"{__name__}.MigrationTool")
        self.tracker = tracker or ExperimentTracker.get_instance()
        self.parser = HistoricalDataParser()
        
        # Migration statistics
        self.stats = {
            'experiments_scanned': 0,
            'experiments_imported': 0,
            'metrics_imported': 0,
            'errors': 0,
            'warnings': 0
        }
    
    def scan_experiment_directory(self, results_dir: Path) -> List[Path]:
        """Scan for experiment directories."""
        experiment_dirs = []
        
        if not results_dir.exists():
            self.logger.warning(f"Results directory does not exist: {results_dir}")
            return experiment_dirs
        
        for item in results_dir.iterdir():
            if item.is_dir() and self._is_experiment_directory(item):
                experiment_dirs.append(item)
        
        return experiment_dirs
    
    def _is_experiment_directory(self, path: Path) -> bool:
        """Check if a directory contains experiment data."""
        indicators = [
            'final_summary_metrics.json',
            'run_*.log',
            'tournament_history.json'
        ]
        
        for indicator in indicators:
            if list(path.glob(indicator)):
                return True
        return False
    
    def import_experiment(self, experiment_dir: Path) -> MigrationResult:
        """Import a single experiment from a directory."""
        experiment_name = f"Migrated_{experiment_dir.name}"
        
        result = MigrationResult(
            experiment_name=experiment_name,
            source_path=str(experiment_dir),
            success=False
        )
        
        try:
            # Parse summary metrics if available
            summary_file = experiment_dir / 'final_summary_metrics.json'
            experiment_data = {}
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    experiment_data = json.load(f)
            
            # Create experiment in tracking system
            experiment_config = experiment_data.get('config', {})
            
            experiment_id = self.tracker.create_experiment(
                name=experiment_name,
                description=f"Migrated from {experiment_dir}",
                config=experiment_config,
                tags=['migrated', 'historical']
            )
            
            result.experiment_id = experiment_id
            result.success = True
            
            # Import final metrics if available
            final_metrics = experiment_data.get('final_metrics', {})
            metrics_count = 0
            
            for metric_name, values in final_metrics.items():
                if isinstance(values, list):
                    for i, value in enumerate(values):
                        if value is not None:
                            try:
                                self.tracker.log_metric(experiment_id, metric_name, float(value), iteration=i+1)
                                metrics_count += 1
                            except Exception as e:
                                result.warnings.append(f"Failed to import metric {metric_name}[{i}]: {e}")
            
            result.metrics_imported = metrics_count
            self.logger.info(f"Successfully imported experiment {experiment_name} with {metrics_count} metrics")
            
        except Exception as e:
            result.errors.append(f"Failed to import experiment: {e}")
            self.logger.error(f"Failed to import experiment from {experiment_dir}: {e}")
        
        return result
    
    def migrate_directory(self, results_dir: Path) -> List[MigrationResult]:
        """Migrate all experiments from a directory."""
        self.logger.info(f"Starting migration from {results_dir}")
        
        experiment_dirs = self.scan_experiment_directory(results_dir)
        results = []
        
        for experiment_dir in experiment_dirs:
            self.logger.info(f"Processing experiment directory: {experiment_dir}")
            result = self.import_experiment(experiment_dir)
            results.append(result)
        
        return results


def main():
    """CLI interface for the migration tool."""
    parser = argparse.ArgumentParser(description="Migrate historical experiment data")
    parser.add_argument('results_dir', type=Path, help='Directory containing historical results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        migration_tool = MigrationTool()
        results = migration_tool.migrate_directory(args.results_dir)
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        total_metrics = sum(r.metrics_imported for r in results)
        
        print(f"\nMigration Summary:")
        print(f"Processed: {len(results)} experiments")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Total metrics imported: {total_metrics}")
        
        return 0 if len(results) == successful else 1
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 