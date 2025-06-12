"""Experiment comparison and analysis tools for YINSH training experiments."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

# Import existing experiment tracking components
from experiments.tracker import MetricsTracker, ExperimentMetrics
from experiments.config import RESULTS_DIR

# Import new tracking system (if available)
try:
    from yinsh_ml.tracking.experiment_tracker import ExperimentTracker
    DATABASE_TRACKING_AVAILABLE = True
except ImportError:
    DATABASE_TRACKING_AVAILABLE = False


@dataclass
class ComparisonMetrics:
    """Container for comparison statistics between experiments."""
    mean: float
    median: float
    min: float
    max: float
    std: float
    count: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExperimentComparison:
    """Container for experiment comparison results."""
    experiment_ids: List[Union[str, int]]
    experiment_names: Dict[Union[str, int], str]
    metric_comparisons: Dict[str, Dict[Union[str, int], ComparisonMetrics]]
    relative_performance: Dict[str, Dict[Union[str, int], float]]
    comparison_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'experiment_ids': self.experiment_ids,
            'experiment_names': self.experiment_names,
            'metric_comparisons': {
                metric: {str(exp_id): comp.to_dict() for exp_id, comp in comparisons.items()}
                for metric, comparisons in self.metric_comparisons.items()
            },
            'relative_performance': {
                metric: {str(exp_id): perf for exp_id, perf in perfs.items()}
                for metric, perfs in self.relative_performance.items()
            },
            'comparison_timestamp': self.comparison_timestamp
        }


class ExperimentComparator:
    """Core class for comparing and analyzing multiple experiments."""
    
    def __init__(self, cache_size: int = 100):
        """
        Initialize the ExperimentComparator.
        
        Args:
            cache_size: Maximum number of experiments to cache in memory
        """
        self.logger = logging.getLogger("ExperimentComparator")
        self.cache_size = cache_size
        
        # Initialize data sources
        self.metrics_tracker = MetricsTracker()
        self.experiment_tracker = None
        if DATABASE_TRACKING_AVAILABLE:
            try:
                self.experiment_tracker = ExperimentTracker()
            except Exception as e:
                self.logger.warning(f"Failed to initialize database tracker: {e}")
        
        # Cache for loaded experiment data
        self._experiment_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
    def load_experiments(self, experiment_identifiers: List[Union[str, int]], 
                        source: str = "auto") -> Dict[Union[str, int], Dict[str, Any]]:
        """
        Load multiple experiments from various sources.
        
        Args:
            experiment_identifiers: List of experiment IDs (strings for files, ints for database)
            source: Data source - "files", "database", or "auto" for automatic detection
            
        Returns:
            Dictionary mapping experiment IDs to experiment data
        """
        loaded_experiments = {}
        
        for exp_id in experiment_identifiers:
            experiment_data = self._load_single_experiment(exp_id, source)
            if experiment_data:
                loaded_experiments[exp_id] = experiment_data
            else:
                self.logger.warning(f"Failed to load experiment {exp_id}")
        
        return loaded_experiments
    
    def _load_single_experiment(self, exp_id: Union[str, int], 
                               source: str = "auto") -> Optional[Dict[str, Any]]:
        """Load a single experiment from cache or data source."""
        cache_key = f"{source}_{exp_id}"
        
        # Check cache first
        if cache_key in self._experiment_cache:
            # Check if cache is still fresh (within 5 minutes)
            if time.time() - self._cache_timestamps[cache_key] < 300:
                return self._experiment_cache[cache_key]
        
        # Load from appropriate source
        experiment_data = None
        
        if source == "auto":
            # Try database first if available, then files
            if isinstance(exp_id, int) and self.experiment_tracker:
                experiment_data = self._load_from_database(exp_id)
            if not experiment_data:
                experiment_data = self._load_from_files(str(exp_id))
        elif source == "database" and self.experiment_tracker:
            experiment_data = self._load_from_database(int(exp_id))
        elif source == "files":
            experiment_data = self._load_from_files(str(exp_id))
        
        # Cache the loaded data
        if experiment_data:
            self._cache_experiment(cache_key, experiment_data)
        
        return experiment_data
    
    def _load_from_database(self, exp_id: int) -> Optional[Dict[str, Any]]:
        """Load experiment from database tracking system."""
        try:
            experiment = self.experiment_tracker.get_experiment(exp_id)
            if not experiment:
                return None
                
            # Get metrics history
            metrics_data = defaultdict(list)
            
            # Get all metric names for this experiment
            all_metrics = self.experiment_tracker.get_experiment_metrics(exp_id)
            
            for metric_entry in all_metrics:
                metric_name = metric_entry['metric_name']
                metric_value = metric_entry['metric_value']
                metrics_data[metric_name].append(metric_value)
            
            # Convert to ExperimentMetrics format for consistency
            metrics = ExperimentMetrics(
                policy_losses=metrics_data.get('policy_loss', []),
                value_losses=metrics_data.get('value_loss', []),
                elo_changes=metrics_data.get('tournament_rating', []),
                game_lengths=metrics_data.get('game_length', []),
                timestamps=metrics_data.get('timestamp', []),
                move_entropies=metrics_data.get('move_entropy', None),
                win_rates=metrics_data.get('win_rate', None),
                search_times=metrics_data.get('search_time', None)
            )
            
            return {
                'id': exp_id,
                'name': experiment['name'],
                'description': experiment.get('description', ''),
                'status': experiment['status'],
                'config': experiment.get('config', {}),
                'metrics': metrics,
                'source': 'database'
            }
            
        except Exception as e:
            self.logger.error(f"Error loading experiment {exp_id} from database: {e}")
            return None
    
    def _load_from_files(self, exp_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment from legacy JSON files."""
        try:
            # Search in all results subdirectories
            for results_subdir in RESULTS_DIR.iterdir():
                if results_subdir.is_dir():
                    result_file = results_subdir / f"{exp_name}.json"
                    if result_file.exists():
                        with open(result_file) as f:
                            data = json.load(f)
                        
                        # Convert to standard format
                        metrics_data = data.get("final_metrics", data.get("metrics", {}))
                        
                        # Create ExperimentMetrics from data
                        metrics = ExperimentMetrics(
                            policy_losses=metrics_data.get('policy_loss', []),
                            value_losses=metrics_data.get('value_loss', []),
                            elo_changes=metrics_data.get('tournament_rating', []),
                            game_lengths=metrics_data.get('game_length', []),
                            timestamps=metrics_data.get('timestamp', []),
                            move_entropies=metrics_data.get('move_entropy', None),
                            win_rates=metrics_data.get('win_rate', None),
                            search_times=metrics_data.get('search_time', None)
                        )
                        
                        return {
                            'id': exp_name,
                            'name': exp_name,
                            'description': f"Loaded from {result_file}",
                            'status': 'completed',
                            'config': data.get('config', {}),
                            'metrics': metrics,
                            'source': 'files'
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading experiment {exp_name} from files: {e}")
            return None
    
    def _cache_experiment(self, cache_key: str, experiment_data: Dict[str, Any]):
        """Cache experiment data with LRU eviction."""
        # Implement simple LRU cache
        if len(self._experiment_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=lambda k: self._cache_timestamps[k])
            del self._experiment_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._experiment_cache[cache_key] = experiment_data
        self._cache_timestamps[cache_key] = time.time()
    
    def compare_experiments(self, experiment_identifiers: Union[List[Union[str, int]], Dict[Union[str, int], Dict[str, Any]]],
                          metrics: Optional[List[str]] = None,
                          source: str = "auto") -> ExperimentComparison:
        """
        Compare multiple experiments side-by-side.
        
        Args:
            experiment_identifiers: List of experiment IDs to compare, or dict of experiment data
            metrics: List of specific metrics to compare (if None, compares all available)
            source: Data source - "files", "database", or "auto"
            
        Returns:
            ExperimentComparison object with detailed comparison results
        """
        # Load all experiments or use provided data
        if isinstance(experiment_identifiers, dict):
            # Direct experiment data provided
            experiments = experiment_identifiers
            experiment_names = {exp_id: f"Experiment {exp_id}" for exp_id in experiments.keys()}
        else:
            # Load experiments from identifiers
            experiments = self.load_experiments(experiment_identifiers, source)
            experiment_names = {exp_id: exp_data['name'] for exp_id, exp_data in experiments.items()}
        
        if not experiments:
            raise ValueError("No experiments could be loaded for comparison")
        
        # Determine metrics to compare
        if metrics is None:
            metrics = self._get_common_metrics(experiments)
        
        # Calculate comparison metrics
        metric_comparisons = {}
        relative_performance = {}
        
        for metric_name in metrics:
            metric_comparisons[metric_name] = {}
            relative_performance[metric_name] = {}
            
            # Calculate statistics for each experiment
            all_values = []
            for exp_id, exp_data in experiments.items():
                values = self._extract_metric_values(exp_data['metrics'], metric_name)
                if values:
                    comparison_metrics = ComparisonMetrics(
                        mean=float(np.mean(values)),
                        median=float(np.median(values)),
                        min=float(np.min(values)),
                        max=float(np.max(values)),
                        std=float(np.std(values)),
                        count=len(values)
                    )
                    metric_comparisons[metric_name][exp_id] = comparison_metrics
                    all_values.extend(values)
            
            # Calculate relative performance (normalized to mean)
            if all_values:
                global_mean = np.mean(all_values)
                for exp_id in metric_comparisons[metric_name]:
                    exp_mean = metric_comparisons[metric_name][exp_id].mean
                    relative_performance[metric_name][exp_id] = (exp_mean - global_mean) / global_mean if global_mean != 0 else 0.0
        
        return ExperimentComparison(
            experiment_ids=list(experiment_identifiers),
            experiment_names=experiment_names,
            metric_comparisons=metric_comparisons,
            relative_performance=relative_performance,
            comparison_timestamp=pd.Timestamp.now().isoformat()
        )
    
    def _get_common_metrics(self, experiments: Dict[Union[str, int], Dict[str, Any]]) -> List[str]:
        """Get list of metrics common to all experiments."""
        all_metric_sets = []
        
        for exp_data in experiments.values():
            metrics_obj = exp_data['metrics']
            available_metrics = []
            
            # Check which metrics have data
            if metrics_obj.policy_losses:
                available_metrics.append('policy_loss')
            if metrics_obj.value_losses:
                available_metrics.append('value_loss')
            if metrics_obj.elo_changes:
                available_metrics.append('elo_changes')
            if metrics_obj.game_lengths:
                available_metrics.append('game_lengths')
            if metrics_obj.move_entropies:
                available_metrics.append('move_entropies')
            if metrics_obj.win_rates:
                available_metrics.append('win_rates')
            if metrics_obj.search_times:
                available_metrics.append('search_times')
            
            all_metric_sets.append(set(available_metrics))
        
        # Return intersection of all sets
        if all_metric_sets:
            return list(set.intersection(*all_metric_sets))
        return []
    
    def _extract_metric_values(self, metrics: ExperimentMetrics, metric_name: str) -> List[float]:
        """Extract values for a specific metric from ExperimentMetrics object."""
        metric_map = {
            'policy_loss': metrics.policy_losses,
            'value_loss': metrics.value_losses,
            'elo_changes': metrics.elo_changes,
            'game_lengths': metrics.game_lengths,
            'move_entropies': metrics.move_entropies,
            'win_rates': metrics.win_rates,
            'search_times': metrics.search_times
        }
        
        values = metric_map.get(metric_name, [])
        return [float(v) for v in values if v is not None] if values else []
    
    def get_best_experiment(self, experiment_identifiers: Union[List[Union[str, int]], Dict[Union[str, int], Dict[str, Any]]],
                           metric: str = "elo_changes",
                           criterion: str = "max",
                           source: str = "auto") -> Optional[Tuple[Union[str, int], float]]:
        """
        Find the best performing experiment based on a specific metric.
        
        Args:
            experiment_identifiers: List of experiment IDs to compare, or dict of experiment data
            metric: Metric to evaluate (default: "elo_changes")
            criterion: "max" for highest value, "min" for lowest value
            source: Data source
            
        Returns:
            Tuple of (experiment_id, performance_value) or None if comparison fails
        """
        comparison = self.compare_experiments(experiment_identifiers, [metric], source)
        
        if metric not in comparison.metric_comparisons:
            return None
        
        best_exp_id = None
        best_value = None
        
        for exp_id, metrics_obj in comparison.metric_comparisons[metric].items():
            value = metrics_obj.mean  # Use mean as the comparison value
            
            if best_value is None:
                best_exp_id = exp_id
                best_value = value
            elif (criterion == "max" and value > best_value) or (criterion == "min" and value < best_value):
                best_exp_id = exp_id
                best_value = value
        
        return (best_exp_id, best_value) if best_exp_id is not None else None
    
    def normalize_experiments(self, experiments: Dict[Union[str, int], Dict[str, Any]],
                            method: str = "z_score") -> Dict[Union[str, int], Dict[str, Any]]:
        """
        Normalize experiment metrics for fair comparison.
        
        Args:
            experiments: Dictionary of experiment data
            method: Normalization method - "z_score", "min_max", or "robust"
            
        Returns:
            Dictionary of experiments with normalized metrics
        """
        if method not in ["z_score", "min_max", "robust"]:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Extract all metric values for normalization parameters
        all_metrics = {}
        for exp_data in experiments.values():
            metrics_obj = exp_data['metrics']
            
            for metric_name in ['policy_loss', 'value_loss', 'elo_changes', 'game_lengths']:
                values = self._extract_metric_values(metrics_obj, metric_name)
                if values:
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].extend(values)
        
        # Calculate normalization parameters
        norm_params = {}
        for metric_name, all_values in all_metrics.items():
            if method == "z_score":
                norm_params[metric_name] = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values)
                }
            elif method == "min_max":
                norm_params[metric_name] = {
                    'min': np.min(all_values),
                    'max': np.max(all_values)
                }
            elif method == "robust":
                norm_params[metric_name] = {
                    'median': np.median(all_values),
                    'mad': np.median(np.abs(all_values - np.median(all_values)))
                }
        
        # Apply normalization to each experiment
        normalized_experiments = {}
        for exp_id, exp_data in experiments.items():
            normalized_exp = exp_data.copy()
            normalized_metrics = self._normalize_experiment_metrics(
                exp_data['metrics'], norm_params, method
            )
            normalized_exp['metrics'] = normalized_metrics
            normalized_experiments[exp_id] = normalized_exp
        
        return normalized_experiments
    
    def _normalize_experiment_metrics(self, metrics: ExperimentMetrics,
                                    norm_params: Dict[str, Dict[str, float]],
                                    method: str) -> ExperimentMetrics:
        """Apply normalization to a single experiment's metrics."""
        normalized_data = {}
        
        # Map attribute names to metric keys used in norm_params
        attr_to_metric_map = {
            'policy_losses': 'policy_loss',
            'value_losses': 'value_loss',
            'elo_changes': 'elo_changes',
            'game_lengths': 'game_lengths'
        }
        
        for attr_name in ['policy_losses', 'value_losses', 'elo_changes', 'game_lengths']:
            values = getattr(metrics, attr_name, [])
            if not values:
                normalized_data[attr_name] = values
                continue
            
            metric_key = attr_to_metric_map[attr_name]
            if metric_key in norm_params:
                params = norm_params[metric_key]
                
                if method == "z_score" and params['std'] != 0:
                    normalized_values = [(v - params['mean']) / params['std'] for v in values]
                elif method == "min_max" and params['max'] != params['min']:
                    normalized_values = [(v - params['min']) / (params['max'] - params['min']) for v in values]
                elif method == "robust" and params['mad'] != 0:
                    normalized_values = [(v - params['median']) / params['mad'] for v in values]
                else:
                    normalized_values = values  # No normalization if parameters invalid
                
                normalized_data[attr_name] = normalized_values
            else:
                normalized_data[attr_name] = values
        
        # Copy optional metrics as-is
        normalized_data['move_entropies'] = metrics.move_entropies
        normalized_data['win_rates'] = metrics.win_rates
        normalized_data['search_times'] = metrics.search_times
        normalized_data['timestamps'] = metrics.timestamps
        
        return ExperimentMetrics(**normalized_data) 