"""Hyperparameter importance analysis for experiment optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from functools import lru_cache
import warnings
import logging
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.inspection import partial_dependence
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError:
    raise ImportError("scikit-learn is required for hyperparameter analysis. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("matplotlib and seaborn not available. Plotting functionality will be disabled.")

from experiments.comparator import ExperimentComparator
from experiments.hyperparameter_analysis_methods import (
    add_interaction_features, correlation_analysis, random_forest_analysis,
    lasso_analysis, create_correlation_matrix
)


@dataclass
class HyperparameterImportance:
    """Container for hyperparameter importance results."""
    hyperparameter: str
    correlation_importance: float
    rf_importance: float
    lasso_importance: float
    composite_score: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class ImportanceAnalysisResult:
    """Container for complete importance analysis results."""
    target_metric: str
    hyperparameter_rankings: List[HyperparameterImportance]
    model_performance: Dict[str, float]  # R² scores for different models
    correlation_matrix: Dict[str, Dict[str, float]]
    data_summary: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'target_metric': self.target_metric,
            'hyperparameter_rankings': [asdict(hp) for hp in self.hyperparameter_rankings],
            'model_performance': self.model_performance,
            'correlation_matrix': self.correlation_matrix,
            'data_summary': self.data_summary,
            'analysis_metadata': self.analysis_metadata
        }


class HyperparameterAnalyzer:
    """Analyzes hyperparameter importance for experiment optimization."""
    
    def __init__(self, experiment_comparator: Optional[ExperimentComparator] = None,
                 cache_size: int = 100, random_state: int = 42):
        """
        Initialize the hyperparameter analyzer.
        
        Args:
            experiment_comparator: ExperimentComparator instance for data loading
            cache_size: Size of LRU cache for analysis results
            random_state: Random seed for reproducibility
        """
        self.comparator = experiment_comparator or ExperimentComparator()
        self.cache_size = cache_size
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Cache for preprocessed data and models
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._model_cache: Dict[str, Any] = {}
    
    @lru_cache(maxsize=100)
    def analyze_hyperparameter_importance(self, 
                                        experiment_identifiers: Union[str, Tuple[str, ...]],
                                        target_metric: str = "elo_changes",
                                        source: str = "auto",
                                        aggregation_method: str = "mean",
                                        include_interactions: bool = False) -> ImportanceAnalysisResult:
        """
        Analyze hyperparameter importance for a given target metric.
        
        Args:
            experiment_identifiers: Experiments to analyze (string or tuple for caching)
            target_metric: Performance metric to analyze
            source: Data source ("files", "database", or "auto")
            aggregation_method: How to aggregate time series metrics ("mean", "final", "max")
            include_interactions: Whether to include interaction features
            
        Returns:
            ImportanceAnalysisResult with complete analysis
        """
        # Convert to tuple for caching if needed
        if isinstance(experiment_identifiers, str):
            exp_ids = (experiment_identifiers,)
        elif isinstance(experiment_identifiers, list):
            exp_ids = tuple(experiment_identifiers)
        else:
            exp_ids = experiment_identifiers
            
        # Load and prepare data
        data_df = self._prepare_analysis_data(exp_ids, target_metric, source, aggregation_method)
        
        if data_df.empty:
            raise ValueError("No valid data found for analysis")
            
        # Separate features and target
        hyperparams_df = data_df.drop(columns=[target_metric, 'experiment_id'], errors='ignore')
        target_values = data_df[target_metric].values
        
        # Add interaction features if requested
        if include_interactions:
            hyperparams_df = add_interaction_features(hyperparams_df)
        
        # Perform different importance analyses
        correlation_results = correlation_analysis(hyperparams_df, target_values)
        rf_results = random_forest_analysis(hyperparams_df, target_values, self.random_state)
        lasso_results = lasso_analysis(hyperparams_df, target_values, self.random_state)
        
        # Combine results and create rankings
        hyperparameter_rankings = self._create_composite_rankings(
            correlation_results, rf_results, lasso_results, hyperparams_df.columns
        )
        
        # Calculate model performance metrics
        model_performance = {
            'random_forest_r2': rf_results['r2_score'],
            'lasso_r2': lasso_results['r2_score'],
            'correlation_max_r2': max([abs(r['correlation']) ** 2 for r in correlation_results.values()])
        }
        
        # Create correlation matrix for all hyperparameters
        correlation_matrix = create_correlation_matrix(hyperparams_df)
        
        # Data summary
        data_summary = {
            'n_experiments': len(data_df),
            'n_hyperparameters': len(hyperparams_df.columns),
            'target_metric': target_metric,
            'target_mean': float(np.mean(target_values)),
            'target_std': float(np.std(target_values)),
            'target_range': (float(np.min(target_values)), float(np.max(target_values)))
        }
        
        # Analysis metadata
        analysis_metadata = {
            'aggregation_method': aggregation_method,
            'include_interactions': include_interactions,
            'random_state': self.random_state,
            'experiment_count': len(exp_ids)
        }
        
        return ImportanceAnalysisResult(
            target_metric=target_metric,
            hyperparameter_rankings=hyperparameter_rankings,
            model_performance=model_performance,
            correlation_matrix=correlation_matrix,
            data_summary=data_summary,
            analysis_metadata=analysis_metadata
        )
    
    def _prepare_analysis_data(self, experiment_identifiers: Tuple[str, ...], 
                             target_metric: str, source: str, 
                             aggregation_method: str) -> pd.DataFrame:
        """Prepare data for hyperparameter analysis."""
        cache_key = f"{experiment_identifiers}_{target_metric}_{source}_{aggregation_method}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Load experiments
        experiments = self.comparator.load_experiments(list(experiment_identifiers), source)
        
        if not experiments:
            return pd.DataFrame()
        
        data_rows = []
        
        for exp_id, exp_data in experiments.items():
            try:
                # Extract hyperparameters from config
                config = exp_data.get('config', {})
                if not config:
                    self.logger.warning(f"No config found for experiment {exp_id}")
                    continue
                
                # Extract target metric from metrics
                metrics = exp_data.get('metrics', {})
                if isinstance(metrics, dict) and hasattr(metrics, target_metric):
                    # Handle ExperimentMetrics object
                    target_values = getattr(metrics, target_metric)
                elif isinstance(metrics, dict) and target_metric in metrics:
                    # Handle dictionary metrics
                    target_values = metrics[target_metric]
                else:
                    self.logger.warning(f"Target metric {target_metric} not found for experiment {exp_id}")
                    continue
                
                # Aggregate target metric
                if isinstance(target_values, list) and target_values:
                    if aggregation_method == "mean":
                        target_value = np.mean(target_values)
                    elif aggregation_method == "final":
                        target_value = target_values[-1]
                    elif aggregation_method == "max":
                        target_value = np.max(target_values)
                    else:
                        target_value = np.mean(target_values)  # Default fallback
                else:
                    target_value = target_values if isinstance(target_values, (int, float)) else 0.0
                
                # Create row with hyperparameters and target
                row = config.copy()
                row[target_metric] = target_value
                row['experiment_id'] = exp_id
                
                data_rows.append(row)
                
            except Exception as e:
                self.logger.error(f"Error processing experiment {exp_id}: {e}")
                continue
        
        if not data_rows:
            return pd.DataFrame()
        
        # Convert to DataFrame and handle data types
        df = pd.DataFrame(data_rows)
        df = self._preprocess_hyperparameters(df, target_metric)
        
        # Cache the result
        self._data_cache[cache_key] = df
        
        return df
    
    def _preprocess_hyperparameters(self, df: pd.DataFrame, target_metric: str) -> pd.DataFrame:
        """Preprocess hyperparameters for analysis."""
        # Remove non-hyperparameter columns
        exclude_cols = {target_metric, 'experiment_id'}
        hyperparam_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        for col in hyperparam_cols:
            if df[col].dtype == 'object':
                # For string categorical variables, use label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            elif df[col].dtype == 'bool':
                # Convert boolean to int
                df[col] = df[col].astype(int)
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Remove constant columns (no variance)
        for col in hyperparam_cols:
            if col in df.columns and df[col].nunique() <= 1:
                df = df.drop(columns=[col])
                self.logger.info(f"Removed constant hyperparameter: {col}")
        
        return df
    
    def _create_composite_rankings(self, correlation_results: Dict, rf_results: Dict, 
                                 lasso_results: Dict, feature_names: List[str]) -> List[HyperparameterImportance]:
        """Create composite rankings combining all importance measures."""
        rankings = []
        
        # Normalize importance scores to [0, 1] range
        rf_importances = np.array([rf_results['feature_importances'].get(f, 0) for f in feature_names])
        lasso_importances = np.array([lasso_results['feature_importances'].get(f, 0) for f in feature_names])
        correlations = np.array([abs(correlation_results.get(f, {}).get('correlation', 0)) for f in feature_names])
        
        # Normalize each measure
        rf_norm = rf_importances / (np.max(rf_importances) + 1e-8)
        lasso_norm = lasso_importances / (np.max(lasso_importances) + 1e-8)
        corr_norm = correlations / (np.max(correlations) + 1e-8)
        
        # Create composite score (weighted average)
        composite_scores = 0.4 * corr_norm + 0.3 * rf_norm + 0.3 * lasso_norm
        
        for i, feature in enumerate(feature_names):
            hp_importance = HyperparameterImportance(
                hyperparameter=feature,
                correlation_importance=float(corr_norm[i]),
                rf_importance=float(rf_norm[i]),
                lasso_importance=float(lasso_norm[i]),
                composite_score=float(composite_scores[i]),
                p_value=correlation_results.get(feature, {}).get('p_value'),
            )
            rankings.append(hp_importance)
        
        # Sort by composite score
        rankings.sort(key=lambda x: x.composite_score, reverse=True)
        
        return rankings
    
    def generate_optimization_suggestions(self, experiment_identifiers: List[str],
                                        target_metric: str = "elo_changes",
                                        n_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate hyperparameter optimization suggestions based on analysis.
        
        Args:
            experiment_identifiers: List of experiments to analyze
            target_metric: Performance metric to optimize
            n_suggestions: Number of suggestions to generate
            
        Returns:
            List of optimization suggestions
        """
        # Get importance analysis
        importance_result = self.analyze_hyperparameter_importance(
            tuple(experiment_identifiers), target_metric
        )
        
        suggestions = []
        
        # Get top important hyperparameters
        top_hyperparams = importance_result.hyperparameter_rankings[:n_suggestions]
        
        for hp in top_hyperparams:
            suggestion = {
                'hyperparameter': hp.hyperparameter,
                'current_importance': hp.composite_score,
                'suggestion_type': self._get_suggestion_type(hp),
                'reasoning': self._generate_reasoning(hp)
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_suggestion_type(self, hp: HyperparameterImportance) -> str:
        """Determine suggestion type based on importance scores."""
        if hp.composite_score > 0.8:
            return "critical_optimization"
        elif hp.composite_score > 0.5:
            return "important_tuning"
        elif hp.composite_score > 0.2:
            return "minor_adjustment"
        else:
            return "monitoring_only"
    
    def _generate_reasoning(self, hp: HyperparameterImportance) -> str:
        """Generate human-readable reasoning for hyperparameter suggestion."""
        reasoning_parts = []
        
        # Importance-based reasoning
        if hp.correlation_importance > 0.7:
            reasoning_parts.append(f"Strong correlation with target metric (r={hp.correlation_importance:.3f})")
        
        if hp.rf_importance > 0.7:
            reasoning_parts.append("High Random Forest feature importance")
        
        if hp.lasso_importance > 0.7:
            reasoning_parts.append("Selected by LASSO regression")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Moderate impact on performance"
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._data_cache.clear()
        self._model_cache.clear()
        # Clear LRU cache
        self.analyze_hyperparameter_importance.cache_clear() 