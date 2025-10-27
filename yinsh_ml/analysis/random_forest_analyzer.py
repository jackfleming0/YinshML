"""Random Forest classifier for non-linear pattern discovery in Yinsh self-play data.

This module implements a comprehensive random forest analysis to discover feature 
interactions and non-linear patterns in game outcomes, including feature importance 
ranking, interaction analysis, and interpretable visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import export_text, plot_tree
import joblib

from .correlation_analyzer import CorrelationAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""
    n_estimators: int = 150
    max_depth: int = 12
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    
    feature_name: str
    importance: float
    std_importance: float
    rank: int
    cumulative_importance: float
    
    def is_important(self, threshold: float = 0.01) -> bool:
        """Check if feature is important based on threshold."""
        return self.importance > threshold


@dataclass
class InteractionAnalysis:
    """Container for feature interaction analysis."""
    
    feature_pair: Tuple[str, str]
    interaction_strength: float
    partial_dependence_data: Dict[str, Any]
    interaction_type: str  # 'synergistic', 'antagonistic', 'neutral'
    
    def is_significant(self, threshold: float = 0.1) -> bool:
        """Check if interaction is significant."""
        return abs(self.interaction_strength) > threshold


@dataclass
class RandomForestAnalysis:
    """Container for complete Random Forest analysis results."""
    
    model: Union[RandomForestClassifier, RandomForestRegressor]
    feature_importances: List[FeatureImportanceResult]
    interactions: List[InteractionAnalysis]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    model_performance: Dict[str, float]
    decision_trees: List[Any]
    analysis_summary: Dict[str, Any]


class RandomForestAnalyzer:
    """Main class for Random Forest analysis of self-play data."""
    
    def __init__(self, data_dir: str = "self_play_data", config: RandomForestConfig = None):
        """Initialize the Random Forest analyzer.
        
        Args:
            data_dir: Directory containing self-play data files
            config: Configuration for Random Forest model
        """
        self.data_dir = Path(data_dir)
        self.config = config or RandomForestConfig()
        self.correlation_analyzer = CorrelationAnalyzer(data_dir)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Feature columns for analysis
        self.feature_columns = [
            'ring_centrality_score', 'ring_spread', 'ring_mobility', 'edge_proximity_score',
            'marker_density_center', 'marker_density_inner', 'marker_density_outer', 'marker_density_edge',
            'potential_runs_count', 'blocking_positions', 'connected_marker_chains_length',
            'completed_runs_differential', 'rings_in_center_count', 'ring_clustering_pattern',
            'marker_pattern_type'
        ]
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load self-play data and prepare for Random Forest analysis.
        
        Returns:
            Tuple of (features_df, target_series, metadata_df)
        """
        logger.info("Loading and preparing data for Random Forest analysis")
        
        # Load raw data
        df = self.correlation_analyzer.load_self_play_data()
        df = self.correlation_analyzer.prepare_outcome_variables(df)
        
        # Prepare features
        features_df = df[self.feature_columns].copy()
        
        # Handle categorical features
        categorical_features = ['ring_clustering_pattern', 'marker_pattern_type']
        for feature in categorical_features:
            if feature in features_df.columns:
                # Encode categorical features
                le = LabelEncoder()
                features_df[feature] = le.fit_transform(features_df[feature].astype(str))
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Prepare target variable (binary classification: win/loss)
        target_series = (df['player_wins'] > 0.5).astype(int)
        
        # Prepare metadata
        metadata_df = df[['game_id', 'turn_number', 'current_player', 'total_turns']].copy()
        
        logger.info(f"Prepared {len(features_df)} samples with {len(features_df.columns)} features")
        logger.info(f"Target distribution: {target_series.value_counts().to_dict()}")
        
        return features_df, target_series, metadata_df
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Train Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained Random Forest classifier
        """
        logger.info("Training Random Forest classifier")
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        # Train the model
        rf.fit(X, y)
        
        logger.info(f"Trained Random Forest with {rf.n_estimators} trees")
        return rf
    
    def evaluate_model(self, rf: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate Random Forest model performance.
        
        Args:
            rf: Trained Random Forest classifier
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("Evaluating Random Forest model performance")
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X, y, cv=self.config.cv_folds, scoring='accuracy')
        
        # Predictions
        y_pred = rf.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        # Get feature importances
        feature_importances = rf.feature_importances_
        
        performance = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'feature_importances': feature_importances.tolist()
        }
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return performance
    
    def analyze_feature_importance(self, rf: RandomForestClassifier, 
                                 feature_names: List[str]) -> List[FeatureImportanceResult]:
        """Analyze feature importance from Random Forest.
        
        Args:
            rf: Trained Random Forest classifier
            feature_names: List of feature names
            
        Returns:
            List of feature importance results
        """
        logger.info("Analyzing feature importance")
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Calculate standard deviation of importances across trees
        std_importances = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        
        # Create feature importance results
        importance_results = []
        cumulative_importance = 0.0
        
        for i, (feature, importance, std_imp) in enumerate(zip(feature_names, importances, std_importances)):
            cumulative_importance += importance
            
            result = FeatureImportanceResult(
                feature_name=feature,
                importance=importance,
                std_importance=std_imp,
                rank=i + 1,
                cumulative_importance=cumulative_importance
            )
            
            importance_results.append(result)
        
        # Sort by importance
        importance_results.sort(key=lambda x: x.importance, reverse=True)
        
        # Update ranks
        for i, result in enumerate(importance_results):
            result.rank = i + 1
        
        logger.info(f"Analyzed importance for {len(importance_results)} features")
        return importance_results
    
    def analyze_feature_interactions(self, rf: RandomForestClassifier, X: pd.DataFrame, 
                                  feature_names: List[str]) -> List[InteractionAnalysis]:
        """Analyze feature interactions using partial dependence.
        
        Args:
            rf: Trained Random Forest classifier
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            List of interaction analysis results
        """
        logger.info("Analyzing feature interactions")
        
        interactions = []
        
        # Analyze top 10 most important features
        top_features = [f.feature_name for f in self.analyze_feature_importance(rf, feature_names)[:10]]
        
        # Analyze pairwise interactions
        for i, feature1 in enumerate(top_features):
            for feature2 in top_features[i+1:]:
                try:
                    # Get feature indices
                    idx1 = feature_names.index(feature1)
                    idx2 = feature_names.index(feature2)
                    
                    # Calculate interaction strength (simplified)
                    interaction_strength = self._calculate_interaction_strength(rf, X, idx1, idx2)
                    
                    # Create interaction analysis
                    interaction = InteractionAnalysis(
                        feature_pair=(feature1, feature2),
                        interaction_strength=interaction_strength,
                        partial_dependence_data={},  # Placeholder for now
                        interaction_type=self._classify_interaction_type(interaction_strength)
                    )
                    
                    interactions.append(interaction)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze interaction {feature1}-{feature2}: {e}")
                    continue
        
        logger.info(f"Analyzed {len(interactions)} feature interactions")
        return interactions
    
    def _calculate_interaction_strength(self, rf: RandomForestClassifier, X: pd.DataFrame, 
                                      idx1: int, idx2: int) -> float:
        """Calculate interaction strength between two features.
        
        Args:
            rf: Trained Random Forest classifier
            X: Feature matrix
            idx1: Index of first feature
            idx2: Index of second feature
            
        Returns:
            Interaction strength score
        """
        # Simplified interaction strength calculation
        # In practice, this would use more sophisticated methods like H-statistic
        
        # Get feature values
        feature1_values = X.iloc[:, idx1].values
        feature2_values = X.iloc[:, idx2].values
        
        # Calculate correlation between features
        correlation = np.corrcoef(feature1_values, feature2_values)[0, 1]
        
        # Calculate interaction as function of correlation and individual importances
        importance1 = rf.feature_importances_[idx1]
        importance2 = rf.feature_importances_[idx2]
        
        # Interaction strength based on correlation and individual importances
        interaction_strength = abs(correlation) * (importance1 + importance2) / 2
        
        return interaction_strength
    
    def _classify_interaction_type(self, interaction_strength: float) -> str:
        """Classify interaction type based on strength.
        
        Args:
            interaction_strength: Calculated interaction strength
            
        Returns:
            Interaction type classification
        """
        if interaction_strength > 0.1:
            return 'synergistic'
        elif interaction_strength < -0.1:
            return 'antagonistic'
        else:
            return 'neutral'
    
    def create_feature_importance_plot(self, importance_results: List[FeatureImportanceResult],
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create feature importance visualization.
        
        Args:
            importance_results: List of feature importance results
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        features = [r.feature_name for r in importance_results[:15]]  # Top 15 features
        importances = [r.importance for r in importance_results[:15]]
        std_importances = [r.std_importance for r in importance_results[:15]]
        
        # Create bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, xerr=std_importances, alpha=0.7, capsize=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Random Forest Feature Importance Rankings', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_interaction_heatmap(self, interactions: List[InteractionAnalysis],
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Create feature interaction heatmap.
        
        Args:
            interactions: List of interaction analysis results
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique features
        all_features = set()
        for interaction in interactions:
            all_features.update(interaction.feature_pair)
        
        features = sorted(list(all_features))
        
        # Create interaction matrix
        interaction_matrix = np.zeros((len(features), len(features)))
        
        for interaction in interactions:
            idx1 = features.index(interaction.feature_pair[0])
            idx2 = features.index(interaction.feature_pair[1])
            interaction_matrix[idx1, idx2] = interaction.interaction_strength
            interaction_matrix[idx2, idx1] = interaction.interaction_strength
        
        # Create heatmap
        sns.heatmap(
            interaction_matrix,
            xticklabels=features,
            yticklabels=features,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt='.3f',
            cbar_kws={'label': 'Interaction Strength'},
            ax=ax
        )
        
        ax.set_title('Feature Interaction Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig
    
    def create_decision_tree_visualization(self, rf: RandomForestClassifier, 
                                        feature_names: List[str],
                                        max_depth: int = 3,
                                        figsize: Tuple[int, int] = (20, 10)) -> plt.Figure:
        """Create decision tree visualization for interpretability.
        
        Args:
            rf: Trained Random Forest classifier
            feature_names: List of feature names
            max_depth: Maximum depth to visualize
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the first tree (most representative)
        tree = rf.estimators_[0]
        
        # Plot decision tree
        plot_tree(tree, 
                 feature_names=feature_names,
                 max_depth=max_depth,
                 filled=True,
                 rounded=True,
                 ax=ax)
        
        ax.set_title('Decision Tree Visualization (First Tree)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_analysis_report(self, analysis: RandomForestAnalysis) -> str:
        """Generate comprehensive analysis report.
        
        Args:
            analysis: Complete Random Forest analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("RANDOM FOREST ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model configuration
        report.append("MODEL CONFIGURATION:")
        report.append(f"  Number of Trees: {self.config.n_estimators}")
        report.append(f"  Max Depth: {self.config.max_depth}")
        report.append(f"  Min Samples Split: {self.config.min_samples_split}")
        report.append(f"  Min Samples Leaf: {self.config.min_samples_leaf}")
        report.append(f"  Max Features: {self.config.max_features}")
        report.append(f"  Cross-Validation Folds: {self.config.cv_folds}")
        report.append("")
        
        # Model performance
        performance = analysis.model_performance
        report.append("MODEL PERFORMANCE:")
        report.append(f"  Accuracy: {performance['accuracy']:.3f}")
        report.append(f"  CV Accuracy: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        report.append(f"  CV Scores: {[f'{score:.3f}' for score in performance['cv_scores']]}")
        report.append("")
        
        # Feature importance
        report.append("TOP 10 FEATURE IMPORTANCES:")
        report.append("-" * 60)
        report.append(f"{'Rank':<5} {'Feature':<35} {'Importance':<12} {'Std Dev':<10}")
        report.append("-" * 60)
        
        for result in analysis.feature_importances[:10]:
            report.append(f"{result.rank:<5} {result.feature_name:<35} "
                        f"{result.importance:<12.3f} {result.std_importance:<10.3f}")
        report.append("")
        
        # Feature interactions
        significant_interactions = [i for i in analysis.interactions if i.is_significant()]
        if significant_interactions:
            report.append("SIGNIFICANT FEATURE INTERACTIONS:")
            report.append("-" * 60)
            report.append(f"{'Feature Pair':<40} {'Strength':<12} {'Type':<15}")
            report.append("-" * 60)
            
            for interaction in significant_interactions[:10]:
                feature_pair = f"{interaction.feature_pair[0]} × {interaction.feature_pair[1]}"
                report.append(f"{feature_pair:<40} {interaction.interaction_strength:<12.3f} "
                            f"{interaction.interaction_type:<15}")
            report.append("")
        
        # Analysis summary
        summary = analysis.analysis_summary
        report.append("ANALYSIS SUMMARY:")
        report.append(f"  Total Features Analyzed: {summary.get('total_features', 0)}")
        report.append(f"  Important Features (>0.01): {summary.get('important_features', 0)}")
        report.append(f"  Significant Interactions: {len(significant_interactions)}")
        report.append(f"  Model Stability (CV std): {performance['cv_std']:.3f}")
        report.append("")
        
        # Strategic insights
        report.append("STRATEGIC INSIGHTS:")
        report.append("-" * 40)
        
        # Top 3 features
        top_features = analysis.feature_importances[:3]
        report.append("Most Important Features:")
        for i, feature in enumerate(top_features, 1):
            report.append(f"  {i}. {feature.feature_name}: {feature.importance:.3f}")
        
        # Feature interactions
        if significant_interactions:
            report.append("\nKey Feature Interactions:")
            for interaction in significant_interactions[:3]:
                report.append(f"  {interaction.feature_pair[0]} × {interaction.feature_pair[1]}: "
                            f"{interaction.interaction_strength:.3f} ({interaction.interaction_type})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, save_plots: bool = True, 
                            output_dir: str = "analysis_output") -> RandomForestAnalysis:
        """Run complete Random Forest analysis pipeline.
        
        Args:
            save_plots: Whether to save visualization plots
            output_dir: Directory to save outputs
            
        Returns:
            Complete Random Forest analysis results
        """
        logger.info("Starting complete Random Forest analysis")
        
        # Load and prepare data
        X, y, metadata = self.load_and_prepare_data()
        
        # Train Random Forest
        rf = self.train_random_forest(X, y)
        
        # Evaluate model
        performance = self.evaluate_model(rf, X, y)
        
        # Analyze feature importance
        feature_importances = self.analyze_feature_importance(rf, X.columns.tolist())
        
        # Analyze feature interactions
        interactions = self.analyze_feature_interactions(rf, X, X.columns.tolist())
        
        # Get decision trees for visualization
        decision_trees = rf.estimators_[:5]  # First 5 trees
        
        # Create analysis summary
        analysis_summary = {
            'total_features': len(X.columns),
            'important_features': len([f for f in feature_importances if f.is_important()]),
            'total_interactions': len(interactions),
            'significant_interactions': len([i for i in interactions if i.is_significant()])
        }
        
        # Create visualizations
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Feature importance plot
            importance_fig = self.create_feature_importance_plot(feature_importances)
            importance_fig.savefig(output_path / "random_forest_feature_importance.png", 
                                  dpi=300, bbox_inches='tight')
            plt.close(importance_fig)
            
            # Interaction heatmap
            interaction_fig = self.create_interaction_heatmap(interactions)
            interaction_fig.savefig(output_path / "feature_interaction_heatmap.png", 
                                  dpi=300, bbox_inches='tight')
            plt.close(interaction_fig)
            
            # Decision tree visualization
            tree_fig = self.create_decision_tree_visualization(rf, X.columns.tolist())
            tree_fig.savefig(output_path / "decision_tree_visualization.png", 
                            dpi=300, bbox_inches='tight')
            plt.close(tree_fig)
        
        # Generate and save report
        analysis = RandomForestAnalysis(
            model=rf,
            feature_importances=feature_importances,
            interactions=interactions,
            cv_scores=performance['cv_scores'],
            cv_mean=performance['cv_mean'],
            cv_std=performance['cv_std'],
            model_performance=performance,
            decision_trees=decision_trees,
            analysis_summary=analysis_summary
        )
        
        report = self.generate_analysis_report(analysis)
        if save_plots:
            with open(output_path / "random_forest_analysis_report.txt", 'w') as f:
                f.write(report)
        
        logger.info("Random Forest analysis completed")
        return analysis


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run analysis
    analyzer = RandomForestAnalyzer()
    analysis = analyzer.run_complete_analysis()
    
    # Print summary
    print(analyzer.generate_analysis_report(analysis))
