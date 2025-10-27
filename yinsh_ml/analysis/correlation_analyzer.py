"""Linear correlation analysis for feature importance in Yinsh self-play data.

This module implements comprehensive correlation analysis to identify features
most correlated with game outcomes, including statistical significance testing
and visualization capabilities.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from scipy import stats
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    
    feature_name: str
    correlation: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    correlation_type: str  # 'pearson' or 'spearman'
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        return self.p_value < alpha
    
    def is_strong(self, threshold: float = 0.1) -> bool:
        """Check if correlation is strong enough."""
        return abs(self.correlation) > threshold


@dataclass
class FeatureImportanceAnalysis:
    """Container for complete feature importance analysis."""
    
    correlation_results: List[CorrelationResult]
    correlation_matrix: pd.DataFrame
    top_features: List[str]
    significant_features: List[str]
    analysis_summary: Dict[str, Any]
    
    def get_top_features(self, n: int = 10) -> List[CorrelationResult]:
        """Get top N features by absolute correlation strength."""
        sorted_results = sorted(
            self.correlation_results,
            key=lambda x: abs(x.correlation),
            reverse=True
        )
        return sorted_results[:n]
    
    def get_significant_features(self, alpha: float = 0.05) -> List[CorrelationResult]:
        """Get all statistically significant features."""
        return [r for r in self.correlation_results if r.is_significant(alpha)]


class CorrelationAnalyzer:
    """Main class for performing correlation analysis on self-play data."""
    
    def __init__(self, data_dir: str = "self_play_data"):
        """Initialize the correlation analyzer.
        
        Args:
            data_dir: Directory containing self-play data files
        """
        self.data_dir = Path(data_dir)
        self.feature_columns = [
            'ring_centrality_score', 'ring_spread', 'ring_mobility', 'edge_proximity_score',
            'marker_density_center', 'marker_density_inner', 'marker_density_outer', 'marker_density_edge',
            'potential_runs_count', 'blocking_positions', 'connected_marker_chains_length',
            'completed_runs_differential', 'rings_in_center_count', 'ring_clustering_pattern',
            'marker_pattern_type'
        ]
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_self_play_data(self) -> pd.DataFrame:
        """Load self-play data from JSON files.
        
        Returns:
            DataFrame with flattened game data including features and outcomes
        """
        logger.info(f"Loading self-play data from {self.data_dir}")
        
        all_data = []
        json_files = list(self.data_dir.glob("game_*.json"))
        
        if not json_files:
            raise ValueError(f"No game files found in {self.data_dir}")
        
        logger.info(f"Found {len(json_files)} game files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    game_data = json.load(f)
                
                # Extract game-level information
                game_id = game_data['game_id']
                winner = game_data['winner']
                final_score = game_data['final_score']
                total_turns = game_data['total_turns']
                
                # Process each turn
                for turn in game_data['turns']:
                    row = {
                        'game_id': game_id,
                        'turn_number': turn['turn_number'],
                        'current_player': turn['current_player'],
                        'winner': winner,
                        'total_turns': total_turns,
                        'white_score': final_score['white'],
                        'black_score': final_score['black']
                    }
                    
                    # Add features
                    features = turn['features']
                    row.update(features)
                    
                    all_data.append(row)
                    
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        
        if df.empty:
            raise ValueError("No valid data loaded")
        
        logger.info(f"Loaded {len(df)} turn records from {len(json_files)} games")
        return df
    
    def prepare_outcome_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare outcome variables for correlation analysis.
        
        Args:
            df: Raw game data DataFrame
            
        Returns:
            DataFrame with outcome variables added
        """
        # Create outcome variables
        df = df.copy()
        
        # 1. Game outcome for current player (1 = win, 0 = loss, 0.5 = draw)
        df['player_wins'] = df.apply(
            lambda row: 1.0 if row['winner'] == row['current_player'] 
            else 0.0 if row['winner'] != 0 
            else 0.5, axis=1
        )
        
        # 2. Score differential (positive = winning)
        df['score_differential'] = df.apply(
            lambda row: row['white_score'] - row['black_score'] if row['current_player'] == 1
            else row['black_score'] - row['white_score'], axis=1
        )
        
        # 3. Normalized score differential
        df['normalized_score_diff'] = df['score_differential'] / df['total_turns']
        
        # 4. Game length (proxy for game complexity)
        df['game_length'] = df['total_turns']
        
        # 5. Turn position (early/mid/late game)
        df['turn_position'] = df['turn_number'] / df['total_turns']
        
        return df
    
    def calculate_correlations(self, df: pd.DataFrame, 
                             outcome_vars: List[str] = None) -> FeatureImportanceAnalysis:
        """Calculate correlations between features and outcome variables.
        
        Args:
            df: DataFrame with features and outcomes
            outcome_vars: List of outcome variables to analyze
            
        Returns:
            FeatureImportanceAnalysis with complete results
        """
        if outcome_vars is None:
            outcome_vars = ['player_wins', 'score_differential', 'normalized_score_diff']
        
        logger.info(f"Calculating correlations for {len(outcome_vars)} outcome variables")
        
        correlation_results = []
        
        # Separate numerical and categorical features
        numerical_features = []
        categorical_features = []
        
        for feature in self.feature_columns:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            # Check if feature is numerical
            if pd.api.types.is_numeric_dtype(df[feature]):
                numerical_features.append(feature)
            else:
                categorical_features.append(feature)
        
        logger.info(f"Found {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        
        # Calculate correlations for each outcome variable
        for outcome_var in outcome_vars:
            if outcome_var not in df.columns:
                logger.warning(f"Outcome variable {outcome_var} not found in data")
                continue
            
            # Process numerical features
            for feature in numerical_features:
                # Get valid data (non-null values)
                valid_mask = df[feature].notna() & df[outcome_var].notna()
                feature_data = df.loc[valid_mask, feature]
                outcome_data = df.loc[valid_mask, outcome_var]
                
                if len(feature_data) < 10:  # Need minimum sample size
                    logger.warning(f"Insufficient data for {feature} vs {outcome_var}")
                    continue
                
                try:
                    # Calculate Pearson correlation
                    pearson_corr, pearson_p = pearsonr(feature_data, outcome_data)
                    
                    # Calculate Spearman correlation (rank-based)
                    spearman_corr, spearman_p = spearmanr(feature_data, outcome_data)
                    
                    # Choose the stronger correlation
                    if abs(pearson_corr) >= abs(spearman_corr):
                        correlation = pearson_corr
                        p_value = pearson_p
                        corr_type = 'pearson'
                    else:
                        correlation = spearman_corr
                        p_value = spearman_p
                        corr_type = 'spearman'
                    
                    # Calculate confidence interval
                    n = len(feature_data)
                    ci = self._calculate_confidence_interval(correlation, n)
                    
                    result = CorrelationResult(
                        feature_name=f"{feature}_vs_{outcome_var}",
                        correlation=correlation,
                        p_value=p_value,
                        confidence_interval=ci,
                        sample_size=n,
                        correlation_type=corr_type
                    )
                    
                    correlation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Correlation calculation failed for {feature} vs {outcome_var}: {e}")
                    continue
            
            # Process categorical features using chi-square test
            for feature in categorical_features:
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(df[feature], df[outcome_var])
                    
                    # Calculate chi-square statistic
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    # Calculate Cramér's V as effect size
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    
                    # Convert to correlation-like measure
                    correlation = cramers_v if cramers_v <= 1.0 else 1.0
                    
                    # Calculate confidence interval (approximate)
                    ci = self._calculate_confidence_interval(correlation, n)
                    
                    result = CorrelationResult(
                        feature_name=f"{feature}_vs_{outcome_var}",
                        correlation=correlation,
                        p_value=p_value,
                        confidence_interval=ci,
                        sample_size=n,
                        correlation_type='cramers_v'
                    )
                    
                    correlation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Categorical correlation calculation failed for {feature} vs {outcome_var}: {e}")
                    continue
        
        # Create correlation matrix for features
        feature_df = df[self.feature_columns].select_dtypes(include=[np.number])
        correlation_matrix = feature_df.corr()
        
        # Identify top features
        top_features = self._identify_top_features(correlation_results)
        significant_features = [r.feature_name for r in correlation_results if r.is_significant()]
        
        # Create analysis summary
        analysis_summary = {
            'total_correlations': len(correlation_results),
            'significant_correlations': len(significant_features),
            'strong_correlations': len([r for r in correlation_results if r.is_strong()]),
            'top_correlation': max([abs(r.correlation) for r in correlation_results]) if correlation_results else 0,
            'outcome_variables': outcome_vars,
            'features_analyzed': len(self.feature_columns)
        }
        
        return FeatureImportanceAnalysis(
            correlation_results=correlation_results,
            correlation_matrix=correlation_matrix,
            top_features=top_features,
            significant_features=significant_features,
            analysis_summary=analysis_summary
        )
    
    def _calculate_confidence_interval(self, correlation: float, n: int, 
                                     alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient.
        
        Args:
            correlation: Correlation coefficient
            n: Sample size
            alpha: Significance level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n < 3:
            return (0.0, 0.0)
        
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        
        # Standard error
        se = 1.0 / np.sqrt(n - 3)
        
        # Critical value
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back to correlation space
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (ci_lower, ci_upper)
    
    def _identify_top_features(self, results: List[CorrelationResult], 
                            n: int = 10) -> List[str]:
        """Identify top features by correlation strength.
        
        Args:
            results: List of correlation results
            n: Number of top features to return
            
        Returns:
            List of top feature names
        """
        sorted_results = sorted(
            results,
            key=lambda x: abs(x.correlation),
            reverse=True
        )
        return [r.feature_name for r in sorted_results[:n]]
    
    def create_correlation_heatmap(self, analysis: FeatureImportanceAnalysis,
                                 figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Create correlation heatmap visualization.
        
        Args:
            analysis: Feature importance analysis results
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            analysis.correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig
    
    def create_outcome_correlation_plot(self, analysis: FeatureImportanceAnalysis,
                                      figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """Create visualization of correlations with outcome variables.
        
        Args:
            analysis: Feature importance analysis results
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        # Get top correlations
        top_results = analysis.get_top_features(15)
        
        if not top_results:
            logger.warning("No correlation results to plot")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        feature_names = [r.feature_name for r in top_results]
        correlations = [r.correlation for r in top_results]
        p_values = [r.p_value for r in top_results]
        
        # Color by significance
        colors = ['red' if p < 0.05 else 'lightblue' for p in p_values]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, correlations, color=colors, alpha=0.7)
        
        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       '*', ha='left', va='center', fontsize=12, color='red')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title('Feature Correlations with Game Outcomes\n(* indicates p < 0.05)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, analysis: FeatureImportanceAnalysis) -> str:
        """Generate a comprehensive text report of the analysis.
        
        Args:
            analysis: Feature importance analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        summary = analysis.analysis_summary
        report.append("SUMMARY STATISTICS:")
        report.append(f"  Total correlations calculated: {summary['total_correlations']}")
        report.append(f"  Statistically significant (p < 0.05): {summary['significant_correlations']}")
        report.append(f"  Strong correlations (|r| > 0.1): {summary['strong_correlations']}")
        report.append(f"  Strongest correlation: {summary['top_correlation']:.3f}")
        report.append(f"  Outcome variables analyzed: {', '.join(summary['outcome_variables'])}")
        report.append(f"  Features analyzed: {summary['features_analyzed']}")
        report.append("")
        
        # Top features
        top_features = analysis.get_top_features(10)
        if top_features:
            report.append("TOP 10 FEATURES BY CORRELATION STRENGTH:")
            report.append("-" * 60)
            report.append(f"{'Feature':<40} {'Correlation':<12} {'P-value':<10} {'Significant':<12}")
            report.append("-" * 60)
            
            for result in top_features:
                significance = "Yes" if result.is_significant() else "No"
                report.append(f"{result.feature_name:<40} {result.correlation:>10.3f} "
                            f"{result.p_value:>8.3f} {significance:>10}")
            report.append("")
        
        # Significant features only
        significant_features = analysis.get_significant_features()
        if significant_features:
            report.append("STATISTICALLY SIGNIFICANT FEATURES (p < 0.05):")
            report.append("-" * 60)
            report.append(f"{'Feature':<40} {'Correlation':<12} {'P-value':<10} {'CI (95%)':<20}")
            report.append("-" * 60)
            
            for result in significant_features:
                ci_str = f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
                report.append(f"{result.feature_name:<40} {result.correlation:>10.3f} "
                            f"{result.p_value:>8.3f} {ci_str:>18}")
            report.append("")
        
        # Strong correlations
        strong_features = [r for r in analysis.correlation_results if r.is_strong()]
        if strong_features:
            report.append("STRONG CORRELATIONS (|r| > 0.1):")
            report.append("-" * 60)
            report.append(f"{'Feature':<40} {'Correlation':<12} {'P-value':<10} {'Sample Size':<12}")
            report.append("-" * 60)
            
            for result in strong_features:
                report.append(f"{result.feature_name:<40} {result.correlation:>10.3f} "
                            f"{result.p_value:>8.3f} {result.sample_size:>10}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, save_plots: bool = True, 
                            output_dir: str = "analysis_output") -> FeatureImportanceAnalysis:
        """Run complete correlation analysis pipeline.
        
        Args:
            save_plots: Whether to save visualization plots
            output_dir: Directory to save outputs
            
        Returns:
            Complete feature importance analysis
        """
        logger.info("Starting complete correlation analysis")
        
        # Load and prepare data
        df = self.load_self_play_data()
        df = self.prepare_outcome_variables(df)
        
        # Perform analysis
        analysis = self.calculate_correlations(df)
        
        # Generate visualizations
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Correlation heatmap
            heatmap_fig = self.create_correlation_heatmap(analysis)
            heatmap_fig.savefig(output_path / "feature_correlation_heatmap.png", 
                              dpi=300, bbox_inches='tight')
            plt.close(heatmap_fig)
            
            # Outcome correlation plot
            outcome_fig = self.create_outcome_correlation_plot(analysis)
            if outcome_fig:
                outcome_fig.savefig(output_path / "outcome_correlations.png", 
                                  dpi=300, bbox_inches='tight')
                plt.close(outcome_fig)
        
        # Generate and save report
        report = self.generate_report(analysis)
        if save_plots:
            with open(output_path / "correlation_analysis_report.txt", 'w') as f:
                f.write(report)
        
        logger.info("Correlation analysis completed")
        return analysis


def validate_correlation_calculations():
    """Validate correlation calculations with synthetic data."""
    logger.info("Validating correlation calculations with synthetic data")
    
    # Create synthetic data with known correlations
    np.random.seed(42)
    n_samples = 1000
    
    # Perfect positive correlation
    x1 = np.random.normal(0, 1, n_samples)
    y1 = x1 + np.random.normal(0, 0.1, n_samples)  # Should be ~0.99
    
    # Perfect negative correlation
    x2 = np.random.normal(0, 1, n_samples)
    y2 = -x2 + np.random.normal(0, 0.1, n_samples)  # Should be ~-0.99
    
    # No correlation
    x3 = np.random.normal(0, 1, n_samples)
    y3 = np.random.normal(0, 1, n_samples)  # Should be ~0
    
    # Test correlations
    analyzer = CorrelationAnalyzer()
    
    # Test perfect positive
    corr1, p1 = pearsonr(x1, y1)
    print(f"Perfect positive correlation test: {corr1:.3f} (expected ~0.99)")
    
    # Test perfect negative
    corr2, p2 = pearsonr(x2, y2)
    print(f"Perfect negative correlation test: {corr2:.3f} (expected ~-0.99)")
    
    # Test no correlation
    corr3, p3 = pearsonr(x3, y3)
    print(f"No correlation test: {corr3:.3f} (expected ~0)")
    
    # Test confidence intervals
    ci1 = analyzer._calculate_confidence_interval(corr1, n_samples)
    print(f"Confidence interval for perfect positive: [{ci1[0]:.3f}, {ci1[1]:.3f}]")
    
    print("Validation completed successfully!")
    return True


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Validate calculations first
    validate_correlation_calculations()
    
    # Run analysis on real data
    analyzer = CorrelationAnalyzer()
    analysis = analyzer.run_complete_analysis()
    
    # Print summary
    print(analyzer.generate_report(analysis))
