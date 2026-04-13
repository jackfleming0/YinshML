"""Phase-based game analysis for Yinsh self-play data.

This module implements game phase segmentation and analyzes how feature 
importance changes throughout game progression, identifying phase-specific 
patterns and strategic insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

from .correlation_analyzer import CorrelationAnalyzer, FeatureImportanceAnalysis, CorrelationResult

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GamePhase(Enum):
    """Game phase enumeration."""
    EARLY = "early"
    MID = "mid"
    LATE = "late"


@dataclass
class PhaseConfig:
    """Configuration for game phase segmentation."""
    early_turns: Tuple[int, int] = (1, 15)
    mid_turns: Tuple[int, int] = (16, 35)
    late_turns: Tuple[int, int] = (36, 999)  # 999 as upper bound
    
    def get_phase(self, turn_number: int) -> GamePhase:
        """Determine game phase based on turn number."""
        if self.early_turns[0] <= turn_number <= self.early_turns[1]:
            return GamePhase.EARLY
        elif self.mid_turns[0] <= turn_number <= self.mid_turns[1]:
            return GamePhase.MID
        else:
            return GamePhase.LATE


@dataclass
class PhaseAnalysisResult:
    """Container for phase-specific analysis results."""
    
    phase: GamePhase
    turn_range: Tuple[int, int]
    sample_size: int
    correlation_results: List[CorrelationResult]
    top_features: List[str]
    significant_features: List[str]
    phase_summary: Dict[str, Any]
    
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


@dataclass
class PhaseComparisonAnalysis:
    """Container for complete phase comparison analysis."""
    
    phase_results: Dict[GamePhase, PhaseAnalysisResult]
    phase_evolution: Dict[str, List[float]]  # Feature correlation evolution across phases
    phase_patterns: Dict[str, Any]
    comparison_summary: Dict[str, Any]
    
    def get_feature_evolution(self, feature_name: str) -> Dict[GamePhase, float]:
        """Get correlation evolution for a specific feature across phases."""
        evolution = {}
        for phase, result in self.phase_results.items():
            # Find correlation for this feature
            for corr_result in result.correlation_results:
                if feature_name in corr_result.feature_name:
                    evolution[phase] = corr_result.correlation
                    break
            else:
                evolution[phase] = 0.0  # Default if not found
        return evolution


class PhaseAnalyzer:
    """Main class for phase-based game analysis."""
    
    def __init__(self, data_dir: str = "self_play_data", phase_config: PhaseConfig = None):
        """Initialize the phase analyzer.
        
        Args:
            data_dir: Directory containing self-play data files
            phase_config: Configuration for phase segmentation
        """
        self.data_dir = Path(data_dir)
        self.phase_config = phase_config or PhaseConfig()
        self.correlation_analyzer = CorrelationAnalyzer(data_dir)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_and_segment_data(self) -> Dict[GamePhase, pd.DataFrame]:
        """Load self-play data and segment by game phases.
        
        Returns:
            Dictionary mapping phases to their respective DataFrames
        """
        logger.info("Loading and segmenting data by game phases")
        
        # Load raw data
        df = self.correlation_analyzer.load_self_play_data()
        df = self.correlation_analyzer.prepare_outcome_variables(df)
        
        # Add phase information
        df['game_phase'] = df['turn_number'].apply(self.phase_config.get_phase)
        
        # Segment by phase
        phase_data = {}
        for phase in GamePhase:
            phase_df = df[df['game_phase'] == phase].copy()
            phase_data[phase] = phase_df
            
            logger.info(f"Phase {phase.value}: {len(phase_df)} turns "
                       f"({len(phase_df['game_id'].unique())} games)")
        
        return phase_data
    
    def analyze_phase_correlations(self, phase_data: Dict[GamePhase, pd.DataFrame]) -> Dict[GamePhase, PhaseAnalysisResult]:
        """Analyze correlations for each game phase.
        
        Args:
            phase_data: Dictionary mapping phases to DataFrames
            
        Returns:
            Dictionary mapping phases to analysis results
        """
        logger.info("Analyzing correlations for each game phase")
        
        phase_results = {}
        
        for phase, df in phase_data.items():
            if df.empty:
                logger.warning(f"No data for phase {phase.value}")
                continue
            
            logger.info(f"Analyzing phase {phase.value} with {len(df)} turns")
            
            # Perform correlation analysis for this phase
            analysis = self.correlation_analyzer.calculate_correlations(df)
            
            # Determine turn range for this phase
            turn_range = (df['turn_number'].min(), df['turn_number'].max())
            
            # Create phase-specific result
            phase_result = PhaseAnalysisResult(
                phase=phase,
                turn_range=turn_range,
                sample_size=len(df),
                correlation_results=analysis.correlation_results,
                top_features=analysis.top_features,
                significant_features=analysis.significant_features,
                phase_summary=analysis.analysis_summary
            )
            
            phase_results[phase] = phase_result
            
            logger.info(f"Phase {phase.value}: {len(analysis.correlation_results)} correlations, "
                       f"{len(analysis.significant_features)} significant")
        
        return phase_results
    
    def analyze_feature_evolution(self, phase_results: Dict[GamePhase, PhaseAnalysisResult]) -> Dict[str, List[float]]:
        """Analyze how feature correlations evolve across phases.
        
        Args:
            phase_results: Dictionary mapping phases to analysis results
            
        Returns:
            Dictionary mapping feature names to correlation evolution lists
        """
        logger.info("Analyzing feature correlation evolution across phases")
        
        # Get all unique feature names
        all_features = set()
        for result in phase_results.values():
            for corr_result in result.correlation_results:
                # Extract base feature name (before _vs_)
                base_feature = corr_result.feature_name.split('_vs_')[0]
                all_features.add(base_feature)
        
        feature_evolution = {}
        
        for feature in all_features:
            evolution = []
            for phase in [GamePhase.EARLY, GamePhase.MID, GamePhase.LATE]:
                if phase in phase_results:
                    # Find correlation for this feature in this phase
                    correlation = 0.0
                    for corr_result in phase_results[phase].correlation_results:
                        if corr_result.feature_name.startswith(feature + '_vs_'):
                            correlation = corr_result.correlation
                            break
                    evolution.append(correlation)
                else:
                    evolution.append(0.0)
            
            feature_evolution[feature] = evolution
        
        return feature_evolution
    
    def identify_phase_patterns(self, phase_results: Dict[GamePhase, PhaseAnalysisResult]) -> Dict[str, Any]:
        """Identify patterns in feature importance across phases.
        
        Args:
            phase_results: Dictionary mapping phases to analysis results
            
        Returns:
            Dictionary containing identified patterns
        """
        logger.info("Identifying phase-specific patterns")
        
        patterns = {
            'early_game_dominators': [],
            'late_game_dominators': [],
            'consistent_predictors': [],
            'phase_specific_features': {},
            'correlation_trends': {}
        }
        
        # Analyze each phase
        for phase, result in phase_results.items():
            top_features = result.get_top_features(5)
            significant_features = result.get_significant_features()
            
            patterns['phase_specific_features'][phase.value] = {
                'top_features': [f.feature_name for f in top_features],
                'significant_count': len(significant_features),
                'sample_size': result.sample_size
            }
        
        # Identify features that dominate in early vs late game
        early_top = set()
        late_top = set()
        
        if GamePhase.EARLY in phase_results:
            early_top = set(f.feature_name.split('_vs_')[0] 
                          for f in phase_results[GamePhase.EARLY].get_top_features(10))
        
        if GamePhase.LATE in phase_results:
            late_top = set(f.feature_name.split('_vs_')[0] 
                         for f in phase_results[GamePhase.LATE].get_top_features(10))
        
        patterns['early_game_dominators'] = list(early_top - late_top)
        patterns['late_game_dominators'] = list(late_top - early_top)
        patterns['consistent_predictors'] = list(early_top & late_top)
        
        return patterns
    
    def create_phase_comparison_plot(self, phase_results: Dict[GamePhase, PhaseAnalysisResult],
                                   figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """Create visualization comparing feature importance across phases.
        
        Args:
            phase_results: Dictionary mapping phases to analysis results
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Importance Across Game Phases', fontsize=16, fontweight='bold')
        
        # Plot 1: Top features by phase
        ax1 = axes[0, 0]
        phases = list(phase_results.keys())
        phase_names = [p.value for p in phases]
        
        # Get top 5 features for each phase
        top_features_by_phase = {}
        max_features = 0
        for phase, result in phase_results.items():
            top_features = result.get_top_features(5)
            correlations = [f.correlation for f in top_features if not np.isnan(f.correlation)]
            top_features_by_phase[phase.value] = correlations
            max_features = max(max_features, len(correlations))
        
        # Create bar plot
        x = np.arange(max_features)
        width = 0.25
        
        for i, (phase_name, correlations) in enumerate(top_features_by_phase.items()):
            # Pad with zeros if needed
            padded_correlations = correlations + [0.0] * (max_features - len(correlations))
            ax1.bar(x + i * width, padded_correlations, width, label=phase_name, alpha=0.8)
        
        ax1.set_xlabel('Feature Rank')
        ax1.set_ylabel('Correlation Strength')
        ax1.set_title('Top 5 Feature Correlations by Phase')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f'#{i+1}' for i in range(max_features)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Significant features count
        ax2 = axes[0, 1]
        significant_counts = [len(result.get_significant_features()) 
                            for result in phase_results.values()]
        
        bars = ax2.bar(phase_names, significant_counts, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_xlabel('Game Phase')
        ax2.set_ylabel('Number of Significant Features')
        ax2.set_title('Statistically Significant Features by Phase')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, significant_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        # Plot 3: Sample sizes
        ax3 = axes[1, 0]
        sample_sizes = [result.sample_size for result in phase_results.values()]
        
        bars = ax3.bar(phase_names, sample_sizes, alpha=0.8, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax3.set_xlabel('Game Phase')
        ax3.set_ylabel('Number of Turns')
        ax3.set_title('Sample Size by Phase')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, sample_sizes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_sizes)*0.01,
                    str(size), ha='center', va='bottom')
        
        # Plot 4: Feature evolution (top 5 most variable features)
        ax4 = axes[1, 1]
        
        # Calculate feature variability across phases
        feature_variability = {}
        for feature in self.correlation_analyzer.feature_columns:
            correlations = []
            for phase in phases:
                if phase in phase_results:
                    for corr_result in phase_results[phase].correlation_results:
                        if corr_result.feature_name.startswith(feature + '_vs_'):
                            correlations.append(corr_result.correlation)
                            break
                    else:
                        correlations.append(0.0)
                else:
                    correlations.append(0.0)
            
            # Calculate standard deviation as variability measure
            variability = np.std(correlations) if len(correlations) > 1 else 0.0
            feature_variability[feature] = variability
        
        # Get top 5 most variable features
        top_variable_features = sorted(feature_variability.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Plot evolution for top variable features
        for feature, _ in top_variable_features:
            correlations = []
            for phase in phases:
                if phase in phase_results:
                    for corr_result in phase_results[phase].correlation_results:
                        if corr_result.feature_name.startswith(feature + '_vs_'):
                            correlations.append(corr_result.correlation)
                            break
                    else:
                        correlations.append(0.0)
                else:
                    correlations.append(0.0)
            
            ax4.plot(phase_names, correlations, marker='o', label=feature, linewidth=2)
        
        ax4.set_xlabel('Game Phase')
        ax4.set_ylabel('Correlation Strength')
        ax4.set_title('Feature Correlation Evolution (Top 5 Most Variable)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_feature_evolution_heatmap(self, feature_evolution: Dict[str, List[float]],
                                       figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Create heatmap showing feature correlation evolution across phases.
        
        Args:
            feature_evolution: Dictionary mapping feature names to correlation evolution
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        features = list(feature_evolution.keys())
        phases = ['Early', 'Mid', 'Late']
        
        # Create correlation matrix
        correlation_matrix = np.array([feature_evolution[feature] for feature in features])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            correlation_matrix,
            xticklabels=phases,
            yticklabels=features,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt='.3f',
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Evolution Across Game Phases', fontsize=14, fontweight='bold')
        ax.set_xlabel('Game Phase', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def generate_phase_report(self, phase_results: Dict[GamePhase, PhaseAnalysisResult],
                            patterns: Dict[str, Any]) -> str:
        """Generate comprehensive report of phase analysis.
        
        Args:
            phase_results: Dictionary mapping phases to analysis results
            patterns: Identified phase patterns
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("PHASE-BASED GAME ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Phase segmentation summary
        report.append("PHASE SEGMENTATION:")
        report.append(f"  Early Game: Turns {self.phase_config.early_turns[0]}-{self.phase_config.early_turns[1]}")
        report.append(f"  Mid Game: Turns {self.phase_config.mid_turns[0]}-{self.phase_config.mid_turns[1]}")
        report.append(f"  Late Game: Turns {self.phase_config.late_turns[0]}+")
        report.append("")
        
        # Phase-specific results
        for phase, result in phase_results.items():
            report.append(f"{phase.value.upper()} GAME ANALYSIS:")
            report.append("-" * 40)
            report.append(f"  Turn Range: {result.turn_range[0]}-{result.turn_range[1]}")
            report.append(f"  Sample Size: {result.sample_size} turns")
            report.append(f"  Total Correlations: {len(result.correlation_results)}")
            report.append(f"  Significant Features: {len(result.significant_features)}")
            report.append(f"  Strong Correlations: {len([r for r in result.correlation_results if r.is_strong()])}")
            
            # Top features for this phase
            top_features = result.get_top_features(5)
            if top_features:
                report.append("  Top Features:")
                for i, feature in enumerate(top_features, 1):
                    if not np.isnan(feature.correlation):
                        report.append(f"    {i}. {feature.feature_name}: {feature.correlation:.3f} (p={feature.p_value:.3f})")
            report.append("")
        
        # Phase patterns
        report.append("PHASE-SPECIFIC PATTERNS:")
        report.append("-" * 40)
        
        if patterns['early_game_dominators']:
            report.append("Early Game Dominators:")
            for feature in patterns['early_game_dominators']:
                report.append(f"  - {feature}")
            report.append("")
        
        if patterns['late_game_dominators']:
            report.append("Late Game Dominators:")
            for feature in patterns['late_game_dominators']:
                report.append(f"  - {feature}")
            report.append("")
        
        if patterns['consistent_predictors']:
            report.append("Consistent Predictors (Early & Late):")
            for feature in patterns['consistent_predictors']:
                report.append(f"  - {feature}")
            report.append("")
        
        # Strategic insights
        report.append("STRATEGIC INSIGHTS:")
        report.append("-" * 40)
        
        # Analyze ring centrality vs ring spread across phases
        ring_centrality_evolution = []
        ring_spread_evolution = []
        
        for phase in [GamePhase.EARLY, GamePhase.MID, GamePhase.LATE]:
            if phase in phase_results:
                # Find ring centrality correlation
                centrality_corr = 0.0
                spread_corr = 0.0
                
                for corr_result in phase_results[phase].correlation_results:
                    if 'ring_centrality_score' in corr_result.feature_name:
                        centrality_corr = corr_result.correlation
                    elif 'ring_spread' in corr_result.feature_name:
                        spread_corr = corr_result.correlation
                
                ring_centrality_evolution.append(centrality_corr)
                ring_spread_evolution.append(spread_corr)
            else:
                ring_centrality_evolution.append(0.0)
                ring_spread_evolution.append(0.0)
        
        report.append("Ring Strategy Evolution:")
        report.append(f"  Ring Centrality: Early={ring_centrality_evolution[0]:.3f}, "
                     f"Mid={ring_centrality_evolution[1]:.3f}, Late={ring_centrality_evolution[2]:.3f}")
        report.append(f"  Ring Spread: Early={ring_spread_evolution[0]:.3f}, "
                     f"Mid={ring_spread_evolution[1]:.3f}, Late={ring_spread_evolution[2]:.3f}")
        
        if abs(ring_centrality_evolution[0]) > abs(ring_centrality_evolution[2]):
            report.append("  → Ring centrality matters more in early game")
        elif abs(ring_centrality_evolution[2]) > abs(ring_centrality_evolution[0]):
            report.append("  → Ring centrality matters more in late game")
        else:
            report.append("  → Ring centrality importance is consistent across phases")
        
        if abs(ring_spread_evolution[0]) > abs(ring_spread_evolution[2]):
            report.append("  → Ring spread matters more in early game")
        elif abs(ring_spread_evolution[2]) > abs(ring_spread_evolution[0]):
            report.append("  → Ring spread matters more in late game")
        else:
            report.append("  → Ring spread importance is consistent across phases")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_analysis(self, save_plots: bool = True, 
                            output_dir: str = "analysis_output") -> PhaseComparisonAnalysis:
        """Run complete phase-based analysis pipeline.
        
        Args:
            save_plots: Whether to save visualization plots
            output_dir: Directory to save outputs
            
        Returns:
            Complete phase comparison analysis
        """
        logger.info("Starting complete phase-based analysis")
        
        # Load and segment data
        phase_data = self.load_and_segment_data()
        
        # Analyze correlations for each phase
        phase_results = self.analyze_phase_correlations(phase_data)
        
        # Analyze feature evolution
        feature_evolution = self.analyze_feature_evolution(phase_results)
        
        # Identify phase patterns
        patterns = self.identify_phase_patterns(phase_results)
        
        # Create visualizations
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Phase comparison plot
            comparison_fig = self.create_phase_comparison_plot(phase_results)
            comparison_fig.savefig(output_path / "phase_comparison_analysis.png", 
                                  dpi=300, bbox_inches='tight')
            plt.close(comparison_fig)
            
            # Feature evolution heatmap
            evolution_fig = self.create_feature_evolution_heatmap(feature_evolution)
            evolution_fig.savefig(output_path / "feature_evolution_heatmap.png", 
                               dpi=300, bbox_inches='tight')
            plt.close(evolution_fig)
        
        # Generate and save report
        report = self.generate_phase_report(phase_results, patterns)
        if save_plots:
            with open(output_path / "phase_analysis_report.txt", 'w') as f:
                f.write(report)
        
        # Create comparison summary
        comparison_summary = {
            'total_phases': len(phase_results),
            'total_correlations': sum(len(r.correlation_results) for r in phase_results.values()),
            'total_significant': sum(len(r.significant_features) for r in phase_results.values()),
            'early_game_dominators': len(patterns['early_game_dominators']),
            'late_game_dominators': len(patterns['late_game_dominators']),
            'consistent_predictors': len(patterns['consistent_predictors'])
        }
        
        logger.info("Phase-based analysis completed")
        
        return PhaseComparisonAnalysis(
            phase_results=phase_results,
            phase_evolution=feature_evolution,
            phase_patterns=patterns,
            comparison_summary=comparison_summary
        )


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run analysis
    analyzer = PhaseAnalyzer()
    analysis = analyzer.run_complete_analysis()
    
    # Print summary
    print(analyzer.generate_phase_report(analysis.phase_results, analysis.phase_patterns))
