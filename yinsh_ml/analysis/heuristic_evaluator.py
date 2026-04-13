"""Heuristic evaluation function based on discovered patterns from analysis.

This module implements a weighted evaluation function that combines features
with weights derived from correlation analysis and random forest importance
scores, optimized for fast MCTS simulations.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

from .correlation_analyzer import CorrelationAnalyzer
from .random_forest_analyzer import RandomForestAnalyzer
from .phase_analyzer import PhaseAnalyzer, GamePhase
from ..game.game_state import GameState
from ..game.constants import Player
from ..analysis.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class EvaluationWeights:
    """Container for evaluation function weights."""
    
    feature_weights: Dict[str, float]
    bias: float
    feature_names: List[str]
    normalization_params: Dict[str, Dict[str, float]]
    
    def get_weight(self, feature_name: str) -> float:
        """Get weight for a specific feature."""
        return self.feature_weights.get(feature_name, 0.0)
    
    def get_normalization_params(self, feature_name: str) -> Dict[str, float]:
        """Get normalization parameters for a feature."""
        return self.normalization_params.get(feature_name, {'mean': 0.0, 'std': 1.0})


@dataclass
class EvaluationConfig:
    """Configuration for heuristic evaluation function."""
    
    top_features_count: int = 7
    use_phase_specific: bool = True
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    bias_adjustment: float = 0.0
    speed_target: int = 1000  # evaluations per second
    random_seed: int = 42


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    score: float
    feature_contributions: Dict[str, float]
    evaluation_time: float
    phase: Optional[GamePhase] = None
    
    def get_top_contributors(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N contributing features."""
        sorted_contributions = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_contributions[:n]


class HeuristicEvaluator:
    """Main class for heuristic evaluation function."""
    
    def __init__(self, data_dir: str = "self_play_data", config: EvaluationConfig = None):
        """Initialize the heuristic evaluator.
        
        Args:
            data_dir: Directory containing self-play data files
            config: Configuration for evaluation function
        """
        self.data_dir = Path(data_dir)
        self.config = config or EvaluationConfig()
        self.feature_extractor = FeatureExtractor()
        
        # Initialize analyzers
        self.correlation_analyzer = CorrelationAnalyzer(data_dir)
        self.random_forest_analyzer = RandomForestAnalyzer(data_dir)
        self.phase_analyzer = PhaseAnalyzer(data_dir)
        
        # Evaluation weights and parameters
        self.weights: Optional[EvaluationWeights] = None
        self.phase_weights: Dict[GamePhase, EvaluationWeights] = {}
        self.scaler: Optional[StandardScaler] = None
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        # Default symmetric weights used when trained weights are unavailable
        self._default_weights = {
            'completed_runs_differential': 1.0,
            'potential_runs_count': 0.2,
            'blocking_positions': 0.2,
            'connected_marker_chains_length': 0.2,
            'ring_mobility': 0.1,
            'ring_centrality_score': 0.05,
            'edge_proximity_score': 0.05,
            # region densities are low signal individually; omit by default
        }
        
    def load_analysis_results(self) -> Tuple[Any, Any, Any]:
        """Load results from previous analyses.
        
        Returns:
            Tuple of (correlation_analysis, random_forest_analysis, phase_analysis)
        """
        logger.info("Loading analysis results for evaluation function")
        
        # Load correlation analysis
        try:
            correlation_analysis = self.correlation_analyzer.run_complete_analysis(save_plots=False)
            logger.info("Loaded correlation analysis results")
        except Exception as e:
            logger.warning(f"Failed to load correlation analysis: {e}")
            correlation_analysis = None
        
        # Load random forest analysis
        try:
            random_forest_analysis = self.random_forest_analyzer.run_complete_analysis(save_plots=False)
            logger.info("Loaded random forest analysis results")
        except Exception as e:
            logger.warning(f"Failed to load random forest analysis: {e}")
            random_forest_analysis = None
        
        # Load phase analysis
        try:
            phase_analysis = self.phase_analyzer.run_complete_analysis(save_plots=False)
            logger.info("Loaded phase analysis results")
        except Exception as e:
            logger.warning(f"Failed to load phase analysis: {e}")
            phase_analysis = None
        
        return correlation_analysis, random_forest_analysis, phase_analysis
    
    def extract_feature_weights(self, correlation_analysis: Any, 
                              random_forest_analysis: Any) -> Dict[str, float]:
        """Extract feature weights from analysis results.
        
        Args:
            correlation_analysis: Correlation analysis results
            random_forest_analysis: Random Forest analysis results
            
        Returns:
            Dictionary mapping feature names to weights
        """
        logger.info("Extracting feature weights from analysis results")
        
        weights = {}
        
        # Extract weights from Random Forest analysis (primary source)
        if random_forest_analysis:
            rf_weights = {}
            for feature_result in random_forest_analysis.feature_importances:
                rf_weights[feature_result.feature_name] = feature_result.importance
            
            # Normalize Random Forest weights
            total_rf_weight = sum(rf_weights.values())
            if total_rf_weight > 0:
                for feature, weight in rf_weights.items():
                    weights[feature] = weight / total_rf_weight
        
        # Extract weights from correlation analysis (secondary source)
        if correlation_analysis:
            corr_weights = {}
            for corr_result in correlation_analysis.correlation_results:
                if corr_result.is_significant() and corr_result.is_strong():
                    # Extract base feature name
                    base_feature = corr_result.feature_name.split('_vs_')[0]
                    corr_weights[base_feature] = abs(corr_result.correlation)
            
            # Normalize correlation weights
            total_corr_weight = sum(corr_weights.values())
            if total_corr_weight > 0:
                for feature, weight in corr_weights.items():
                    # Combine with Random Forest weights (weighted average)
                    rf_weight = weights.get(feature, 0.0)
                    corr_weight = weight / total_corr_weight
                    weights[feature] = 0.7 * rf_weight + 0.3 * corr_weight
        
        # Select top features
        top_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_features = top_features[:self.config.top_features_count]
        
        # Normalize top feature weights
        total_weight = sum(weight for _, weight in top_features)
        if total_weight > 0:
            normalized_weights = {feature: weight / total_weight for feature, weight in top_features}
        else:
            normalized_weights = {feature: 1.0 / len(top_features) for feature, _ in top_features}
        
        logger.info(f"Extracted weights for {len(normalized_weights)} top features")
        for feature, weight in normalized_weights.items():
            logger.info(f"  {feature}: {weight:.3f}")
        
        return normalized_weights
    
    def create_normalization_params(self, feature_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Create normalization parameters for features.
        
        Args:
            feature_weights: Dictionary of feature weights
            
        Returns:
            Dictionary of normalization parameters
        """
        logger.info("Creating normalization parameters")
        
        # Load data to calculate normalization parameters
        df = self.correlation_analyzer.load_self_play_data()
        df = self.correlation_analyzer.prepare_outcome_variables(df)
        
        normalization_params = {}
        
        for feature_name in feature_weights.keys():
            if feature_name in df.columns:
                feature_data = df[feature_name].dropna()
                
                if len(feature_data) > 0:
                    if self.config.normalization_method == 'standard':
                        mean_val = feature_data.mean()
                        std_val = feature_data.std()
                        if std_val == 0:
                            std_val = 1.0
                        normalization_params[feature_name] = {
                            'mean': mean_val,
                            'std': std_val
                        }
                    elif self.config.normalization_method == 'minmax':
                        min_val = feature_data.min()
                        max_val = feature_data.max()
                        if max_val == min_val:
                            max_val = min_val + 1.0
                        normalization_params[feature_name] = {
                            'min': min_val,
                            'max': max_val
                        }
                    else:  # robust
                        median_val = feature_data.median()
                        mad_val = np.median(np.abs(feature_data - median_val))
                        if mad_val == 0:
                            mad_val = 1.0
                        normalization_params[feature_name] = {
                            'median': median_val,
                            'mad': mad_val
                        }
                else:
                    # Default parameters
                    normalization_params[feature_name] = {'mean': 0.0, 'std': 1.0}
            else:
                # Default parameters
                normalization_params[feature_name] = {'mean': 0.0, 'std': 1.0}
        
        logger.info(f"Created normalization parameters for {len(normalization_params)} features")
        return normalization_params
    
    def train_evaluation_function(self) -> EvaluationWeights:
        """Train the evaluation function using analysis results.
        
        Returns:
            Trained evaluation weights
        """
        logger.info("Training heuristic evaluation function")
        
        # Load analysis results
        correlation_analysis, random_forest_analysis, phase_analysis = self.load_analysis_results()
        
        # Extract feature weights
        feature_weights = self.extract_feature_weights(correlation_analysis, random_forest_analysis)
        
        # Create normalization parameters
        normalization_params = self.create_normalization_params(feature_weights)
        
        # Calculate bias (average outcome)
        df = self.correlation_analyzer.load_self_play_data()
        df = self.correlation_analyzer.prepare_outcome_variables(df)
        bias = df['player_wins'].mean() + self.config.bias_adjustment
        
        # Create evaluation weights
        weights = EvaluationWeights(
            feature_weights=feature_weights,
            bias=bias,
            feature_names=list(feature_weights.keys()),
            normalization_params=normalization_params
        )
        
        self.weights = weights
        logger.info(f"Trained evaluation function with {len(feature_weights)} features, bias: {bias:.3f}")
        
        return weights
    
    def normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize a feature value.
        
        Args:
            feature_name: Name of the feature
            value: Raw feature value
            
        Returns:
            Normalized feature value
        """
        if feature_name not in self.weights.normalization_params:
            return value
        
        params = self.weights.normalization_params[feature_name]
        
        if self.config.normalization_method == 'standard':
            return (value - params['mean']) / params['std']
        elif self.config.normalization_method == 'minmax':
            return (value - params['min']) / (params['max'] - params['min'])
        else:  # robust
            return (value - params['median']) / params['mad']
    
    def extract_features(self, game_state: GameState, player: Player) -> Dict[str, float]:
        """Extract features for evaluation.
        
        Args:
            game_state: Current game state
            player: Player to evaluate for
            
        Returns:
            Dictionary of extracted features
        """
        # Extract all features
        feature_vector = self.feature_extractor.extract_all_features(game_state, player)
        
        # Convert to dictionary
        features = feature_vector.to_dict()
        
        # Return only the features we have weights for
        return {name: features.get(name, 0.0) for name in self.weights.feature_names}
    
    def evaluate_position(self, game_state: GameState, player: Player) -> EvaluationResult:
        """Evaluate a game position.
        
        Args:
            game_state: Current game state
            player: Player to evaluate for
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        # Use symmetric evaluation: (features(player) - features(opponent)) · w
        start = time.time()
        # If weights not trained, use default symmetric weights with zero bias
        if self.weights is None:
            p_feats = self.feature_extractor.extract_all_features(game_state, player).to_dict()
            o_feats = self.feature_extractor.extract_all_features(game_state, player.opponent).to_dict()
            score = 0.0
            feature_contributions = {}
            for name, weight in self._default_weights.items():
                pv = p_feats.get(name, 0.0)
                ov = o_feats.get(name, 0.0)
                diff = pv - ov
                contribution = weight * diff
                feature_contributions[name] = contribution
                score += contribution
        else:
            # Trained weights path
            p_feats = self.extract_features(game_state, player)
            o_feats = self.extract_features(game_state, player.opponent)
            score = 0.0  # antisymmetric baseline
            feature_contributions = {}
            for feature_name, p_val in p_feats.items():
                # Compute normalized difference
                p_norm = self.normalize_feature(feature_name, p_val)
                o_val = o_feats.get(feature_name, 0.0)
                o_norm = self.normalize_feature(feature_name, o_val)
                diff = p_norm - o_norm
                weight = self.weights.get_weight(feature_name)
                contribution = weight * diff
                feature_contributions[feature_name] = contribution
                score += contribution
        
        evaluation_time = time.time() - start
        
        # Update performance tracking
        self.evaluation_count += 1
        self.total_evaluation_time += evaluation_time
        
        return EvaluationResult(
            score=score,
            feature_contributions=feature_contributions,
            evaluation_time=evaluation_time
        )
    
    def evaluate_position_fast(self, game_state: GameState, player: Player) -> float:
        """Fast evaluation for MCTS (optimized version).
        
        Args:
            game_state: Current game state
            player: Player to evaluate for
            
        Returns:
            Evaluation score
        """
        # Symmetric fast path
        if self.weights is None:
            p = self.feature_extractor.extract_all_features(game_state, player).to_dict()
            o = self.feature_extractor.extract_all_features(game_state, player.opponent).to_dict()
            score = 0.0
            for name, weight in self._default_weights.items():
                score += weight * (p.get(name, 0.0) - o.get(name, 0.0))
            return score
        else:
            p = self.feature_extractor.extract_all_features(game_state, player).to_dict()
            o = self.feature_extractor.extract_all_features(game_state, player.opponent).to_dict()
            score = 0.0  # antisymmetric baseline
            for feature_name in self.weights.feature_names:
                pn = self.normalize_feature(feature_name, p.get(feature_name, 0.0))
                on = self.normalize_feature(feature_name, o.get(feature_name, 0.0))
                diff = pn - on
                weight = self.weights.get_weight(feature_name)
                score += weight * diff
            return score
    
    def benchmark_evaluation_speed(self, num_evaluations: int = 1000) -> Dict[str, float]:
        """Benchmark evaluation speed.
        
        Args:
            num_evaluations: Number of evaluations to perform
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking evaluation speed with {num_evaluations} evaluations")
        
        # Create a simple test game state
        from ..game.game_state import GameState
        from ..game.constants import Player
        
        # This is a simplified test - in practice you'd use real game states
        test_game_state = GameState()
        test_player = Player.WHITE
        
        # Benchmark fast evaluation
        start_time = time.time()
        for _ in range(num_evaluations):
            self.evaluate_position_fast(test_game_state, test_player)
        fast_time = time.time() - start_time
        
        # Benchmark full evaluation
        start_time = time.time()
        for _ in range(min(100, num_evaluations)):  # Fewer for full evaluation
            self.evaluate_position(test_game_state, test_player)
        full_time = time.time() - start_time
        
        # Calculate metrics
        fast_evaluations_per_second = num_evaluations / fast_time
        full_evaluations_per_second = min(100, num_evaluations) / full_time
        
        results = {
            'fast_evaluations_per_second': fast_evaluations_per_second,
            'full_evaluations_per_second': full_evaluations_per_second,
            'fast_time_total': fast_time,
            'full_time_total': full_time,
            'meets_speed_target': fast_evaluations_per_second >= self.config.speed_target
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Fast evaluation: {fast_evaluations_per_second:.1f} evaluations/second")
        logger.info(f"  Full evaluation: {full_evaluations_per_second:.1f} evaluations/second")
        logger.info(f"  Meets speed target: {results['meets_speed_target']}")
        
        return results
    
    def validate_evaluation_scores(self) -> Dict[str, Any]:
        """Validate evaluation scores against known positions.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating evaluation scores")
        
        # Load data
        df = self.correlation_analyzer.load_self_play_data()
        df = self.correlation_analyzer.prepare_outcome_variables(df)
        
        # Sample some positions for validation
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=self.config.random_seed)
        
        # Calculate evaluation scores
        evaluation_scores = []
        actual_outcomes = []
        
        for _, row in sample_df.iterrows():
            # This is a simplified validation - in practice you'd reconstruct game states
            # For now, we'll use the feature values directly
            score = self.weights.bias
            
            for feature_name in self.weights.feature_names:
                if feature_name in row:
                    value = row[feature_name]
                    normalized_value = self.normalize_feature(feature_name, value)
                    weight = self.weights.get_weight(feature_name)
                    score += weight * normalized_value
            
            evaluation_scores.append(score)
            actual_outcomes.append(row['player_wins'])
        
        # Calculate correlation between evaluation scores and actual outcomes
        correlation = np.corrcoef(evaluation_scores, actual_outcomes)[0, 1]
        
        # Calculate accuracy
        predicted_outcomes = [1 if score > 0.5 else 0 for score in evaluation_scores]
        accuracy = sum(1 for pred, actual in zip(predicted_outcomes, actual_outcomes) if pred == actual) / len(actual_outcomes)
        
        results = {
            'correlation': correlation,
            'accuracy': accuracy,
            'sample_size': sample_size,
            'mean_evaluation_score': np.mean(evaluation_scores),
            'std_evaluation_score': np.std(evaluation_scores)
        }
        
        logger.info(f"Validation results:")
        logger.info(f"  Correlation with outcomes: {correlation:.3f}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Mean evaluation score: {results['mean_evaluation_score']:.3f}")
        
        return results
    
    def benchmark_against_random(self) -> Dict[str, float]:
        """Benchmark evaluation function against random baseline.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Benchmarking against random baseline")
        
        # Load data
        df = self.correlation_analyzer.load_self_play_data()
        df = self.correlation_analyzer.prepare_outcome_variables(df)
        
        # Sample positions
        sample_size = min(200, len(df))
        sample_df = df.sample(n=sample_size, random_state=self.config.random_seed)
        
        # Calculate evaluation scores
        evaluation_scores = []
        actual_outcomes = []
        
        for _, row in sample_df.iterrows():
            score = self.weights.bias
            
            for feature_name in self.weights.feature_names:
                if feature_name in row:
                    value = row[feature_name]
                    normalized_value = self.normalize_feature(feature_name, value)
                    weight = self.weights.get_weight(feature_name)
                    score += weight * normalized_value
            
            evaluation_scores.append(score)
            actual_outcomes.append(row['player_wins'])
        
        # Calculate performance metrics
        correlation = np.corrcoef(evaluation_scores, actual_outcomes)[0, 1]
        accuracy = sum(1 for score, outcome in zip(evaluation_scores, actual_outcomes) 
                      if (score > 0.5) == (outcome > 0.5)) / len(actual_outcomes)
        
        # Random baseline
        random_accuracy = 0.5  # Random guessing
        random_correlation = 0.0  # No correlation
        
        results = {
            'heuristic_correlation': correlation,
            'heuristic_accuracy': accuracy,
            'random_correlation': random_correlation,
            'random_accuracy': random_accuracy,
            'correlation_improvement': correlation - random_correlation,
            'accuracy_improvement': accuracy - random_accuracy
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Heuristic correlation: {correlation:.3f}")
        logger.info(f"  Heuristic accuracy: {accuracy:.3f}")
        logger.info(f"  Random correlation: {random_correlation:.3f}")
        logger.info(f"  Random accuracy: {random_accuracy:.3f}")
        logger.info(f"  Correlation improvement: {results['correlation_improvement']:.3f}")
        logger.info(f"  Accuracy improvement: {results['accuracy_improvement']:.3f}")
        
        return results
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation function report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("HEURISTIC EVALUATION FUNCTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Configuration
        report.append("CONFIGURATION:")
        report.append(f"  Top Features Count: {self.config.top_features_count}")
        report.append(f"  Use Phase Specific: {self.config.use_phase_specific}")
        report.append(f"  Normalization Method: {self.config.normalization_method}")
        report.append(f"  Speed Target: {self.config.speed_target} evaluations/second")
        report.append("")
        
        # Feature weights
        if self.weights:
            report.append("FEATURE WEIGHTS:")
            report.append("-" * 40)
            report.append(f"{'Feature':<35} {'Weight':<10} {'Normalization':<20}")
            report.append("-" * 40)
            
            for feature_name in self.weights.feature_names:
                weight = self.weights.get_weight(feature_name)
                norm_params = self.weights.normalization_params.get(feature_name, {})
                norm_str = f"mean={norm_params.get('mean', 0):.2f}, std={norm_params.get('std', 1):.2f}"
                report.append(f"{feature_name:<35} {weight:<10.3f} {norm_str:<20}")
            report.append("")
            
            report.append(f"Bias: {self.weights.bias:.3f}")
            report.append("")
        
        # Performance metrics
        if self.evaluation_count > 0:
            avg_evaluation_time = self.total_evaluation_time / self.evaluation_count
            evaluations_per_second = 1.0 / avg_evaluation_time if avg_evaluation_time > 0 else 0
            
            report.append("PERFORMANCE METRICS:")
            report.append(f"  Total Evaluations: {self.evaluation_count}")
            report.append(f"  Average Evaluation Time: {avg_evaluation_time:.6f} seconds")
            report.append(f"  Evaluations Per Second: {evaluations_per_second:.1f}")
            report.append(f"  Meets Speed Target: {evaluations_per_second >= self.config.speed_target}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_complete_training(self, save_model: bool = True, 
                            output_dir: str = "analysis_output") -> Dict[str, Any]:
        """Run complete training and validation pipeline.
        
        Args:
            save_model: Whether to save the trained model
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting complete heuristic evaluation function training")
        
        # Train evaluation function
        weights = self.train_evaluation_function()
        
        # Benchmark speed
        speed_results = self.benchmark_evaluation_speed()
        
        # Validate scores
        validation_results = self.validate_evaluation_scores()
        
        # Benchmark against random
        benchmark_results = self.benchmark_against_random()
        
        # Generate report
        report = self.generate_evaluation_report()
        
        # Save results
        if save_model:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save model
            model_data = {
                'weights': weights,
                'config': self.config,
                'speed_results': speed_results,
                'validation_results': validation_results,
                'benchmark_results': benchmark_results
            }
            
            joblib.dump(model_data, output_path / "heuristic_evaluator_model.pkl")
            
            # Save report
            with open(output_path / "heuristic_evaluator_report.txt", 'w') as f:
                f.write(report)
        
        logger.info("Heuristic evaluation function training completed")
        
        return {
            'weights': weights,
            'speed_results': speed_results,
            'validation_results': validation_results,
            'benchmark_results': benchmark_results,
            'report': report
        }


if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run training
    evaluator = HeuristicEvaluator()
    results = evaluator.run_complete_training()
    
    # Print summary
    print(evaluator.generate_evaluation_report())





