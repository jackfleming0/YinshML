"""A/B testing framework for comparing agent configurations."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .heuristic_agent import HeuristicAgentConfig
    from .tournament import TournamentMetrics, TournamentConfig
    from .statistics import StatisticalAnalyzer, SignificanceTest
else:
    HeuristicAgentConfig = Any
    TournamentMetrics = Any
    TournamentConfig = Any
    StatisticalAnalyzer = Any
    SignificanceTest = Any

logger = logging.getLogger(__name__)


@dataclass
class AgentVariant:
    """Configuration for an agent variant in an A/B test."""
    name: str
    """Unique name for this variant."""
    
    config: HeuristicAgentConfig
    """Agent configuration."""
    
    description: str = ""
    """Optional description of the variant."""


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    """Experiment name."""
    
    variants: List[AgentVariant]
    """List of agent variants to test."""
    
    num_games_per_matchup: int = 100
    """Number of games to play for each matchup."""
    
    randomization_seed: Optional[int] = None
    """Random seed for reproducibility."""
    
    tournament_config: Optional[TournamentConfig] = None
    """Optional tournament configuration for large-scale tests."""
    
    output_path: Optional[str] = None
    """Path to save experiment results."""
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if len(self.variants) < 2:
            raise ValueError("At least 2 variants required for A/B testing")
        
        if len(set(v.name for v in self.variants)) != len(self.variants):
            raise ValueError("Variant names must be unique")


@dataclass
class MatchupResult:
    """Results for a single matchup between two variants."""
    variant1_name: str
    variant2_name: str
    variant1_metrics: TournamentMetrics
    variant2_metrics: TournamentMetrics
    significance_test: Optional[SignificanceTest] = None


@dataclass
class ExperimentResults:
    """Complete results from an A/B test experiment."""
    experiment_name: str
    config: ExperimentConfig
    matchup_results: List[MatchupResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "config": asdict(self.config),
            "matchup_results": [
                {
                    "variant1_name": m.variant1_name,
                    "variant2_name": m.variant2_name,
                    "variant1_metrics": m.variant1_metrics.to_dict(),
                    "variant2_metrics": m.variant2_metrics.to_dict(),
                    "significance_test": asdict(m.significance_test) if m.significance_test else None,
                }
                for m in self.matchup_results
            ],
            "timestamp": self.timestamp,
        }


class ABTestRunner:
    """Runner for A/B testing experiments."""
    
    def __init__(self):
        """Initialize the A/B test runner."""
        self.logger = logging.getLogger(__name__)
        from .statistics import StatisticalAnalyzer
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def run_experiment(
        self,
        config: ExperimentConfig,
    ) -> ExperimentResults:
        """Run a complete A/B test experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment results
        """
        self.logger.info(f"Starting A/B test experiment: {config.name}")
        self.logger.info(f"Variants: {[v.name for v in config.variants]}")
        self.logger.info(f"Games per matchup: {config.num_games_per_matchup}")
        
        # Set random seed for reproducibility
        if config.randomization_seed is not None:
            random.seed(config.randomization_seed)
        
        results = ExperimentResults(
            experiment_name=config.name,
            config=config,
        )
        
        # Run all pairwise matchups
        for i, variant1 in enumerate(config.variants):
            for variant2 in config.variants[i+1:]:
                self.logger.info(
                    f"Running matchup: {variant1.name} vs {variant2.name}"
                )
                
                matchup_result = self._run_matchup(
                    variant1,
                    variant2,
                    config,
                )
                results.matchup_results.append(matchup_result)
        
        # Save results if output path specified
        if config.output_path:
            self._save_results(config.output_path, results)
        
        self.logger.info(f"Experiment complete: {config.name}")
        return results
    
    def _run_matchup(
        self,
        variant1: AgentVariant,
        variant2: AgentVariant,
        config: ExperimentConfig,
    ) -> MatchupResult:
        """Run a single matchup between two variants.
        
        Args:
            variant1: First variant
            variant2: Second variant
            config: Experiment configuration
            
        Returns:
            Matchup result
        """
        from .tournament import TournamentEvaluator, TournamentConfig
        
        # Create tournament evaluators for each variant
        evaluator1 = TournamentEvaluator(
            heuristic_config=variant1.config,
        )
        evaluator2 = TournamentEvaluator(
            heuristic_config=variant2.config,
        )
        
        # Use tournament config if provided, otherwise use defaults
        if config.tournament_config:
            tournament_config = config.tournament_config
            tournament_config.num_games = config.num_games_per_matchup
        else:
            tournament_config = TournamentConfig(
                num_games=config.num_games_per_matchup,
                concurrent_workers=2,  # Reduced for pairwise testing
            )
        
        # Run tournaments: variant1 vs random, variant2 vs random
        # Then compare results
        metrics1 = evaluator1.run_tournament(
            games=config.num_games_per_matchup,
        )
        metrics2 = evaluator2.run_tournament(
            games=config.num_games_per_matchup,
        )
        
        # Perform statistical significance test
        significance_test = self.statistical_analyzer.test_significance(
            metrics1,
            metrics2,
        )
        
        return MatchupResult(
            variant1_name=variant1.name,
            variant2_name=variant2.name,
            variant1_metrics=metrics1,
            variant2_metrics=metrics2,
            significance_test=significance_test,
        )
    
    def analyze_results(
        self,
        results: ExperimentResults,
    ) -> Dict[str, Any]:
        """Analyze experiment results and provide interpretation.
        
        Args:
            results: Experiment results
            
        Returns:
            Analysis dictionary with recommendations
        """
        analysis = {
            "experiment_name": results.experiment_name,
            "total_matchups": len(results.matchup_results),
            "significant_differences": [],
            "recommendations": [],
        }
        
        # Find significant differences
        for matchup in results.matchup_results:
            if matchup.significance_test and matchup.significance_test.significant:
                analysis["significant_differences"].append({
                    "variant1": matchup.variant1_name,
                    "variant2": matchup.variant2_name,
                    "variant1_win_rate": matchup.variant1_metrics.win_rate,
                    "variant2_win_rate": matchup.variant2_metrics.win_rate,
                    "p_value": matchup.significance_test.p_value,
                })
        
        # Generate recommendations
        if not analysis["significant_differences"]:
            analysis["recommendations"].append(
                "No significant differences detected. Consider increasing sample size."
            )
        else:
            # Find best performing variant
            best_variant = None
            best_win_rate = -1.0
            
            for matchup in results.matchup_results:
                if matchup.variant1_metrics.win_rate > best_win_rate:
                    best_win_rate = matchup.variant1_metrics.win_rate
                    best_variant = matchup.variant1_name
                if matchup.variant2_metrics.win_rate > best_win_rate:
                    best_win_rate = matchup.variant2_metrics.win_rate
                    best_variant = matchup.variant2_name
            
            if best_variant:
                analysis["recommendations"].append(
                    f"Best performing variant: {best_variant} "
                    f"(win rate: {best_win_rate:.3f})"
                )
        
        return analysis
    
    def generate_dashboard(
        self,
        results: ExperimentResults,
    ) -> str:
        """Generate a formatted dashboard report.
        
        Args:
            results: Experiment results
            
        Returns:
            Formatted markdown dashboard
        """
        lines = [f"# A/B Test Dashboard: {results.experiment_name}\n"]
        lines.append(f"**Experiment Date**: {time.ctime(results.timestamp)}\n")
        
        # Variant summary
        lines.append("## Variants\n")
        for variant in results.config.variants:
            lines.append(f"- **{variant.name}**: {variant.description or 'No description'}")
        lines.append("")
        
        # Matchup results table
        lines.append("## Matchup Results\n")
        lines.append(
            "| Variant 1 | Variant 2 | V1 Win Rate | V2 Win Rate | "
            "Significant | p-value |"
        )
        lines.append("|------------|-----------|-------------|-------------|-------------|---------|")
        
        for matchup in results.matchup_results:
            significant = "Yes" if (
                matchup.significance_test and matchup.significance_test.significant
            ) else "No"
            p_value = (
                matchup.significance_test.p_value
                if matchup.significance_test
                else 1.0
            )
            lines.append(
                f"| {matchup.variant1_name} | {matchup.variant2_name} | "
                f"{matchup.variant1_metrics.win_rate:.3f} | "
                f"{matchup.variant2_metrics.win_rate:.3f} | "
                f"{significant} | {p_value:.4f} |"
            )
        
        # Analysis
        analysis = self.analyze_results(results)
        lines.append("\n## Analysis\n")
        
        if analysis["significant_differences"]:
            lines.append("### Significant Differences\n")
            for diff in analysis["significant_differences"]:
                lines.append(
                    f"- **{diff['variant1']} vs {diff['variant2']}**: "
                    f"Win rates {diff['variant1_win_rate']:.3f} vs "
                    f"{diff['variant2_win_rate']:.3f} (p={diff['p_value']:.4f})"
                )
        else:
            lines.append("No significant differences detected.\n")
        
        lines.append("\n### Recommendations\n")
        for rec in analysis["recommendations"]:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def calculate_statistical_power(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """Calculate required sample size for statistical power.
        
        Args:
            effect_size: Minimum detectable effect size (e.g., 0.05 for 5% difference)
            alpha: Significance level (default 0.05)
            power: Desired statistical power (default 0.80)
            
        Returns:
            Required sample size per group
        
        Note:
            Simplified calculation using normal approximation.
            For more accurate results, consider using scipy.stats.
        """
        # Z-scores
        z_alpha = 1.96  # For alpha=0.05 (two-tailed)
        z_power = 0.84  # For power=0.80
        
        # Simplified sample size calculation for two-proportion test
        p1 = 0.5  # Baseline win rate
        p2 = p1 + effect_size
        
        p_pooled = (p1 + p2) / 2
        
        numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) +
                     z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        denominator = (p2 - p1) ** 2
        
        n = numerator / denominator if denominator > 0 else 1000
        return int(math.ceil(n))
    
    def _save_results(
        self,
        output_path: str,
        results: ExperimentResults,
    ) -> None:
        """Save experiment results to file.
        
        Args:
            output_path: Path to save results
            results: Experiment results to save
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w") as f:
                json.dump(results.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved experiment results to {output_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save experiment results: {e}")

