"""Statistical analysis tools for tournament results and performance metrics."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .tournament import TournamentMetrics
else:
    TournamentMetrics = Any

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a statistic."""
    lower: float
    upper: float
    confidence: float  # e.g., 0.95 for 95% confidence
    
    def __str__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}] @ {self.confidence*100:.1f}%"


@dataclass
class SignificanceTest:
    """Results of a statistical significance test."""
    p_value: float
    significant: bool
    test_name: str
    statistic: float
    alpha: float = 0.05
    
    def __str__(self) -> str:
        status = "significant" if self.significant else "not significant"
        return f"{self.test_name}: p={self.p_value:.4f}, {status} (α={self.alpha})"


@dataclass
class ELORating:
    """ELO rating for an agent."""
    agent_name: str
    rating: float
    games_played: int
    wins: int
    losses: int
    draws: int
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for tournament results."""
    
    def __init__(self):
        """Initialize the statistical analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def compute_confidence_interval(
        self,
        win_rate: float,
        total_games: int,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Compute confidence interval for win rate using Wilson score method.
        
        Args:
            win_rate: Observed win rate (0-1)
            total_games: Total number of games played
            confidence: Confidence level (default 0.95 for 95%)
            
        Returns:
            Confidence interval
            
        Note:
            Uses Wilson score interval which is more accurate for small samples
            than normal approximation.
        """
        if total_games == 0:
            return ConfidenceInterval(0.0, 1.0, confidence)
        
        if win_rate == 0.0:
            # Special case: no wins
            z = self._z_score(confidence)
            denominator = 1 + (z**2 / total_games)
            lower = 0.0
            upper = (z**2 / total_games) / denominator
            return ConfidenceInterval(lower, upper, confidence)
        
        if win_rate == 1.0:
            # Special case: all wins
            z = self._z_score(confidence)
            denominator = 1 + (z**2 / total_games)
            lower = 1 / denominator
            upper = 1.0
            return ConfidenceInterval(lower, upper, confidence)
        
        # Wilson score interval
        z = self._z_score(confidence)
        n = total_games
        p = win_rate
        
        denominator = 1 + (z**2 / n)
        center = (p + (z**2 / (2 * n))) / denominator
        margin = (z / denominator) * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return ConfidenceInterval(lower, upper, confidence)
    
    def test_significance(
        self,
        baseline_metrics: TournamentMetrics,
        test_metrics: TournamentMetrics,
        alpha: float = 0.05,
    ) -> SignificanceTest:
        """Test statistical significance of difference between two metrics.
        
        Args:
            baseline_metrics: Baseline tournament metrics
            test_metrics: Test tournament metrics
            alpha: Significance level (default 0.05)
            
        Returns:
            Significance test results
        """
        # Use chi-square test for win/loss/draw comparison
        return self._chi_square_test(baseline_metrics, test_metrics, alpha)
    
    def _chi_square_test(
        self,
        baseline: TournamentMetrics,
        test: TournamentMetrics,
        alpha: float,
    ) -> SignificanceTest:
        """Perform chi-square test for independence.
        
        Args:
            baseline: Baseline metrics
            test: Test metrics
            alpha: Significance level
            
        Returns:
            Significance test results
        """
        # Create contingency table: [wins, losses, draws]
        baseline_counts = [baseline.wins, baseline.losses, baseline.draws]
        test_counts = [test.wins, test.losses, test.draws]
        
        total_baseline = sum(baseline_counts)
        total_test = sum(test_counts)
        total_all = total_baseline + total_test
        
        if total_all == 0:
            return SignificanceTest(
                p_value=1.0,
                significant=False,
                test_name="chi-square",
                statistic=0.0,
                alpha=alpha,
            )
        
        # Calculate expected frequencies
        wins_total = baseline.wins + test.wins
        losses_total = baseline.losses + test.losses
        draws_total = baseline.draws + test.draws
        
        expected_baseline = [
            (wins_total * total_baseline) / total_all,
            (losses_total * total_baseline) / total_all,
            (draws_total * total_baseline) / total_all,
        ]
        expected_test = [
            (wins_total * total_test) / total_all,
            (losses_total * total_test) / total_all,
            (draws_total * total_test) / total_all,
        ]
        
        # Calculate chi-square statistic
        chi_square = 0.0
        observed = baseline_counts + test_counts
        expected = expected_baseline + expected_test
        
        for obs, exp in zip(observed, expected):
            if exp > 0:
                chi_square += ((obs - exp) ** 2) / exp
        
        # Degrees of freedom: (rows - 1) * (cols - 1) = (2 - 1) * (3 - 1) = 2
        df = 2
        p_value = self._chi_square_p_value(chi_square, df)
        significant = p_value < alpha
        
        return SignificanceTest(
            p_value=p_value,
            significant=significant,
            test_name="chi-square",
            statistic=chi_square,
            alpha=alpha,
        )
    
    def calculate_elo_ratings(
        self,
        match_results: List[Dict[str, Any]],
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
    ) -> List[ELORating]:
        """Calculate ELO ratings from match results.
        
        Args:
            match_results: List of match result dictionaries with keys:
                - 'agent1': name of first agent
                - 'agent2': name of second agent
                - 'winner': 'agent1', 'agent2', or 'draw'
            initial_rating: Starting ELO rating (default 1500)
            k_factor: K-factor for rating updates (default 32)
            
        Returns:
            List of ELO ratings for each agent
        """
        # Initialize ratings
        ratings: Dict[str, float] = {}
        stats: Dict[str, Dict[str, int]] = {}
        
        for match in match_results:
            agent1 = match.get("agent1", "agent1")
            agent2 = match.get("agent2", "agent2")
            winner = match.get("winner", "draw")
            
            # Initialize if needed
            if agent1 not in ratings:
                ratings[agent1] = initial_rating
                stats[agent1] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            if agent2 not in ratings:
                ratings[agent2] = initial_rating
                stats[agent2] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
            
            # Calculate expected scores
            expected1 = self._expected_score(ratings[agent1], ratings[agent2])
            expected2 = 1.0 - expected1
            
            # Determine actual scores
            if winner == "agent1":
                actual1, actual2 = 1.0, 0.0
                stats[agent1]["wins"] += 1
                stats[agent2]["losses"] += 1
            elif winner == "agent2":
                actual1, actual2 = 0.0, 1.0
                stats[agent1]["losses"] += 1
                stats[agent2]["wins"] += 1
            else:  # draw
                actual1, actual2 = 0.5, 0.5
                stats[agent1]["draws"] += 1
                stats[agent2]["draws"] += 1
            
            stats[agent1]["games"] += 1
            stats[agent2]["games"] += 1
            
            # Update ratings
            ratings[agent1] += k_factor * (actual1 - expected1)
            ratings[agent2] += k_factor * (actual2 - expected2)
        
        # Build ELO rating objects
        elo_ratings = []
        for agent_name, rating in ratings.items():
            agent_stats = stats[agent_name]
            elo_ratings.append(ELORating(
                agent_name=agent_name,
                rating=rating,
                games_played=agent_stats["games"],
                wins=agent_stats["wins"],
                losses=agent_stats["losses"],
                draws=agent_stats["draws"],
            ))
        
        # Sort by rating (highest first)
        elo_ratings.sort(key=lambda x: x.rating, reverse=True)
        return elo_ratings
    
    def generate_performance_report(
        self,
        metrics_list: List[TournamentMetrics],
        agent_names: Optional[List[str]] = None,
        confidence: float = 0.95,
    ) -> str:
        """Generate a formatted performance report.
        
        Args:
            metrics_list: List of tournament metrics for different agents/configs
            agent_names: Optional list of agent names (defaults to "Agent 1", "Agent 2", etc.)
            confidence: Confidence level for intervals
            
        Returns:
            Formatted markdown report string
        """
        if not metrics_list:
            return "No metrics provided."
        
        if agent_names is None:
            agent_names = [f"Agent {i+1}" for i in range(len(metrics_list))]
        
        lines = ["# Tournament Performance Report\n"]
        lines.append(f"Confidence Level: {confidence*100:.1f}%\n")
        
        # Summary table
        lines.append("## Summary\n")
        lines.append("| Agent | Games | Win Rate | CI Lower | CI Upper | Avg Length |")
        lines.append("|-------|-------|----------|----------|----------|------------|")
        
        for name, metrics in zip(agent_names, metrics_list):
            ci = self.compute_confidence_interval(
                metrics.win_rate,
                metrics.total_games,
                confidence,
            )
            lines.append(
                f"| {name} | {metrics.total_games} | "
                f"{metrics.win_rate:.3f} | {ci.lower:.3f} | {ci.upper:.3f} | "
                f"{metrics.average_game_length:.1f} |"
            )
        
        # Statistical comparisons
        if len(metrics_list) >= 2:
            lines.append("\n## Statistical Comparisons\n")
            baseline = metrics_list[0]
            for i, (name, metrics) in enumerate(zip(agent_names[1:], metrics_list[1:]), 1):
                test = self.test_significance(baseline, metrics)
                lines.append(f"### {agent_names[0]} vs {name}")
                lines.append(f"- **Test**: {test.test_name}")
                lines.append(f"- **Statistic**: {test.statistic:.4f}")
                lines.append(f"- **p-value**: {test.p_value:.4f}")
                lines.append(f"- **Significant**: {'Yes' if test.significant else 'No'}")
                lines.append("")
        
        # ELO ratings (if we have pairwise match data)
        # This would require match-level data, so we'll skip for now
        
        return "\n".join(lines)
    
    def _z_score(self, confidence: float) -> float:
        """Get z-score for given confidence level.
        
        Args:
            confidence: Confidence level (e.g., 0.95)
            
        Returns:
            Z-score
        """
        # Common z-scores
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
        }
        return z_scores.get(confidence, 1.960)  # Default to 95%
    
    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score for ELO rating.
        
        Args:
            rating1: Rating of first agent
            rating2: Rating of second agent
            
        Returns:
            Expected score (0-1) for first agent
        """
        return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
    
    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Approximate p-value for chi-square statistic.
        
        Args:
            chi_square: Chi-square statistic
            df: Degrees of freedom
            
        Returns:
            Approximate p-value
        
        Note:
            This is a simplified approximation. For production use,
            consider using scipy.stats.chi2 for more accurate values.
        """
        # Simple approximation using critical values
        # For df=2: critical values are approximately 5.991 (α=0.05), 9.210 (α=0.01)
        if df == 2:
            if chi_square < 5.991:
                return 0.05 + (chi_square / 5.991) * 0.05  # Rough interpolation
            elif chi_square < 9.210:
                return 0.01 + ((chi_square - 5.991) / (9.210 - 5.991)) * 0.04
            else:
                return max(0.001, 0.01 * (9.210 / chi_square) ** 2)
        
        # Fallback: very rough approximation
        return max(0.001, min(1.0, math.exp(-chi_square / (2 * df))))

