"""Performance profiling framework for agent execution metrics."""

from __future__ import annotations

import logging
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .heuristic_agent import HeuristicAgent
    from .tournament import TournamentMetrics
    from ..self_play.quality_metrics import GameQualityMetrics
else:
    HeuristicAgent = Any
    TournamentMetrics = Any
    GameQualityMetrics = Any

logger = logging.getLogger(__name__)


@dataclass
class ProfileMetrics:
    """Performance profiling metrics for an agent."""
    move_times: List[float] = field(default_factory=list)
    """List of move computation times in seconds."""
    
    memory_usage_mb: float = 0.0
    """Peak memory usage in megabytes."""
    
    nodes_per_second: float = 0.0
    """Average nodes evaluated per second."""
    
    game_quality_score: float = 0.0
    """Composite game quality score (0-1)."""
    
    total_games: int = 0
    """Number of games profiled."""
    
    @property
    def avg_move_time(self) -> float:
        """Average move time in seconds."""
        return statistics.mean(self.move_times) if self.move_times else 0.0
    
    @property
    def min_move_time(self) -> float:
        """Minimum move time in seconds."""
        return min(self.move_times) if self.move_times else 0.0
    
    @property
    def max_move_time(self) -> float:
        """Maximum move time in seconds."""
        return max(self.move_times) if self.move_times else 0.0
    
    @property
    def p95_move_time(self) -> float:
        """95th percentile move time in seconds."""
        if not self.move_times:
            return 0.0
        sorted_times = sorted(self.move_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def p99_move_time(self) -> float:
        """99th percentile move time in seconds."""
        if not self.move_times:
            return 0.0
        sorted_times = sorted(self.move_times)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def std_move_time(self) -> float:
        """Standard deviation of move times."""
        if len(self.move_times) < 2:
            return 0.0
        return statistics.stdev(self.move_times)


@dataclass
class RegressionAlert:
    """Alert for performance regression detection."""
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    severity: str  # "low", "medium", "high"
    threshold: float
    
    def __str__(self) -> str:
        direction = "increase" if self.change_percent > 0 else "decrease"
        return (
            f"{self.metric_name}: {direction} of {abs(self.change_percent):.1f}% "
            f"({self.baseline_value:.4f} -> {self.current_value:.4f}) "
            f"[{self.severity} severity]"
        )


class PerformanceProfiler:
    """Performance monitoring and profiling system."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        """Initialize the performance profiler.
        
        Args:
            enable_memory_tracking: Whether to track memory usage (requires tracemalloc)
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.logger = logging.getLogger(__name__)
        self._memory_snapshots: List[float] = []
    
    def profile_agent(
        self,
        agent: HeuristicAgent,
        num_games: int = 10,
        opponent_factory: Optional[callable] = None,
    ) -> ProfileMetrics:
        """Profile an agent's performance over multiple games.
        
        Args:
            agent: Agent to profile
            num_games: Number of games to play
            opponent_factory: Factory function for creating opponent (defaults to random)
            
        Returns:
            Profile metrics
        """
        if self.enable_memory_tracking:
            tracemalloc.start()
        
        move_times: List[float] = []
        total_nodes = 0
        quality_scores: List[float] = []
        
        from ..game.game_state import GameState
        from ..game.constants import Player
        from ..game.moves import MoveGenerator
        
        if opponent_factory is None:
            from ..self_play.random_policy import RandomMovePolicy
            from ..self_play.policies import PolicyConfig
            opponent_factory = lambda: RandomMovePolicy(PolicyConfig())
        
        for game_num in range(num_games):
            state = GameState()
            state.current_player = Player.WHITE if game_num % 2 == 0 else Player.BLACK
            opponent = opponent_factory()
            
            game_move_times: List[float] = []
            turn_count = 0
            
            while turn_count < 200 and not state.is_terminal():
                valid_moves = MoveGenerator.get_valid_moves(state.board, state)
                if not valid_moves:
                    break
                
                if state.current_player == Player.WHITE:
                    start = time.perf_counter()
                    move = agent.select_move(state)
                    elapsed = time.perf_counter() - start
                    game_move_times.append(elapsed)
                    move_times.append(elapsed)
                    
                    # Track nodes evaluated
                    nodes = agent.last_search_stats.get("nodes_evaluated", 0)
                    total_nodes += nodes
                else:
                    move = opponent.select_move(state)
                
                if not state.make_move(move):
                    break
                
                turn_count += 1
                
                # Track memory periodically
                if self.enable_memory_tracking and turn_count % 10 == 0:
                    current, peak = tracemalloc.get_traced_memory()
                    self._memory_snapshots.append(peak / (1024 * 1024))  # Convert to MB
            
            # Calculate game quality (simplified)
            if turn_count > 0:
                avg_move_time = statistics.mean(game_move_times) if game_move_times else 0.0
                # Quality score: inverse of move time (normalized), higher is better
                quality_score = 1.0 / (1.0 + avg_move_time * 10.0)  # Rough normalization
                quality_scores.append(quality_score)
        
        # Calculate final metrics
        total_time = sum(move_times)
        nodes_per_second = total_nodes / total_time if total_time > 0 else 0.0
        
        memory_usage = 0.0
        if self.enable_memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = peak / (1024 * 1024)  # Convert to MB
            tracemalloc.stop()
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        return ProfileMetrics(
            move_times=move_times,
            memory_usage_mb=memory_usage,
            nodes_per_second=nodes_per_second,
            game_quality_score=avg_quality,
            total_games=num_games,
        )
    
    def detect_regressions(
        self,
        current_metrics: ProfileMetrics,
        baseline_metrics: ProfileMetrics,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[RegressionAlert]:
        """Detect performance regressions compared to baseline.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            thresholds: Optional custom thresholds for regression detection
                Keys: metric names, Values: percentage change threshold
            
        Returns:
            List of regression alerts
        """
        if thresholds is None:
            thresholds = {
                "avg_move_time": 20.0,  # 20% increase is concerning
                "p95_move_time": 25.0,
                "p99_move_time": 30.0,
                "memory_usage": 50.0,  # 50% increase in memory
                "nodes_per_second": -10.0,  # 10% decrease in throughput
            }
        
        alerts: List[RegressionAlert] = []
        
        # Check move time metrics
        baseline_avg = baseline_metrics.avg_move_time
        current_avg = current_metrics.avg_move_time
        if baseline_avg > 0:
            change_pct = ((current_avg - baseline_avg) / baseline_avg) * 100
            threshold = thresholds.get("avg_move_time", 20.0)
            if abs(change_pct) > threshold:
                severity = self._determine_severity(abs(change_pct), threshold)
                alerts.append(RegressionAlert(
                    metric_name="avg_move_time",
                    baseline_value=baseline_avg,
                    current_value=current_avg,
                    change_percent=change_pct,
                    severity=severity,
                    threshold=threshold,
                ))
        
        # Check p95 move time
        baseline_p95 = baseline_metrics.p95_move_time
        current_p95 = current_metrics.p95_move_time
        if baseline_p95 > 0:
            change_pct = ((current_p95 - baseline_p95) / baseline_p95) * 100
            threshold = thresholds.get("p95_move_time", 25.0)
            if abs(change_pct) > threshold:
                severity = self._determine_severity(abs(change_pct), threshold)
                alerts.append(RegressionAlert(
                    metric_name="p95_move_time",
                    baseline_value=baseline_p95,
                    current_value=current_p95,
                    change_percent=change_pct,
                    severity=severity,
                    threshold=threshold,
                ))
        
        # Check p99 move time
        baseline_p99 = baseline_metrics.p99_move_time
        current_p99 = current_metrics.p99_move_time
        if baseline_p99 > 0:
            change_pct = ((current_p99 - baseline_p99) / baseline_p99) * 100
            threshold = thresholds.get("p99_move_time", 30.0)
            if abs(change_pct) > threshold:
                severity = self._determine_severity(abs(change_pct), threshold)
                alerts.append(RegressionAlert(
                    metric_name="p99_move_time",
                    baseline_value=baseline_p99,
                    current_value=current_p99,
                    change_percent=change_pct,
                    severity=severity,
                    threshold=threshold,
                ))
        
        # Check memory usage
        if self.enable_memory_tracking:
            baseline_mem = baseline_metrics.memory_usage_mb
            current_mem = current_metrics.memory_usage_mb
            if baseline_mem > 0:
                change_pct = ((current_mem - baseline_mem) / baseline_mem) * 100
                threshold = thresholds.get("memory_usage", 50.0)
                if change_pct > threshold:  # Only alert on increases
                    severity = self._determine_severity(change_pct, threshold)
                    alerts.append(RegressionAlert(
                        metric_name="memory_usage_mb",
                        baseline_value=baseline_mem,
                        current_value=current_mem,
                        change_percent=change_pct,
                        severity=severity,
                        threshold=threshold,
                    ))
        
        # Check throughput (nodes per second)
        baseline_throughput = baseline_metrics.nodes_per_second
        current_throughput = current_metrics.nodes_per_second
        if baseline_throughput > 0:
            change_pct = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
            threshold = thresholds.get("nodes_per_second", -10.0)
            if change_pct < threshold:  # Alert on decreases
                severity = self._determine_severity(abs(change_pct), abs(threshold))
                alerts.append(RegressionAlert(
                    metric_name="nodes_per_second",
                    baseline_value=baseline_throughput,
                    current_value=current_throughput,
                    change_percent=change_pct,
                    severity=severity,
                    threshold=abs(threshold),
                ))
        
        return alerts
    
    def identify_bottlenecks(self, profile_data: ProfileMetrics) -> List[str]:
        """Identify performance bottlenecks from profile data.
        
        Args:
            profile_data: Profile metrics to analyze
            
        Returns:
            List of bottleneck descriptions
        """
        bottlenecks: List[str] = []
        
        # Check for high variance in move times (indicates inconsistent performance)
        if profile_data.std_move_time > profile_data.avg_move_time * 0.5:
            bottlenecks.append(
                f"High variance in move times (std={profile_data.std_move_time:.4f}s, "
                f"avg={profile_data.avg_move_time:.4f}s)"
            )
        
        # Check for slow p99 times
        if profile_data.p99_move_time > profile_data.avg_move_time * 3.0:
            bottlenecks.append(
                f"Slow tail performance: p99={profile_data.p99_move_time:.4f}s "
                f"(avg={profile_data.avg_move_time:.4f}s)"
            )
        
        # Check memory usage
        if profile_data.memory_usage_mb > 500:  # Arbitrary threshold
            bottlenecks.append(
                f"High memory usage: {profile_data.memory_usage_mb:.1f}MB"
            )
        
        # Check throughput
        if profile_data.nodes_per_second < 1000:  # Arbitrary threshold
            bottlenecks.append(
                f"Low throughput: {profile_data.nodes_per_second:.1f} nodes/sec"
            )
        
        return bottlenecks
    
    def generate_performance_report(
        self,
        metrics: ProfileMetrics,
        baseline: Optional[ProfileMetrics] = None,
    ) -> str:
        """Generate a formatted performance report.
        
        Args:
            metrics: Current performance metrics
            baseline: Optional baseline metrics for comparison
            
        Returns:
            Formatted markdown report string
        """
        lines = ["# Performance Profile Report\n"]
        
        # Summary statistics
        lines.append("## Summary Statistics\n")
        lines.append(f"- **Games Profiled**: {metrics.total_games}")
        lines.append(f"- **Total Moves**: {len(metrics.move_times)}")
        lines.append(f"- **Average Move Time**: {metrics.avg_move_time:.4f}s")
        lines.append(f"- **Min Move Time**: {metrics.min_move_time:.4f}s")
        lines.append(f"- **Max Move Time**: {metrics.max_move_time:.4f}s")
        lines.append(f"- **P95 Move Time**: {metrics.p95_move_time:.4f}s")
        lines.append(f"- **P99 Move Time**: {metrics.p99_move_time:.4f}s")
        lines.append(f"- **Std Dev Move Time**: {metrics.std_move_time:.4f}s")
        lines.append(f"- **Memory Usage**: {metrics.memory_usage_mb:.1f}MB")
        lines.append(f"- **Nodes/Second**: {metrics.nodes_per_second:.1f}")
        lines.append(f"- **Game Quality Score**: {metrics.game_quality_score:.3f}")
        
        # Comparison with baseline
        if baseline:
            lines.append("\n## Comparison with Baseline\n")
            lines.append("| Metric | Baseline | Current | Change |")
            lines.append("|--------|----------|---------|--------|")
            
            comparisons = [
                ("Avg Move Time", baseline.avg_move_time, metrics.avg_move_time),
                ("P95 Move Time", baseline.p95_move_time, metrics.p95_move_time),
                ("P99 Move Time", baseline.p99_move_time, metrics.p99_move_time),
                ("Memory Usage (MB)", baseline.memory_usage_mb, metrics.memory_usage_mb),
                ("Nodes/Second", baseline.nodes_per_second, metrics.nodes_per_second),
            ]
            
            for name, base_val, curr_val in comparisons:
                if base_val > 0:
                    change_pct = ((curr_val - base_val) / base_val) * 100
                    change_str = f"{change_pct:+.1f}%"
                else:
                    change_str = "N/A"
                lines.append(f"| {name} | {base_val:.4f} | {curr_val:.4f} | {change_str} |")
            
            # Regression alerts
            alerts = self.detect_regressions(metrics, baseline)
            if alerts:
                lines.append("\n## Regression Alerts\n")
                for alert in alerts:
                    lines.append(f"- **{alert.severity.upper()}**: {alert}")
        
        # Bottleneck identification
        bottlenecks = self.identify_bottlenecks(metrics)
        if bottlenecks:
            lines.append("\n## Identified Bottlenecks\n")
            for bottleneck in bottlenecks:
                lines.append(f"- {bottleneck}")
        
        return "\n".join(lines)
    
    def _determine_severity(self, change_percent: float, threshold: float) -> str:
        """Determine severity level for a regression.
        
        Args:
            change_percent: Percentage change
            threshold: Threshold for alerting
            
        Returns:
            Severity level: "low", "medium", or "high"
        """
        ratio = abs(change_percent) / threshold if threshold > 0 else 0
        if ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"

