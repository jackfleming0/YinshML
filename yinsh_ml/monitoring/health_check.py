"""
Memory Management Health Check System

This module provides comprehensive health assessment for memory pools,
allocation patterns, and system performance.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .metrics import MemoryMetrics, MemoryMetricsCollector

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    score: float  # 0.0 to 1.0, where 1.0 is perfect health
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: float


class MemoryHealthChecker:
    """Comprehensive health checker for memory management system."""
    
    def __init__(self, metrics_collector: MemoryMetricsCollector):
        self.metrics_collector = metrics_collector
        
        # Health check thresholds
        self.thresholds = {
            'pool_utilization_warning': 75.0,
            'pool_utilization_critical': 90.0,
            'hit_rate_warning': 60.0,
            'hit_rate_critical': 40.0,
            'fragmentation_warning': 0.2,
            'fragmentation_critical': 0.35,
            'latency_warning': 500.0,  # microseconds
            'latency_critical': 2000.0,
            'memory_pressure_warning': 0.8,
            'memory_pressure_critical': 0.9,
            'allocation_rate_variance_warning': 0.3,
            'allocation_rate_variance_critical': 0.5
        }
    
    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        current_metrics = self.metrics_collector.get_current_metrics()
        history = self.metrics_collector.get_metrics_history(3600)  # Last hour
        
        # Individual health checks
        checks = [
            self._check_pool_health(current_metrics),
            self._check_allocation_performance(current_metrics, history),
            self._check_memory_fragmentation(current_metrics, history),
            self._check_system_resources(current_metrics),
            self._check_allocation_patterns(history),
            self._check_trend_analysis(history)
        ]
        
        # Calculate overall health
        overall_status, overall_score = self._calculate_overall_health(checks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks, current_metrics)
        
        return {
            'timestamp': time.time(),
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'individual_checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'score': check.score,
                    'message': check.message,
                    'details': check.details,
                    'recommendations': check.recommendations
                }
                for check in checks
            ],
            'summary_recommendations': recommendations,
            'metrics_summary': self._get_metrics_summary(current_metrics)
        }
    
    def _check_pool_health(self, metrics: MemoryMetrics) -> HealthCheckResult:
        """Check health of memory pools."""
        details = {}
        recommendations = []
        issues = []
        total_score = 0.0
        pool_count = 0
        
        for pool_name in metrics.pool_utilization.keys():
            pool_count += 1
            utilization = metrics.pool_utilization.get(pool_name, 0)
            hit_rate = metrics.pool_hit_rates.get(pool_name, 0)
            current_size = metrics.pool_sizes.get(pool_name, 0)
            max_size = metrics.pool_max_sizes.get(pool_name, 0)
            
            # Calculate pool score
            util_score = self._score_utilization(utilization)
            hit_score = self._score_hit_rate(hit_rate)
            pool_score = (util_score + hit_score) / 2
            total_score += pool_score
            
            details[f'{pool_name}_utilization'] = utilization
            details[f'{pool_name}_hit_rate'] = hit_rate
            details[f'{pool_name}_size'] = current_size
            details[f'{pool_name}_score'] = pool_score
            
            # Check for issues
            if utilization >= self.thresholds['pool_utilization_critical']:
                issues.append(f"{pool_name} pool critically full ({utilization:.1f}%)")
                recommendations.append(f"Increase {pool_name} pool initial size")
            elif utilization >= self.thresholds['pool_utilization_warning']:
                issues.append(f"{pool_name} pool utilization high ({utilization:.1f}%)")
                recommendations.append(f"Monitor {pool_name} pool growth")
            
            if hit_rate <= self.thresholds['hit_rate_critical']:
                issues.append(f"{pool_name} pool hit rate critically low ({hit_rate:.1f}%)")
                recommendations.append(f"Increase {pool_name} pool size or review allocation patterns")
            elif hit_rate <= self.thresholds['hit_rate_warning']:
                issues.append(f"{pool_name} pool hit rate low ({hit_rate:.1f}%)")
                recommendations.append(f"Consider tuning {pool_name} pool configuration")
        
        # Calculate overall pool health
        avg_score = total_score / pool_count if pool_count > 0 else 0.0
        
        if not issues:
            status = HealthStatus.HEALTHY
            message = f"All {pool_count} memory pools are healthy"
        elif any("critically" in issue for issue in issues):
            status = HealthStatus.CRITICAL
            message = f"Critical issues detected in memory pools: {'; '.join(issues[:2])}"
        else:
            status = HealthStatus.WARNING
            message = f"Warning issues detected in memory pools: {'; '.join(issues[:2])}"
        
        return HealthCheckResult(
            name="Pool Health",
            status=status,
            score=avg_score,
            message=message,
            details=details,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _check_allocation_performance(self, metrics: MemoryMetrics, 
                                    history: List[MemoryMetrics]) -> HealthCheckResult:
        """Check allocation performance metrics."""
        details = {
            'p50_latency': metrics.allocation_latency_p50,
            'p95_latency': metrics.allocation_latency_p95,
            'p99_latency': metrics.allocation_latency_p99,
            'allocations_per_second': metrics.allocations_per_second
        }
        
        recommendations = []
        issues = []
        
        # Check latency
        p95_latency = metrics.allocation_latency_p95
        if p95_latency >= self.thresholds['latency_critical']:
            issues.append(f"Critical allocation latency: {p95_latency:.0f}μs (P95)")
            recommendations.append("Investigate memory pool sizing and fragmentation")
            status = HealthStatus.CRITICAL
            latency_score = 0.2
        elif p95_latency >= self.thresholds['latency_warning']:
            issues.append(f"High allocation latency: {p95_latency:.0f}μs (P95)")
            recommendations.append("Consider increasing pool sizes")
            status = HealthStatus.WARNING
            latency_score = 0.6
        else:
            latency_score = 1.0 - (p95_latency / self.thresholds['latency_warning'])
            latency_score = max(0.0, min(1.0, latency_score))
        
        # Check allocation rate stability
        if len(history) >= 10:
            rates = [m.allocations_per_second for m in history[-10:]]
            avg_rate = sum(rates) / len(rates)
            variance = sum((r - avg_rate) ** 2 for r in rates) / len(rates)
            cv = (variance ** 0.5) / avg_rate if avg_rate > 0 else 0
            
            details['allocation_rate_cv'] = cv
            
            if cv >= self.thresholds['allocation_rate_variance_critical']:
                issues.append(f"Highly unstable allocation rate (CV: {cv:.2f})")
                recommendations.append("Investigate allocation pattern irregularities")
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                rate_score = 0.3
            elif cv >= self.thresholds['allocation_rate_variance_warning']:
                issues.append(f"Unstable allocation rate (CV: {cv:.2f})")
                recommendations.append("Monitor allocation patterns")
                rate_score = 0.7
            else:
                rate_score = 1.0
        else:
            rate_score = 0.8  # Neutral score for insufficient data
        
        # Overall performance score
        performance_score = (latency_score + rate_score) / 2
        
        if not issues:
            status = HealthStatus.HEALTHY
            message = "Allocation performance is optimal"
        else:
            message = f"Performance issues detected: {'; '.join(issues[:2])}"
        
        return HealthCheckResult(
            name="Allocation Performance",
            status=status,
            score=performance_score,
            message=message,
            details=details,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _check_memory_fragmentation(self, metrics: MemoryMetrics, 
                                  history: List[MemoryMetrics]) -> HealthCheckResult:
        """Check memory fragmentation levels."""
        fragmentation = metrics.fragmentation_index
        details = {'fragmentation_index': fragmentation}
        recommendations = []
        
        # Score fragmentation (lower is better)
        if fragmentation >= self.thresholds['fragmentation_critical']:
            status = HealthStatus.CRITICAL
            message = f"Critical memory fragmentation: {fragmentation:.2f}"
            recommendations.extend([
                "Consider pool defragmentation",
                "Review allocation patterns",
                "Increase pool sizes to reduce pressure"
            ])
            score = 0.2
        elif fragmentation >= self.thresholds['fragmentation_warning']:
            status = HealthStatus.WARNING
            message = f"High memory fragmentation: {fragmentation:.2f}"
            recommendations.extend([
                "Monitor fragmentation trends",
                "Consider pool tuning"
            ])
            score = 0.6
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory fragmentation is acceptable: {fragmentation:.2f}"
            score = 1.0 - (fragmentation / self.thresholds['fragmentation_warning'])
            score = max(0.0, min(1.0, score))
        
        # Check fragmentation trend
        if len(history) >= 5:
            recent_frag = [m.fragmentation_index for m in history[-5:]]
            trend = (recent_frag[-1] - recent_frag[0]) / len(recent_frag)
            details['fragmentation_trend'] = trend
            
            if trend > 0.02:  # Increasing fragmentation
                recommendations.append("Fragmentation is increasing - investigate allocation patterns")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
        
        return HealthCheckResult(
            name="Memory Fragmentation",
            status=status,
            score=score,
            message=message,
            details=details,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _check_system_resources(self, metrics: MemoryMetrics) -> HealthCheckResult:
        """Check system resource utilization."""
        memory_pressure = metrics.memory_pressure
        cpu_usage = metrics.cpu_usage
        system_memory = metrics.system_memory_usage
        
        details = {
            'memory_pressure': memory_pressure,
            'cpu_usage': cpu_usage,
            'system_memory_usage': system_memory
        }
        
        recommendations = []
        issues = []
        
        # Check memory pressure
        if memory_pressure >= self.thresholds['memory_pressure_critical']:
            issues.append(f"Critical memory pressure: {memory_pressure:.2f}")
            recommendations.append("Reduce memory usage or add more RAM")
            status = HealthStatus.CRITICAL
            memory_score = 0.1
        elif memory_pressure >= self.thresholds['memory_pressure_warning']:
            issues.append(f"High memory pressure: {memory_pressure:.2f}")
            recommendations.append("Monitor memory usage closely")
            status = HealthStatus.WARNING
            memory_score = 0.5
        else:
            memory_score = 1.0 - memory_pressure
            status = HealthStatus.HEALTHY
        
        # Check CPU usage
        if cpu_usage >= 90.0:
            issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            recommendations.append("Investigate CPU-intensive operations")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            cpu_score = 0.3
        elif cpu_usage >= 70.0:
            cpu_score = 0.7
        else:
            cpu_score = 1.0 - (cpu_usage / 100.0)
        
        # Overall system score
        system_score = (memory_score + cpu_score) / 2
        
        if not issues:
            message = "System resources are healthy"
        else:
            message = f"System resource issues: {'; '.join(issues)}"
        
        return HealthCheckResult(
            name="System Resources",
            status=status,
            score=system_score,
            message=message,
            details=details,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _check_allocation_patterns(self, history: List[MemoryMetrics]) -> HealthCheckResult:
        """Analyze allocation patterns for anomalies."""
        if len(history) < 10:
            return HealthCheckResult(
                name="Allocation Patterns",
                status=HealthStatus.UNKNOWN,
                score=0.5,
                message="Insufficient data for pattern analysis",
                details={'sample_count': len(history)},
                recommendations=["Collect more metrics data"],
                timestamp=time.time()
            )
        
        # Analyze allocation rates
        rates = [m.allocations_per_second for m in history]
        avg_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        min_rate = min(rates)
        
        details = {
            'avg_allocation_rate': avg_rate,
            'max_allocation_rate': max_rate,
            'min_allocation_rate': min_rate,
            'rate_range': max_rate - min_rate
        }
        
        recommendations = []
        
        # Check for rate spikes
        spike_threshold = avg_rate * 3
        spikes = [r for r in rates if r > spike_threshold]
        
        if spikes:
            details['spike_count'] = len(spikes)
            recommendations.append(f"Investigate {len(spikes)} allocation rate spikes")
            status = HealthStatus.WARNING
            score = 0.6
            message = f"Detected {len(spikes)} allocation rate anomalies"
        else:
            status = HealthStatus.HEALTHY
            score = 0.9
            message = "Allocation patterns are stable"
        
        return HealthCheckResult(
            name="Allocation Patterns",
            status=status,
            score=score,
            message=message,
            details=details,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _check_trend_analysis(self, history: List[MemoryMetrics]) -> HealthCheckResult:
        """Analyze trends in key metrics."""
        if len(history) < 20:
            return HealthCheckResult(
                name="Trend Analysis",
                status=HealthStatus.UNKNOWN,
                score=0.5,
                message="Insufficient data for trend analysis",
                details={'sample_count': len(history)},
                recommendations=["Collect more historical data"],
                timestamp=time.time()
            )
        
        # Analyze trends in key metrics
        recent = history[-10:]
        older = history[-20:-10]
        
        def calculate_trend(recent_values, older_values):
            recent_avg = sum(recent_values) / len(recent_values)
            older_avg = sum(older_values) / len(older_values)
            return (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
        
        # Memory pressure trend
        memory_trend = calculate_trend(
            [m.memory_pressure for m in recent],
            [m.memory_pressure for m in older]
        )
        
        # Fragmentation trend
        frag_trend = calculate_trend(
            [m.fragmentation_index for m in recent],
            [m.fragmentation_index for m in older]
        )
        
        # Latency trend
        latency_trend = calculate_trend(
            [m.allocation_latency_p95 for m in recent],
            [m.allocation_latency_p95 for m in older]
        )
        
        details = {
            'memory_pressure_trend': memory_trend,
            'fragmentation_trend': frag_trend,
            'latency_trend': latency_trend
        }
        
        recommendations = []
        issues = []
        
        # Check concerning trends
        if memory_trend > 0.1:
            issues.append("Memory pressure increasing")
            recommendations.append("Investigate memory usage growth")
        
        if frag_trend > 0.05:
            issues.append("Fragmentation increasing")
            recommendations.append("Review allocation patterns")
        
        if latency_trend > 0.2:
            issues.append("Allocation latency increasing")
            recommendations.append("Investigate performance degradation")
        
        if issues:
            status = HealthStatus.WARNING
            score = 0.6
            message = f"Concerning trends detected: {'; '.join(issues)}"
        else:
            status = HealthStatus.HEALTHY
            score = 0.9
            message = "All trends are stable or improving"
        
        return HealthCheckResult(
            name="Trend Analysis",
            status=status,
            score=score,
            message=message,
            details=details,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _score_utilization(self, utilization: float) -> float:
        """Score pool utilization (optimal around 60-80%)."""
        if 60 <= utilization <= 80:
            return 1.0
        elif utilization < 60:
            return 0.7 + (utilization / 60) * 0.3
        else:  # utilization > 80
            return max(0.0, 1.0 - ((utilization - 80) / 20))
    
    def _score_hit_rate(self, hit_rate: float) -> float:
        """Score pool hit rate (higher is better)."""
        return min(1.0, hit_rate / 90.0)  # 90% hit rate = perfect score
    
    def _calculate_overall_health(self, checks: List[HealthCheckResult]) -> Tuple[HealthStatus, float]:
        """Calculate overall health status and score."""
        if not checks:
            return HealthStatus.UNKNOWN, 0.0
        
        # Count status levels
        critical_count = sum(1 for c in checks if c.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for c in checks if c.status == HealthStatus.WARNING)
        
        # Calculate weighted average score
        total_score = sum(c.score for c in checks)
        avg_score = total_score / len(checks)
        
        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return overall_status, avg_score
    
    def _generate_recommendations(self, checks: List[HealthCheckResult], 
                                metrics: MemoryMetrics) -> List[str]:
        """Generate high-level recommendations based on all checks."""
        all_recommendations = []
        for check in checks:
            all_recommendations.extend(check.recommendations)
        
        # Deduplicate and prioritize
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        # Add general recommendations based on overall state
        if any(c.status == HealthStatus.CRITICAL for c in checks):
            unique_recommendations.insert(0, "Immediate attention required - critical issues detected")
        
        return unique_recommendations[:10]  # Limit to top 10
    
    def _get_metrics_summary(self, metrics: MemoryMetrics) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            'timestamp': metrics.timestamp,
            'total_pools': len(metrics.pool_utilization),
            'avg_pool_utilization': sum(metrics.pool_utilization.values()) / len(metrics.pool_utilization) if metrics.pool_utilization else 0,
            'avg_hit_rate': sum(metrics.pool_hit_rates.values()) / len(metrics.pool_hit_rates) if metrics.pool_hit_rates else 0,
            'allocation_latency_p95': metrics.allocation_latency_p95,
            'fragmentation_index': metrics.fragmentation_index,
            'memory_pressure': metrics.memory_pressure,
            'system_memory_usage': metrics.system_memory_usage
        } 