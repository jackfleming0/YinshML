"""
Memory Management Alerting System

This module provides configurable alerting for memory management metrics,
including threshold-based alerts, trend analysis, and notification delivery.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

from .metrics import MemoryMetrics, MemoryMetricsCollector

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Types of alert conditions."""
    THRESHOLD = "threshold"           # Value exceeds threshold
    TREND = "trend"                  # Value trending up/down
    RATE_OF_CHANGE = "rate_of_change" # Rate of change exceeds limit
    ANOMALY = "anomaly"              # Statistical anomaly detection


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    metric_path: str                 # e.g., "pool_utilization.game_state"
    condition: AlertCondition
    severity: AlertSeverity
    
    # Threshold conditions
    threshold_value: Optional[float] = None
    threshold_operator: str = ">"    # >, <, >=, <=, ==, !=
    
    # Trend conditions
    trend_window_seconds: float = 300.0  # 5 minutes
    trend_threshold: float = 0.1         # 10% change
    
    # Rate of change conditions
    rate_window_seconds: float = 60.0    # 1 minute
    rate_threshold: float = 0.05         # 5% per minute
    
    # Anomaly detection
    anomaly_sensitivity: float = 2.0     # Standard deviations
    anomaly_window_size: int = 100       # Historical samples
    
    # Alert management
    cooldown_seconds: float = 300.0      # 5 minutes between same alerts
    enabled: bool = True
    description: str = ""
    
    # Custom evaluation function
    custom_evaluator: Optional[Callable[[MemoryMetrics], bool]] = None
    
    # Internal state
    last_triggered: float = field(default=0.0, init=False)
    trigger_count: int = field(default=0, init=False)


@dataclass
class Alert:
    """An active alert instance."""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_value: Any
    threshold_value: Optional[float] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp,
            'metric_value': self.metric_value,
            'threshold_value': self.threshold_value,
            'additional_context': self.additional_context
        }


class AlertManager:
    """Manages alert rules and notifications."""
    
    def __init__(self, metrics_collector: MemoryMetricsCollector):
        self.metrics_collector = metrics_collector
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Notification handlers
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        # Historical data for trend/anomaly detection
        self.metric_history: Dict[str, deque] = {}
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self.lock:
            self.rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
            return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alert rule."""
        with self.lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = True
                logger.info(f"Enabled alert rule: {rule_name}")
                return True
            return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alert rule."""
        with self.lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = False
                logger.info(f"Disabled alert rule: {rule_name}")
                return True
            return False
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler function."""
        with self.lock:
            self.notification_handlers.append(handler)
            logger.info("Added notification handler")
    
    def start_monitoring(self, check_interval: float = 30.0) -> None:
        """Start alert monitoring."""
        if self.running:
            logger.warning("Alert monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            name="AlertMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started alert monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped alert monitoring")
    
    def _monitoring_loop(self, check_interval: float) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                time.sleep(1.0)
    
    def _check_alerts(self) -> None:
        """Check all alert rules against current metrics."""
        current_metrics = self.metrics_collector.get_current_metrics()
        current_time = time.time()
        
        with self.lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Check cooldown
                if current_time - rule.last_triggered < rule.cooldown_seconds:
                    continue
                
                # Evaluate rule
                try:
                    should_alert, alert_message, metric_value = self._evaluate_rule(rule, current_metrics)
                    
                    if should_alert:
                        alert = Alert(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=alert_message,
                            timestamp=current_time,
                            metric_value=metric_value,
                            threshold_value=rule.threshold_value,
                            additional_context=self._get_alert_context(rule, current_metrics)
                        )
                        
                        self._trigger_alert(rule, alert)
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, metrics: MemoryMetrics) -> tuple:
        """Evaluate a single alert rule."""
        # Custom evaluator takes precedence
        if rule.custom_evaluator:
            try:
                result = rule.custom_evaluator(metrics)
                return result, f"Custom rule {rule.name} triggered", "custom"
            except Exception as e:
                logger.error(f"Error in custom evaluator for {rule.name}: {e}")
                return False, "", None
        
        # Get metric value
        metric_value = self._get_metric_value(metrics, rule.metric_path)
        if metric_value is None:
            return False, "", None
        
        # Store historical data
        if rule.metric_path not in self.metric_history:
            self.metric_history[rule.metric_path] = deque(maxlen=rule.anomaly_window_size)
        self.metric_history[rule.metric_path].append((time.time(), metric_value))
        
        # Evaluate based on condition type
        if rule.condition == AlertCondition.THRESHOLD:
            return self._evaluate_threshold(rule, metric_value)
        elif rule.condition == AlertCondition.TREND:
            return self._evaluate_trend(rule, metric_value)
        elif rule.condition == AlertCondition.RATE_OF_CHANGE:
            return self._evaluate_rate_of_change(rule, metric_value)
        elif rule.condition == AlertCondition.ANOMALY:
            return self._evaluate_anomaly(rule, metric_value)
        
        return False, "", metric_value
    
    def _get_metric_value(self, metrics: MemoryMetrics, path: str) -> Any:
        """Extract metric value using dot notation path."""
        try:
            value = metrics
            for part in path.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        except Exception:
            return None
    
    def _evaluate_threshold(self, rule: AlertRule, value: float) -> tuple:
        """Evaluate threshold condition."""
        if rule.threshold_value is None:
            return False, "", value
        
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y
        }
        
        op_func = operators.get(rule.threshold_operator)
        if not op_func:
            return False, "", value
        
        if op_func(value, rule.threshold_value):
            message = (f"{rule.metric_path} = {value:.3f} {rule.threshold_operator} "
                      f"{rule.threshold_value:.3f}")
            return True, message, value
        
        return False, "", value
    
    def _evaluate_trend(self, rule: AlertRule, current_value: float) -> tuple:
        """Evaluate trend condition."""
        history = self.metric_history.get(rule.metric_path, deque())
        if len(history) < 2:
            return False, "", current_value
        
        # Get values within trend window
        cutoff_time = time.time() - rule.trend_window_seconds
        recent_values = [(t, v) for t, v in history if t >= cutoff_time]
        
        if len(recent_values) < 2:
            return False, "", current_value
        
        # Calculate trend
        oldest_value = recent_values[0][1]
        change_percent = (current_value - oldest_value) / oldest_value if oldest_value != 0 else 0
        
        if abs(change_percent) >= rule.trend_threshold:
            direction = "increasing" if change_percent > 0 else "decreasing"
            message = (f"{rule.metric_path} {direction} by {change_percent*100:.1f}% "
                      f"over {rule.trend_window_seconds}s")
            return True, message, current_value
        
        return False, "", current_value
    
    def _evaluate_rate_of_change(self, rule: AlertRule, current_value: float) -> tuple:
        """Evaluate rate of change condition."""
        history = self.metric_history.get(rule.metric_path, deque())
        if len(history) < 2:
            return False, "", current_value
        
        # Get value from rate window ago
        cutoff_time = time.time() - rule.rate_window_seconds
        past_values = [(t, v) for t, v in history if t <= cutoff_time]
        
        if not past_values:
            return False, "", current_value
        
        past_value = past_values[-1][1]  # Most recent value before cutoff
        rate = (current_value - past_value) / rule.rate_window_seconds
        rate_percent = rate / past_value if past_value != 0 else 0
        
        if abs(rate_percent) >= rule.rate_threshold:
            message = (f"{rule.metric_path} changing at {rate_percent*100:.1f}%/min "
                      f"(threshold: {rule.rate_threshold*100:.1f}%/min)")
            return True, message, current_value
        
        return False, "", current_value
    
    def _evaluate_anomaly(self, rule: AlertRule, current_value: float) -> tuple:
        """Evaluate anomaly condition using statistical analysis."""
        history = self.metric_history.get(rule.metric_path, deque())
        if len(history) < rule.anomaly_window_size // 2:
            return False, "", current_value
        
        # Calculate statistics from historical data
        values = [v for t, v in history]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return False, "", current_value
        
        # Check if current value is anomalous
        z_score = abs(current_value - mean) / std_dev
        
        if z_score >= rule.anomaly_sensitivity:
            message = (f"{rule.metric_path} = {current_value:.3f} is anomalous "
                      f"(z-score: {z_score:.2f}, threshold: {rule.anomaly_sensitivity})")
            return True, message, current_value
        
        return False, "", current_value
    
    def _get_alert_context(self, rule: AlertRule, metrics: MemoryMetrics) -> Dict[str, Any]:
        """Get additional context for an alert."""
        context = {
            'rule_description': rule.description,
            'system_cpu_usage': metrics.cpu_usage,
            'system_memory_usage': metrics.system_memory_usage,
            'fragmentation_index': metrics.fragmentation_index,
            'memory_pressure': metrics.memory_pressure,
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics.timestamp))
        }
        
        # Add pool-specific context if relevant
        if 'pool_' in rule.metric_path:
            context['pool_utilization'] = metrics.pool_utilization
            context['pool_hit_rates'] = metrics.pool_hit_rates
            context['pool_sizes'] = metrics.pool_sizes
        
        return context
    
    def _trigger_alert(self, rule: AlertRule, alert: Alert) -> None:
        """Trigger an alert and send notifications."""
        with self.lock:
            # Update rule state
            rule.last_triggered = alert.timestamp
            rule.trigger_count += 1
            
            # Add to active alerts and history
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")
            
            # Send notifications
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in notification handler: {e}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts."""
        with self.lock:
            if severity:
                return [a for a in self.active_alerts if a.severity == severity]
            return list(self.active_alerts)
    
    def clear_alert(self, rule_name: str) -> bool:
        """Clear active alerts for a specific rule."""
        with self.lock:
            initial_count = len(self.active_alerts)
            self.active_alerts = [a for a in self.active_alerts if a.rule_name != rule_name]
            cleared_count = initial_count - len(self.active_alerts)
            
            if cleared_count > 0:
                logger.info(f"Cleared {cleared_count} alerts for rule {rule_name}")
                return True
            return False
    
    def clear_all_alerts(self) -> int:
        """Clear all active alerts."""
        with self.lock:
            count = len(self.active_alerts)
            self.active_alerts.clear()
            logger.info(f"Cleared all {count} active alerts")
            return count
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status."""
        with self.lock:
            active_by_severity = {}
            for severity in AlertSeverity:
                active_by_severity[severity.value] = len([
                    a for a in self.active_alerts if a.severity == severity
                ])
            
            rule_stats = {}
            for rule_name, rule in self.rules.items():
                rule_stats[rule_name] = {
                    'enabled': rule.enabled,
                    'trigger_count': rule.trigger_count,
                    'last_triggered': rule.last_triggered,
                    'cooldown_remaining': max(0, rule.cooldown_seconds - (time.time() - rule.last_triggered))
                }
            
            return {
                'total_rules': len(self.rules),
                'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
                'active_alerts': len(self.active_alerts),
                'active_by_severity': active_by_severity,
                'total_alert_history': len(self.alert_history),
                'rule_stats': rule_stats,
                'monitoring_active': self.running
            }


# Predefined alert rules for common scenarios
def create_default_alert_rules() -> List[AlertRule]:
    """Create a set of default alert rules for memory management."""
    return [
        # High memory utilization
        AlertRule(
            name="high_pool_utilization",
            metric_path="pool_utilization.game_state",
            condition=AlertCondition.THRESHOLD,
            severity=AlertSeverity.WARNING,
            threshold_value=80.0,
            threshold_operator=">=",
            description="GameState pool utilization is high"
        ),
        
        # Critical memory utilization
        AlertRule(
            name="critical_pool_utilization",
            metric_path="pool_utilization.game_state",
            condition=AlertCondition.THRESHOLD,
            severity=AlertSeverity.CRITICAL,
            threshold_value=95.0,
            threshold_operator=">=",
            description="GameState pool utilization is critical"
        ),
        
        # Low hit rate
        AlertRule(
            name="low_pool_hit_rate",
            metric_path="pool_hit_rates.game_state",
            condition=AlertCondition.THRESHOLD,
            severity=AlertSeverity.WARNING,
            threshold_value=50.0,
            threshold_operator="<",
            description="GameState pool hit rate is low"
        ),
        
        # High fragmentation
        AlertRule(
            name="high_fragmentation",
            metric_path="fragmentation_index",
            condition=AlertCondition.THRESHOLD,
            severity=AlertSeverity.WARNING,
            threshold_value=0.25,
            threshold_operator=">=",
            description="Memory fragmentation is high"
        ),
        
        # High allocation latency
        AlertRule(
            name="high_allocation_latency",
            metric_path="allocation_latency_p95",
            condition=AlertCondition.THRESHOLD,
            severity=AlertSeverity.WARNING,
            threshold_value=1000.0,  # 1ms
            threshold_operator=">=",
            description="Memory allocation latency is high"
        ),
        
        # Memory pressure
        AlertRule(
            name="memory_pressure",
            metric_path="memory_pressure",
            condition=AlertCondition.THRESHOLD,
            severity=AlertSeverity.CRITICAL,
            threshold_value=0.9,
            threshold_operator=">=",
            description="System memory pressure is critical"
        ),
        
        # Allocation rate anomaly
        AlertRule(
            name="allocation_rate_anomaly",
            metric_path="allocations_per_second",
            condition=AlertCondition.ANOMALY,
            severity=AlertSeverity.INFO,
            anomaly_sensitivity=2.5,
            description="Unusual allocation rate detected"
        )
    ]


# Notification handlers
def console_notification_handler(alert: Alert) -> None:
    """Simple console notification handler."""
    print(f"🚨 [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")


def log_notification_handler(alert: Alert) -> None:
    """Log-based notification handler."""
    level = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.CRITICAL: logging.CRITICAL
    }.get(alert.severity, logging.WARNING)
    
    logger.log(level, f"Alert {alert.rule_name}: {alert.message}")


def webhook_notification_handler(webhook_url: str) -> Callable[[Alert], None]:
    """Create a webhook notification handler."""
    def handler(alert: Alert) -> None:
        try:
            import requests
            payload = {
                'alert': alert.to_dict(),
                'timestamp': alert.timestamp
            }
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    return handler 