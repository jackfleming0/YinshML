"""
Memory Management Dashboard for YINSH ML.

This module provides real-time dashboards and visualization tools for monitoring
memory pool performance, allocation patterns, and system health.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .metrics import MemoryMetricsCollector, MemoryMetrics, MetricsExporter
from .alerts import AlertManager, Alert, AlertSeverity
from .health_check import MemoryHealthChecker, HealthStatus

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for the memory management dashboard."""
    update_interval: float = 5.0  # seconds
    history_duration: float = 3600.0  # 1 hour
    max_data_points: int = 1000
    enable_alerts: bool = True
    enable_health_checks: bool = True
    
    # Visual settings
    theme: str = "dark"  # "light" or "dark"
    chart_height: int = 400
    chart_width: int = 800
    
    # Export settings
    auto_export_interval: float = 300.0  # 5 minutes
    export_formats: List[str] = field(default_factory=lambda: ["json", "prometheus"])
    export_directory: Optional[str] = None

class MemoryDashboard:
    """
    Real-time memory management dashboard with multiple visualization backends.
    
    Supports both Streamlit web interface and programmatic data export.
    """
    
    def __init__(self, 
                 metrics_collector: MemoryMetricsCollector,
                 config: Optional[DashboardConfig] = None):
        """
        Initialize the memory dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            config: Dashboard configuration
        """
        self.metrics_collector = metrics_collector
        self.config = config or DashboardConfig()
        
        self.exporter = MetricsExporter(metrics_collector)
        
        # Initialize optional components
        self.alert_manager = None
        self.health_checker = None
        
        if self.config.enable_alerts:
            self.alert_manager = AlertManager(metrics_collector)
            self._setup_default_alerts()
            
        if self.config.enable_health_checks:
            self.health_checker = MemoryHealthChecker(metrics_collector)
        
        # Dashboard state
        self._is_running = False
        self._update_thread = None
        self._last_export_time = 0.0
        
        # Data storage for dashboard
        self._dashboard_data = {
            'metrics_history': [],
            'alerts_history': [],
            'health_status': None,
            'last_update': 0.0
        }
        
        logger.info("Memory dashboard initialized")
    
    def _setup_default_alerts(self):
        """Set up default alert rules for memory monitoring."""
        if not self.alert_manager:
            return
            
        from .alerts import AlertRule, AlertCondition, AlertSeverity, create_default_alert_rules
        
        # Add the default rules
        default_rules = create_default_alert_rules()
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
        
        # Add console notification handler
        from .alerts import console_notification_handler
        self.alert_manager.add_notification_handler(console_notification_handler)
        
        logger.info(f"Set up {len(default_rules)} default alert rules")
    
    def start_monitoring(self):
        """Start the dashboard monitoring and data collection."""
        if self._is_running:
            logger.warning("Dashboard monitoring already running")
            return
        
        self._is_running = True
        
        # Start metrics collection if not already running
        if not self.metrics_collector._is_collecting:
            self.metrics_collector.start_collection()
        
        # Start alert monitoring if available
        if self.alert_manager:
            self.alert_manager.start_monitoring()
        
        # Start dashboard update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        logger.info("Dashboard monitoring started")
    
    def stop_monitoring(self):
        """Stop the dashboard monitoring."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop alert monitoring
        if self.alert_manager:
            self.alert_manager.stop_monitoring()
        
        # Wait for update thread to finish
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5.0)
        
        logger.info("Dashboard monitoring stopped")
    
    def _update_loop(self):
        """Main dashboard update loop."""
        while self._is_running:
            try:
                self._update_dashboard_data()
                self._check_auto_export()
                time.sleep(self.config.update_interval)
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _update_dashboard_data(self):
        """Update the dashboard data with latest metrics."""
        current_time = time.time()
        
        # Get current metrics
        current_metrics = self.metrics_collector.get_current_metrics()
        if current_metrics:
            self._dashboard_data['metrics_history'].append(current_metrics)
        
        # Trim history to maximum data points
        if len(self._dashboard_data['metrics_history']) > self.config.max_data_points:
            excess = len(self._dashboard_data['metrics_history']) - self.config.max_data_points
            self._dashboard_data['metrics_history'] = self._dashboard_data['metrics_history'][excess:]
        
        # Get alerts if available
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            self._dashboard_data['alerts_history'] = active_alerts
        
        # Get health status if available
        if self.health_checker:
            try:
                health_result = self.health_checker.run_comprehensive_health_check()
                self._dashboard_data['health_status'] = health_result
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
        
        self._dashboard_data['last_update'] = current_time
    
    def _check_auto_export(self):
        """Check if it's time to auto-export data."""
        current_time = time.time()
        if current_time - self._last_export_time >= self.config.auto_export_interval:
            try:
                self._auto_export_data()
                self._last_export_time = current_time
            except Exception as e:
                logger.error(f"Auto-export failed: {e}")
    
    def _auto_export_data(self):
        """Automatically export data in configured formats."""
        if not self.config.export_directory:
            return
        
        export_dir = Path(self.config.export_directory)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_name in self.config.export_formats:
            try:
                if format_name == "json":
                    filepath = export_dir / f"memory_metrics_{timestamp}.json"
                    data = self.exporter.export_json(include_history=True)
                    with open(filepath, 'w') as f:
                        f.write(data)
                
                elif format_name == "prometheus":
                    filepath = export_dir / f"memory_metrics_{timestamp}.prom"
                    data = self.exporter.export_prometheus()
                    with open(filepath, 'w') as f:
                        f.write(data)
                
                elif format_name == "csv":
                    filepath = export_dir / f"memory_metrics_{timestamp}.csv"
                    self.exporter.export_csv(str(filepath))
                
                logger.debug(f"Exported {format_name} data to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export {format_name} data: {e}")
    
    def get_grafana_dashboard(self) -> Dict[str, Any]:
        """
        Generate a Grafana dashboard configuration for memory monitoring.
        
        Returns:
            Dictionary containing Grafana dashboard JSON
        """
        return self.exporter.export_grafana_dashboard()
    
    def export_prometheus_config(self, output_path: str):
        """
        Export Prometheus configuration for memory metrics scraping.
        
        Args:
            output_path: Path to save the Prometheus config
        """
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "yinsh-memory-metrics",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"]  # Adjust as needed
                        }
                    ],
                    "scrape_interval": "5s",
                    "metrics_path": "/metrics"
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Prometheus config exported to {output_path}")
    
    def create_streamlit_dashboard(self):
        """
        Create a Streamlit-based web dashboard.
        
        This method sets up the Streamlit interface for real-time monitoring.
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available. Install with: pip install streamlit plotly")
        
        st.set_page_config(
            page_title="YINSH ML Memory Management Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 YINSH ML Memory Management Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        
        if st.sidebar.button("Start Monitoring" if not self._is_running else "Stop Monitoring"):
            if self._is_running:
                self.stop_monitoring()
                st.sidebar.success("Monitoring stopped")
            else:
                self.start_monitoring()
                st.sidebar.success("Monitoring started")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        
        # Manual refresh button
        if st.sidebar.button("Refresh Now"):
            st.rerun()
        
        # Auto-refresh mechanism (simplified)
        if auto_refresh:
            # Create placeholder for auto-refresh status
            refresh_placeholder = st.sidebar.empty()
            refresh_placeholder.info(f"Auto-refreshing every {self.config.update_interval}s")
            
            # Set up auto-refresh using st.rerun
            time.sleep(self.config.update_interval)
            st.rerun()
        
        # Main dashboard content - always render
        self._render_dashboard()
    
    def _render_dashboard(self):
        """Render the complete dashboard with all sections."""
        self._render_overview_section()
        self._render_pool_metrics_section()
        self._render_allocation_metrics_section()
        self._render_system_metrics_section()
        
        if self.alert_manager:
            self._render_alerts_section()
        
        if self.health_checker:
            self._render_health_section()
    
    def _render_overview_section(self):
        """Render the overview section of the dashboard."""
        st.header("📊 System Overview")
        
        current_metrics = self.metrics_collector.get_current_metrics()
        if not current_metrics:
            st.warning("No metrics data available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Memory Used",
                f"{current_metrics.total_memory_used / (1024**2):.1f} MB",
                delta=None
            )
        
        with col2:
            st.metric(
                "Memory Pressure",
                f"{current_metrics.memory_pressure:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Fragmentation Index",
                f"{current_metrics.fragmentation_index:.1%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Allocations/sec",
                f"{current_metrics.allocations_per_second:.1f}",
                delta=None
            )
    
    def _render_pool_metrics_section(self):
        """Render pool-specific metrics."""
        st.header("🏊 Memory Pool Metrics")
        
        metrics_history = self._dashboard_data['metrics_history']
        if not metrics_history:
            st.info("No pool metrics data available")
            return
        
        # Pool utilization chart
        fig = go.Figure()
        
        # Get pool names
        latest_metrics = metrics_history[-1]
        pool_names = list(latest_metrics.pool_utilization.keys())
        
        for pool_name in pool_names:
            utilization_values = []
            timestamps = []
            
            for metrics in metrics_history[-100:]:  # Last 100 data points
                if pool_name in metrics.pool_utilization:
                    utilization_values.append(metrics.pool_utilization[pool_name] * 100)
                    timestamps.append(datetime.fromtimestamp(metrics.timestamp))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=utilization_values,
                mode='lines+markers',
                name=f"{pool_name} Utilization (%)",
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Memory Pool Utilization Over Time",
            xaxis_title="Time",
            yaxis_title="Utilization (%)",
            height=self.config.chart_height,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pool hit rates
        col1, col2 = st.columns(2)
        
        with col1:
            if latest_metrics.pool_hit_rates:
                hit_rate_df = pd.DataFrame([
                    {"Pool": pool, "Hit Rate": rate * 100}
                    for pool, rate in latest_metrics.pool_hit_rates.items()
                ])
                
                fig_hit = px.bar(
                    hit_rate_df,
                    x="Pool",
                    y="Hit Rate",
                    title="Current Pool Hit Rates",
                    color="Hit Rate",
                    color_continuous_scale="Viridis"
                )
                fig_hit.update_layout(height=300)
                st.plotly_chart(fig_hit, use_container_width=True)
        
        with col2:
            if latest_metrics.pool_sizes:
                size_df = pd.DataFrame([
                    {"Pool": pool, "Current Size": size, "Max Size": latest_metrics.pool_max_sizes.get(pool, size)}
                    for pool, size in latest_metrics.pool_sizes.items()
                ])
                
                fig_sizes = go.Figure()
                fig_sizes.add_trace(go.Bar(
                    x=size_df["Pool"],
                    y=size_df["Current Size"],
                    name="Current Size",
                    marker_color="lightblue"
                ))
                fig_sizes.add_trace(go.Bar(
                    x=size_df["Pool"],
                    y=size_df["Max Size"],
                    name="Max Size",
                    marker_color="darkblue"
                ))
                
                fig_sizes.update_layout(
                    title="Pool Sizes",
                    height=300,
                    barmode='group'
                )
                st.plotly_chart(fig_sizes, use_container_width=True)
    
    def _render_allocation_metrics_section(self):
        """Render allocation performance metrics."""
        st.header("⚡ Allocation Performance")
        
        metrics_history = self._dashboard_data['metrics_history']
        if not metrics_history:
            st.info("No allocation metrics data available")
            return
        
        # Allocation latency percentiles
        timestamps = []
        p50_values = []
        p95_values = []
        p99_values = []
        
        for metrics in metrics_history[-100:]:
            timestamps.append(datetime.fromtimestamp(metrics.timestamp))
            p50_values.append(metrics.allocation_latency_p50)
            p95_values.append(metrics.allocation_latency_p95)
            p99_values.append(metrics.allocation_latency_p99)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=p50_values, name="P50", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=timestamps, y=p95_values, name="P95", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=timestamps, y=p99_values, name="P99", line=dict(color="red")))
        
        fig.update_layout(
            title="Allocation Latency Percentiles (μs)",
            xaxis_title="Time",
            yaxis_title="Latency (μs)",
            height=self.config.chart_height
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Allocation rate
        col1, col2 = st.columns(2)
        
        with col1:
            alloc_timestamps = []
            alloc_rates = []
            dealloc_rates = []
            
            for metrics in metrics_history[-50:]:
                alloc_timestamps.append(datetime.fromtimestamp(metrics.timestamp))
                alloc_rates.append(metrics.allocations_per_second)
                dealloc_rates.append(metrics.deallocations_per_second)
            
            fig_rates = go.Figure()
            fig_rates.add_trace(go.Scatter(
                x=alloc_timestamps, 
                y=alloc_rates, 
                name="Allocations/sec",
                line=dict(color="blue")
            ))
            fig_rates.add_trace(go.Scatter(
                x=alloc_timestamps, 
                y=dealloc_rates, 
                name="Deallocations/sec",
                line=dict(color="red")
            ))
            
            fig_rates.update_layout(
                title="Allocation/Deallocation Rates",
                height=300
            )
            st.plotly_chart(fig_rates, use_container_width=True)
    
    def _render_system_metrics_section(self):
        """Render system-wide metrics."""
        st.header("🖥️ System Metrics")
        
        current_metrics = self.metrics_collector.get_current_metrics()
        if not current_metrics:
            st.info("No system metrics data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{current_metrics.cpu_usage:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "System Memory Usage",
                f"{current_metrics.system_memory_usage:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "GC Collections",
                f"{current_metrics.gc_collections}",
                delta=None
            )
    
    def _render_alerts_section(self):
        """Render active alerts section."""
        st.header("🚨 Active Alerts")
        
        if not self.alert_manager:
            st.info("Alert manager not available")
            return
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        if not active_alerts:
            st.success("No active alerts")
            return
        
        for alert in active_alerts:
            severity_color = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "orange", 
                AlertSeverity.CRITICAL: "red"
            }.get(alert.severity, "gray")
            
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {severity_color}; margin: 10px 0;">
                <strong>{alert.severity.value.upper()}</strong>: {alert.rule_name}<br>
                {alert.message}<br>
                <small>Triggered: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_health_section(self):
        """Render system health section."""
        st.header("🏥 System Health")
        
        if not self.health_checker:
            st.info("Health checker not available")
            return
        
        health_status = self._dashboard_data.get('health_status')
        if not health_status:
            st.info("No health check data available")
            return
        
        overall_status = health_status.get('overall_status', HealthStatus.UNKNOWN)
        overall_score = health_status.get('overall_score', 0.0)
        
        status_color = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.WARNING: "orange",
            HealthStatus.CRITICAL: "red",
            HealthStatus.UNKNOWN: "gray"
        }.get(overall_status, "gray")
        
        st.markdown(f"""
        <div style="padding: 20px; background-color: {status_color}20; border-radius: 10px; margin: 10px 0;">
            <h3 style="color: {status_color};">Overall Health: {overall_status.value.upper()}</h3>
            <p>Health Score: {overall_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Health check details
        checks = health_status.get('checks', [])
        for check in checks:
            with st.expander(f"{check['name']} - {check['status']}"):
                st.write(f"**Score:** {check['score']:.1%}")
                st.write(f"**Message:** {check['message']}")
                
                if check.get('recommendations'):
                    st.write("**Recommendations:**")
                    for rec in check['recommendations']:
                        st.write(f"- {rec}")

def create_streamlit_app(metrics_collector: MemoryMetricsCollector, 
                        config: Optional[DashboardConfig] = None):
    """
    Create and run a Streamlit dashboard application.
    
    Args:
        metrics_collector: Metrics collector instance
        config: Dashboard configuration
    """
    dashboard = MemoryDashboard(metrics_collector, config)
    dashboard.create_streamlit_dashboard()
    return dashboard 