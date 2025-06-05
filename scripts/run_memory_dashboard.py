#!/usr/bin/env python3
"""
Memory Management Dashboard Launcher for YINSH ML.

This script provides a command-line interface for launching and configuring
the memory management dashboard with various monitoring options.
"""

import argparse
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.monitoring import (
    MemoryMetricsCollector,
    MemoryDashboard,
    DashboardConfig,
    create_streamlit_app
)

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('memory_dashboard.log')
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YINSH ML Memory Management Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Streamlit dashboard
  python scripts/run_memory_dashboard.py --mode streamlit
  
  # Run console monitoring with alerts
  python scripts/run_memory_dashboard.py --mode console --enable-alerts
  
  # Export metrics to Prometheus format
  python scripts/run_memory_dashboard.py --mode export --format prometheus --output metrics.prom
  
  # Run with custom update interval
  python scripts/run_memory_dashboard.py --mode streamlit --update-interval 2.0
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["streamlit", "console", "export"],
        default="streamlit",
        help="Dashboard mode (default: streamlit)"
    )
    
    parser.add_argument(
        "--update-interval",
        type=float,
        default=5.0,
        help="Update interval in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--max-data-points",
        type=int,
        default=1000,
        help="Maximum data points to keep in memory (default: 1000)"
    )
    
    parser.add_argument(
        "--enable-alerts",
        action="store_true",
        help="Enable alerting system"
    )
    
    parser.add_argument(
        "--enable-health-checks",
        action="store_true",
        help="Enable health monitoring"
    )
    
    parser.add_argument(
        "--export-directory",
        type=str,
        help="Directory for auto-exporting metrics"
    )
    
    parser.add_argument(
        "--export-interval",
        type=float,
        default=300.0,
        help="Auto-export interval in seconds (default: 300)"
    )
    
    # Export mode specific options
    parser.add_argument(
        "--format",
        choices=["json", "csv", "prometheus", "grafana"],
        default="json",
        help="Export format (default: json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for export mode"
    )
    
    parser.add_argument(
        "--duration", 
        type=float,
        help="Duration to run in seconds (for console/export modes)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard (default: 8501)"
    )
    
    return parser.parse_args()

def create_memory_components():
    """Create memory components for monitoring."""
    try:
        # Create memory pools for monitoring
        from yinsh_ml.memory import GameStatePool, TensorPool, GameStatePoolConfig, TensorPoolConfig
        from yinsh_ml.memory.config import GrowthPolicy
        from yinsh_ml.game import GameState
        import torch
        
        # Factory functions
        def game_state_factory():
            return GameState()
        
        def tensor_factory(shape, dtype, device):
            return torch.zeros(shape, dtype=dtype, device=device)
        
        # Create game state pool
        gs_config = GameStatePoolConfig(
            initial_size=100,
            growth_policy=GrowthPolicy.LINEAR,
            enable_statistics=True,
            training_mode=True,
            factory_func=game_state_factory
        )
        game_state_pool = GameStatePool(gs_config)
        
        # Create tensor pool  
        tensor_config = TensorPoolConfig(
            initial_size=50,
            enable_statistics=True,
            factory_func=tensor_factory
        )
        tensor_pool = TensorPool(tensor_config)
        
        # Create a simple container to hold the pools
        class MemoryComponents:
            def __init__(self, game_pool, tensor_pool):
                self.game_state_pool = game_pool
                self.tensor_pool = tensor_pool
                
        return MemoryComponents(game_state_pool, tensor_pool)
        
    except Exception as e:
        logger.warning(f"Could not create memory components: {e}")
        logger.info("Running dashboard in standalone mode")
        return None

def run_streamlit_dashboard(args):
    """Run the Streamlit-based dashboard."""
    try:
        import streamlit.web.cli as stcli
        import streamlit as st
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit plotly")
        return 1
    
    # Create metrics collector
    memory_components = create_memory_components()
    metrics_collector = MemoryMetricsCollector(memory_components)
    
    # Create dashboard config
    config = DashboardConfig(
        update_interval=args.update_interval,
        max_data_points=args.max_data_points,
        enable_alerts=args.enable_alerts,
        enable_health_checks=args.enable_health_checks,
        export_directory=args.export_directory,
        auto_export_interval=args.export_interval
    )
    
    # Create temporary script for Streamlit
    dashboard_script = project_root / "temp_dashboard.py"
    
    dashboard_code = f"""
import sys
sys.path.insert(0, '{project_root}')

from yinsh_ml.monitoring import MemoryMetricsCollector, DashboardConfig, create_streamlit_app
from yinsh_ml.memory import GameStatePool, TensorPool, GameStatePoolConfig, TensorPoolConfig
from yinsh_ml.memory.config import GrowthPolicy
from yinsh_ml.game import GameState
import torch

# Create components
try:
    # Factory functions
    def game_state_factory():
        return GameState()
    
    def tensor_factory(shape, dtype, device):
        return torch.zeros(shape, dtype=dtype, device=device)
    
    gs_config = GameStatePoolConfig(
        initial_size=100, 
        growth_policy=GrowthPolicy.LINEAR, 
        enable_statistics=True, 
        training_mode=True,
        factory_func=game_state_factory
    )
    game_state_pool = GameStatePool(gs_config)
    
    tensor_config = TensorPoolConfig(
        initial_size=50, 
        enable_statistics=True,
        factory_func=tensor_factory
    )
    tensor_pool = TensorPool(tensor_config)
    
    class MemoryComponents:
        def __init__(self, game_pool, tensor_pool):
            self.game_state_pool = game_pool
            self.tensor_pool = tensor_pool
    
    memory_components = MemoryComponents(game_state_pool, tensor_pool)
except Exception as e:
    print(f"Warning: Could not create memory components: {{e}}")
    memory_components = None

metrics_collector = MemoryMetricsCollector(memory_components)

config = DashboardConfig(
    update_interval={args.update_interval},
    max_data_points={args.max_data_points},
    enable_alerts={args.enable_alerts},
    enable_health_checks={args.enable_health_checks},
    export_directory={repr(args.export_directory)},
    auto_export_interval={args.export_interval}
)

# Create and run dashboard
create_streamlit_app(metrics_collector, config)
"""
    
    try:
        with open(dashboard_script, 'w') as f:
            f.write(dashboard_code)
        
        # Run Streamlit
        sys.argv = [
            "streamlit", "run", str(dashboard_script),
            "--server.port", str(args.port),
            "--server.headless", "false"
        ]
        
        logger.info(f"Starting Streamlit dashboard on port {args.port}")
        stcli.main()
        
    finally:
        # Clean up temporary file
        if dashboard_script.exists():
            dashboard_script.unlink()
    
    return 0

def run_console_dashboard(args):
    """Run the console-based monitoring dashboard."""
    memory_components = create_memory_components()
    metrics_collector = MemoryMetricsCollector(memory_components)
    
    config = DashboardConfig(
        update_interval=args.update_interval,
        max_data_points=args.max_data_points,
        enable_alerts=args.enable_alerts,
        enable_health_checks=args.enable_health_checks,
        export_directory=args.export_directory,
        auto_export_interval=args.export_interval
    )
    
    dashboard = MemoryDashboard(metrics_collector, config)
    
    # Set up signal handler for clean shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        dashboard.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting console dashboard...")
        dashboard.start_monitoring()
        
        start_time = time.time()
        
        while True:
            # Get current metrics
            current_metrics = metrics_collector.get_current_metrics()
            
            if current_metrics:
                # Clear screen and display metrics
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("=" * 80)
                print(f"YINSH ML Memory Management Dashboard - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                
                print(f"\n📊 Memory Overview:")
                print(f"  Total Memory Used: {current_metrics.total_memory_used / (1024**2):.1f} MB")
                print(f"  Memory Pressure: {current_metrics.memory_pressure:.1%}")
                print(f"  Fragmentation Index: {current_metrics.fragmentation_index:.1%}")
                
                print(f"\n⚡ Allocation Performance:")
                print(f"  Allocations/sec: {current_metrics.allocations_per_second:.1f}")
                print(f"  Deallocations/sec: {current_metrics.deallocations_per_second:.1f}")
                print(f"  Latency P50: {current_metrics.allocation_latency_p50:.1f} μs")
                print(f"  Latency P95: {current_metrics.allocation_latency_p95:.1f} μs")
                print(f"  Latency P99: {current_metrics.allocation_latency_p99:.1f} μs")
                
                if current_metrics.pool_utilization:
                    print(f"\n🏊 Pool Utilization:")
                    for pool_name, utilization in current_metrics.pool_utilization.items():
                        print(f"  {pool_name}: {utilization:.1%}")
                
                if dashboard.alert_manager:
                    active_alerts = dashboard.alert_manager.get_active_alerts()
                    if active_alerts:
                        print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
                        for alert in active_alerts[-5:]:  # Show last 5 alerts
                            print(f"  [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")
                    else:
                        print(f"\n✅ No active alerts")
                
                print(f"\nPress Ctrl+C to stop monitoring...")
            
            # Check duration limit
            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info(f"Reached duration limit of {args.duration} seconds")
                break
            
            time.sleep(args.update_interval)
    
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted by user")
    finally:
        dashboard.stop_monitoring()
    
    return 0

def run_export_mode(args):
    """Run in export mode to generate metric files."""
    memory_components = create_memory_components()
    metrics_collector = MemoryMetricsCollector(memory_components)
    
    config = DashboardConfig(
        update_interval=args.update_interval,
        enable_alerts=args.enable_alerts,
        enable_health_checks=args.enable_health_checks
    )
    
    dashboard = MemoryDashboard(metrics_collector, config)
    
    try:
        logger.info("Starting metrics collection for export...")
        dashboard.start_monitoring()
        
        # Collect data for specified duration or default
        duration = args.duration or 60.0  # Default 1 minute
        logger.info(f"Collecting metrics for {duration} seconds...")
        
        time.sleep(duration)
        
        # Export data
        output_path = args.output or f"memory_metrics_{int(time.time())}.{args.format}"
        
        logger.info(f"Exporting metrics in {args.format} format to {output_path}")
        
        if args.format == "json":
            data = dashboard.exporter.export_json(include_history=True)
            with open(output_path, 'w') as f:
                f.write(data)
        
        elif args.format == "csv":
            dashboard.exporter.export_csv(output_path)
        
        elif args.format == "prometheus":
            data = dashboard.exporter.export_prometheus()
            with open(output_path, 'w') as f:
                f.write(data)
        
        elif args.format == "grafana":
            data = dashboard.exporter.export_grafana_dashboard()
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported successfully to {output_path}")
        
    finally:
        dashboard.stop_monitoring()
    
    return 0

def main():
    """Main entry point."""
    args = parse_args()
    
    setup_logging(args.log_level)
    
    logger.info(f"Starting YINSH ML Memory Dashboard in {args.mode} mode")
    
    try:
        if args.mode == "streamlit":
            return run_streamlit_dashboard(args)
        elif args.mode == "console":
            return run_console_dashboard(args)
        elif args.mode == "export":
            return run_export_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
            
    except Exception as e:
        logger.error(f"Dashboard failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 