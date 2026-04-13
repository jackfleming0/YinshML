#!/usr/bin/env python3
"""Monitor large-scale self-play data collection."""

import argparse
import json
import logging
import os
import psutil
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfPlayMonitor:
    """Monitor large-scale self-play data collection."""
    
    def __init__(self, config_path: str, output_dir: str):
        """Initialize the monitor.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Directory where self-play data is being written
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.config = self._load_config()
        self.start_time = datetime.now()
        self.last_check_time = self.start_time
        self.last_game_count = 0
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create monitoring log file
        self.monitor_log = self.output_dir / "monitoring.log"
        self._setup_file_logging()
        
        logger.info(f"Self-play monitor initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target games: {self.config['data_collection']['target_games']}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_file_logging(self):
        """Set up file logging for monitoring data."""
        file_handler = logging.FileHandler(self.monitor_log)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _get_game_count(self) -> int:
        """Get current number of completed games."""
        try:
            # Look for parquet files or game records
            parquet_files = list(self.output_dir.glob("*.parquet"))
            if parquet_files:
                # Estimate based on file count and batch size
                batch_size = self.config['storage']['parquet_batch_size']
                return len(parquet_files) * batch_size
            
            # Look for JSON game files
            json_files = list(self.output_dir.glob("game_*.json"))
            return len(json_files)
            
        except Exception as e:
            logger.warning(f"Could not determine game count: {e}")
            return 0
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information."""
        try:
            usage = shutil.disk_usage(self.output_dir)
            total_gb = usage.total / (1024**3)
            used_gb = (usage.total - usage.free) / (1024**3)
            free_gb = usage.free / (1024**3)
            percent_used = (used_gb / total_gb) * 100
            
            return {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'percent_used': percent_used
            }
        except Exception as e:
            logger.warning(f"Could not get disk usage: {e}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {}
    
    def _calculate_progress_metrics(self) -> Dict[str, Any]:
        """Calculate progress metrics."""
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        current_game_count = self._get_game_count()
        
        # Calculate games per hour
        if elapsed_time.total_seconds() > 0:
            games_per_hour = (current_game_count / elapsed_time.total_seconds()) * 3600
        else:
            games_per_hour = 0
        
        # Calculate recent games per hour
        time_since_last_check = current_time - self.last_check_time
        if time_since_last_check.total_seconds() > 0:
            recent_games_per_hour = ((current_game_count - self.last_game_count) / 
                                    time_since_last_check.total_seconds()) * 3600
        else:
            recent_games_per_hour = 0
        
        # Calculate ETA
        target_games = self.config['data_collection']['target_games']
        if games_per_hour > 0 and current_game_count < target_games:
            remaining_games = target_games - current_game_count
            eta_seconds = (remaining_games / games_per_hour) * 3600
            eta = current_time + timedelta(seconds=eta_seconds)
        else:
            eta = None
        
        # Update tracking variables
        self.last_check_time = current_time
        self.last_game_count = current_game_count
        
        return {
            'current_games': current_game_count,
            'target_games': target_games,
            'progress_percent': (current_game_count / target_games) * 100 if target_games > 0 else 0,
            'games_per_hour': games_per_hour,
            'recent_games_per_hour': recent_games_per_hour,
            'elapsed_time': elapsed_time,
            'eta': eta
        }
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # Check games per hour threshold
        min_games_per_hour = self.config['monitoring']['alert_threshold_games_per_hour']
        if metrics['recent_games_per_hour'] < min_games_per_hour and metrics['current_games'] > 100:
            alerts.append(f"Low performance: {metrics['recent_games_per_hour']:.1f} games/hour "
                         f"(threshold: {min_games_per_hour})")
        
        # Check disk usage threshold
        disk_usage = self._get_disk_usage()
        if disk_usage:
            max_disk_usage = self.config['monitoring']['alert_threshold_disk_usage']
            if disk_usage['percent_used'] > max_disk_usage:
                alerts.append(f"High disk usage: {disk_usage['percent_used']:.1f}% "
                             f"(threshold: {max_disk_usage}%)")
        
        return alerts
    
    def _log_status(self, metrics: Dict[str, Any], alerts: List[str]):
        """Log current status."""
        logger.info("=== Self-Play Status ===")
        logger.info(f"Games completed: {metrics['current_games']}/{metrics['target_games']} "
                   f"({metrics['progress_percent']:.1f}%)")
        logger.info(f"Games per hour: {metrics['games_per_hour']:.1f} "
                   f"(recent: {metrics['recent_games_per_hour']:.1f})")
        logger.info(f"Elapsed time: {metrics['elapsed_time']}")
        
        if metrics['eta']:
            logger.info(f"Estimated completion: {metrics['eta']}")
        
        # Log system resources
        disk_usage = self._get_disk_usage()
        if disk_usage:
            logger.info(f"Disk usage: {disk_usage['used_gb']:.1f}GB/{disk_usage['total_gb']:.1f}GB "
                       f"({disk_usage['percent_used']:.1f}%)")
        
        memory_usage = self._get_memory_usage()
        if memory_usage:
            logger.info(f"Memory usage: {memory_usage['used_gb']:.1f}GB/{memory_usage['total_gb']:.1f}GB "
                       f"({memory_usage['percent_used']:.1f}%)")
        
        # Log alerts
        if alerts:
            logger.warning("=== ALERTS ===")
            for alert in alerts:
                logger.warning(alert)
    
    def _save_status_report(self, metrics: Dict[str, Any], alerts: List[str]):
        """Save status report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'alerts': alerts,
            'disk_usage': self._get_disk_usage(),
            'memory_usage': self._get_memory_usage(),
            'config': self.config
        }
        
        # Convert datetime objects to strings for JSON serialization
        if metrics['eta']:
            report['metrics']['eta'] = metrics['eta'].isoformat()
        report['metrics']['elapsed_time'] = str(metrics['elapsed_time'])
        
        report_file = self.output_dir / "status_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def run(self, check_interval: int = 300):
        """Run the monitoring loop.
        
        Args:
            check_interval: Seconds between status checks
        """
        logger.info(f"Starting monitoring loop (check interval: {check_interval}s)")
        
        try:
            while self.running:
                # Calculate metrics
                metrics = self._calculate_progress_metrics()
                alerts = self._check_alerts(metrics)
                
                # Log status
                self._log_status(metrics, alerts)
                
                # Save status report
                self._save_status_report(metrics, alerts)
                
                # Check if target reached
                if metrics['current_games'] >= metrics['target_games']:
                    logger.info("🎉 Target number of games reached!")
                    break
                
                # Wait for next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            logger.info("Monitoring stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Monitor large-scale self-play data collection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory where self-play data is being written')
    parser.add_argument('--check-interval', type=int, default=300,
                       help='Seconds between status checks (default: 300)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.output_dir):
        logger.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    # Create and run monitor
    monitor = SelfPlayMonitor(args.config, args.output_dir)
    monitor.run(args.check_interval)


if __name__ == '__main__':
    main()
