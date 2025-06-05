"""
Reporting and visualization components for benchmark results.

This module provides various output formats for benchmark results including
HTML reports, CSV exports, plots, and JSON data.
"""

import json
import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict
import base64
from io import BytesIO

from .benchmark_framework import BenchmarkResult

logger = logging.getLogger(__name__)

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - plotting features disabled")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not available - advanced data processing disabled")


class JSONReporter:
    """Export benchmark results to JSON format."""
    
    def __init__(self):
        """Initialize JSON reporter."""
        pass
    
    def export(self, results: List[BenchmarkResult], filepath: str) -> None:
        """
        Export results to JSON file.
        
        Args:
            results: List of benchmark results
            filepath: Output file path
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'results': [result.to_dict() for result in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class CSVReporter:
    """Export benchmark results to CSV format."""
    
    def __init__(self):
        """Initialize CSV reporter."""
        pass
    
    def export(self, results: List[BenchmarkResult], filepath: str) -> None:
        """
        Export results to CSV file.
        
        Args:
            results: List of benchmark results
            filepath: Output file path
        """
        if not results:
            logger.warning("No results to export")
            return
        
        # Flatten results for CSV
        flattened_data = []
        for result in results:
            base_data = {
                'name': result.name,
                'description': result.description,
                'timestamp': result.timestamp.isoformat(),
                'iterations': result.iterations,
            }
            
            # Add statistics
            stats = result.calculate_statistics()
            for metric_name, metric_stats in stats.items():
                if isinstance(metric_stats, dict):
                    for stat_name, value in metric_stats.items():
                        base_data[f'{metric_name}_{stat_name}'] = value
                else:
                    base_data[metric_name] = metric_stats
            
            # Add average metrics
            avg_metrics = result.avg_metrics.to_dict()
            for category, metrics in avg_metrics.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        base_data[f'avg_{category}_{metric_name}'] = value
                else:
                    base_data[f'avg_{category}'] = metrics
            
            # Add custom metrics from first iteration (as representative)
            if result.iteration_metrics:
                for metric_name, value in result.iteration_metrics[0].custom_metrics.items():
                    base_data[f'custom_{metric_name}'] = value
            
            flattened_data.append(base_data)
        
        # Write CSV
        if flattened_data:
            fieldnames = flattened_data[0].keys()
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
        
        logger.info(f"Exported {len(results)} results to {filepath}")


class PlotReporter:
    """Generate plots and visualizations from benchmark results."""
    
    def __init__(self):
        """Initialize plot reporter."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for PlotReporter")
    
    def create_performance_comparison(self, 
                                    results: List[BenchmarkResult], 
                                    metric_name: str = 'duration_ms',
                                    output_path: Optional[str] = None) -> Optional[str]:
        """
        Create a performance comparison chart.
        
        Args:
            results: List of benchmark results
            metric_name: Name of metric to compare
            output_path: Path to save plot (if None, returns base64 encoded image)
            
        Returns:
            Path to saved file or base64 encoded image data
        """
        if not results:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        names = [result.name for result in results]
        values = []
        errors = []
        
        for result in results:
            stats = result.calculate_statistics()
            if metric_name in stats and 'mean' in stats[metric_name]:
                values.append(stats[metric_name]['mean'])
                errors.append(stats[metric_name].get('std_dev', 0))
            else:
                values.append(0)
                errors.append(0)
        
        # Create bar chart
        bars = ax.bar(range(len(names)), values, yerr=errors, capsize=5)
        
        # Customize chart
        ax.set_xlabel('Benchmark')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}')
        ax.set_title(f'Performance Comparison: {metric_name}')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            # Return base64 encoded image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64
    
    def create_memory_usage_timeline(self, 
                                   results: List[BenchmarkResult],
                                   output_path: Optional[str] = None) -> Optional[str]:
        """
        Create a memory usage timeline chart.
        
        Args:
            results: List of benchmark results
            output_path: Path to save plot
            
        Returns:
            Path to saved file or base64 encoded image data
        """
        if not results:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Process data
        timestamps = []
        peak_memory = []
        allocated_memory = []
        
        for result in results:
            timestamps.append(result.timestamp)
            peak_memory.append(result.avg_metrics.memory_peak_mb)
            allocated_memory.append(result.avg_metrics.memory_allocated_mb)
        
        # Plot peak memory
        ax1.plot(timestamps, peak_memory, 'o-', label='Peak Memory', color='red')
        ax1.set_ylabel('Peak Memory (MB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot allocated memory
        ax2.plot(timestamps, allocated_memory, 's-', label='Allocated Memory', color='blue')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Allocated Memory (MB)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64
    
    def create_pool_efficiency_chart(self, 
                                   results: List[BenchmarkResult],
                                   output_path: Optional[str] = None) -> Optional[str]:
        """
        Create a memory pool efficiency chart.
        
        Args:
            results: List of benchmark results
            output_path: Path to save plot
            
        Returns:
            Path to saved file or base64 encoded image data
        """
        if not results:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract pool efficiency data
        names = []
        hit_rates = []
        utilizations = []
        
        for result in results:
            if result.iteration_metrics:
                # Look for pool-related metrics
                for metrics in result.iteration_metrics[:1]:  # Just first iteration
                    custom = metrics.custom_metrics
                    if 'pool_hit_rate' in custom and 'pool_utilization' in custom:
                        names.append(result.name)
                        hit_rates.append(custom['pool_hit_rate'])
                        utilizations.append(custom['pool_utilization'])
                        break
        
        if not names:
            logger.warning("No pool efficiency data found in results")
            return None
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, hit_rates, width, label='Hit Rate', alpha=0.8)
        bars2 = ax.bar(x + width/2, utilizations, width, label='Utilization', alpha=0.8)
        
        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Rate (0-1)')
        ax.set_title('Memory Pool Efficiency')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', '\n') for name in names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return image_base64


class HTMLReporter:
    """Generate comprehensive HTML reports from benchmark results."""
    
    def __init__(self):
        """Initialize HTML reporter."""
        self.plot_reporter = PlotReporter() if HAS_MATPLOTLIB else None
    
    def export(self, results: List[BenchmarkResult], filepath: str) -> None:
        """
        Export results to HTML report.
        
        Args:
            results: List of benchmark results
            filepath: Output file path
        """
        html_content = self._generate_html(results)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {filepath}")
    
    def _generate_html(self, results: List[BenchmarkResult]) -> str:
        """Generate HTML content for the report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YinshML Benchmark Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>YinshML Memory Management Benchmark Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Benchmarks: {len(results)}</p>
        </header>
        
        <section class="summary">
            {self._generate_summary_section(results)}
        </section>
        
        <section class="visualizations">
            {self._generate_visualizations_section(results)}
        </section>
        
        <section class="detailed-results">
            {self._generate_detailed_results_section(results)}
        </section>
        
        <footer>
            <p>YinshML Benchmark Framework</p>
        </footer>
    </div>
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the HTML report."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .summary {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .summary-card {
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }
        
        .summary-card h4 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
        }
        
        .visualizations {
            margin-bottom: 30px;
        }
        
        .chart-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chart-container h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .results-table th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }
        
        .results-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .results-table tr:hover {
            background-color: #f1f1f1;
        }
        
        .metric-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .metric-badge.good {
            background-color: #2ecc71;
            color: white;
        }
        
        .metric-badge.warning {
            background-color: #f39c12;
            color: white;
        }
        
        .metric-badge.poor {
            background-color: #e74c3c;
            color: white;
        }
        
        footer {
            border-top: 1px solid #bdc3c7;
            padding-top: 20px;
            margin-top: 30px;
            text-align: center;
            color: #7f8c8d;
        }
        """
    
    def _generate_summary_section(self, results: List[BenchmarkResult]) -> str:
        """Generate summary section HTML."""
        if not results:
            return "<p>No results to display.</p>"
        
        # Calculate summary statistics
        total_iterations = sum(r.iterations for r in results)
        avg_duration = sum(r.avg_metrics.duration_ns for r in results) / len(results) / 1_000_000
        avg_memory = sum(r.avg_metrics.memory_peak_mb for r in results) / len(results)
        
        # Count memory pool benchmarks
        pool_benchmarks = len([r for r in results if 'pool' in r.name.lower() or 'memory' in r.name.lower()])
        
        html = f"""
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h4>Total Benchmarks</h4>
                <div class="metric-value">{len(results)}</div>
            </div>
            <div class="summary-card">
                <h4>Total Iterations</h4>
                <div class="metric-value">{total_iterations:,}</div>
            </div>
            <div class="summary-card">
                <h4>Average Duration</h4>
                <div class="metric-value">{avg_duration:.2f} ms</div>
            </div>
            <div class="summary-card">
                <h4>Average Memory Usage</h4>
                <div class="metric-value">{avg_memory:.2f} MB</div>
            </div>
            <div class="summary-card">
                <h4>Memory Pool Tests</h4>
                <div class="metric-value">{pool_benchmarks}</div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_visualizations_section(self, results: List[BenchmarkResult]) -> str:
        """Generate visualizations section HTML."""
        if not self.plot_reporter:
            return "<h2>Visualizations</h2><p>Visualization features require matplotlib to be installed.</p>"
        
        html = "<h2>Visualizations</h2>"
        
        # Performance comparison chart
        perf_chart = self.plot_reporter.create_performance_comparison(results)
        if perf_chart:
            html += f"""
            <div class="chart-container">
                <h3>Performance Comparison</h3>
                <img src="data:image/png;base64,{perf_chart}" alt="Performance Comparison Chart" />
            </div>
            """
        
        # Memory usage timeline
        memory_chart = self.plot_reporter.create_memory_usage_timeline(results)
        if memory_chart:
            html += f"""
            <div class="chart-container">
                <h3>Memory Usage Timeline</h3>
                <img src="data:image/png;base64,{memory_chart}" alt="Memory Usage Timeline" />
            </div>
            """
        
        # Pool efficiency chart
        pool_chart = self.plot_reporter.create_pool_efficiency_chart(results)
        if pool_chart:
            html += f"""
            <div class="chart-container">
                <h3>Memory Pool Efficiency</h3>
                <img src="data:image/png;base64,{pool_chart}" alt="Pool Efficiency Chart" />
            </div>
            """
        
        return html
    
    def _generate_detailed_results_section(self, results: List[BenchmarkResult]) -> str:
        """Generate detailed results section HTML."""
        html = "<h2>Detailed Results</h2>"
        
        # Create results table
        html += """
        <table class="results-table">
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Iterations</th>
                    <th>Avg Duration (ms)</th>
                    <th>Peak Memory (MB)</th>
                    <th>Success Rate</th>
                    <th>Custom Metrics</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for result in results:
            stats = result.calculate_statistics()
            success_rate = stats.get('success_rate', 1.0)
            
            # Determine success rate badge
            if success_rate >= 0.95:
                success_badge = 'good'
            elif success_rate >= 0.8:
                success_badge = 'warning'
            else:
                success_badge = 'poor'
            
            # Extract key custom metrics
            custom_metrics_str = ""
            if result.iteration_metrics:
                custom = result.iteration_metrics[0].custom_metrics
                key_metrics = []
                for key, value in custom.items():
                    if key in ['pool_hit_rate', 'allocations_per_sec', 'memory_efficiency']:
                        if isinstance(value, float):
                            key_metrics.append(f"{key}: {value:.3f}")
                        else:
                            key_metrics.append(f"{key}: {value}")
                custom_metrics_str = ", ".join(key_metrics[:3])  # Limit to 3 metrics
            
            html += f"""
                <tr>
                    <td><strong>{result.name}</strong><br><small>{result.description}</small></td>
                    <td>{result.iterations}</td>
                    <td>{result.avg_metrics.duration_ns / 1_000_000:.2f}</td>
                    <td>{result.avg_metrics.memory_peak_mb:.2f}</td>
                    <td><span class="metric-badge {success_badge}">{success_rate:.1%}</span></td>
                    <td><small>{custom_metrics_str}</small></td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html 