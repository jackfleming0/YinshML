"""
Comprehensive report generation system for experiment analysis.

This module provides functionality to generate detailed analysis reports
in multiple formats (HTML, PDF, Markdown) from experiment comparison results,
statistical analysis, and hyperparameter importance data.
"""

import os
import json
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

import jinja2
import numpy as np
import pandas as pd

from .comparator import ExperimentComparison
from .statistical_analysis import StatisticalTestResult, MultipleComparisonResult  
from .hyperparameter_analysis import ImportanceAnalysisResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Experiment Analysis Report"
    author: str = "YinshML Analysis System"
    include_executive_summary: bool = True
    include_experiment_overview: bool = True
    include_performance_comparison: bool = True
    include_statistical_analysis: bool = True
    include_hyperparameter_analysis: bool = True
    include_visualizations: bool = True
    include_conclusions: bool = True
    include_technical_appendix: bool = True
    logo_path: Optional[str] = None
    custom_css: Optional[str] = None


class ExperimentReportGenerator:
    """
    Comprehensive report generator for experiment analysis results.
    
    This class can generate reports in multiple formats (HTML, PDF, Markdown)
    combining experiment comparisons, statistical analysis, and visualizations
    into cohesive, publication-ready documents.
    """
    
    def __init__(self, 
                 template_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            template_dir: Directory containing custom templates
            output_dir: Default output directory for reports
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        template_loader = jinja2.FileSystemLoader([
            self.template_dir,
            self._get_builtin_template_dir()
        ])
        self.jinja_env = jinja2.Environment(
            loader=template_loader,
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Register custom filters
        self._register_custom_filters()
        
        logger.info(f"ExperimentReportGenerator initialized with templates from {self.template_dir}")
    
    def _get_default_template_dir(self) -> str:
        """Get the default template directory."""
        return str(Path(__file__).parent / "templates")
    
    def _get_builtin_template_dir(self) -> str:
        """Get the built-in template directory."""
        return str(Path(__file__).parent / "templates" / "builtin")
    
    def _register_custom_filters(self):
        """Register custom Jinja2 filters for report generation."""
        
        def format_number(value, precision=3):
            """Format numbers with appropriate precision."""
            if isinstance(value, (int, float)):
                if abs(value) < 0.001:
                    return f"{value:.2e}"
                elif abs(value) < 1:
                    return f"{value:.{precision}f}"
                else:
                    return f"{value:.{min(precision, 2)}f}"
            return str(value)
        
        def format_percentage(value, precision=1):
            """Format values as percentages."""
            if isinstance(value, (int, float)):
                return f"{value * 100:.{precision}f}%"
            return str(value)
        
        def format_pvalue(value):
            """Format p-values with scientific notation for very small values."""
            if isinstance(value, (int, float)):
                if value < 0.001:
                    return f"{value:.2e}"
                else:
                    return f"{value:.3f}"
            return str(value)
        
        def encode_image(image_path):
            """Encode image as base64 for embedding in HTML."""
            try:
                if isinstance(image_path, (str, Path)) and Path(image_path).exists():
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                    img_b64 = base64.b64encode(img_data).decode('utf-8')
                    ext = Path(image_path).suffix.lower()
                    mime_type = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg', 
                        '.jpeg': 'image/jpeg',
                        '.svg': 'image/svg+xml'
                    }.get(ext, 'image/png')
                    return f"data:{mime_type};base64,{img_b64}"
            except Exception as e:
                logger.warning(f"Failed to encode image {image_path}: {e}")
            return ""
        
        # Register filters
        self.jinja_env.filters['format_number'] = format_number
        self.jinja_env.filters['format_percentage'] = format_percentage
        self.jinja_env.filters['format_pvalue'] = format_pvalue
        self.jinja_env.filters['encode_image'] = encode_image
    
    def generate_html_report(self,
                           comparison_data: Optional[ExperimentComparison] = None,
                           statistical_results: Optional[Union[StatisticalTestResult, 
                                                             List[StatisticalTestResult],
                                                             MultipleComparisonResult]] = None,
                           importance_analysis: Optional[ImportanceAnalysisResult] = None,
                           visualizations: Optional[Dict[str, Any]] = None,
                           config: Optional[ReportConfig] = None,
                           template: str = 'default_html.html',
                           output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            comparison_data: Experiment comparison results
            statistical_results: Statistical analysis results  
            importance_analysis: Hyperparameter importance analysis
            visualizations: Dictionary of visualization data/paths
            config: Report configuration
            template: Template name to use
            output_path: Custom output path
            
        Returns:
            Path to generated HTML report
        """
        config = config or ReportConfig()
        
        # Prepare report data
        report_data = self._prepare_report_data(
            comparison_data, statistical_results, importance_analysis, 
            visualizations, config
        )
        
        # Load and render template
        try:
            template_obj = self.jinja_env.get_template(template)
        except jinja2.TemplateNotFound:
            logger.warning(f"Template {template} not found, using default")
            template_obj = self.jinja_env.from_string(self._get_default_html_template())
        
        html_content = template_obj.render(**report_data)
        
        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"experiment_report_{timestamp}.html"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return str(output_path)
    
    def generate_markdown_report(self,
                               comparison_data: Optional[ExperimentComparison] = None,
                               statistical_results: Optional[Union[StatisticalTestResult, 
                                                                 List[StatisticalTestResult],
                                                                 MultipleComparisonResult]] = None,
                               importance_analysis: Optional[ImportanceAnalysisResult] = None,
                               visualizations: Optional[Dict[str, Any]] = None,
                               config: Optional[ReportConfig] = None,
                               template: str = 'default_markdown.md',
                               output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Markdown report.
        
        Args:
            comparison_data: Experiment comparison results
            statistical_results: Statistical analysis results  
            importance_analysis: Hyperparameter importance analysis
            visualizations: Dictionary of visualization data/paths
            config: Report configuration
            template: Template name to use
            output_path: Custom output path
            
        Returns:
            Path to generated Markdown report
        """
        config = config or ReportConfig()
        
        # Prepare report data
        report_data = self._prepare_report_data(
            comparison_data, statistical_results, importance_analysis, 
            visualizations, config
        )
        
        # Load and render template
        try:
            template_obj = self.jinja_env.get_template(template)
        except jinja2.TemplateNotFound:
            logger.warning(f"Template {template} not found, using default")
            template_obj = self.jinja_env.from_string(self._get_default_markdown_template())
        
        markdown_content = template_obj.render(**report_data)
        
        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"experiment_report_{timestamp}.md"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write Markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report generated: {output_path}")
        return str(output_path)
    
    def generate_pdf_report(self,
                          comparison_data: Optional[ExperimentComparison] = None,
                          statistical_results: Optional[Union[StatisticalTestResult, 
                                                            List[StatisticalTestResult],
                                                            MultipleComparisonResult]] = None,
                          importance_analysis: Optional[ImportanceAnalysisResult] = None,
                          visualizations: Optional[Dict[str, Any]] = None,
                          config: Optional[ReportConfig] = None,
                          template: str = 'default_pdf.html',
                          output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive PDF report.
        
        Args:
            comparison_data: Experiment comparison results
            statistical_results: Statistical analysis results  
            importance_analysis: Hyperparameter importance analysis
            visualizations: Dictionary of visualization data/paths
            config: Report configuration
            template: Template name to use
            output_path: Custom output path
            
        Returns:
            Path to generated PDF report
        """
        # Try to import PDF generation library
        try:
            import weasyprint
            has_weasyprint = True
        except ImportError:
            has_weasyprint = False
        
        if not has_weasyprint:
            logger.warning("WeasyPrint not available, generating HTML version instead")
            html_path = self.generate_html_report(
                comparison_data, statistical_results, importance_analysis,
                visualizations, config, template='default_html.html'
            )
            logger.info(f"PDF generation not available. HTML report saved: {html_path}")
            return html_path
        
        # Generate HTML first
        config = config or ReportConfig()
        
        # Create temporary HTML file for PDF conversion
        html_content = self._generate_pdf_html_content(
            comparison_data, statistical_results, importance_analysis, 
            visualizations, config, template
        )
        
        # Convert to PDF
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"experiment_report_{timestamp}.pdf"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate PDF using WeasyPrint
        html_doc = weasyprint.HTML(string=html_content)
        html_doc.write_pdf(str(output_path))
        
        logger.info(f"PDF report generated: {output_path}")
        return str(output_path)
    
    def generate_all_formats(self,
                           comparison_data: Optional[ExperimentComparison] = None,
                           statistical_results: Optional[Union[StatisticalTestResult, 
                                                             List[StatisticalTestResult],
                                                             MultipleComparisonResult]] = None,
                           importance_analysis: Optional[ImportanceAnalysisResult] = None,
                           visualizations: Optional[Dict[str, Any]] = None,
                           config: Optional[ReportConfig] = None,
                           base_name: str = "experiment_report") -> Dict[str, str]:
        """
        Generate reports in all available formats.
        
        Args:
            comparison_data: Experiment comparison results
            statistical_results: Statistical analysis results  
            importance_analysis: Hyperparameter importance analysis
            visualizations: Dictionary of visualization data/paths
            config: Report configuration
            base_name: Base name for output files
            
        Returns:
            Dictionary mapping format names to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs = {}
        
        # Generate HTML report
        html_path = self.generate_html_report(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config,
            output_path=self.output_dir / f"{base_name}_{timestamp}.html"
        )
        outputs['html'] = html_path
        
        # Generate Markdown report
        md_path = self.generate_markdown_report(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config,
            output_path=self.output_dir / f"{base_name}_{timestamp}.md"
        )
        outputs['markdown'] = md_path
        
        # Generate PDF report (if available)
        pdf_path = self.generate_pdf_report(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config,
            output_path=self.output_dir / f"{base_name}_{timestamp}.pdf"
        )
        outputs['pdf'] = pdf_path
        
        logger.info(f"All format reports generated with base name: {base_name}_{timestamp}")
        return outputs
    
    def _prepare_report_data(self,
                           comparison_data: Optional[ExperimentComparison],
                           statistical_results: Optional[Union[StatisticalTestResult, 
                                                             List[StatisticalTestResult],
                                                             MultipleComparisonResult]],
                           importance_analysis: Optional[ImportanceAnalysisResult],
                           visualizations: Optional[Dict[str, Any]],
                           config: ReportConfig) -> Dict[str, Any]:
        """Prepare all data needed for report generation."""
        
        # Generate timestamp and metadata
        generation_time = datetime.now()
        
        report_data = {
            'config': config,
            'generation_time': generation_time,
            'timestamp_formatted': generation_time.strftime("%Y-%m-%d %H:%M:%S"),
            'comparison_data': comparison_data,
            'statistical_results': statistical_results,
            'importance_analysis': importance_analysis,
            'visualizations': visualizations or {},
            'has_comparison': comparison_data is not None,
            'has_statistical': statistical_results is not None,
            'has_importance': importance_analysis is not None,
            'has_visualizations': bool(visualizations)
        }
        
        # Process experiment comparison data
        if comparison_data:
            report_data.update(self._process_comparison_data(comparison_data))
        
        # Process statistical results
        if statistical_results:
            report_data.update(self._process_statistical_data(statistical_results))
        
        # Process hyperparameter importance
        if importance_analysis:
            report_data.update(self._process_importance_data(importance_analysis))
        
        return report_data
    
    def _process_comparison_data(self, comparison_data: ExperimentComparison) -> Dict[str, Any]:
        """Process experiment comparison data for reporting."""
        processed = {
            'experiment_count': len(comparison_data.experiment_ids),
            'experiment_ids': comparison_data.experiment_ids,
            'experiment_names': getattr(comparison_data, 'experiment_names', {}),
            'metrics': list(comparison_data.metric_comparisons.keys()),
            'comparison_timestamp': getattr(comparison_data, 'comparison_timestamp', 'N/A')
        }
        
        # Create summary tables
        if comparison_data.metric_comparisons:
            summary_tables = {}
            for metric_name, metric_data in comparison_data.metric_comparisons.items():
                table_data = []
                for exp_id, metrics in metric_data.items():
                    exp_name = processed['experiment_names'].get(exp_id, exp_id)
                    table_data.append({
                        'experiment_id': exp_id,
                        'experiment_name': exp_name,
                        'mean': metrics.mean,
                        'median': metrics.median,
                        'std': metrics.std,
                        'min': metrics.min,
                        'max': metrics.max,
                        'count': metrics.count
                    })
                summary_tables[metric_name] = table_data
            processed['summary_tables'] = summary_tables
        
        return processed
    
    def _process_statistical_data(self, 
                                statistical_results: Union[StatisticalTestResult, 
                                                         List[StatisticalTestResult],
                                                         MultipleComparisonResult]) -> Dict[str, Any]:
        """Process statistical analysis results for reporting."""
        processed = {'statistical_summary': []}
        
        if isinstance(statistical_results, list):
            for result in statistical_results:
                processed['statistical_summary'].append(self._format_statistical_result(result))
        elif isinstance(statistical_results, (StatisticalTestResult, MultipleComparisonResult)):
            processed['statistical_summary'].append(self._format_statistical_result(statistical_results))
        
        return processed
    
    def _format_statistical_result(self, result) -> Dict[str, Any]:
        """Format a single statistical result for display."""
        if hasattr(result, 'test_type'):
            return {
                'test_type': result.test_type,
                'test_name': result.test_type.replace('_', ' ').title(),  # For backwards compatibility
                'p_value': result.p_value,
                'statistic': getattr(result, 'statistic', 'N/A'),
                'effect_size': getattr(result, 'effect_size', 'N/A'),
                'confidence_interval': getattr(result, 'confidence_interval', 'N/A'),
                'interpretation': getattr(result, 'interpretation', 'N/A'),
                'significant': getattr(result, 'significant', False),
                'alpha': getattr(result, 'alpha', 0.05)
            }
        return {'raw_result': str(result)}
    
    def _process_importance_data(self, importance_analysis: ImportanceAnalysisResult) -> Dict[str, Any]:
        """Process hyperparameter importance data for reporting."""
        processed = {
            'target_metric': importance_analysis.target_metric,
            'hyperparameter_count': len(importance_analysis.hyperparameter_rankings),
            'model_performance': importance_analysis.model_performance,
            'data_summary': importance_analysis.data_summary
        }
        
        # Create importance ranking table
        importance_table = []
        for hp in importance_analysis.hyperparameter_rankings:
            importance_table.append({
                'hyperparameter': hp.hyperparameter,
                'composite_score': hp.composite_score,
                'correlation_importance': hp.correlation_importance,
                'rf_importance': hp.rf_importance,
                'lasso_importance': hp.lasso_importance,
                'p_value': getattr(hp, 'p_value', None),
                'confidence_interval': getattr(hp, 'confidence_interval', None)
            })
        
        processed['importance_table'] = importance_table
        return processed
    
    def _generate_pdf_html_content(self, *args, **kwargs) -> str:
        """Generate HTML content optimized for PDF conversion."""
        # For now, use the same HTML generation but with PDF-specific styling
        config = kwargs.get('config', ReportConfig())
        config.custom_css = self._get_pdf_css()
        kwargs['config'] = config
        
        # Generate HTML content (without saving to file)
        report_data = self._prepare_report_data(
            args[0] if args else None,  # comparison_data
            args[1] if len(args) > 1 else None,  # statistical_results
            args[2] if len(args) > 2 else None,  # importance_analysis
            args[3] if len(args) > 3 else None,  # visualizations
            config
        )
        
        template_obj = self.jinja_env.from_string(self._get_default_html_template())
        return template_obj.render(**report_data)
    
    def _get_default_html_template(self) -> str:
        """Get the default HTML template."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 { color: #333; }
        h1 { border-bottom: 3px solid #007acc; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th { background-color: #f8f9fa; font-weight: 600; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        .metric-value { font-weight: bold; color: #007acc; }
        .timestamp { color: #666; font-style: italic; }
        .section { margin: 30px 0; }
        .summary-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
        }
        {% if config.custom_css %}{{ config.custom_css }}{% endif %}
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ config.title }}</h1>
        <p class="timestamp">Generated on {{ timestamp_formatted }} by {{ config.author }}</p>
        
        {% if config.include_executive_summary %}
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <p><strong>Analysis Overview:</strong></p>
                <ul>
                    {% if has_comparison %}<li>Analyzed {{ experiment_count }} experiments across {{ metrics|length }} metrics</li>{% endif %}
                    {% if has_statistical %}<li>Statistical significance testing performed</li>{% endif %}
                    {% if has_importance %}<li>Hyperparameter importance analysis completed</li>{% endif %}
                    {% if has_visualizations %}<li>Comprehensive visualizations generated</li>{% endif %}
                </ul>
            </div>
        </div>
        {% endif %}
        
        {% if config.include_experiment_overview and has_comparison %}
        <div class="section">
            <h2>Experiment Overview</h2>
            <p><strong>Experiments Analyzed:</strong> {{ experiment_count }}</p>
            <p><strong>Metrics Evaluated:</strong> {{ metrics|join(', ') }}</p>
            <p><strong>Comparison Timestamp:</strong> {{ comparison_timestamp }}</p>
        </div>
        {% endif %}
        
        {% if config.include_performance_comparison and summary_tables %}
        <div class="section">
            <h2>Performance Comparison</h2>
            {% for metric_name, table_data in summary_tables.items() %}
            <h3>{{ metric_name.replace('_', ' ').title() }}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in table_data %}
                    <tr>
                        <td>{{ row.experiment_name or row.experiment_id }}</td>
                        <td class="metric-value">{{ row.mean|format_number }}</td>
                        <td>{{ row.median|format_number }}</td>
                        <td>{{ row.std|format_number }}</td>
                        <td>{{ row.min|format_number }}</td>
                        <td>{{ row.max|format_number }}</td>
                        <td>{{ row.count }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if config.include_statistical_analysis and statistical_summary %}
        <div class="section">
            <h2>Statistical Analysis</h2>
            {% for result in statistical_summary %}
            <div class="summary-box">
                <h4>{{ result.test_name or 'Statistical Test' }}</h4>
                <p><strong>P-value:</strong> {{ result.p_value|format_pvalue }}</p>
                {% if result.statistic != 'N/A' %}<p><strong>Test Statistic:</strong> {{ result.statistic|format_number }}</p>{% endif %}
                {% if result.effect_size != 'N/A' %}<p><strong>Effect Size:</strong> {{ result.effect_size|format_number }}</p>{% endif %}
                {% if result.interpretation != 'N/A' %}<p><strong>Interpretation:</strong> {{ result.interpretation }}</p>{% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if config.include_hyperparameter_analysis and importance_table %}
        <div class="section">
            <h2>Hyperparameter Analysis</h2>
            <p><strong>Target Metric:</strong> {{ target_metric }}</p>
            <p><strong>Model Performance:</strong></p>
            <ul>
                {% for key, value in model_performance.items() %}
                <li>{{ key.replace('_', ' ').title() }}: {{ value|format_number }}</li>
                {% endfor %}
            </ul>
            
            <h3>Hyperparameter Importance Rankings</h3>
            <table>
                <thead>
                    <tr>
                        <th>Hyperparameter</th>
                        <th>Composite Score</th>
                        <th>Correlation</th>
                        <th>Random Forest</th>
                        <th>LASSO</th>
                        {% if importance_table[0].p_value is not none %}<th>P-value</th>{% endif %}
                    </tr>
                </thead>
                <tbody>
                    {% for hp in importance_table %}
                    <tr>
                        <td><strong>{{ hp.hyperparameter }}</strong></td>
                        <td class="metric-value">{{ hp.composite_score|format_number }}</td>
                        <td>{{ hp.correlation_importance|format_number }}</td>
                        <td>{{ hp.rf_importance|format_number }}</td>
                        <td>{{ hp.lasso_importance|format_number }}</td>
                        {% if hp.p_value is not none %}<td>{{ hp.p_value|format_pvalue }}</td>{% endif %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% if config.include_visualizations and has_visualizations %}
        <div class="section">
            <h2>Visualizations</h2>
            <p>Interactive visualizations and charts are available in the analysis output.</p>
            {% for viz_name, viz_path in visualizations.items() %}
            <div>
                <h4>{{ viz_name.replace('_', ' ').title() }}</h4>
                {% if viz_path.endswith(('.png', '.jpg', '.jpeg', '.svg')) %}
                <img src="{{ viz_path|encode_image }}" alt="{{ viz_name }}" style="max-width: 100%; height: auto;">
                {% else %}
                <p>Visualization available at: <a href="{{ viz_path }}">{{ viz_path }}</a></p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Report Information</h2>
            <p><strong>Generated by:</strong> {{ config.author }}</p>
            <p><strong>Generated on:</strong> {{ timestamp_formatted }}</p>
            <p><strong>Analysis System:</strong> YinshML Experiment Analysis Framework</p>
        </div>
    </div>
</body>
</html>'''
    
    def _get_default_markdown_template(self) -> str:
        """Get the default Markdown template."""
        return '''# {{ config.title }}

**Generated on:** {{ timestamp_formatted }}  
**Author:** {{ config.author }}

---

{% if config.include_executive_summary %}
## Executive Summary

**Analysis Overview:**
{% if has_comparison %}
- Analyzed {{ experiment_count }} experiments across {{ metrics|length }} metrics
{% endif %}
{% if has_statistical %}
- Statistical significance testing performed
{% endif %}
{% if has_importance %}
- Hyperparameter importance analysis completed
{% endif %}
{% if has_visualizations %}
- Comprehensive visualizations generated
{% endif %}

{% endif %}

{% if config.include_experiment_overview and has_comparison %}
## Experiment Overview

- **Experiments Analyzed:** {{ experiment_count }}
- **Metrics Evaluated:** {{ metrics|join(', ') }}
- **Comparison Timestamp:** {{ comparison_timestamp }}

{% endif %}

{% if config.include_performance_comparison and summary_tables %}
## Performance Comparison

{% for metric_name, table_data in summary_tables.items() %}
### {{ metric_name.replace('_', ' ').title() }}

| Experiment | Mean | Median | Std Dev | Min | Max | Count |
|------------|------|--------|---------|-----|-----|-------|
{% for row in table_data %}
| {{ row.experiment_name or row.experiment_id }} | {{ row.mean|format_number }} | {{ row.median|format_number }} | {{ row.std|format_number }} | {{ row.min|format_number }} | {{ row.max|format_number }} | {{ row.count }} |
{% endfor %}

{% endfor %}
{% endif %}

{% if config.include_statistical_analysis and statistical_summary %}
## Statistical Analysis

{% for result in statistical_summary %}
### {{ result.test_name or 'Statistical Test' }}

- **P-value:** {{ result.p_value|format_pvalue }}
{% if result.statistic != 'N/A' %}
- **Test Statistic:** {{ result.statistic|format_number }}
{% endif %}
{% if result.effect_size != 'N/A' %}
- **Effect Size:** {{ result.effect_size|format_number }}
{% endif %}
{% if result.interpretation != 'N/A' %}
- **Interpretation:** {{ result.interpretation }}
{% endif %}

{% endfor %}
{% endif %}

{% if config.include_hyperparameter_analysis and importance_table %}
## Hyperparameter Analysis

- **Target Metric:** {{ target_metric }}

**Model Performance:**
{% for key, value in model_performance.items() %}
- {{ key.replace('_', ' ').title() }}: {{ value|format_number }}
{% endfor %}

### Hyperparameter Importance Rankings

| Hyperparameter | Composite Score | Correlation | Random Forest | LASSO |{% if importance_table[0].p_value is not none %} P-value |{% endif %}
|----------------|----------------|-------------|---------------|-------|{% if importance_table[0].p_value is not none %}---------|{% endif %}
{% for hp in importance_table %}
| **{{ hp.hyperparameter }}** | {{ hp.composite_score|format_number }} | {{ hp.correlation_importance|format_number }} | {{ hp.rf_importance|format_number }} | {{ hp.lasso_importance|format_number }} |{% if hp.p_value is not none %} {{ hp.p_value|format_pvalue }} |{% endif %}
{% endfor %}

{% endif %}

{% if config.include_visualizations and has_visualizations %}
## Visualizations

{% for viz_name, viz_path in visualizations.items() %}
### {{ viz_name.replace('_', ' ').title() }}

{% if viz_path.endswith(('.png', '.jpg', '.jpeg', '.svg')) %}
![{{ viz_name }}]({{ viz_path }})
{% else %}
Visualization available at: [{{ viz_path }}]({{ viz_path }})
{% endif %}

{% endfor %}
{% endif %}

---

**Report Information:**
- **Generated by:** {{ config.author }}
- **Generated on:** {{ timestamp_formatted }}
- **Analysis System:** YinshML Experiment Analysis Framework
'''
    
    def _get_pdf_css(self) -> str:
        """Get CSS optimized for PDF generation."""
        return '''
        @page {
            size: A4;
            margin: 2cm;
        }
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.4;
        }
        h1 { font-size: 18pt; page-break-after: avoid; }
        h2 { font-size: 14pt; page-break-after: avoid; margin-top: 20pt; }
        h3 { font-size: 12pt; page-break-after: avoid; }
        table { 
            page-break-inside: avoid;
            font-size: 9pt;
        }
        .section { page-break-inside: avoid; }
        '''


# Convenience function for quick report generation
def generate_experiment_report(comparison_data: Optional[ExperimentComparison] = None,
                             statistical_results: Optional[Union[StatisticalTestResult, 
                                                               List[StatisticalTestResult],
                                                               MultipleComparisonResult]] = None,
                             importance_analysis: Optional[ImportanceAnalysisResult] = None,
                             visualizations: Optional[Dict[str, Any]] = None,
                             output_format: str = 'html',
                             output_path: Optional[str] = None,
                             config: Optional[ReportConfig] = None) -> str:
    """
    Convenience function to generate a single experiment report.
    
    Args:
        comparison_data: Experiment comparison results
        statistical_results: Statistical analysis results  
        importance_analysis: Hyperparameter importance analysis
        visualizations: Dictionary of visualization data/paths
        output_format: Format to generate ('html', 'pdf', 'markdown', 'all')
        output_path: Custom output path
        config: Report configuration
        
    Returns:
        Path to generated report (or dict of paths if format='all')
    """
    generator = ExperimentReportGenerator()
    
    if output_format.lower() == 'html':
        return generator.generate_html_report(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config, output_path=output_path
        )
    elif output_format.lower() == 'pdf':
        return generator.generate_pdf_report(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config, output_path=output_path
        )
    elif output_format.lower() == 'markdown':
        return generator.generate_markdown_report(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config, output_path=output_path
        )
    elif output_format.lower() == 'all':
        return generator.generate_all_formats(
            comparison_data, statistical_results, importance_analysis,
            visualizations, config
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}") 