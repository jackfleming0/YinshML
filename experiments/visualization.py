"""Visualization utilities for experiment comparison and analysis results."""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Static visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Interactive visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Internal imports
from .comparator import ExperimentComparator, ExperimentComparison, ComparisonMetrics
from .statistical_analysis import StatisticalTestResult, MultipleComparisonResult
from .hyperparameter_analysis import ImportanceAnalysisResult, HyperparameterImportance

# Configure logging
logger = logging.getLogger(__name__)

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')


class ExperimentVisualizer:
    """Comprehensive visualization utilities for experiment analysis results."""
    
    def __init__(
        self,
        style: str = 'whitegrid',
        palette: str = 'husl',
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        save_format: str = 'png'
    ):
        """Initialize the ExperimentVisualizer."""
        self.style = style
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi
        self.save_format = save_format
        
        # Set up matplotlib and seaborn styling
        sns.set_style(self.style)
        sns.set_palette(self.palette)
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        
        # Color schemes for different chart types
        self.color_schemes = {
            'categorical': sns.color_palette(self.palette, 10),
            'sequential': sns.color_palette('viridis', 10),
            'diverging': sns.color_palette('RdBu', 10),
            'statistical': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        }
        
        logger.info(f"ExperimentVisualizer initialized with style='{style}', palette='{palette}'")
    
    def plot_experiment_comparison(
        self,
        comparison_result: ExperimentComparison,
        metric: str = 'elo_changes',
        chart_type: str = 'bar',
        save_path: Optional[Path] = None,
        interactive: bool = False
    ) -> Union[Figure, go.Figure]:
        """
        Create experiment comparison visualization.
        
        Args:
            comparison_result: Results from ExperimentComparator.compare_experiments()
            metric: Metric to visualize
            chart_type: Type of chart ('bar', 'box')
            save_path: Path to save the figure
            interactive: Whether to create interactive plot with Plotly
            
        Returns:
            Matplotlib Figure or Plotly Figure object
        """
        logger.info(f"Creating experiment comparison plot for metric '{metric}' (type: {chart_type})")
        
        # Extract data from comparison result
        if metric not in comparison_result.metric_comparisons:
            logger.warning(f"Metric '{metric}' not found in comparison results")
            available_metrics = list(comparison_result.metric_comparisons.keys())
            logger.info(f"Available metrics: {available_metrics}")
            if available_metrics:
                metric = available_metrics[0]
                logger.info(f"Using metric '{metric}' instead")
            else:
                raise ValueError("No metrics available in comparison results")
        
        metric_data = comparison_result.metric_comparisons[metric]
        experiment_names = [comparison_result.experiment_names.get(exp_id, str(exp_id)) 
                          for exp_id in comparison_result.experiment_ids]
        
        if interactive:
            return self._plot_comparison_interactive(
                metric_data, experiment_names, metric, chart_type, save_path
            )
        else:
            return self._plot_comparison_static(
                metric_data, experiment_names, metric, chart_type, save_path
            )
    
    def _plot_comparison_static(
        self,
        metric_data: Dict[Union[str, int], ComparisonMetrics],
        experiment_names: List[str],
        metric: str,
        chart_type: str,
        save_path: Optional[Path]
    ) -> Figure:
        """Create static comparison plot using matplotlib."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Extract values for plotting
        means = []
        stds = []
        experiment_labels = []
        
        for exp_id, name in zip(metric_data.keys(), experiment_names):
            metrics = metric_data[exp_id]
            means.append(metrics.mean)
            stds.append(metrics.std)
            experiment_labels.append(name)
        
        colors = self.color_schemes['categorical'][:len(experiment_labels)]
        
        if chart_type == 'bar':
            bars = ax.bar(experiment_labels, means, yerr=stds, capsize=5, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_ylabel(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Experiment')
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Styling
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Rotate x-axis labels if there are many experiments
        if len(experiment_labels) > 5:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, format=self.save_format, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Static plot saved to {save_path}")
        
        return fig
    
    def _plot_comparison_interactive(
        self,
        metric_data: Dict[Union[str, int], ComparisonMetrics],
        experiment_names: List[str],
        metric: str,
        chart_type: str,
        save_path: Optional[Path]
    ) -> go.Figure:
        """Create interactive comparison plot using Plotly."""
        fig = go.Figure()
        
        # Extract values for plotting
        means = []
        stds = []
        experiment_labels = []
        
        for exp_id, name in zip(metric_data.keys(), experiment_names):
            metrics = metric_data[exp_id]
            means.append(metrics.mean)
            stds.append(metrics.std)
            experiment_labels.append(name)
        
        if chart_type == 'bar':
            fig.add_trace(go.Bar(
                x=experiment_labels,
                y=means,
                error_y=dict(type='data', array=stds),
                marker=dict(color=px.colors.qualitative.Set3[:len(experiment_labels)]),
                text=[f'{mean:.2f}' for mean in means],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                             f'{metric}: %{{y:.2f}} ± %{{error_y.array:.2f}}<br>' +
                             '<extra></extra>'
            ))
        
        # Layout and styling
        title = f'{metric.replace("_", " ").title()} Comparison'
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title="Experiment",
            yaxis_title=f'{metric.replace("_", " ").title()}',
            template='plotly_white',
            font=dict(size=12),
            showlegend=False,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        # Save if path provided
        if save_path:
            html_path = save_path.with_suffix('.html')
            fig.write_html(html_path)
            logger.info(f"Interactive plot saved to {html_path}")
        
        return fig
    
    def plot_hyperparameter_importance(
        self,
        importance_result: ImportanceAnalysisResult,
        top_n: int = 10,
        include_all_scores: bool = True,
        save_path: Optional[Path] = None,
        interactive: bool = False
    ) -> Union[Figure, go.Figure]:
        """
        Create hyperparameter importance visualization.
        
        Args:
            importance_result: Results from HyperparameterAnalyzer
            top_n: Number of top hyperparameters to show
            include_all_scores: Whether to show individual importance scores
            save_path: Path to save the figure
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib Figure or Plotly Figure object
        """
        logger.info(f"Creating hyperparameter importance plot (top {top_n})")
        
        # Get top N hyperparameters by composite score
        top_hyperparams = importance_result.hyperparameter_rankings[:top_n]
        
        if interactive:
            return self._plot_hyperparameter_importance_interactive(
                top_hyperparams, include_all_scores, save_path
            )
        else:
            return self._plot_hyperparameter_importance_static(
                top_hyperparams, include_all_scores, save_path
            )
    
    def _plot_hyperparameter_importance_static(
        self,
        hyperparams: List[HyperparameterImportance],
        include_all_scores: bool,
        save_path: Optional[Path]
    ) -> Figure:
        """Create static hyperparameter importance plot."""
        if include_all_scores:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax1 = plt.subplots(figsize=self.figure_size)
        
        # Main composite score plot
        names = [hp.hyperparameter for hp in hyperparams]
        scores = [hp.composite_score for hp in hyperparams]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax1.barh(names, scores, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Composite Importance Score')
        ax1.set_title('Hyperparameter Importance Rankings', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # Individual scores breakdown plot
        if include_all_scores:
            score_data = {
                'Correlation': [hp.correlation_importance for hp in hyperparams],
                'Random Forest': [hp.rf_importance for hp in hyperparams],
                'LASSO': [hp.lasso_importance for hp in hyperparams]
            }
            
            x = np.arange(len(names))
            width = 0.25
            
            for i, (score_type, values) in enumerate(score_data.items()):
                offset = (i - 1) * width
                bars = ax2.bar(x + offset, values, width, label=score_type, 
                              alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax2.set_xlabel('Hyperparameters')
            ax2.set_ylabel('Importance Score')
            ax2.set_title('Individual Importance Scores', fontsize=16, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, format=self.save_format, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Hyperparameter importance plot saved to {save_path}")
        
        return fig
    
    def _plot_hyperparameter_importance_interactive(
        self,
        hyperparams: List[HyperparameterImportance],
        include_all_scores: bool,
        save_path: Optional[Path]
    ) -> go.Figure:
        """Create interactive hyperparameter importance plot."""
        if include_all_scores:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Composite Importance', 'Individual Scores'),
                horizontal_spacing=0.12
            )
        else:
            fig = go.Figure()
        
        # Main composite score plot
        names = [hp.hyperparameter for hp in hyperparams]
        scores = [hp.composite_score for hp in hyperparams]
        
        trace1 = go.Bar(
            y=names,
            x=scores,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score")
            ),
            text=[f'{score:.3f}' for score in scores],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Composite Score: %{x:.3f}<extra></extra>'
        )
        
        if include_all_scores:
            fig.add_trace(trace1, row=1, col=1)
        else:
            fig.add_trace(trace1)
        
        # Individual scores breakdown plot
        if include_all_scores:
            score_data = {
                'Correlation': [hp.correlation_importance for hp in hyperparams],
                'Random Forest': [hp.rf_importance for hp in hyperparams],
                'LASSO': [hp.lasso_importance for hp in hyperparams]
            }
            
            for score_type, values in score_data.items():
                fig.add_trace(go.Bar(
                    x=names,
                    y=values,
                    name=score_type,
                    hovertemplate='<b>%{x}</b><br>' + f'{score_type}: %{{y:.3f}}<extra></extra>'
                ), row=1, col=2)
        
        # Layout and styling
        title = 'Hyperparameter Importance Analysis'
        
        if include_all_scores:
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=18)),
                template='plotly_white',
                height=600,
                showlegend=True,
                barmode='group'
            )
            fig.update_xaxes(title_text="Composite Score", row=1, col=1)
            fig.update_yaxes(title_text="Hyperparameters", row=1, col=1)
            fig.update_xaxes(title_text="Hyperparameters", row=1, col=2)
            fig.update_yaxes(title_text="Importance Score", row=1, col=2)
        else:
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=18)),
                xaxis_title="Composite Score",
                yaxis_title="Hyperparameters",
                template='plotly_white',
                height=600,
                showlegend=False
            )
        
        # Save if path provided
        if save_path:
            html_path = save_path.with_suffix('.html')
            fig.write_html(html_path)
            logger.info(f"Interactive hyperparameter importance plot saved to {html_path}")
        
        return fig
    
    def plot_correlation_heatmap(
        self,
        importance_result: ImportanceAnalysisResult,
        save_path: Optional[Path] = None,
        interactive: bool = False
    ) -> Union[Figure, go.Figure]:
        """
        Create correlation heatmap for hyperparameters.
        
        Args:
            importance_result: Results from HyperparameterAnalyzer
            save_path: Path to save the figure
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib Figure or Plotly Figure object
        """
        logger.info("Creating hyperparameter correlation heatmap")
        
        correlation_matrix = importance_result.correlation_matrix
        
        if interactive:
            return self._plot_correlation_heatmap_interactive(correlation_matrix, save_path)
        else:
            return self._plot_correlation_heatmap_static(correlation_matrix, save_path)
    
    def _plot_correlation_heatmap_static(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        save_path: Optional[Path]
    ) -> Figure:
        """Create static correlation heatmap."""
        # Convert to pandas DataFrame for easier plotting
        df = pd.DataFrame(correlation_matrix)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create heatmap
        sns.heatmap(df, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('Hyperparameter Correlation Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Hyperparameters')
        ax.set_ylabel('Hyperparameters')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, format=self.save_format, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        return fig
    
    def _plot_correlation_heatmap_interactive(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        save_path: Optional[Path]
    ) -> go.Figure:
        """Create interactive correlation heatmap."""
        # Convert to arrays for Plotly
        variables = list(correlation_matrix.keys())
        z_matrix = [[correlation_matrix[var1][var2] for var2 in variables] for var1 in variables]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=variables,
            y=variables,
            colorscale='RdBu',
            zmid=0,
            text=[[f'{correlation_matrix[var1][var2]:.3f}' for var2 in variables] for var1 in variables],
            texttemplate='%{text}',
            textfont={'size': 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text='Hyperparameter Correlation Matrix', x=0.5, font=dict(size=18)),
            xaxis_title='Hyperparameters',
            yaxis_title='Hyperparameters',
            template='plotly_white',
            height=600,
            width=600
        )
        
        # Save if path provided
        if save_path:
            html_path = save_path.with_suffix('.html')
            fig.write_html(html_path)
            logger.info(f"Interactive correlation heatmap saved to {html_path}")
        
        return fig 