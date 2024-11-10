"""Interactive dashboard for YINSH training experiments."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from experiments.config import RESULTS_SUBDIRS
from experiments.tracker import MetricsTracker, ExperimentMetrics


class ExperimentDashboard:
    def __init__(self):
        self.logger = logging.getLogger("ExperimentDashboard")
        self.tracker = MetricsTracker()

    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="YINSH Training Analysis",
            page_icon="🎮",
            layout="wide"
        )

        st.title("YINSH Training Experiments Dashboard")

        # Sidebar navigation
        experiment_type = st.sidebar.selectbox(
            "Experiment Type",
            ["learning_rate", "mcts", "temperature"]
        )

        # Load data
        metrics_by_config = self.tracker.load_experiment_results(experiment_type)
        if not metrics_by_config:
            st.error(f"No results found for {experiment_type} experiments")
            return

        # Analysis summary
        self._show_analysis_summary(experiment_type)

        # Main content
        col1, col2 = st.columns(2)

        with col1:
            self._plot_learning_curves(metrics_by_config)
            self._plot_elo_progression(metrics_by_config)

        with col2:
            self._plot_training_stability(metrics_by_config)
            if experiment_type == "temperature":
                self._plot_entropy_analysis(metrics_by_config)
            else:
                self._plot_game_length_distribution(metrics_by_config)

        # Detailed metrics
        st.subheader("Detailed Metrics")
        tabs = st.tabs(["Learning", "Performance", "Stability"])

        with tabs[0]:
            self._show_learning_metrics(metrics_by_config)
        with tabs[1]:
            self._show_performance_metrics(metrics_by_config)
        with tabs[2]:
            self._show_stability_metrics(metrics_by_config)

    def _show_analysis_summary(self, experiment_type: str):
        """Show summary of experiment analysis."""
        recommendations = self.tracker.get_recommendations(experiment_type)
        analysis = self.tracker.analyze_learning_dynamics(experiment_type)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Best Configuration",
                recommendations['best_configuration'],
                f"+{recommendations['improvement_potential']:.1%}" if 'improvement_potential' in recommendations else None
            )

        with col2:
            if 'best_configuration' in recommendations:
                best_row = analysis[analysis['config'] == recommendations['best_configuration']]
                baseline_row = analysis[analysis['config'] == 'baseline']

                if not best_row.empty:
                    # Get baseline ELO if available
                    baseline_elo = (
                        baseline_row.iloc[0]['final_elo']
                        if not baseline_row.empty
                        else 0.0
                    )

                    current_elo = best_row.iloc[0]['final_elo']
                    elo_change = current_elo - baseline_elo

                    st.metric(
                        "Final ELO",
                        f"{current_elo:.1f}",
                        f"{elo_change:+.1f}" if baseline_elo != 0 else None
                    )
                else:
                    st.metric("Final ELO", "No data")

        with col3:
            if 'best_configuration' in recommendations:
                best_row = analysis[analysis['config'] == recommendations['best_configuration']]
                baseline_row = analysis[analysis['config'] == 'baseline']

                if not best_row.empty:
                    # Get baseline training time if available
                    baseline_time = (
                        baseline_row.iloc[0]['training_time']
                        if not baseline_row.empty
                        else 0.0
                    )

                    current_time = best_row.iloc[0]['training_time']
                    time_change = current_time - baseline_time

                    st.metric(
                        "Training Time",
                        f"{current_time / 3600:.1f}h",
                        f"{time_change / 3600:+.1f}h" if baseline_time != 0 else None
                    )
                else:
                    st.metric("Training Time", "No data")

        # Show recommendations
        if recommendations.get('suggestions'):
            with st.expander("Recommendations"):
                for suggestion in recommendations['suggestions']:
                    st.write(f"• {suggestion}")
        else:
            st.info("No specific recommendations available yet.")

        # Show raw analysis data if requested
        if st.checkbox("Show Raw Analysis Data"):
            st.dataframe(analysis)

    def _plot_learning_curves(self, metrics_by_config: Dict):
        """Plot learning curves with Plotly."""
        st.subheader("Learning Curves")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Policy Loss", "Value Loss")
        )

        for config, metrics in metrics_by_config.items():
            # Policy loss
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(metrics.policy_losses))),
                    y=metrics.policy_losses,
                    name=f"{config} - Policy",
                    mode='lines+markers'
                ),
                row=1, col=1
            )

            # Value loss
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(metrics.value_losses))),
                    y=metrics.value_losses,
                    name=f"{config} - Value",
                    mode='lines+markers'
                ),
                row=2, col=1
            )

        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Iteration")
        fig.update_yaxes(title_text="Loss")

        st.plotly_chart(fig, use_container_width=True)

    def _plot_elo_progression(self, metrics_by_config: Dict):
        """Plot ELO progression."""
        st.subheader("ELO Progression")

        fig = go.Figure()

        for config, metrics in metrics_by_config.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(metrics.elo_changes))),
                    y=metrics.elo_changes,
                    name=config,
                    mode='lines+markers'
                )
            )

        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="ELO Change",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_training_stability(self, metrics_by_config: Dict):
        """Plot training stability metrics."""
        st.subheader("Training Stability")

        # Calculate stability metrics
        stability_data = []
        for config, metrics in metrics_by_config.items():
            policy_stability = self.tracker._calculate_stability(metrics.policy_losses)
            value_stability = self.tracker._calculate_stability(metrics.value_losses)
            elo_stability = self.tracker._calculate_stability(metrics.elo_changes)

            stability_data.append({
                'config': config,
                'Policy Loss': policy_stability,
                'Value Loss': value_stability,
                'ELO': elo_stability
            })

        df = pd.DataFrame(stability_data)

        fig = go.Figure(data=[
            go.Bar(name='Policy Loss', x=df['config'], y=df['Policy Loss']),
            go.Bar(name='Value Loss', x=df['config'], y=df['Value Loss']),
            go.Bar(name='ELO', x=df['config'], y=df['ELO'])
        ])

        fig.update_layout(
            barmode='group',
            xaxis_title="Configuration",
            yaxis_title="Stability Score",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_entropy_analysis(self, metrics_by_config: Dict):
        """Plot move entropy analysis for temperature experiments."""
        st.subheader("Move Diversity")

        fig = go.Figure()

        for config, metrics in metrics_by_config.items():
            if metrics.move_entropies:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(metrics.move_entropies))),
                        y=metrics.move_entropies,
                        name=config,
                        mode='lines+markers'
                    )
                )

        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Move Entropy",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_game_length_distribution(self, metrics_by_config: Dict):
        """Plot game length distribution."""
        st.subheader("Game Length Distribution")

        fig = go.Figure()

        for config, metrics in metrics_by_config.items():
            fig.add_trace(
                go.Histogram(
                    x=metrics.game_lengths,
                    name=config,
                    nbinsx=20,
                    opacity=0.7
                )
            )

        fig.update_layout(
            barmode='overlay',
            xaxis_title="Game Length",
            yaxis_title="Count",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _show_learning_metrics(self, metrics_by_config: Dict):
        """Show detailed learning metrics."""
        data = []
        for config, metrics in metrics_by_config.items():
            data.append({
                'Configuration': config,
                'Final Policy Loss': metrics.policy_losses[-1],
                'Final Value Loss': metrics.value_losses[-1],
                'Policy Loss Trend': self.tracker._calculate_trend(metrics.policy_losses),
                'Value Loss Trend': self.tracker._calculate_trend(metrics.value_losses),
                'Policy Converged': self.tracker._check_convergence(metrics.policy_losses),
                'Value Converged': self.tracker._check_convergence(metrics.value_losses)
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

    def _show_performance_metrics(self, metrics_by_config: Dict):
        """Show detailed performance metrics."""
        data = []
        for config, metrics in metrics_by_config.items():
            data.append({
                'Configuration': config,
                'Final ELO': metrics.elo_changes[-1],
                'Peak ELO': max(metrics.elo_changes),
                'Average Game Length': np.mean(metrics.game_lengths),
                'Total Training Time': sum(metrics.timestamps) / 3600  # Convert to hours
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

    def _show_stability_metrics(self, metrics_by_config: Dict):
        """Show detailed stability metrics."""
        data = []
        for config, metrics in metrics_by_config.items():
            data.append({
                'Configuration': config,
                'Policy Stability': self.tracker._calculate_stability(metrics.policy_losses),
                'Value Stability': self.tracker._calculate_stability(metrics.value_losses),
                'ELO Stability': self.tracker._calculate_stability(metrics.elo_changes),
                'Game Length Variance': np.std(metrics.game_lengths)
            })

        df = pd.DataFrame(data)
        st.dataframe(df)


def main():
    """Run the dashboard."""
    dashboard = ExperimentDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()