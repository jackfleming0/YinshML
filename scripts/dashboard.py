"""Training dashboard for YINSH ML model."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
import numpy as np
from typing import List, Dict
import glob
import logging

from train import get_mode_settings

logger = logging.getLogger(__name__)

def load_metrics(model_dir: Path) -> List[Dict]:
    """Load all metrics files from a training directory."""
    metrics_files = model_dir.glob("iteration_*/metrics.json")
    all_metrics = []

    for metrics_file in sorted(metrics_files, key=lambda x: x.parent.name):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

                # Add iteration info from directory name
                iteration = int(metrics_file.parent.name.split('_')[1])
                metrics['iteration'] = iteration

                # Ensure all required fields exist with proper types
                required_fields = {
                    'game_lengths': 0.0,
                    'ring_mobility': 0.0,
                    'win_rates': 0.0,
                    'draw_rates': 0.0,
                    'policy_losses': 0.0,
                    'value_losses': 0.0,
                    'avg_temperature': 0.0,
                    'early_game_entropy': 0.0,
                    'late_game_entropy': 0.0,
                    'move_selection_confidence': 0.0
                }

                # Fill in missing fields with defaults
                for field, default in required_fields.items():
                    if field not in metrics:
                        logger.warning(f"Missing field {field} in {metrics_file}, using default {default}")
                        metrics[field] = default
                    else:
                        # Ensure numeric type
                        metrics[field] = float(metrics[field])

                all_metrics.append(metrics)

        except Exception as e:
            logger.error(f"Error loading metrics from {metrics_file}: {str(e)}")
            continue

    return all_metrics

def plot_loss_curves(df: pd.DataFrame):
    """Plot policy and value loss curves with temperature metrics."""
    fig = go.Figure()

    # Policy loss
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['policy_losses'],
        name='Policy Loss',
        line=dict(color='blue', width=2),
        mode='lines+markers'
    ))

    # Value loss
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['value_losses'],
        name='Value Loss',
        line=dict(color='red', width=2),
        mode='lines+markers'
    ))

    # Add temperature if available
    if 'avg_temperature' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['avg_temperature'],
            name='Temperature',
            line=dict(color='green', width=2, dash='dot'),
            yaxis='y2'
        ))

    fig.update_layout(
        title='Training Losses & Temperature',
        xaxis_title='Iteration',
        yaxis_title='Loss',
        yaxis2=dict(
            title='Temperature',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_entropy_curves(df: pd.DataFrame):
    """Plot early vs late game entropy."""
    if 'early_game_entropy' in df.columns and 'late_game_entropy' in df.columns:
        fig = go.Figure()

        # Early game entropy
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['early_game_entropy'],
            name='Early Game Entropy',
            line=dict(color='orange', width=2),
            mode='lines+markers'
        ))

        # Late game entropy
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['late_game_entropy'],
            name='Late Game Entropy',
            line=dict(color='purple', width=2),
            mode='lines+markers'
        ))

        fig.update_layout(
            title='Policy Entropy Over Game Phases',
            xaxis_title='Iteration',
            yaxis_title='Entropy',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_metrics_df(metrics: List[Dict]) -> pd.DataFrame:
    """Convert metrics list to DataFrame with proper type conversion."""
    df = pd.DataFrame(metrics)

    # Explicitly convert columns to float
    numeric_columns = ['ring_mobility', 'game_lengths', 'win_rates',
                       'draw_rates', 'policy_losses', 'value_losses',
                       'avg_temperature', 'early_game_entropy', 'late_game_entropy',
                       'move_selection_confidence']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Add validation
            logger.debug(f"Column {col} range: {df[col].min()}-{df[col].max()}")

    return df

def plot_temperature_metrics(df: pd.DataFrame):
    """Plot temperature-related metrics."""
    fig = go.Figure()

    # Temperature
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['avg_temperature'],
        name='Average Temperature',
        line=dict(color='green', width=2),
        mode='lines+markers'
    ))

    # Move selection confidence
    if 'move_selection_confidence' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['move_selection_confidence'],
            name='Move Confidence',
            line=dict(color='blue', width=2, dash='dot'),
            mode='lines+markers'
        ))

    fig.update_layout(
        title='Temperature and Move Selection Confidence',
        xaxis_title='Iteration',
        yaxis_title='Value',
        yaxis_range=[0, 1],
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_win_rates(df: pd.DataFrame):
    """Plot win and draw rates."""
    fig = go.Figure()

    # Win rates
    win_cols = [col for col in df.columns if col.startswith('win_rates_')]
    if win_cols:
        win_rates = df[win_cols].mean(axis=1)
        fig.add_trace(go.Scatter(
            y=win_rates,
            name='Win Rate',
            line=dict(color='green', width=2)
        ))

    # Draw rates
    draw_cols = [col for col in df.columns if col.startswith('draw_rates_')]
    if draw_cols:
        draw_rates = df[draw_cols].mean(axis=1)
        fig.add_trace(go.Scatter(
            y=draw_rates,
            name='Draw Rate',
            line=dict(color='gray', width=2)
        ))

    fig.update_layout(
        title='Game Outcomes',
        xaxis_title='Iteration',
        yaxis_title='Rate',
        height=400,
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig, use_container_width=True)


def load_game_data(model_dir: Path, iteration: int) -> np.ndarray:
    """Load game data for a specific iteration."""
    game_file = model_dir / f"iteration_{iteration}/games.npy"
    if game_file.exists():
        return np.load(game_file, allow_pickle=True).item()
    return None

def plot_ring_mobility(df: pd.DataFrame):
    """Plot ring mobility over time."""
    fig = go.Figure()

    # Check if we have the ring_mobility column
    if 'ring_mobility' in df.columns:
        fig.add_trace(go.Scatter(
            y=df['ring_mobility'],  # Use single column instead of trying to average
            name='Ring Mobility',
            line=dict(color='purple', width=2)
        ))

        fig.update_layout(
            title='Average Ring Mobility',
            xaxis_title='Iteration',
            yaxis_title='Moves per Ring',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No ring mobility data available yet")

def plot_game_length_histogram(df: pd.DataFrame):
    """Plot histogram of game lengths."""
    if 'game_lengths' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['game_lengths'].iloc[-20:],  # Last 20 games
            nbinsx=20,
            name='Game Length Distribution'
        ))
        fig.update_layout(
            title='Recent Game Length Distribution',
            xaxis_title='Number of Moves',
            yaxis_title='Frequency',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_win_distribution(df: pd.DataFrame):
    """Plot pie chart of game outcomes."""
    if 'win_rates' in df.columns and 'draw_rates' in df.columns:
        latest_idx = df.index[-1]
        win_rate = df.loc[latest_idx, 'win_rates']
        draw_rate = df.loc[latest_idx, 'draw_rates']
        loss_rate = 1 - win_rate - draw_rate

        fig = go.Figure(data=[go.Pie(
            labels=['Wins', 'Draws', 'Losses'],
            values=[win_rate, draw_rate, loss_rate],
            hole=.3
        )])
        fig.update_layout(title='Game Outcome Distribution')
        st.plotly_chart(fig, use_container_width=True)

def check_training_stability(df: pd.DataFrame) -> Dict[str, bool]:
    """Check if training metrics indicate stability."""
    recent = df.iloc[-5:]

    checks = {
        'policy_loss': recent['policy_losses'].diff().mean() < 0,
        'value_loss': recent['value_losses'].diff().mean() < 0,
        'ring_mobility': recent['ring_mobility'].mean() > 2.0,
        'game_length': 20 < recent['avg_game_lengths'].mean() < 200
    }

    return {
        'stable': all(checks.values()),
        'reason': ', '.join(k for k, v in checks.items() if not v)
    }

def show_config_info(mode: str):
    """Display training configuration."""
    settings = get_mode_settings(mode)
    with st.expander("Training Configuration"):
        config_cols = st.columns(4)
        with config_cols[0]:
            st.metric("Games per Iteration", settings['games_per_iteration'])
        with config_cols[1]:
            st.metric("MCTS Simulations", settings['mcts_simulations'])
        with config_cols[2]:
            st.metric("Epochs per Iteration", settings['epochs_per_iteration'])
        with config_cols[3]:
            st.metric("Total Iterations", settings['num_iterations'])

def show_training_status(df: pd.DataFrame):
    """Show training status indicators."""
    status_cols = st.columns(5)

    # Policy Loss Trend
    with status_cols[0]:
        policy_trend = df['policy_losses'].diff().mean()
        st.metric(
            "Policy Loss Trend",
            f"{policy_trend:.4f}",
            delta=policy_trend,
            delta_color="inverse"
        )

    # Ring Mobility Trend
    with status_cols[1]:
        mobility_trend = df['ring_mobility'].diff().mean()
        st.metric(
            "Mobility Trend",
            f"{mobility_trend:.2f}",
            delta=mobility_trend,
            delta_color="normal"
        )

    # Temperature Metrics
    with status_cols[2]:
        if 'avg_temperature' in df.columns:
            current_temp = df['avg_temperature'].iloc[-1]
            temp_trend = df['avg_temperature'].diff().mean()
            st.metric(
                "Avg Temperature",
                f"{current_temp:.2f}",
                delta=temp_trend,
                delta_color="normal"
            )

    # Early vs Late Game Entropy
    with status_cols[3]:
        if 'early_game_entropy' in df.columns and 'late_game_entropy' in df.columns:
            entropy_diff = df['early_game_entropy'].iloc[-1] - df['late_game_entropy'].iloc[-1]
            st.metric(
                "Early-Late Entropy Diff",
                f"{entropy_diff:.2f}",
                delta=entropy_diff,
                delta_color="normal"
            )

    # Move Selection Confidence
    with status_cols[4]:
        if 'move_selection_confidence' in df.columns:
            confidence = df['move_selection_confidence'].iloc[-1]
            confidence_trend = df['move_selection_confidence'].diff().mean()
            st.metric(
                "Move Confidence",
                f"{confidence:.2%}",
                delta=confidence_trend,
                delta_color="normal"
            )


def main():
    st.set_page_config(
        page_title="YINSH Training Dashboard",
        page_icon="🎮",
        layout="wide"
    )

    # Sidebar for controls
    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox(
        "Training Mode",
        ["tiny", "quick", "dev", "full"],
        format_func=lambda x: x.capitalize()
    )

    # Load data
    model_dir = Path(f"models/training_{mode}")
    if not model_dir.exists():
        st.error(f"No training data found for {mode} mode")
        return

    # Load and convert metrics
    raw_metrics = load_metrics(model_dir)
    if not raw_metrics:
        st.warning("No training data available yet")
        return

    df = create_metrics_df(raw_metrics)

    # Main header with training info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("YINSH Training Dashboard")
    with col2:
        st.metric("Current Mode", mode.upper())
    with col3:
        training_time = (df['timestamp'].max() - df['timestamp'].min()) / 3600 if 'timestamp' in df.columns else 0
        st.metric("Training Time", f"{training_time:.1f}h")

    # Show training status
    show_training_status(df)

    # Key metrics in a row
    if len(df) > 0:
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            latest_win_rate = df['win_rates'].iloc[-1] if 'win_rates' in df.columns else 0
            st.metric("Latest Win Rate", f"{latest_win_rate:.1%}")
        with metrics_cols[1]:
            latest_mobility = df['ring_mobility'].iloc[-1] if 'ring_mobility' in df.columns else 0
            st.metric("Ring Mobility", f"{latest_mobility:.2f}")
        with metrics_cols[2]:
            latest_policy_loss = df['policy_losses'].iloc[-1] if 'policy_losses' in df.columns else 0
            st.metric("Policy Loss", f"{latest_policy_loss:.4f}")
        with metrics_cols[3]:
            latest_value_loss = df['value_losses'].iloc[-1] if 'value_losses' in df.columns else 0
            st.metric("Value Loss", f"{latest_value_loss:.4f}")

    # Training progress and game outcomes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Progress")
        if 'policy_losses' in df.columns and 'value_losses' in df.columns:
            plot_loss_curves(df)
        else:
            st.warning("No loss data available yet")

    with col2:
        st.subheader("Game Outcomes")
        if 'win_rates' in df.columns and 'draw_rates' in df.columns:
            plot_win_rates(df)
        else:
            st.warning("No game outcome data available yet")

    # Temperature metrics section
    st.subheader("Temperature Analysis")
    col1, col2 = st.columns(2)

    with col1:
        plot_entropy_curves(df)
    with col2:
        plot_temperature_metrics(df)

    # Ring mobility and game length
    if 'ring_mobility' in df.columns:
        st.subheader("Ring Mobility")
        plot_ring_mobility(df)

    if 'game_lengths' in df.columns:
        st.subheader("Game Length Distribution")
        plot_game_length_histogram(df)

    # Show raw data if requested
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Raw Metrics Data")
        st.write(df)

    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh", value=True):
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()