# analysis/tournament_viz.py

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from collections import defaultdict


def load_data(base_path: Path):
    with open(base_path / "elo_ratings.json") as f:
        elo_data = json.load(f)
    with open(base_path / "match_history.json") as f:
        match_data = json.load(f)
    with open(base_path / "tournament_history.json") as f:
        tournament_data = json.load(f)
    return elo_data, match_data, tournament_data


def plot_elo_progression(elo_data):
    ratings = elo_data['current_ratings']
    df = pd.DataFrame([
        {'iteration': int(k.split('_')[1]), 'elo': v}
        for k, v in ratings.items()
    ]).sort_values('iteration')

    # Linear trend
    z = np.polyfit(df['iteration'], df['elo'], 1)
    p = np.poly1d(z)

    # Rolling averages
    df['MA5'] = df['elo'].rolling(window=5).mean()
    df['MA10'] = df['elo'].rolling(window=10).mean()
    df['MA20'] = df['elo'].rolling(window=20).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(df['iteration'], df['elo'], marker='o', alpha=0.4, label='ELO')
    plt.plot(df['iteration'], p(df['iteration']), "r--", alpha=0.6,
             label=f'Trend: {z[0]:.2f} ELO/iteration')
    plt.plot(df['iteration'], df['MA5'], 'g-', alpha=0.7, label='5-iter MA')
    plt.plot(df['iteration'], df['MA10'], 'y-', alpha=0.7, label='10-iter MA')
    plt.plot(df['iteration'], df['MA20'], 'm-', alpha=0.7, label='20-iter MA')

    plt.title('ELO Rating Progression')
    plt.xlabel('Iteration')
    plt.ylabel('ELO Rating')
    plt.grid(True)
    plt.legend()

    return plt


def analyze_plateau_performance(tournament_data):
    """Analyze win rates by opponent distance during key periods."""
    # Define periods
    periods = {
        'early': (0, 30),
        'plateau': (30, 50),
        'late': (50, 85)
    }

    # Collect win rates by distance for each period
    period_results = defaultdict(list)

    for tournament_id, data in tournament_data.items():
        current_iter = data['iteration']
        for period_name, (start, end) in periods.items():
            if start <= current_iter < end:
                for match in data['results']:
                    if match['white_model'] == f'iteration_{current_iter}':
                        opponent_iter = int(match['black_model'].split('_')[1])
                        distance = current_iter - opponent_iter
                        win_rate = match['white_wins'] / 10
                        period_results[period_name].append({
                            'distance': distance,
                            'win_rate': win_rate
                        })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'early': 'blue', 'plateau': 'red', 'late': 'green'}

    for period, results in period_results.items():
        df = pd.DataFrame(results)
        df = df.groupby('distance')['win_rate'].mean().reset_index()
        ax.plot(df['distance'], df['win_rate'],
                color=colors[period], label=f'{period} ({periods[period][0]}-{periods[period][1]})',
                marker='o')

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Opponent Distance (iterations)')
    plt.ylabel('Win Rate')
    plt.title('Win Rate by Opponent Distance Across Training Periods')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt

def analyze_color_bias(match_data):
    matches_df = pd.DataFrame(match_data)

    results = []
    for model in set(matches_df['white_model']) | set(matches_df['black_model']):
        white_games = matches_df[matches_df['white_model'] == model]
        black_games = matches_df[matches_df['black_model'] == model]

        # Calculate total games and wins for normalization
        total_games = len(white_games) + len(black_games)
        total_wins = white_games['white_wins'].sum() + black_games['black_wins'].sum()

        if total_games > 0:
            overall_winrate = total_wins / total_games

            # Calculate relative win rates
            white_wr = white_games['white_wins'].sum() / len(white_games) if len(white_games) > 0 else 0
            black_wr = black_games['black_wins'].sum() / len(black_games) if len(black_games) > 0 else 0

            # Normalize by dividing by overall win rate
            white_relative = white_wr / overall_winrate if overall_winrate > 0 else 1
            black_relative = black_wr / overall_winrate if overall_winrate > 0 else 1

            results.append({
                'model': int(model.split('_')[1]),
                'white_relative': white_relative,
                'black_relative': black_relative
            })

    df = pd.DataFrame(results).sort_values('model')

    plt.figure(figsize=(15, 5))
    plt.plot(df['model'], df['white_relative'], label='White Relative Performance', color='blue')
    plt.plot(df['model'], df['black_relative'], label='Black Relative Performance', color='red')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Relative Performance by Color (1.0 = Average)')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Win Rate')
    plt.legend()
    plt.grid(True)
    return plt


def analyze_win_distribution(match_data):
    matches_df = pd.DataFrame(match_data)
    results = []
    for model in set(matches_df['white_model']) | set(matches_df['black_model']):
        white_games = matches_df[matches_df['white_model'] == model]
        black_games = matches_df[matches_df['black_model'] == model]

        total_games = len(white_games) + len(black_games)
        total_wins = white_games['white_wins'].sum() + black_games['black_wins'].sum()
        white_wins = white_games['white_wins'].sum()
        black_wins = black_games['black_wins'].sum()

        if total_wins > 0:
            results.append({
                'model': int(model.split('_')[1]),
                'white_win_pct': white_wins / total_wins,
                'black_win_pct': black_wins / total_wins,
                # need to divide by 10 to get actual win rate
                'overall_winrate': ((total_wins / total_games) * 0.1)  # Convert to proper ratio
            })

    df = pd.DataFrame(results).sort_values('model')

    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()

    # Add debug prints
    print("Overall win rates:", df['overall_winrate'].tolist())
    print("Y-axis limits:", ax2.get_ylim())

    # Reduced alpha for bars
    ax1.bar(df['model'], df['white_win_pct'], label='Wins as White', color='blue', alpha=0.3)
    ax1.bar(df['model'], df['black_win_pct'], bottom=df['white_win_pct'],
            label='Wins as Black', color='red', alpha=0.3)

    # Enhanced line plot
    ax2.plot(df['model'], df['overall_winrate'],
             color='green',  # Changed from green for better visibility
             linewidth=2,
             marker='o',
             markersize=4,
             alpha=1.0,  # Explicit alpha
             zorder=5)   # Ensure line appears above grid

    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distribution of Wins (White vs Black)')
    ax2.set_ylabel('Overall Win Rate')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Win Distribution by Color and Overall Performance')
    plt.grid(True, alpha=0.2)

    return plt

def get_week_run_path():
   # Get the directory containing the script
   script_dir = Path(__file__).parent.absolute()
   # Go up to project root and then down to week_run directory
   week_run_path = script_dir.parent / "checkpoints/combined/week_run"
   return week_run_path


def plot_historical_performance(tournament_data):
    """Track how each model performs against all previous versions."""
    results = []
    for tournament_id, data in tournament_data.items():
        current_iter = data['iteration']
        for match in data['results']:
            if match['white_model'] == f'iteration_{current_iter}':
                opponent_iter = int(match['black_model'].split('_')[1])
                win_rate = match['white_wins'] / 10  # 10 games per match
                results.append({'current': current_iter, 'opponent': opponent_iter, 'win_rate': win_rate})

    df = pd.DataFrame(results)

    # Plot heatmap of win rates
    pivot_df = df.pivot(index='current', columns='opponent', values='win_rate')
    sns.heatmap(pivot_df, cmap='RdYlBu', center=0.5)
    plt.title('Win Rates vs Historical Models')
    return plt


def plot_rolling_strength(tournament_data, window=5):
    """Compare models against their recent predecessors."""
    results = []
    for tournament_id, data in tournament_data.items():
        current_iter = data['iteration']
        recent_wins = 0
        total_games = 0
        for match in data['results']:
            opponent_iter = int(match['black_model'].split('_')[1])
            if current_iter - window <= opponent_iter < current_iter:
                if match['white_model'] == f'iteration_{current_iter}':
                    recent_wins += match['white_wins']
                    total_games += 10
        if total_games > 0:
            results.append({'iteration': current_iter, 'win_rate': recent_wins / total_games})

    df = pd.DataFrame(results)
    plt.plot(df['iteration'], df['win_rate'], marker='o')
    plt.title(f'Win Rate vs Previous {window} Iterations')
    return plt


def analyze_win_rate_by_distance(tournament_data):
    """Analyze win rates based on iteration distance."""
    distance_results = []

    for tournament_id, data in tournament_data.items():
        current_iter = data['iteration']
        for match in data['results']:
            if match['white_model'] == f'iteration_{current_iter}':
                opponent_iter = int(match['black_model'].split('_')[1])
                distance = current_iter - opponent_iter
                win_rate = match['white_wins'] / 10
                distance_results.append({
                    'distance': distance,
                    'win_rate': win_rate,
                    'current_iter': current_iter
                })

    df = pd.DataFrame(distance_results)

    # Group by distance and calculate average win rate
    avg_by_distance = df.groupby('distance')['win_rate'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.scatter(df['distance'], df['win_rate'], alpha=0.2, label='Individual Matches')
    plt.plot(avg_by_distance['distance'], avg_by_distance['win_rate'],
             color='red', linewidth=2, label='Average Win Rate')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration Distance')
    plt.ylabel('Win Rate')
    plt.title('Win Rate by Iteration Distance')
    plt.legend()

    return plt

def plot_game_lengths(match_data):
    matches_df = pd.DataFrame(match_data)
    plt.figure(figsize=(15, 5))
    plt.hist(matches_df['avg_game_length'], bins=30)
    plt.title('Distribution of Game Lengths')
    plt.xlabel('Average Game Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    return plt


def main():
    base_path = get_week_run_path()
    elo_data, match_data, tournament_data = load_data(base_path)

    # Create analysis directory if it doesn't exist
    output_dir = Path(__file__).parent.absolute()
    output_dir.mkdir(exist_ok=True)

    # Save plots with absolute paths
    elo_fig = plot_elo_progression(elo_data)
    elo_fig.savefig(output_dir / 'elo_progression.png')

    color_fig = analyze_color_bias(match_data)
    color_fig.savefig(output_dir / 'color_bias.png')

    length_fig = plot_game_lengths(match_data)
    length_fig.savefig(output_dir / 'game_lengths.png')

    win_dist_fig = analyze_win_distribution(match_data)
    win_dist_fig.savefig(output_dir / 'win_distribution.png')

    distance_fig = analyze_win_rate_by_distance(tournament_data)
    distance_fig.savefig(output_dir / 'win_rate_by_distance.png')

    historical_fig = plot_historical_performance(tournament_data)
    historical_fig.savefig(output_dir / 'historical_performance.png')

    rolling_fig = plot_rolling_strength(tournament_data)
    rolling_fig.savefig(output_dir / 'rolling_strength.png')

    plateau_perf = analyze_plateau_performance(tournament_data)
    plateau_perf.savefig(output_dir / 'plateau_performance.png')

if __name__ == "__main__":
    main()