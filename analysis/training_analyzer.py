"""Experiment results analysis for YINSH training."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


def analyze_training_data(metrics_dir: Path, checkpoints_dir: Path):
    """Analyze training metrics and tournament results."""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    metrics_path = project_root / metrics_dir
    checkpoints_path = project_root / checkpoints_dir

    print(f"Looking for metrics in: {metrics_path}")
    print(f"Looking for checkpoints in: {checkpoints_path}")

    # Check files
    tournament_file = checkpoints_path / "tournament_history.json"
    elo_file = checkpoints_path / "elo_ratings.json"

    # Read tournament file
    tournament_data = {}
    try:
        with open(tournament_file) as f:
            content = f.read()
            print(f"Tournament file content length: {len(content)} bytes")
            tournament_data = json.loads(content)
    except Exception as e:
        print(f"Error reading tournament file: {e}")

    # Read ELO file
    elo_ratings = {}
    try:
        with open(elo_file) as f:
            content = f.read()
            print(f"ELO file content length: {len(content)} bytes")
            elo_data = json.loads(content)
            if 'current_ratings' not in elo_data:
                print("Warning: 'current_ratings' not found in ELO data")
                print(f"Available keys: {list(elo_data.keys())}")
            elo_ratings = elo_data.get('current_ratings', {})
    except Exception as e:
        print(f"Error reading ELO file: {e}")

    # Process metrics
    data = defaultdict(list)
    win_rates_by_distance = defaultdict(list)

    try:
        for file in sorted(metrics_path.glob("iteration_*.json")):
            iter_num = int(file.stem.split('_')[1])
            with open(file) as f:
                metrics = json.load(f)['metrics']
                print(f"\nMetrics structure for iteration {iter_num}:")
                print(json.dumps(metrics['summary_stats'], indent=2))

                tournament_key = next((k for k in tournament_data if
                                   tournament_data[k]['iteration'] == iter_num), None)

                data['elo'].append(float(elo_ratings.get(f'iteration_{iter_num}', 0)))

                # More flexible value accuracy extraction
                value_head_data = metrics.get('summary_stats', {}).get('value_head', {})
                if isinstance(value_head_data, dict):
                    accuracy = (value_head_data.get('accuracy_by_phase', {})
                              .get('main_game', {}).get('mean', 0))
                else:
                    accuracy = 0.0
                data['value_acc'].append(accuracy)

                # Extract policy loss and gradient norm with defaults
                dynamics = metrics.get('summary_stats', {}).get('learning_dynamics', {})
                data['policy_loss'].append(dynamics.get('policy_loss_trend', 0))
                data['gradient_norm'].append(np.mean(
                    dynamics.get('gradient_norms', [0])
                ))

                # Extract game length
                games = metrics.get('games', [])
                if games:
                    data['game_length'].append(np.mean([g.get('length', 0) for g in games]))
                else:
                    data['game_length'].append(0)

                if tournament_key:
                    for match in tournament_data[tournament_key]['results']:
                        distance = abs(int(match['black_model'].split('_')[1]) - iter_num)
                        win_rate = match['white_wins'] / (match['white_wins'] + match['black_wins'])
                        win_rates_by_distance[distance].append(win_rate)

    except Exception as e:
        print(f"Error processing metrics: {e}")
        import traceback
        traceback.print_exc()

    # Create visualization
    # Only create plots if we have data
    if not any(data.values()):
        print("No data collected to visualize")
        return None, {}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ELO and Value Accuracy
    if data['elo'] and data['value_acc']:
        axes[0, 0].plot(data['elo'], label='ELO', marker='o')
        axes[0, 0].plot(data['value_acc'], label='Value Accuracy', marker='o')
        axes[0, 0].set_title('ELO vs Value Accuracy')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
    else:
        print("No ELO or accuracy data to plot")

    # Policy Loss and Gradient Norm
    if data['policy_loss'] and data['gradient_norm']:
        axes[0, 1].plot(data['policy_loss'], label='Policy Loss', marker='o')
        axes[0, 1].plot(data['gradient_norm'], label='Gradient Norm', marker='o')
        axes[0, 1].set_title('Training Dynamics')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
    else:
        print("No training dynamics data to plot")

    # Game Length
    if data['game_length']:
        axes[1, 0].plot(data['game_length'], marker='o')
        axes[1, 0].set_title('Average Game Length')
        axes[1, 0].grid(True)
    else:
        print("No game length data to plot")

    # Win Rates by Distance
    if win_rates_by_distance:
        distances = sorted(win_rates_by_distance.keys())
        avg_rates = [np.mean(win_rates_by_distance[d]) for d in distances]
        axes[1, 1].plot(distances, avg_rates, marker='o')
        axes[1, 1].set_title('Win Rate by Opponent Distance')
        axes[1, 1].grid(True)
    else:
        print("No win rate data to plot")

    stats = {
        'elo_trend': np.polyfit(range(len(data['elo'])), data['elo'], 1)[0],
        'value_acc_final': np.mean(data['value_acc'][-10:]),
        'policy_loss_final': np.mean(data['policy_loss'][-10:]),
        'gradient_norm_final': np.mean(data['gradient_norm'][-10:])
    }

    return fig, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_path', type=str, required=True,
                        help="Path to metrics directory")
    parser.add_argument('--checkpoints_path', type=str, required=True,
                        help="Path to checkpoints directory")
    args = parser.parse_args()

    fig, stats = analyze_training_data(
        metrics_dir=Path(args.metrics_path),
        checkpoints_dir=Path(args.checkpoints_path)
    )

    # Save the plot
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'training_analysis.png')

    print(f"Training Statistics:\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()