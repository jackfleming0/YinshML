import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import argparse


class MetricsAnalyzer:
    def __init__(self, experiment_path: Path):
        # Get the directory where the script is located
        script_dir = Path(__file__).parent
        # Go up one level to project root and then to the metrics path
        self.metrics_dir = script_dir.parent / experiment_path
        print(f"Looking for metrics in: {self.metrics_dir}")
        if not self.metrics_dir.exists():
            print(f"Directory not found: {self.metrics_dir}")
            print(f"Current directory: {Path.cwd()}")
            print(f"Script directory: {script_dir}")
            print(f"Available directories in project root:", list(script_dir.parent.glob("*")))

    def load_iterations(self):
        """Load all iteration files into a list."""
        print(f"Looking for files in: {self.metrics_dir}")
        print(f"Directory exists: {self.metrics_dir.exists()}")
        if self.metrics_dir.exists():
            print("Contents:", list(self.metrics_dir.glob("*")))

        metrics_by_iteration = []
        files = sorted(self.metrics_dir.glob("iteration_*.json"))
        if not files:
            print(f"No iteration_*.json files found in {self.metrics_dir}")
            print("Available files:", list(self.metrics_dir.glob("*")))
            return metrics_by_iteration

        for file in files:
            print(f"\nLoading {file}")
            try:
                with open(file) as f:
                    data = json.load(f)
                    print(f"Keys in file: {data.keys()}")
                    if not data:
                        print(f"Warning: Empty data in {file}")
                        continue
                    metrics_by_iteration.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nLoaded {len(metrics_by_iteration)} iteration files")
        return metrics_by_iteration


    def analyze_value_head(self, metrics_by_iteration):
        """Extract and analyze value head performance."""
        value_stats = []

        for i, metrics in enumerate(metrics_by_iteration):
            try:
                enhanced = metrics['enhanced_metrics']
                phases = enhanced['summary']['phases']
                standard = metrics['metrics']

                # Extract learning rates safely
                lr_dict = standard.get('learning_rates', {})
                policy_lr = float(lr_dict.get('policy', 0))
                value_lr = float(lr_dict.get('value', 0))

                stat = {
                    'iteration': i,
                    'placement_accuracy': phases.get('placement', {}).get('value_accuracy', 0),
                    'main_game_accuracy': phases.get('main_game', {}).get('value_accuracy', 0),
                    'ring_removal_accuracy': phases.get('ring_removal', {}).get('value_accuracy', 0),
                    'placement_confidence': phases.get('placement', {}).get('avg_confidence', 0),
                    'main_game_confidence': phases.get('main_game', {}).get('avg_confidence', 0),
                    'ring_removal_confidence': phases.get('ring_removal', {}).get('avg_confidence', 0),
                    'policy_lr': policy_lr,
                    'value_lr': value_lr,
                    'policy_loss': float(standard.get('policy_loss', 0)),
                    'value_loss': float(standard.get('value_loss', 0))
                }
                value_stats.append(stat)

            except Exception as e:
                self.logger.error(f"Error processing iteration {i}: {e}")

        df = pd.DataFrame(value_stats)
        return df

    def plot_value_analysis(self, value_df):
        """Create comprehensive visualization of value head metrics."""
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # 1. Accuracy by phase
        for col in ['placement_accuracy', 'main_game_accuracy', 'ring_removal_accuracy']:
            axs[0, 0].plot(value_df['iteration'], value_df[col],
                           label=col.split('_')[0], marker='o', alpha=0.7)
        axs[0, 0].set_title('Value Accuracy by Phase')
        axs[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
        axs[0, 0].legend()
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Learning Rates
        if value_df['policy_lr'].max() > 0:  # Only plot if we have valid learning rates
            axs[0, 1].plot(value_df['iteration'], value_df['policy_lr'],
                           label='Policy LR', marker='o')
            axs[0, 1].plot(value_df['iteration'], value_df['value_lr'],
                           label='Value LR', marker='o')
            axs[0, 1].set_title('Learning Rate Schedule')
            if value_df['policy_lr'].min() > 0:  # Check if log scale is appropriate
                axs[0, 1].set_yscale('log')
            axs[0, 1].legend()
            axs[0, 1].grid(True, alpha=0.3)

        # 3. Loss Curves
        axs[1, 0].plot(value_df['iteration'], value_df['policy_loss'],
                       label='Policy Loss', marker='o')
        axs[1, 0].plot(value_df['iteration'], value_df['value_loss'],
                       label='Value Loss', marker='o')
        axs[1, 0].set_title('Training Losses')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Phase-Specific Confidence
        for phase in ['placement', 'main_game', 'ring_removal']:
            conf_col = f'{phase}_confidence'
            acc_col = f'{phase}_accuracy'
            acc = value_df[acc_col].iloc[-1]
            conf = value_df[conf_col].iloc[-1]
            axs[1, 1].scatter(value_df['iteration'], value_df[conf_col],
                              label=f'{phase} ({acc:.3f} acc)', alpha=0.7)
        axs[1, 1].set_title('Confidence by Phase')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        # 5. Learning Rate vs Loss
        if value_df['policy_lr'].max() > 0:
            sc = axs[2, 0].scatter(value_df['policy_lr'], value_df['policy_loss'],
                                   c=value_df['iteration'], cmap='viridis')
            plt.colorbar(sc, ax=axs[2, 0], label='Iteration')
            axs[2, 0].set_title('Learning Rate vs Loss')
            if value_df['policy_lr'].min() > 0:
                axs[2, 0].set_xscale('log')
            axs[2, 0].grid(True, alpha=0.3)

        # 6. Phase Accuracy Distribution
        phase_data = []
        for phase in ['placement', 'main_game', 'ring_removal']:
            phase_data.extend([(phase, acc) for acc in value_df[f'{phase}_accuracy']])

        phase_df = pd.DataFrame(phase_data, columns=['Phase', 'Accuracy'])
        sns.violinplot(data=phase_df, x='Phase', y='Accuracy', ax=axs[2, 1])
        axs[2, 1].set_title('Accuracy Distribution by Phase')
        axs[2, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        axs[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def analyze_training_dynamics(self, metrics_by_iteration):
        """Extract and analyze training process metrics."""
        training_stats = []

        for metrics in metrics_by_iteration:
            standard = metrics['metrics']
            training_stats.append({
                'iteration': metrics['iteration'],
                'policy_loss': standard.get('policy_loss', 0),
                'value_loss': standard.get('value_loss', 0),
                'gradient_norm': standard.get('gradient_norm', 0),
                'learning_rate': standard.get('learning_rate', {}).get('policy', 0)
            })

        return pd.DataFrame(training_stats)

    def print_summary(self, value_df):
        """Print comprehensive training summary."""
        print("\nTraining Summary:")
        print("\nValue Head Performance by Phase:")
        for phase in ['placement', 'main_game', 'ring_removal']:
            acc_col = f'{phase}_accuracy'
            conf_col = f'{phase}_confidence'
            start_acc = value_df[acc_col].iloc[0]
            end_acc = value_df[acc_col].iloc[-1]
            peak_acc = value_df[acc_col].max()
            final_conf = value_df[conf_col].iloc[-1]

            print(f"\n{phase.upper()}:")
            print(f"  Accuracy: {start_acc:.3f} → {end_acc:.3f} [Peak: {peak_acc:.3f}]")
            print(f"  Final Confidence: {final_conf:.3f}")

            # Calculate if overconfident
            if end_acc < 0.55 and final_conf > 0.6:
                print("  ⚠️ Warning: Potentially overconfident predictions")

        print("\nLearning Dynamics:")
        if value_df['policy_lr'].max() > 0:
            print(f"  Learning Rates (Final) - Policy: {value_df['policy_lr'].iloc[-1]:.2e}, "
                  f"Value: {value_df['value_lr'].iloc[-1]:.2e}")
        print(f"  Final Losses - Policy: {value_df['policy_loss'].iloc[-1]:.4f}, "
              f"Value: {value_df['value_loss'].iloc[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_path', type=str, required=True,
                        help="Path to metrics directory")
    args = parser.parse_args()

    analyzer = MetricsAnalyzer(Path(args.metrics_path))
    metrics = analyzer.load_iterations()

    # Analyze value head
    if not metrics:
        print("No metrics data found to analyze")
        return

    value_df = analyzer.analyze_value_head(metrics)
    if value_df is None or value_df.empty:
        print("No value head data to analyze")
        return

    # Create iteration column if it doesn't exist
    if 'iteration' not in value_df.columns:
        value_df['iteration'] = range(len(value_df))

    fig = analyzer.plot_value_analysis(value_df)

    # Print detailed summary
    analyzer.print_summary(value_df)

    # Save visualization
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'value_analysis.png')
    print(f"\nAnalysis plots saved to {output_dir}/value_analysis.png")

if __name__ == "__main__":
    main()