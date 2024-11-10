from experiments.config import get_experiment_config
from experiments.runner import ExperimentRunner
import torch


def test_learning_rate_experiments():
    """Test a quick learning rate experiment."""
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Try the low learning rate configuration
    result = runner.run_experiment(
        experiment_type="learning_rate",
        config_name="low_lr"
    )

    print("\nLearning Rate Experiment Results:")
    print(f"Policy Loss: {result['policy_losses'][-1]:.4f}")
    print(f"Value Loss: {result['value_losses'][-1]:.4f}")
    print(f"ELO Change: {result['elo_changes'][-1]:.1f}")


def test_all_experiments():
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')

    experiment_types = {
        "learning_rate": ["baseline", "low_lr", "high_regularization", "large_batch"],  # Added baseline
        "mcts": ["baseline", "deep_search", "very_deep_search"],                            # Added baseline
        "temperature": ["baseline", "high_exploration"]                 # Added baseline
    }

    for exp_type, configs in experiment_types.items():
        print(f"\nTesting {exp_type} experiments...")
        for config in configs:
            print(f"Running configuration: {config}")
            result = runner.run_experiment(exp_type, config)
            print(f"Completed with ELO change: {result['elo_changes'][-1]:.1f}")


def main():
    print("Testing experiment framework...")
    test_all_experiments()

    # Start dashboard
    print("\nStarting dashboard...")
    import streamlit
    streamlit.run("experiments/visualizer.py")




if __name__ == "__main__":
    main()