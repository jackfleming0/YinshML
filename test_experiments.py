from experiments.config import get_experiment_config, BaseExperimentConfig
from experiments.runner import ExperimentRunner
import torch
import argparse


def test_experiment_configs(quick_test: bool = True):
    """Test experiment configurations systematically."""
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Override ALL configs for quick tests
    if quick_test:
        from experiments.config import (
            LEARNING_RATE_EXPERIMENTS,
            MCTS_EXPERIMENTS,
            TEMPERATURE_EXPERIMENTS,
            LearningRateConfig,
            MCTSConfig,
            TemperatureConfig
        )

        # Set quick test parameters
        quick_params = {
            'num_iterations': 2,
            'games_per_iteration': 5,
            'epochs_per_iteration': 1,
            'batches_per_epoch': 10
        }

        # Override learning rate experiments
        for config_name in LEARNING_RATE_EXPERIMENTS:
            original_config = LEARNING_RATE_EXPERIMENTS[config_name]
            LEARNING_RATE_EXPERIMENTS[config_name] = LearningRateConfig(
                lr=original_config.lr,
                weight_decay=original_config.weight_decay,
                batch_size=32,  # Reduced batch size
                lr_schedule=original_config.lr_schedule,
                warmup_steps=original_config.warmup_steps,
                **quick_params
            )

        # Override MCTS experiments
        for config_name in MCTS_EXPERIMENTS:
            original_config = MCTS_EXPERIMENTS[config_name]
            MCTS_EXPERIMENTS[config_name] = MCTSConfig(
                num_simulations=50,  # Reduced from typical values like 100-400
                c_puct=original_config.c_puct,
                dirichlet_alpha=original_config.dirichlet_alpha,
                value_weight=original_config.value_weight,
                **quick_params
            )

        # Override temperature experiments
        for config_name in TEMPERATURE_EXPERIMENTS:
            original_config = TEMPERATURE_EXPERIMENTS[config_name]
            TEMPERATURE_EXPERIMENTS[config_name] = TemperatureConfig(
                initial_temp=original_config.initial_temp,
                final_temp=original_config.final_temp,
                annealing_steps=10,  # Reduced from typical values like 30-50
                temp_schedule=original_config.temp_schedule,
                mcts_simulations=50,  # Reduced from 100
                **quick_params
            )

    # Define all experiment configurations to test
    experiment_types = {
        "learning_rate": [
            "baseline",
            "conservative",
            "aggressive",
            "high_reg",
            "warmup",
            "large_batch"
        ],
        "mcts": [
            "baseline",
            "deep_search",
            "exploratory",
            "balanced"
        ],
        "temperature": [
            "baseline",
            "high_exploration",
            "slow_annealing",
            "adaptive",
            "cyclical"
        ],
        "combined": [
            "balanced_optimizer",
            "aggressive_search"
        ]
    }

    results = {}
    failed_configs = []

    for exp_type, configs in experiment_types.items():
        print(f"\nTesting {exp_type} experiments:")
        results[exp_type] = {}

        for config in configs:
            print(f"Running configuration: {config}")
            try:
                result = runner.run_experiment(exp_type, config)
                _print_metrics(result)
                results[exp_type][config] = result
            except Exception as e:
                failed_configs.append((exp_type, config, str(e)))
                print(f"❌ Failed: {str(e)}")
                continue

    return results, failed_configs


def _print_metrics(result: dict):
    """Print experiment metrics in a consistent format."""
    print(f"Policy Loss: {result['policy_losses'][-1]:.4f}")
    print(f"Value Loss: {result['value_losses'][-1]:.4f}")
    print(f"ELO Change: {result['elo_changes'][-1]:.1f}")
    if 'move_entropies' in result:
        print(f"Move Entropy: {result['move_entropies'][-1]:.3f}")
    print("-" * 40)


def test_learning_rate_experiments():
    """Test a quick learning rate experiment."""
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
    result = runner.run_experiment("learning_rate", "low_lr")

    print("\nLearning Rate Experiment Results:")
    _print_metrics(result)


def test_temperature_experiment():
    """Test a quick temperature experiment with baseline config."""
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
    result = runner.run_experiment("temperature", "baseline")

    print("\nTemperature Experiment Results:")
    _print_metrics(result)


def test_all_experiments():
    """Run all experiments (full version)."""
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')

    experiment_types = {
        "learning_rate": ["baseline", "low_lr", "high_regularization", "large_batch"],
        "mcts": ["baseline", "deep_search", "very_deep_search"],
        "temperature": ["baseline", "high_exploration"]
    }

    for exp_type, configs in experiment_types.items():
        print(f"\nTesting {exp_type} experiments...")
        for config in configs:
            print(f"Running configuration: {config}")
            result = runner.run_experiment(exp_type, config)
            _print_metrics(result)


def test_combined_configs(quick_test: bool = True):
    """Test combined experiment configurations."""
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Override base config for quick tests
    if quick_test:
        BaseExperimentConfig.num_iterations = 2
        BaseExperimentConfig.games_per_iteration = 5
        BaseExperimentConfig.epochs_per_iteration = 1

    combined_configs = {
        "balanced_optimizer": {
            "lr": 0.0005,
            "weight_decay": 2e-4,
            "num_simulations": 200,
            "initial_temp": 1.5,
            "temp_schedule": "cosine"
        },
        "aggressive_search": {
            "lr": 0.001,
            "num_simulations": 300,
            "c_puct": 1.5,
            "initial_temp": 2.0
        }
    }

    results = {}
    failed_configs = []

    print("\nTesting combined configurations:")
    for config_name, settings in combined_configs.items():
        print(f"\nTesting {config_name}...")
        try:
            # Create combined experiment configuration
            result = runner.run_experiment(
                experiment_type="combined",
                config_name=config_name
            )
            _print_metrics(result)
            results[config_name] = result
            print(f"✓ {config_name} completed successfully")

        except Exception as e:
            failed_configs.append(("combined", config_name, str(e)))
            print(f"✗ {config_name} failed: {str(e)}")
            continue

    return results, failed_configs

def run_quick_test(experiment_type: str = None, config_name: str = None):
    """Run a quick test of specific or all configurations."""
    if experiment_type and config_name:
        runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nQuick test of {experiment_type}/{config_name}")
        result = runner.run_experiment(experiment_type, config_name)
        _print_metrics(result)
    else:
        results, failures = test_experiment_configs(quick_test=True)
        if failures:
            print("\nFailed configurations:")
            for exp_type, config, error in failures:
                print(f"• {exp_type}/{config}: {error}")


def main():
    parser = argparse.ArgumentParser(description='Test YINSH experiment configurations')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick validation with minimal iterations')
    parser.add_argument('--type',
                        choices=['learning_rate', 'mcts', 'temperature', 'combined'],
                        help='Test specific experiment type')
    parser.add_argument('--config', help='Test specific configuration')
    parser.add_argument('--dashboard', action='store_true',
                        help='Start Streamlit dashboard after tests')
    parser.add_argument('--combined-only', action='store_true',
                        help='Only test combined configurations')

    args = parser.parse_args()

    if args.combined_only:
        print("Running combined configuration tests...")
        results, failures = test_combined_configs(quick_test=args.quick)
    elif args.quick:
        print("Running quick validation tests...")
        run_quick_test(args.type, args.config)
    elif args.type and args.config:
        print(f"Testing specific configuration: {args.type}/{args.config}")
        runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
        result = runner.run_experiment(args.type, args.config)
        _print_metrics(result)
    else:
        print("Running full test suite...")
        test_all_experiments()

    if failures:
        print("\nFailed configurations:")
        for exp_type, config, error in failures:
            print(f"• {exp_type}/{config}: {error}")

    if args.dashboard:
        print("\nStarting dashboard...")
        import streamlit
        streamlit.run("experiments/visualizer.py")


if __name__ == "__main__":
    main()