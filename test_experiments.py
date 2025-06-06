import argparse
import warnings
from typing import List, Dict

import numpy as np
import torch

from experiments.config import get_experiment_config
from experiments.runner import ExperimentRunner
from yinsh_ml.training.trainer import YinshTrainer  # Add this import

warnings.filterwarnings('ignore', message='Torch version.*has not been tested with coremltools')

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


        # Override combined experiments for quick testing
        for config_name in COMBINED_EXPERIMENTS:
            original_config = COMBINED_EXPERIMENTS[config_name]
            COMBINED_EXPERIMENTS[config_name] = CombinedConfig(
                lr=original_config.lr,
                weight_decay=original_config.weight_decay,
                batch_size=16,  # Reduced for quick testing
                num_simulations=25,  # Reduced for quick testing
                c_puct=original_config.c_puct,
                dirichlet_alpha=original_config.dirichlet_alpha,
                value_weight=original_config.value_weight,
                initial_temp=original_config.initial_temp,
                final_temp=original_config.final_temp,
                lr_schedule=original_config.lr_schedule,
                temp_schedule=original_config.temp_schedule,
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
            "aggressive_search",
            "value_head_config"
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


def _print_metrics(result: dict) -> None:
    """Pretty‑print the latest metrics, falling back gracefully if any are missing."""

    def latest(series_name: str, default: str = "–") -> str:
        """Return the last element of a list in `result`, formatted as a string."""
        series = result.get(series_name, [])
        return f"{series[-1]:.4f}" if series else default

    print(f"Policy Loss       : {latest('policy_losses')}")
    print(f"Value  Loss       : {latest('value_losses')}")
    print(f"ELO Change        : {latest('elo_changes', default='–')[:-2]}")  # strip trailing zeros

    # Optional series ----------------------------------------------------------
    if "value_accuracies" in result:
        acc = result["value_accuracies"][-1] if result["value_accuracies"] else None
        print(f"Value Accuracy    : {acc:.2%}" if acc is not None else "Value Accuracy    : –")

    if "move_accuracies" in result and result["move_accuracies"]:
        mv = result["move_accuracies"][-1]
        print(f"Move Top‑1 Acc    : {mv.get('top_1_accuracy', 0):.2%}")
        print(f"Move Top‑3 Acc    : {mv.get('top_3_accuracy', 0):.2%}")

    if "policy_entropy" in result:
        pe = result["policy_entropy"][-1] if result["policy_entropy"] else None
        print(f"Policy Entropy    : {pe:.3f}" if pe is not None else "Policy Entropy    : –")

    if "move_entropies" in result:
        me = result["move_entropies"][-1] if result["move_entropies"] else None
        print(f"Move Entropy      : {me:.3f}" if me is not None else "Move Entropy      : –")

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
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    experiment_types = {
        "learning_rate": ["baseline", "conservative", "aggressive", "high_reg", "warmup", "large_batch"],
        "mcts": ["baseline", "deep_search", "very_deep_search", "exploratory", "balanced"],
        "temperature": ["baseline", "high_exploration", "slow_annealing", "adaptive", "cyclical"],
        "combined": ["aggressive_search","balanced_optimizer"]
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

    # Import necessary configs
    from experiments.config import COMBINED_EXPERIMENTS, CombinedConfig

    if quick_test:
        # Quick test parameters
        quick_params = {
            'num_iterations': 3,
            'games_per_iteration': 10,  # Reduced from 5
            'epochs_per_iteration': 1,
            'batches_per_epoch': 10
        }

        # Override COMBINED_EXPERIMENTS with quick test values
        COMBINED_EXPERIMENTS["balanced_optimizer"] = CombinedConfig(
            lr=0.0005,
            weight_decay=2e-4,
            batch_size=16,  # Reduced batch size
            num_simulations=25,  # Reduced simulations
            initial_temp=1.5,
            temp_schedule="cosine",
            **quick_params
        )

        COMBINED_EXPERIMENTS["aggressive_search"] = CombinedConfig(
            lr=0.001,
            weight_decay=1e-4,
            batch_size=16,  # Reduced batch size
            num_simulations=25,  # Reduced simulations
            c_puct=1.5,
            initial_temp=2.0,
            **quick_params
        )

    results = {}
    failed_configs = []

    print("\nTesting combined configurations:")
    for config_name in ["balanced_optimizer", "aggressive_search"]:
        print(f"\nTesting {config_name}...")
        try:
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


def run_quick_test(experiment_type: str, config_name: str):
    """Run a quick validation test for the experiment configuration."""
    print("Running quick validation tests...")

    # Create a very small config for testing
    if experiment_type == "learning_rate":
        config = get_experiment_config(experiment_type, config_name)
        if config:
            # Override with tiny values for quick testing
            config.num_iterations = 1  # Just one iteration
            config.games_per_iteration = 2  # Minimal games
            config.epochs_per_iteration = 1  # One epoch
            config.batches_per_epoch = 2  # Just two batches
            config.batch_size = 32  # Smaller batch size

            print(f"\nQuick test of {experiment_type}/{config_name}")
            print("Using minimal configuration for rapid testing:")
            print(f"- {config.num_iterations} iteration")
            print(f"- {config.games_per_iteration} games per iteration")
            print(f"- {config.epochs_per_iteration} epoch with {config.batches_per_epoch} batches")
            print(f"- Batch size of {config.batch_size}")
            print("\nStarting test...")

            runner = ExperimentRunner(device='cpu')  # Use CPU for testing
            result = runner.run_experiment(experiment_type, config_name)

            if result:
                print("\nQuick test completed successfully!")
                return True

    return False

# Update check_value_head_health function for quick debugging
def check_value_head_health(trainer: YinshTrainer, game_states: List[np.ndarray],
                            game_outcomes: List[int]) -> Dict:
    """Run a quick health check on value head predictions."""
    with torch.no_grad():
        states = torch.FloatTensor(game_states).to(trainer.device)
        outcomes = torch.FloatTensor(game_outcomes).to(trainer.device)

        _, values = trainer.network.network(states)

        stats = {
            'mean_pred': float(values.mean()),
            'std_pred': float(values.std()),
            'max_pred': float(values.max()),
            'min_pred': float(values.min()),
            'num_saturated': float((torch.abs(values) > 0.99).sum()),
            'correlation': float(torch.corrcoef(torch.stack([values.squeeze(), outcomes]))[0, 1])
        }

        return stats


def check_hardware():
    import torch
    import psutil

    print("\n=== Hardware Configuration ===")

    # CPU Info
    print("\nCPU Information:")
    print(f"CPU Count: {psutil.cpu_count()} cores")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    print(f"Memory: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")


    # GPU Info
    print("\nGPU Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB")
        gpu_allocated = torch.cuda.memory_allocated()
        gpu_max_allocated = torch.cuda.max_memory_allocated()

        if gpu_max_allocated != 0:
            gpu_usage = gpu_allocated / gpu_max_allocated
        else:
            gpu_usage = 0  # or any other default value that makes sense for your application

        print(f"GPU Usage: {gpu_usage:.2%}")

    # PyTorch device that will be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPyTorch will use device: {device}")

    print("\n===========================\n")

    # Test tensor operations
    if torch.cuda.is_available():
        print("Running quick GPU test...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        z = torch.matmul(x, y)
        end.record()
        torch.cuda.synchronize()
        print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")


def main():
    check_hardware()

    parser = argparse.ArgumentParser(description='Test YINSH experiment configurations')
    # REMOVED type argument:
    # parser.add_argument('--type',
    #                     choices=['learning_rate', 'mcts', 'temperature', 'combined'],
    #                     help='Test specific experiment type')
    parser.add_argument('--config', required=True, # Make config required now
                        help='Test specific configuration name (e.g., "smoke", "value_head_config")')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick validation with minimal iterations (Note: May need config adjustments)')
    parser.add_argument('--device',
                        choices=['cuda', 'mps', 'cpu'],
                        default='mps' if torch.backends.mps.is_available() else
                        ('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to run on (default: best available)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging') # Added debug flag propagation
    parser.add_argument('--dashboard', action='store_true',
                        help='Start Streamlit dashboard after tests')
    # REMOVED combined-only as --config specifies the run now
    # parser.add_argument('--combined-only', action='store_true',
    #                     help='Only test combined configurations')


    args = parser.parse_args()
    print(f"Using device: {args.device}") # Use the selected device

    failures = [] # Keep track of failures if running multiple tests

    # Initialize Runner with selected device and debug flag
    runner = ExperimentRunner(device=args.device, debug=args.debug)

    # --- Simplified Logic: Run the specified config ---
    print(f"Testing specific configuration: {args.config}")
    try:
        # MODIFIED: Pass only config name to run_experiment
        result = runner.run_experiment(args.config)
        if result: # Check if result is not empty (indicating success)
            _print_metrics(result) # Assuming _print_metrics is defined elsewhere
            print(f"✓ Config '{args.config}' completed.")
        else:
            failures.append(("", args.config, "ExperimentRunner returned empty results"))
            print(f"✗ Config '{args.config}' failed (returned empty results).")

    except Exception as e:
        failures.append(("", args.config, str(e)))
        print(f"✗ Config '{args.config}' failed with exception: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    # --- Removed old logic branches for --quick, --combined-only, test_all_experiments ---
    # If you need quick testing, you should ideally define specific "smoke" or "quick" configs
    # in your config.py rather than modifying them on the fly here.

    # --- Report Failures ---
    if failures:
        print("\nFailed configurations:")
        for _, config, error in failures:
            print(f"• {config}: {error}")

    # --- Optional Dashboard ---
    if args.dashboard:
        print("\nStarting dashboard...")
        # Make sure streamlit is installed: pip install streamlit
        try:
            import streamlit.web.cli as stcli
            # Assuming visualizer.py is in the same directory or adjust path
            dashboard_path = Path(__file__).parent / "visualizer.py"
            if dashboard_path.exists():
                 # Use sys.argv manipulation for Streamlit run
                 import sys
                 sys.argv = ["streamlit", "run", str(dashboard_path)]
                 stcli.main()
            else:
                 print(f"Dashboard file not found at {dashboard_path}")
        except ImportError:
             print("Streamlit is not installed. Cannot start dashboard. `pip install streamlit`")
        except Exception as e:
             print(f"Failed to start dashboard: {e}")


if __name__ == "__main__":
    main()