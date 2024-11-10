"""Configuration for YINSH training experiments."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LearningRateConfig:
    lr: float
    weight_decay: float
    batch_size: int
    num_iterations: int = 5
    games_per_iteration: int = 20
    epochs_per_iteration: int = 2
    batches_per_epoch: int = 100  # Added this parameter


@dataclass
class MCTSConfig:
    num_simulations: int
    num_iterations: int = 5
    games_per_iteration: int = 20
    epochs_per_iteration: int = 2
    temperature: float = 1.0


@dataclass
class TemperatureConfig:
    initial_temp: float
    final_temp: float
    annealing_steps: int
    num_iterations: int = 5
    games_per_iteration: int = 20
    epochs_per_iteration: int = 2
    mcts_simulations: int = 100


# Experiment configurations
LEARNING_RATE_EXPERIMENTS = {
    "baseline": LearningRateConfig(
        lr=0.001,            # Default learning rate
        weight_decay=1e-4,   # Default weight decay
        batch_size=256,      # Default batch size
        num_iterations=5,
        games_per_iteration=20,
        epochs_per_iteration=2,
        batches_per_epoch=100
    ),
    "low_lr": LearningRateConfig(
        lr=0.0001,
        weight_decay=1e-4,
        batch_size=256
    ),
    "high_regularization": LearningRateConfig(
        lr=0.001,
        weight_decay=1e-3,
        batch_size=256
    ),
    "large_batch": LearningRateConfig(
        lr=0.001,
        weight_decay=1e-4,
        batch_size=512
    )
}

MCTS_EXPERIMENTS = {
    "baseline": MCTSConfig(
        num_simulations=100,  # Default MCTS simulations
        num_iterations=5,
        games_per_iteration=20,
        epochs_per_iteration=2
    ),
    "deep_search": MCTSConfig(
        num_simulations=200
    ),
    "very_deep_search": MCTSConfig(
        num_simulations=400
    )
}

TEMPERATURE_EXPERIMENTS = {
    "baseline": TemperatureConfig(
        initial_temp=1.0,     # Default temperature settings
        final_temp=0.2,
        annealing_steps=30,
        num_iterations=5,
        games_per_iteration=20,
        epochs_per_iteration=2
    ),
    "high_exploration": TemperatureConfig(
        initial_temp=2.0,
        final_temp=0.1,
        annealing_steps=50
    ),
    "slow_annealing": TemperatureConfig(
        initial_temp=1.0,
        final_temp=0.5,
        annealing_steps=40
    )
}

# Results directory structure
RESULTS_DIR = Path("results")
RESULTS_SUBDIRS = {
    "learning_rate": RESULTS_DIR / "learning_rate",
    "mcts": RESULTS_DIR / "mcts",
    "temperature": RESULTS_DIR / "temperature"
}

# Create results directories
for dir_path in RESULTS_SUBDIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)


def validate_config(config_name: str, config_dict: Dict) -> bool:
    """Validate experiment configuration."""
    try:
        if config_name not in config_dict:
            logger.error(f"Configuration '{config_name}' not found")
            return False

        config = config_dict[config_name]

        # Check required fields based on config type
        if isinstance(config, LearningRateConfig):
            assert config.lr > 0, "Learning rate must be positive"
            assert config.weight_decay >= 0, "Weight decay must be non-negative"
            assert config.batch_size > 0, "Batch size must be positive"

        elif isinstance(config, MCTSConfig):
            assert config.num_simulations > 0, "Number of simulations must be positive"

        elif isinstance(config, TemperatureConfig):
            assert config.initial_temp > config.final_temp, "Initial temperature should be higher than final"
            assert config.annealing_steps > 0, "Annealing steps must be positive"

        return True

    except AssertionError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating configuration: {str(e)}")
        return False


def get_experiment_config(experiment_type: str, config_name: str) -> Optional[object]:
    """Get specific experiment configuration."""
    config_maps = {
        "learning_rate": LEARNING_RATE_EXPERIMENTS,
        "mcts": MCTS_EXPERIMENTS,
        "temperature": TEMPERATURE_EXPERIMENTS
    }

    if experiment_type not in config_maps:
        logger.error(f"Unknown experiment type: {experiment_type}")
        return None

    config_dict = config_maps[experiment_type]

    if not validate_config(config_name, config_dict):
        return None

    return config_dict[config_name]