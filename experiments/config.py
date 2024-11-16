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
class BaseExperimentConfig:
    """Base configuration shared across all experiment types."""
    num_iterations: int
    games_per_iteration: int
    epochs_per_iteration: int
    batches_per_epoch: int

    def __init__(self,
                 num_iterations: int = 10,
                 games_per_iteration: int = 75,
                 epochs_per_iteration: int = 3,
                 batches_per_epoch: int = 75):
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.epochs_per_iteration = epochs_per_iteration
        self.batches_per_epoch = batches_per_epoch

@dataclass
class LearningRateConfig(BaseExperimentConfig):
    """Learning rate specific configuration."""
    lr: float
    weight_decay: float
    batch_size: int
    lr_schedule: str = "constant"
    warmup_steps: int = 0

    def __init__(self,
                 lr: float,
                 weight_decay: float,
                 batch_size: int,
                 lr_schedule: str = "constant",
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps

@dataclass
class MCTSConfig(BaseExperimentConfig):
    """MCTS specific configuration."""
    num_simulations: int
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    value_weight: float = 1.0

    def __init__(self,
                 num_simulations: int,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 value_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.value_weight = value_weight

@dataclass
class TemperatureConfig(BaseExperimentConfig):
    """Temperature annealing configuration."""
    initial_temp: float
    final_temp: float
    annealing_steps: int
    temp_schedule: str = "linear"
    mcts_simulations: int = 100

    def __init__(self,
                 initial_temp: float,
                 final_temp: float,
                 annealing_steps: int,
                 temp_schedule: str = "linear",
                 mcts_simulations: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_steps = annealing_steps
        self.temp_schedule = temp_schedule
        self.mcts_simulations = mcts_simulations

@dataclass
class CombinedConfig(BaseExperimentConfig):
    """Configuration for combined experiments."""
    # Learning rate params
    lr: float
    weight_decay: float
    batch_size: int
    lr_schedule: str = "constant"
    warmup_steps: int = 0

    # MCTS params
    num_simulations: int = 100
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    value_weight: float = 1.0

    # Temperature params
    initial_temp: float = 1.0
    final_temp: float = 0.2
    temp_schedule: str = "linear"

    def __init__(self,
                 lr: float,
                 weight_decay: float,
                 batch_size: int,
                 num_simulations: int = 100,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 value_weight: float = 1.0,
                 initial_temp: float = 1.0,
                 final_temp: float = 0.2,
                 lr_schedule: str = "constant",
                 temp_schedule: str = "linear",
                 warmup_steps: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.value_weight = value_weight
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.temp_schedule = temp_schedule

# Experiment configurations
LEARNING_RATE_EXPERIMENTS = {
    # Hypothesis: Current standard settings provide a good reference point
    "baseline": LearningRateConfig(
        lr=0.001,
        weight_decay=1e-4,
        batch_size=256
    ),

    # Hypothesis: Slower learning with cosine annealing might prevent the ELO decline
    # we're seeing by avoiding overfitting early in training
    "conservative": LearningRateConfig(
        lr=0.0001,
        weight_decay=1e-4,
        batch_size=256,
        lr_schedule="cosine"  # Add LR scheduling
    ),

    # Hypothesis: More aggressive initial learning with step schedule might help
    # escape local optima in early training, while steps prevent divergence
    "aggressive": LearningRateConfig(
        lr=0.003,
        weight_decay=1e-4,
        batch_size=256,
        lr_schedule="step"
    ),

    # Hypothesis: Stronger regularization might help align training loss with actual
    # playing strength by preventing memorization of specific patterns
    "high_reg": LearningRateConfig(
        lr=0.001,
        weight_decay=1e-3,
        batch_size=256
    ),

    # Hypothesis: Gradual warmup might help establish better initial representations
    # before full-speed learning begins
    "warmup": LearningRateConfig(
        lr=0.001,
        weight_decay=1e-4,
        batch_size=256,
        warmup_steps=1000,
        lr_schedule="cosine"
    ),

    # Hypothesis: Larger batches with scaled learning rate might provide more
    # stable gradient estimates and better generalization
    "large_batch": LearningRateConfig(
        lr=0.002,  # Increased LR for larger batch
        weight_decay=1e-4,
        batch_size=512
    )
}

MCTS_EXPERIMENTS = {

    "baseline": MCTSConfig(
        num_simulations=100,
        c_puct=1.0
    ),

    # Hypothesis: More simulation depth might improve move quality but risks
    # overfitting to specific positions
    "deep_search": MCTSConfig(
        num_simulations=200,
        c_puct=1.0
    ),

    "very_deep_search": MCTSConfig(
        num_simulations=400,
        c_puct=1.0
    ),

    # Hypothesis: More exploration in MCTS might help discover novel strategies
    # that basic search misses, increasing strategic diversity
    "exploratory": MCTSConfig(
        num_simulations=100,
        c_puct=2.0,  # More exploration
        dirichlet_alpha=0.5  # More noise
    ),

    # Hypothesis: Rebalancing between policy and value networks might help
    # align immediate moves with long-term strategy better
    "balanced": MCTSConfig(
        num_simulations=200,
        c_puct=1.5,
        value_weight=0.7  # Balance between policy and value
    )
}

TEMPERATURE_EXPERIMENTS = {
    "baseline": TemperatureConfig(
        initial_temp=1.0,
        final_temp=0.2,
        annealing_steps=30
    ),

    # Hypothesis: More early exploration followed by stronger exploitation
    # might help discover better strategies before locking in behavior
    "high_exploration": TemperatureConfig(
        initial_temp=2.0,
        final_temp=0.1,
        annealing_steps=50
    ),

    # Hypothesis: Maintaining higher randomness longer might prevent premature
    # convergence to suboptimal strategies
    "slow_annealing": TemperatureConfig(
        initial_temp=1.0,
        final_temp=0.5,
        annealing_steps=40
    ),

    # Hypothesis: Exponential decay might provide better balance between
    # exploration and exploitation than linear decay
    "adaptive": TemperatureConfig(
        initial_temp=1.5,
        final_temp=0.2,
        annealing_steps=40,
        temp_schedule="exponential"  # Add different schedules
    ),

    # Hypothesis: Periodically increasing temperature might help escape local
    # optima throughout training, similar to cyclical learning rates
    "cyclical": TemperatureConfig(  # Temperature cycling
        initial_temp=1.0,
        final_temp=0.3,
        annealing_steps=30,
        temp_schedule="cyclical"
    )
}

COMBINED_EXPERIMENTS = {

    # Hypothesis: A balanced approach combining moderate values of all parameters
    # might provide more robust learning than extremes in any one dimension
    "balanced_optimizer": CombinedConfig(
        lr=0.0005,
        weight_decay=2e-4,
        batch_size=256,
        num_simulations=200,
        initial_temp=1.5,
        temp_schedule="cosine"
    ),

    # Hypothesis: Combining deep search with high exploration might discover
    # sophisticated strategies that simpler approaches miss
    "aggressive_search": CombinedConfig(
        lr=0.001,
        weight_decay=1e-4,
        batch_size=256,
        num_simulations=300,
        c_puct=1.5,
        initial_temp=2.0
    )
}

# Results directory structure
RESULTS_DIR = Path("results")
RESULTS_SUBDIRS = {
    "learning_rate": RESULTS_DIR / "learning_rate",
    "mcts": RESULTS_DIR / "mcts",
    "temperature": RESULTS_DIR / "temperature",
    "combined": RESULTS_DIR / "combined"

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
        "temperature": TEMPERATURE_EXPERIMENTS,
        "combined": COMBINED_EXPERIMENTS
    }

    if experiment_type not in config_maps:
        logger.error(f"Unknown experiment type: {experiment_type}")
        return None

    config_dict = config_maps[experiment_type]

    if not validate_config(config_name, config_dict):
        return None

    return config_dict[config_name]