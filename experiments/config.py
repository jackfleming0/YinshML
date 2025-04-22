# experiments/config.py

"""Configuration for YINSH training experiments."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CombinedConfig:
    """Unified configuration for YINSH experiments."""

    # --------------------------------------------------------------------------
    # Core Training Loop Parameters (Previously in BaseExperimentConfig)
    # --------------------------------------------------------------------------
    num_iterations: int = 10
    games_per_iteration: int = 75
    epochs_per_iteration: int = 3
    batches_per_epoch: int = 75 # Note: Trainer might override this dynamically based on experience buffer size

    # --------------------------------------------------------------------------
    # Optimizer/Learning Rate Parameters (Previously in LearningRateConfig)
    # --------------------------------------------------------------------------
    lr: float = 0.001                # Base learning rate (often for policy head)
    weight_decay: float = 1e-4
    batch_size: int = 256
    lr_schedule: str = "constant"    # Options: "constant", "cosine", "step", "cyclical"
    warmup_steps: int = 0
    value_head_lr_factor: float = 5.0 # Multiplier for value head LR relative to base 'lr'
    l2_reg: float = 0.0               # L2 regularization coefficient (passed to trainer)

    # --------------------------------------------------------------------------
    # MCTS Parameters (Previously in MCTSConfig)
    # --------------------------------------------------------------------------
    num_simulations: int = 100        # Default/Early game simulation budget
    late_simulations: Optional[int] = None # Simulation budget after switch_ply (defaults to num_simulations if None)
    simulation_switch_ply: int = 20   # Ply at which simulation budget might change
    c_puct: float = 1.0               # Exploration constant in UCB
    dirichlet_alpha: float = 0.3      # Noise for root policy exploration
    value_weight: float = 1.0         # Weighting factor for value in UCB selection (MCTS._select_action)
    max_depth: int = 20               # Max search depth in MCTS (optional, might prevent overly deep searches)

    # --------------------------------------------------------------------------
    # Temperature Annealing Parameters (Previously in TemperatureConfig)
    # --------------------------------------------------------------------------
    initial_temp: float = 1.0         # Starting temperature for action selection probabilities
    final_temp: float = 0.2           # Final temperature
    annealing_steps: int = 30         # Number of moves over which to anneal temperature
    temp_schedule: str = "linear"     # Options: "linear", "cosine", "exponential", "cyclical"
    temp_clamp_fraction: float = 0.60 # Fraction of annealing_steps before clamping temp to final_temp
    # --- Optional advanced temperature schedule params ---
    temp_decay_half_life: float = 0.5 # For exponential decay
    temp_start_decay_at: float = 0.5  # Fraction of annealing steps to start exponential decay

    # --------------------------------------------------------------------------
    # Loss Function Parameters (Passed to Trainer)
    # --------------------------------------------------------------------------
    value_loss_weights: Tuple[float, float] = (0.5, 0.5) # Weights for (MSE, CrossEntropy) components of value loss

    # --------------------------------------------------------------------------
    # (Optional) Add any other experiment-specific parameters here
    # --------------------------------------------------------------------------
    # Example: attention_specific_param: bool = False


# ==========================================================================
# Experiment Configurations Dictionary
# ==========================================================================
# Now only contains 'CombinedConfig' instances
COMBINED_EXPERIMENTS: Dict[str, CombinedConfig] = {

    # --- Keep your existing CombinedConfig definitions here ---
    # Example:
    "balanced_optimizer": CombinedConfig(
        lr=0.0005,
        weight_decay=2e-4,
        batch_size=256,
        num_simulations=200,
        initial_temp=1.5,
        temp_schedule="cosine",
        num_iterations=50, # Example: Override base defaults if needed
        games_per_iteration=100,
    ),

    "aggressive_search": CombinedConfig(
        lr=0.001,
        weight_decay=1e-4,
        batch_size=256,
        num_simulations=300,
        c_puct=1.5,
        initial_temp=2.0,
        num_iterations=50,
        games_per_iteration=100,
    ),

    "short_baseline": CombinedConfig(
        num_iterations=10,
        games_per_iteration=25,
        epochs_per_iteration=3,
        batches_per_epoch=25, # Note: Trainer might adjust this based on experience
        lr=0.0005,
        weight_decay=2e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=1000,
        num_simulations=200,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        value_weight=1.0,
        initial_temp=2.0,
        final_temp=0.1,
        temp_schedule="exponential"
    ),

    "value_head_eval": CombinedConfig(
        num_iterations=40,
        games_per_iteration=75,
        epochs_per_iteration=5,
        batches_per_epoch=50,
        lr=0.0005,
        weight_decay=2e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=1000,
        num_simulations=200,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        value_weight=1.0,
        initial_temp=2.0,
        final_temp=0.1,
        temp_schedule="exponential"
    ),

    "week_run": CombinedConfig(
        num_iterations=100,
        games_per_iteration=75,
        epochs_per_iteration=5,
        batches_per_epoch=50,
        lr=0.0002,
        weight_decay=5e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,
        num_simulations=200,
        c_puct=1.2,
        dirichlet_alpha=0.3,
        value_weight=0.8,
        initial_temp=1.5,
        final_temp=0.2,
        temp_schedule="exponential"
    ),

    "value_head_config": CombinedConfig(
        num_iterations=100,
        games_per_iteration=75,
        epochs_per_iteration=7,
        batches_per_epoch=50,
        lr=0.0001,
        weight_decay=5e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,
        num_simulations=250,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        value_weight=1.2,
        initial_temp=1.8,
        final_temp=0.1,
        temp_schedule="exponential"
    ),

    "value_head_config2": CombinedConfig(
        num_iterations=100,
        games_per_iteration=75,
        epochs_per_iteration=7,
        batches_per_epoch=50,
        lr=0.0002,
        weight_decay=1e-3,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=1000,
        num_simulations=300,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        value_weight=1.2,
        initial_temp=2.0,
        final_temp=0.1,
        temp_schedule="exponential"
    ),

    "m3_final_revised": CombinedConfig(
        num_iterations=168,
        games_per_iteration=120,
        epochs_per_iteration=5,
        batches_per_epoch=100,
        lr=0.0001,
        weight_decay=5e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,
        num_simulations=200,
        c_puct=2.0,
        dirichlet_alpha=0.4,
        value_weight=1.1,
        initial_temp=2.5,
        final_temp=0.2,
        temp_schedule="cosine"
    ),

    "m3_final_revised2": CombinedConfig(
        num_iterations=50,
        games_per_iteration=120,
        epochs_per_iteration=8,
        batches_per_epoch=75,
        lr=0.0003,
        weight_decay=5e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,
        num_simulations=300,
        c_puct=2.0,
        dirichlet_alpha=0.4,
        value_weight=0.7,
        initial_temp=3.0,
        final_temp=0.2,
        temp_schedule="cosine"
    ),

    "m3_final_revised_v3": CombinedConfig(
        num_iterations=50,
        games_per_iteration=500,
        epochs_per_iteration=10,
        batches_per_epoch=250,
        lr=0.0005,
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,
        num_simulations=400,
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.0,
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "separate_value_head": CombinedConfig(
        num_iterations=50,
        games_per_iteration=500,
        epochs_per_iteration=10,
        batches_per_epoch=250,
        lr=0.0005,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,
        num_simulations=400,
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,
        value_loss_weights=(0.5, 0.5),
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb15_testing": CombinedConfig(
        num_iterations=50,
        games_per_iteration=100,
        epochs_per_iteration=5,
        batches_per_epoch=150,
        lr=0.0005,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,
        num_simulations=100,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,
        value_loss_weights=(0.5, 0.5),
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb16_testing": CombinedConfig(
        num_iterations=50,
        games_per_iteration=100,
        epochs_per_iteration=5,
        batches_per_epoch=150,
        lr=0.0005,
        value_head_lr_factor=3.0,
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,
        num_simulations=200,
        c_puct=2.0,
        dirichlet_alpha=0.15,
        value_weight=1.5,
        value_loss_weights=(0.5, 0.5),
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb18_testing": CombinedConfig(
        num_iterations=50,
        games_per_iteration=100,
        epochs_per_iteration=5,
        batches_per_epoch=150,
        lr=0.0007,
        value_head_lr_factor=4.0,
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=2000,
        num_simulations=250,
        c_puct=2.0,
        dirichlet_alpha=0.15,
        value_weight=1.5,
        value_loss_weights=(0.5, 0.5),
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb20_testing": CombinedConfig(
        num_iterations=25,
        games_per_iteration=150,
        epochs_per_iteration=4,
        batches_per_epoch=150, # Kept original value, but trainer might adjust
        lr=0.0007,
        value_head_lr_factor=4.0,
        weight_decay=1e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=3000,
        num_simulations=125,
        c_puct=2.0,
        dirichlet_alpha=0.17,
        value_weight=1.9,
        value_loss_weights=(0.6, 0.4),
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb26_testing": CombinedConfig(
        num_iterations=25,
        games_per_iteration=200,
        epochs_per_iteration=4,
        batches_per_epoch=150, # Kept original value
        lr=0.0007,
        value_head_lr_factor=7.0,
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=3000,
        num_simulations=125,
        c_puct=2.0,
        dirichlet_alpha=0.17,
        value_weight=1.9,
        value_loss_weights=(0.7, 0.3),
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "mar2_testing": CombinedConfig(
        num_iterations=25,
        games_per_iteration=200,
        epochs_per_iteration=4,
        batches_per_epoch=150, # Kept original value
        lr=0.0007,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=2500,
        num_simulations=200,
        c_puct=2.0,
        dirichlet_alpha=0.17,
        value_weight=1.9,
        value_loss_weights=(0.65, 0.35),
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "mar7_test": CombinedConfig(
        num_iterations=40,
        games_per_iteration=205,
        epochs_per_iteration=8,
        batches_per_epoch=50,
        lr=0.0007,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cyclical",
        warmup_steps=2500,
        num_simulations=50,
        c_puct=2.0,
        dirichlet_alpha=0.17,
        value_weight=1.9,
        value_loss_weights=(0.65, 0.35),
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "mar_19_test": CombinedConfig(
        num_iterations=20,
        games_per_iteration=205,
        epochs_per_iteration=8,
        batches_per_epoch=50,
        lr=0.0007,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cyclical",
        warmup_steps=2500,
        num_simulations=50,
        c_puct=2.0,
        dirichlet_alpha=0.17,
        value_weight=1.9,
        value_loss_weights=(0.65, 0.35),
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "mar_22_test": CombinedConfig(
        num_iterations=20,
        games_per_iteration=205,
        epochs_per_iteration=8,
        batches_per_epoch=50,
        lr=0.0007,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cyclical",
        warmup_steps=2500,
        num_simulations=200,
        c_puct=2.0,
        dirichlet_alpha=0.17,
        value_weight=1.9,
        value_loss_weights=(0.65, 0.35),
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "improved_value_head_mar26": CombinedConfig(
        num_iterations=20,
        games_per_iteration=300,
        epochs_per_iteration=5,
        batches_per_epoch=50,
        lr=0.0005,
        value_head_lr_factor=8.0,
        weight_decay=5e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=1500,
        num_simulations=150,
        c_puct=3.0,
        dirichlet_alpha=0.2,
        value_weight=0.8,
        value_loss_weights=(0.7, 0.3),
        initial_temp=1.8,
        final_temp=0.1,
        temp_schedule="cosine"
    ),

    "improved_value_head_apr1": CombinedConfig(
        num_iterations=20,
        games_per_iteration=300,
        epochs_per_iteration=5,
        batches_per_epoch=50,
        lr=0.0005,
        value_head_lr_factor=18.0,
        weight_decay=5e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=1500,
        num_simulations=150,
        c_puct=3.0,
        dirichlet_alpha=0.2,
        value_weight=0.8,
        value_loss_weights=(0.7, 0.3),
        initial_temp=1.8,
        final_temp=0.1,
        temp_schedule="cosine"
    ),

    "more_games_apr7": CombinedConfig(
        num_iterations=10,
        games_per_iteration=1500,
        epochs_per_iteration=5,
        batches_per_epoch=50,
        lr=0.0005,
        value_head_lr_factor=18.0,
        weight_decay=5e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=1500,
        num_simulations=50,
        c_puct=3.0,
        dirichlet_alpha=0.2,
        value_weight=0.8,
        value_loss_weights=(0.7, 0.3),
        initial_temp=1.8,
        final_temp=0.1,
        temp_schedule="cosine"
    ),

    "separate_value_head_2": CombinedConfig(
        num_iterations=50,
        games_per_iteration=500,
        epochs_per_iteration=10,
        batches_per_epoch=250,
        lr=0.0005,
        value_head_lr_factor=15.0,
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,
        num_simulations=400,
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,
        value_loss_weights=(0.7, 0.3),
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "041725_balanced": CombinedConfig(
        num_iterations=15,
        games_per_iteration=800,
        num_simulations=120,
        late_simulations=60, # Corrected: default is None, needs explicit value if different
        c_puct=2.5,
        dirichlet_alpha=0.17,
        epochs_per_iteration=8,
        batches_per_epoch=60,
        batch_size=512,
        lr=6e-4,
        value_head_lr_factor=6.0,
        weight_decay=1e-4,
        lr_schedule="cosine",
        warmup_steps=2000,
        value_weight=1.0,
        value_loss_weights=(0.5, 0.5),
        initial_temp=1.6,
        final_temp=0.15,
        temp_schedule="cosine"
    ),

    "separate_value_head_smoke": CombinedConfig(
        num_iterations=5,
        games_per_iteration=2,
        epochs_per_iteration=2,
        batches_per_epoch=2,
        lr=0.0005,
        value_head_lr_factor=5.0,
        weight_decay=1e-4,
        batch_size=128, # Reduced batch size for smoke test
        lr_schedule="cosine",
        warmup_steps=10, # Reduced warmup
        # --- MCTS ---
        num_simulations=2, # <<<<< INTENDED LOW VALUE
        late_simulations=2, # Match early sims for simplicity here
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,
        # --- Value Loss ---
        value_loss_weights=(0.5, 0.5),
        # --- Temperature ---
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        annealing_steps=5, # Short anneal for smoke test
        temp_clamp_fraction=0.5, # Clamp quickly
        temp_decay_half_life=0.35, # Keep other temp params
        temp_start_decay_at=0.25
    ),

    "attention_config": CombinedConfig(
        num_iterations=100,
        games_per_iteration=75,
        epochs_per_iteration=5,
        batches_per_epoch=200,
        lr=0.0005,
        weight_decay=2e-4,
        batch_size=32,
        lr_schedule="cosine",
        warmup_steps=1000,
        num_simulations=100,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        value_weight=1.0,
        initial_temp=1.5,
        final_temp=0.1,
        # attention_specific_param=True # Example if needed
    ),

    "smoke": CombinedConfig(
        num_iterations=3,
        games_per_iteration=2,
        epochs_per_iteration=2,
        batches_per_epoch=5, # Reduced batches further
        lr=0.0002,
        weight_decay=5e-4,
        batch_size=64, # Reduced batch size
        lr_schedule="cosine",
        warmup_steps=10, # Reduced warmup
        # --- MCTS ---
        num_simulations=2, # <<<<< INTENDED LOW VALUE
        late_simulations=2,
        c_puct=1.2,
        dirichlet_alpha=0.3,
        value_weight=0.8,
        # --- Temperature ---
        initial_temp=1.5,
        final_temp=0.2,
        temp_schedule="exponential",
        annealing_steps=5,
        temp_clamp_fraction=0.5,
    )

    # --- Add any other CombinedConfig entries here ---

}

# ==========================================================================
# Results Directory Structure (Simplified)
# ==========================================================================
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure base results dir exists

# Removed RESULTS_SUBDIRS as we only have one type now


# ==========================================================================
# Configuration Loading Function (Simplified)
# ==========================================================================
def get_experiment_config(config_name: str) -> Optional[CombinedConfig]:
    """
    Get specific experiment configuration from the COMBINED_EXPERIMENTS dictionary.

    Args:
        config_name: The name of the configuration to retrieve (e.g., "smoke").

    Returns:
        The corresponding CombinedConfig object if found, otherwise None.
    """
    if config_name not in COMBINED_EXPERIMENTS:
        logger.error(f"Configuration '{config_name}' not found in COMBINED_EXPERIMENTS.")
        return None

    config = COMBINED_EXPERIMENTS[config_name]

    # Basic validation inherent in dataclass structure (type hints)
    # Add any crucial cross-parameter checks here if needed, e.g.:
    try:
        assert config.num_simulations > 0, "num_simulations must be positive"
        assert config.lr > 0, "Learning rate must be positive"
        assert config.batch_size > 0, "Batch size must be positive"
        if config.late_simulations is not None:
             assert config.late_simulations > 0, "late_simulations must be positive if specified"
        # Add more assertions as needed...
    except AssertionError as e:
        logger.error(f"Configuration validation failed for '{config_name}': {e}")
        return None

    logger.info(f"Successfully loaded configuration '{config_name}'")
    return config


if __name__ == "__main__":
    # Example of loading a config
    smoke_config = get_experiment_config("smoke")
    if smoke_config:
        print("\nSmoke Config Loaded Successfully!")
        print(f"Num Simulations: {smoke_config.num_simulations}") # Verify the value
        print(f"LR: {smoke_config.lr}")
        # print(vars(smoke_config)) # Print all fields if needed
    else:
        print("\nFailed to Load Smoke Config.")

    attention_config = get_experiment_config("attention_config")
    if attention_config:
        print("\nAttention Config Loaded Successfully!")
    else:
        print("\nFailed to Load Attention Config.")