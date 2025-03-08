"""Configuration for YINSH training experiments."""

from dataclasses import dataclass
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
    temp_decay_half_life: float = 0.5 # Add this
    temp_start_decay_at: float = 0.5 # Add this
    value_head_lr_factor: float = 5.0  # Add this line
    value_loss_weights: Tuple[float, float] = (0.5, 0.5) # Add this line - equal weights to start
    max_depth: int = 20

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
                 temp_decay_half_life: float = 0.5,  # Add this
                 temp_start_decay_at: float = 0.5,  # Add this
                 value_head_lr_factor: float = 5.0,  # Add this line
                 value_loss_weights: Tuple[float, float] = (0.5, 0.5),  # Add this line
                 max_depth: int = 20,
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
        self.temp_decay_half_life = temp_decay_half_life
        self.temp_start_decay_at = temp_start_decay_at
        self.value_head_lr_factor = value_head_lr_factor  # Add this line
        self.value_loss_weights = value_loss_weights # Add this line
        self.max_depth = max_depth

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
    ),

    "short_baseline": CombinedConfig(
        # This configuration is designed to thoroughly evaluate the new value head by:
        # 1. Using longer training to see if value predictions remain stable
        # 2. Starting with high temperature to encourage exploration
        # 3. Transitioning to low temperature to test value head's exploitation
        # 4. Using deeper MCTS to get better value estimates
        # 5. Employing cosine learning rate schedule for better convergence
        #
        # Training parameters

        num_iterations=10, #  was 10
        games_per_iteration=25, #  was 25
        epochs_per_iteration=3,
        batches_per_epoch=25,

        # Learning rate parameters
        lr=0.0005,  # Conservative base learning rate
        weight_decay=2e-4,  # Moderate regularization
        batch_size=256,  # Standard batch size
        lr_schedule="cosine",  # Smooth learning rate decay
        warmup_steps=1000,  # Gradual warmup for stability

        # MCTS parameters
        num_simulations=200,  # Deep enough to test value predictions (200)
        c_puct=1.5,  # Slightly higher exploration
        dirichlet_alpha=0.3,  # Standard noise
        value_weight=1.0,  # Full value weighting

        # Temperature parameters - crucial for testing value head
        initial_temp=2.0,  # Start with high exploration
        final_temp=0.1,  # End with strong exploitation
        temp_schedule="exponential"  # Faster transition once value head stabilizes

    ),

    "value_head_eval": CombinedConfig(
        # Comments explaining the test:
        # This configuration is designed to thoroughly evaluate the new value head by:
        # 1. Using longer training to see if value predictions remain stable
        # 2. Starting with high temperature to encourage exploration
        # 3. Transitioning to low temperature to test value head's exploitation
        # 4. Using deeper MCTS to get better value estimates
        # 5. Employing cosine learning rate schedule for better convergence
        #
        # Training parameters

        num_iterations=40,  # Longer run to see learning dynamics
        games_per_iteration=75,  # Decent sample size
        epochs_per_iteration=5,  # More training per batch
        batches_per_epoch=50,  # Substantial training

        # Learning rate parameters
        lr=0.0005,  # Conservative base learning rate
        weight_decay=2e-4,  # Moderate regularization
        batch_size=256,  # Standard batch size
        lr_schedule="cosine",  # Smooth learning rate decay
        warmup_steps=1000,  # Gradual warmup for stability

        # MCTS parameters
        num_simulations=200,  # Deep enough to test value predictions
        c_puct=1.5,  # Slightly higher exploration
        dirichlet_alpha=0.3,  # Standard noise
        value_weight=1.0,  # Full value weighting

        # Temperature parameters - crucial for testing value head
        initial_temp=2.0,  # Start with high exploration
        final_temp=0.1,  # End with strong exploitation
        temp_schedule="exponential"  # Faster transition once value head stabilizes

    ),

    # gonna run this for a week while i'm out of town and i'm hoping I see some dang improvements!
    "week_run": CombinedConfig(
        # Training parameters
        num_iterations=100,  # About 1 week with current timing
        games_per_iteration=75,  # 50 was about 55 minutes per iteration.
        epochs_per_iteration=5,  # Increased from 3 for better learning
        batches_per_epoch=50,

        # Learning rates - more conservative
        lr=0.0002,  # Reduced from 0.0005
        weight_decay=5e-4,  # Increased from 2e-4
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,  # Longer warmup

        # MCTS parameters
        num_simulations=200,  # Slight increase from 100
        c_puct=1.2,  # Slightly more exploratory
        dirichlet_alpha=0.3,
        value_weight=0.8,  # Reduced from 1.0 to rely less on struggling value head

        # Temperature parameters
        initial_temp=1.5,  # Higher initial exploration
        final_temp=0.2,
        temp_schedule="exponential"  # Smoother transition
    ),

    "value_head_config": CombinedConfig(
        # Core training settings
        num_iterations=100,
        games_per_iteration=75,
        epochs_per_iteration=7,  # Increased from 5
        batches_per_epoch=50,

        # Value head focus
        lr=0.0001,  # Lower learning rate
        weight_decay=5e-4,  # Keep current regularization
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,

        # MCTS parameters
        num_simulations=250,  # Increased from 200
        c_puct=1.0,  # More conservative
        dirichlet_alpha=0.3,
        value_weight=1.2,  # Increased value influence

        # Temperature parameters
        initial_temp=1.8,  # Higher exploration
        final_temp=0.1,  # Stronger exploitation
        temp_schedule="exponential"
    ),

    "value_head_config2": CombinedConfig(

        # This configuration (value_head_config2) balances three key insights:
        # First, the moderate increase in learning rate (0.0002) paired with cosine scheduling should allow
        # more effective learning while maintaining stability.
        # Second, the combination of higher initial temperature (2.0) and moderate MCTS depth (300 simulations)
        # creates more diverse training positions - particularly important in the complex placement and main game
        # phases where the move space is large.
        # Third, the slight increase in value weight (1.2) and regularization (1e-3) aims to improve value head
        # learning without overwhelming the successful policy learning we've observed.
        # This balanced approach should help the value head learn better position evaluation while
        # preserving the strong move accuracy we've already achieved.

        # Training parameters
        num_iterations=100,
        games_per_iteration=75,
        epochs_per_iteration=7,
        batches_per_epoch=50,

        # Learning rates
        lr=0.0002,  # Modest increase for learning
        weight_decay=1e-3,  # Increased regularization
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=1000,

        # MCTS parameters
        num_simulations=300,  # Moderate simulation depth
        c_puct=1.5,  # Balanced exploration
        dirichlet_alpha=0.3,
        value_weight=1.2,  # Slight emphasis on value prediction

        # Temperature parameters - accounting for move space complexity
        initial_temp=2.0,  # Higher early exploration
        final_temp=0.1,  # Strong final exploitation
        temp_schedule="exponential"  # Smoother transition
    ),

    "m3_final_revised": CombinedConfig(

 # one last run on the good computer to see how this bad boy does

        # Training parameters
        num_iterations=168,
        games_per_iteration=120,
        epochs_per_iteration=5,
        batches_per_epoch=100,

        # Learning rates
        lr=0.0001,  # Modest increase for learning
        weight_decay=5e-4,  # Increased regularization
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,

        # MCTS parameters
        num_simulations=200,  # Moderate simulation depth
        c_puct=2.0,  # Balanced exploration
        dirichlet_alpha=0.4,
        value_weight=1.1,  # Slight emphasis on value prediction

        # Temperature parameters - accounting for move space complexity
        initial_temp=2.5,  # Higher early exploration
        final_temp=0.2,  # Strong final exploitation
        temp_schedule="cosine"  # Smoother transition
    ),

    "m3_final_revised2": CombinedConfig(

        # second last run on the good computer to see how this bad boy does

        # Training parameters
        num_iterations=50,
        games_per_iteration=120,
        epochs_per_iteration=8,
        batches_per_epoch=75,

        # Learning rates
        lr=0.0003,  # Modest increase for learning
        weight_decay=5e-4,  # Increased regularization
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,

        # MCTS parameters
        num_simulations=300,  # Moderate simulation depth
        c_puct=2.0,  # Balanced exploration
        dirichlet_alpha=0.4,
        value_weight=0.7,  # Slight emphasis on value prediction

        # Temperature parameters - accounting for move space complexity
        initial_temp=3.0,  # Higher early exploration
        final_temp=0.2,  # Strong final exploitation
        temp_schedule="cosine"  # Smoother transition
    ),

    "m3_final_revised_v3": CombinedConfig(

        # second last run on the good computer to see how this bad boy does

        # Training parameters
        num_iterations=50,
        games_per_iteration=500,
        epochs_per_iteration=10,
        batches_per_epoch=250,

        # Learning rates
        lr=0.0005,  # Slightly increased learning rate
        weight_decay=1e-4,  # Reduced regularization to improve learning
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,

        # MCTS parameters
        num_simulations=400,  # Increased simulation depth
        c_puct=4.0,  # Increased exploration
        dirichlet_alpha=0.3,
        value_weight=1.0,  # Balanced value prediction

        # Temperature parameters - accounting for move space complexity
        initial_temp=1.0,  # Lower initial exploration, more focused on learned policy
        final_temp=0.1,  # Strong final exploitation, allowing model to play best moves
        temp_schedule="cosine",  # Smooth transition
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "separate_value_head": CombinedConfig(
        # Training parameters (no changes)
        num_iterations=50,
        games_per_iteration=500,
        epochs_per_iteration=10,
        batches_per_epoch=250,

        # Learning rates
        lr=0.0005,  # Base learning rate (for policy head)
        value_head_lr_factor=5.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,

        # MCTS parameters
        num_simulations=400,
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,  # Increased value weight
        value_loss_weights=(0.5, 0.5),  # Add this line

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb15_testing": CombinedConfig(
        # Training parameters (no changes)
        num_iterations=50,
        games_per_iteration=100,
        epochs_per_iteration=5,
        batches_per_epoch=150,

        # Learning rates
        lr=0.0005,  # Base learning rate (for policy head)
        value_head_lr_factor=5.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,

        # MCTS parameters
        num_simulations=100,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,  # Increased value weight
        value_loss_weights=(0.5, 0.5),  # Add this line

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb16_testing": CombinedConfig(
        # in runner: ring_weight = 1.0 + iteration * 0.2  # or some schedule. this makes ring placement more important as iterations go up.
        # Training parameters (no changes)
        num_iterations=50,
        games_per_iteration=100,
        epochs_per_iteration=5,
        batches_per_epoch=150,

        # Learning rates
        lr=0.0005,  # Base learning rate (for policy head)
        value_head_lr_factor=3.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,

        # MCTS parameters
        num_simulations=200,
        c_puct=2.0,
        dirichlet_alpha=0.15,
        value_weight=1.5,  # Increased value weight
        value_loss_weights=(0.5, 0.5),  # Add this line

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb18_testing": CombinedConfig(
        # ring_weight = 1.0 + iteration * 0.025  # or some schedule. this makes ring placement more important as iterations go up.
        # Training parameters (no changes)
        num_iterations=50,
        games_per_iteration=100,
        epochs_per_iteration=5,
        batches_per_epoch=150,

        # Learning rates
        lr=0.0007,  # Base learning rate (for policy head)
        value_head_lr_factor=4.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=2000,

        # MCTS parameters
        num_simulations=250,
        c_puct=2.0,
        dirichlet_alpha=0.15,
        value_weight=1.5,  # Increased value weight
        value_loss_weights=(0.5, 0.5),  # Add this line

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb20_testing": CombinedConfig(
        # ring_weight = 1.0 + iteration * 0.0125  # or some schedule. this makes ring placement more important as iterations go up.
        # I changed the ring_placement_weight in trainer.py to be 0.25 instead of 1.0. I think this'll focus more on main game in early phases.
        # Training parameters
        num_iterations= 25, #dropped iterations
        games_per_iteration=150, #but bumped up games
#        games_per_iteration=2, #but bumped up games
        epochs_per_iteration=4, #fewer epochs
#        epochs_per_iteration=1,  # fewer epochs

#        batches_per_epoch=150,
#        batches_per_epoch=3,

        # Learning rates
        lr=0.0007,  # Base learning rate (for policy head)
        value_head_lr_factor=4.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=3000, #maybe in the middle?

        # MCTS parameters
        num_simulations=125,
#        num_simulations=20,
        c_puct=2.0,
        dirichlet_alpha=0.17, #slight bump to flatten move distribution, hopefully
        value_weight=1.9,  # Increased value weight more
        value_loss_weights=(0.6, 0.4),  # also put this a bit more towards MSE

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "feb26_testing": CombinedConfig(
        # ring_weight = 1.0 + iteration * 0.0125  # or some schedule. this makes ring placement more important as iterations go up.
        # I changed the ring_placement_weight in trainer.py to be 0.25 instead of 1.0. I think this'll focus more on main game in early phases.
        # Training parameters
        num_iterations= 25, #dropped iterations
        games_per_iteration=200, #but bumped up games
#        games_per_iteration=2, #but bumped up games
        epochs_per_iteration=4, #fewer epochs
#        epochs_per_iteration=1,  # fewer epochs

#        batches_per_epoch=150,
#        batches_per_epoch=3,

        # Learning rates
        lr=0.0007,  # Base learning rate (for policy head)
        value_head_lr_factor=7.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=3000, #maybe in the middle?

        # MCTS parameters
        num_simulations=125,
#        num_simulations=20,
        c_puct=2.0,
        dirichlet_alpha=0.17, #slight bump to flatten move distribution, hopefully
        value_weight=1.9,  # Increased value weight more
        value_loss_weights=(0.7, 0.3),  # also put this a bit more towards MSE

        # Temperature parameters
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "mar2_testing": CombinedConfig(
        # ring_weight = 1.0 + iteration * 0.0125  # or some schedule. this makes ring placement more important as iterations go up.
        # I changed the ring_placement_weight in trainer.py to be 0.25 instead of 1.0. I think this'll focus more on main game in early phases.
        # Training parameters
        num_iterations= 25, #dropped iterations
        games_per_iteration=200, #but bumped up games
#        games_per_iteration=2, #but bumped up games
        epochs_per_iteration=4, #fewer epochs
#        epochs_per_iteration=1,  # fewer epochs

#        batches_per_epoch=150,
#        batches_per_epoch=3,

        # Learning rates
        lr=0.0007,  # Base learning rate (for policy head)
        value_head_lr_factor=5.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cosine",
        warmup_steps=2500, #maybe in the middle?

        # MCTS parameters
        num_simulations=200,
#        num_simulations=20,
        c_puct=2.0,
        dirichlet_alpha=0.17, #slight bump to flatten move distribution, hopefully
        value_weight=1.9,  # Increased value weight more
        value_loss_weights=(0.65, 0.35),  # also put this a bit more towards MSE

        # Temperature parameters
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "mar7_test": CombinedConfig(
        # phase weights are now defined in trainer.py via phase_weight dict
        # Training parameters
        num_iterations= 40, #dropped iterations
        games_per_iteration=200, #but bumped up games
#        games_per_iteration=2, #but bumped up games
        epochs_per_iteration=8, #fewer epochs
#        epochs_per_iteration=1,  # fewer epochs

#        batches_per_epoch=150,
#        batches_per_epoch=3,

        # Learning rates
        lr=0.0007,  # Base learning rate (for policy head)
        value_head_lr_factor=5.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=384,
        lr_schedule="cyclical",
        warmup_steps=2500, #maybe in the middle?

        # MCTS parameters
        num_simulations=50,
#        num_simulations=20,
        c_puct=2.0,
        dirichlet_alpha=0.17, #slight bump to flatten move distribution, hopefully
        value_weight=1.9,  # Increased value weight more
        value_loss_weights=(0.65, 0.35),  # also put this a bit more towards MSE

        # Temperature parameters
        initial_temp=1.5,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

"separate_value_head_2": CombinedConfig(
        # Training parameters (no changes)
        num_iterations=50,
        games_per_iteration=500,
        epochs_per_iteration=10,
        batches_per_epoch=250,

        # Learning rates
        lr=0.0005,  # Base learning rate (for policy head)
        value_head_lr_factor=15.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=512,
        lr_schedule="cosine",
        warmup_steps=4000,

        # MCTS parameters
        num_simulations=400,
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,  # Increased value weight
        value_loss_weights=(0.7, 0.3),  # Add this line

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "separate_value_head_smoke": CombinedConfig(
        # Training parameters (no changes)
        num_iterations=3,
        games_per_iteration=2,
        epochs_per_iteration=2,
        batches_per_epoch=2,

        # Learning rates
        lr=0.0005,  # Base learning rate (for policy head)
        value_head_lr_factor=5.0,  # Value head learning rate will be lr * value_head_lr_factor
        weight_decay=1e-4,
        batch_size=128,
        lr_schedule="cosine",
        warmup_steps=4000,

        # MCTS parameters
        num_simulations=15,
        c_puct=4.0,
        dirichlet_alpha=0.3,
        value_weight=1.5,  # Increased value weight
        value_loss_weights=(0.5, 0.5),  # Add this line

        # Temperature parameters
        initial_temp=1.0,
        final_temp=0.1,
        temp_schedule="cosine",
        temp_decay_half_life=0.35,
        temp_start_decay_at=0.25
    ),

    "attention_config": CombinedConfig(

        # The attention_config builds on our learnings from value_head_config2 and addresses its limitations:
        # While value_head_config2 showed strong ring removal performance (64.9%) but struggled with placement
        # (48.7%) and main game phases (51.4%), this configuration adds spatial attention to help capture
        # complex board relationships.

        # Key architectural changes are paired with specific parameter choices:
        # 1. The attention mechanisms need a lower learning rate (0.0005) and higher regularization (2e-4) for
        #    stable training of the new spatial attention parameters.
        # 2. We reduce batch size to 32 (from 256) to allow more frequent updates during the initial learning
        #    of attention patterns.
        # 3. We maintain the higher initial temperature (1.5) but reduce simulations (100 from 300) as the
        #    attention mechanism should provide better immediate position evaluation.

        # The hypothesis is that attention will help the model understand spatial relationships between pieces,
        # particularly in the placement and main game phases where value_head_config2 struggled to learn
        # beyond random performance.

        # Training parameters
        num_iterations=100,
        games_per_iteration=75,     # Keep same as value_head_config2
        epochs_per_iteration=5,     # Reduced from 7 due to smaller batches
        batches_per_epoch=200,      # Increased to compensate for smaller batch size

        # Learning parameters
        lr=0.0005,                  # Higher than value_head but with smaller batches
        weight_decay=2e-4,          # Moderate regularization for attention
        batch_size=32,              # Smaller batches for attention training
        lr_schedule="cosine",
        warmup_steps=1000,          # Keep warmup for attention training

        # MCTS parameters
        num_simulations=100,        # Reduced from 300, relying more on attention
        c_puct=1.5,                 # Keep balanced exploration
        dirichlet_alpha=0.3,
        value_weight=1.0,           # Neutral weight to start

        # Temperature parameters
        initial_temp=1.5,           # High but not as extreme as value_head_config2
        final_temp=0.1,             # Keep strong final exploitation
    ),

    "smoke": CombinedConfig(
        # Training parameters
        num_iterations=3,  # About 1 week with current timing
        games_per_iteration=2,  # Reduced from 75 for speed
        epochs_per_iteration=2,  # Increased from 3 for better learning
        batches_per_epoch=50,

        # Learning rates - more conservative
        lr=0.0002,  # Reduced from 0.0005
        weight_decay=5e-4,  # Increased from 2e-4
        batch_size=256,
        lr_schedule="cosine",
        warmup_steps=2000,  # Longer warmup

        # MCTS parameters
        num_simulations=2,  # Slight increase from 100
        c_puct=1.2,  # Slightly more exploratory
        dirichlet_alpha=0.3,
        value_weight=0.8,  # Reduced from 1.0 to rely less on struggling value head

        # Temperature parameters
        initial_temp=1.5,  # Higher initial exploration
        final_temp=0.2,
        temp_schedule="exponential"  # Smoother transition
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

if __name__ == "__main__":
    config = get_experiment_config("combined", "attention_config")
    if config:
        print("Attention Config Loaded Successfully!")
    else:
        print("Failed to Load Attention Config.")