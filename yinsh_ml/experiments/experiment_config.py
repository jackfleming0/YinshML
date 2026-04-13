"""
Typed experiment configuration schema with validation.

All hyperparameters for training runs are defined here with:
- Type annotations for IDE support
- Default values based on current codebase
- Validation to catch configuration errors early
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Learning rate and optimizer configuration."""
    policy_lr: float = 0.001
    value_lr_factor: float = 5.0  # Value LR = policy_lr * factor
    policy_weight_decay: float = 1e-4
    value_weight_decay: float = 1e-3
    value_momentum: float = 0.9
    l2_reg: float = 0.0

    # Scheduler configuration
    scheduler_type: Literal['step', 'cosine', 'cyclic', 'none'] = 'step'
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.9

    def validate(self):
        """Validate optimizer configuration."""
        if self.policy_lr <= 0:
            raise ValueError(f"policy_lr must be positive, got {self.policy_lr}")
        if self.value_lr_factor <= 0:
            raise ValueError(f"value_lr_factor must be positive, got {self.value_lr_factor}")
        if self.scheduler_gamma <= 0 or self.scheduler_gamma > 1:
            raise ValueError(f"scheduler_gamma must be in (0, 1], got {self.scheduler_gamma}")


@dataclass
class MCTSConfig:
    """MCTS search configuration."""
    early_simulations: int = 100
    late_simulations: int = 100
    simulation_switch_ply: int = 20
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise
    max_depth: int = 500

    # Evaluation mode
    evaluation_mode: Literal['pure_neural', 'pure_heuristic', 'hybrid'] = 'hybrid'
    heuristic_weight: float = 0.7  # Only used in hybrid mode

    # Batched MCTS (Phase 2.1 feature)
    use_batched_mcts: bool = True
    mcts_batch_size: int = 32

    def validate(self):
        """Validate MCTS configuration."""
        if self.early_simulations < 1:
            raise ValueError(f"early_simulations must be >= 1, got {self.early_simulations}")
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be positive, got {self.c_puct}")
        if not 0 <= self.heuristic_weight <= 1:
            raise ValueError(f"heuristic_weight must be in [0, 1], got {self.heuristic_weight}")


@dataclass
class TemperatureConfig:
    """Temperature schedule for move selection."""
    initial: float = 1.0
    final: float = 0.1
    annealing_steps: int = 30
    schedule: Literal['linear', 'exponential', 'cosine'] = 'linear'
    clamp_fraction: float = 0.6  # Fraction of game where temp is clamped to final

    def validate(self):
        """Validate temperature configuration."""
        if self.initial <= 0:
            raise ValueError(f"initial temperature must be positive, got {self.initial}")
        if self.final <= 0:
            raise ValueError(f"final temperature must be positive, got {self.final}")
        if self.annealing_steps < 1:
            raise ValueError(f"annealing_steps must be >= 1, got {self.annealing_steps}")


@dataclass
class ValueHeadConfig:
    """Value head architecture and loss configuration."""
    mode: Literal['classification', 'regression'] = 'classification'
    num_classes: int = 7  # For classification mode
    discrimination_weight: float = 0.5  # Weight for discrimination loss

    def validate(self):
        """Validate value head configuration."""
        if self.mode == 'classification' and self.num_classes < 3:
            raise ValueError(f"num_classes must be >= 3 for classification, got {self.num_classes}")
        if not 0 <= self.discrimination_weight <= 2:
            raise ValueError(f"discrimination_weight should be in [0, 2], got {self.discrimination_weight}")


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    batch_size: int = 256
    epochs_per_iteration: int = 4
    games_per_iteration: int = 100
    max_buffer_size: int = 10000
    batches_per_epoch: str = 'auto'  # 'auto' or integer

    # Phase weights for sampling
    phase_weights: Dict[str, float] = field(default_factory=lambda: {
        'RING_PLACEMENT': 0.5,
        'MAIN_GAME': 2.0,
        'RING_REMOVAL': 0.5
    })

    def validate(self):
        """Validate training configuration."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs_per_iteration < 1:
            raise ValueError(f"epochs_per_iteration must be >= 1, got {self.epochs_per_iteration}")
        if self.max_buffer_size < self.batch_size:
            raise ValueError(f"max_buffer_size ({self.max_buffer_size}) must be >= batch_size ({self.batch_size})")


@dataclass
class TournamentConfig:
    """Tournament evaluation configuration."""
    games_per_match: int = 20
    promotion_threshold: float = 0.55  # Wilson lower bound threshold

    def validate(self):
        """Validate tournament configuration."""
        if self.games_per_match < 2:
            raise ValueError(f"games_per_match must be >= 2, got {self.games_per_match}")
        if not 0.5 <= self.promotion_threshold <= 1.0:
            raise ValueError(f"promotion_threshold must be in [0.5, 1.0], got {self.promotion_threshold}")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    name: str = "unnamed_experiment"
    description: str = ""
    iterations: int = 10

    # Component configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)

    # Early stopping
    early_stop_enabled: bool = True
    early_stop_elo_threshold: float = 1400  # Stop if ELO drops below this
    early_stop_patience: int = 5  # Number of consecutive rejections before stopping

    # Peak detection (stop when training regresses from peak)
    peak_detection_enabled: bool = True
    peak_patience: int = 3  # Consecutive iterations below peak before stopping
    peak_regression_threshold: float = 30.0  # ELO points below peak to consider regression

    # Checkpoint retention (keep top N checkpoints by ELO)
    checkpoint_retention_count: int = 5  # Number of top checkpoints to keep (0 = keep all, -1 = delete rejected)

    # Observability
    verbosity: Literal['debug', 'info', 'warning'] = 'info'
    tensorboard_enabled: bool = True
    save_checkpoints: bool = True

    def validate(self):
        """Validate the entire configuration."""
        errors = []

        if self.iterations < 1:
            errors.append(f"iterations must be >= 1, got {self.iterations}")

        # Validate sub-configs
        for config_name in ['training', 'optimizer', 'mcts', 'temperature', 'value_head', 'tournament']:
            config = getattr(self, config_name)
            try:
                config.validate()
            except ValueError as e:
                errors.append(f"{config_name}: {e}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_mode_settings(self) -> Dict[str, Any]:
        """Convert to mode_settings dict for TrainingSupervisor compatibility."""
        return {
            # Optimizer
            'lr': self.optimizer.policy_lr,
            'value_head_lr_factor': self.optimizer.value_lr_factor,
            'l2_reg': self.optimizer.l2_reg,

            # Training
            'batch_size': self.training.batch_size,
            'max_buffer_size': self.training.max_buffer_size,
            'batches_per_epoch': self.training.batches_per_epoch,

            # MCTS
            'num_simulations': self.mcts.early_simulations,
            'late_simulations': self.mcts.late_simulations,
            'simulation_switch_ply': self.mcts.simulation_switch_ply,
            'c_puct': self.mcts.c_puct,
            'dirichlet_alpha': self.mcts.dirichlet_alpha,
            'max_depth': self.mcts.max_depth,
            'evaluation_mode': self.mcts.evaluation_mode,
            'heuristic_weight': self.mcts.heuristic_weight,
            'use_batched_mcts': self.mcts.use_batched_mcts,
            'mcts_batch_size': self.mcts.mcts_batch_size,

            # Temperature
            'initial_temp': self.temperature.initial,
            'final_temp': self.temperature.final,
            'annealing_steps': self.temperature.annealing_steps,
            'temp_schedule': self.temperature.schedule,
            'temp_clamp_fraction': self.temperature.clamp_fraction,

            # Value head
            'discrimination_weight': self.value_head.discrimination_weight,
            'value_loss_weights': (0.5, 0.5),  # Legacy format

            # Tournament
            'promotion_threshold': self.tournament.promotion_threshold,

            # Checkpoint retention
            'checkpoint_retention_count': self.checkpoint_retention_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Handle nested configs
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'optimizer' in data and isinstance(data['optimizer'], dict):
            data['optimizer'] = OptimizerConfig(**data['optimizer'])
        if 'mcts' in data and isinstance(data['mcts'], dict):
            data['mcts'] = MCTSConfig(**data['mcts'])
        if 'temperature' in data and isinstance(data['temperature'], dict):
            data['temperature'] = TemperatureConfig(**data['temperature'])
        if 'value_head' in data and isinstance(data['value_head'], dict):
            data['value_head'] = ValueHeadConfig(**data['value_head'])
        if 'tournament' in data and isinstance(data['tournament'], dict):
            data['tournament'] = TournamentConfig(**data['tournament'])

        # Handle experiment-level fields
        config_fields = {
            'name', 'description', 'iterations',
            'training', 'optimizer', 'mcts', 'temperature', 'value_head', 'tournament',
            'early_stop_enabled', 'early_stop_elo_threshold', 'early_stop_patience',
            'peak_detection_enabled', 'peak_patience', 'peak_regression_threshold',
            'checkpoint_retention_count',
            'verbosity', 'tensorboard_enabled', 'save_checkpoints'
        }
        filtered_data = {k: v for k, v in data.items() if k in config_fields}

        return cls(**filtered_data)


def load_config(path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Handle nested 'experiment' key
    if 'experiment' in data:
        # Merge experiment metadata with top-level
        exp_data = data.pop('experiment')
        data = {**exp_data, **data}

    config = ExperimentConfig.from_dict(data)
    logger.info(f"Loaded config from {path}: {config.name}")
    return config


def validate_config(config: ExperimentConfig) -> bool:
    """Validate configuration, raising ValueError on failure."""
    config.validate()
    return True


def save_config(config: ExperimentConfig, path: str):
    """Save experiment configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def diff_configs(config1: ExperimentConfig, config2: ExperimentConfig) -> Dict[str, tuple]:
    """
    Compare two configs and return differences.

    Returns:
        Dict mapping parameter path to (value1, value2) tuples
    """
    def flatten_dict(d: Dict, prefix: str = '') -> Dict[str, Any]:
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, key))
            else:
                items[key] = v
        return items

    flat1 = flatten_dict(config1.to_dict())
    flat2 = flatten_dict(config2.to_dict())

    all_keys = set(flat1.keys()) | set(flat2.keys())
    diffs = {}

    for key in all_keys:
        v1 = flat1.get(key)
        v2 = flat2.get(key)
        if v1 != v2:
            diffs[key] = (v1, v2)

    return diffs
