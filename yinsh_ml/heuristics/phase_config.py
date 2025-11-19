"""Configuration system for phase detection parameters.

This module provides a configurable system for phase boundary thresholds and
transition parameters, enabling easy tuning and experimentation.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class PhaseConfig:
    """Configuration for game phase detection and transitions.
    
    This class encapsulates all configurable parameters for phase detection,
    including boundary thresholds and transition parameters.
    
    Attributes:
        early_max_moves: Maximum move count for Early phase (default: 15)
        mid_max_moves: Maximum move count for Mid phase (default: 35)
        transition_window: Number of moves before/after boundary for transitions
                          (default: 2)
        interpolation_method: Method for phase transitions - 'linear' or 'sigmoid'
                             (default: 'linear')
        
    Example:
        >>> config = PhaseConfig(early_max_moves=20, mid_max_moves=40)
        >>> print(config.early_max_moves)
        20
    """
    early_max_moves: int = 15
    mid_max_moves: int = 35
    transition_window: int = 2
    interpolation_method: str = 'linear'
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.early_max_moves <= 0 or self.mid_max_moves <= 0:
            raise ValueError(
                "Phase boundaries must be positive integers: "
                f"early_max_moves={self.early_max_moves}, "
                f"mid_max_moves={self.mid_max_moves}"
            )
        
        if self.early_max_moves >= self.mid_max_moves:
            raise ValueError(
                f"'early_max_moves' ({self.early_max_moves}) must be less than "
                f"'mid_max_moves' ({self.mid_max_moves})"
            )
        
        if self.transition_window < 0:
            raise ValueError(
                f"'transition_window' must be non-negative, got {self.transition_window}"
            )
        
        if self.interpolation_method not in ('linear', 'sigmoid'):
            raise ValueError(
                f"'interpolation_method' must be 'linear' or 'sigmoid', "
                f"got '{self.interpolation_method}'"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseConfig':
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary containing configuration values
            
        Returns:
            PhaseConfig instance
            
        Raises:
            ValueError: If dictionary contains invalid values
        """
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            filepath: Path to JSON file for saving
            
        Raises:
            IOError: If file cannot be written
            ValueError: If configuration is invalid
        """
        self._validate()  # Ensure valid before saving
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PhaseConfig':
        """Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON file to load
            
        Returns:
            PhaseConfig instance loaded from file
            
        Raises:
            IOError: If file cannot be read
            ValueError: If file contains invalid configuration
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = cls.from_dict(data)
        config._validate()  # Validate after loading
        return config
    
    def update(self, **kwargs: Any) -> 'PhaseConfig':
        """Create a new configuration with updated values.
        
        This method creates a new PhaseConfig instance with updated values
        rather than modifying the existing one, ensuring immutability.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            New PhaseConfig instance with updated values
            
        Raises:
            ValueError: If any updated value is invalid
            
        Example:
            >>> config = PhaseConfig()
            >>> new_config = config.update(early_max_moves=20)
            >>> print(new_config.early_max_moves)
            20
        """
        current = self.to_dict()
        current.update(kwargs)
        return self.from_dict(current)


# Default configuration instance
DEFAULT_PHASE_CONFIG = PhaseConfig()

