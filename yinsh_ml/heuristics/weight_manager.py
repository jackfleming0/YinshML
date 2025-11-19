"""Weight management system for YINSH heuristic evaluator.

This module provides functionality for loading, saving, and managing
heuristic weights through configuration files and runtime updates.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import shutil
from datetime import datetime


class WeightManager:
    """Manages heuristic weights with file persistence and runtime updates.
    
    This class provides functionality to:
    - Load weights from JSON configuration files
    - Save weights to configuration files
    - Update weights at runtime
    - Validate weight constraints
    - Create backups of weight configurations
    - Restore from backups
    
    Example:
        >>> manager = WeightManager()
        >>> weights = manager.load_from_file('weights.json')
        >>> manager.update_weights('early', 'completed_runs_differential', 12.0)
        >>> manager.save_to_file('weights.json')
    """
    
    # Valid feature names
    VALID_FEATURES = [
        'completed_runs_differential',
        'potential_runs_count',
        'connected_marker_chains',
        'ring_positioning',
        'ring_spread',
        'board_control',
    ]
    
    # Valid phase names
    VALID_PHASES = ['early', 'mid', 'late']
    
    # Default weight constraints (min, max)
    DEFAULT_CONSTRAINTS = {
        'completed_runs_differential': (0.0, 50.0),
        'potential_runs_count': (0.0, 50.0),
        'connected_marker_chains': (0.0, 50.0),
        'ring_positioning': (0.0, 50.0),
        'ring_spread': (0.0, 50.0),
        'board_control': (0.0, 50.0),
    }
    
    def __init__(
        self,
        constraints: Optional[Dict[str, tuple]] = None,
        backup_dir: Optional[str] = None
    ):
        """Initialize the weight manager.
        
        Args:
            constraints: Optional dictionary mapping feature names to (min, max) tuples.
                        If None, DEFAULT_CONSTRAINTS will be used.
            backup_dir: Optional directory for storing weight backups.
                       If None, backups will be stored in './weight_backups'
        """
        self.constraints = constraints or self.DEFAULT_CONSTRAINTS.copy()
        self.backup_dir = Path(backup_dir) if backup_dir else Path('./weight_backups')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Current weights (loaded or default)
        self.weights: Dict[str, Dict[str, float]] = {}
    
    def load_from_file(self, filepath: str) -> Dict[str, Dict[str, float]]:
        """Load weights from a JSON configuration file.
        
        Args:
            filepath: Path to the JSON configuration file
            
        Returns:
            Dictionary with phase keys ('early', 'mid', 'late') mapping to
            feature weight dictionaries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid weight structure
            json.JSONDecodeError: If the file is not valid JSON
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Weight configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        self._validate_weight_structure(data)
        
        # Validate constraints
        self._validate_weight_constraints(data)
        
        self.weights = data
        return self.weights.copy()
    
    def save_to_file(
        self,
        filepath: str,
        create_backup: bool = True
    ) -> None:
        """Save current weights to a JSON configuration file.
        
        Args:
            filepath: Path to save the JSON configuration file
            create_backup: If True, creates a backup before saving
            
        Raises:
            ValueError: If weights are not set or invalid
            IOError: If file cannot be written
        """
        if not self.weights:
            raise ValueError("No weights loaded. Load weights first or use set_default_weights()")
        
        filepath = Path(filepath)
        
        # Create backup if requested and file exists
        if create_backup and filepath.exists():
            self._create_backup(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate before saving
        self._validate_weight_structure(self.weights)
        self._validate_weight_constraints(self.weights)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.weights, f, indent=2)
    
    def update_weights(
        self,
        phase: str,
        feature: str,
        value: float
    ) -> None:
        """Update a specific weight value at runtime.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            feature: Feature name (e.g., 'completed_runs_differential')
            value: New weight value
            
        Raises:
            ValueError: If phase, feature, or value is invalid
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid phase '{phase}'. Must be one of {self.VALID_PHASES}"
            )
        
        if feature not in self.VALID_FEATURES:
            raise ValueError(
                f"Invalid feature '{feature}'. Must be one of {self.VALID_FEATURES}"
            )
        
        # Validate constraint
        if feature in self.constraints:
            min_val, max_val = self.constraints[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Weight value {value} for '{feature}' is out of range "
                    f"[{min_val}, {max_val}]"
                )
        
        # Initialize phase if needed
        if phase not in self.weights:
            self.weights[phase] = {}
        
        # Update weight
        self.weights[phase][feature] = float(value)
    
    def update_phase_weights(
        self,
        phase: str,
        weights: Dict[str, float]
    ) -> None:
        """Update all weights for a specific phase.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            weights: Dictionary mapping feature names to weight values
            
        Raises:
            ValueError: If phase or weights are invalid
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid phase '{phase}'. Must be one of {self.VALID_PHASES}"
            )
        
        # Validate all weights before updating
        for feature, value in weights.items():
            if feature not in self.VALID_FEATURES:
                raise ValueError(
                    f"Invalid feature '{feature}'. Must be one of {self.VALID_FEATURES}"
                )
            
            if feature in self.constraints:
                min_val, max_val = self.constraints[feature]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Weight value {value} for '{feature}' is out of range "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Update phase weights
        if phase not in self.weights:
            self.weights[phase] = {}
        
        self.weights[phase].update({k: float(v) for k, v in weights.items()})
    
    def set_default_weights(self, weights: Dict[str, Dict[str, float]]) -> None:
        """Set default weights (used when no file is loaded).
        
        Args:
            weights: Dictionary with phase keys mapping to feature weight dictionaries
            
        Raises:
            ValueError: If weights structure is invalid
        """
        self._validate_weight_structure(weights)
        self._validate_weight_constraints(weights)
        self.weights = weights.copy()
    
    def get_weights(self) -> Dict[str, Dict[str, float]]:
        """Get current weights.
        
        Returns:
            Copy of current weights dictionary
        """
        return self.weights.copy()
    
    def get_weight(self, phase: str, feature: str) -> Optional[float]:
        """Get a specific weight value.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            feature: Feature name
            
        Returns:
            Weight value or None if not set
        """
        return self.weights.get(phase, {}).get(feature)
    
    def _validate_weight_structure(self, weights: Dict[str, Any]) -> None:
        """Validate weight dictionary structure.
        
        Args:
            weights: Weight dictionary to validate
            
        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(weights, dict):
            raise ValueError(f"Weights must be a dictionary, got {type(weights)}")
        
        # Check for required phases
        for phase in self.VALID_PHASES:
            if phase not in weights:
                raise ValueError(f"Missing required phase '{phase}' in weights")
            
            if not isinstance(weights[phase], dict):
                raise ValueError(f"Weights for phase '{phase}' must be a dictionary")
            
            # Check for required features
            for feature in self.VALID_FEATURES:
                if feature not in weights[phase]:
                    raise ValueError(
                        f"Missing required feature '{feature}' in phase '{phase}'"
                    )
                
                value = weights[phase][feature]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Weight value for '{phase}.{feature}' must be numeric, "
                        f"got {type(value)}"
                    )
    
    def _validate_weight_constraints(self, weights: Dict[str, Any]) -> None:
        """Validate weight values against constraints.
        
        Args:
            weights: Weight dictionary to validate
            
        Raises:
            ValueError: If any weight violates constraints
        """
        for phase in self.VALID_PHASES:
            if phase not in weights:
                continue
            
            for feature, value in weights[phase].items():
                if feature in self.constraints:
                    min_val, max_val = self.constraints[feature]
                    if not (min_val <= value <= max_val):
                        raise ValueError(
                            f"Weight value {value} for '{phase}.{feature}' is out of range "
                            f"[{min_val}, {max_val}]"
                        )
    
    def _create_backup(self, filepath: Path) -> None:
        """Create a backup of the weight file.
        
        Args:
            filepath: Path to the file to backup
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(filepath, backup_path)
    
    def list_backups(self) -> list:
        """List all available weight backups.
        
        Returns:
            List of backup file paths
        """
        if not self.backup_dir.exists():
            return []
        
        return sorted(self.backup_dir.glob('*.json'), reverse=True)
    
    def restore_from_backup(self, backup_path: str) -> Dict[str, Dict[str, float]]:
        """Restore weights from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            Restored weights dictionary
            
        Raises:
            FileNotFoundError: If backup file doesn't exist
            ValueError: If backup file is invalid
        """
        return self.load_from_file(backup_path)

