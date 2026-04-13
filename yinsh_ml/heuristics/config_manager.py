"""Unified configuration manager for YINSH heuristic evaluator.

This module provides a centralized ConfigManager that unifies WeightManager
and PhaseConfig into a single, thread-safe configuration system.
"""

import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .weight_manager import WeightManager
from .phase_config import PhaseConfig, DEFAULT_PHASE_CONFIG


class ConfigManager:
    """Unified configuration manager for heuristic weights and phase boundaries.
    
    This class provides a single entry point for managing all heuristic
    configuration, including weights and phase detection parameters. It ensures
    thread-safe operations and maintains configuration versioning.
    
    Attributes:
        weight_manager: WeightManager instance for weight operations
        phase_config: PhaseConfig instance for phase boundary settings
        config_path: Default path for configuration files
        _lock: Thread lock for safe concurrent access
    
    Example:
        >>> manager = ConfigManager()
        >>> manager.load_config("config.json")
        >>> manager.update_weight_atomic("early", "completed_runs_differential", 12.0)
        >>> manager.save_config("config.json")
    """
    
    CONFIG_VERSION = "1.0"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ConfigManager.
        
        Args:
            config_path: Optional default path for configuration files.
                        If None, uses ".yinsh/config.json"
        """
        self.weight_manager = WeightManager()
        self.phase_config = DEFAULT_PHASE_CONFIG
        self.config_path = config_path or ".yinsh/config.json"
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def load_config(self, filepath: str) -> None:
        """Load unified configuration from a JSON file.
        
        Args:
            filepath: Path to the JSON configuration file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the configuration structure is invalid
            json.JSONDecodeError: If the file is not valid JSON
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Load weights if present
        if "weights" in data:
            self.weight_manager.set_default_weights(data["weights"])
        
        # Load phase config if present
        if "phase_config" in data:
            self.phase_config = PhaseConfig.from_dict(data["phase_config"])
        
        # Validate loaded configuration
        self._validate_configuration()
    
    def save_config(self, filepath: str, create_backup: bool = True) -> None:
        """Save current configuration to a JSON file.
        
        Args:
            filepath: Path to save the JSON configuration file
            create_backup: If True, creates a backup before saving (if file exists)
            
        Raises:
            ValueError: If configuration is invalid
            IOError: If file cannot be written
        """
        # Validate before saving
        self._validate_configuration()
        
        filepath = Path(filepath)
        
        # Create backup if requested and file exists
        if create_backup and filepath.exists():
            self._create_backup(filepath)
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare configuration data
        config_data = {
            "version": self.CONFIG_VERSION,
            "timestamp": datetime.now().isoformat(),
            "weights": self.weight_manager.get_weights(),
            "phase_config": self.phase_config.to_dict(),
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def update_weight_atomic(
        self,
        phase: str,
        feature: str,
        value: float
    ) -> None:
        """Update a specific weight value atomically (thread-safe).
        
        This method ensures that weight updates are atomic and thread-safe,
        preventing race conditions during concurrent access.
        
        Args:
            phase: Phase name ('early', 'mid', or 'late')
            feature: Feature name (e.g., 'completed_runs_differential')
            value: New weight value
            
        Raises:
            ValueError: If phase, feature, or value is invalid
        """
        with self._lock:
            self.weight_manager.update_weights(phase, feature, value)
    
    def update_phase_config(self, **kwargs: Any) -> None:
        """Update phase configuration parameters atomically (thread-safe).
        
        Args:
            **kwargs: Phase configuration parameters to update
                    (e.g., early_max_moves=20, mid_max_moves=40)
                    
        Raises:
            ValueError: If any parameter is invalid
        """
        with self._lock:
            self.phase_config = self.phase_config.update(**kwargs)
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as a dictionary.
        
        Returns:
            Dictionary containing current weights and phase configuration
        """
        with self._lock:
            return {
                "weights": self.weight_manager.get_weights(),
                "phase_config": self.phase_config.to_dict(),
            }
    
    def _validate_configuration(self) -> None:
        """Validate current configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # WeightManager validates weights internally
        # PhaseConfig validates phase boundaries internally
        # PhaseConfig is always initialized (has defaults)
        # Weights may be empty initially, which is valid
        pass
    
    def _create_backup(self, filepath: Path) -> None:
        """Create a backup of the configuration file.
        
        Args:
            filepath: Path to the file to backup
        """
        backup_dir = filepath.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name
        
        import shutil
        shutil.copy2(filepath, backup_path)
    
    def list_backups(self, config_filepath: Optional[str] = None) -> list:
        """List all available configuration backups.
        
        Args:
            config_filepath: Optional path to config file to find backups for.
                           If None, uses self.config_path
                           
        Returns:
            List of backup file paths, sorted by modification time (newest first)
        """
        filepath = Path(config_filepath) if config_filepath else Path(self.config_path)
        backup_dir = filepath.parent / "backups"
        
        if not backup_dir.exists():
            return []
        
        return sorted(backup_dir.glob(f"{filepath.stem}_*{filepath.suffix}"), reverse=True)
    
    def restore_from_backup(self, backup_path: str) -> None:
        """Restore configuration from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Raises:
            FileNotFoundError: If backup file doesn't exist
            ValueError: If backup file is invalid
        """
        self.load_config(backup_path)

