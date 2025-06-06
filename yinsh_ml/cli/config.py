"""
Configuration management for YinshML CLI.

Handles loading and managing CLI configuration settings from files and environment.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CLIConfig:
    """Manages CLI configuration settings."""
    
    DEFAULT_CONFIG = {
        # Database settings
        'database_path': None,  # Will be auto-detected if None
        
        # Output formatting
        'output_format': 'table',  # table, json, csv
        'color_output': True,
        'show_timestamps': True,
        'max_rows': 50,
        
        # Experiment defaults
        'default_status': 'running',
        'auto_capture_git': True,
        'auto_capture_system': True,
        'auto_capture_environment': True,
        
        # CLI behavior
        'confirm_destructive': True,
        'verbose': False,
        'quiet': False,
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file. If None, uses default locations.
        """
        self._config = self.DEFAULT_CONFIG.copy()
        self._config_file = config_file or self._find_config_file()
        self._load_config()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        # Check for config files in order of preference
        config_locations = [
            # Current directory
            Path.cwd() / '.yinsh-track.json',
            Path.cwd() / 'yinsh-track.json',
            # Home directory
            Path.home() / '.yinsh-track.json',
            Path.home() / '.config' / 'yinsh-track.json',
            # System-wide
            Path('/etc/yinsh-track.json'),
        ]
        
        for config_path in config_locations:
            if config_path.exists() and config_path.is_file():
                logger.debug(f"Found config file: {config_path}")
                return str(config_path)
        
        return None
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Load from file if it exists
        if self._config_file and os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                    logger.debug(f"Loaded config from {self._config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file {self._config_file}: {e}")
        
        # Override with environment variables
        self._load_env_config()
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'YINSH_TRACK_DB_PATH': 'database_path',
            'YINSH_TRACK_OUTPUT_FORMAT': 'output_format', 
            'YINSH_TRACK_COLOR': 'color_output',
            'YINSH_TRACK_VERBOSE': 'verbose',
            'YINSH_TRACK_QUIET': 'quiet',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if config_key in ['color_output', 'verbose', 'quiet']:
                    self._config[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    self._config[config_key] = env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with multiple values."""
        self._config.update(updates)
    
    def save(self, config_file: str = None):
        """Save current configuration to file."""
        target_file = config_file or self._config_file
        if not target_file:
            # Create default config file in user's home directory
            config_dir = Path.home() / '.config'
            config_dir.mkdir(exist_ok=True)
            target_file = str(config_dir / 'yinsh-track.json')
        
        try:
            with open(target_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Configuration saved to {target_file}")
        except IOError as e:
            logger.error(f"Failed to save config to {target_file}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self):
        return f"CLIConfig(config_file={self._config_file})"


# Global configuration instance
_config = None

def get_config() -> CLIConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = CLIConfig()
    return _config

def set_config(config: CLIConfig):
    """Set global configuration instance."""
    global _config
    _config = config 