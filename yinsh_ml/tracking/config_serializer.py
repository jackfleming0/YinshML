"""
Configuration serialization and management utilities for experiment tracking.

Provides enhanced configuration handling including:
- Comprehensive configuration capture from multiple sources
- Sensitive value masking for security
- Configuration comparison and diff utilities
- Configuration reconstruction from tracking data
"""

import os
import json
import copy
import logging
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import re

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationMetadata:
    """Metadata about a configuration snapshot."""
    timestamp: str
    source: str  # 'experiment', 'system', 'environment', 'file'
    version: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None
    masked_keys: Optional[List[str]] = None


class ConfigurationSerializer:
    """
    Enhanced configuration serialization and management utility.
    
    Provides comprehensive configuration handling for experiment tracking
    including automatic capture, sensitive value masking, and comparison tools.
    """
    
    # Default sensitive key patterns (case-insensitive)
    DEFAULT_SENSITIVE_PATTERNS = [
        r'.*password.*', r'.*secret.*', r'.*key.*', r'.*token.*', 
        r'.*private.*', r'.*credential.*', r'.*auth.*', r'.*pass.*',
        r'.*pwd.*', r'.*apikey.*', r'.*api_key.*', r'.*access_key.*',
        r'.*secret_key.*', r'.*private_key.*', r'.*session.*'
    ]
    
    # Configuration file patterns to search for
    CONFIG_FILE_PATTERNS = [
        '*.json', '*.yaml', '*.yml', '*.toml', '*.ini', 
        'config.*', '*.config', 'settings.*', '*.settings',
        '.env', '.env.*', 'environment.*'
    ]
    
    def __init__(self, 
                 sensitive_patterns: Optional[List[str]] = None,
                 mask_value: str = "***MASKED***",
                 max_config_files: int = 20,
                 max_file_size_mb: float = 10.0):
        """
        Initialize configuration serializer.
        
        Args:
            sensitive_patterns: Custom regex patterns for sensitive keys
            mask_value: Value to use for masking sensitive data
            max_config_files: Maximum number of config files to process
            max_file_size_mb: Maximum file size to process (MB)
        """
        self.sensitive_patterns = sensitive_patterns or self.DEFAULT_SENSITIVE_PATTERNS
        self.mask_value = mask_value
        self.max_config_files = max_config_files
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        
        # Compile regex patterns for performance
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.sensitive_patterns
        ]
    
    def is_sensitive_key(self, key: str) -> bool:
        """
        Check if a configuration key contains sensitive information.
        
        Args:
            key: Configuration key name
            
        Returns:
            True if the key appears to contain sensitive information
        """
        return any(pattern.match(key) for pattern in self._compiled_patterns)
    
    def mask_sensitive_values(self, config: Dict[str, Any], 
                            in_place: bool = False) -> Tuple[Dict[str, Any], List[str]]:
        """
        Mask sensitive values in configuration dictionary.
        
        Args:
            config: Configuration dictionary to mask
            in_place: Whether to modify the original dictionary
            
        Returns:
            Tuple of (masked_config, list_of_masked_keys)
        """
        if not in_place:
            config = copy.deepcopy(config)
        
        masked_keys = []
        
        def _mask_recursive(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if self.is_sensitive_key(key):
                        obj[key] = self.mask_value
                        masked_keys.append(current_path)
                    else:
                        obj[key] = _mask_recursive(value, current_path)
                return obj
            elif isinstance(obj, list):
                return [_mask_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            else:
                return obj
        
        masked_config = _mask_recursive(config)
        return masked_config, masked_keys
    
    def serialize_dataclass_config(self, config_obj: Any) -> Dict[str, Any]:
        """
        Serialize a dataclass configuration object to dictionary.
        
        Args:
            config_obj: Dataclass configuration object
            
        Returns:
            Dictionary representation of the configuration
        """
        if is_dataclass(config_obj):
            return asdict(config_obj)
        elif hasattr(config_obj, '__dict__'):
            return vars(config_obj)
        else:
            return {'value': config_obj, 'type': type(config_obj).__name__}
    
    def capture_comprehensive_configuration(self, 
                                          user_config: Optional[Dict[str, Any]] = None,
                                          experiment_path: Optional[Path] = None,
                                          include_environment: bool = True,
                                          include_system_config: bool = True) -> Dict[str, Any]:
        """
        Capture comprehensive configuration from multiple sources.
        
        Args:
            user_config: User-provided experiment configuration
            experiment_path: Path to experiment directory to scan for config files
            include_environment: Whether to include environment variables
            include_system_config: Whether to include system configuration
            
        Returns:
            Comprehensive configuration dictionary
        """
        comprehensive_config = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'capture_sources': [],
                'total_masked_keys': 0
            }
        }
        
        # 1. User-provided configuration
        if user_config:
            masked_user_config, masked_keys = self.mask_sensitive_values(user_config)
            comprehensive_config['user_config'] = {
                'data': masked_user_config,
                'metadata': ConfigurationMetadata(
                    timestamp=datetime.now().isoformat(),
                    source='user_provided',
                    checksum=self._calculate_checksum(user_config),
                    size_bytes=len(json.dumps(user_config)),
                    masked_keys=masked_keys
                ).__dict__
            }
            comprehensive_config['metadata']['capture_sources'].append('user_config')
            comprehensive_config['metadata']['total_masked_keys'] += len(masked_keys)
        
        # 2. Environment variables
        if include_environment:
            env_config = self._capture_environment_config()
            if env_config:
                comprehensive_config['environment_config'] = env_config
                comprehensive_config['metadata']['capture_sources'].append('environment')
                comprehensive_config['metadata']['total_masked_keys'] += len(
                    env_config.get('metadata', {}).get('masked_keys', [])
                )
        
        # 3. Configuration files
        if experiment_path and experiment_path.exists():
            file_configs = self._capture_file_configurations(experiment_path)
            if file_configs:
                comprehensive_config['file_configs'] = file_configs
                comprehensive_config['metadata']['capture_sources'].append('files')
                
                # Count masked keys from all files
                for file_config in file_configs.values():
                    comprehensive_config['metadata']['total_masked_keys'] += len(
                        file_config.get('metadata', {}).get('masked_keys', [])
                    )
        
        # 4. System configuration
        if include_system_config:
            system_config = self._capture_system_configuration()
            if system_config:
                comprehensive_config['system_config'] = system_config
                comprehensive_config['metadata']['capture_sources'].append('system')
        
        return comprehensive_config
    
    def _capture_environment_config(self) -> Optional[Dict[str, Any]]:
        """Capture relevant environment variables."""
        try:
            # Get environment variables that might be configuration-related
            config_env_vars = {}
            
            for key, value in os.environ.items():
                # Include variables that look like configuration
                if any(pattern in key.upper() for pattern in [
                    'CONFIG', 'SETTING', 'PARAM', 'OPT', 'FLAG', 
                    'YINSH', 'ML', 'TRAIN', 'EXPERIMENT', 'MODEL'
                ]):
                    config_env_vars[key] = value
            
            if not config_env_vars:
                return None
            
            # Mask sensitive values
            masked_config, masked_keys = self.mask_sensitive_values(config_env_vars)
            
            return {
                'data': masked_config,
                'metadata': ConfigurationMetadata(
                    timestamp=datetime.now().isoformat(),
                    source='environment_variables',
                    checksum=self._calculate_checksum(config_env_vars),
                    size_bytes=len(json.dumps(config_env_vars)),
                    masked_keys=masked_keys
                ).__dict__
            }
            
        except Exception as e:
            logger.warning(f"Failed to capture environment configuration: {e}")
            return None
    
    def _capture_file_configurations(self, experiment_path: Path) -> Dict[str, Dict[str, Any]]:
        """Capture configuration from files in experiment directory."""
        file_configs = {}
        files_processed = 0
        
        try:
            for pattern in self.CONFIG_FILE_PATTERNS:
                if files_processed >= self.max_config_files:
                    break
                    
                for config_file in experiment_path.glob(pattern):
                    if files_processed >= self.max_config_files:
                        break
                    
                    try:
                        # Check file size
                        if config_file.stat().st_size > self.max_file_size_bytes:
                            logger.warning(f"Skipping large config file: {config_file}")
                            continue
                        
                        # Read and parse file
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        parsed_config = self._parse_configuration_file(config_file.name, content)
                        if parsed_config is not None:
                            # Mask sensitive values
                            masked_config, masked_keys = self.mask_sensitive_values(parsed_config)
                            
                            file_configs[str(config_file.name)] = {
                                'data': masked_config,
                                'metadata': ConfigurationMetadata(
                                    timestamp=datetime.now().isoformat(),
                                    source=f'file:{config_file.name}',
                                    checksum=self._calculate_checksum(parsed_config),
                                    size_bytes=len(content),
                                    masked_keys=masked_keys
                                ).__dict__,
                                'file_info': {
                                    'path': str(config_file),
                                    'size': config_file.stat().st_size,
                                    'modified_time': config_file.stat().st_mtime
                                }
                            }
                            files_processed += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to process config file {config_file}: {e}")
                        continue
            
            return file_configs
            
        except Exception as e:
            logger.warning(f"Failed to capture file configurations: {e}")
            return {}
    
    def _parse_configuration_file(self, filename: str, content: str) -> Optional[Dict[str, Any]]:
        """Parse configuration file based on its format."""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.json':
                return json.loads(content)
            elif file_ext in ['.yaml', '.yml']:
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    logger.warning(f"PyYAML not available, skipping {filename}")
                    return None
            elif file_ext == '.toml':
                try:
                    import toml
                    return toml.loads(content)
                except ImportError:
                    logger.warning(f"toml library not available, skipping {filename}")
                    return None
            elif file_ext == '.ini':
                try:
                    import configparser
                    config = configparser.ConfigParser()
                    config.read_string(content)
                    return {section: dict(config[section]) for section in config.sections()}
                except Exception as e:
                    logger.warning(f"Failed to parse INI file {filename}: {e}")
                    return None
            elif filename.startswith('.env'):
                # Parse environment file
                env_config = {}
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_config[key.strip()] = value.strip()
                return env_config
            else:
                # Try to parse as JSON first, then YAML
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        return yaml.safe_load(content)
                    except:
                        return None
                        
        except Exception as e:
            logger.warning(f"Failed to parse configuration file {filename}: {e}")
            return None
    
    def _capture_system_configuration(self) -> Optional[Dict[str, Any]]:
        """Capture system-level configuration."""
        try:
            system_config = {
                'python_path': os.environ.get('PYTHONPATH', ''),
                'working_directory': str(Path.cwd()),
                'user_home': str(Path.home()),
                'temp_directory': os.environ.get('TMPDIR', '/tmp'),
            }
            
            # Add system-specific paths and settings
            if os.name == 'posix':
                system_config.update({
                    'shell': os.environ.get('SHELL', ''),
                    'path': os.environ.get('PATH', ''),
                })
            elif os.name == 'nt':
                system_config.update({
                    'comspec': os.environ.get('COMSPEC', ''),
                    'path': os.environ.get('PATH', ''),
                })
            
            # Mask sensitive values
            masked_config, masked_keys = self.mask_sensitive_values(system_config)
            
            return {
                'data': masked_config,
                'metadata': ConfigurationMetadata(
                    timestamp=datetime.now().isoformat(),
                    source='system_configuration',
                    checksum=self._calculate_checksum(system_config),
                    size_bytes=len(json.dumps(system_config)),
                    masked_keys=masked_keys
                ).__dict__
            }
            
        except Exception as e:
            logger.warning(f"Failed to capture system configuration: {e}")
            return None
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate checksum for configuration data."""
        try:
            config_str = json.dumps(config, sort_keys=True, default=str)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def compare_configurations(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two configuration dictionaries and highlight differences.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'identical': False,
            'differences': [],
            'added_keys': [],
            'removed_keys': [],
            'modified_keys': [],
            'summary': {}
        }
        
        def _compare_recursive(obj1: Any, obj2: Any, path: str = "") -> None:
            if type(obj1) != type(obj2):
                comparison['differences'].append({
                    'path': path,
                    'type': 'type_change',
                    'old_type': type(obj1).__name__,
                    'new_type': type(obj2).__name__,
                    'old_value': str(obj1),
                    'new_value': str(obj2)
                })
                return
            
            if isinstance(obj1, dict):
                keys1, keys2 = set(obj1.keys()), set(obj2.keys())
                
                # Added keys
                for key in keys2 - keys1:
                    full_path = f"{path}.{key}" if path else key
                    comparison['added_keys'].append(full_path)
                    comparison['differences'].append({
                        'path': full_path,
                        'type': 'added',
                        'new_value': obj2[key]
                    })
                
                # Removed keys
                for key in keys1 - keys2:
                    full_path = f"{path}.{key}" if path else key
                    comparison['removed_keys'].append(full_path)
                    comparison['differences'].append({
                        'path': full_path,
                        'type': 'removed',
                        'old_value': obj1[key]
                    })
                
                # Common keys
                for key in keys1 & keys2:
                    full_path = f"{path}.{key}" if path else key
                    _compare_recursive(obj1[key], obj2[key], full_path)
            
            elif isinstance(obj1, list):
                if len(obj1) != len(obj2):
                    comparison['differences'].append({
                        'path': path,
                        'type': 'list_length_change',
                        'old_length': len(obj1),
                        'new_length': len(obj2)
                    })
                
                for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                    _compare_recursive(item1, item2, f"{path}[{i}]")
            
            else:
                if obj1 != obj2:
                    comparison['modified_keys'].append(path)
                    comparison['differences'].append({
                        'path': path,
                        'type': 'value_change',
                        'old_value': obj1,
                        'new_value': obj2
                    })
        
        _compare_recursive(config1, config2)
        
        # Generate summary
        comparison['identical'] = len(comparison['differences']) == 0
        comparison['summary'] = {
            'total_differences': len(comparison['differences']),
            'added_keys_count': len(comparison['added_keys']),
            'removed_keys_count': len(comparison['removed_keys']),
            'modified_keys_count': len(comparison['modified_keys'])
        }
        
        return comparison
    
    def reconstruct_configuration(self, config_snapshot: Dict[str, Any], 
                                unmask_sensitive: bool = False) -> Dict[str, Any]:
        """
        Reconstruct configuration from a tracking snapshot.
        
        Args:
            config_snapshot: Configuration snapshot from tracking system
            unmask_sensitive: Whether to attempt to unmask sensitive values (not recommended)
            
        Returns:
            Reconstructed configuration dictionary
        """
        reconstructed = {}
        
        # Extract user configuration
        if 'user_config' in config_snapshot:
            user_config = config_snapshot['user_config']
            if isinstance(user_config, dict) and 'data' in user_config:
                reconstructed['user_config'] = user_config['data']
            else:
                reconstructed['user_config'] = user_config
        
        # Extract file configurations
        if 'file_configs' in config_snapshot:
            reconstructed['file_configs'] = {}
            for filename, file_config in config_snapshot['file_configs'].items():
                if isinstance(file_config, dict) and 'data' in file_config:
                    reconstructed['file_configs'][filename] = file_config['data']
                else:
                    reconstructed['file_configs'][filename] = file_config
        
        # Extract environment configuration
        if 'environment_config' in config_snapshot:
            env_config = config_snapshot['environment_config']
            if isinstance(env_config, dict) and 'data' in env_config:
                reconstructed['environment_config'] = env_config['data']
            else:
                reconstructed['environment_config'] = env_config
        
        # Add metadata about reconstruction
        reconstructed['_reconstruction_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'unmask_attempted': unmask_sensitive,
            'warning': 'Sensitive values remain masked for security' if not unmask_sensitive else None
        }
        
        return reconstructed
    
    def validate_configuration_integrity(self, config_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the integrity of a configuration snapshot.
        
        Args:
            config_snapshot: Configuration snapshot to validate
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'checksums_verified': 0,
            'checksums_failed': 0
        }
        
        def _validate_config_section(section_name: str, section_data: Any) -> None:
            if not isinstance(section_data, dict):
                validation['warnings'].append(f"Section {section_name} is not a dictionary")
                return
            
            # Check for metadata
            if 'metadata' in section_data:
                metadata = section_data['metadata']
                
                # Verify checksum if available
                if 'checksum' in metadata and 'data' in section_data:
                    try:
                        calculated_checksum = self._calculate_checksum(section_data['data'])
                        if calculated_checksum == metadata['checksum']:
                            validation['checksums_verified'] += 1
                        else:
                            validation['checksums_failed'] += 1
                            validation['errors'].append(
                                f"Checksum mismatch in {section_name}: "
                                f"expected {metadata['checksum']}, got {calculated_checksum}"
                            )
                    except Exception as e:
                        validation['warnings'].append(f"Failed to verify checksum for {section_name}: {e}")
        
        # Validate each configuration section
        for section_name in ['user_config', 'environment_config', 'system_config']:
            if section_name in config_snapshot:
                _validate_config_section(section_name, config_snapshot[section_name])
        
        # Validate file configurations
        if 'file_configs' in config_snapshot:
            for filename, file_config in config_snapshot['file_configs'].items():
                _validate_config_section(f"file_configs.{filename}", file_config)
        
        # Overall validation status
        validation['valid'] = len(validation['errors']) == 0
        
        return validation 