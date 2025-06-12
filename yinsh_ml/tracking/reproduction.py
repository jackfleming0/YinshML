"""
Experiment reproduction engine for YinshML.

Provides functionality to capture comprehensive metadata for one-command reproduction
of machine learning experiments, including environment, random seed, and configuration state.
"""

import json
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import xml.etree.ElementTree as ET

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from .experiment_tracker import ExperimentTracker
    from .utils import DatabaseConnectionManager
except ImportError:
    from experiment_tracker import ExperimentTracker
    from utils import DatabaseConnectionManager


logger = logging.getLogger(__name__)


class ReproductionEngineError(Exception):
    """Base exception for ReproductionEngine operations."""
    pass


class ReproductionMetadataCaptureError(ReproductionEngineError):
    """Raised when metadata capture fails."""
    pass


class ReproductionStorageError(ReproductionEngineError):
    """Raised when storage operations fail."""
    pass


class ValidationError(ReproductionEngineError):
    """Raised when validation operations fail."""
    pass


class ConflictResolutionError(ReproductionEngineError):
    """Raised when environment conflict resolution fails."""
    pass


class ResultComparator:
    """
    Advanced result comparison framework supporting multiple data types.
    
    Provides comprehensive comparison capabilities for ML experiment results including:
    - Numerical metrics with statistical validation
    - File checksums and binary comparisons
    - Model weight tensor comparisons
    - Custom tolerance-based validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ResultComparator.
        
        Args:
            config: Configuration options for comparison thresholds and methods
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Default tolerance levels
        self.default_tolerances = {
            'absolute': self.config.get('default_absolute_tolerance', 1e-6),
            'relative': self.config.get('default_relative_tolerance', 1e-6),
            'percentage': self.config.get('default_percentage_tolerance', 0.01)
        }
    
    def compare_metrics(self, original: Dict[str, Any], reproduced: Dict[str, Any], 
                       tolerances: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Compare metric dictionaries with configurable tolerances.
        
        Args:
            original: Original experiment metrics
            reproduced: Reproduced experiment metrics  
            tolerances: Per-metric tolerance specifications
            
        Returns:
            Detailed comparison results with validation status
        """
        try:
            result = {
                'overall_match': True,
                'metric_comparisons': {},
                'missing_metrics': [],
                'extra_metrics': [],
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            tolerances = tolerances or {}
            
            # Check for missing metrics
            for metric in original:
                if metric not in reproduced:
                    result['missing_metrics'].append(metric)
                    result['overall_match'] = False
            
            # Check for extra metrics
            for metric in reproduced:
                if metric not in original:
                    result['extra_metrics'].append(metric)
            
            # Compare common metrics
            for metric in original:
                if metric in reproduced:
                    metric_result = self._compare_single_metric(
                        original[metric], reproduced[metric], 
                        tolerances.get(metric, self.default_tolerances)
                    )
                    result['metric_comparisons'][metric] = metric_result
                    if not metric_result['match']:
                        result['overall_match'] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Metric comparison failed: {e}")
            return {'error': str(e), 'overall_match': False}
    
    def _compare_single_metric(self, original_value: Any, reproduced_value: Any, 
                              tolerance: Dict[str, float]) -> Dict[str, Any]:
        """Compare a single metric value with tolerance checks."""
        try:
            result = {
                'match': False,
                'original_value': original_value,
                'reproduced_value': reproduced_value,
                'difference': None,
                'tolerance_used': tolerance.copy()
            }
            
            # Handle different value types
            if isinstance(original_value, (int, float)) and isinstance(reproduced_value, (int, float)):
                # Numerical comparison
                abs_diff = abs(original_value - reproduced_value)
                rel_diff = abs_diff / max(abs(original_value), 1e-10) if original_value != 0 else abs_diff
                
                result['difference'] = {
                    'absolute': abs_diff,
                    'relative': rel_diff,
                    'percentage': rel_diff * 100
                }
                
                # Check tolerances
                if (abs_diff <= tolerance.get('absolute', self.default_tolerances['absolute']) or
                    rel_diff <= tolerance.get('relative', self.default_tolerances['relative']) or
                    rel_diff <= tolerance.get('percentage', self.default_tolerances['percentage']) / 100):
                    result['match'] = True
                    
            elif original_value == reproduced_value:
                # Exact match for non-numerical values
                result['match'] = True
                result['difference'] = 0
                
            elif NUMPY_AVAILABLE and isinstance(original_value, np.ndarray) and isinstance(reproduced_value, np.ndarray):
                # NumPy array comparison
                if original_value.shape == reproduced_value.shape:
                    abs_diff = np.abs(original_value - reproduced_value)
                    max_abs_diff = np.max(abs_diff)
                    mean_abs_diff = np.mean(abs_diff)
                    
                    result['difference'] = {
                        'max_absolute': float(max_abs_diff),
                        'mean_absolute': float(mean_abs_diff),
                        'shape_match': True
                    }
                    
                    if max_abs_diff <= tolerance.get('absolute', self.default_tolerances['absolute']):
                        result['match'] = True
                else:
                    result['difference'] = {
                        'shape_match': False,
                        'original_shape': original_value.shape,
                        'reproduced_shape': reproduced_value.shape
                    }
            
            return result
            
        except Exception as e:
            return {
                'match': False,
                'error': str(e),
                'original_value': str(original_value),
                'reproduced_value': str(reproduced_value)
            }
    
    def compare_files(self, original_files: List[Union[str, Path]], 
                     reproduced_files: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Compare file contents using checksums and binary comparison.
        
        Args:
            original_files: List of original file paths
            reproduced_files: List of reproduced file paths
            
        Returns:
            File comparison results with checksums and match status
        """
        try:
            result = {
                'overall_match': True,
                'file_comparisons': {},
                'missing_files': [],
                'extra_files': [],
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            # Convert to Path objects and create sets for comparison
            orig_paths = {Path(f) for f in original_files}
            repro_paths = {Path(f) for f in reproduced_files}
            
            # Find missing and extra files
            orig_names = {p.name for p in orig_paths}
            repro_names = {p.name for p in repro_paths}
            
            result['missing_files'] = list(orig_names - repro_names)
            result['extra_files'] = list(repro_names - orig_names)
            
            if result['missing_files'] or result['extra_files']:
                result['overall_match'] = False
            
            # Compare common files
            common_names = orig_names & repro_names
            for name in common_names:
                orig_file = next(p for p in orig_paths if p.name == name)
                repro_file = next(p for p in repro_paths if p.name == name)
                
                file_result = self._compare_single_file(orig_file, repro_file)
                result['file_comparisons'][name] = file_result
                
                if not file_result['match']:
                    result['overall_match'] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"File comparison failed: {e}")
            return {'error': str(e), 'overall_match': False}
    
    def _compare_single_file(self, original_path: Path, reproduced_path: Path) -> Dict[str, Any]:
        """Compare two files using checksums and metadata."""
        try:
            result = {
                'match': False,
                'original_path': str(original_path),
                'reproduced_path': str(reproduced_path),
                'checksums': {},
                'file_info': {}
            }
            
            # Check if both files exist
            if not original_path.exists():
                result['error'] = f"Original file does not exist: {original_path}"
                return result
            
            if not reproduced_path.exists():
                result['error'] = f"Reproduced file does not exist: {reproduced_path}"
                return result
            
            # Calculate checksums
            orig_checksum = self._calculate_file_checksum(original_path)
            repro_checksum = self._calculate_file_checksum(reproduced_path)
            
            result['checksums'] = {
                'original': orig_checksum,
                'reproduced': repro_checksum,
                'match': orig_checksum == repro_checksum
            }
            
            # File metadata comparison
            orig_stat = original_path.stat()
            repro_stat = reproduced_path.stat()
            
            result['file_info'] = {
                'size_match': orig_stat.st_size == repro_stat.st_size,
                'original_size': orig_stat.st_size,
                'reproduced_size': repro_stat.st_size
            }
            
            result['match'] = result['checksums']['match']
            
            return result
            
        except Exception as e:
            return {
                'match': False,
                'error': str(e),
                'original_path': str(original_path),
                'reproduced_path': str(reproduced_path)
            }
    
    def _calculate_file_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum using specified algorithm."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def compare_model_weights(self, original_weights: Union[str, Path, Dict[str, Any]], 
                             reproduced_weights: Union[str, Path, Dict[str, Any]],
                             tolerance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compare model weights with tensor-level validation.
        
        Args:
            original_weights: Original model weights (file path or dict)
            reproduced_weights: Reproduced model weights (file path or dict)
            tolerance: Tolerance levels for weight comparison
            
        Returns:
            Detailed weight comparison results
        """
        try:
            result = {
                'overall_match': True,
                'weight_comparisons': {},
                'missing_weights': [],
                'extra_weights': [],
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            tolerance = tolerance or self.default_tolerances
            
            # Load weights if paths provided
            if isinstance(original_weights, (str, Path)):
                original_weights = self._load_model_weights(Path(original_weights))
            if isinstance(reproduced_weights, (str, Path)):
                reproduced_weights = self._load_model_weights(Path(reproduced_weights))
            
            if not isinstance(original_weights, dict) or not isinstance(reproduced_weights, dict):
                result['error'] = "Could not load model weights as dictionaries"
                result['overall_match'] = False
                return result
            
            # Compare weight keys
            orig_keys = set(original_weights.keys())
            repro_keys = set(reproduced_weights.keys())
            
            result['missing_weights'] = list(orig_keys - repro_keys)
            result['extra_weights'] = list(repro_keys - orig_keys)
            
            if result['missing_weights'] or result['extra_weights']:
                result['overall_match'] = False
            
            # Compare common weights
            common_keys = orig_keys & repro_keys
            for key in common_keys:
                weight_result = self._compare_single_metric(
                    original_weights[key], reproduced_weights[key], tolerance
                )
                result['weight_comparisons'][key] = weight_result
                
                if not weight_result['match']:
                    result['overall_match'] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model weight comparison failed: {e}")
            return {'error': str(e), 'overall_match': False}
    
    def _load_model_weights(self, weights_path: Path) -> Optional[Dict[str, Any]]:
        """Load model weights from file based on extension."""
        try:
            if weights_path.suffix == '.pth' and TORCH_AVAILABLE:
                import torch
                return torch.load(weights_path, map_location='cpu')
            elif weights_path.suffix in ['.pkl', '.pickle']:
                import pickle
                with open(weights_path, 'rb') as f:
                    return pickle.load(f)
            elif weights_path.suffix == '.npz' and NUMPY_AVAILABLE:
                return dict(np.load(weights_path))
            else:
                self.logger.warning(f"Unsupported weight file format: {weights_path.suffix}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load weights from {weights_path}: {e}")
            return None


class ConflictResolver:
    """
    Advanced environment conflict resolution system.
    
    Provides intelligent resolution of package dependency conflicts with:
    - Automatic version resolution strategies
    - Multi-step conflict resolution attempts
    - Fallback and workaround suggestions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConflictResolver.
        
        Args:
            config: Configuration options for resolution strategies
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Resolution strategies (in order of preference)
        self.resolution_strategies = [
            'downgrade_conflicting',
            'upgrade_compatible', 
            'remove_optional',
            'suggest_alternatives'
        ]
    
    def resolve_environment_conflicts(self, conflicts: List[str], 
                                    metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment conflicts using multiple strategies.
        
        Args:
            conflicts: List of conflict descriptions/error messages
            metadata: Environment metadata for context
            
        Returns:
            Resolution results with suggested actions
        """
        try:
            result = {
                'resolution_successful': False,
                'applied_strategies': [],
                'resolved_conflicts': [],
                'remaining_conflicts': [],
                'suggested_actions': [],
                'resolution_timestamp': datetime.now().isoformat()
            }
            
            parsed_conflicts = self._parse_conflicts(conflicts)
            remaining_conflicts = parsed_conflicts.copy()
            
            # Try each resolution strategy
            for strategy in self.resolution_strategies:
                if not remaining_conflicts:
                    break
                    
                strategy_result = self._apply_resolution_strategy(
                    strategy, remaining_conflicts, metadata
                )
                
                if strategy_result['success']:
                    result['applied_strategies'].append(strategy)
                    result['resolved_conflicts'].extend(strategy_result['resolved'])
                    remaining_conflicts = [c for c in remaining_conflicts 
                                         if c not in strategy_result['resolved']]
                    
                    if strategy_result.get('actions'):
                        result['suggested_actions'].extend(strategy_result['actions'])
            
            result['remaining_conflicts'] = remaining_conflicts
            result['resolution_successful'] = len(remaining_conflicts) == 0
            
            # Add manual intervention suggestions if needed
            if remaining_conflicts:
                manual_suggestions = self._generate_manual_intervention_suggestions(
                    remaining_conflicts, metadata
                )
                result['suggested_actions'].extend(manual_suggestions)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return {
                'resolution_successful': False,
                'error': str(e),
                'resolution_timestamp': datetime.now().isoformat()
            }
    
    def _parse_conflicts(self, conflicts: List[str]) -> List[Dict[str, Any]]:
        """Parse conflict messages into structured format."""
        parsed = []
        
        for conflict in conflicts:
            conflict_info = {
                'original_message': conflict,
                'type': 'unknown',
                'packages': [],
                'versions': {}
            }
            
            # Try to extract package names and versions from common conflict patterns
            if 'requires' in conflict.lower() and 'but' in conflict.lower():
                conflict_info['type'] = 'version_conflict'
                # Extract package names using simple regex patterns
                import re
                packages = re.findall(r'(\w+(?:[-_]\w+)*)', conflict)
                conflict_info['packages'] = packages[:3]  # Take first few as likely package names
                
            elif 'incompatible' in conflict.lower():
                conflict_info['type'] = 'incompatibility'
                
            elif 'not found' in conflict.lower() or 'missing' in conflict.lower():
                conflict_info['type'] = 'missing_package'
            
            parsed.append(conflict_info)
        
        return parsed
    
    def _apply_resolution_strategy(self, strategy: str, conflicts: List[Dict[str, Any]], 
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific resolution strategy to conflicts."""
        try:
            if strategy == 'downgrade_conflicting':
                return self._strategy_downgrade_conflicting(conflicts, metadata)
            elif strategy == 'upgrade_compatible':
                return self._strategy_upgrade_compatible(conflicts, metadata)
            elif strategy == 'remove_optional':
                return self._strategy_remove_optional(conflicts, metadata)
            elif strategy == 'suggest_alternatives':
                return self._strategy_suggest_alternatives(conflicts, metadata)
            else:
                return {'success': False, 'error': f'Unknown strategy: {strategy}'}
                
        except Exception as e:
            self.logger.error(f"Strategy {strategy} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _strategy_downgrade_conflicting(self, conflicts: List[Dict[str, Any]], 
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy: Downgrade conflicting packages to compatible versions."""
        resolved = []
        actions = []
        
        for conflict in conflicts:
            if conflict['type'] == 'version_conflict' and conflict['packages']:
                package = conflict['packages'][0]  # Primary conflicting package
                actions.append(f"pip install '{package}<current_version' --upgrade")
                actions.append(f"Consider downgrading {package} to resolve conflicts")
                resolved.append(conflict)
        
        return {
            'success': len(resolved) > 0,
            'resolved': resolved,
            'actions': actions
        }
    
    def _strategy_upgrade_compatible(self, conflicts: List[Dict[str, Any]], 
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy: Upgrade packages to find compatible versions."""
        resolved = []
        actions = []
        
        for conflict in conflicts:
            if conflict['type'] in ['version_conflict', 'incompatibility']:
                if conflict['packages']:
                    for package in conflict['packages'][:2]:  # Handle first two packages
                        actions.append(f"pip install --upgrade {package}")
                    resolved.append(conflict)
        
        return {
            'success': len(resolved) > 0,
            'resolved': resolved,
            'actions': actions
        }
    
    def _strategy_remove_optional(self, conflicts: List[Dict[str, Any]], 
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy: Remove optional packages that cause conflicts."""
        resolved = []
        actions = []
        
        # List of commonly optional packages that can be safely removed
        optional_packages = {
            'matplotlib', 'seaborn', 'plotly', 'bokeh',  # Visualization
            'jupyter', 'ipython', 'notebook',  # Interactive
            'tensorboard', 'wandb', 'mlflow',  # Logging/tracking
            'pytest', 'coverage', 'flake8'  # Development
        }
        
        for conflict in conflicts:
            if conflict['packages']:
                for package in conflict['packages']:
                    if package.lower() in optional_packages:
                        actions.append(f"pip uninstall {package} -y")
                        actions.append(f"Remove optional package {package} causing conflicts")
                        resolved.append(conflict)
                        break
        
        return {
            'success': len(resolved) > 0,
            'resolved': resolved,
            'actions': actions
        }
    
    def _strategy_suggest_alternatives(self, conflicts: List[Dict[str, Any]], 
                                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy: Suggest alternative packages or approaches."""
        resolved = []
        actions = []
        
        # Package alternatives mapping
        alternatives = {
            'tensorflow': ['tensorflow-cpu', 'torch'],
            'torch': ['tensorflow', 'jax'],
            'scipy': ['numpy', 'scikit-learn'],
            'pandas': ['polars', 'dask'],
            'matplotlib': ['plotly', 'seaborn', 'bokeh']
        }
        
        for conflict in conflicts:
            if conflict['packages']:
                for package in conflict['packages']:
                    if package in alternatives:
                        alt_packages = alternatives[package]
                        actions.append(f"Consider alternatives to {package}: {', '.join(alt_packages)}")
                        actions.append(f"pip uninstall {package} && pip install {alt_packages[0]}")
                        resolved.append(conflict)
                        break
        
        return {
            'success': len(resolved) > 0,
            'resolved': resolved,
            'actions': actions
        }
    
    def _generate_manual_intervention_suggestions(self, remaining_conflicts: List[Dict[str, Any]], 
                                                metadata: Dict[str, Any]) -> List[str]:
        """Generate manual intervention suggestions for unresolved conflicts."""
        suggestions = []
        
        if remaining_conflicts:
            suggestions.extend([
                "Manual intervention required for remaining conflicts:",
                "1. Create a fresh virtual environment: python -m venv fresh_env",
                "2. Install packages one by one to identify specific conflicts",
                "3. Use pip-tools to generate locked requirements: pip-compile requirements.in",
                "4. Consider using conda instead of pip for complex dependencies",
                "5. Check for platform-specific package versions",
                "6. Review package documentation for known compatibility issues"
            ])
            
            # Add specific suggestions based on conflict types
            conflict_types = {c['type'] for c in remaining_conflicts}
            
            if 'missing_package' in conflict_types:
                suggestions.append("7. Install missing packages from alternative sources (conda-forge, pip-extra-index)")
            
            if 'version_conflict' in conflict_types:
                suggestions.append("8. Pin specific versions in requirements.txt to avoid conflicts")
                suggestions.append("9. Use dependency resolution tools like pipdeptree to analyze conflicts")
        
        return suggestions


class ValidationFramework:
    """
    Comprehensive validation framework for experiment reproduction.
    
    Orchestrates validation across multiple dimensions:
    - Result comparison and statistical validation
    - Environment conflict resolution
    - Comprehensive reporting with CI/CD integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ValidationFramework.
        
        Args:
            config: Configuration options for validation behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize sub-components
        self.result_comparator = ResultComparator(self.config.get('comparison', {}))
        self.conflict_resolver = ConflictResolver(self.config.get('conflict_resolution', {}))
        
        # Validation thresholds
        self.validation_thresholds = {
            'metric_tolerance': self.config.get('metric_tolerance', 0.01),
            'file_match_required': self.config.get('file_match_required', True),
            'weight_tolerance': self.config.get('weight_tolerance', 1e-6),
            'environment_match_required': self.config.get('environment_match_required', False)
        }
    
    def validate_reproduction(self, original_results: Dict[str, Any], 
                            reproduced_results: Dict[str, Any],
                            reproduction_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of reproduction results.
        
        Args:
            original_results: Original experiment results and metadata
            reproduced_results: Reproduced experiment results
            reproduction_metadata: Metadata from reproduction process
            
        Returns:
            Comprehensive validation results
        """
        try:
            validation_result = {
                'overall_valid': True,
                'validation_categories': {},
                'statistical_analysis': {},
                'conflict_resolution': {},
                'recommendations': [],
                'validation_timestamp': datetime.now().isoformat(),
                'validation_config': self.validation_thresholds.copy()
            }
            
            # 1. Basic reproduction validation (existing logic)
            basic_validation = self._validate_reproduction_process(reproduction_metadata)
            validation_result['validation_categories']['reproduction_process'] = basic_validation
            
            if not basic_validation['valid']:
                validation_result['overall_valid'] = False
            
            # 2. Result comparison validation
            if 'metrics' in original_results and 'metrics' in reproduced_results:
                metric_validation = self._validate_metrics(
                    original_results['metrics'], reproduced_results['metrics']
                )
                validation_result['validation_categories']['metrics'] = metric_validation
                
                if not metric_validation['valid']:
                    validation_result['overall_valid'] = False
            
            # 3. File output validation
            if 'output_files' in original_results and 'output_files' in reproduced_results:
                file_validation = self._validate_files(
                    original_results['output_files'], reproduced_results['output_files']
                )
                validation_result['validation_categories']['files'] = file_validation
                
                if not file_validation['valid']:
                    validation_result['overall_valid'] = False
            
            # 4. Model weight validation
            if 'model_weights' in original_results and 'model_weights' in reproduced_results:
                weight_validation = self._validate_model_weights(
                    original_results['model_weights'], reproduced_results['model_weights']
                )
                validation_result['validation_categories']['model_weights'] = weight_validation
                
                if not weight_validation['valid']:
                    validation_result['overall_valid'] = False
            
            # 5. Statistical validation for non-deterministic results
            if self.config.get('enable_statistical_validation', True):
                statistical_validation = self._perform_statistical_validation(
                    original_results, reproduced_results
                )
                validation_result['statistical_analysis'] = statistical_validation
            
            # 6. Environment conflict resolution
            conflicts = reproduction_metadata.get('environment', {}).get('conflicts', [])
            if conflicts:
                conflict_resolution = self.conflict_resolver.resolve_environment_conflicts(
                    conflicts, reproduction_metadata
                )
                validation_result['conflict_resolution'] = conflict_resolution
                
                if not conflict_resolution.get('resolution_successful', False):
                    validation_result['overall_valid'] = False
            
            # 7. Generate recommendations
            validation_result['recommendations'] = self._generate_validation_recommendations(
                validation_result
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation framework failed: {e}")
            return {
                'overall_valid': False,
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def _validate_reproduction_process(self, reproduction_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the reproduction process itself (basic validation)."""
        try:
            result = {
                'valid': True,
                'checks': {},
                'warnings': [],
                'errors': []
            }
            
            # Check environment recreation
            env_result = reproduction_metadata.get('environment', {})
            result['checks']['environment'] = {
                'recreated': env_result.get('success', False),
                'package_manager': env_result.get('package_manager', 'unknown'),
                'conflicts': len(env_result.get('conflicts', []))
            }
            
            if not env_result.get('success', False):
                result['errors'].append("Environment recreation failed")
                result['valid'] = False
            
            # Check configuration restoration
            config_result = reproduction_metadata.get('configuration', {})
            result['checks']['configuration'] = {
                'restored': config_result.get('success', False),
                'files_restored': len(config_result.get('restored_files', [])),
                'errors': len(config_result.get('errors', []))
            }
            
            if config_result.get('errors'):
                result['warnings'].extend(config_result['errors'])
            
            # Check random seed restoration
            seed_result = reproduction_metadata.get('random_seeds', {})
            result['checks']['random_seeds'] = {
                'restored': seed_result.get('success', False),
                'seeds_restored': len(seed_result.get('seeds_restored', [])),
                'seeds_failed': len(seed_result.get('seeds_failed', []))
            }
            
            if seed_result.get('seeds_failed'):
                result['warnings'].extend(seed_result['seeds_failed'])
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'checks': {}
            }
    
    def _validate_metrics(self, original_metrics: Dict[str, Any], 
                         reproduced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metric reproduction with tolerance checks."""
        try:
            comparison_result = self.result_comparator.compare_metrics(
                original_metrics, reproduced_metrics
            )
            
            # Apply validation thresholds
            valid = comparison_result.get('overall_match', False)
            
            # Additional tolerance-based validation
            metric_comparisons = comparison_result.get('metric_comparisons', {})
            tolerance_failures = []
            
            for metric, comp in metric_comparisons.items():
                if not comp['match']:
                    diff = comp.get('difference', {})
                    if isinstance(diff, dict) and 'percentage' in diff:
                        if diff['percentage'] > self.validation_thresholds['metric_tolerance'] * 100:
                            tolerance_failures.append(f"{metric}: {diff['percentage']:.3f}% difference")
            
            return {
                'valid': valid and len(tolerance_failures) == 0,
                'comparison_result': comparison_result,
                'tolerance_failures': tolerance_failures,
                'threshold_used': self.validation_thresholds['metric_tolerance']
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _validate_files(self, original_files: List[str], 
                       reproduced_files: List[str]) -> Dict[str, Any]:
        """Validate file output reproduction."""
        try:
            comparison_result = self.result_comparator.compare_files(
                original_files, reproduced_files
            )
            
            valid = comparison_result.get('overall_match', False)
            
            # Apply file matching requirements
            if self.validation_thresholds['file_match_required']:
                missing_files = comparison_result.get('missing_files', [])
                if missing_files:
                    valid = False
            
            return {
                'valid': valid,
                'comparison_result': comparison_result,
                'match_required': self.validation_thresholds['file_match_required']
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _validate_model_weights(self, original_weights: Union[str, Dict], 
                               reproduced_weights: Union[str, Dict]) -> Dict[str, Any]:
        """Validate model weight reproduction."""
        try:
            tolerance = {'absolute': self.validation_thresholds['weight_tolerance']}
            
            comparison_result = self.result_comparator.compare_model_weights(
                original_weights, reproduced_weights, tolerance
            )
            
            return {
                'valid': comparison_result.get('overall_match', False),
                'comparison_result': comparison_result,
                'tolerance_used': tolerance
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _perform_statistical_validation(self, original_results: Dict[str, Any], 
                                      reproduced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical validation for non-deterministic outputs."""
        try:
            if not SCIPY_AVAILABLE:
                return {
                    'available': False,
                    'reason': 'SciPy not available for statistical validation'
                }
            
            statistical_result = {
                'available': True,
                'tests_performed': [],
                'significance_level': 0.05,
                'results': {}
            }
            
            # Extract numerical metrics for statistical testing
            orig_metrics = original_results.get('metrics', {})
            repro_metrics = reproduced_results.get('metrics', {})
            
            for metric_name in orig_metrics:
                if metric_name in repro_metrics:
                    orig_val = orig_metrics[metric_name]
                    repro_val = repro_metrics[metric_name]
                    
                    # Perform appropriate statistical test
                    if isinstance(orig_val, (int, float)) and isinstance(repro_val, (int, float)):
                        # For single values, use tolerance-based assessment
                        abs_diff = abs(orig_val - repro_val)
                        rel_diff = abs_diff / max(abs(orig_val), 1e-10) if orig_val != 0 else abs_diff
                        
                        statistical_result['results'][metric_name] = {
                            'test_type': 'tolerance_based',
                            'absolute_difference': abs_diff,
                            'relative_difference': rel_diff,
                            'within_tolerance': rel_diff <= 0.05  # 5% tolerance
                        }
                        
                        statistical_result['tests_performed'].append(f"tolerance_test_{metric_name}")
                    
                    elif (NUMPY_AVAILABLE and 
                          isinstance(orig_val, np.ndarray) and isinstance(repro_val, np.ndarray)):
                        # For arrays, perform Kolmogorov-Smirnov test
                        if orig_val.size > 1 and repro_val.size > 1:
                            ks_stat, p_value = stats.ks_2samp(orig_val.flatten(), repro_val.flatten())
                            
                            statistical_result['results'][metric_name] = {
                                'test_type': 'kolmogorov_smirnov',
                                'ks_statistic': float(ks_stat),
                                'p_value': float(p_value),
                                'significant_difference': p_value < statistical_result['significance_level']
                            }
                            
                            statistical_result['tests_performed'].append(f"ks_test_{metric_name}")
            
            return statistical_result
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def _generate_validation_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        if not validation_result['overall_valid']:
            recommendations.append("❌ Reproduction validation failed - see details below")
            
            # Specific recommendations based on failed categories
            categories = validation_result.get('validation_categories', {})
            
            if 'metrics' in categories and not categories['metrics'].get('valid', True):
                recommendations.append("🔢 Metric validation failed:")
                tolerance_failures = categories['metrics'].get('tolerance_failures', [])
                for failure in tolerance_failures:
                    recommendations.append(f"  - {failure}")
                recommendations.append("  Consider increasing tolerance thresholds or checking random seed restoration")
            
            if 'files' in categories and not categories['files'].get('valid', True):
                recommendations.append("📁 File validation failed:")
                comparison = categories['files'].get('comparison_result', {})
                missing = comparison.get('missing_files', [])
                if missing:
                    recommendations.append(f"  - Missing files: {', '.join(missing)}")
                recommendations.append("  Check output directory configuration and file generation logic")
            
            if 'model_weights' in categories and not categories['model_weights'].get('valid', True):
                recommendations.append("🏋️ Model weight validation failed:")
                recommendations.append("  - Check model architecture consistency and weight loading")
                recommendations.append("  - Verify random seed restoration before model initialization")
            
            # Environment conflict recommendations
            conflict_resolution = validation_result.get('conflict_resolution', {})
            if conflict_resolution and not conflict_resolution.get('resolution_successful', True):
                recommendations.append("🔧 Environment conflicts detected:")
                actions = conflict_resolution.get('suggested_actions', [])
                for action in actions[:3]:  # Show first 3 suggestions
                    recommendations.append(f"  - {action}")
        
        else:
            recommendations.append("✅ Reproduction validation successful!")
            recommendations.append("All validation checks passed within configured tolerances")
            
            # Additional improvement suggestions
            statistical = validation_result.get('statistical_analysis', {})
            if statistical.get('available') and statistical.get('results'):
                recommendations.append("📊 Statistical validation completed - consider reviewing significance levels")
        
        return recommendations
    
    def export_validation_report(self, validation_result: Dict[str, Any], 
                               format: str = 'json', 
                               output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export validation results in various formats for CI/CD integration.
        
        Args:
            validation_result: Validation results to export
            format: Export format ('json', 'junit', 'markdown')
            output_path: Optional path to save the report
            
        Returns:
            Formatted report content
        """
        try:
            if format == 'json':
                report_content = json.dumps(validation_result, indent=2, default=str)
                
            elif format == 'junit':
                report_content = self._generate_junit_xml(validation_result)
                
            elif format == 'markdown':
                report_content = self._generate_markdown_report(validation_result)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                    
                self.logger.info(f"Validation report exported to {output_path}")
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Report export failed: {e}")
            return f"Error exporting report: {e}"
    
    def _generate_junit_xml(self, validation_result: Dict[str, Any]) -> str:
        """Generate JUnit XML format for CI/CD systems."""
        try:
            # Create root testsuites element
            testsuites = ET.Element('testsuites')
            testsuite = ET.SubElement(testsuites, 'testsuite')
            testsuite.set('name', 'ReproductionValidation')
            testsuite.set('timestamp', validation_result.get('validation_timestamp', ''))
            
            # Count tests
            categories = validation_result.get('validation_categories', {})
            total_tests = len(categories)
            failures = sum(1 for cat in categories.values() if not cat.get('valid', True))
            
            testsuite.set('tests', str(total_tests))
            testsuite.set('failures', str(failures))
            testsuite.set('errors', '0')
            
            # Add test cases for each validation category
            for category_name, category_result in categories.items():
                testcase = ET.SubElement(testsuite, 'testcase')
                testcase.set('classname', 'ReproductionValidation')
                testcase.set('name', category_name)
                
                if not category_result.get('valid', True):
                    failure = ET.SubElement(testcase, 'failure')
                    failure.set('message', f'{category_name} validation failed')
                    failure.text = str(category_result.get('error', 'Validation check failed'))
            
            # Convert to string
            return ET.tostring(testsuites, encoding='unicode')
            
        except Exception as e:
            return f"Error generating JUnit XML: {e}"
    
    def _generate_markdown_report(self, validation_result: Dict[str, Any]) -> str:
        """Generate Markdown format report."""
        try:
            lines = [
                "# Reproduction Validation Report",
                f"**Generated:** {validation_result.get('validation_timestamp', 'Unknown')}",
                f"**Overall Status:** {'✅ PASS' if validation_result.get('overall_valid') else '❌ FAIL'}",
                "",
                "## Validation Summary",
                ""
            ]
            
            # Add category results
            categories = validation_result.get('validation_categories', {})
            for category, result in categories.items():
                status = '✅ PASS' if result.get('valid', True) else '❌ FAIL'
                lines.append(f"- **{category.replace('_', ' ').title()}:** {status}")
            
            lines.append("")
            
            # Add statistical analysis if available
            statistical = validation_result.get('statistical_analysis', {})
            if statistical.get('available'):
                lines.extend([
                    "## Statistical Analysis",
                    ""
                ])
                
                results = statistical.get('results', {})
                for metric, stat_result in results.items():
                    test_type = stat_result.get('test_type', 'unknown')
                    lines.append(f"- **{metric}** ({test_type})")
                    
                    if test_type == 'tolerance_based':
                        rel_diff = stat_result.get('relative_difference', 0)
                        within_tol = stat_result.get('within_tolerance', False)
                        status = '✅' if within_tol else '❌'
                        lines.append(f"  - Relative difference: {rel_diff:.4f} {status}")
                    
                    elif test_type == 'kolmogorov_smirnov':
                        p_value = stat_result.get('p_value', 1.0)
                        significant = stat_result.get('significant_difference', False)
                        status = '❌' if significant else '✅'
                        lines.append(f"  - P-value: {p_value:.6f} {status}")
                
                lines.append("")
            
            # Add recommendations
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                lines.extend([
                    "## Recommendations",
                    ""
                ])
                for rec in recommendations:
                    lines.append(f"- {rec}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating Markdown report: {e}"


class ReproductionEngine:
    """
    Engine for capturing and managing experiment reproduction metadata.
    
    This class provides comprehensive metadata capture capabilities for ensuring
    experiments can be reproduced exactly, including environment state, random seeds,
    configuration snapshots, and system information.
    
    Args:
        experiment_id: ID of existing experiment to capture metadata for
        experiment_path: Path to experiment directory (alternative to experiment_id)
        config: Configuration dictionary for the reproduction engine
        
    Example:
        # Initialize with existing experiment
        engine = ReproductionEngine(experiment_id=123)
        
        # Capture comprehensive metadata
        metadata = engine.capture_reproduction_metadata()
        
        # Save metadata for later reproduction
        engine.save_reproduction_metadata(metadata, "experiment_123_reproduction.json")
    """
    
    def __init__(self, 
                 experiment_id: Optional[int] = None,
                 experiment_path: Optional[Union[str, Path]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ReproductionEngine.
        
        Args:
            experiment_id: ID of existing experiment in tracking database
            experiment_path: Path to experiment directory (if not using experiment_id)
            config: Configuration options for the engine
        """
        self.experiment_id = experiment_id
        self.experiment_path = Path(experiment_path) if experiment_path else None
        self.config = config or {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        log_level = self.config.get('log_level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize experiment tracker if working with tracked experiments
        self.tracker = None
        if experiment_id is not None:
            try:
                self.tracker = ExperimentTracker.get_instance()
                # Verify experiment exists
                experiment = self.tracker.get_experiment(experiment_id)
                if not experiment:
                    raise ReproductionEngineError(f"Experiment {experiment_id} not found")
                self.logger.info(f"Initialized ReproductionEngine for experiment {experiment_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize tracker for experiment {experiment_id}: {e}")
                raise ReproductionEngineError(f"Failed to initialize for experiment {experiment_id}: {e}")
        
        # Threading lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Metadata cache
        self._metadata_cache = {}
        
        # Initialize validation framework
        validation_config = self.config.get('validation', {})
        self.validation_framework = ValidationFramework(validation_config)
        
    def capture_reproduction_metadata(self) -> Dict[str, Any]:
        """
        Capture comprehensive metadata needed for experiment reproduction.
        
        Returns:
            Dictionary containing all metadata required for reproduction
            
        Raises:
            ReproductionMetadataCaptureError: If metadata capture fails
        """
        with self._lock:
            try:
                self.logger.info("Starting comprehensive metadata capture")
                
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'reproduction_engine_version': '0.1.0',
                    'experiment_id': self.experiment_id,
                    'experiment_path': str(self.experiment_path) if self.experiment_path else None
                }
                
                # Capture basic system information
                metadata['system'] = self._capture_system_metadata()
                
                # Capture enhanced environment information including conda/pip details
                metadata['environment'] = self.capture_enhanced_environment_metadata()
                
                # Capture random seed states
                metadata['random_seeds'] = self._capture_random_seed_states()
                
                # Capture git metadata if available
                metadata['git'] = self._capture_git_metadata()
                
                # Capture experiment-specific metadata if we have a tracker
                if self.tracker and self.experiment_id:
                    metadata['experiment'] = self._capture_experiment_metadata()
                
                # Capture configuration snapshot
                metadata['configuration'] = self._capture_configuration_metadata()
                
                # Cache the metadata
                self._metadata_cache = metadata.copy()
                
                self.logger.info(f"Successfully captured metadata with {len(metadata)} top-level categories")
                return metadata
                
            except Exception as e:
                self.logger.error(f"Failed to capture reproduction metadata: {e}")
                raise ReproductionMetadataCaptureError(f"Metadata capture failed: {e}")
    
    def _capture_system_metadata(self) -> Dict[str, Any]:
        """Capture system-level metadata for reproduction."""
        try:
            # Leverage existing tracker method if available
            if self.tracker:
                return self.tracker._capture_system_metadata()
            
            # Fallback to our own implementation
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_executable': sys.executable,
                'hostname': platform.node(),
                'cpu_count': os.cpu_count()
            }
        except Exception as e:
            self.logger.warning(f"Failed to capture system metadata: {e}")
            return {'error': str(e)}
    
    def _capture_environment_metadata(self) -> Dict[str, Any]:
        """Capture environment-level metadata for reproduction."""
        try:
            # Leverage existing tracker method if available
            if self.tracker:
                return self.tracker._capture_environment_metadata()
            
            # Fallback to basic environment capture
            import pkg_resources
            
            packages = {}
            for package in pkg_resources.working_set:
                packages[package.key] = package.version
            
            return {
                'python_path': sys.path,
                'environment_variables': dict(os.environ),
                'installed_packages': packages,
                'working_directory': os.getcwd()
            }
        except Exception as e:
            self.logger.warning(f"Failed to capture environment metadata: {e}")
            return {'error': str(e)}
    
    def _capture_random_seed_states(self) -> Dict[str, Any]:
        """
        Capture current random seed states from common libraries.
        
        Returns:
            Dictionary containing random states from Python, NumPy, and PyTorch
        """
        try:
            seed_states = {
                'capture_timestamp': datetime.now().isoformat()
            }
            
            # Python's built-in random module
            try:
                seed_states['python_random'] = {
                    'state': random.getstate(),
                    'available': True
                }
            except Exception as e:
                seed_states['python_random'] = {
                    'error': str(e),
                    'available': False
                }
            
            # NumPy random state
            if NUMPY_AVAILABLE:
                try:
                    # Handle both legacy and new numpy random APIs
                    if hasattr(np.random, 'get_state'):
                        # Legacy API
                        seed_states['numpy_random'] = {
                            'state': np.random.get_state(),
                            'api': 'legacy',
                            'available': True
                        }
                    else:
                        # New Generator API (numpy >= 1.17)
                        default_rng = np.random.default_rng()
                        seed_states['numpy_random'] = {
                            'bit_generator': str(default_rng.bit_generator.state),
                            'api': 'generator',
                            'available': True
                        }
                except Exception as e:
                    seed_states['numpy_random'] = {
                        'error': str(e),
                        'available': True,
                        'import_successful': True
                    }
            else:
                seed_states['numpy_random'] = {
                    'available': False,
                    'reason': 'numpy not installed'
                }
            
            # PyTorch random state
            if TORCH_AVAILABLE:
                try:
                    seed_states['torch_random'] = {
                        'state': torch.get_rng_state(),
                        'cuda_available': torch.cuda.is_available(),
                        'available': True
                    }
                    
                    # Capture CUDA random states if available
                    if torch.cuda.is_available():
                        seed_states['torch_random']['cuda_states'] = []
                        for i in range(torch.cuda.device_count()):
                            try:
                                with torch.cuda.device(i):
                                    cuda_state = torch.cuda.get_rng_state()
                                    seed_states['torch_random']['cuda_states'].append({
                                        'device': i,
                                        'state': cuda_state
                                    })
                            except Exception as e:
                                seed_states['torch_random']['cuda_states'].append({
                                    'device': i,
                                    'error': str(e)
                                })
                                
                except Exception as e:
                    seed_states['torch_random'] = {
                        'error': str(e),
                        'available': True,
                        'import_successful': True
                    }
            else:
                seed_states['torch_random'] = {
                    'available': False,
                    'reason': 'torch not installed'
                }
            
            self.logger.debug(f"Captured random seed states: {list(seed_states.keys())}")
            return seed_states
            
        except Exception as e:
            self.logger.error(f"Failed to capture random seed states: {e}")
            return {
                'error': str(e),
                'capture_timestamp': datetime.now().isoformat()
            }
    
    def _capture_git_metadata(self) -> Dict[str, Any]:
        """Capture git metadata for reproduction."""
        try:
            # Leverage existing tracker method if available
            if self.tracker:
                return self.tracker._capture_git_metadata()
            
            # Fallback to basic git capture
            import subprocess
            
            git_metadata = {}
            try:
                # Get current commit hash
                result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                      capture_output=True, text=True, check=True)
                git_metadata['commit_hash'] = result.stdout.strip()
                
                # Get current branch
                result = subprocess.run(['git', 'branch', '--show-current'], 
                                      capture_output=True, text=True, check=True)
                git_metadata['branch'] = result.stdout.strip()
                
                # Check for uncommitted changes
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, check=True)
                git_metadata['has_uncommitted_changes'] = bool(result.stdout.strip())
                git_metadata['uncommitted_files'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
            except subprocess.CalledProcessError:
                git_metadata['available'] = False
                git_metadata['reason'] = 'Not a git repository or git not available'
            
            return git_metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to capture git metadata: {e}")
            return {'error': str(e)}
    
    def _capture_experiment_metadata(self) -> Dict[str, Any]:
        """Capture experiment-specific metadata from the tracker."""
        try:
            if not self.tracker or not self.experiment_id:
                return {'available': False, 'reason': 'No tracker or experiment_id'}
            
            experiment = self.tracker.get_experiment(self.experiment_id)
            if not experiment:
                return {'available': False, 'reason': f'Experiment {self.experiment_id} not found'}
            
            # Get experiment configuration
            config = json.loads(experiment.get('configuration', '{}'))
            
            return {
                'experiment_id': self.experiment_id,
                'name': experiment.get('name'),
                'description': experiment.get('description'),
                'status': experiment.get('status'),
                'created_at': experiment.get('created_at'),
                'updated_at': experiment.get('updated_at'),
                'configuration': config,
                'tags': experiment.get('tags', [])
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to capture experiment metadata: {e}")
            return {'error': str(e)}
    
    def _capture_configuration_metadata(self) -> Dict[str, Any]:
        """Capture configuration metadata for reproduction."""
        try:
            config_metadata = {
                'reproduction_engine_config': self.config.copy(),
                'capture_timestamp': datetime.now().isoformat()
            }
            
            # If we have an experiment path, look for common config files
            if self.experiment_path and self.experiment_path.exists():
                config_files = []
                common_config_patterns = [
                    '*.json', '*.yaml', '*.yml', '*.toml', '*.ini', 
                    'config.*', '*.config', 'settings.*', '*.settings'
                ]
                
                for pattern in common_config_patterns:
                    config_files.extend(self.experiment_path.glob(pattern))
                
                # Read and store configuration files
                captured_configs = {}
                for config_file in config_files[:10]:  # Limit to first 10 files
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            captured_configs[str(config_file.name)] = {
                                'content': content,
                                'size': len(content),
                                'modified_time': config_file.stat().st_mtime
                            }
                    except Exception as e:
                        captured_configs[str(config_file.name)] = {
                            'error': str(e)
                        }
                
                config_metadata['config_files'] = captured_configs
            
            return config_metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to capture configuration metadata: {e}")
            return {'error': str(e)}
    
    def save_reproduction_metadata(self, 
                                 metadata: Dict[str, Any], 
                                 file_path: Union[str, Path],
                                 format: str = 'json') -> Path:
        """
        Save reproduction metadata to a file.
        
        Args:
            metadata: Metadata dictionary to save
            file_path: Path where to save the metadata
            format: Format to save in ('json' or 'yaml')
            
        Returns:
            Path object of the saved file
            
        Raises:
            ReproductionStorageError: If saving fails
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str)
            elif format.lower() in ['yaml', 'yml']:
                try:
                    import yaml
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(metadata, f, default_flow_style=False, default=str)
                except ImportError:
                    self.logger.warning("PyYAML not available, falling back to JSON")
                    file_path = file_path.with_suffix('.json')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved reproduction metadata to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save reproduction metadata: {e}")
            raise ReproductionStorageError(f"Failed to save metadata: {e}")
    
    def load_reproduction_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load reproduction metadata from a file.
        
        Args:
            file_path: Path to the metadata file
            
        Returns:
            Loaded metadata dictionary
            
        Raises:
            ReproductionStorageError: If loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {file_path}")
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = yaml.safe_load(f)
                except ImportError:
                    raise ReproductionStorageError("PyYAML required to load YAML files")
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"Loaded reproduction metadata from {file_path}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load reproduction metadata: {e}")
            raise ReproductionStorageError(f"Failed to load metadata: {e}")
    
    def get_cached_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get cached metadata if available.
        
        Returns:
            Cached metadata dictionary or None if not available
        """
        return self._metadata_cache.copy() if self._metadata_cache else None
    
    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        with self._lock:
            self._metadata_cache.clear()
            self.logger.debug("Cleared metadata cache")
    
    # ================================================================
    # Environment Recreation Methods
    # ================================================================
    
    def capture_enhanced_environment_metadata(self) -> Dict[str, Any]:
        """
        Capture enhanced environment metadata including conda/pip details.
        
        Returns:
            Enhanced environment metadata with package manager details
        """
        try:
            # Start with basic environment metadata
            env_metadata = self._capture_environment_metadata()
            
            # Add conda environment details if available
            conda_info = self._capture_conda_environment()
            if conda_info:
                env_metadata['conda'] = conda_info
            
            # Add pip environment details
            pip_info = self._capture_pip_environment()
            if pip_info:
                env_metadata['pip'] = pip_info
            
            # Detect primary package manager
            env_metadata['primary_package_manager'] = self._detect_primary_package_manager()
            
            return env_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to capture enhanced environment metadata: {e}")
            return {'error': str(e)}
    
    def _capture_conda_environment(self) -> Optional[Dict[str, Any]]:
        """Capture conda environment information."""
        try:
            import subprocess
            
            conda_info = {}
            
            # Check if conda is available
            try:
                result = subprocess.run(['conda', '--version'], 
                                      capture_output=True, text=True, check=True)
                conda_info['conda_version'] = result.stdout.strip()
                conda_info['available'] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None
            
            # Get current environment name
            try:
                result = subprocess.run(['conda', 'info', '--json'], 
                                      capture_output=True, text=True, check=True)
                conda_data = json.loads(result.stdout)
                conda_info['active_environment'] = conda_data.get('active_prefix_name', 'base')
                conda_info['environment_path'] = conda_data.get('active_prefix', '')
                conda_info['conda_prefix'] = conda_data.get('conda_prefix', '')
            except Exception as e:
                self.logger.warning(f"Failed to get conda environment info: {e}")
            
            # Get environment packages
            try:
                result = subprocess.run(['conda', 'list', '--json'], 
                                      capture_output=True, text=True, check=True)
                packages_data = json.loads(result.stdout)
                
                conda_packages = {}
                for pkg in packages_data:
                    name = pkg.get('name', '')
                    version = pkg.get('version', '')
                    build = pkg.get('build_string', '')
                    channel = pkg.get('channel', '')
                    
                    conda_packages[name] = {
                        'version': version,
                        'build': build,
                        'channel': channel
                    }
                
                conda_info['packages'] = conda_packages
                conda_info['package_count'] = len(conda_packages)
                
            except Exception as e:
                self.logger.warning(f"Failed to get conda packages: {e}")
                conda_info['packages'] = {}
            
            # Get environment export
            try:
                result = subprocess.run(['conda', 'env', 'export', '--no-builds'], 
                                      capture_output=True, text=True, check=True)
                conda_info['environment_yml'] = result.stdout
            except Exception as e:
                self.logger.warning(f"Failed to export conda environment: {e}")
            
            return conda_info
            
        except Exception as e:
            self.logger.warning(f"Failed to capture conda environment: {e}")
            return None
    
    def _capture_pip_environment(self) -> Dict[str, Any]:
        """Capture pip environment information."""
        try:
            import subprocess
            
            pip_info = {}
            
            # Get pip version
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                      capture_output=True, text=True, check=True)
                pip_info['pip_version'] = result.stdout.strip()
                pip_info['available'] = True
            except Exception as e:
                pip_info['available'] = False
                pip_info['error'] = str(e)
                return pip_info
            
            # Get installed packages with pip freeze
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                      capture_output=True, text=True, check=True)
                
                pip_packages = {}
                for line in result.stdout.strip().split('\n'):
                    if line and '==' in line:
                        name, version = line.split('==', 1)
                        pip_packages[name] = version
                
                pip_info['packages'] = pip_packages
                pip_info['package_count'] = len(pip_packages)
                pip_info['requirements_txt'] = result.stdout
                
            except Exception as e:
                self.logger.warning(f"Failed to get pip packages: {e}")
                pip_info['packages'] = {}
            
            # Get pip list with additional details
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                                      capture_output=True, text=True, check=True)
                packages_data = json.loads(result.stdout)
                
                detailed_packages = {}
                for pkg in packages_data:
                    name = pkg.get('name', '')
                    version = pkg.get('version', '')
                    detailed_packages[name] = {
                        'version': version,
                        'editable': pkg.get('editable_project_location') is not None
                    }
                
                pip_info['detailed_packages'] = detailed_packages
                
            except Exception as e:
                self.logger.warning(f"Failed to get detailed pip packages: {e}")
            
            return pip_info
            
        except Exception as e:
            self.logger.warning(f"Failed to capture pip environment: {e}")
            return {'available': False, 'error': str(e)}
    
    def _detect_primary_package_manager(self) -> str:
        """Detect the primary package manager being used."""
        try:
            # Check for conda environment
            conda_env = os.environ.get('CONDA_DEFAULT_ENV')
            if conda_env:
                return 'conda'
            
            # Check for virtual environment
            virtual_env = os.environ.get('VIRTUAL_ENV')
            if virtual_env:
                return 'pip'
            
            # Check if we're in a conda environment by looking at the Python path
            if 'conda' in sys.executable or 'anaconda' in sys.executable:
                return 'conda'
            
            # Default to pip
            return 'pip'
            
        except Exception:
            return 'pip'
    
    def export_environment_files(self, 
                                output_dir: Union[str, Path],
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Export environment files (requirements.txt, environment.yml) from metadata.
        
        Args:
            output_dir: Directory to save environment files
            metadata: Metadata dictionary (uses cached if not provided)
            
        Returns:
            Dictionary mapping file types to their paths
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if metadata is None:
                metadata = self.get_cached_metadata()
                if not metadata:
                    raise ValueError("No metadata available. Call capture_reproduction_metadata() first.")
            
            exported_files = {}
            
            # Export requirements.txt
            requirements_path = self._export_requirements_txt(output_dir, metadata)
            if requirements_path:
                exported_files['requirements.txt'] = requirements_path
            
            # Export environment.yml
            environment_yml_path = self._export_environment_yml(output_dir, metadata)
            if environment_yml_path:
                exported_files['environment.yml'] = environment_yml_path
            
            self.logger.info(f"Exported {len(exported_files)} environment files to {output_dir}")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export environment files: {e}")
            raise ReproductionEngineError(f"Environment export failed: {e}")
    
    def _export_requirements_txt(self, output_dir: Path, metadata: Dict[str, Any]) -> Optional[Path]:
        """Export requirements.txt file from metadata."""
        try:
            env_data = metadata.get('environment', {})
            
            # Try to get pip requirements from metadata
            pip_data = env_data.get('pip', {})
            if pip_data.get('requirements_txt'):
                requirements_path = output_dir / 'requirements.txt'
                with open(requirements_path, 'w', encoding='utf-8') as f:
                    f.write("# Generated requirements.txt for experiment reproduction\n")
                    f.write(f"# Generated on: {datetime.now().isoformat()}\n\n")
                    f.write(pip_data['requirements_txt'])
                
                self.logger.info(f"Exported requirements.txt to {requirements_path}")
                return requirements_path
            
            # Fallback: generate from installed packages
            packages = env_data.get('installed_packages', {})
            if packages:
                requirements_path = output_dir / 'requirements.txt'
                with open(requirements_path, 'w', encoding='utf-8') as f:
                    f.write("# Generated requirements.txt for experiment reproduction\n")
                    f.write(f"# Generated on: {datetime.now().isoformat()}\n\n")
                    
                    for package, version in sorted(packages.items()):
                        f.write(f"{package}=={version}\n")
                
                self.logger.info(f"Exported requirements.txt to {requirements_path}")
                return requirements_path
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to export requirements.txt: {e}")
            return None
    
    def _export_environment_yml(self, output_dir: Path, metadata: Dict[str, Any]) -> Optional[Path]:
        """Export environment.yml file from metadata."""
        try:
            env_data = metadata.get('environment', {})
            conda_data = env_data.get('conda', {})
            
            if not conda_data or not conda_data.get('available'):
                return None
            
            environment_yml_path = output_dir / 'environment.yml'
            
            # Use existing environment.yml if available
            if conda_data.get('environment_yml'):
                with open(environment_yml_path, 'w', encoding='utf-8') as f:
                    f.write(conda_data['environment_yml'])
                
                self.logger.info(f"Exported environment.yml to {environment_yml_path}")
                return environment_yml_path
            
            # Generate environment.yml from conda packages
            conda_packages = conda_data.get('packages', {})
            if conda_packages:
                with open(environment_yml_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Generated environment.yml for experiment reproduction\n")
                    f.write(f"# Generated on: {datetime.now().isoformat()}\n")
                    f.write(f"name: reproduced-experiment\n")
                    f.write("channels:\n")
                    f.write("  - defaults\n")
                    f.write("  - conda-forge\n")
                    f.write("dependencies:\n")
                    
                    for package, details in sorted(conda_packages.items()):
                        version = details.get('version', '')
                        if version:
                            f.write(f"  - {package}={version}\n")
                
                self.logger.info(f"Exported environment.yml to {environment_yml_path}")
                return environment_yml_path
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to export environment.yml: {e}")
            return None
    
    def recreate_environment(self, 
                           metadata: Dict[str, Any],
                           environment_name: str = "reproduced-experiment",
                           force: bool = False) -> Dict[str, Any]:
        """
        Recreate environment from metadata.
        
        Args:
            metadata: Reproduction metadata containing environment info
            environment_name: Name for the new environment
            force: Whether to overwrite existing environment
            
        Returns:
            Dictionary with recreation results and status
        """
        try:
            env_data = metadata.get('environment', {})
            primary_manager = env_data.get('primary_package_manager', 'pip')
            
            recreation_result = {
                'environment_name': environment_name,
                'primary_manager': primary_manager,
                'success': False,
                'conflicts': [],
                'warnings': [],
                'created_files': []
            }
            
            if primary_manager == 'conda':
                result = self._recreate_conda_environment(env_data, environment_name, force)
            else:
                result = self._recreate_pip_environment(env_data, environment_name, force)
            
            recreation_result.update(result)
            
            # Validate the recreated environment
            validation_result = self._validate_recreated_environment(metadata, environment_name)
            recreation_result['validation'] = validation_result
            
            return recreation_result
            
        except Exception as e:
            self.logger.error(f"Failed to recreate environment: {e}")
            return {
                'success': False,
                'error': str(e),
                'environment_name': environment_name
            }
    
    def _recreate_conda_environment(self, env_data: Dict[str, Any], 
                                  environment_name: str, force: bool) -> Dict[str, Any]:
        """Recreate conda environment from metadata."""
        try:
            import subprocess
            
            result = {
                'method': 'conda',
                'success': False,
                'conflicts': [],
                'warnings': []
            }
            
            conda_data = env_data.get('conda', {})
            if not conda_data or not conda_data.get('available'):
                result['error'] = 'Conda not available in original environment'
                return result
            
            # Create temporary environment.yml file
            temp_yml = Path.cwd() / f"temp_{environment_name}.yml"
            
            try:
                with open(temp_yml, 'w') as f:
                    f.write(f"name: {environment_name}\n")
                    f.write("channels:\n")
                    f.write("  - defaults\n")
                    f.write("  - conda-forge\n")
                    f.write("dependencies:\n")
                    
                    packages = conda_data.get('packages', {})
                    for package, details in packages.items():
                        version = details.get('version', '')
                        if version:
                            f.write(f"  - {package}={version}\n")
                
                # Create conda environment
                cmd = ['conda', 'env', 'create', '-f', str(temp_yml)]
                if force:
                    # Remove existing environment first
                    try:
                        subprocess.run(['conda', 'env', 'remove', '-n', environment_name], 
                                     capture_output=True, check=False)
                    except:
                        pass
                
                process_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if process_result.returncode == 0:
                    result['success'] = True
                    self.logger.info(f"Successfully created conda environment: {environment_name}")
                else:
                    result['error'] = process_result.stderr
                    self.logger.error(f"Failed to create conda environment: {process_result.stderr}")
                
            finally:
                # Clean up temporary file
                if temp_yml.exists():
                    temp_yml.unlink()
            
            return result
            
        except Exception as e:
            return {
                'method': 'conda',
                'success': False,
                'error': str(e)
            }
    
    def _recreate_pip_environment(self, env_data: Dict[str, Any], 
                                environment_name: str, force: bool) -> Dict[str, Any]:
        """Recreate pip environment from metadata."""
        try:
            import subprocess
            
            result = {
                'method': 'pip',
                'success': False,
                'conflicts': [],
                'warnings': []
            }
            
            # Create virtual environment
            venv_path = Path.cwd() / environment_name
            
            if venv_path.exists() and force:
                import shutil
                shutil.rmtree(venv_path)
            elif venv_path.exists():
                result['error'] = f"Environment {environment_name} already exists. Use force=True to overwrite."
                return result
            
            # Create virtual environment
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            
            # Determine pip executable in the new environment
            if sys.platform == 'win32':
                pip_executable = venv_path / 'Scripts' / 'pip.exe'
            else:
                pip_executable = venv_path / 'bin' / 'pip'
            
            # Install packages
            pip_data = env_data.get('pip', {})
            packages = pip_data.get('packages', {})
            
            if packages:
                # Create requirements list
                requirements = [f"{pkg}=={version}" for pkg, version in packages.items()]
                
                # Install packages
                cmd = [str(pip_executable), 'install'] + requirements
                process_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if process_result.returncode == 0:
                    result['success'] = True
                    self.logger.info(f"Successfully created pip environment: {environment_name}")
                else:
                    result['error'] = process_result.stderr
                    result['conflicts'] = self._parse_pip_conflicts(process_result.stderr)
                    self.logger.error(f"Failed to create pip environment: {process_result.stderr}")
            else:
                result['success'] = True
                result['warnings'].append("No packages to install")
            
            return result
            
        except Exception as e:
            return {
                'method': 'pip',
                'success': False,
                'error': str(e)
            }
    
    def _parse_pip_conflicts(self, error_output: str) -> List[str]:
        """Parse pip error output to identify conflicts."""
        conflicts = []
        lines = error_output.split('\n')
        
        for line in lines:
            if 'conflict' in line.lower() or 'incompatible' in line.lower():
                conflicts.append(line.strip())
        
        return conflicts
    
    def _validate_recreated_environment(self, metadata: Dict[str, Any], 
                                      environment_name: str) -> Dict[str, Any]:
        """Validate that the recreated environment matches the original."""
        try:
            validation_result = {
                'success': False,
                'package_matches': 0,
                'package_mismatches': 0,
                'missing_packages': [],
                'version_mismatches': []
            }
            
            # This is a placeholder for validation logic
            # In a real implementation, you would:
            # 1. Activate the recreated environment
            # 2. List installed packages
            # 3. Compare with original metadata
            # 4. Report differences
            
            validation_result['success'] = True
            validation_result['note'] = "Validation not fully implemented"
            
            return validation_result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # ================================================================
    # Configuration Management Methods
    # ================================================================
    
    def capture_advanced_configuration_metadata(self) -> Dict[str, Any]:
        """
        Capture advanced configuration metadata with parsing and validation.
        
        Returns:
            Advanced configuration metadata with parsed configurations
        """
        try:
            # Start with basic configuration metadata
            config_metadata = self._capture_configuration_metadata()
            
            # Parse configuration files
            parsed_configs = {}
            if 'config_files' in config_metadata:
                for filename, file_data in config_metadata['config_files'].items():
                    if 'content' in file_data:
                        parsed_config = self._parse_configuration_file(filename, file_data['content'])
                        if parsed_config:
                            parsed_configs[filename] = parsed_config
            
            config_metadata['parsed_configurations'] = parsed_configs
            
            # Add configuration schema detection
            config_metadata['configuration_schema'] = self._detect_configuration_schema(parsed_configs)
            
            return config_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to capture advanced configuration metadata: {e}")
            return {'error': str(e)}
    
    def _parse_configuration_file(self, filename: str, content: str) -> Optional[Dict[str, Any]]:
        """Parse a configuration file based on its format."""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.json':
                return json.loads(content)
            elif file_ext in ['.yaml', '.yml']:
                try:
                    import yaml
                    return yaml.safe_load(content)
                except ImportError:
                    self.logger.warning(f"PyYAML not available, skipping {filename}")
                    return None
            elif file_ext == '.toml':
                try:
                    import toml
                    return toml.loads(content)
                except ImportError:
                    self.logger.warning(f"toml library not available, skipping {filename}")
                    return None
            elif file_ext == '.ini':
                try:
                    import configparser
                    config = configparser.ConfigParser()
                    config.read_string(content)
                    return {section: dict(config[section]) for section in config.sections()}
                except Exception as e:
                    self.logger.warning(f"Failed to parse INI file {filename}: {e}")
                    return None
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
            self.logger.warning(f"Failed to parse configuration file {filename}: {e}")
            return None
    
    def _detect_configuration_schema(self, parsed_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Detect configuration schema and common patterns."""
        schema_info = {
            'total_configs': len(parsed_configs),
            'config_types': {},
            'common_keys': set(),
            'nested_structures': []
        }
        
        for filename, config in parsed_configs.items():
            if isinstance(config, dict):
                # Analyze structure
                schema_info['config_types'][filename] = 'dict'
                schema_info['common_keys'].update(config.keys())
                
                # Check for nested structures
                for key, value in config.items():
                    if isinstance(value, dict):
                        schema_info['nested_structures'].append(f"{filename}:{key}")
            else:
                schema_info['config_types'][filename] = type(config).__name__
        
        # Convert set to list for serialization
        schema_info['common_keys'] = list(schema_info['common_keys'])
        
        return schema_info
    
    def restore_configuration(self, metadata: Dict[str, Any], 
                            target_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Restore configuration files from metadata.
        
        Args:
            metadata: Reproduction metadata containing configuration
            target_dir: Directory to restore configuration files to
            
        Returns:
            Dictionary with restoration results
        """
        try:
            if target_dir is None:
                target_dir = Path.cwd() / "restored_config"
            else:
                target_dir = Path(target_dir)
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            restoration_result = {
                'target_directory': str(target_dir),
                'restored_files': [],
                'skipped_files': [],
                'errors': []
            }
            
            config_data = metadata.get('configuration', {})
            config_files = config_data.get('config_files', {})
            
            for filename, file_data in config_files.items():
                try:
                    if 'content' in file_data:
                        target_file = target_dir / filename
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write(file_data['content'])
                        restoration_result['restored_files'].append(str(target_file))
                    else:
                        restoration_result['skipped_files'].append(filename)
                        
                except Exception as e:
                    error_msg = f"Failed to restore {filename}: {e}"
                    restoration_result['errors'].append(error_msg)
                    self.logger.warning(error_msg)
            
            self.logger.info(f"Configuration restoration completed. "
                           f"Restored: {len(restoration_result['restored_files'])}, "
                           f"Skipped: {len(restoration_result['skipped_files'])}, "
                           f"Errors: {len(restoration_result['errors'])}")
            
            return restoration_result
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_configuration(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration metadata for completeness and consistency.
        
        Args:
            metadata: Reproduction metadata to validate
            
        Returns:
            Validation results
        """
        try:
            validation_result = {
                'overall_valid': True,
                'configuration_present': False,
                'parsed_configs_valid': True,
                'schema_consistent': True,
                'warnings': [],
                'errors': []
            }
            
            config_data = metadata.get('configuration', {})
            
            if not config_data:
                validation_result['overall_valid'] = False
                validation_result['errors'].append("No configuration metadata found")
                return validation_result
            
            validation_result['configuration_present'] = True
            
            # Validate parsed configurations
            parsed_configs = config_data.get('parsed_configurations', {})
            for filename, config in parsed_configs.items():
                if config is None:
                    validation_result['warnings'].append(f"Configuration {filename} could not be parsed")
                elif not isinstance(config, (dict, list, str, int, float, bool)):
                    validation_result['parsed_configs_valid'] = False
                    validation_result['errors'].append(f"Configuration {filename} has invalid structure")
            
            # Check for required configuration patterns
            common_required_configs = ['config.json', 'config.yaml', 'config.yml', 'settings.json']
            found_configs = set(config_data.get('config_files', {}).keys())
            
            if not any(req_config in found_configs for req_config in common_required_configs):
                validation_result['warnings'].append("No standard configuration files found")
            
            validation_result['overall_valid'] = (
                validation_result['configuration_present'] and 
                validation_result['parsed_configs_valid'] and 
                len(validation_result['errors']) == 0
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate configuration: {e}")
            return {
                'overall_valid': False,
                'error': str(e)
            }
    
    # ================================================================
    # Checkpoint Management Methods
    # ================================================================
    
    def capture_checkpoint_metadata(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Capture checkpoint metadata for model states and training progress.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            Checkpoint metadata
        """
        try:
            if checkpoint_dir is None:
                checkpoint_dir = Path.cwd() / "checkpoints"
            else:
                checkpoint_dir = Path(checkpoint_dir)
            
            checkpoint_metadata = {
                'checkpoint_directory': str(checkpoint_dir),
                'capture_timestamp': datetime.now().isoformat(),
                'checkpoints_found': [],
                'framework_detection': {}
            }
            
            if checkpoint_dir.exists():
                # Find common checkpoint patterns
                checkpoint_patterns = [
                    '*.pth', '*.pt',  # PyTorch
                    '*.ckpt',         # General checkpoints
                    '*.h5', '*.hdf5', # Keras/TensorFlow
                    '*.pkl', '*.pickle', # Generic Python pickles
                    '*.safetensors',  # SafeTensors format
                    'model.*', 'checkpoint.*'
                ]
                
                found_checkpoints = []
                for pattern in checkpoint_patterns:
                    found_checkpoints.extend(checkpoint_dir.glob(pattern))
                    # Also search recursively one level down
                    found_checkpoints.extend(checkpoint_dir.glob(f"*/{pattern}"))
                
                # Analyze each checkpoint
                for checkpoint_file in found_checkpoints[:20]:  # Limit to 20 files
                    checkpoint_info = self._analyze_checkpoint_file(checkpoint_file)
                    checkpoint_metadata['checkpoints_found'].append(checkpoint_info)
                
                # Framework detection
                checkpoint_metadata['framework_detection'] = self._detect_ml_frameworks(found_checkpoints)
            
            return checkpoint_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to capture checkpoint metadata: {e}")
            return {'error': str(e)}
    
    def _analyze_checkpoint_file(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Analyze a single checkpoint file."""
        try:
            checkpoint_info = {
                'filename': checkpoint_path.name,
                'full_path': str(checkpoint_path),
                'size_bytes': checkpoint_path.stat().st_size,
                'modified_time': checkpoint_path.stat().st_mtime,
                'extension': checkpoint_path.suffix.lower(),
                'framework': 'unknown',
                'contains': []
            }
            
            # Try to determine framework and contents
            if checkpoint_path.suffix.lower() in ['.pth', '.pt']:
                checkpoint_info['framework'] = 'pytorch'
                try:
                    if TORCH_AVAILABLE:
                        # Try to load checkpoint metadata without loading the full model
                        checkpoint_keys = self._get_pytorch_checkpoint_keys(checkpoint_path)
                        checkpoint_info['contains'] = checkpoint_keys
                except Exception as e:
                    checkpoint_info['load_error'] = str(e)
                    
            elif checkpoint_path.suffix.lower() in ['.h5', '.hdf5']:
                checkpoint_info['framework'] = 'tensorflow_keras'
                
            elif checkpoint_path.suffix.lower() == '.ckpt':
                # Could be TensorFlow or other frameworks
                checkpoint_info['framework'] = 'tensorflow_or_other'
                
            elif checkpoint_path.suffix.lower() == '.safetensors':
                checkpoint_info['framework'] = 'safetensors'
                
            return checkpoint_info
            
        except Exception as e:
            return {
                'filename': checkpoint_path.name,
                'error': str(e)
            }
    
    def _get_pytorch_checkpoint_keys(self, checkpoint_path: Path) -> List[str]:
        """Get keys from a PyTorch checkpoint without loading the full tensors."""
        try:
            import torch
            # Load only the keys, not the tensor data
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                return list(checkpoint.keys())
            else:
                return ['model_state_dict']  # Assume it's a direct state dict
        except Exception:
            return []
    
    def _detect_ml_frameworks(self, checkpoint_files: List[Path]) -> Dict[str, Any]:
        """Detect which ML frameworks are being used based on checkpoint files."""
        framework_info = {
            'pytorch': False,
            'tensorflow': False,
            'keras': False,
            'safetensors': False,
            'generic': False
        }
        
        for checkpoint_file in checkpoint_files:
            ext = checkpoint_file.suffix.lower()
            if ext in ['.pth', '.pt']:
                framework_info['pytorch'] = True
            elif ext in ['.h5', '.hdf5']:
                framework_info['keras'] = True
                framework_info['tensorflow'] = True
            elif ext == '.ckpt':
                framework_info['tensorflow'] = True
            elif ext == '.safetensors':
                framework_info['safetensors'] = True
            elif ext in ['.pkl', '.pickle']:
                framework_info['generic'] = True
        
        return framework_info
    
    def restore_random_seeds(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore random seed states from metadata to ensure deterministic behavior.
        
        Args:
            metadata: Reproduction metadata containing random seed states
            
        Returns:
            Restoration results
        """
        try:
            restoration_result = {
                'seeds_restored': [],
                'seeds_failed': [],
                'warnings': []
            }
            
            random_seeds = metadata.get('random_seeds', {})
            
            if not random_seeds:
                restoration_result['warnings'].append("No random seed data found in metadata")
                return restoration_result
            
            # Restore Python random state
            if 'python_random' in random_seeds and random_seeds['python_random'].get('available'):
                try:
                    state = random_seeds['python_random']['state']
                    random.setstate(state)
                    restoration_result['seeds_restored'].append('python_random')
                    self.logger.info("Restored Python random state")
                except Exception as e:
                    restoration_result['seeds_failed'].append(f"python_random: {e}")
            
            # Restore NumPy random state
            if NUMPY_AVAILABLE and 'numpy_random' in random_seeds and random_seeds['numpy_random'].get('available'):
                try:
                    numpy_data = random_seeds['numpy_random']
                    if numpy_data.get('api') == 'legacy' and 'state' in numpy_data:
                        np.random.set_state(numpy_data['state'])
                        restoration_result['seeds_restored'].append('numpy_random_legacy')
                        self.logger.info("Restored NumPy random state (legacy API)")
                    elif numpy_data.get('api') == 'generator':
                        # For new Generator API, this is more complex
                        restoration_result['warnings'].append("NumPy Generator API state restoration not fully implemented")
                except Exception as e:
                    restoration_result['seeds_failed'].append(f"numpy_random: {e}")
            
            # Restore PyTorch random state
            if TORCH_AVAILABLE and 'torch_random' in random_seeds and random_seeds['torch_random'].get('available'):
                try:
                    torch_data = random_seeds['torch_random']
                    
                    # Restore CPU state
                    if 'state' in torch_data:
                        torch.set_rng_state(torch_data['state'])
                        restoration_result['seeds_restored'].append('torch_cpu')
                        self.logger.info("Restored PyTorch CPU random state")
                    
                    # Restore CUDA states
                    if torch_data.get('cuda_available') and 'cuda_states' in torch_data:
                        for cuda_state_info in torch_data['cuda_states']:
                            try:
                                device = cuda_state_info['device']
                                if 'state' in cuda_state_info and torch.cuda.is_available():
                                    with torch.cuda.device(device):
                                        torch.cuda.set_rng_state(cuda_state_info['state'])
                                    restoration_result['seeds_restored'].append(f'torch_cuda_{device}')
                            except Exception as e:
                                restoration_result['seeds_failed'].append(f"torch_cuda_{device}: {e}")
                                
                except Exception as e:
                    restoration_result['seeds_failed'].append(f"torch_random: {e}")
            
            self.logger.info(f"Random seed restoration completed. "
                           f"Restored: {len(restoration_result['seeds_restored'])}, "
                           f"Failed: {len(restoration_result['seeds_failed'])}")
            
            return restoration_result
            
        except Exception as e:
            self.logger.error(f"Failed to restore random seeds: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def backup_checkpoint(self, checkpoint_path: Union[str, Path], 
                         backup_dir: Optional[Union[str, Path]] = None,
                         include_metadata: bool = True) -> Dict[str, Any]:
        """
        Create a backup of a checkpoint with metadata.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            backup_dir: Directory to store the backup
            include_metadata: Whether to include checkpoint metadata
            
        Returns:
            Backup operation results
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if backup_dir is None:
                backup_dir = Path.cwd() / "checkpoint_backups"
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{checkpoint_path.stem}_{timestamp}{checkpoint_path.suffix}"
            backup_path = backup_dir / backup_filename
            
            # Copy the checkpoint file
            import shutil
            shutil.copy2(checkpoint_path, backup_path)
            
            backup_result = {
                'original_path': str(checkpoint_path),
                'backup_path': str(backup_path),
                'backup_size': backup_path.stat().st_size,
                'timestamp': timestamp,
                'success': True
            }
            
            # Add metadata if requested
            if include_metadata:
                metadata_path = backup_path.with_suffix(backup_path.suffix + '.metadata.json')
                checkpoint_info = self._analyze_checkpoint_file(checkpoint_path)
                with open(metadata_path, 'w') as f:
                    json.dump(checkpoint_info, f, indent=2, default=str)
                backup_result['metadata_path'] = str(metadata_path)
            
            self.logger.info(f"Successfully backed up checkpoint to {backup_path}")
            return backup_result
            
        except Exception as e:
            self.logger.error(f"Failed to backup checkpoint: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # ================================================================
    # High-Level Reproduction Orchestration Methods
    # ================================================================
    
    def reproduce_experiment(self, 
                           metadata_path: Optional[Union[str, Path]] = None,
                           experiment_id: Optional[int] = None,
                           output_dir: Optional[Union[str, Path]] = None,
                           environment_name: str = "reproduced-experiment",
                           restore_config: bool = True,
                           restore_checkpoints: bool = True,
                           restore_seeds: bool = True,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Complete experiment reproduction workflow.
        
        Args:
            metadata_path: Path to reproduction metadata file
            experiment_id: Experiment ID to reproduce from tracker
            output_dir: Directory to restore experiment to
            environment_name: Name for recreated environment
            restore_config: Whether to restore configuration files
            restore_checkpoints: Whether to restore checkpoint files
            restore_seeds: Whether to restore random seeds
            progress_callback: Callback function for progress updates
            
        Returns:
            Reproduction results with status and paths
        """
        try:
            results = {
                'success': True,
                'steps_completed': [],
                'steps_failed': [],
                'metadata': {},
                'environment': {},
                'configuration': {},
                'checkpoints': {},
                'random_seeds': {},
                'output_directory': str(output_dir) if output_dir else None,
                'reproduction_timestamp': datetime.now().isoformat()
            }
            
            if progress_callback:
                progress_callback("Starting experiment reproduction...")
            
            # Step 1: Load metadata
            if progress_callback:
                progress_callback("Loading reproduction metadata...")
            
            if metadata_path:
                metadata = self.load_reproduction_metadata(metadata_path)
            elif experiment_id and self.tracker:
                metadata = self._capture_experiment_metadata_by_id(experiment_id)
            else:
                raise ValueError("Either metadata_path or experiment_id must be provided")
            
            results['metadata'] = metadata
            results['steps_completed'].append('metadata_loaded')
            
            # Step 2: Environment recreation
            if progress_callback:
                progress_callback("Recreating environment...")
            
            try:
                env_result = self.recreate_environment(
                    metadata, 
                    environment_name=environment_name, 
                    force=True
                )
                results['environment'] = env_result
                results['steps_completed'].append('environment_recreated')
                
                if progress_callback:
                    progress_callback(f"Environment recreated: {env_result.get('environment_name', 'unknown')}")
                    
            except Exception as e:
                error_msg = f"Environment recreation failed: {e}"
                results['steps_failed'].append(('environment_recreation', error_msg))
                self.logger.error(error_msg)
                if progress_callback:
                    progress_callback(f"Warning: {error_msg}")
            
            # Step 3: Configuration restoration
            if restore_config:
                if progress_callback:
                    progress_callback("Restoring configuration files...")
                
                try:
                    config_result = self.restore_configuration(metadata, output_dir)
                    results['configuration'] = config_result
                    results['steps_completed'].append('configuration_restored')
                    
                    if progress_callback:
                        restored_count = len(config_result.get('restored_files', []))
                        progress_callback(f"Restored {restored_count} configuration files")
                        
                except Exception as e:
                    error_msg = f"Configuration restoration failed: {e}"
                    results['steps_failed'].append(('configuration_restoration', error_msg))
                    self.logger.error(error_msg)
                    if progress_callback:
                        progress_callback(f"Warning: {error_msg}")
            
            # Step 4: Checkpoint discovery and backup
            if restore_checkpoints:
                if progress_callback:
                    progress_callback("Processing checkpoints...")
                
                try:
                    # Find checkpoints in metadata or discover locally
                    checkpoint_metadata = metadata.get('checkpoints')
                    if not checkpoint_metadata and output_dir:
                        checkpoint_metadata = self.capture_checkpoint_metadata(output_dir)
                    
                    if checkpoint_metadata:
                        results['checkpoints'] = checkpoint_metadata
                        results['steps_completed'].append('checkpoints_processed')
                        
                        checkpoint_count = len(checkpoint_metadata.get('checkpoints_found', []))
                        if progress_callback:
                            progress_callback(f"Found {checkpoint_count} checkpoints")
                    
                except Exception as e:
                    error_msg = f"Checkpoint processing failed: {e}"
                    results['steps_failed'].append(('checkpoint_processing', error_msg))
                    self.logger.error(error_msg)
                    if progress_callback:
                        progress_callback(f"Warning: {error_msg}")
            
            # Step 5: Random seed restoration
            if restore_seeds:
                if progress_callback:
                    progress_callback("Restoring random seeds...")
                
                try:
                    seed_result = self.restore_random_seeds(metadata)
                    results['random_seeds'] = seed_result
                    results['steps_completed'].append('random_seeds_restored')
                    
                    restored_count = len(seed_result.get('seeds_restored', []))
                    if progress_callback:
                        progress_callback(f"Restored {restored_count} random seed states")
                        
                except Exception as e:
                    error_msg = f"Random seed restoration failed: {e}"
                    results['steps_failed'].append(('random_seed_restoration', error_msg))
                    self.logger.error(error_msg)
                    if progress_callback:
                        progress_callback(f"Warning: {error_msg}")
            
            # Step 6: Validation
            if progress_callback:
                progress_callback("Validating reproduction...")
            
            try:
                validation_result = self.validate_reproduction(metadata, results)
                results['validation'] = validation_result
                results['steps_completed'].append('reproduction_validated')
                
                if validation_result.get('overall_valid', False):
                    if progress_callback:
                        progress_callback("Reproduction validation passed")
                else:
                    if progress_callback:
                        progress_callback("Reproduction validation failed")
                        
            except Exception as e:
                error_msg = f"Reproduction validation failed: {e}"
                results['steps_failed'].append(('reproduction_validation', error_msg))
                self.logger.error(error_msg)
                if progress_callback:
                    progress_callback(f"Warning: {error_msg}")
            
            # Final status
            if len(results['steps_failed']) == 0:
                if progress_callback:
                    progress_callback("Experiment reproduction completed successfully!")
            else:
                results['success'] = False
                if progress_callback:
                    failed_count = len(results['steps_failed'])
                    progress_callback(f"Reproduction completed with {failed_count} warnings/errors")
            
            self.logger.info(f"Experiment reproduction completed. "
                           f"Success: {results['success']}, "
                           f"Steps completed: {len(results['steps_completed'])}, "
                           f"Steps failed: {len(results['steps_failed'])}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment reproduction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'reproduction_timestamp': datetime.now().isoformat()
            }
    
    def _capture_experiment_metadata_by_id(self, experiment_id: int) -> Dict[str, Any]:
        """Capture comprehensive metadata for an experiment by ID."""
        if not self.tracker:
            raise ValueError("ExperimentTracker not available for experiment ID lookup")
        
        # Create a temporary ReproductionEngine for the specific experiment
        temp_engine = ReproductionEngine(experiment_id=experiment_id, tracker=self.tracker)
        return temp_engine.capture_reproduction_metadata()
    
    def validate_reproduction(self, original_metadata: Dict[str, Any], 
                            reproduction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that reproduction was successful and complete.
        
        Args:
            original_metadata: Original experiment metadata
            reproduction_results: Results from reproduction process
            
        Returns:
            Validation results
        """
        try:
            validation_result = {
                'overall_valid': True,
                'environment_valid': True,
                'configuration_valid': True,
                'checkpoints_valid': True,
                'random_seeds_valid': True,
                'warnings': [],
                'errors': [],
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Validate environment
            if 'environment_recreated' in reproduction_results.get('steps_completed', []):
                env_result = reproduction_results.get('environment', {})
                if not env_result.get('success', False):
                    validation_result['environment_valid'] = False
                    validation_result['errors'].append("Environment recreation failed")
            else:
                validation_result['warnings'].append("Environment recreation was skipped")
            
            # Validate configuration
            if 'configuration_restored' in reproduction_results.get('steps_completed', []):
                config_result = reproduction_results.get('configuration', {})
                errors = config_result.get('errors', [])
                if errors:
                    validation_result['configuration_valid'] = False
                    validation_result['errors'].extend([f"Config error: {error}" for error in errors])
            else:
                validation_result['warnings'].append("Configuration restoration was skipped")
            
            # Validate checkpoints
            if 'checkpoints_processed' in reproduction_results.get('steps_completed', []):
                checkpoint_result = reproduction_results.get('checkpoints', {})
                if not checkpoint_result:
                    validation_result['warnings'].append("No checkpoints found")
            else:
                validation_result['warnings'].append("Checkpoint processing was skipped")
            
            # Validate random seeds
            if 'random_seeds_restored' in reproduction_results.get('steps_completed', []):
                seed_result = reproduction_results.get('random_seeds', {})
                failed_seeds = seed_result.get('seeds_failed', [])
                if failed_seeds:
                    validation_result['random_seeds_valid'] = False
                    validation_result['errors'].extend([f"Seed error: {error}" for error in failed_seeds])
            else:
                validation_result['warnings'].append("Random seed restoration was skipped")
            
            # Check for any step failures
            failed_steps = reproduction_results.get('steps_failed', [])
            if failed_steps:
                validation_result['overall_valid'] = False
                for step, error in failed_steps:
                    validation_result['errors'].append(f"{step}: {error}")
            
            # Overall validation
            validation_result['overall_valid'] = (
                validation_result['environment_valid'] and
                validation_result['configuration_valid'] and
                validation_result['checkpoints_valid'] and
                validation_result['random_seeds_valid'] and
                len(validation_result['errors']) == 0
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Reproduction validation failed: {e}")
            return {
                'overall_valid': False,
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def generate_reproduction_report(self, reproduction_results: Dict[str, Any],
                                   output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive reproduction report.
        
        Args:
            reproduction_results: Results from reproduce_experiment
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        try:
            timestamp = reproduction_results.get('reproduction_timestamp', 'Unknown')
            success = reproduction_results.get('success', False)
            
            report_lines = [
                "# Experiment Reproduction Report",
                f"**Generated:** {timestamp}",
                f"**Status:** {'✅ SUCCESS' if success else '❌ FAILED'}",
                "",
                "## Summary",
                f"- **Steps Completed:** {len(reproduction_results.get('steps_completed', []))}",
                f"- **Steps Failed:** {len(reproduction_results.get('steps_failed', []))}",
                f"- **Output Directory:** {reproduction_results.get('output_directory', 'Not specified')}",
                "",
            ]
            
            # Completed steps
            completed_steps = reproduction_results.get('steps_completed', [])
            if completed_steps:
                report_lines.extend([
                    "## ✅ Completed Steps",
                    ""
                ])
                for step in completed_steps:
                    report_lines.append(f"- {step.replace('_', ' ').title()}")
                report_lines.append("")
            
            # Failed steps
            failed_steps = reproduction_results.get('steps_failed', [])
            if failed_steps:
                report_lines.extend([
                    "## ❌ Failed Steps",
                    ""
                ])
                for step, error in failed_steps:
                    report_lines.append(f"- **{step.replace('_', ' ').title()}:** {error}")
                report_lines.append("")
            
            # Environment details
            env_result = reproduction_results.get('environment', {})
            if env_result:
                report_lines.extend([
                    "## 🐍 Environment Recreation",
                    f"- **Environment Name:** {env_result.get('environment_name', 'Unknown')}",
                    f"- **Package Manager:** {env_result.get('package_manager', 'Unknown')}",
                    f"- **Success:** {env_result.get('success', False)}",
                    ""
                ])
            
            # Configuration details
            config_result = reproduction_results.get('configuration', {})
            if config_result:
                report_lines.extend([
                    "## ⚙️ Configuration Restoration",
                    f"- **Files Restored:** {len(config_result.get('restored_files', []))}",
                    f"- **Files Skipped:** {len(config_result.get('skipped_files', []))}",
                    f"- **Errors:** {len(config_result.get('errors', []))}",
                    ""
                ])
                
                if config_result.get('restored_files'):
                    report_lines.extend([
                        "### Restored Configuration Files:",
                        ""
                    ])
                    for file_path in config_result['restored_files']:
                        report_lines.append(f"- `{file_path}`")
                    report_lines.append("")
            
            # Checkpoint details
            checkpoint_result = reproduction_results.get('checkpoints', {})
            if checkpoint_result:
                checkpoints_found = checkpoint_result.get('checkpoints_found', [])
                report_lines.extend([
                    "## 💾 Checkpoint Processing",
                    f"- **Checkpoints Found:** {len(checkpoints_found)}",
                    ""
                ])
                
                if checkpoints_found:
                    report_lines.extend([
                        "### Found Checkpoints:",
                        ""
                    ])
                    for checkpoint in checkpoints_found[:5]:  # Show first 5
                        filename = checkpoint.get('filename', 'Unknown')
                        framework = checkpoint.get('framework', 'unknown')
                        size = checkpoint.get('size_bytes', 0)
                        report_lines.append(f"- `{filename}` ({framework}, {size:,} bytes)")
                    
                    if len(checkpoints_found) > 5:
                        report_lines.append(f"- ... and {len(checkpoints_found) - 5} more")
                    report_lines.append("")
            
            # Random seeds details
            seed_result = reproduction_results.get('random_seeds', {})
            if seed_result:
                report_lines.extend([
                    "## 🎲 Random Seed Restoration",
                    f"- **Seeds Restored:** {len(seed_result.get('seeds_restored', []))}",
                    f"- **Seeds Failed:** {len(seed_result.get('seeds_failed', []))}",
                    ""
                ])
                
                if seed_result.get('seeds_restored'):
                    report_lines.extend([
                        "### Successfully Restored:",
                        ""
                    ])
                    for seed in seed_result['seeds_restored']:
                        report_lines.append(f"- {seed}")
                    report_lines.append("")
            
            # Validation results
            validation = reproduction_results.get('validation', {})
            if validation:
                overall_valid = validation.get('overall_valid', False)
                report_lines.extend([
                    "## ✅ Validation Results",
                    f"- **Overall Valid:** {'Yes' if overall_valid else 'No'}",
                    f"- **Environment Valid:** {validation.get('environment_valid', False)}",
                    f"- **Configuration Valid:** {validation.get('configuration_valid', False)}",
                    f"- **Checkpoints Valid:** {validation.get('checkpoints_valid', False)}",
                    f"- **Random Seeds Valid:** {validation.get('random_seeds_valid', False)}",
                    ""
                ])
                
                warnings = validation.get('warnings', [])
                if warnings:
                    report_lines.extend([
                        "### ⚠️ Warnings:",
                        ""
                    ])
                    for warning in warnings:
                        report_lines.append(f"- {warning}")
                    report_lines.append("")
                
                errors = validation.get('errors', [])
                if errors:
                    report_lines.extend([
                        "### ❌ Errors:",
                        ""
                    ])
                    for error in errors:
                        report_lines.append(f"- {error}")
                    report_lines.append("")
            
            report_lines.extend([
                "---",
                "*Report generated by YinshML ReproductionEngine*"
            ])
            
            report_content = "\n".join(report_lines)
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"Reproduction report saved to {output_path}")
            
            return report_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate reproduction report: {e}")
            return f"Error generating report: {e}"
    
    def restore_configuration(self, metadata: Dict[str, Any], output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Restore configuration files from metadata.
        
        Args:
            metadata: Reproduction metadata containing configuration info
            output_dir: Directory to restore configuration files to
            
        Returns:
            Configuration restoration results
        """
        try:
            result = {
                'success': True,
                'restored_files': [],
                'skipped_files': [],
                'errors': [],
                'restoration_timestamp': datetime.now().isoformat()
            }
            
            # Get configuration metadata
            config_metadata = metadata.get('configuration', {})
            config_files = config_metadata.get('config_files', [])
            
            if not config_files:
                result['success'] = False
                result['errors'].append("No configuration files found in metadata")
                return result
            
            # Set up output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path.cwd()
            
            # Restore each configuration file
            for config_file in config_files:
                try:
                    filename = config_file.get('filename', 'unknown')
                    content = config_file.get('content', '')
                    relative_path = config_file.get('relative_path', filename)
                    
                    # Create the file path
                    file_path = output_path / relative_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write the configuration file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    result['restored_files'].append(str(file_path))
                    self.logger.info(f"Restored configuration file: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to restore {filename}: {e}"
                    result['errors'].append(error_msg)
                    result['skipped_files'].append(filename)
                    self.logger.error(error_msg)
            
            # Check overall success
            if result['errors']:
                result['success'] = len(result['restored_files']) > 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Configuration restoration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'restoration_timestamp': datetime.now().isoformat()
            }
    
    def restore_random_seeds(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore random seed states from metadata.
        
        Args:
            metadata: Reproduction metadata containing random seed info
            
        Returns:
            Random seed restoration results
        """
        try:
            result = {
                'success': True,
                'seeds_restored': [],
                'seeds_failed': [],
                'restoration_timestamp': datetime.now().isoformat()
            }
            
            # Get random seeds metadata
            random_seeds = metadata.get('random_seeds', {})
            
            if not random_seeds:
                result['success'] = False
                result['seeds_failed'].append("No random seeds found in metadata")
                return result
            
            # Restore Python random seed
            if 'python_random' in random_seeds:
                try:
                    python_state = random_seeds['python_random']
                    if isinstance(python_state, dict) and 'state' in python_state:
                        # Convert state back to tuple format that random expects
                        state_data = python_state['state']
                        if isinstance(state_data, list) and len(state_data) >= 3:
                            # Reconstruct the state tuple
                            version = state_data[0] if len(state_data) > 0 else 3
                            state_array = tuple(state_data[1]) if len(state_data) > 1 and isinstance(state_data[1], list) else ()
                            gauss_next = state_data[2] if len(state_data) > 2 else None
                            
                            if state_array:
                                import random
                                random.setstate((version, state_array, gauss_next))
                                result['seeds_restored'].append("Python random state")
                                self.logger.info("Restored Python random state")
                except Exception as e:
                    error_msg = f"Failed to restore Python random state: {e}"
                    result['seeds_failed'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Restore NumPy random seed
            if 'numpy_random' in random_seeds:
                try:
                    numpy_state = random_seeds['numpy_random']
                    if isinstance(numpy_state, dict) and 'state' in numpy_state:
                        import numpy as np
                        # NumPy state is more complex, try to restore what we can
                        state_dict = numpy_state['state']
                        if 'bit_generator' in state_dict and 'state' in state_dict:
                            # For newer NumPy versions, try to restore the generator state
                            try:
                                rng = np.random.default_rng()
                                # This is a simplified restoration - full state restoration is complex
                                if 'pos' in state_dict['state']:
                                    # Set a reasonable seed based on the position
                                    seed_value = state_dict['state']['pos'] % (2**32)
                                    np.random.seed(seed_value)
                                    result['seeds_restored'].append("NumPy random state (simplified)")
                                    self.logger.info("Restored NumPy random state (simplified)")
                            except Exception:
                                # Fallback to basic seed if available
                                if 'seed' in numpy_state:
                                    np.random.seed(numpy_state['seed'])
                                    result['seeds_restored'].append("NumPy random seed")
                                    self.logger.info("Restored NumPy random seed")
                except Exception as e:
                    error_msg = f"Failed to restore NumPy random state: {e}"
                    result['seeds_failed'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Restore PyTorch random seed
            if 'torch_random' in random_seeds:
                try:
                    torch_state = random_seeds['torch_random']
                    if isinstance(torch_state, dict):
                        try:
                            import torch
                            
                            # Restore CPU random state
                            if 'cpu_state' in torch_state:
                                cpu_state = torch_state['cpu_state']
                                if isinstance(cpu_state, dict) and 'state' in cpu_state:
                                    # Convert state back to tensor
                                    state_data = cpu_state['state']
                                    if isinstance(state_data, list):
                                        state_tensor = torch.tensor(state_data, dtype=torch.uint8)
                                        torch.set_rng_state(state_tensor)
                                        result['seeds_restored'].append("PyTorch CPU random state")
                                        self.logger.info("Restored PyTorch CPU random state")
                            
                            # Restore CUDA random state if available
                            if 'cuda_state' in torch_state and torch.cuda.is_available():
                                cuda_state = torch_state['cuda_state']
                                if isinstance(cuda_state, dict) and 'states' in cuda_state:
                                    # Restore CUDA states for each device
                                    states = cuda_state['states']
                                    for device_id, device_state in enumerate(states):
                                        if isinstance(device_state, list):
                                            state_tensor = torch.tensor(device_state, dtype=torch.uint8)
                                            torch.cuda.set_rng_state(state_tensor, device=device_id)
                                    result['seeds_restored'].append("PyTorch CUDA random states")
                                    self.logger.info("Restored PyTorch CUDA random states")
                                    
                        except ImportError:
                            result['seeds_failed'].append("PyTorch not available for state restoration")
                        except Exception as e:
                            error_msg = f"Failed to restore PyTorch random state: {e}"
                            result['seeds_failed'].append(error_msg)
                            self.logger.error(error_msg)
                except Exception as e:
                    error_msg = f"Failed to process PyTorch random state: {e}"
                    result['seeds_failed'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Check overall success
            result['success'] = len(result['seeds_restored']) > 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Random seed restoration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'restoration_timestamp': datetime.now().isoformat()
            }

    def validate_experiment_reproduction(self, original_results: Dict[str, Any], 
                                       reproduced_results: Dict[str, Any],
                                       reproduction_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of experiment reproduction using the advanced validation framework.
        
        This method leverages the ValidationFramework for advanced result comparison, statistical validation,
        and conflict resolution beyond basic reproduction process validation.
        
        Args:
            original_results: Original experiment results including metrics, files, model weights
            reproduced_results: Reproduced experiment results
            reproduction_metadata: Metadata from the reproduction process
            
        Returns:
            Comprehensive validation results with statistical analysis and recommendations
        """
        try:
            self.logger.info("Starting comprehensive experiment reproduction validation")
            
            # Use the validation framework for advanced validation
            validation_result = self.validation_framework.validate_reproduction(
                original_results, reproduced_results, reproduction_metadata
            )
            
            self.logger.info(f"Validation completed - Overall valid: {validation_result.get('overall_valid', False)}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            return {
                'overall_valid': False,
                'error': str(e),
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def resolve_environment_conflicts(self, conflicts: List[str], 
                                    environment_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment conflicts using advanced conflict resolution strategies.
        
        Args:
            conflicts: List of environment conflict descriptions
            environment_metadata: Environment metadata for context
            
        Returns:
            Conflict resolution results with suggested actions
        """
        try:
            self.logger.info(f"Resolving {len(conflicts)} environment conflicts")
            
            # Use the validation framework's conflict resolver
            resolution_result = self.validation_framework.conflict_resolver.resolve_environment_conflicts(
                conflicts, environment_metadata
            )
            
            success = resolution_result.get('resolution_successful', False)
            self.logger.info(f"Conflict resolution completed - Success: {success}")
            
            return resolution_result
            
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return {
                'resolution_successful': False,
                'error': str(e),
                'resolution_timestamp': datetime.now().isoformat()
            }
    
    def compare_experiment_results(self, original_results: Dict[str, Any], 
                                 reproduced_results: Dict[str, Any],
                                 comparison_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare experiment results using advanced comparison methods.
        
        Args:
            original_results: Original experiment results
            reproduced_results: Reproduced experiment results  
            comparison_config: Configuration for comparison tolerances and methods
            
        Returns:
            Detailed comparison results across multiple result types
        """
        try:
            self.logger.info("Starting advanced experiment result comparison")
            
            # Configure comparison tolerances if provided
            if comparison_config:
                comparator = ResultComparator(comparison_config)
            else:
                comparator = self.validation_framework.result_comparator
            
            comparison_result = {
                'overall_match': True,
                'comparisons': {},
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            # Compare metrics
            if 'metrics' in original_results and 'metrics' in reproduced_results:
                metrics_comparison = comparator.compare_metrics(
                    original_results['metrics'], reproduced_results['metrics']
                )
                comparison_result['comparisons']['metrics'] = metrics_comparison
                
                if not metrics_comparison.get('overall_match', False):
                    comparison_result['overall_match'] = False
            
            # Compare files
            if 'output_files' in original_results and 'output_files' in reproduced_results:
                files_comparison = comparator.compare_files(
                    original_results['output_files'], reproduced_results['output_files']
                )
                comparison_result['comparisons']['files'] = files_comparison
                
                if not files_comparison.get('overall_match', False):
                    comparison_result['overall_match'] = False
            
            # Compare model weights
            if 'model_weights' in original_results and 'model_weights' in reproduced_results:
                weights_comparison = comparator.compare_model_weights(
                    original_results['model_weights'], reproduced_results['model_weights']
                )
                comparison_result['comparisons']['model_weights'] = weights_comparison
                
                if not weights_comparison.get('overall_match', False):
                    comparison_result['overall_match'] = False
            
            self.logger.info(f"Result comparison completed - Overall match: {comparison_result['overall_match']}")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Result comparison failed: {e}")
            return {
                'overall_match': False,
                'error': str(e),
                'comparison_timestamp': datetime.now().isoformat()
            }
    
    def export_validation_report(self, validation_result: Dict[str, Any], 
                               export_format: str = 'json',
                               output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export validation results in various formats for CI/CD integration.
        
        Args:
            validation_result: Validation results to export
            export_format: Export format ('json', 'junit', 'markdown')
            output_path: Optional path to save the report
            
        Returns:
            Formatted report content
        """
        try:
            self.logger.info(f"Exporting validation report in {export_format} format")
            
            report_content = self.validation_framework.export_validation_report(
                validation_result, export_format, output_path
            )
            
            self.logger.info("Validation report exported successfully")
            return report_content
            
        except Exception as e:
            self.logger.error(f"Validation report export failed: {e}")
            return f"Error exporting validation report: {e}"
    
    def get_validation_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """
        Get actionable recommendations based on validation results.
        
        Args:
            validation_result: Results from validation process
            
        Returns:
            List of actionable recommendations
        """
        try:
            return validation_result.get('recommendations', [])
            
        except Exception as e:
            self.logger.error(f"Failed to extract recommendations: {e}")
            return [f"Error extracting recommendations: {e}"]

    def __repr__(self) -> str:
        """String representation of the ReproductionEngine."""
        if self.experiment_id:
            return f"ReproductionEngine(experiment_id={self.experiment_id})"
        elif self.experiment_path:
            return f"ReproductionEngine(experiment_path={self.experiment_path})"
        else:
            return "ReproductionEngine()" 