"""
Capture modules for automatic metadata collection in YinshML experiments.

This module provides specialized classes for capturing various types of metadata
including git repository information, environment variables, system information,
and dataset fingerprints.
"""

import subprocess
import logging
import os
import platform
import socket
import sys
import pkg_resources
import hashlib
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CaptureError(Exception):
    """Base exception for capture operations."""
    pass


class GitCaptureError(CaptureError):
    """Raised when git capture operations fail."""
    pass


class EnvironmentCaptureError(CaptureError):
    """Raised when environment capture operations fail."""
    pass


class DatasetCaptureError(CaptureError):
    """Raised when dataset capture operations fail."""
    pass


class GitCapture:
    """
    Captures git repository metadata with improved modularity and error handling.
    
    This class extracts git information including commit hash, branch, working
    directory status, remote URL, and commit details. It provides both subprocess-based
    and optional GitPython-based implementations.
    """
    
    def __init__(self, repository_path: Optional[str] = None, 
                 fail_silently: bool = True,
                 capture_untracked: bool = False,
                 capture_diff_summary: bool = False,
                 command_timeout: int = 30):
        """
        Initialize GitCapture instance.
        
        Args:
            repository_path: Path to git repository. If None, uses current directory.
            fail_silently: If True, returns default values on errors. If False, raises exceptions.
            capture_untracked: Whether to capture list of untracked files.
            capture_diff_summary: Whether to capture a summary of uncommitted changes.
            command_timeout: Timeout in seconds for git commands (default: 30).
        """
        self.repository_path = Path(repository_path) if repository_path else Path.cwd()
        self.fail_silently = fail_silently
        self.capture_untracked = capture_untracked
        self.capture_diff_summary = capture_diff_summary
        self.command_timeout = command_timeout
        
    def _run_git_command(self, command: List[str]) -> str:
        """
        Execute a git command and return the output.
        
        Args:
            command: Git command arguments as a list.
            
        Returns:
            Command output as string.
            
        Raises:
            GitCaptureError: If command fails and fail_silently is False.
        """
        try:
            result = subprocess.run(
                ['git'] + command,
                capture_output=True,
                text=True,
                cwd=self.repository_path,
                timeout=self.command_timeout  # Prevent hanging
            )
            
            if result.returncode != 0:
                error_msg = f"Git command failed: {' '.join(command)}. Error: {result.stderr.strip()}"
                if not self.fail_silently:
                    raise GitCaptureError(error_msg)
                logger.warning(error_msg)
                return ""
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            error_msg = f"Git command timed out: {' '.join(command)}"
            if not self.fail_silently:
                raise GitCaptureError(error_msg)
            logger.warning(error_msg)
            return ""
        except FileNotFoundError:
            error_msg = "Git command not found - git not installed or not in PATH"
            if not self.fail_silently:
                raise GitCaptureError(error_msg)
            logger.warning(error_msg)
            return ""
        except Exception as e:
            error_msg = f"Unexpected error running git command: {e}"
            if not self.fail_silently:
                raise GitCaptureError(error_msg) from e
            logger.warning(error_msg)
            return ""
    
    def is_git_repository(self) -> bool:
        """
        Check if the current path is inside a git repository.
        
        Returns:
            True if inside a git repository, False otherwise.
        """
        try:
            output = self._run_git_command(['rev-parse', '--is-inside-work-tree'])
            return output.lower() == 'true'
        except GitCaptureError:
            return False
    
    def get_commit_hash(self, short: bool = False) -> str:
        """
        Get the current commit hash.
        
        Args:
            short: If True, returns short hash. Otherwise returns full hash.
            
        Returns:
            Commit hash as string, or 'unknown' if unavailable.
        """
        command = ['rev-parse']
        if short:
            command.append('--short')
        command.append('HEAD')
        
        result = self._run_git_command(command)
        return result if result else 'unknown'
    
    def get_branch_name(self) -> str:
        """
        Get the current branch name.
        
        Returns:
            Branch name as string, or 'unknown' if unavailable.
        """
        result = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
        return result if result else 'unknown'
    
    def get_remote_url(self, remote: str = 'origin') -> Optional[str]:
        """
        Get the remote URL for the specified remote.
        
        Args:
            remote: Name of the remote (default: 'origin').
            
        Returns:
            Remote URL as string, or None if unavailable.
        """
        result = self._run_git_command(['config', '--get', f'remote.{remote}.url'])
        return result if result else None
    
    def get_commit_message(self) -> Optional[str]:
        """
        Get the message of the current commit.
        
        Returns:
            Commit message as string, or None if unavailable.
        """
        result = self._run_git_command(['log', '-1', '--pretty=format:%s'])
        return result if result else None
    
    def get_commit_author(self) -> Optional[str]:
        """
        Get the author of the current commit.
        
        Returns:
            Author string in format "Name <email>", or None if unavailable.
        """
        result = self._run_git_command(['log', '-1', '--pretty=format:%an <%ae>'])
        return result if result else None
    
    def get_commit_timestamp(self) -> Optional[str]:
        """
        Get the timestamp of the current commit.
        
        Returns:
            ISO format timestamp string, or None if unavailable.
        """
        result = self._run_git_command(['log', '-1', '--pretty=format:%ai'])
        return result if result else None
    
    def get_working_directory_status(self) -> Dict[str, Any]:
        """
        Get working directory status including clean state and file lists.
        
        Returns:
            Dictionary containing working directory status information.
        """
        status_info = {
            'is_clean': True,
            'modified_files': [],
            'untracked_files': [],
            'staged_files': []
        }
        
        # Check basic status
        porcelain_output = self._run_git_command(['status', '--porcelain'])
        if not porcelain_output:
            return status_info
        
        status_info['is_clean'] = False
        
        # Parse porcelain output
        for line in porcelain_output.split('\n'):
            if len(line) < 3:
                continue
                
            status_code = line[:2]
            filename = line[2:].strip()  # Remove leading space and whitespace
            
            # Check for modifications (M in first or second position)
            if 'M' in status_code:
                status_info['modified_files'].append(filename)
            
            # Check for staged files (first position not space or ?)
            if status_code[0] not in [' ', '?']:
                status_info['staged_files'].append(filename)
            
            # Check for untracked files (??)
            if status_code == '??':
                status_info['untracked_files'].append(filename)
        
        return status_info
    
    def get_diff_summary(self) -> Dict[str, Any]:
        """
        Get a summary of uncommitted changes.
        
        Returns:
            Dictionary containing diff statistics.
        """
        diff_summary = {
            'lines_added': 0,
            'lines_removed': 0,
            'files_changed': 0
        }
        
        if not self.capture_diff_summary:
            return diff_summary
        
        # Get diff statistics
        diff_output = self._run_git_command(['diff', '--numstat'])
        if not diff_output:
            return diff_summary
        
        for line in diff_output.split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        added = int(parts[0]) if parts[0] != '-' else 0
                        removed = int(parts[1]) if parts[1] != '-' else 0
                        diff_summary['lines_added'] += added
                        diff_summary['lines_removed'] += removed
                        diff_summary['files_changed'] += 1
                    except ValueError:
                        continue
        
        return diff_summary
    
    def capture(self) -> Dict[str, Any]:
        """
        Capture all git metadata in a single operation.
        
        Returns:
            Dictionary containing all available git information.
        """
        git_info = {
            'commit': 'unknown',
            'commit_short': 'unknown',
            'branch': 'unknown',
            'remote_url': None,
            'commit_message': None,
            'author': None,
            'timestamp': None,
            'working_directory_clean': True,
            'capture_timestamp': datetime.now().isoformat(),
            'repository_path': str(self.repository_path)
        }
        
        try:
            # Check if we're in a git repository
            if not self.is_git_repository():
                logger.warning(f"Path {self.repository_path} is not inside a git repository")
                return git_info
            
            # Capture basic information
            git_info.update({
                'commit': self.get_commit_hash(short=False),
                'commit_short': self.get_commit_hash(short=True),
                'branch': self.get_branch_name(),
                'remote_url': self.get_remote_url(),
                'commit_message': self.get_commit_message(),
                'author': self.get_commit_author(),
                'timestamp': self.get_commit_timestamp(),
            })
            
            # Capture working directory status
            status_info = self.get_working_directory_status()
            git_info['working_directory_clean'] = status_info['is_clean']
            git_info.update(status_info)
            
            # Optionally capture diff summary
            if self.capture_diff_summary:
                git_info['diff_summary'] = self.get_diff_summary()
            
            logger.debug(f"Captured git metadata: commit={git_info['commit_short']}, "
                        f"branch={git_info['branch']}, clean={git_info['working_directory_clean']}")
            
        except Exception as e:
            error_msg = f"Failed to capture git metadata: {e}"
            if not self.fail_silently:
                raise GitCaptureError(error_msg) from e
            logger.warning(error_msg)
        
        return git_info
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return configuration and git information.
        
        Returns:
            Dictionary containing git configuration and captured information.
        """
        config = {
            'command_timeout': self.command_timeout,
            'capture_untracked': self.capture_untracked,
            'capture_diff_summary': self.capture_diff_summary,
            'fail_silently': self.fail_silently,
            'repository_path': str(self.repository_path)
        }
        git_info = self.capture()
        return {**config, **git_info}


class EnvironmentCapture:
    """
    Captures Python environment and system metadata with comprehensive filtering and optimization.
    
    This class extracts environment variables, installed packages, system information,
    and framework-specific metadata (like PyTorch/CUDA info). It provides configurable
    filtering for sensitive data and performance optimizations.
    """
    
    def __init__(self, fail_silently: bool = True,
                 capture_packages: bool = True,
                 capture_env_vars: bool = True,
                 capture_system_info: bool = True,
                 capture_frameworks: bool = True,
                 custom_env_vars: List[str] = None,
                 exclude_env_vars: List[str] = None):
        """
        Initialize EnvironmentCapture instance.
        
        Args:
            fail_silently: If True, captures what it can on errors. If False, raises exceptions.
            capture_packages: Whether to capture installed Python packages.
            capture_env_vars: Whether to capture environment variables.
            capture_system_info: Whether to capture system information.
            capture_frameworks: Whether to capture framework-specific info (PyTorch, etc.).
            custom_env_vars: Additional environment variables to capture.
            exclude_env_vars: Environment variables to explicitly exclude.
        """
        self.fail_silently = fail_silently
        self.capture_packages = capture_packages
        self.capture_env_vars = capture_env_vars
        self.capture_system_info = capture_system_info
        self.capture_frameworks = capture_frameworks
        
        # Default environment variables to capture
        self.default_env_vars = [
            'PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'TF_CPP_MIN_LOG_LEVEL', 'PYTORCH_CUDA_ALLOC_CONF',
            'HOME', 'USER', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV'
        ]
        
        # Add custom environment variables
        if custom_env_vars:
            self.default_env_vars.extend(custom_env_vars)
        
        # Sensitive environment variables to exclude by default
        self.sensitive_env_vars = [
            'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'PRIVATE', 'CREDENTIAL',
            'API_KEY', 'AUTH', 'PASS', 'PWD', 'APIKEY'
        ]
        
        # Add custom exclusions
        if exclude_env_vars:
            self.sensitive_env_vars.extend(exclude_env_vars)
        
        # Cache for expensive operations
        self._package_cache = None
        self._system_cache = None
        
    def _is_sensitive_env_var(self, var_name: str) -> bool:
        """
        Check if an environment variable name contains sensitive information.
        
        Args:
            var_name: Environment variable name.
            
        Returns:
            True if the variable appears to contain sensitive information.
        """
        var_upper = var_name.upper()
        return any(sensitive in var_upper for sensitive in self.sensitive_env_vars)
    
    def get_python_version(self) -> str:
        """
        Get the Python version information.
        
        Returns:
            Python version string.
        """
        try:
            return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        except Exception as e:
            if not self.fail_silently:
                raise EnvironmentCaptureError(f"Failed to get Python version: {e}")
            logger.warning(f"Failed to get Python version: {e}")
            return "unknown"
    
    def get_installed_packages(self) -> Dict[str, str]:
        """
        Get installed Python packages with versions.
        
        Returns:
            Dictionary mapping package names to versions.
        """
        if not self.capture_packages:
            return {}
        
        # Return cached result if available
        if self._package_cache is not None:
            return self._package_cache
        
        packages = {}
        try:
            for pkg in pkg_resources.working_set:
                packages[pkg.project_name] = pkg.version
            
            self._package_cache = packages
            logger.debug(f"Captured {len(packages)} installed packages")
            
        except Exception as e:
            error_msg = f"Failed to capture installed packages: {e}"
            if not self.fail_silently:
                raise EnvironmentCaptureError(error_msg)
            logger.warning(error_msg)
        
        return packages
    
    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get relevant environment variables with sensitive data filtering.
        
        Returns:
            Dictionary of environment variables.
        """
        if not self.capture_env_vars:
            return {}
        
        env_vars = {}
        try:
            # Capture specified environment variables
            for var in self.default_env_vars:
                value = os.environ.get(var)
                if value is not None and not self._is_sensitive_env_var(var):
                    env_vars[var] = value
            
            logger.debug(f"Captured {len(env_vars)} environment variables")
            
        except Exception as e:
            error_msg = f"Failed to capture environment variables: {e}"
            if not self.fail_silently:
                raise EnvironmentCaptureError(error_msg)
            logger.warning(error_msg)
        
        return env_vars
    
    def get_system_information(self) -> Dict[str, Any]:
        """
        Get basic system information.
        
        Returns:
            Dictionary containing system information.
        """
        if not self.capture_system_info:
            return {}
        
        # Return cached result if available
        if self._system_cache is not None:
            return self._system_cache
        
        system_info = {}
        try:
            system_info.update({
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'hostname': socket.gethostname(),
                'python_implementation': platform.python_implementation(),
                'python_version': self.get_python_version()
            })
            
            self._system_cache = system_info
            logger.debug(f"Captured system information with {len(system_info)} fields")
            
        except Exception as e:
            error_msg = f"Failed to capture system information: {e}"
            if not self.fail_silently:
                raise EnvironmentCaptureError(error_msg)
            logger.warning(error_msg)
        
        return system_info
    
    def get_framework_information(self) -> Dict[str, Any]:
        """
        Get framework-specific information (PyTorch, TensorFlow, etc.).
        
        Returns:
            Dictionary containing framework information.
        """
        if not self.capture_frameworks:
            return {}
        
        framework_info = {}
        
        # PyTorch information
        try:
            import torch
            torch_info = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            
            # Add CUDNN version if available
            if torch.cuda.is_available():
                try:
                    torch_info['cudnn_version'] = torch.backends.cudnn.version()
                    torch_info['device_names'] = [
                        torch.cuda.get_device_name(i) 
                        for i in range(torch.cuda.device_count())
                    ]
                except Exception as e:
                    logger.debug(f"Could not get CUDA device details: {e}")
            
            framework_info['torch'] = torch_info
            logger.debug("Captured PyTorch information")
            
        except ImportError:
            logger.debug("PyTorch not available")
        except Exception as e:
            error_msg = f"Failed to capture PyTorch information: {e}"
            if not self.fail_silently:
                raise EnvironmentCaptureError(error_msg)
            logger.warning(error_msg)
        
        # TensorFlow information
        try:
            import tensorflow as tf
            tf_info = {
                'version': tf.__version__,
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
                'gpu_devices': [device.name for device in tf.config.list_physical_devices('GPU')]
            }
            framework_info['tensorflow'] = tf_info
            logger.debug("Captured TensorFlow information")
            
        except ImportError:
            logger.debug("TensorFlow not available")
        except Exception as e:
            error_msg = f"Failed to capture TensorFlow information: {e}"
            if not self.fail_silently:
                raise EnvironmentCaptureError(error_msg)
            logger.warning(error_msg)
        
        # NumPy information
        try:
            import numpy as np
            framework_info['numpy'] = {
                'version': np.__version__
            }
            logger.debug("Captured NumPy information")
            
        except ImportError:
            logger.debug("NumPy not available")
        except Exception as e:
            logger.warning(f"Failed to capture NumPy information: {e}")
        
        return framework_info
    
    def capture(self) -> Dict[str, Any]:
        """
        Capture all environment metadata in a single operation.
        
        Returns:
            Dictionary containing all available environment information.
        """
        env_info = {
            'capture_timestamp': datetime.now().isoformat(),
            'python_version': self.get_python_version()
        }
        
        try:
            # Capture system information
            if self.capture_system_info:
                env_info['system'] = self.get_system_information()
            
            # Capture environment variables
            if self.capture_env_vars:
                env_info['environment_variables'] = self.get_environment_variables()
            
            # Capture installed packages
            if self.capture_packages:
                env_info['installed_packages'] = self.get_installed_packages()
            
            # Capture framework information
            if self.capture_frameworks:
                env_info['frameworks'] = self.get_framework_information()
            
            logger.debug(f"Captured environment metadata with {len(env_info)} categories")
            
        except Exception as e:
            error_msg = f"Failed to capture environment metadata: {e}"
            if not self.fail_silently:
                raise EnvironmentCaptureError(error_msg) from e
            logger.warning(error_msg)
        
        return env_info
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Alias for capture() method for backward compatibility.
        
        Returns:
            Dictionary containing all available environment information.
        """
        return self.capture()
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get configuration parameters for this EnvironmentCapture instance.
        
        Returns:
            Dictionary containing configuration settings.
        """
        return {
            'fail_silently': self.fail_silently,
            'capture_packages': self.capture_packages,
            'capture_env_vars': self.capture_env_vars,
            'capture_system_info': self.capture_system_info,
            'capture_frameworks': self.capture_frameworks,
            'default_env_vars': self.default_env_vars,
            'sensitive_env_vars': self.sensitive_env_vars,
            'custom_env_vars': getattr(self, 'custom_env_vars', []),
            'exclude_env_vars': getattr(self, 'exclude_env_vars', [])
        }
    
    def clear_cache(self) -> None:
        """
        Clear cached results to force fresh capture on next call.
        """
        self._package_cache = None
        self._system_cache = None
        logger.debug("Cleared environment capture cache")


class DatasetFingerprinter:
    """
    Generates reproducible content-based fingerprints of datasets for ML experiment tracking.
    
    This class can fingerprint various dataset formats including CSV files, JSON files,
    pandas DataFrames, numpy arrays, and other common ML data formats. It uses streaming
    and sampling approaches for large datasets to ensure efficient processing.
    """
    
    def __init__(self, max_sample_size: int = 10000,
                 chunk_size: int = 1000,
                 hash_algorithm: str = 'sha256',
                 normalize_whitespace: bool = True,
                 sort_for_consistency: bool = True,
                 fail_silently: bool = True):
        """
        Initialize DatasetFingerprinter instance.
        
        Args:
            max_sample_size: Maximum number of rows to sample for large datasets.
            chunk_size: Size of chunks when processing large datasets.
            hash_algorithm: Hash algorithm to use ('sha256', 'md5', 'sha1').
            normalize_whitespace: Whether to normalize whitespace in text data.
            sort_for_consistency: Whether to sort data for order-independent hashing.
            fail_silently: If True, captures what it can on errors. If False, raises exceptions.
        """
        self.max_sample_size = max_sample_size
        self.chunk_size = chunk_size
        self.hash_algorithm = hash_algorithm.lower()
        self.normalize_whitespace = normalize_whitespace
        self.sort_for_consistency = sort_for_consistency
        self.fail_silently = fail_silently
        
        # Validate hash algorithm
        import hashlib
        if self.hash_algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Hash algorithm '{hash_algorithm}' not available")
        
        self._hash_func = getattr(hashlib, self.hash_algorithm)
        
    def _normalize_data(self, data: Any) -> bytes:
        """
        Normalize data to bytes for consistent hashing.
        
        Args:
            data: Data to normalize.
            
        Returns:
            Normalized data as bytes.
        """
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            if self.normalize_whitespace:
                data = ' '.join(data.split())  # Normalize whitespace
            return data.encode('utf-8')
        elif isinstance(data, (int, float)):
            return str(data).encode('utf-8')
        elif data is None:
            return b'NULL'
        else:
            return str(data).encode('utf-8')
    
    def _sample_dataframe(self, df, max_rows: int = None) -> Any:
        """
        Sample a DataFrame if it's larger than the maximum sample size.
        
        Args:
            df: pandas DataFrame to sample.
            max_rows: Maximum number of rows to include.
            
        Returns:
            Sampled DataFrame.
        """
        max_rows = max_rows or self.max_sample_size
        
        if len(df) <= max_rows:
            return df
        
        # Use systematic sampling for reproducibility
        step = len(df) // max_rows
        if step < 1:
            step = 1
        
        sampled_indices = list(range(0, len(df), step))[:max_rows]
        return df.iloc[sampled_indices]
    
    def fingerprint_dataframe(self, df, include_index: bool = False) -> Dict[str, Any]:
        """
        Generate fingerprint for a pandas DataFrame.
        
        Args:
            df: pandas DataFrame to fingerprint.
            include_index: Whether to include index in fingerprint.
            
        Returns:
            Dictionary containing fingerprint and metadata.
        """
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            # Sample if necessary
            original_size = len(df)
            sampled_df = self._sample_dataframe(df)
            
            # Prepare data for hashing
            data_for_hash = []
            
            # Add column information
            columns_info = []
            for col in sampled_df.columns:
                col_info = {
                    'name': str(col),
                    'dtype': str(sampled_df[col].dtype),
                    'null_count': int(sampled_df[col].isnull().sum())
                }
                columns_info.append(col_info)
            
            # Sort columns for consistency if enabled
            if self.sort_for_consistency:
                columns_info.sort(key=lambda x: x['name'])
                df_to_hash = sampled_df.reindex(sorted(sampled_df.columns), axis=1)
            else:
                df_to_hash = sampled_df
            
            # Include index if requested
            if include_index:
                data_for_hash.append(self._normalize_data(str(df_to_hash.index.tolist())))
            
            # Process data in chunks for memory efficiency
            for chunk_start in range(0, len(df_to_hash), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(df_to_hash))
                chunk = df_to_hash.iloc[chunk_start:chunk_end]
                
                # Convert chunk to normalized string representation
                for _, row in chunk.iterrows():
                    row_data = []
                    for col in (sorted(chunk.columns) if self.sort_for_consistency else chunk.columns):
                        row_data.append(self._normalize_data(row[col]))
                    data_for_hash.append(b'|'.join(row_data))
            
            # Sort rows for consistency if enabled
            if self.sort_for_consistency:
                data_for_hash.sort()
            
            # Compute hash
            hasher = self._hash_func()
            for data_bytes in data_for_hash:
                hasher.update(data_bytes)
            
            content_hash = hasher.hexdigest()
            
            # Prepare metadata
            metadata = {
                'fingerprint_algorithm': self.hash_algorithm,
                'content_hash': content_hash,
                'original_shape': df.shape,
                'sampled_shape': sampled_df.shape,
                'sample_ratio': len(sampled_df) / len(df) if len(df) > 0 else 0,
                'columns': columns_info,
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'fingerprint_timestamp': datetime.now().isoformat(),
                'included_index': include_index,
                'normalized_whitespace': self.normalize_whitespace,
                'sorted_for_consistency': self.sort_for_consistency
            }
            
            logger.debug(f"Generated DataFrame fingerprint: {content_hash[:16]}... "
                        f"(shape: {df.shape}, sampled: {sampled_df.shape})")
            
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to fingerprint DataFrame: {e}"
            if not self.fail_silently:
                raise CaptureError(error_msg) from e
            logger.warning(error_msg)
            return {'error': error_msg, 'fingerprint_timestamp': datetime.now().isoformat()}
    
    def fingerprint_numpy_array(self, arr, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Generate fingerprint for a numpy array.
        
        Args:
            arr: numpy array to fingerprint.
            include_metadata: Whether to include array metadata.
            
        Returns:
            Dictionary containing fingerprint and metadata.
        """
        try:
            import numpy as np
            
            if not isinstance(arr, np.ndarray):
                raise ValueError("Input must be a numpy array")
            
            # Sample if array is too large
            original_shape = arr.shape
            total_elements = arr.size
            
            if total_elements > self.max_sample_size:
                # Flatten and sample systematically
                flat_arr = arr.flatten()
                step = total_elements // self.max_sample_size
                sampled_indices = list(range(0, total_elements, step))[:self.max_sample_size]
                sampled_arr = flat_arr[sampled_indices]
            else:
                sampled_arr = arr.flatten()
            
            # Sort for consistency if enabled
            if self.sort_for_consistency and arr.dtype.kind in ['i', 'f', 'U', 'S']:  # numeric or string types
                sampled_arr = np.sort(sampled_arr)
            
            # Convert to bytes for hashing
            if arr.dtype.kind in ['U', 'S']:  # String types
                data_bytes = b'|'.join([self._normalize_data(str(x)) for x in sampled_arr])
            else:
                # For numeric types, convert to string with consistent precision
                if arr.dtype.kind == 'f':  # float types
                    str_data = '|'.join([f"{x:.10g}" for x in sampled_arr])
                else:
                    str_data = '|'.join([str(x) for x in sampled_arr])
                data_bytes = str_data.encode('utf-8')
            
            # Compute hash
            hasher = self._hash_func()
            hasher.update(data_bytes)
            content_hash = hasher.hexdigest()
            
            # Prepare metadata
            metadata = {
                'fingerprint_algorithm': self.hash_algorithm,
                'content_hash': content_hash,
                'original_shape': original_shape,
                'original_size': total_elements,
                'sampled_size': len(sampled_arr),
                'sample_ratio': len(sampled_arr) / total_elements if total_elements > 0 else 0,
                'dtype': str(arr.dtype),
                'fingerprint_timestamp': datetime.now().isoformat(),
                'sorted_for_consistency': self.sort_for_consistency
            }
            
            if include_metadata:
                try:
                    metadata.update({
                        'memory_usage_mb': round(arr.nbytes / 1024 / 1024, 2),
                        'min_value': float(np.min(arr)) if arr.dtype.kind in ['i', 'f'] else None,
                        'max_value': float(np.max(arr)) if arr.dtype.kind in ['i', 'f'] else None,
                        'mean_value': float(np.mean(arr)) if arr.dtype.kind in ['i', 'f'] else None
                    })
                except Exception as stats_error:
                    logger.debug(f"Could not compute array statistics: {stats_error}")
            
            logger.debug(f"Generated numpy array fingerprint: {content_hash[:16]}... "
                        f"(shape: {original_shape}, sampled: {len(sampled_arr)})")
            
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to fingerprint numpy array: {e}"
            if not self.fail_silently:
                raise CaptureError(error_msg) from e
            logger.warning(error_msg)
            return {'error': error_msg, 'fingerprint_timestamp': datetime.now().isoformat()}
    
    def fingerprint_file(self, file_path: str, file_format: str = None) -> Dict[str, Any]:
        """
        Generate fingerprint for a file-based dataset.
        
        Args:
            file_path: Path to the dataset file.
            file_format: Format of the file ('csv', 'json', 'parquet', etc.). Auto-detected if None.
            
        Returns:
            Dictionary containing fingerprint and metadata.
        """
        try:
            import pandas as pd
            from pathlib import Path
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Auto-detect format if not specified
            if file_format is None:
                file_format = file_path.suffix.lower().lstrip('.')
            
            # Load data based on format
            if file_format in ['csv', 'tsv']:
                df = pd.read_csv(file_path, nrows=self.max_sample_size)
            elif file_format == 'json':
                df = pd.read_json(file_path, lines=file_format=='jsonl').head(self.max_sample_size)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path).head(self.max_sample_size)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, nrows=self.max_sample_size)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Generate fingerprint for the loaded DataFrame
            fingerprint_result = self.fingerprint_dataframe(df)
            
            # Add file-specific metadata
            file_stats = file_path.stat()
            fingerprint_result.update({
                'source_file': str(file_path),
                'file_format': file_format,
                'file_size_mb': round(file_stats.st_size / 1024 / 1024, 2),
                'file_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'is_sample': len(df) >= self.max_sample_size
            })
            
            logger.debug(f"Generated file fingerprint for {file_path}: "
                        f"{fingerprint_result.get('content_hash', 'N/A')[:16]}...")
            
            return fingerprint_result
            
        except Exception as e:
            error_msg = f"Failed to fingerprint file {file_path}: {e}"
            if not self.fail_silently:
                raise CaptureError(error_msg) from e
            logger.warning(error_msg)
            return {'error': error_msg, 'source_file': str(file_path), 
                    'fingerprint_timestamp': datetime.now().isoformat()}
    
    def fingerprint_dict_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate fingerprint for dictionary-based data.
        
        Args:
            data: Dictionary containing data to fingerprint.
            
        Returns:
            Dictionary containing fingerprint and metadata.
        """
        try:
            import json
            
            # Sort keys for consistency if enabled
            if self.sort_for_consistency:
                sorted_data = dict(sorted(data.items()))
            else:
                sorted_data = data
            
            # Convert to JSON string for hashing
            json_str = json.dumps(sorted_data, sort_keys=self.sort_for_consistency, 
                                default=str, separators=(',', ':'))
            
            # Normalize whitespace if enabled
            if self.normalize_whitespace:
                json_str = ' '.join(json_str.split())
            
            # Compute hash
            hasher = self._hash_func()
            hasher.update(json_str.encode('utf-8'))
            content_hash = hasher.hexdigest()
            
            # Prepare metadata
            metadata = {
                'fingerprint_algorithm': self.hash_algorithm,
                'content_hash': content_hash,
                'data_type': 'dictionary',
                'num_keys': len(data),
                'keys': list(data.keys())[:100],  # Limit to first 100 keys
                'fingerprint_timestamp': datetime.now().isoformat(),
                'sorted_for_consistency': self.sort_for_consistency,
                'normalized_whitespace': self.normalize_whitespace
            }
            
            logger.debug(f"Generated dictionary fingerprint: {content_hash[:16]}... "
                        f"({len(data)} keys)")
            
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to fingerprint dictionary data: {e}"
            if not self.fail_silently:
                raise CaptureError(error_msg) from e
            logger.warning(error_msg)
            return {'error': error_msg, 'fingerprint_timestamp': datetime.now().isoformat()}
    
    def fingerprint_dataset(self, dataset: Any, dataset_type: str = None) -> Dict[str, Any]:
        """
        Universal fingerprinting method that handles various dataset types.
        
        Args:
            dataset: Dataset to fingerprint (DataFrame, array, file path, etc.).
            dataset_type: Type hint for the dataset ('dataframe', 'numpy', 'file', 'dict').
            
        Returns:
            Dictionary containing fingerprint and metadata.
        """
        try:
            # Auto-detect dataset type if not specified
            if dataset_type is None:
                if hasattr(dataset, 'iloc'):  # pandas DataFrame
                    dataset_type = 'dataframe'
                elif hasattr(dataset, 'shape') and hasattr(dataset, 'dtype'):  # numpy array
                    dataset_type = 'numpy'
                elif isinstance(dataset, (str, Path)):  # file path
                    dataset_type = 'file'
                elif isinstance(dataset, dict):  # dictionary
                    dataset_type = 'dict'
                else:
                    raise ValueError(f"Cannot auto-detect dataset type for: {type(dataset)}")
            
            # Route to appropriate fingerprinting method
            if dataset_type == 'dataframe':
                return self.fingerprint_dataframe(dataset)
            elif dataset_type == 'numpy':
                return self.fingerprint_numpy_array(dataset)
            elif dataset_type == 'file':
                return self.fingerprint_file(dataset)
            elif dataset_type == 'dict':
                return self.fingerprint_dict_data(dataset)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
                
        except Exception as e:
            error_msg = f"Failed to fingerprint dataset: {e}"
            if not self.fail_silently:
                raise CaptureError(error_msg) from e
            logger.warning(error_msg)
            return {'error': error_msg, 'fingerprint_timestamp': datetime.now().isoformat()}
    
    def compare_fingerprints(self, fingerprint1: Dict[str, Any], 
                           fingerprint2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two dataset fingerprints.
        
        Args:
            fingerprint1: First fingerprint to compare.
            fingerprint2: Second fingerprint to compare.
            
        Returns:
            Dictionary containing comparison results.
        """
        try:
            comparison = {
                'identical': False,
                'content_match': False,
                'algorithm_match': False,
                'differences': []
            }
            
            # Check if fingerprints have the required fields
            if 'content_hash' not in fingerprint1 or 'content_hash' not in fingerprint2:
                comparison['differences'].append('Missing content_hash in one or both fingerprints')
                return comparison
            
            # Compare algorithms
            algo1 = fingerprint1.get('fingerprint_algorithm', 'unknown')
            algo2 = fingerprint2.get('fingerprint_algorithm', 'unknown')
            comparison['algorithm_match'] = algo1 == algo2
            
            if not comparison['algorithm_match']:
                comparison['differences'].append(f'Different algorithms: {algo1} vs {algo2}')
            
            # Compare content hashes
            hash1 = fingerprint1['content_hash']
            hash2 = fingerprint2['content_hash']
            comparison['content_match'] = hash1 == hash2
            
            if not comparison['content_match']:
                comparison['differences'].append(f'Different content hashes: {hash1[:16]}... vs {hash2[:16]}...')
            
            # Check overall identity
            comparison['identical'] = comparison['content_match'] and comparison['algorithm_match']
            
            # Compare shapes/sizes if available
            shape1 = fingerprint1.get('original_shape') or fingerprint1.get('original_size')
            shape2 = fingerprint2.get('original_shape') or fingerprint2.get('original_size')
            
            if shape1 and shape2:
                if shape1 != shape2:
                    comparison['differences'].append(f'Different shapes/sizes: {shape1} vs {shape2}')
            
            logger.debug(f"Fingerprint comparison: identical={comparison['identical']}, "
                        f"content_match={comparison['content_match']}")
            
            return comparison
            
        except Exception as e:
            error_msg = f"Failed to compare fingerprints: {e}"
            if not self.fail_silently:
                raise CaptureError(error_msg) from e
            logger.warning(error_msg)
            return {'error': error_msg}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary for serialization.
        
        Returns:
            Dictionary containing fingerprinter configuration.
        """
        return {
            'max_sample_size': self.max_sample_size,
            'chunk_size': self.chunk_size,
            'hash_algorithm': self.hash_algorithm,
            'normalize_whitespace': self.normalize_whitespace,
            'sort_for_consistency': self.sort_for_consistency,
            'fail_silently': self.fail_silently
        }


class CaptureManager:
    """
    Unified manager that orchestrates all capture modules for ML experiment tracking.
    
    This class provides a simple interface to capture git information, environment metadata,
    and dataset fingerprints in a coordinated manner. It handles errors gracefully to ensure
    that failures in individual capture modules don't prevent the overall capture process.
    """
    
    def __init__(self, 
                 enable_git: bool = True,
                 enable_environment: bool = True,
                 enable_dataset: bool = True,
                 git_options: dict = None,
                 env_options: dict = None,
                 dataset_options: dict = None,
                 fail_silently: bool = True):
        """
        Initialize the CaptureManager with configurable capture components.
        
        Args:
            enable_git: Whether to capture git information
            enable_environment: Whether to capture environment metadata
            enable_dataset: Whether to enable dataset fingerprinting
            git_options: Options dict to pass to GitCapture constructor
            env_options: Options dict to pass to EnvironmentCapture constructor
            dataset_options: Options dict to pass to DatasetFingerprinter constructor
            fail_silently: Whether to suppress errors from individual capture modules
        """
        self.enable_git = enable_git
        self.enable_environment = enable_environment
        self.enable_dataset = enable_dataset
        self.fail_silently = fail_silently
        
        # Initialize capture modules with provided options
        self.git_capture = None
        self.env_capture = None
        self.dataset_fingerprinter = None
        
        if self.enable_git:
            git_opts = git_options or {}
            git_opts.setdefault('fail_silently', fail_silently)
            try:
                self.git_capture = GitCapture(**git_opts)
            except Exception as e:
                if not fail_silently:
                    raise
                logger.warning(f"Failed to initialize GitCapture: {e}")
        
        if self.enable_environment:
            env_opts = env_options or {}
            env_opts.setdefault('fail_silently', fail_silently)
            try:
                self.env_capture = EnvironmentCapture(**env_opts)
            except Exception as e:
                if not fail_silently:
                    raise
                logger.warning(f"Failed to initialize EnvironmentCapture: {e}")
        
        if self.enable_dataset:
            dataset_opts = dataset_options or {}
            dataset_opts.setdefault('fail_silently', fail_silently)
            try:
                self.dataset_fingerprinter = DatasetFingerprinter(**dataset_opts)
            except Exception as e:
                if not fail_silently:
                    raise
                logger.warning(f"Failed to initialize DatasetFingerprinter: {e}")
    
    def capture_all(self, datasets=None, lightweight=True):
        """
        Capture all enabled information in a single coordinated operation.
        
        Args:
            datasets: Optional list/dict of datasets to fingerprint
            lightweight: If True, use faster but less comprehensive capture options
            
        Returns:
            dict: Combined capture data with keys 'git', 'environment', 'datasets', 'metadata'
        """
        capture_data = {
            'metadata': {
                'capture_timestamp': datetime.utcnow().isoformat() + 'Z',
                'capture_manager_version': '1.0.0',
                'enabled_modules': {
                    'git': self.enable_git and self.git_capture is not None,
                    'environment': self.enable_environment and self.env_capture is not None,
                    'dataset': self.enable_dataset and self.dataset_fingerprinter is not None
                },
                'lightweight_mode': lightweight
            }
        }
        
        # Capture git information
        if self.enable_git and self.git_capture is not None:
            try:
                git_data = self.git_capture.capture()
                capture_data['git'] = git_data
                logger.debug("Successfully captured git information")
            except Exception as e:
                if not self.fail_silently:
                    raise
                logger.warning(f"Failed to capture git information: {e}")
                capture_data['git'] = {'error': str(e), 'capture_failed': True}
        
        # Capture environment information
        if self.enable_environment and self.env_capture is not None:
            try:
                env_data = self.env_capture.capture()
                capture_data['environment'] = env_data
                logger.debug("Successfully captured environment information")
            except Exception as e:
                if not self.fail_silently:
                    raise
                logger.warning(f"Failed to capture environment information: {e}")
                capture_data['environment'] = {'error': str(e), 'capture_failed': True}
        
        # Capture dataset fingerprints
        if self.enable_dataset and self.dataset_fingerprinter is not None and datasets is not None:
            try:
                dataset_data = self._capture_datasets(datasets, lightweight)
                capture_data['datasets'] = dataset_data
                logger.debug(f"Successfully captured {len(dataset_data)} dataset fingerprints")
            except Exception as e:
                if not self.fail_silently:
                    raise
                logger.warning(f"Failed to capture dataset information: {e}")
                capture_data['datasets'] = {'error': str(e), 'capture_failed': True}
        
        return capture_data
    
    def _capture_datasets(self, datasets, lightweight=True):
        """
        Capture fingerprints for multiple datasets.
        
        Args:
            datasets: List of datasets, dict mapping names to datasets, or single dataset
            lightweight: Whether to use fast sampling for large datasets
            
        Returns:
            dict: Dataset fingerprints keyed by dataset name/index
        """
        dataset_fingerprints = {}
        
        # Handle different input formats
        if isinstance(datasets, dict):
            # Dict mapping names to datasets
            dataset_items = datasets.items()
        elif isinstance(datasets, (list, tuple)):
            # List of datasets - use indices as names
            dataset_items = enumerate(datasets)
        else:
            # Single dataset
            dataset_items = [('dataset', datasets)]
        
        for name, dataset in dataset_items:
            try:
                fingerprint = self.dataset_fingerprinter.fingerprint_dataset(dataset)
                dataset_fingerprints[str(name)] = fingerprint
            except Exception as e:
                if not self.fail_silently:
                    raise
                logger.warning(f"Failed to fingerprint dataset '{name}': {e}")
                dataset_fingerprints[str(name)] = {'error': str(e), 'fingerprint_failed': True}
        
        return dataset_fingerprints
    
    def capture_git_only(self):
        """Convenience method to capture only git information."""
        if not self.enable_git or self.git_capture is None:
            return None
        try:
            return self.git_capture.capture()
        except Exception as e:
            if not self.fail_silently:
                raise
            logger.warning(f"Failed to capture git information: {e}")
            return {'error': str(e), 'capture_failed': True}
    
    def capture_environment_only(self):
        """Convenience method to capture only environment information."""
        if not self.enable_environment or self.env_capture is None:
            return None
        try:
            return self.env_capture.capture()
        except Exception as e:
            if not self.fail_silently:
                raise
            logger.warning(f"Failed to capture environment information: {e}")
            return {'error': str(e), 'capture_failed': True}
    
    def fingerprint_dataset(self, dataset, name=None):
        """Convenience method to fingerprint a single dataset."""
        if not self.enable_dataset or self.dataset_fingerprinter is None:
            return None
        try:
            fingerprint = self.dataset_fingerprinter.fingerprint_dataset(dataset)
            if name:
                return {name: fingerprint}
            return fingerprint
        except Exception as e:
            if not self.fail_silently:
                raise
            logger.warning(f"Failed to fingerprint dataset: {e}")
            return {'error': str(e), 'fingerprint_failed': True}
    
    def get_configuration(self):
        """Get the current configuration of all capture modules."""
        config = {
            'capture_manager': {
                'enabled_modules': {
                    'git': self.enable_git,
                    'environment': self.enable_environment,
                    'dataset': self.enable_dataset
                },
                'fail_silently': self.fail_silently
            }
        }
        
        if self.git_capture:
            config['git_capture'] = {
                'command_timeout': self.git_capture.command_timeout,
                'capture_untracked': self.git_capture.capture_untracked,
                'capture_diff_summary': self.git_capture.capture_diff_summary,
                'fail_silently': self.git_capture.fail_silently
            }
        
        if self.env_capture:
            config['environment_capture'] = self.env_capture.get_configuration()
        
        if self.dataset_fingerprinter:
            config['dataset_fingerprinter'] = self.dataset_fingerprinter.to_dict()
        
        return config


# Export main classes for easier imports
__all__ = [
    'GitCapture', 'GitCaptureError',
    'EnvironmentCapture', 'EnvironmentCaptureError', 
    'DatasetFingerprinter',
    'CaptureManager'
] 