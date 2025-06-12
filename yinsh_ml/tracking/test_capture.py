"""
Unit tests for the capture module functionality.
"""

import unittest
import tempfile
import shutil
import subprocess
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from .capture import GitCapture, GitCaptureError, EnvironmentCapture, EnvironmentCaptureError, DatasetFingerprinter, CaptureManager


class TestGitCapture(unittest.TestCase):
    """Test cases for GitCapture class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.git_repo_path = Path(self.temp_dir) / "test_repo"
        self.git_repo_path.mkdir()
        
        # Initialize a test git repository
        os.chdir(self.git_repo_path)
        subprocess.run(['git', 'init'], capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], capture_output=True)
        
        # Create initial commit
        test_file = self.git_repo_path / "test.txt"
        test_file.write_text("Initial content")
        subprocess.run(['git', 'add', 'test.txt'], capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], capture_output=True)
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir("/")  # Change away from temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_git_capture_basic_functionality(self):
        """Test basic git capture functionality."""
        capture = GitCapture(repository_path=str(self.git_repo_path))
        git_info = capture.capture()
        
        self.assertNotEqual(git_info['commit'], 'unknown')
        self.assertNotEqual(git_info['commit_short'], 'unknown')
        self.assertIn(git_info['branch'], ['master', 'main'])  # Handle both git default branches
        self.assertTrue(git_info['working_directory_clean'])
        self.assertEqual(git_info['commit_message'], 'Initial commit')
        self.assertEqual(git_info['author'], 'Test User <test@example.com>')
        
    def test_git_capture_with_modifications(self):
        """Test git capture with uncommitted changes."""
        # Modify a file
        test_file = self.git_repo_path / "test.txt"
        test_file.write_text("Modified content")
        
        # Add untracked file
        new_file = self.git_repo_path / "new.txt"
        new_file.write_text("New file content")
        
        capture = GitCapture(repository_path=str(self.git_repo_path), capture_untracked=True)
        git_info = capture.capture()
        
        self.assertFalse(git_info['working_directory_clean'])
        self.assertIn('test.txt', git_info['modified_files'])
        self.assertIn('new.txt', git_info['untracked_files'])
        
    def test_git_capture_with_diff_summary(self):
        """Test git capture with diff summary enabled."""
        # Modify a file
        test_file = self.git_repo_path / "test.txt"
        test_file.write_text("Modified content\nNew line added")
        
        capture = GitCapture(
            repository_path=str(self.git_repo_path), 
            capture_diff_summary=True
        )
        git_info = capture.capture()
        
        self.assertIn('diff_summary', git_info)
        self.assertGreater(git_info['diff_summary']['lines_added'], 0)
        self.assertGreater(git_info['diff_summary']['files_changed'], 0)
        
    def test_git_capture_non_repository(self):
        """Test git capture in non-git directory."""
        non_git_path = Path(self.temp_dir) / "non_git"
        non_git_path.mkdir()
        
        capture = GitCapture(repository_path=str(non_git_path))
        git_info = capture.capture()
        
        self.assertEqual(git_info['commit'], 'unknown')
        self.assertEqual(git_info['branch'], 'unknown')
        self.assertTrue(git_info['working_directory_clean'])
        
    def test_git_capture_fail_silently_false(self):
        """Test git capture with fail_silently=False."""
        non_git_path = Path(self.temp_dir) / "non_git"
        non_git_path.mkdir()
        
        capture = GitCapture(repository_path=str(non_git_path), fail_silently=False)
        
        # Should not raise exception for non-git directory (this is expected behavior)
        git_info = capture.capture()
        self.assertEqual(git_info['commit'], 'unknown')
        
    @patch('subprocess.run')
    def test_git_capture_command_timeout(self, mock_run):
        """Test handling of git command timeouts."""
        mock_run.side_effect = subprocess.TimeoutExpired(['git', 'rev-parse', 'HEAD'], 30)
        
        capture = GitCapture(repository_path=str(self.git_repo_path))
        result = capture._run_git_command(['rev-parse', 'HEAD'])
        
        self.assertEqual(result, "")
        
    @patch('subprocess.run')
    def test_git_capture_command_not_found(self, mock_run):
        """Test handling when git command is not found."""
        mock_run.side_effect = FileNotFoundError("git command not found")
        
        capture = GitCapture(repository_path=str(self.git_repo_path))
        result = capture._run_git_command(['rev-parse', 'HEAD'])
        
        self.assertEqual(result, "")
        
    def test_individual_methods(self):
        """Test individual git capture methods."""
        capture = GitCapture(repository_path=str(self.git_repo_path))
        
        # Test individual methods
        self.assertTrue(capture.is_git_repository())
        self.assertNotEqual(capture.get_commit_hash(), 'unknown')
        self.assertNotEqual(capture.get_commit_hash(short=True), 'unknown')
        self.assertIn(capture.get_branch_name(), ['master', 'main'])  # Git default branch
        self.assertEqual(capture.get_commit_message(), 'Initial commit')
        self.assertEqual(capture.get_commit_author(), 'Test User <test@example.com>')
        self.assertIsNotNone(capture.get_commit_timestamp())
        
        # Test status
        status = capture.get_working_directory_status()
        self.assertTrue(status['is_clean'])
        self.assertEqual(len(status['modified_files']), 0)
        self.assertEqual(len(status['untracked_files']), 0)
        
    def test_to_dict_alias(self):
        """Test that to_dict() is an alias for capture()."""
        capture = GitCapture(repository_path=str(self.git_repo_path))
        
        dict_result = capture.to_dict()
        capture_result = capture.capture()
        
        # Results should be identical (within timestamp tolerance)
        self.assertEqual(dict_result['commit'], capture_result['commit'])
        self.assertEqual(dict_result['branch'], capture_result['branch'])


class TestEnvironmentCapture(unittest.TestCase):
    """Test cases for EnvironmentCapture class."""
    
    def test_environment_capture_basic_functionality(self):
        """Test basic environment capture functionality."""
        capture = EnvironmentCapture()
        env_info = capture.capture()
        
        # Check basic structure
        self.assertIn('capture_timestamp', env_info)
        self.assertIn('python_version', env_info)
        self.assertIn('system', env_info)
        self.assertIn('environment_variables', env_info)
        self.assertIn('installed_packages', env_info)
        self.assertIn('frameworks', env_info)
        
        # Check python version format
        python_version = env_info['python_version']
        self.assertRegex(python_version, r'\d+\.\d+\.\d+')
        
    def test_environment_capture_with_disabled_components(self):
        """Test environment capture with specific components disabled."""
        capture = EnvironmentCapture(
            capture_packages=False,
            capture_env_vars=False,
            capture_system_info=False,
            capture_frameworks=False
        )
        env_info = capture.capture()
        
        # Should have basic info but empty collections for disabled components
        self.assertIn('capture_timestamp', env_info)
        self.assertIn('python_version', env_info)
        self.assertEqual(env_info.get('system', {}), {})
        self.assertEqual(env_info.get('environment_variables', {}), {})
        self.assertEqual(env_info.get('installed_packages', {}), {})
        self.assertEqual(env_info.get('frameworks', {}), {})
        
    def test_python_version_capture(self):
        """Test Python version capture."""
        capture = EnvironmentCapture()
        version = capture.get_python_version()
        
        # Should be in format X.Y.Z
        self.assertRegex(version, r'\d+\.\d+\.\d+')
        
    def test_system_information_capture(self):
        """Test system information capture."""
        capture = EnvironmentCapture()
        system_info = capture.get_system_information()
        
        # Check for expected fields
        expected_fields = [
            'platform', 'platform_release', 'platform_version',
            'architecture', 'hostname', 'python_implementation', 'python_version'
        ]
        
        for field in expected_fields:
            self.assertIn(field, system_info)
            self.assertIsInstance(system_info[field], str)
            self.assertNotEqual(system_info[field], '')
            
    def test_environment_variables_capture(self):
        """Test environment variables capture."""
        capture = EnvironmentCapture()
        env_vars = capture.get_environment_variables()
        
        # Should be a dictionary
        self.assertIsInstance(env_vars, dict)
        
        # Check that PATH is captured if available
        if 'PATH' in os.environ:
            self.assertIn('PATH', env_vars)
            
    def test_sensitive_environment_variable_filtering(self):
        """Test that sensitive environment variables are filtered out."""
        # Set a test sensitive environment variable
        os.environ['TEST_SECRET_KEY'] = 'sensitive_value'
        os.environ['TEST_API_KEY'] = 'api_value'
        
        try:
            capture = EnvironmentCapture(custom_env_vars=['TEST_SECRET_KEY', 'TEST_API_KEY'])
            env_vars = capture.get_environment_variables()
            
            # These should be filtered out due to sensitive keywords
            self.assertNotIn('TEST_SECRET_KEY', env_vars)
            self.assertNotIn('TEST_API_KEY', env_vars)
            
        finally:
            # Clean up test environment variables
            os.environ.pop('TEST_SECRET_KEY', None)
            os.environ.pop('TEST_API_KEY', None)
            
    def test_custom_environment_variables(self):
        """Test capturing custom environment variables."""
        # Set a test non-sensitive environment variable
        os.environ['TEST_CUSTOM_VAR'] = 'custom_value'
        
        try:
            capture = EnvironmentCapture(custom_env_vars=['TEST_CUSTOM_VAR'])
            env_vars = capture.get_environment_variables()
            
            # Should be captured since it's not sensitive
            self.assertIn('TEST_CUSTOM_VAR', env_vars)
            self.assertEqual(env_vars['TEST_CUSTOM_VAR'], 'custom_value')
            
        finally:
            # Clean up test environment variable
            os.environ.pop('TEST_CUSTOM_VAR', None)
            
    def test_installed_packages_capture(self):
        """Test installed packages capture."""
        capture = EnvironmentCapture()
        packages = capture.get_installed_packages()
        
        # Should be a dictionary
        self.assertIsInstance(packages, dict)
        
        # Should contain some packages (at least the test dependencies)
        self.assertGreater(len(packages), 0)
        
        # Check that package versions are strings
        for name, version in packages.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(version, str)
            self.assertNotEqual(version, '')
            
    def test_framework_information_capture(self):
        """Test framework information capture."""
        capture = EnvironmentCapture()
        frameworks = capture.get_framework_information()
        
        # Should be a dictionary
        self.assertIsInstance(frameworks, dict)
        
        # Check for NumPy (should be available in test environment)
        if 'numpy' in frameworks:
            self.assertIn('version', frameworks['numpy'])
            self.assertIsInstance(frameworks['numpy']['version'], str)
            
    def test_caching_functionality(self):
        """Test that caching works for expensive operations."""
        capture = EnvironmentCapture()
        
        # First call should populate cache
        packages1 = capture.get_installed_packages()
        system1 = capture.get_system_information()
        
        # Second call should use cache (same objects)
        packages2 = capture.get_installed_packages()
        system2 = capture.get_system_information()
        
        # Should be the same objects (cached)
        self.assertIs(packages1, packages2)
        self.assertIs(system1, system2)
        
        # Clear cache and try again
        capture.clear_cache()
        packages3 = capture.get_installed_packages()
        system3 = capture.get_system_information()
        
        # Should be different objects (new capture)
        self.assertIsNot(packages1, packages3)
        self.assertIsNot(system1, system3)
        
        # But content should be the same
        self.assertEqual(packages1, packages3)
        self.assertEqual(system1, system3)
        
    def test_fail_silently_false(self):
        """Test behavior when fail_silently is False."""
        capture = EnvironmentCapture(fail_silently=False)
        
        # Should not raise exceptions for normal operations
        env_info = capture.capture()
        self.assertIsInstance(env_info, dict)
        
    @patch('yinsh_ml.tracking.capture.pkg_resources')
    def test_package_capture_error_handling(self, mock_pkg_resources):
        """Test error handling during package capture."""
        mock_pkg_resources.working_set = None
        mock_pkg_resources.side_effect = Exception("Package error")
        
        # With fail_silently=True (default)
        capture = EnvironmentCapture()
        packages = capture.get_installed_packages()
        self.assertEqual(packages, {})
        
        # With fail_silently=False
        capture_strict = EnvironmentCapture(fail_silently=False)
        with self.assertRaises(EnvironmentCaptureError):
            capture_strict.get_installed_packages()
            
    @patch('yinsh_ml.tracking.capture.platform')
    def test_system_info_error_handling(self, mock_platform):
        """Test error handling during system info capture."""
        mock_platform.system.side_effect = Exception("System error")
        
        # With fail_silently=True (default)
        capture = EnvironmentCapture()
        system_info = capture.get_system_information()
        self.assertEqual(system_info, {})
        
        # With fail_silently=False
        capture_strict = EnvironmentCapture(fail_silently=False)
        with self.assertRaises(EnvironmentCaptureError):
            capture_strict.get_system_information()
            
    def test_to_dict_alias(self):
        """Test that to_dict() is an alias for capture()."""
        capture = EnvironmentCapture()
        
        dict_result = capture.to_dict()
        capture_result = capture.capture()
        
        # Results should be identical (within timestamp tolerance)
        self.assertEqual(dict_result['python_version'], capture_result['python_version'])
        self.assertEqual(dict_result.keys(), capture_result.keys())


class TestDatasetFingerprinter(unittest.TestCase):
    """Test cases for DatasetFingerprinter class."""
    
    def setUp(self):
        """Set up test environment."""
        self.fingerprinter = DatasetFingerprinter(max_sample_size=100, chunk_size=10)
        
    def test_dataset_fingerprinter_initialization(self):
        """Test DatasetFingerprinter initialization."""
        fingerprinter = DatasetFingerprinter(
            max_sample_size=500,
            chunk_size=50,
            hash_algorithm='sha256',
            normalize_whitespace=True,
            sort_for_consistency=True,
            fail_silently=False
        )
        
        self.assertEqual(fingerprinter.max_sample_size, 500)
        self.assertEqual(fingerprinter.chunk_size, 50)
        self.assertEqual(fingerprinter.hash_algorithm, 'sha256')
        self.assertTrue(fingerprinter.normalize_whitespace)
        self.assertTrue(fingerprinter.sort_for_consistency)
        self.assertFalse(fingerprinter.fail_silently)
        
    def test_invalid_hash_algorithm(self):
        """Test initialization with invalid hash algorithm."""
        with self.assertRaises(ValueError):
            DatasetFingerprinter(hash_algorithm='invalid_algorithm')
            
    def test_normalize_data(self):
        """Test data normalization functionality."""
        # Test string normalization
        result = self.fingerprinter._normalize_data("  hello   world  ")
        self.assertEqual(result, b"hello world")
        
        # Test numeric normalization
        result = self.fingerprinter._normalize_data(42)
        self.assertEqual(result, b"42")
        
        # Test None normalization
        result = self.fingerprinter._normalize_data(None)
        self.assertEqual(result, b"NULL")
        
        # Test bytes passthrough
        result = self.fingerprinter._normalize_data(b"test")
        self.assertEqual(result, b"test")
        
    def test_fingerprint_dataframe_basic(self):
        """Test basic DataFrame fingerprinting."""
        import pandas as pd
        
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = self.fingerprinter.fingerprint_dataframe(df)
        
        # Check required fields
        self.assertIn('content_hash', result)
        self.assertIn('fingerprint_algorithm', result)
        self.assertIn('original_shape', result)
        self.assertIn('columns', result)
        
        # Check data integrity
        self.assertEqual(result['original_shape'], (5, 3))
        self.assertEqual(result['fingerprint_algorithm'], 'sha256')
        self.assertEqual(len(result['columns']), 3)
        
    def test_fingerprint_dataframe_consistency(self):
        """Test that identical DataFrames produce identical fingerprints."""
        import pandas as pd
        
        df1 = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        
        result1 = self.fingerprinter.fingerprint_dataframe(df1)
        result2 = self.fingerprinter.fingerprint_dataframe(df2)
        
        self.assertEqual(result1['content_hash'], result2['content_hash'])
        
    def test_fingerprint_dataframe_with_different_order(self):
        """Test DataFrame fingerprinting with different row order."""
        import pandas as pd
        
        df1 = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'x': [3, 1, 2], 'y': ['c', 'a', 'b']})
        
        # With sort_for_consistency=True, should be identical
        fingerprinter_sorted = DatasetFingerprinter(sort_for_consistency=True)
        result1 = fingerprinter_sorted.fingerprint_dataframe(df1)
        result2 = fingerprinter_sorted.fingerprint_dataframe(df2)
        self.assertEqual(result1['content_hash'], result2['content_hash'])
        
        # With sort_for_consistency=False, should be different
        fingerprinter_unsorted = DatasetFingerprinter(sort_for_consistency=False)
        result3 = fingerprinter_unsorted.fingerprint_dataframe(df1)
        result4 = fingerprinter_unsorted.fingerprint_dataframe(df2)
        self.assertNotEqual(result3['content_hash'], result4['content_hash'])
        
    def test_fingerprint_numpy_array(self):
        """Test numpy array fingerprinting."""
        import numpy as np
        
        arr = np.array([1, 2, 3, 4, 5])
        result = self.fingerprinter.fingerprint_numpy_array(arr)
        
        # Check required fields
        self.assertIn('content_hash', result)
        self.assertIn('original_shape', result)
        self.assertIn('dtype', result)
        
        # Check data integrity
        self.assertEqual(result['original_shape'], (5,))
        self.assertEqual(result['original_size'], 5)
        
    def test_fingerprint_numpy_array_consistency(self):
        """Test numpy array fingerprinting consistency."""
        import numpy as np
        
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        
        result1 = self.fingerprinter.fingerprint_numpy_array(arr1)
        result2 = self.fingerprinter.fingerprint_numpy_array(arr2)
        
        self.assertEqual(result1['content_hash'], result2['content_hash'])
        
    def test_fingerprint_numpy_array_large_sampling(self):
        """Test numpy array fingerprinting with large arrays (sampling)."""
        import numpy as np
        
        # Create array larger than max_sample_size
        arr = np.random.rand(200)  # Larger than max_sample_size=100
        result = self.fingerprinter.fingerprint_numpy_array(arr)
        
        self.assertEqual(result['original_size'], 200)
        self.assertEqual(result['sampled_size'], 100)  # Should be sampled
        self.assertLess(result['sample_ratio'], 1.0)
        
    def test_fingerprint_dict_data(self):
        """Test dictionary data fingerprinting."""
        data = {
            'key1': 'value1',
            'key2': [1, 2, 3],
            'key3': {'nested': 'dict'}
        }
        
        result = self.fingerprinter.fingerprint_dict_data(data)
        
        # Check required fields
        self.assertIn('content_hash', result)
        self.assertIn('num_keys', result)
        self.assertIn('keys', result)
        
        # Check data integrity
        self.assertEqual(result['num_keys'], 3)
        self.assertEqual(set(result['keys']), {'key1', 'key2', 'key3'})
        
    def test_fingerprint_dict_consistency(self):
        """Test dictionary fingerprinting consistency."""
        data1 = {'a': 1, 'b': 2}
        data2 = {'b': 2, 'a': 1}  # Different order
        
        result1 = self.fingerprinter.fingerprint_dict_data(data1)
        result2 = self.fingerprinter.fingerprint_dict_data(data2)
        
        # Should be identical due to sort_for_consistency=True
        self.assertEqual(result1['content_hash'], result2['content_hash'])
        
    def test_fingerprint_dataset_auto_detection(self):
        """Test universal fingerprint_dataset method with auto-detection."""
        import pandas as pd
        import numpy as np
        
        # Test DataFrame auto-detection
        df = pd.DataFrame({'x': [1, 2, 3]})
        result_df = self.fingerprinter.fingerprint_dataset(df)
        self.assertIn('original_shape', result_df)
        
        # Test numpy array auto-detection
        arr = np.array([1, 2, 3])
        result_arr = self.fingerprinter.fingerprint_dataset(arr)
        self.assertIn('original_shape', result_arr)
        
        # Test dictionary auto-detection
        data = {'key': 'value'}
        result_dict = self.fingerprinter.fingerprint_dataset(data)
        self.assertIn('num_keys', result_dict)
        
    def test_compare_fingerprints(self):
        """Test fingerprint comparison functionality."""
        import pandas as pd
        
        df1 = pd.DataFrame({'x': [1, 2, 3]})
        df2 = pd.DataFrame({'x': [1, 2, 3]})
        df3 = pd.DataFrame({'x': [4, 5, 6]})
        
        fp1 = self.fingerprinter.fingerprint_dataframe(df1)
        fp2 = self.fingerprinter.fingerprint_dataframe(df2)
        fp3 = self.fingerprinter.fingerprint_dataframe(df3)
        
        # Compare identical fingerprints
        comparison_identical = self.fingerprinter.compare_fingerprints(fp1, fp2)
        self.assertTrue(comparison_identical['identical'])
        self.assertTrue(comparison_identical['content_match'])
        self.assertTrue(comparison_identical['algorithm_match'])
        
        # Compare different fingerprints
        comparison_different = self.fingerprinter.compare_fingerprints(fp1, fp3)
        self.assertFalse(comparison_different['identical'])
        self.assertFalse(comparison_different['content_match'])
        self.assertTrue(comparison_different['algorithm_match'])
        
    def test_sample_dataframe(self):
        """Test DataFrame sampling functionality."""
        import pandas as pd
        
        # Create DataFrame larger than max_sample_size
        df = pd.DataFrame({'x': range(200)})  # Larger than max_sample_size=100
        
        sampled_df = self.fingerprinter._sample_dataframe(df, max_rows=50)
        self.assertEqual(len(sampled_df), 50)
        
        # Test with DataFrame smaller than max_sample_size
        small_df = pd.DataFrame({'x': range(10)})
        sampled_small = self.fingerprinter._sample_dataframe(small_df)
        self.assertEqual(len(sampled_small), 10)  # Should return original
        
    def test_configuration_to_dict(self):
        """Test configuration serialization."""
        config = self.fingerprinter.to_dict()
        
        expected_keys = {
            'max_sample_size', 'chunk_size', 'hash_algorithm',
            'normalize_whitespace', 'sort_for_consistency', 'fail_silently'
        }
        
        self.assertEqual(set(config.keys()), expected_keys)
        self.assertEqual(config['max_sample_size'], 100)
        self.assertEqual(config['hash_algorithm'], 'sha256')
        
    def test_error_handling_fail_silently_true(self):
        """Test error handling with fail_silently=True."""
        fingerprinter = DatasetFingerprinter(fail_silently=True)
        
        # Test with invalid input
        result = fingerprinter.fingerprint_dataframe("not_a_dataframe")
        self.assertIn('error', result)
        
    def test_error_handling_fail_silently_false(self):
        """Test error handling with fail_silently=False."""
        fingerprinter = DatasetFingerprinter(fail_silently=False)
        
        # Test with invalid input
        with self.assertRaises(Exception):
            fingerprinter.fingerprint_dataframe("not_a_dataframe")
            
    def test_different_hash_algorithms(self):
        """Test different hash algorithms produce different results."""
        import pandas as pd
        
        df = pd.DataFrame({'x': [1, 2, 3]})
        
        fp_sha256 = DatasetFingerprinter(hash_algorithm='sha256').fingerprint_dataframe(df)
        fp_md5 = DatasetFingerprinter(hash_algorithm='md5').fingerprint_dataframe(df)
        
        self.assertNotEqual(fp_sha256['content_hash'], fp_md5['content_hash'])
        self.assertEqual(fp_sha256['fingerprint_algorithm'], 'sha256')
        self.assertEqual(fp_md5['fingerprint_algorithm'], 'md5')
        
    def test_whitespace_normalization(self):
        """Test whitespace normalization behavior."""
        import pandas as pd
        
        df1 = pd.DataFrame({'text': ['hello  world', ' test ']})
        df2 = pd.DataFrame({'text': ['hello world', 'test']})
        
        # With normalization enabled
        fp_normalized = DatasetFingerprinter(normalize_whitespace=True).fingerprint_dataframe(df1)
        fp_clean = DatasetFingerprinter(normalize_whitespace=True).fingerprint_dataframe(df2)
        self.assertEqual(fp_normalized['content_hash'], fp_clean['content_hash'])
        
        # With normalization disabled
        fp_no_norm1 = DatasetFingerprinter(normalize_whitespace=False).fingerprint_dataframe(df1)
        fp_no_norm2 = DatasetFingerprinter(normalize_whitespace=False).fingerprint_dataframe(df2)
        self.assertNotEqual(fp_no_norm1['content_hash'], fp_no_norm2['content_hash'])


class TestCaptureManager(unittest.TestCase):
    """Test cases for CaptureManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temp directory for git tests
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Initialize a git repository
        subprocess.run(['git', 'init'], capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], capture_output=True)
        
        # Create an initial commit
        with open('test.txt', 'w') as f:
            f.write('test')
        subprocess.run(['git', 'add', '.'], capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], capture_output=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_capture_manager_initialization(self):
        """Test CaptureManager initialization with different configurations."""
        # Test default initialization (all modules enabled)
        manager = CaptureManager()
        self.assertTrue(manager.enable_git)
        self.assertTrue(manager.enable_environment)
        self.assertTrue(manager.enable_dataset)
        self.assertIsNotNone(manager.git_capture)
        self.assertIsNotNone(manager.env_capture)
        self.assertIsNotNone(manager.dataset_fingerprinter)
        
        # Test selective module initialization
        manager_git_only = CaptureManager(enable_environment=False, enable_dataset=False)
        self.assertTrue(manager_git_only.enable_git)
        self.assertFalse(manager_git_only.enable_environment)
        self.assertFalse(manager_git_only.enable_dataset)
        self.assertIsNotNone(manager_git_only.git_capture)
        self.assertIsNone(manager_git_only.env_capture)
        self.assertIsNone(manager_git_only.dataset_fingerprinter)
    
    def test_capture_manager_with_options(self):
        """Test CaptureManager initialization with custom options."""
        git_opts = {'command_timeout': 10, 'capture_untracked': False}
        env_opts = {'capture_packages': False}
        dataset_opts = {'hash_algorithm': 'md5'}
        
        manager = CaptureManager(
            git_options=git_opts,
            env_options=env_opts,
            dataset_options=dataset_opts
        )
        
        self.assertEqual(manager.git_capture.command_timeout, 10)
        self.assertFalse(manager.git_capture.capture_untracked)
        self.assertFalse(manager.env_capture.capture_packages)
        self.assertEqual(manager.dataset_fingerprinter.hash_algorithm, 'md5')
    
    def test_capture_all_basic_functionality(self):
        """Test the capture_all method with basic functionality."""
        manager = CaptureManager()
        
        # Test basic capture without datasets
        result = manager.capture_all()
        
        # Check structure
        self.assertIn('metadata', result)
        self.assertIn('git', result)
        self.assertIn('environment', result)
        self.assertNotIn('datasets', result)  # No datasets provided
        
        # Check metadata
        metadata = result['metadata']
        self.assertIn('capture_timestamp', metadata)
        self.assertIn('capture_manager_version', metadata)
        self.assertIn('enabled_modules', metadata)
        self.assertTrue(metadata['enabled_modules']['git'])
        self.assertTrue(metadata['enabled_modules']['environment'])
        self.assertTrue(metadata['enabled_modules']['dataset'])
        
        # Check git data
        self.assertIn('commit', result['git'])
        self.assertIn('branch', result['git'])
        
        # Check environment data
        self.assertIn('python_version', result['environment'])
        self.assertIn('system', result['environment'])
        self.assertIn('platform', result['environment']['system'])
    
    def test_capture_all_with_datasets(self):
        """Test capture_all with dataset fingerprinting."""
        manager = CaptureManager()
        
        # Create test datasets
        import pandas as pd
        import numpy as np
        
        datasets = {
            'train_data': pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}),
            'test_data': np.array([[1, 2], [3, 4]]),
            'validation_data': {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        }
        
        result = manager.capture_all(datasets=datasets)
        
        # Check that datasets were captured
        self.assertIn('datasets', result)
        dataset_data = result['datasets']
        
        self.assertIn('train_data', dataset_data)
        self.assertIn('test_data', dataset_data)
        self.assertIn('validation_data', dataset_data)
        
        # Check that each dataset has fingerprint information
        for dataset_name, fingerprint in dataset_data.items():
            self.assertIn('content_hash', fingerprint)
            self.assertIn('fingerprint_algorithm', fingerprint)
            self.assertIn('fingerprint_timestamp', fingerprint)
    
    def test_capture_individual_modules(self):
        """Test individual capture methods."""
        manager = CaptureManager()
        
        # Test git-only capture
        git_result = manager.capture_git_only()
        self.assertIsNotNone(git_result)
        self.assertIn('commit', git_result)
        self.assertIn('branch', git_result)
        
        # Test environment-only capture
        env_result = manager.capture_environment_only()
        self.assertIsNotNone(env_result)
        self.assertIn('python_version', env_result)
        self.assertIn('system', env_result)
        self.assertIn('platform', env_result['system'])
        
        # Test dataset fingerprinting
        import pandas as pd
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        dataset_result = manager.fingerprint_dataset(df, name='test_df')
        self.assertIsNotNone(dataset_result)
        self.assertIn('test_df', dataset_result)
        self.assertIn('content_hash', dataset_result['test_df'])
    
    def test_disabled_modules(self):
        """Test behavior when modules are disabled."""
        manager = CaptureManager(enable_git=False, enable_environment=False, enable_dataset=False)
        
        # Individual captures should return None
        self.assertIsNone(manager.capture_git_only())
        self.assertIsNone(manager.capture_environment_only())
        self.assertIsNone(manager.fingerprint_dataset({'x': [1, 2, 3]}))
        
        # capture_all should only have metadata
        result = manager.capture_all()
        self.assertIn('metadata', result)
        self.assertNotIn('git', result)
        self.assertNotIn('environment', result)
        self.assertNotIn('datasets', result)
        
        # Check metadata reflects disabled modules
        enabled_modules = result['metadata']['enabled_modules']
        self.assertFalse(enabled_modules['git'])
        self.assertFalse(enabled_modules['environment'])
        self.assertFalse(enabled_modules['dataset'])
    
    def test_error_handling_fail_silently_true(self):
        """Test error handling when fail_silently=True."""
        # Test with non-git directory for git capture
        non_git_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        
        try:
            os.chdir(non_git_dir)
            manager = CaptureManager(fail_silently=True)
            
            result = manager.capture_all()
            
            # Should have metadata even if git capture fails
            self.assertIn('metadata', result)
            
            # Git section might have error information or be missing
            if 'git' in result:
                # If present, might contain error info
                git_data = result['git']
                if 'error' in git_data:
                    self.assertTrue(git_data.get('capture_failed', False))
            
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(non_git_dir)
    
    def test_get_configuration(self):
        """Test configuration retrieval."""
        git_opts = {'command_timeout': 15, 'capture_untracked': True}
        env_opts = {'capture_packages': False}
        dataset_opts = {'hash_algorithm': 'sha1'}
        
        manager = CaptureManager(
            git_options=git_opts,
            env_options=env_opts,
            dataset_options=dataset_opts
        )
        
        config = manager.get_configuration()
        
        # Check top-level configuration
        self.assertIn('capture_manager', config)
        self.assertIn('git_capture', config)
        self.assertIn('environment_capture', config)
        self.assertIn('dataset_fingerprinter', config)
        
        # Check specific configurations
        self.assertEqual(config['git_capture']['command_timeout'], 15)
        self.assertTrue(config['git_capture']['capture_untracked'])
        self.assertFalse(config['environment_capture']['capture_packages'])
        self.assertEqual(config['dataset_fingerprinter']['hash_algorithm'], 'sha1')


if __name__ == '__main__':
    unittest.main() 