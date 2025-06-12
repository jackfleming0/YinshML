"""
Tests for the ReproductionEngine class.

Tests the metadata capture, storage, and loading capabilities of the reproduction engine.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    from .reproduction import ReproductionEngine, ReproductionEngineError
    from .experiment_tracker import ExperimentTracker
except ImportError:
    from reproduction import ReproductionEngine, ReproductionEngineError
    from experiment_tracker import ExperimentTracker


class TestReproductionEngine(unittest.TestCase):
    """Test the ReproductionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_experiment_path = Path(self.temp_dir) / "test_experiment"
        self.test_experiment_path.mkdir(exist_ok=True)
        
        # Create a test config file
        test_config = {"model_type": "neural_network", "learning_rate": 0.01}
        with open(self.test_experiment_path / "config.json", "w") as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_without_experiment_id(self):
        """Test ReproductionEngine initialization without experiment_id."""
        engine = ReproductionEngine(
            experiment_path=self.test_experiment_path,
            config={'log_level': 'DEBUG'}
        )
        
        self.assertIsNone(engine.experiment_id)
        self.assertEqual(engine.experiment_path, self.test_experiment_path)
        self.assertEqual(engine.config['log_level'], 'DEBUG')
        self.assertIsNone(engine.tracker)
    
    def test_init_with_invalid_experiment_id(self):
        """Test ReproductionEngine initialization with invalid experiment_id."""
        with patch('yinsh_ml.tracking.reproduction.ExperimentTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker.get_experiment.return_value = None
            mock_tracker_class.get_instance.return_value = mock_tracker
            
            with self.assertRaises(ReproductionEngineError):
                ReproductionEngine(experiment_id=999)
    
    def test_capture_random_seed_states(self):
        """Test random seed state capture."""
        engine = ReproductionEngine()
        
        with patch('random.getstate') as mock_getstate:
            mock_getstate.return_value = (3, (2147483648, 1))
            
            seed_states = engine._capture_random_seed_states()
            
            self.assertIn('capture_timestamp', seed_states)
            self.assertIn('python_random', seed_states)
            self.assertTrue(seed_states['python_random']['available'])
            self.assertEqual(seed_states['python_random']['state'], (3, (2147483648, 1)))
    
    def test_capture_random_seed_states_with_numpy(self):
        """Test random seed state capture with NumPy available."""
        engine = ReproductionEngine()
        
        with patch('yinsh_ml.tracking.reproduction.NUMPY_AVAILABLE', True):
            with patch('yinsh_ml.tracking.reproduction.np') as mock_np:
                mock_np.random.get_state.return_value = ('MT19937', [1, 2, 3], 624)
                
                seed_states = engine._capture_random_seed_states()
                
                self.assertIn('numpy_random', seed_states)
                self.assertTrue(seed_states['numpy_random']['available'])
                self.assertEqual(seed_states['numpy_random']['api'], 'legacy')
    
    def test_capture_random_seed_states_with_torch(self):
        """Test random seed state capture with PyTorch available."""
        engine = ReproductionEngine()
        
        with patch('yinsh_ml.tracking.reproduction.TORCH_AVAILABLE', True):
            with patch('yinsh_ml.tracking.reproduction.torch') as mock_torch:
                mock_torch.get_rng_state.return_value = b'torch_state'
                mock_torch.cuda.is_available.return_value = False
                
                seed_states = engine._capture_random_seed_states()
                
                self.assertIn('torch_random', seed_states)
                self.assertTrue(seed_states['torch_random']['available'])
                self.assertEqual(seed_states['torch_random']['state'], b'torch_state')
                self.assertFalse(seed_states['torch_random']['cuda_available'])
    
    def test_capture_system_metadata(self):
        """Test system metadata capture."""
        engine = ReproductionEngine()
        
        system_metadata = engine._capture_system_metadata()
        
        self.assertIn('platform', system_metadata)
        self.assertIn('system', system_metadata)
        self.assertIn('python_version', system_metadata)
        self.assertIn('cpu_count', system_metadata)
    
    def test_capture_environment_metadata(self):
        """Test environment metadata capture."""
        engine = ReproductionEngine()
        
        env_metadata = engine._capture_environment_metadata()
        
        self.assertIn('python_path', env_metadata)
        self.assertIn('environment_variables', env_metadata)
        self.assertIn('installed_packages', env_metadata)
        self.assertIn('working_directory', env_metadata)
    
    def test_capture_git_metadata(self):
        """Test git metadata capture."""
        engine = ReproductionEngine()
        
        with patch('subprocess.run') as mock_run:
            # Mock successful git commands
            mock_run.side_effect = [
                Mock(stdout='abc123\n', returncode=0),  # commit hash
                Mock(stdout='main\n', returncode=0),     # branch
                Mock(stdout='', returncode=0)            # status --porcelain
            ]
            
            git_metadata = engine._capture_git_metadata()
            
            self.assertEqual(git_metadata['commit_hash'], 'abc123')
            self.assertEqual(git_metadata['branch'], 'main')
            self.assertFalse(git_metadata['has_uncommitted_changes'])
            self.assertEqual(git_metadata['uncommitted_files'], [])
    
    def test_capture_configuration_metadata(self):
        """Test configuration metadata capture."""
        engine = ReproductionEngine(experiment_path=self.test_experiment_path)
        
        config_metadata = engine._capture_configuration_metadata()
        
        self.assertIn('capture_timestamp', config_metadata)
        self.assertIn('reproduction_engine_config', config_metadata)
        self.assertIn('config_files', config_metadata)
        self.assertIn('config.json', config_metadata['config_files'])
    
    def test_capture_reproduction_metadata(self):
        """Test comprehensive metadata capture."""
        engine = ReproductionEngine(experiment_path=self.test_experiment_path)
        
        metadata = engine.capture_reproduction_metadata()
        
        self.assertIn('timestamp', metadata)
        self.assertIn('reproduction_engine_version', metadata)
        self.assertIn('system', metadata)
        self.assertIn('environment', metadata)
        self.assertIn('random_seeds', metadata)
        self.assertIn('git', metadata)
        self.assertIn('configuration', metadata)
        
        # Verify metadata is cached
        cached_metadata = engine.get_cached_metadata()
        self.assertEqual(metadata, cached_metadata)
    
    def test_save_and_load_metadata_json(self):
        """Test saving and loading metadata in JSON format."""
        engine = ReproductionEngine()
        metadata = {
            'test_key': 'test_value',
            'timestamp': '2023-01-01T00:00:00',
            'nested': {'key': 'value'}
        }
        
        output_file = Path(self.temp_dir) / "test_metadata.json"
        
        # Save metadata
        saved_path = engine.save_reproduction_metadata(metadata, output_file)
        self.assertEqual(saved_path, output_file)
        self.assertTrue(output_file.exists())
        
        # Load metadata
        loaded_metadata = engine.load_reproduction_metadata(output_file)
        self.assertEqual(loaded_metadata, metadata)
    
    def test_save_metadata_yaml_fallback(self):
        """Test saving metadata in YAML format with fallback to JSON."""
        engine = ReproductionEngine()
        metadata = {'test_key': 'test_value'}
        
        output_file = Path(self.temp_dir) / "test_metadata.yaml"
        
        # Mock PyYAML not being available
        with patch('builtins.__import__', side_effect=ImportError):
            saved_path = engine.save_reproduction_metadata(metadata, output_file, format='yaml')
            
            # Should fallback to JSON
            self.assertEqual(saved_path.suffix, '.json')
            self.assertTrue(saved_path.exists())
    
    def test_load_nonexistent_file(self):
        """Test loading metadata from nonexistent file."""
        engine = ReproductionEngine()
        nonexistent_file = Path(self.temp_dir) / "nonexistent.json"
        
        with self.assertRaises(Exception):
            engine.load_reproduction_metadata(nonexistent_file)
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        engine = ReproductionEngine()
        
        # Capture metadata to populate cache
        engine.capture_reproduction_metadata()
        self.assertIsNotNone(engine.get_cached_metadata())
        
        # Clear cache
        engine.clear_cache()
        self.assertIsNone(engine.get_cached_metadata())
    
    def test_repr(self):
        """Test string representation of ReproductionEngine."""
        # Test with experiment_path
        engine2 = ReproductionEngine(experiment_path="/test/path")
        self.assertEqual(repr(engine2), "ReproductionEngine(experiment_path=/test/path)")
        
        # Test with neither
        engine3 = ReproductionEngine()
        self.assertEqual(repr(engine3), "ReproductionEngine()")
        
        # Test with experiment_id (mocked)
        with patch('yinsh_ml.tracking.reproduction.ExperimentTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker.get_experiment.return_value = {'id': 123, 'name': 'Test Experiment'}
            mock_tracker_class.get_instance.return_value = mock_tracker
            
            engine1 = ReproductionEngine(experiment_id=123)
            self.assertEqual(repr(engine1), "ReproductionEngine(experiment_id=123)")
    
    def test_integration_with_experiment_tracker(self):
        """Test integration with ExperimentTracker."""
        with patch('yinsh_ml.tracking.reproduction.ExperimentTracker') as mock_tracker_class:
            # Mock the tracker and experiment
            mock_tracker = Mock()
            mock_experiment = {
                'name': 'Test Experiment',
                'description': 'Test description',
                'status': 'completed',
                'configuration': '{"param1": "value1"}',
                'created_at': '2023-01-01T00:00:00',
                'updated_at': '2023-01-01T12:00:00',
                'tags': ['test', 'reproduction']
            }
            
            mock_tracker.get_experiment.return_value = mock_experiment
            mock_tracker._capture_system_metadata.return_value = {'system': 'test'}
            mock_tracker._capture_environment_metadata.return_value = {'env': 'test'}
            mock_tracker._capture_git_metadata.return_value = {'git': 'test'}
            
            mock_tracker_class.get_instance.return_value = mock_tracker
            
            # Test with valid experiment
            engine = ReproductionEngine(experiment_id=123)
            self.assertEqual(engine.experiment_id, 123)
            self.assertIsNotNone(engine.tracker)
            
            # Test metadata capture with tracker integration
            metadata = engine.capture_reproduction_metadata()
            
            self.assertIn('experiment', metadata)
            self.assertEqual(metadata['experiment']['name'], 'Test Experiment')
            self.assertEqual(metadata['experiment']['experiment_id'], 123)
            
            # Verify tracker methods were called
            mock_tracker._capture_system_metadata.assert_called_once()
            mock_tracker._capture_environment_metadata.assert_called_once()
            mock_tracker._capture_git_metadata.assert_called_once()
    
    def test_capture_enhanced_environment_metadata(self):
        """Test enhanced environment metadata capture."""
        engine = ReproductionEngine()
        
        with patch.object(engine, '_capture_conda_environment') as mock_conda:
            with patch.object(engine, '_capture_pip_environment') as mock_pip:
                mock_conda.return_value = {'conda_version': '4.10.0', 'available': True}
                mock_pip.return_value = {'pip_version': '21.0.0', 'available': True}
                
                enhanced_env = engine.capture_enhanced_environment_metadata()
                
                self.assertIn('conda', enhanced_env)
                self.assertIn('pip', enhanced_env)
                self.assertIn('primary_package_manager', enhanced_env)
    
    def test_capture_conda_environment(self):
        """Test conda environment capture."""
        engine = ReproductionEngine()
        
        with patch('subprocess.run') as mock_run:
            # Mock conda commands
            mock_run.side_effect = [
                Mock(stdout='conda 4.10.0\n', returncode=0),  # conda --version
                Mock(stdout='{"active_prefix_name": "test_env"}', returncode=0),  # conda info --json
                Mock(stdout='[{"name": "numpy", "version": "1.21.0"}]', returncode=0),  # conda list --json
                Mock(stdout='name: test_env\ndependencies:\n  - numpy=1.21.0', returncode=0)  # conda env export
            ]
            
            conda_info = engine._capture_conda_environment()
            
            self.assertIsNotNone(conda_info)
            self.assertTrue(conda_info['available'])
            self.assertEqual(conda_info['active_environment'], 'test_env')
            self.assertIn('numpy', conda_info['packages'])
    
    def test_capture_pip_environment(self):
        """Test pip environment capture."""
        engine = ReproductionEngine()
        
        with patch('subprocess.run') as mock_run:
            # Mock pip commands
            mock_run.side_effect = [
                Mock(stdout='pip 21.0.0\n', returncode=0),  # pip --version
                Mock(stdout='numpy==1.21.0\nscipy==1.7.0\n', returncode=0),  # pip freeze
                Mock(stdout='[{"name": "numpy", "version": "1.21.0"}]', returncode=0)  # pip list --format=json
            ]
            
            pip_info = engine._capture_pip_environment()
            
            self.assertTrue(pip_info['available'])
            self.assertIn('numpy', pip_info['packages'])
            self.assertEqual(pip_info['packages']['numpy'], '1.21.0')
            self.assertIn('requirements_txt', pip_info)
    
    def test_detect_primary_package_manager(self):
        """Test primary package manager detection."""
        engine = ReproductionEngine()
        
        # Test conda environment detection
        with patch.dict(os.environ, {'CONDA_DEFAULT_ENV': 'test_env'}):
            self.assertEqual(engine._detect_primary_package_manager(), 'conda')
        
        # Test virtual environment detection
        with patch.dict(os.environ, {'VIRTUAL_ENV': '/path/to/venv'}, clear=True):
            self.assertEqual(engine._detect_primary_package_manager(), 'pip')
        
        # Test default case
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(engine._detect_primary_package_manager(), 'pip')
    
    def test_export_environment_files(self):
        """Test environment file export."""
        engine = ReproductionEngine()
        
        # Create test metadata
        metadata = {
            'environment': {
                'pip': {
                    'requirements_txt': 'numpy==1.21.0\nscipy==1.7.0\n',
                    'available': True
                },
                'conda': {
                    'available': False
                }
            }
        }
        
        output_dir = Path(self.temp_dir) / "export_test"
        
        exported_files = engine.export_environment_files(output_dir, metadata)
        
        self.assertIn('requirements.txt', exported_files)
        self.assertTrue(exported_files['requirements.txt'].exists())
        
        # Check file content
        with open(exported_files['requirements.txt'], 'r') as f:
            content = f.read()
            self.assertIn('numpy==1.21.0', content)
            self.assertIn('scipy==1.7.0', content)
    
    def test_export_requirements_txt(self):
        """Test requirements.txt export."""
        engine = ReproductionEngine()
        output_dir = Path(self.temp_dir)
        
        metadata = {
            'environment': {
                'pip': {
                    'requirements_txt': 'numpy==1.21.0\n'
                }
            }
        }
        
        requirements_path = engine._export_requirements_txt(output_dir, metadata)
        
        self.assertIsNotNone(requirements_path)
        self.assertTrue(requirements_path.exists())
        
        with open(requirements_path, 'r') as f:
            content = f.read()
            self.assertIn('numpy==1.21.0', content)
    
    def test_export_environment_yml(self):
        """Test environment.yml export."""
        engine = ReproductionEngine()
        output_dir = Path(self.temp_dir)
        
        metadata = {
            'environment': {
                'conda': {
                    'available': True,
                    'packages': {
                        'numpy': {'version': '1.21.0'},
                        'scipy': {'version': '1.7.0'}
                    }
                }
            }
        }
        
        yml_path = engine._export_environment_yml(output_dir, metadata)
        
        self.assertIsNotNone(yml_path)
        self.assertTrue(yml_path.exists())
        
        with open(yml_path, 'r') as f:
            content = f.read()
            self.assertIn('numpy=1.21.0', content)
            self.assertIn('scipy=1.7.0', content)
    
    def test_recreate_environment_pip(self):
        """Test pip environment recreation."""
        engine = ReproductionEngine()
        
        metadata = {
            'environment': {
                'primary_package_manager': 'pip',
                'pip': {
                    'packages': {'numpy': '1.21.0'}
                }
            }
        }
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stderr='')
            
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.return_value = False
                
                result = engine.recreate_environment(metadata, 'test-env')
                
                self.assertEqual(result['environment_name'], 'test-env')
                self.assertEqual(result['primary_manager'], 'pip')
    
    def test_parse_pip_conflicts(self):
        """Test pip conflict parsing."""
        engine = ReproductionEngine()
        
        error_output = """
        ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
        This behaviour is the source of the following dependency conflicts.
        package-a 1.0.0 requires package-b>=2.0.0, but you have package-b 1.5.0 which is incompatible.
        """
        
        conflicts = engine._parse_pip_conflicts(error_output)
        
        self.assertTrue(len(conflicts) > 0)
        self.assertTrue(any('incompatible' in conflict for conflict in conflicts))


if __name__ == '__main__':
    unittest.main() 