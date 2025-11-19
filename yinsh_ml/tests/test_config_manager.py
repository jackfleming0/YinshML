"""Tests for unified ConfigManager system.

This test suite verifies that ConfigManager correctly unifies WeightManager
and PhaseConfig with thread-safe operations and proper validation.
"""

import unittest
import json
import tempfile
import threading
import os
from pathlib import Path

from yinsh_ml.heuristics.config_manager import ConfigManager
from yinsh_ml.heuristics.weight_manager import WeightManager
from yinsh_ml.heuristics.phase_config import PhaseConfig


class TestConfigManager(unittest.TestCase):
    """Test suite for ConfigManager unified configuration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that ConfigManager initializes correctly."""
        manager = ConfigManager()
        
        self.assertIsInstance(manager.weight_manager, WeightManager)
        self.assertIsInstance(manager.phase_config, PhaseConfig)
        self.assertIsNotNone(manager._lock)
    
    def test_load_config_from_file(self):
        """Test loading configuration from a JSON file."""
        # Create test config file
        config_data = {
            "version": "1.0",
            "weights": {
                "early": {
                    "completed_runs_differential": 10.0,
                    "potential_runs_count": 8.0,
                    "connected_marker_chains": 6.0,
                    "ring_positioning": 5.0,
                    "ring_spread": 4.0,
                    "board_control": 3.0,
                },
                "mid": {
                    "completed_runs_differential": 12.0,
                    "potential_runs_count": 10.0,
                    "connected_marker_chains": 8.0,
                    "ring_positioning": 7.0,
                    "ring_spread": 6.0,
                    "board_control": 5.0,
                },
                "late": {
                    "completed_runs_differential": 15.0,
                    "potential_runs_count": 12.0,
                    "connected_marker_chains": 10.0,
                    "ring_positioning": 9.0,
                    "ring_spread": 8.0,
                    "board_control": 7.0,
                },
            },
            "phase_config": {
                "early_max_moves": 20,
                "mid_max_moves": 40,
                "transition_window": 3,
                "interpolation_method": "linear",
            },
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        manager = ConfigManager()
        manager.load_config(self.config_path)
        
        # Verify weights loaded
        weights = manager.weight_manager.get_weights()
        self.assertEqual(weights["early"]["completed_runs_differential"], 10.0)
        self.assertEqual(weights["mid"]["completed_runs_differential"], 12.0)
        self.assertEqual(weights["late"]["completed_runs_differential"], 15.0)
        
        # Verify phase config loaded
        self.assertEqual(manager.phase_config.early_max_moves, 20)
        self.assertEqual(manager.phase_config.mid_max_moves, 40)
        self.assertEqual(manager.phase_config.transition_window, 3)
    
    def test_save_config_to_file(self):
        """Test saving configuration to a JSON file."""
        manager = ConfigManager()
        
        # Set some weights
        manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        
        # Update phase config
        manager.phase_config = manager.phase_config.update(
            early_max_moves=20,
            mid_max_moves=40,
        )
        
        # Save config
        manager.save_config(self.config_path)
        
        # Verify file exists and contains correct data
        self.assertTrue(os.path.exists(self.config_path))
        
        with open(self.config_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertIn("weights", saved_data)
        self.assertIn("phase_config", saved_data)
        self.assertEqual(saved_data["weights"]["early"]["completed_runs_differential"], 10.0)
        self.assertEqual(saved_data["phase_config"]["early_max_moves"], 20)
    
    def test_atomic_weight_update(self):
        """Test thread-safe atomic weight updates."""
        manager = ConfigManager()
        manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        
        # Update weight atomically
        manager.update_weight_atomic("early", "completed_runs_differential", 15.0)
        
        # Verify update
        weight = manager.weight_manager.get_weight("early", "completed_runs_differential")
        self.assertEqual(weight, 15.0)
    
    def test_concurrent_access(self):
        """Test thread safety with concurrent access."""
        manager = ConfigManager()
        manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        
        updates = []
        errors = []
        
        def update_worker(phase, feature, value):
            try:
                manager.update_weight_atomic(phase, feature, value)
                updates.append((phase, feature, value))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads updating different weights
        threads = []
        test_cases = [
            ("early", "completed_runs_differential", 11.0),
            ("mid", "completed_runs_differential", 13.0),
            ("late", "completed_runs_differential", 16.0),
            ("early", "potential_runs_count", 9.0),
            ("mid", "potential_runs_count", 11.0),
        ]
        
        for phase, feature, value in test_cases:
            thread = threading.Thread(
                target=update_worker,
                args=(phase, feature, value)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        # Verify all updates were applied
        self.assertEqual(len(updates), len(test_cases))
        
        # Verify final state is consistent
        self.assertEqual(
            manager.weight_manager.get_weight("early", "completed_runs_differential"),
            11.0
        )
        self.assertEqual(
            manager.weight_manager.get_weight("mid", "completed_runs_differential"),
            13.0
        )
        self.assertEqual(
            manager.weight_manager.get_weight("late", "completed_runs_differential"),
            16.0
        )
    
    def test_validation_errors(self):
        """Test that validation errors are raised for invalid configurations."""
        manager = ConfigManager()
        
        # Set default weights first (needed for validation)
        manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        
        # Test invalid weight value (out of range)
        with self.assertRaises(ValueError):
            manager.update_weight_atomic("early", "completed_runs_differential", 100.0)
        
        # Test invalid phase name
        with self.assertRaises(ValueError):
            manager.update_weight_atomic("invalid_phase", "completed_runs_differential", 10.0)
        
        # Test invalid feature name
        with self.assertRaises(ValueError):
            manager.update_weight_atomic("early", "invalid_feature", 10.0)
    
    def test_config_versioning(self):
        """Test configuration version tracking."""
        manager = ConfigManager()
        
        # Set initial config
        manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        
        # Save config (should create version)
        manager.save_config(self.config_path)
        
        # Verify version is saved
        with open(self.config_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertIn("version", saved_data)
        self.assertIsInstance(saved_data["version"], str)
    
    def test_get_current_config(self):
        """Test retrieving current configuration."""
        manager = ConfigManager()
        
        # Set some weights
        manager.weight_manager.set_default_weights({
            "early": {
                "completed_runs_differential": 10.0,
                "potential_runs_count": 8.0,
                "connected_marker_chains": 6.0,
                "ring_positioning": 5.0,
                "ring_spread": 4.0,
                "board_control": 3.0,
            },
            "mid": {
                "completed_runs_differential": 12.0,
                "potential_runs_count": 10.0,
                "connected_marker_chains": 8.0,
                "ring_positioning": 7.0,
                "ring_spread": 6.0,
                "board_control": 5.0,
            },
            "late": {
                "completed_runs_differential": 15.0,
                "potential_runs_count": 12.0,
                "connected_marker_chains": 10.0,
                "ring_positioning": 9.0,
                "ring_spread": 8.0,
                "board_control": 7.0,
            },
        })
        
        # Get current config
        config = manager.get_current_config()
        
        self.assertIn("weights", config)
        self.assertIn("phase_config", config)
        self.assertEqual(config["weights"]["early"]["completed_runs_differential"], 10.0)


if __name__ == '__main__':
    unittest.main()

