#!/usr/bin/env python3
"""Quick verification script for Task 4 completion.

This script performs a quick smoke test to verify all Task 4 components
are working correctly.
"""

import sys
from pathlib import Path

def verify_imports():
    """Verify all required modules can be imported."""
    print("Verifying imports...")
    try:
        from yinsh_ml.heuristics import (
            YinshHeuristics,
            WeightManager,
            extract_all_features,
            detect_phase,
            GamePhaseCategory,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_evaluator():
    """Verify evaluator can be instantiated and used."""
    print("\nVerifying evaluator...")
    try:
        from yinsh_ml.heuristics import YinshHeuristics
        from yinsh_ml.game.game_state import GameState
        from yinsh_ml.game.constants import Player
        
        evaluator = YinshHeuristics()
        game_state = GameState()
        score = evaluator.evaluate_position(game_state, Player.WHITE)
        
        print(f"✅ Evaluator works - score: {score}")
        return True
    except Exception as e:
        print(f"❌ Evaluator error: {e}")
        return False

def verify_weight_management():
    """Verify weight management functionality."""
    print("\nVerifying weight management...")
    try:
        from yinsh_ml.heuristics import YinshHeuristics
        import tempfile
        import os
        
        evaluator = YinshHeuristics()
        
        # Test get_weight
        weight = evaluator.get_weight('early', 'completed_runs_differential')
        assert weight is not None, "Weight should not be None"
        
        # Test update_weight
        original = weight
        evaluator.update_weight('early', 'completed_runs_differential', original + 1.0)
        updated = evaluator.get_weight('early', 'completed_runs_differential')
        assert updated == original + 1.0, "Weight should be updated"
        
        # Test save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            evaluator.save_weights_to_file(temp_file, create_backup=False)
            assert os.path.exists(temp_file), "File should be created"
            
            new_evaluator = YinshHeuristics()
            new_evaluator.load_weights_from_file(temp_file)
            assert new_evaluator.get_weight('early', 'completed_runs_differential') == updated
            
            print("✅ Weight management works")
            return True
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    except Exception as e:
        print(f"❌ Weight management error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_performance():
    """Verify performance optimizations are in place."""
    print("\nVerifying performance optimizations...")
    try:
        from yinsh_ml.heuristics import YinshHeuristics
        
        evaluator = YinshHeuristics()
        
        # Check for cached attributes
        assert hasattr(evaluator, '_early_weights'), "Cached weights should exist"
        assert hasattr(evaluator, '_mid_weights'), "Cached weights should exist"
        assert hasattr(evaluator, '_late_weights'), "Cached weights should exist"
        assert hasattr(evaluator, '_feature_names'), "Feature names should be cached"
        
        print("✅ Performance optimizations in place")
        return True
    except Exception as e:
        print(f"❌ Performance verification error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Task 4 Verification Script")
    print("=" * 60)
    
    checks = [
        verify_imports,
        verify_evaluator,
        verify_weight_management,
        verify_performance,
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    if all(results):
        print("✅ All checks passed! Task 4 implementation is complete.")
        print("\nTo run comprehensive tests:")
        print("  python yinsh_ml/heuristics/test_heuristics_complete.py")
        print("\nTo run performance benchmark:")
        print("  python yinsh_ml/heuristics/benchmark_evaluator.py")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

