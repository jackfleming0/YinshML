#!/usr/bin/env python3
"""Verification script for database utilities."""

import sys
import tempfile
import os
import traceback

def test_basic_functionality():
    """Test basic database utility functionality."""
    print("=== Testing Database Utilities ===")
    
    try:
        # Test imports
        from utils import (
            initialize_database, create_experiment, add_metric_to_experiment, 
            get_database_stats, close_all_connections, query_experiments,
            get_experiment_by_id, add_metrics_bulk
        )
        print("✓ All imports successful")
        
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        test_db = os.path.join(temp_dir, 'test.db')
        
        # Test database initialization
        db = initialize_database(test_db)
        print("✓ Database initialized")
        
        # Test experiment creation
        config = {'model': 'ResNet50', 'batch_size': 32}
        environment = {'python': '3.9', 'pytorch': '1.10'}
        exp_id = create_experiment(
            'test_experiment', 'abc123', 'main', 
            config, environment, 
            tags=['test', 'verification']
        )
        print(f"✓ Created experiment {exp_id}")
        
        # Test metric addition
        add_metric_to_experiment(exp_id, 'accuracy', 0.95, 1)
        print("✓ Added single metric")
        
        # Test bulk metrics
        metrics = [
            {'name': 'loss', 'value': 0.05, 'iteration': 1},
            {'name': 'accuracy', 'value': 0.97, 'iteration': 2}
        ]
        add_metrics_bulk(exp_id, metrics)
        print("✓ Added bulk metrics")
        
        # Test querying
        exp = get_experiment_by_id(exp_id)
        assert exp is not None, "Experiment should exist"
        assert exp['name'] == 'test_experiment', "Experiment name should match"
        print("✓ Retrieved experiment data")
        
        # Test tag querying
        tagged_exps = query_experiments(tags=['test'])
        assert len(tagged_exps) == 1, "Should find 1 experiment with 'test' tag"
        print("✓ Tag-based querying works")
        
        # Test database stats
        stats = get_database_stats()
        assert stats['experiment_count'] >= 1, "Should have at least 1 experiment"
        assert stats['metric_count'] >= 3, "Should have at least 3 metrics"
        print(f"✓ Database stats: {stats['experiment_count']} experiments, {stats['metric_count']} metrics")
        
        # Cleanup
        close_all_connections()
        os.unlink(test_db)
        os.rmdir(temp_dir)
        print("✓ Cleanup completed")
        
        print("\n✅ ALL TESTS PASSED! Database utilities are working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling capabilities."""
    print("\n=== Testing Error Handling ===")
    
    try:
        from utils import validate_experiment_data, ValidationError
        
        # Test validation errors
        try:
            validate_experiment_data("", "commit", "branch", {}, {})
            print("❌ Should have raised ValidationError for empty name")
            return False
        except ValidationError:
            print("✓ Validation correctly rejects empty experiment name")
        
        try:
            validate_experiment_data("test", "commit", "branch", "not_dict", {})
            print("❌ Should have raised ValidationError for invalid config")
            return False
        except ValidationError:
            print("✓ Validation correctly rejects invalid config type")
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_transaction_handling():
    """Test transaction capabilities."""
    print("\n=== Testing Transaction Handling ===")
    
    try:
        from utils import initialize_database, with_transaction, close_all_connections
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        test_db = os.path.join(temp_dir, 'transaction_test.db')
        
        db = initialize_database(test_db)
        
        @with_transaction
        def test_transaction(conn):
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, git_commit, git_branch, status, config_json, environment_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("transaction_test", "abc123", "main", "running", "{}", "{}"))
            return cursor.lastrowid
        
        exp_id = test_transaction()
        assert isinstance(exp_id, int), "Should return experiment ID"
        print("✓ Transaction decorator works")
        
        # Cleanup
        close_all_connections()
        os.unlink(test_db)
        os.rmdir(temp_dir)
        
        print("✅ Transaction handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Transaction test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = True
    
    success &= test_basic_functionality()
    success &= test_error_handling()
    success &= test_transaction_handling()
    
    if success:
        print("\n🎉 ALL VERIFICATION TESTS PASSED!")
        print("Database utilities are fully functional and ready for use.")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED!")
        print("Database utilities need fixes before they can be considered complete.")
        sys.exit(1) 