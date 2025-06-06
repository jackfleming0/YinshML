#!/usr/bin/env python3
"""Debug script to find where tests hang."""

import sys
import os
import tempfile
import time

def debug_step(step_name, func):
    """Run a debugging step with timeout indication."""
    print(f"🔍 Testing: {step_name}")
    start_time = time.time()
    try:
        result = func()
        elapsed = time.time() - start_time
        print(f"✅ {step_name} - SUCCESS ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {step_name} - FAILED ({elapsed:.2f}s): {e}")
        import traceback
        traceback.print_exc()
        return None

def test_imports():
    """Test just importing modules."""
    from utils import initialize_database
    return "Imports successful"

def test_database_creation_no_migrations():
    """Test database creation without migrations."""
    from database import ExperimentDatabase
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    # Try without migrations first
    db = ExperimentDatabase(test_db, use_migrations=False)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    return "Database creation without migrations successful"

def test_database_creation_with_migrations():
    """Test database creation with migrations."""
    from database import ExperimentDatabase
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    # Try with migrations
    db = ExperimentDatabase(test_db, use_migrations=True)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    return "Database creation with migrations successful"

def test_utils_initialize():
    """Test utils initialize_database function."""
    from utils import initialize_database, close_all_connections
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    db = initialize_database(test_db)
    
    # Cleanup
    close_all_connections()
    import shutil
    shutil.rmtree(temp_dir)
    return "Utils initialize_database successful"

def test_connection_manager():
    """Test connection manager singleton."""
    from utils import DatabaseConnectionManager, close_all_connections
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    # Reset singleton
    DatabaseConnectionManager._instance = None
    
    manager = DatabaseConnectionManager(test_db)
    print(f"   Manager created, db_path: {manager.db_path}")
    
    # Cleanup
    close_all_connections()
    DatabaseConnectionManager._instance = None
    import shutil
    shutil.rmtree(temp_dir)
    return "Connection manager test successful"

def test_simple_experiment_creation():
    """Test creating a simple experiment."""
    from utils import initialize_database, create_experiment, close_all_connections
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    db = initialize_database(test_db)
    print("   Database initialized, attempting experiment creation...")
    
    exp_id = create_experiment('test', 'commit', 'branch', {}, {})
    print(f"   Experiment created with ID: {exp_id}")
    
    # Cleanup
    close_all_connections()
    import shutil
    shutil.rmtree(temp_dir)
    return f"Simple experiment creation successful (ID: {exp_id})"

def test_schema_verification():
    """Test schema verification separately."""
    from database import ExperimentDatabase
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    db = ExperimentDatabase(test_db, use_migrations=False)
    print("   Database created, testing schema verification...")
    
    result = db.verify_schema()
    print(f"   Schema verification result: {result}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    return f"Schema verification: {result}"

def main():
    """Run debugging steps systematically."""
    print("🐛 DEBUGGING DATABASE UTILITIES HANG ISSUE")
    print("=" * 50)
    
    # Test each component separately
    tests = [
        ("Import modules", test_imports),
        ("Database creation (no migrations)", test_database_creation_no_migrations),
        ("Schema verification", test_schema_verification),
        ("Database creation (with migrations)", test_database_creation_with_migrations),
        ("Connection manager", test_connection_manager),
        ("Utils initialize_database", test_utils_initialize),
        ("Simple experiment creation", test_simple_experiment_creation),
    ]
    
    for test_name, test_func in tests:
        result = debug_step(test_name, test_func)
        if result is None:
            print(f"\n💥 HANGING/FAILING AT: {test_name}")
            print("This is where the issue occurs!")
            break
        print(f"   Result: {result}")
        print()
        
        # Add a small delay to see progress
        time.sleep(0.1)
    
    print("\n🏁 Debug session complete!")

if __name__ == '__main__':
    main() 