#!/usr/bin/env python3
"""Debug connection manager specifically."""

import sys
import os
import tempfile
import time
import threading

def test_connection_manager_step_by_step():
    """Test connection manager initialization step by step."""
    print("🔍 Testing connection manager step by step...")
    
    # Step 1: Import
    print("Step 1: Importing DatabaseConnectionManager...")
    from utils import DatabaseConnectionManager
    print("✅ Import successful")
    
    # Step 2: Create temp directory
    print("Step 2: Creating temp directory...")
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    print(f"✅ Temp directory: {temp_dir}")
    print(f"✅ Test DB path: {test_db}")
    
    # Step 3: Reset singleton (if it exists)
    print("Step 3: Resetting singleton...")
    if hasattr(DatabaseConnectionManager, '_instance'):
        print(f"   Current singleton instance: {DatabaseConnectionManager._instance}")
        DatabaseConnectionManager._instance = None
        print("✅ Singleton reset")
    else:
        print("✅ No existing singleton")
    
    # Step 4: Check threading info
    print("Step 4: Threading info...")
    print(f"   Current thread: {threading.current_thread()}")
    print(f"   Thread count: {threading.active_count()}")
    
    # Step 5: Create manager with verbose output
    print("Step 5: Creating DatabaseConnectionManager...")
    try:
        print("   About to call DatabaseConnectionManager() constructor...")
        manager = DatabaseConnectionManager(test_db)
        print("✅ Manager created successfully!")
        print(f"   Manager db_path: {manager.db_path}")
        print(f"   Manager instance: {manager}")
        
        # Step 6: Test getting connection
        print("Step 6: Testing get_connection...")
        conn = manager.get_connection()
        print(f"✅ Connection obtained: {conn}")
        
        # Step 7: Test cursor
        print("Step 7: Testing cursor creation...")
        cursor = conn.cursor()
        print(f"✅ Cursor created: {cursor}")
        
        # Step 8: Simple query
        print("Step 8: Testing simple query...")
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"✅ Query result: {result}")
        
    except Exception as e:
        print(f"❌ Failed at manager creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("Step 9: Cleanup...")
    try:
        from utils import close_all_connections
        close_all_connections()
        DatabaseConnectionManager._instance = None
        print("✅ Connections closed")
        
        import shutil
        shutil.rmtree(temp_dir)
        print("✅ Temp directory cleaned up")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("🏁 Connection manager test completed successfully!")
    return True

def test_sqlite_directly():
    """Test SQLite directly without our wrapper."""
    print("🔍 Testing SQLite directly...")
    
    import sqlite3
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'direct_test.db')
    
    print(f"   Creating SQLite connection to: {test_db}")
    conn = sqlite3.connect(test_db)
    print("✅ Direct SQLite connection successful")
    
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    print(f"✅ Direct query result: {result}")
    
    conn.close()
    print("✅ Direct connection closed")
    
    import shutil
    shutil.rmtree(temp_dir)
    print("✅ Direct test cleanup complete")

def main():
    """Run focused connection manager debugging."""
    print("🐛 DEBUGGING CONNECTION MANAGER SPECIFICALLY")
    print("=" * 50)
    
    # Test direct SQLite first
    test_sqlite_directly()
    print()
    
    # Test our connection manager
    test_connection_manager_step_by_step()

if __name__ == '__main__':
    main() 