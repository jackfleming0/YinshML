#!/usr/bin/env python3
"""Test the singleton fix step by step."""

import sys
import os
import tempfile
import signal

def alarm_handler(signum, frame):
    print("❌ HANG DETECTED!")
    sys.exit(1)

signal.signal(signal.SIGALRM, alarm_handler)

try:
    print("🔍 Testing singleton fix step by step...")
    
    # Import and reset any existing singleton
    print("Step 1: Import and reset singleton...")
    signal.alarm(5)
    from utils import DatabaseConnectionManager
    if hasattr(DatabaseConnectionManager, '_instance'):
        DatabaseConnectionManager._instance = None
    signal.alarm(0)
    print("✅ Import and reset successful")
    
    # Create temp database
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    print(f"   Test DB: {test_db}")
    
    # Test creating singleton with no path (like global instance)
    print("Step 2: Create singleton with no path...")
    signal.alarm(5)
    manager1 = DatabaseConnectionManager()
    signal.alarm(0)
    print(f"✅ Manager1 created: {manager1}")
    print(f"   Manager1 db_path: {getattr(manager1, 'db_path', 'MISSING')}")
    
    # Test creating singleton with path (should trigger set_database_path)
    print("Step 3: Create singleton with path (should call set_database_path)...")
    signal.alarm(10)  # Give more time for this
    manager2 = DatabaseConnectionManager(test_db)
    signal.alarm(0)
    print(f"✅ Manager2 created: {manager2}")
    print(f"   Manager2 db_path: {getattr(manager2, 'db_path', 'MISSING')}")
    print(f"   Same instance: {manager1 is manager2}")
    
    # Test getting connection
    print("Step 4: Test getting connection...")
    signal.alarm(10)
    conn = manager2.get_connection()
    signal.alarm(0)
    print(f"✅ Connection obtained: {conn}")
    
    print("🎉 All singleton tests passed!")
    
except Exception as e:
    signal.alarm(0)
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
finally:
    signal.alarm(0)
    # Cleanup
    import shutil
    if 'temp_dir' in locals():
        shutil.rmtree(temp_dir, ignore_errors=True) 