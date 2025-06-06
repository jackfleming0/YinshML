#!/usr/bin/env python3
"""Minimal debug of connection manager."""

import os
import tempfile
import sys
import signal

def alarm_handler(signum, frame):
    print("❌ ALARM: Operation taking too long, likely hanging!")
    sys.exit(1)

# Set up alarm for 10 seconds
signal.signal(signal.SIGALRM, alarm_handler)

try:
    print("🔍 Starting minimal connection manager debug...")
    
    # Step 1: Test SQLite directly first
    print("Step 1: Testing direct SQLite...")
    import sqlite3
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    print(f"   DB path: {test_db}")
    
    signal.alarm(5)  # 5 second timeout
    conn = sqlite3.connect(test_db)
    signal.alarm(0)  # Cancel alarm
    print("✅ Direct SQLite connection works")
    
    # Test WAL mode specifically
    print("Step 2: Testing WAL mode...")
    signal.alarm(5)
    conn.execute("PRAGMA journal_mode = WAL")
    signal.alarm(0)
    print("✅ WAL mode works")
    
    conn.close()
    
    # Step 3: Test our connection manager import
    print("Step 3: Importing DatabaseConnectionManager...")
    signal.alarm(5)
    from utils import DatabaseConnectionManager
    signal.alarm(0)
    print("✅ Import successful")
    
    # Step 4: Test manager initialization
    print("Step 4: Creating manager instance...")
    signal.alarm(10)  # Give more time for this
    manager = DatabaseConnectionManager(test_db)
    signal.alarm(0)
    print("✅ Manager created!")
    
    # Step 5: Test get_connection method
    print("Step 5: Getting connection...")
    signal.alarm(10)
    conn = manager.get_connection()
    signal.alarm(0)
    print("✅ Connection obtained!")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    signal.alarm(0)  # Cancel any pending alarm
    print(f"❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    signal.alarm(0)  # Ensure alarm is cancelled
    # Cleanup
    import shutil
    if 'temp_dir' in locals():
        shutil.rmtree(temp_dir, ignore_errors=True) 