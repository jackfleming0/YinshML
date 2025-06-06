#!/usr/bin/env python3
"""Test if WAL mode configuration is causing hangs."""

import sqlite3
import tempfile
import os
import signal
import sys

def alarm_handler(signum, frame):
    print("❌ HANG DETECTED: Operation taking too long!")
    sys.exit(1)

signal.signal(signal.SIGALRM, alarm_handler)

def test_wal_configurations():
    """Test different WAL configurations to isolate the issue."""
    
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    try:
        print("🔍 Testing different SQLite configurations...")
        
        # Test 1: Basic connection
        print("Test 1: Basic connection...")
        signal.alarm(5)
        conn = sqlite3.connect(test_db)
        signal.alarm(0)
        print("✅ Basic connection works")
        
        # Test 2: Foreign keys
        print("Test 2: Foreign keys...")
        signal.alarm(5)
        conn.execute("PRAGMA foreign_keys = ON")
        signal.alarm(0)
        print("✅ Foreign keys work")
        
        # Test 3: WAL mode
        print("Test 3: WAL mode...")
        signal.alarm(5)
        result = conn.execute("PRAGMA journal_mode = WAL").fetchone()
        signal.alarm(0)
        print(f"✅ WAL mode result: {result}")
        
        # Test 4: Synchronous mode
        print("Test 4: Synchronous mode...")
        signal.alarm(5)
        conn.execute("PRAGMA synchronous = NORMAL")
        signal.alarm(0)
        print("✅ Synchronous mode works")
        
        # Test 5: Cache size
        print("Test 5: Cache size...")
        signal.alarm(5)
        conn.execute("PRAGMA cache_size = 10000")
        signal.alarm(0)
        print("✅ Cache size works")
        
        # Test 6: Temp store
        print("Test 6: Temp store...")
        signal.alarm(5)
        conn.execute("PRAGMA temp_store = MEMORY")
        signal.alarm(0)
        print("✅ Temp store works")
        
        # Test 7: Memory mapping
        print("Test 7: Memory mapping...")
        signal.alarm(5)
        conn.execute("PRAGMA mmap_size = 268435456")
        signal.alarm(0)
        print("✅ Memory mapping works")
        
        # Test 8: Row factory
        print("Test 8: Row factory...")
        signal.alarm(5)
        conn.row_factory = sqlite3.Row
        signal.alarm(0)
        print("✅ Row factory works")
        
        # Test 9: All together with new connection
        print("Test 9: All optimizations on new connection...")
        conn.close()
        
        signal.alarm(10)
        conn = sqlite3.connect(
            test_db,
            timeout=30.0,
            check_same_thread=False
        )
        
        # Apply all optimizations
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")
        conn.row_factory = sqlite3.Row
        signal.alarm(0)
        print("✅ All optimizations work together")
        
        # Test 10: Simple query after optimizations
        print("Test 10: Query after optimizations...")
        signal.alarm(5)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        signal.alarm(0)
        print(f"✅ Query result: {result}")
        
        conn.close()
        print("🎉 All SQLite configuration tests passed!")
        
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    test_wal_configurations() 