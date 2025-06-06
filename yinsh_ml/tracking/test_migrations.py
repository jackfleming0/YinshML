#!/usr/bin/env python3
"""
Test suite for the database migration system.

Tests migration functionality including applying, rolling back,
and managing multiple migrations with proper version tracking.
"""

import sqlite3
import tempfile
import os
from pathlib import Path

from .migrations import Migration, MigrationManager, InitialSchemaMigration
from .database import ExperimentDatabase


class SampleMigration1(Migration):
    """Sample migration that adds a column to the experiments table."""
    
    def __init__(self):
        super().__init__("002", "Add description column to experiments table")
    
    def up(self, conn: sqlite3.Connection):
        """Add description column."""
        conn.execute("ALTER TABLE experiments ADD COLUMN description TEXT")
        
    def down(self, conn: sqlite3.Connection):
        """Remove description column (by recreating table without it)."""
        # SQLite doesn't support DROP COLUMN easily, so we recreate the table
        # This is just for testing - in practice we'd handle this differently
        
        # Create temporary table with original schema
        conn.execute("""
            CREATE TABLE experiments_temp (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                git_commit TEXT NOT NULL,
                git_branch TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                config_json TEXT NOT NULL,
                environment_json TEXT NOT NULL,
                notes TEXT
            )
        """)
        
        # Copy data (excluding description column)
        conn.execute("""
            INSERT INTO experiments_temp 
            SELECT id, name, timestamp, git_commit, git_branch, status, config_json, environment_json, notes
            FROM experiments
        """)
        
        # Drop original table and rename temp
        conn.execute("DROP TABLE experiments")
        conn.execute("ALTER TABLE experiments_temp RENAME TO experiments")
        
        # Recreate indexes for experiments table
        conn.execute("CREATE INDEX idx_experiments_timestamp ON experiments(timestamp)")
        conn.execute("CREATE INDEX idx_experiments_status ON experiments(status)")
        conn.execute("CREATE INDEX idx_experiments_git_branch ON experiments(git_branch)")


class SampleMigration2(Migration):
    """Sample migration that adds a table."""
    
    def __init__(self):
        super().__init__("003", "Add experiment_logs table")
    
    def up(self, conn: sqlite3.Connection):
        """Create experiment_logs table."""
        conn.execute("""
            CREATE TABLE experiment_logs (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        """)
        
        # Add index for efficient querying
        conn.execute("CREATE INDEX idx_experiment_logs_experiment_id ON experiment_logs(experiment_id)")
        
    def down(self, conn: sqlite3.Connection):
        """Drop experiment_logs table."""
        conn.execute("DROP TABLE IF EXISTS experiment_logs")


def test_migration_system():
    """Test the migration system functionality."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    
    try:
        print("Testing migration manager initialization...")
        manager = MigrationManager(db_path)
        
        # Should have the initial migration registered
        assert len(manager.migrations) == 1
        assert manager.migrations[0].version == "001"
        assert isinstance(manager.migrations[0], InitialSchemaMigration)
        
        print("Testing initial migration application...")
        success = manager.migrate_to_latest()
        assert success
        
        # Verify migration was recorded
        applied = manager.get_applied_migrations()
        assert applied == ["001"]
        assert manager.get_current_version() == "001"
        
        print("Testing custom migration...")
        migration1 = SampleMigration1()
        manager.register_migration(migration1)
        
        # Apply custom migration
        success = manager.migrate_to_latest()
        assert success
        
        applied = manager.get_applied_migrations()
        assert applied == ["001", "002"]
        assert manager.get_current_version() == "002"
        
        # Verify schema changes
        with manager._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check that description column was added
            cursor.execute("PRAGMA table_info(experiments)")
            columns = {row[1] for row in cursor.fetchall()}
            assert 'description' in columns
        
        print("Testing migration rollback...")
        success = manager.rollback_to_version("001")
        assert success
        
        applied = manager.get_applied_migrations()
        assert applied == ["001"]
        assert manager.get_current_version() == "001"
        
        # Verify description column was removed
        with manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(experiments)")
            columns = {row[1] for row in cursor.fetchall()}
            assert 'description' not in columns
        
        print("Testing database integration...")
        # Test ExperimentDatabase with migrations
        db = ExperimentDatabase(db_path, use_migrations=True)
        assert db.verify_schema()
        
        # Test normal operations
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, git_commit, git_branch, config_json, environment_json)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_exp", "abc123", "main", "{}", "{}"))
            
            cursor.execute("SELECT COUNT(*) FROM experiments")
            assert cursor.fetchone()[0] == 1
        
        print("✅ All migration tests passed!")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_migration_status():
    """Test migration status reporting."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    
    try:
        manager = MigrationManager(db_path)
        migration1 = SampleMigration1()
        manager.register_migration(migration1)
        
        # Initial status
        status = manager.get_migration_status()
        assert status['current_version'] is None
        assert status['total_migrations'] == 2
        assert status['applied_count'] == 0
        assert status['pending_count'] == 2
        
        # Apply first migration
        manager.apply_migration(manager.migrations[0])
        
        status = manager.get_migration_status()
        assert status['current_version'] == "001"
        assert status['applied_count'] == 1
        assert status['pending_count'] == 1
        
        print("✅ Migration status test passed!")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    test_migration_system()
    test_migration_status() 