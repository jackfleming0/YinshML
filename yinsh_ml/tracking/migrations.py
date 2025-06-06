"""
Database migration system for YinshML experiment tracking.

Provides version-controlled schema changes with support for both forward
and rollback migrations. Tracks applied migrations in a dedicated table.
"""

import sqlite3
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Migration(ABC):
    """
    Abstract base class for database migrations.
    
    Each migration must implement up() and down() methods for forward
    and rollback operations respectively.
    """
    
    def __init__(self, version: str, description: str):
        """
        Initialize migration.
        
        Args:
            version: Unique version identifier (e.g., "001", "20231201_001")
            description: Human-readable description of the migration
        """
        self.version = version
        self.description = description
        self.timestamp = datetime.now()
    
    @abstractmethod
    def up(self, conn: sqlite3.Connection):
        """
        Apply the migration (forward operation).
        
        Args:
            conn: SQLite database connection
        """
        pass
    
    @abstractmethod
    def down(self, conn: sqlite3.Connection):
        """
        Rollback the migration (reverse operation).
        
        Args:
            conn: SQLite database connection
        """
        pass
    
    def __str__(self):
        return f"Migration {self.version}: {self.description}"


class InitialSchemaMigration(Migration):
    """
    Initial migration that creates the base schema from subtasks 1.1 and 1.2.
    
    This migration establishes the experiments, metrics, and tags tables
    along with all the performance indexes.
    """
    
    def __init__(self):
        super().__init__("001", "Create initial schema with experiments, metrics, tags tables and indexes")
    
    def up(self, conn: sqlite3.Connection):
        """Create the initial schema with all tables and indexes."""
        logger.info("Applying initial schema migration")
        
        # Create experiments table
        conn.execute("""
            CREATE TABLE experiments (
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
        
        # Create metrics table
        conn.execute("""
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                iteration INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        """)
        
        # Create tags table
        conn.execute("""
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(experiment_id, tag)
            )
        """)
        
        # Create all performance indexes
        self._create_indexes(conn)
        
        logger.info("Initial schema migration applied successfully")
    
    def down(self, conn: sqlite3.Connection):
        """Drop all tables and indexes."""
        logger.info("Rolling back initial schema migration")
        
        # Drop indexes first (though they'll be dropped with tables anyway)
        index_names = [
            'idx_metrics_experiment_id', 'idx_metrics_experiment_metric',
            'idx_metrics_metric_name', 'idx_metrics_timestamp',
            'idx_tags_experiment_id', 'idx_tags_tag',
            'idx_experiments_timestamp', 'idx_experiments_status',
            'idx_experiments_git_branch'
        ]
        
        for index_name in index_names:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        
        # Drop tables in reverse dependency order
        conn.execute("DROP TABLE IF EXISTS tags")
        conn.execute("DROP TABLE IF EXISTS metrics")
        conn.execute("DROP TABLE IF EXISTS experiments")
        
        logger.info("Initial schema migration rolled back successfully")
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """Create all performance indexes as defined in subtask 1.2."""
        # Metrics table indexes
        conn.execute("CREATE INDEX idx_metrics_experiment_id ON metrics(experiment_id)")
        conn.execute("CREATE INDEX idx_metrics_experiment_metric ON metrics(experiment_id, metric_name)")
        conn.execute("CREATE INDEX idx_metrics_metric_name ON metrics(metric_name)")
        conn.execute("CREATE INDEX idx_metrics_timestamp ON metrics(timestamp)")
        
        # Tags table indexes
        conn.execute("CREATE INDEX idx_tags_experiment_id ON tags(experiment_id)")
        conn.execute("CREATE INDEX idx_tags_tag ON tags(tag)")
        
        # Experiments table indexes
        conn.execute("CREATE INDEX idx_experiments_timestamp ON experiments(timestamp)")
        conn.execute("CREATE INDEX idx_experiments_status ON experiments(status)")
        conn.execute("CREATE INDEX idx_experiments_git_branch ON experiments(git_branch)")


class MigrationManager:
    """
    Manages database migrations including tracking applied versions
    and executing pending migrations.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize migration manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.migrations: List[Migration] = []
        
        # Register built-in migrations
        self.register_migration(InitialSchemaMigration())
    
    def register_migration(self, migration: Migration):
        """
        Register a migration with the manager.
        
        Args:
            migration: Migration instance to register
        """
        self.migrations.append(migration)
        # Keep migrations sorted by version
        self.migrations.sort(key=lambda m: m.version)
    
    def _get_connection(self):
        """Get database connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _ensure_migrations_table(self, conn: sqlite3.Connection):
        """Create migrations tracking table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def get_applied_migrations(self) -> List[str]:
        """
        Get list of applied migration versions.
        
        Returns:
            List of migration versions that have been applied
        """
        with self._get_connection() as conn:
            self._ensure_migrations_table(conn)
            
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM migrations ORDER BY version")
            return [row[0] for row in cursor.fetchall()]
    
    def get_pending_migrations(self) -> List[Migration]:
        """
        Get list of migrations that haven't been applied yet.
        
        Returns:
            List of Migration objects that need to be applied
        """
        applied_versions = set(self.get_applied_migrations())
        return [m for m in self.migrations if m.version not in applied_versions]
    
    def get_current_version(self) -> Optional[str]:
        """
        Get the current schema version (latest applied migration).
        
        Returns:
            Version string of the latest applied migration, or None if no migrations applied
        """
        applied = self.get_applied_migrations()
        return applied[-1] if applied else None
    
    def apply_migration(self, migration: Migration) -> bool:
        """
        Apply a single migration.
        
        Args:
            migration: Migration to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                self._ensure_migrations_table(conn)
                
                # Check if migration is already applied
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM migrations WHERE version = ?", (migration.version,))
                if cursor.fetchone():
                    logger.warning(f"Migration {migration.version} already applied")
                    return True
                
                logger.info(f"Applying migration {migration}")
                
                # Apply the migration
                migration.up(conn)
                
                # Record the migration as applied
                cursor.execute("""
                    INSERT INTO migrations (version, description, applied_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (migration.version, migration.description))
                
                conn.commit()
                logger.info(f"Migration {migration.version} applied successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            return False
    
    def rollback_migration(self, migration: Migration) -> bool:
        """
        Rollback a single migration.
        
        Args:
            migration: Migration to rollback
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                self._ensure_migrations_table(conn)
                
                # Check if migration is applied
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM migrations WHERE version = ?", (migration.version,))
                if not cursor.fetchone():
                    logger.warning(f"Migration {migration.version} not applied, cannot rollback")
                    return True
                
                logger.info(f"Rolling back migration {migration}")
                
                # Rollback the migration
                migration.down(conn)
                
                # Remove the migration record
                cursor.execute("DELETE FROM migrations WHERE version = ?", (migration.version,))
                
                conn.commit()
                logger.info(f"Migration {migration.version} rolled back successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            return False
    
    def migrate_to_latest(self) -> bool:
        """
        Apply all pending migrations to bring database to latest schema.
        
        Returns:
            True if all migrations applied successfully, False otherwise
        """
        pending = self.get_pending_migrations()
        
        if not pending:
            logger.info("Database is up to date, no migrations to apply")
            return True
        
        logger.info(f"Applying {len(pending)} pending migrations")
        
        for migration in pending:
            if not self.apply_migration(migration):
                logger.error(f"Failed to apply migration {migration.version}, stopping")
                return False
        
        logger.info("All pending migrations applied successfully")
        return True
    
    def rollback_to_version(self, target_version: Optional[str]) -> bool:
        """
        Rollback migrations to a specific version.
        
        Args:
            target_version: Version to rollback to, or None to rollback all
            
        Returns:
            True if rollback successful, False otherwise
        """
        applied = self.get_applied_migrations()
        
        if target_version and target_version not in applied:
            logger.error(f"Target version {target_version} not found in applied migrations")
            return False
        
        # Find migrations to rollback (in reverse order)
        if target_version:
            target_index = applied.index(target_version)
            to_rollback = applied[target_index + 1:]
        else:
            to_rollback = applied
        
        to_rollback.reverse()  # Rollback in reverse order
        
        if not to_rollback:
            logger.info("No migrations to rollback")
            return True
        
        logger.info(f"Rolling back {len(to_rollback)} migrations")
        
        for version in to_rollback:
            # Find the migration object
            migration = next((m for m in self.migrations if m.version == version), None)
            if not migration:
                logger.error(f"Migration {version} not found in registered migrations")
                return False
            
            if not self.rollback_migration(migration):
                logger.error(f"Failed to rollback migration {version}, stopping")
                return False
        
        logger.info("Rollback completed successfully")
        return True
    
    def get_migration_status(self) -> dict:
        """
        Get detailed status of all migrations.
        
        Returns:
            Dictionary with migration status information
        """
        applied_versions = set(self.get_applied_migrations())
        
        status = {
            'current_version': self.get_current_version(),
            'total_migrations': len(self.migrations),
            'applied_count': len(applied_versions),
            'pending_count': len(self.get_pending_migrations()),
            'migrations': []
        }
        
        for migration in self.migrations:
            status['migrations'].append({
                'version': migration.version,
                'description': migration.description,
                'applied': migration.version in applied_versions
            })
        
        return status 