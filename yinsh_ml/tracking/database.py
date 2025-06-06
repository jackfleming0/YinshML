"""
Database module for YinshML experiment tracking.

Implements the SQLite schema as specified in Appendix A of the experiment tracking PRD.
Provides database creation, connection management, and core schema operations.
"""

import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ExperimentDatabase:
    """
    SQLite database interface for experiment tracking.
    
    Implements the schema from Appendix A with tables for experiments,
    metrics, and tags. Provides connection management and schema operations.
    """
    
    def __init__(self, db_path: Union[str, Path], use_migrations: bool = True):
        """
        Initialize the experiment database.
        
        Args:
            db_path: Path to the SQLite database file
            use_migrations: Whether to use migration system for schema management
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_migrations = use_migrations
        
        # Initialize database with schema or migrations
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with schema or migrations."""
        if self.use_migrations:
            # Use migration system for schema management
            self._initialize_with_migrations()
        else:
            # Legacy direct schema creation
            with self.get_connection() as conn:
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Create tables if they don't exist
                self._create_schema(conn)
                conn.commit()
    
    def _initialize_with_migrations(self):
        """Initialize database using the migration system."""
        try:
            from .migrations import MigrationManager
        except ImportError:
            # Handle case where relative import fails (e.g., in standalone scripts)
            from migrations import MigrationManager
        
        migration_manager = MigrationManager(str(self.db_path))
        success = migration_manager.migrate_to_latest()
        
        if not success:
            logger.error("Failed to apply migrations during database initialization")
            raise RuntimeError("Database migration failed")
        
        logger.info("Database initialized with migration system")
    
    def get_migration_manager(self):
        """
        Get the migration manager for this database.
        
        Returns:
            MigrationManager: Migration manager instance
        """
        try:
            from .migrations import MigrationManager
        except ImportError:
            # Handle case where relative import fails (e.g., in standalone scripts)
            from migrations import MigrationManager
        return MigrationManager(str(self.db_path))
    
    def _create_schema(self, conn: sqlite3.Connection):
        """
        Create the database schema according to Appendix A specifications.
        
        Args:
            conn: SQLite connection object
        """
        # Core experiment tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
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
        
        # Time-series metrics table with foreign key to experiments
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                iteration INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
        """)
        
        # Tags table for experiment organization and search
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(experiment_id, tag)
            )
        """)
        
        # Create indexes for efficient querying
        self._create_indexes(conn)
        
        logger.info("Database schema created successfully")
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """
        Create database indexes for efficient querying.
        
        Implements indexes as specified in subtask 1.2 to optimize common
        query patterns for experiment tracking operations.
        
        Args:
            conn: SQLite connection object
        """
        # Index on experiment_id in metrics table
        # Optimizes: JOIN operations between experiments and metrics
        # Query pattern: SELECT * FROM experiments e JOIN metrics m ON e.id = m.experiment_id
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id 
            ON metrics(experiment_id)
        """)
        
        # Composite index on (experiment_id, metric_name) in metrics table
        # Optimizes: Retrieving specific metrics for a given experiment
        # Query pattern: SELECT * FROM metrics WHERE experiment_id = ? AND metric_name = ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_experiment_metric 
            ON metrics(experiment_id, metric_name)
        """)
        
        # Index on metric_name for cross-experiment metric queries
        # Optimizes: Finding all experiments with a specific metric
        # Query pattern: SELECT DISTINCT experiment_id FROM metrics WHERE metric_name = ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_metric_name 
            ON metrics(metric_name)
        """)
        
        # Index on timestamp in metrics table for time-range queries
        # Optimizes: Filtering metrics by time range
        # Query pattern: SELECT * FROM metrics WHERE timestamp BETWEEN ? AND ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
            ON metrics(timestamp)
        """)
        
        # Index on experiment_id in tags table
        # Optimizes: JOIN operations between experiments and tags
        # Query pattern: SELECT * FROM experiments e JOIN tags t ON e.id = t.experiment_id
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_experiment_id 
            ON tags(experiment_id)
        """)
        
        # Index on tag for filtering experiments by tag
        # Optimizes: Finding experiments with specific tags
        # Query pattern: SELECT experiment_id FROM tags WHERE tag = ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_tag 
            ON tags(tag)
        """)
        
        # Index on timestamp in experiments table for time-range queries
        # Optimizes: Filtering experiments by creation time
        # Query pattern: SELECT * FROM experiments WHERE timestamp BETWEEN ? AND ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_timestamp 
            ON experiments(timestamp)
        """)
        
        # Index on status in experiments table for status filtering
        # Optimizes: Finding experiments by status (running, completed, failed, etc.)
        # Query pattern: SELECT * FROM experiments WHERE status = ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_status 
            ON experiments(status)
        """)
        
        # Index on git_branch for filtering by branch
        # Optimizes: Finding experiments from specific git branches
        # Query pattern: SELECT * FROM experiments WHERE git_branch = ?
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_git_branch 
            ON experiments(git_branch)
        """)
        
        logger.info("Database indexes created successfully")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection with foreign keys enabled
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA foreign_keys = ON")
            # Enable row factory for named access to columns
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def verify_schema(self) -> bool:
        """
        Verify that the database schema is correctly created.
        
        Returns:
            bool: True if schema is valid, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check that all required tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('experiments', 'metrics', 'tags')
                """)
                
                tables = [row[0] for row in cursor.fetchall()]
                required_tables = {'experiments', 'metrics', 'tags'}
                
                if not required_tables.issubset(set(tables)):
                    missing_tables = required_tables - set(tables)
                    logger.error(f"Missing tables: {missing_tables}")
                    return False
                
                # Verify foreign key constraints are enabled
                cursor.execute("PRAGMA foreign_keys")
                fk_enabled = cursor.fetchone()[0]
                if not fk_enabled:
                    logger.error("Foreign key constraints are not enabled")
                    return False
                
                # Verify indexes are created
                if not self.verify_indexes():
                    logger.error("Index verification failed")
                    return False
                
                logger.info("Database schema verification successful")
                return True
                
        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> list:
        """
        Get column information for a specific table.
        
        Args:
            table_name: Name of the table to inspect
            
        Returns:
            list: Table column information
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            return cursor.fetchall()
    
    def get_indexes_info(self) -> dict:
        """
        Get information about all indexes in the database.
        
        Returns:
            dict: Dictionary mapping table names to their indexes
        """
        indexes_info = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all indexes
            cursor.execute("""
                SELECT name, tbl_name, sql 
                FROM sqlite_master 
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
                ORDER BY tbl_name, name
            """)
            
            for row in cursor.fetchall():
                index_name, table_name, sql = row
                if table_name not in indexes_info:
                    indexes_info[table_name] = []
                indexes_info[table_name].append({
                    'name': index_name,
                    'sql': sql
                })
        
        return indexes_info
    
    def verify_indexes(self) -> bool:
        """
        Verify that all required indexes have been created.
        
        Returns:
            bool: True if all indexes exist, False otherwise
        """
        expected_indexes = {
            'metrics': [
                'idx_metrics_experiment_id',
                'idx_metrics_experiment_metric',
                'idx_metrics_metric_name',
                'idx_metrics_timestamp'
            ],
            'tags': [
                'idx_tags_experiment_id',
                'idx_tags_tag'
            ],
            'experiments': [
                'idx_experiments_timestamp',
                'idx_experiments_status',
                'idx_experiments_git_branch'
            ]
        }
        
        indexes_info = self.get_indexes_info()
        
        for table, expected_idx_list in expected_indexes.items():
            if table not in indexes_info:
                logger.error(f"No indexes found for table {table}")
                return False
            
            actual_indexes = [idx['name'] for idx in indexes_info[table]]
            
            for expected_idx in expected_idx_list:
                if expected_idx not in actual_indexes:
                    logger.error(f"Missing index {expected_idx} for table {table}")
                    return False
        
        logger.info("All required indexes verified successfully")
        return True
    
    def close(self):
        """Clean up database resources."""
        # Context manager handles connection cleanup automatically
        pass


def create_database(db_path: Union[str, Path], use_migrations: bool = True) -> ExperimentDatabase:
    """
    Create and initialize a new experiment tracking database.
    
    Args:
        db_path: Path where the SQLite database should be created
        use_migrations: Whether to use migration system for schema management (default: True)
        
    Returns:
        ExperimentDatabase: Initialized database instance
    """
    logger.info(f"Creating experiment database at {db_path}")
    db = ExperimentDatabase(db_path, use_migrations=use_migrations)
    
    # Verify the schema was created correctly
    if not db.verify_schema():
        raise RuntimeError("Failed to create valid database schema")
    
    logger.info("Experiment database created and verified successfully")
    return db 