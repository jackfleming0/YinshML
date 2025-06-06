"""
Database utility functions for YinshML experiment tracking.

Provides high-level utilities for database operations including connection management,
transaction handling, common operations, and SQLite optimizations.
"""

import sqlite3
import logging
import json
import threading
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Tuple
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

try:
    from .database import ExperimentDatabase
except ImportError:
    from database import ExperimentDatabase

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Exception for database connection issues."""
    pass


class TransactionError(DatabaseError):
    """Exception for transaction handling issues."""
    pass


class ValidationError(DatabaseError):
    """Exception for data validation issues."""
    pass


class DatabaseConnectionManager:
    """
    Singleton connection manager for efficient database connection reuse.
    
    Provides thread-safe connection management with automatic cleanup
    and SQLite optimizations.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseConnectionManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = None):
        if not self._initialized:
            self.db_path = db_path
            self._connections = {}
            self._connection_lock = threading.Lock()
            self._initialized = True
            logger.info("DatabaseConnectionManager initialized")
        elif db_path is not None and db_path != self.db_path:
            # Update database path even if already initialized
            self.set_database_path(db_path)
    
    def set_database_path(self, db_path: str):
        """Set the database path for future connections."""
        with self._connection_lock:
            if self.db_path != db_path:
                # Close existing connections when changing database
                self._close_all_connections_unsafe()
                self.db_path = db_path
                logger.info(f"Database path set to: {db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection for the current thread.
        
        Returns:
            sqlite3.Connection: Database connection with optimizations applied
            
        Raises:
            ConnectionError: If unable to connect to database
        """
        if not self.db_path:
            raise ConnectionError("Database path not set")
        
        thread_id = threading.get_ident()
        
        with self._connection_lock:
            if thread_id not in self._connections:
                try:
                    conn = sqlite3.connect(
                        self.db_path,
                        timeout=30.0,  # 30 second timeout
                        check_same_thread=False
                    )
                    
                    # Apply SQLite optimizations
                    self._configure_connection(conn)
                    
                    self._connections[thread_id] = conn
                    logger.debug(f"Created new connection for thread {thread_id}")
                    
                except sqlite3.Error as e:
                    raise ConnectionError(f"Failed to connect to database: {e}")
            
            return self._connections[thread_id]
    
    def _configure_connection(self, conn: sqlite3.Connection):
        """Apply SQLite optimizations and settings to a connection."""
        try:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            
            # Optimize for performance
            conn.execute("PRAGMA synchronous = NORMAL")  # Balance safety and speed
            conn.execute("PRAGMA cache_size = 10000")    # 10MB cache
            conn.execute("PRAGMA temp_store = MEMORY")    # Store temp tables in memory
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory mapping
            
            # Set row factory for named access
            conn.row_factory = sqlite3.Row
            
            logger.debug("Applied SQLite optimizations to connection")
            
        except sqlite3.Error as e:
            logger.warning(f"Failed to apply some SQLite optimizations: {e}")
    
    def close_connection(self, thread_id: int = None):
        """Close connection for a specific thread (current thread if not specified)."""
        if thread_id is None:
            thread_id = threading.get_ident()
        
        with self._connection_lock:
            if thread_id in self._connections:
                self._connections[thread_id].close()
                del self._connections[thread_id]
                logger.debug(f"Closed connection for thread {thread_id}")
    
    def _close_all_connections_unsafe(self):
        """Close all active connections without acquiring lock. Internal use only."""
        for thread_id, conn in list(self._connections.items()):
            try:
                conn.close()
                logger.debug(f"Closed connection for thread {thread_id}")
            except sqlite3.Error as e:
                logger.warning(f"Error closing connection for thread {thread_id}: {e}")
        
        self._connections.clear()
        logger.info("All database connections closed")
    
    def close_all_connections(self):
        """Close all active connections."""
        with self._connection_lock:
            self._close_all_connections_unsafe()
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        with self._connection_lock:
            return len(self._connections)


# Global connection manager instance
_connection_manager = DatabaseConnectionManager()


def initialize_database(db_path: str, use_migrations: bool = True) -> ExperimentDatabase:
    """
    Initialize the experiment tracking database with optimizations.
    
    Args:
        db_path: Path to the SQLite database file
        use_migrations: Whether to use migration system for schema management
        
    Returns:
        ExperimentDatabase: Initialized database instance
        
    Raises:
        DatabaseError: If initialization fails
    """
    try:
        # Set the database path in the connection manager
        _connection_manager.set_database_path(db_path)
        
        # Create and initialize the database
        db = ExperimentDatabase(db_path, use_migrations=use_migrations)
        
        # Verify schema is correct
        if not db.verify_schema():
            raise DatabaseError("Database schema verification failed")
        
        logger.info(f"Database initialized successfully at {db_path}")
        return db
        
    except Exception as e:
        raise DatabaseError(f"Failed to initialize database: {e}")


@contextmanager
def get_connection():
    """
    Context manager for getting a database connection.
    
    Yields:
        sqlite3.Connection: Database connection
        
    Raises:
        ConnectionError: If unable to get connection
    """
    try:
        conn = _connection_manager.get_connection()
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise ConnectionError(f"Failed to get database connection: {e}")


@contextmanager
def transaction_context():
    """
    Context manager for database transactions.
    
    Automatically commits on success or rolls back on error.
    
    Yields:
        sqlite3.Connection: Database connection in transaction
        
    Raises:
        TransactionError: If transaction fails
    """
    try:
        with get_connection() as conn:
            conn.execute("BEGIN")
            try:
                yield conn
                conn.commit()
                logger.debug("Transaction committed successfully")
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back due to error: {e}")
                raise TransactionError(f"Transaction failed: {e}")
    except ConnectionError:
        raise
    except Exception as e:
        raise TransactionError(f"Transaction setup failed: {e}")


def with_transaction(func):
    """
    Decorator to wrap a function in a database transaction.
    
    The decorated function will receive a 'conn' parameter as the first argument.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with transaction_context() as conn:
            return func(conn, *args, **kwargs)
    return wrapper


def validate_experiment_data(name: str, git_commit: str, git_branch: str, 
                           config: Dict[str, Any], environment: Dict[str, Any]) -> None:
    """
    Validate experiment data before insertion.
    
    Args:
        name: Experiment name
        git_commit: Git commit hash
        git_branch: Git branch name
        config: Configuration dictionary
        environment: Environment dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Experiment name must be a non-empty string")
    
    if not git_commit or not isinstance(git_commit, str):
        raise ValidationError("Git commit must be a non-empty string")
    
    if not git_branch or not isinstance(git_branch, str):
        raise ValidationError("Git branch must be a non-empty string")
    
    if not isinstance(config, dict):
        raise ValidationError("Config must be a dictionary")
    
    if not isinstance(environment, dict):
        raise ValidationError("Environment must be a dictionary")
    
    # Validate JSON serializable
    try:
        json.dumps(config)
        json.dumps(environment)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Config or environment not JSON serializable: {e}")


@with_transaction
def create_experiment(conn: sqlite3.Connection, name: str, git_commit: str, git_branch: str,
                     config: Dict[str, Any], environment: Dict[str, Any],
                     status: str = "running", notes: str = None, 
                     tags: List[str] = None) -> int:
    """
    Create a new experiment with optional tags.
    
    Args:
        conn: Database connection (provided by decorator)
        name: Experiment name
        git_commit: Git commit hash
        git_branch: Git branch name
        config: Configuration dictionary
        environment: Environment dictionary
        status: Experiment status (default: "running")
        notes: Optional notes
        tags: Optional list of tags
        
    Returns:
        int: ID of the created experiment
        
    Raises:
        ValidationError: If data validation fails
        DatabaseError: If database operation fails
    """
    try:
        # Validate input data
        validate_experiment_data(name, git_commit, git_branch, config, environment)
        
        # Serialize dictionaries to JSON
        config_json = json.dumps(config)
        environment_json = json.dumps(environment)
        
        # Insert experiment
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (name, git_commit, git_branch, status, 
                                   config_json, environment_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, git_commit, git_branch, status, config_json, environment_json, notes))
        
        experiment_id = cursor.lastrowid
        
        # Add tags if provided
        if tags:
            add_tags_to_experiment(conn, experiment_id, tags)
        
        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id
        
    except ValidationError:
        raise
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to create experiment: {e}")


def add_tags_to_experiment(conn: sqlite3.Connection, experiment_id: int, 
                          tags: List[str]) -> None:
    """
    Add tags to an experiment.
    
    Args:
        conn: Database connection
        experiment_id: ID of the experiment
        tags: List of tag strings
        
    Raises:
        ValidationError: If validation fails
        DatabaseError: If database operation fails
    """
    try:
        if not isinstance(tags, list):
            raise ValidationError("Tags must be a list")
        
        if not tags:
            return  # Nothing to add
        
        # Validate tags are strings
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValidationError("All tags must be non-empty strings")
        
        # Insert tags (ignore duplicates due to UNIQUE constraint)
        cursor = conn.cursor()
        for tag in tags:
            try:
                cursor.execute("""
                    INSERT INTO tags (experiment_id, tag)
                    VALUES (?, ?)
                """, (experiment_id, tag.strip()))
            except sqlite3.IntegrityError:
                # Tag already exists for this experiment, skip
                logger.debug(f"Tag '{tag}' already exists for experiment {experiment_id}")
        
        logger.debug(f"Added {len(tags)} tags to experiment {experiment_id}")
        
    except ValidationError:
        raise
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to add tags: {e}")


@with_transaction
def add_metric_to_experiment(conn: sqlite3.Connection, experiment_id: int, 
                           metric_name: str, metric_value: float, 
                           iteration: int, timestamp: datetime = None) -> int:
    """
    Add a metric value to an experiment.
    
    Args:
        conn: Database connection (provided by decorator)
        experiment_id: ID of the experiment
        metric_name: Name of the metric
        metric_value: Numeric value of the metric
        iteration: Iteration number
        timestamp: Optional timestamp (defaults to current time)
        
    Returns:
        int: ID of the created metric record
        
    Raises:
        ValidationError: If validation fails
        DatabaseError: If database operation fails
    """
    try:
        # Validate inputs
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValidationError("Metric name must be a non-empty string")
        
        if not isinstance(metric_value, (int, float)):
            raise ValidationError("Metric value must be numeric")
        
        if not isinstance(iteration, int) or iteration < 0:
            raise ValidationError("Iteration must be a non-negative integer")
        
        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Insert metric
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO metrics (experiment_id, metric_name, metric_value, iteration, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (experiment_id, metric_name.strip(), float(metric_value), iteration, timestamp))
        
        metric_id = cursor.lastrowid
        logger.debug(f"Added metric {metric_name}={metric_value} to experiment {experiment_id}")
        
        return metric_id
        
    except ValidationError:
        raise
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to add metric: {e}")


def add_metrics_bulk(experiment_id: int, metrics: List[Dict[str, Any]]) -> List[int]:
    """
    Add multiple metrics to an experiment in a single transaction.
    
    Args:
        experiment_id: ID of the experiment
        metrics: List of metric dictionaries with keys: name, value, iteration, timestamp (optional)
        
    Returns:
        List[int]: List of metric IDs
        
    Raises:
        ValidationError: If validation fails
        DatabaseError: If database operation fails
    """
    try:
        with transaction_context() as conn:
            metric_ids = []
            
            for metric in metrics:
                if not isinstance(metric, dict):
                    raise ValidationError("Each metric must be a dictionary")
                
                required_keys = {'name', 'value', 'iteration'}
                if not required_keys.issubset(metric.keys()):
                    raise ValidationError(f"Metric must contain keys: {required_keys}")
                
                timestamp = metric.get('timestamp')
                if timestamp and isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                # Use the add_metric_to_experiment logic without transaction decorator
                metric_id = add_metric_to_experiment.__wrapped__(
                    conn, experiment_id, metric['name'], metric['value'], 
                    metric['iteration'], timestamp
                )
                metric_ids.append(metric_id)
            
            logger.info(f"Added {len(metrics)} metrics to experiment {experiment_id}")
            return metric_ids
            
    except ValidationError:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to add metrics in bulk: {e}")


def query_experiments(status: str = None, tags: List[str] = None, 
                     start_date: date = None, end_date: date = None,
                     limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query experiments with optional filtering.
    
    Args:
        status: Filter by experiment status
        tags: Filter by tags (experiment must have ALL specified tags)
        start_date: Filter experiments created after this date
        end_date: Filter experiments created before this date
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List[Dict[str, Any]]: List of experiment dictionaries
        
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT DISTINCT e.* FROM experiments e"
            params = []
            conditions = []
            
            # Join with tags if needed
            if tags:
                query += " JOIN tags t ON e.id = t.experiment_id"
            
            # Add conditions
            if status:
                conditions.append("e.status = ?")
                params.append(status)
            
            if tags:
                # Create condition for having all specified tags
                tag_conditions = " AND ".join(["t.tag = ?"] * len(tags))
                conditions.append(f"({tag_conditions})")
                params.extend(tags)
                
                # Add GROUP BY and HAVING for tag count matching
                query += f" WHERE {' AND '.join(conditions)}"
                query += " GROUP BY e.id HAVING COUNT(DISTINCT t.tag) = ?"
                params.append(len(tags))
            elif conditions:
                query += f" WHERE {' AND '.join(conditions)}"
            
            # Add date filtering
            if start_date or end_date:
                date_conditions = []
                if start_date:
                    date_conditions.append("DATE(e.timestamp) >= ?")
                    params.append(start_date.isoformat())
                if end_date:
                    date_conditions.append("DATE(e.timestamp) <= ?")
                    params.append(end_date.isoformat())
                
                if conditions or tags:
                    if not tags:  # Only add WHERE if not already added
                        query += " WHERE " + " AND ".join(date_conditions)
                    else:
                        # Need to modify the query structure for date conditions with tags
                        query = query.replace(" GROUP BY", f" AND {' AND '.join(date_conditions)} GROUP BY")
                else:
                    query += " WHERE " + " AND ".join(date_conditions)
            
            # Add ordering and pagination
            query += " ORDER BY e.timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries and parse JSON fields
            experiments = []
            for row in rows:
                exp = dict(row)
                exp['config'] = json.loads(exp['config_json'])
                exp['environment'] = json.loads(exp['environment_json'])
                del exp['config_json']
                del exp['environment_json']
                experiments.append(exp)
            
            logger.debug(f"Found {len(experiments)} experiments matching criteria")
            return experiments
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to query experiments: {e}")


def get_experiment_by_id(experiment_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific experiment by ID.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Optional[Dict[str, Any]]: Experiment dictionary or None if not found
        
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert to dictionary and parse JSON
            exp = dict(row)
            exp['config'] = json.loads(exp['config_json'])
            exp['environment'] = json.loads(exp['environment_json'])
            del exp['config_json']
            del exp['environment_json']
            
            return exp
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to get experiment {experiment_id}: {e}")


def get_experiment_metrics(experiment_id: int, metric_name: str = None) -> List[Dict[str, Any]]:
    """
    Get metrics for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        metric_name: Optional filter by metric name
        
    Returns:
        List[Dict[str, Any]]: List of metric dictionaries
        
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if metric_name:
                cursor.execute("""
                    SELECT * FROM metrics 
                    WHERE experiment_id = ? AND metric_name = ?
                    ORDER BY iteration, timestamp
                """, (experiment_id, metric_name))
            else:
                cursor.execute("""
                    SELECT * FROM metrics 
                    WHERE experiment_id = ?
                    ORDER BY metric_name, iteration, timestamp
                """, (experiment_id,))
            
            return [dict(row) for row in cursor.fetchall()]
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to get metrics for experiment {experiment_id}: {e}")


def get_experiment_tags(experiment_id: int) -> List[str]:
    """
    Get tags for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        List[str]: List of tag strings
        
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tag FROM tags WHERE experiment_id = ? ORDER BY tag", (experiment_id,))
            return [row[0] for row in cursor.fetchall()]
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to get tags for experiment {experiment_id}: {e}")


def update_experiment_status(experiment_id: int, status: str) -> None:
    """
    Update the status of an experiment.
    
    Args:
        experiment_id: ID of the experiment
        status: New status
        
    Raises:
        ValidationError: If validation fails
        DatabaseError: If update fails
    """
    try:
        if not isinstance(status, str) or not status.strip():
            raise ValidationError("Status must be a non-empty string")
        
        with transaction_context() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments SET status = ? WHERE id = ?
            """, (status.strip(), experiment_id))
            
            if cursor.rowcount == 0:
                raise DatabaseError(f"Experiment {experiment_id} not found")
            
            logger.info(f"Updated experiment {experiment_id} status to {status}")
            
    except ValidationError:
        raise
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to update experiment status: {e}")


def delete_experiment(experiment_id: int) -> None:
    """
    Delete an experiment and all associated data.
    
    Args:
        experiment_id: ID of the experiment to delete
        
    Raises:
        DatabaseError: If deletion fails
    """
    try:
        with transaction_context() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            
            if cursor.rowcount == 0:
                raise DatabaseError(f"Experiment {experiment_id} not found")
            
            logger.info(f"Deleted experiment {experiment_id} and associated data")
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to delete experiment {experiment_id}: {e}")


def get_database_stats() -> Dict[str, Any]:
    """
    Get database statistics and information.
    
    Returns:
        Dict[str, Any]: Database statistics
        
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Table counts
            cursor.execute("SELECT COUNT(*) FROM experiments")
            stats['experiment_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM metrics")
            stats['metric_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM tags")
            stats['tag_count'] = cursor.fetchone()[0]
            
            # Status distribution
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM experiments 
                GROUP BY status 
                ORDER BY COUNT(*) DESC
            """)
            stats['status_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Most common tags
            cursor.execute("""
                SELECT tag, COUNT(*) as count
                FROM tags 
                GROUP BY tag 
                ORDER BY count DESC 
                LIMIT 10
            """)
            stats['top_tags'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Database size info
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['database_size_bytes'] = page_count * page_size
            
            # Connection info
            stats['active_connections'] = _connection_manager.get_connection_count()
            
            return stats
            
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to get database statistics: {e}")


def close_all_connections():
    """Close all database connections."""
    _connection_manager.close_all_connections()


def set_database_path(db_path: str):
    """Set the database path for the connection manager."""
    _connection_manager.set_database_path(db_path) 