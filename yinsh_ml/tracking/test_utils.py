"""
Unit tests for database utility functions.
"""

import unittest
import tempfile
import os
import json
import threading
import time
from datetime import datetime, date
from pathlib import Path

from .utils import (
    DatabaseConnectionManager, initialize_database, get_connection, transaction_context,
    with_transaction, validate_experiment_data, create_experiment, add_tags_to_experiment,
    add_metric_to_experiment, add_metrics_bulk, query_experiments, get_experiment_by_id,
    get_experiment_metrics, get_experiment_tags, update_experiment_status, 
    delete_experiment, get_database_stats, close_all_connections, set_database_path,
    DatabaseError, ConnectionError, TransactionError, ValidationError
)
from .database import ExperimentDatabase


class TestDatabaseConnectionManager(unittest.TestCase):
    """Test the database connection manager singleton."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_utils.db")
        # Reset singleton for clean testing
        DatabaseConnectionManager._instance = None
    
    def tearDown(self):
        close_all_connections()
        if os.path.exists(self.test_db):
            os.unlink(self.test_db)
        os.rmdir(self.temp_dir)
        # Reset singleton after test
        DatabaseConnectionManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that DatabaseConnectionManager is a singleton."""
        manager1 = DatabaseConnectionManager(self.test_db)
        manager2 = DatabaseConnectionManager()
        
        self.assertIs(manager1, manager2)
        self.assertEqual(manager1.db_path, self.test_db)
    
    def test_connection_creation(self):
        """Test connection creation and management."""
        # Initialize database first
        initialize_database(self.test_db)
        
        manager = DatabaseConnectionManager(self.test_db)
        conn = manager.get_connection()
        
        self.assertIsNotNone(conn)
        self.assertEqual(manager.get_connection_count(), 1)
        
        # Same thread should get same connection
        conn2 = manager.get_connection()
        self.assertIs(conn, conn2)
        self.assertEqual(manager.get_connection_count(), 1)
    
    def test_connection_thread_safety(self):
        """Test that different threads get different connections."""
        initialize_database(self.test_db)
        manager = DatabaseConnectionManager(self.test_db)
        
        connections = {}
        
        def get_connection_in_thread(thread_id):
            connections[thread_id] = manager.get_connection()
        
        # Create connections in different threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=get_connection_in_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have a different connection
        self.assertEqual(len(connections), 3)
        conn_ids = [id(conn) for conn in connections.values()]
        self.assertEqual(len(set(conn_ids)), 3)  # All unique
    
    def test_connection_optimization(self):
        """Test that connections have proper SQLite optimizations."""
        initialize_database(self.test_db)
        manager = DatabaseConnectionManager(self.test_db)
        conn = manager.get_connection()
        
        # Test that row factory is set
        self.assertEqual(conn.row_factory, manager._connections[threading.get_ident()].row_factory)
        
        # Test some PRAGMA settings
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        self.assertEqual(cursor.fetchone()[0], 1)  # Should be enabled
        
        cursor.execute("PRAGMA journal_mode")
        self.assertEqual(cursor.fetchone()[0], "wal")  # Should be WAL mode
    
    def test_close_connections(self):
        """Test closing connections."""
        initialize_database(self.test_db)
        manager = DatabaseConnectionManager(self.test_db)
        
        # Create a connection
        conn = manager.get_connection()
        self.assertEqual(manager.get_connection_count(), 1)
        
        # Close all connections
        manager.close_all_connections()
        self.assertEqual(manager.get_connection_count(), 0)
    
    def test_path_change(self):
        """Test changing database path closes existing connections."""
        initialize_database(self.test_db)
        manager = DatabaseConnectionManager(self.test_db)
        
        # Create a connection
        conn = manager.get_connection()
        self.assertEqual(manager.get_connection_count(), 1)
        
        # Change path
        new_db = os.path.join(self.temp_dir, "new_test.db")
        manager.set_database_path(new_db)
        
        # Connections should be closed
        self.assertEqual(manager.get_connection_count(), 0)
        self.assertEqual(manager.db_path, new_db)


class TestUtilityFunctions(unittest.TestCase):
    """Test database utility functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_utils.db")
        
        # Initialize database
        self.db = initialize_database(self.test_db)
        
        # Sample data for testing
        self.sample_config = {
            "model": "ResNet50",
            "batch_size": 32,
            "learning_rate": 0.001
        }
        self.sample_environment = {
            "python_version": "3.9.0",
            "pytorch_version": "1.10.0",
            "cuda_version": "11.2"
        }
    
    def tearDown(self):
        close_all_connections()
        if os.path.exists(self.test_db):
            os.unlink(self.test_db)
        os.rmdir(self.temp_dir)
        # Reset singleton
        DatabaseConnectionManager._instance = None
    
    def test_initialize_database(self):
        """Test database initialization."""
        # Database should be initialized and schema verified
        self.assertTrue(self.db.verify_schema())
        
        # Should be able to get database stats
        stats = get_database_stats()
        self.assertIn('experiment_count', stats)
        self.assertEqual(stats['experiment_count'], 0)
    
    def test_validation_functions(self):
        """Test data validation."""
        # Valid data should pass
        validate_experiment_data(
            "test_exp", "abc123", "main", 
            self.sample_config, self.sample_environment
        )
        
        # Invalid data should raise ValidationError
        with self.assertRaises(ValidationError):
            validate_experiment_data("", "abc123", "main", {}, {})  # Empty name
        
        with self.assertRaises(ValidationError):
            validate_experiment_data("test", "", "main", {}, {})  # Empty commit
        
        with self.assertRaises(ValidationError):
            validate_experiment_data("test", "abc123", "", {}, {})  # Empty branch
        
        with self.assertRaises(ValidationError):
            validate_experiment_data("test", "abc123", "main", "not_dict", {})  # Invalid config
        
        with self.assertRaises(ValidationError):
            validate_experiment_data("test", "abc123", "main", {}, "not_dict")  # Invalid environment
    
    def test_create_experiment(self):
        """Test experiment creation."""
        exp_id = create_experiment(
            "test_experiment", 
            "abc123def456", 
            "feature_branch",
            self.sample_config,
            self.sample_environment,
            status="running",
            notes="Test experiment",
            tags=["test", "resnet"]
        )
        
        self.assertIsInstance(exp_id, int)
        self.assertGreater(exp_id, 0)
        
        # Verify experiment was created
        exp = get_experiment_by_id(exp_id)
        self.assertIsNotNone(exp)
        self.assertEqual(exp['name'], "test_experiment")
        self.assertEqual(exp['git_commit'], "abc123def456")
        self.assertEqual(exp['git_branch'], "feature_branch")
        self.assertEqual(exp['status'], "running")
        self.assertEqual(exp['notes'], "Test experiment")
        self.assertEqual(exp['config'], self.sample_config)
        self.assertEqual(exp['environment'], self.sample_environment)
        
        # Verify tags were added
        tags = get_experiment_tags(exp_id)
        self.assertEqual(set(tags), {"test", "resnet"})
    
    def test_add_metric_to_experiment(self):
        """Test adding metrics to experiments."""
        # Create experiment first
        exp_id = create_experiment(
            "metric_test", "abc123", "main", 
            self.sample_config, self.sample_environment
        )
        
        # Add metric
        metric_id = add_metric_to_experiment(
            exp_id, "accuracy", 0.95, 10
        )
        
        self.assertIsInstance(metric_id, int)
        self.assertGreater(metric_id, 0)
        
        # Verify metric was added
        metrics = get_experiment_metrics(exp_id)
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertEqual(metric['experiment_id'], exp_id)
        self.assertEqual(metric['metric_name'], "accuracy")
        self.assertEqual(metric['metric_value'], 0.95)
        self.assertEqual(metric['iteration'], 10)
        
        # Test metric validation
        with self.assertRaises(ValidationError):
            add_metric_to_experiment(exp_id, "", 0.95, 10)  # Empty name
        
        with self.assertRaises(ValidationError):
            add_metric_to_experiment(exp_id, "loss", "not_numeric", 10)  # Non-numeric value
        
        with self.assertRaises(ValidationError):
            add_metric_to_experiment(exp_id, "loss", 0.5, -1)  # Negative iteration
    
    def test_add_metrics_bulk(self):
        """Test bulk metric addition."""
        # Create experiment
        exp_id = create_experiment(
            "bulk_test", "abc123", "main",
            self.sample_config, self.sample_environment
        )
        
        # Prepare bulk metrics
        metrics = [
            {"name": "accuracy", "value": 0.95, "iteration": 1},
            {"name": "loss", "value": 0.05, "iteration": 1},
            {"name": "accuracy", "value": 0.97, "iteration": 2},
            {"name": "loss", "value": 0.03, "iteration": 2}
        ]
        
        metric_ids = add_metrics_bulk(exp_id, metrics)
        
        self.assertEqual(len(metric_ids), 4)
        self.assertTrue(all(isinstance(mid, int) for mid in metric_ids))
        
        # Verify all metrics were added
        all_metrics = get_experiment_metrics(exp_id)
        self.assertEqual(len(all_metrics), 4)
        
        # Test filtering by metric name
        accuracy_metrics = get_experiment_metrics(exp_id, "accuracy")
        self.assertEqual(len(accuracy_metrics), 2)
        self.assertTrue(all(m['metric_name'] == "accuracy" for m in accuracy_metrics))
    
    def test_experiment_querying(self):
        """Test experiment querying with various filters."""
        # Create multiple experiments with different attributes
        exp1_id = create_experiment(
            "exp1", "commit1", "main", {"type": "train"}, {},
            status="completed", tags=["resnet", "cifar10"]
        )
        
        exp2_id = create_experiment(
            "exp2", "commit2", "feature", {"type": "eval"}, {},
            status="running", tags=["vgg", "cifar10"]
        )
        
        exp3_id = create_experiment(
            "exp3", "commit3", "main", {"type": "train"}, {},
            status="failed", tags=["resnet", "imagenet"]
        )
        
        # Test query by status
        completed_exps = query_experiments(status="completed")
        self.assertEqual(len(completed_exps), 1)
        self.assertEqual(completed_exps[0]['id'], exp1_id)
        
        # Test query by tags
        resnet_exps = query_experiments(tags=["resnet"])
        self.assertEqual(len(resnet_exps), 2)
        resnet_ids = {exp['id'] for exp in resnet_exps}
        self.assertEqual(resnet_ids, {exp1_id, exp3_id})
        
        # Test query by multiple tags (AND logic)
        resnet_cifar_exps = query_experiments(tags=["resnet", "cifar10"])
        self.assertEqual(len(resnet_cifar_exps), 1)
        self.assertEqual(resnet_cifar_exps[0]['id'], exp1_id)
        
        # Test pagination
        all_exps = query_experiments(limit=2)
        self.assertEqual(len(all_exps), 2)
        
        remaining_exps = query_experiments(limit=2, offset=2)
        self.assertEqual(len(remaining_exps), 1)
        
        # Test date filtering (today's experiments)
        today = date.today()
        today_exps = query_experiments(start_date=today, end_date=today)
        self.assertEqual(len(today_exps), 3)  # All created today
    
    def test_experiment_management(self):
        """Test experiment status updates and deletion."""
        # Create experiment
        exp_id = create_experiment(
            "mgmt_test", "abc123", "main",
            self.sample_config, self.sample_environment,
            status="running"
        )
        
        # Update status
        update_experiment_status(exp_id, "completed")
        
        exp = get_experiment_by_id(exp_id)
        self.assertEqual(exp['status'], "completed")
        
        # Test invalid status update
        with self.assertRaises(ValidationError):
            update_experiment_status(exp_id, "")  # Empty status
        
        # Delete experiment
        delete_experiment(exp_id)
        
        # Should not be found after deletion
        exp = get_experiment_by_id(exp_id)
        self.assertIsNone(exp)
        
        # Should raise error for non-existent experiment
        with self.assertRaises(DatabaseError):
            delete_experiment(exp_id)
    
    def test_tags_management(self):
        """Test tag operations."""
        # Create experiment
        exp_id = create_experiment(
            "tag_test", "abc123", "main",
            self.sample_config, self.sample_environment
        )
        
        # Add initial tags
        with get_connection() as conn:
            add_tags_to_experiment(conn, exp_id, ["initial", "test"])
        
        tags = get_experiment_tags(exp_id)
        self.assertEqual(set(tags), {"initial", "test"})
        
        # Add more tags (including duplicate)
        with get_connection() as conn:
            add_tags_to_experiment(conn, exp_id, ["new", "test"])  # "test" is duplicate
        
        tags = get_experiment_tags(exp_id)
        self.assertEqual(set(tags), {"initial", "test", "new"})  # No duplicate
        
        # Test tag validation
        with get_connection() as conn:
            with self.assertRaises(ValidationError):
                add_tags_to_experiment(conn, exp_id, [""])  # Empty tag
            
            with self.assertRaises(ValidationError):
                add_tags_to_experiment(conn, exp_id, "not_a_list")  # Not a list
    
    def test_database_stats(self):
        """Test database statistics."""
        # Get initial stats
        stats = get_database_stats()
        initial_exp_count = stats['experiment_count']
        
        # Create some test data
        exp1_id = create_experiment(
            "stats_test1", "abc123", "main",
            self.sample_config, self.sample_environment,
            status="completed", tags=["test", "stats"]
        )
        
        exp2_id = create_experiment(
            "stats_test2", "def456", "feature",
            self.sample_config, self.sample_environment,
            status="running", tags=["test"]
        )
        
        # Add some metrics
        add_metric_to_experiment(exp1_id, "accuracy", 0.95, 1)
        add_metric_to_experiment(exp1_id, "loss", 0.05, 1)
        add_metric_to_experiment(exp2_id, "accuracy", 0.90, 1)
        
        # Get updated stats
        stats = get_database_stats()
        
        self.assertEqual(stats['experiment_count'], initial_exp_count + 2)
        self.assertEqual(stats['metric_count'], 3)
        self.assertGreater(stats['tag_count'], 0)
        
        # Check status distribution
        self.assertIn('status_distribution', stats)
        self.assertIn('completed', stats['status_distribution'])
        self.assertIn('running', stats['status_distribution'])
        
        # Check top tags
        self.assertIn('top_tags', stats)
        self.assertIn('test', stats['top_tags'])
        
        # Check database size
        self.assertIn('database_size_bytes', stats)
        self.assertGreater(stats['database_size_bytes'], 0)
        
        # Check connection count
        self.assertIn('active_connections', stats)
        self.assertGreaterEqual(stats['active_connections'], 0)
    
    def test_transaction_handling(self):
        """Test transaction context and decorators."""
        
        @with_transaction
        def create_exp_with_metrics(conn, name):
            """Helper function that uses transaction decorator."""
            # This should all happen in one transaction
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, git_commit, git_branch, status, config_json, environment_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, "abc123", "main", "running", "{}", "{}"))
            
            exp_id = cursor.lastrowid
            
            # Add a metric
            cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value, iteration)
                VALUES (?, ?, ?, ?)
            """, (exp_id, "test_metric", 1.0, 1))
            
            return exp_id
        
        # Test successful transaction
        exp_id = create_exp_with_metrics("transaction_test")
        self.assertIsInstance(exp_id, int)
        
        # Verify both experiment and metric were created
        exp = get_experiment_by_id(exp_id)
        self.assertIsNotNone(exp)
        
        metrics = get_experiment_metrics(exp_id)
        self.assertEqual(len(metrics), 1)
        
        # Test transaction rollback
        @with_transaction
        def failing_transaction(conn):
            # Create an experiment
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, git_commit, git_branch, status, config_json, environment_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("failing_test", "abc123", "main", "running", "{}", "{}"))
            
            exp_id = cursor.lastrowid
            
            # This should cause a rollback
            raise Exception("Intentional failure")
        
        # This should raise TransactionError and rollback
        with self.assertRaises(TransactionError):
            failing_transaction()
        
        # Verify no experiment was created due to rollback
        failing_exps = query_experiments()
        failing_names = [exp['name'] for exp in failing_exps]
        self.assertNotIn("failing_test", failing_names)
    
    def test_error_handling(self):
        """Test various error conditions."""
        # Test connection without database path
        DatabaseConnectionManager._instance = None
        manager = DatabaseConnectionManager()
        
        with self.assertRaises(ConnectionError):
            manager.get_connection()
        
        # Test operations on non-existent experiment
        with self.assertRaises(DatabaseError):
            update_experiment_status(99999, "completed")
        
        with self.assertRaises(DatabaseError):
            delete_experiment(99999)
        
        # Test invalid database path
        with self.assertRaises(DatabaseError):
            initialize_database("/invalid/path/database.db")


class TestConcurrency(unittest.TestCase):
    """Test concurrent database operations."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_concurrent.db")
        self.db = initialize_database(self.test_db)
        
        self.sample_config = {"test": True}
        self.sample_environment = {"test": True}
    
    def tearDown(self):
        close_all_connections()
        if os.path.exists(self.test_db):
            os.unlink(self.test_db)
        os.rmdir(self.temp_dir)
        DatabaseConnectionManager._instance = None
    
    def test_concurrent_experiment_creation(self):
        """Test creating experiments concurrently."""
        results = []
        errors = []
        
        def create_experiment_thread(thread_id):
            try:
                exp_id = create_experiment(
                    f"concurrent_test_{thread_id}",
                    f"commit_{thread_id}",
                    "main",
                    self.sample_config,
                    self.sample_environment
                )
                results.append(exp_id)
            except Exception as e:
                errors.append(e)
        
        # Create experiments in multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_experiment_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All experiments should be created successfully
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 5)
        self.assertEqual(len(set(results)), 5)  # All unique IDs
        
        # Verify all experiments exist
        for exp_id in results:
            exp = get_experiment_by_id(exp_id)
            self.assertIsNotNone(exp)
    
    def test_concurrent_metric_addition(self):
        """Test adding metrics concurrently to the same experiment."""
        # Create an experiment
        exp_id = create_experiment(
            "metric_concurrent_test", "abc123", "main",
            self.sample_config, self.sample_environment
        )
        
        results = []
        errors = []
        
        def add_metric_thread(thread_id):
            try:
                metric_id = add_metric_to_experiment(
                    exp_id, f"metric_{thread_id}", float(thread_id), thread_id
                )
                results.append(metric_id)
            except Exception as e:
                errors.append(e)
        
        # Add metrics in multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=add_metric_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All metrics should be added successfully
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 10)
        
        # Verify all metrics exist
        metrics = get_experiment_metrics(exp_id)
        self.assertEqual(len(metrics), 10)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main() 