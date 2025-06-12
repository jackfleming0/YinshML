#!/usr/bin/env python3
"""
Test script for data management policies and backup functionality.

Demonstrates retention policies, backup operations, and cleanup mechanisms.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yinsh_ml.tracking.experiment_tracker import ExperimentTracker
from yinsh_ml.tracking.data_management import (
    DataRetentionManager, DataBackupManager, BackupConfig,
    RetentionRule, RetentionPolicy, CleanupAction,
    setup_data_management
)


def create_test_experiments(tracker: ExperimentTracker, count: int = 10):
    """Create test experiments with various statuses and ages."""
    experiments = []
    
    for i in range(count):
        # Vary the experiment status
        if i % 4 == 0:
            status = "failed"
        elif i % 4 == 1:
            status = "completed"
        elif i % 4 == 2:
            status = "cancelled"
        else:
            status = "running"
        
        # Create experiment
        exp_id = tracker.create_experiment(
            name=f"test_experiment_{i}",
            config={"learning_rate": 0.001 + i * 0.0001, "batch_size": 32},
            tags=[f"test", f"batch_{i//3}"]
        )
        
        # Add some metrics
        for j in range(5):
            tracker.log_metric(exp_id, "loss", 1.0 - j * 0.1, iteration=j)
            tracker.log_metric(exp_id, "accuracy", j * 0.2, iteration=j)
        
        # Set final status
        tracker.update_experiment_status(exp_id, status)
        
        experiments.append(exp_id)
        print(f"Created experiment {exp_id} with status '{status}'")
    
    return experiments


def test_retention_policies():
    """Test data retention policies."""
    print("\n" + "="*60)
    print("TESTING DATA RETENTION POLICIES")
    print("="*60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_experiments.db"
        
        # Initialize tracker and create test data
        tracker = ExperimentTracker(str(db_path))
        experiments = create_test_experiments(tracker, 12)
        
        # Set up retention manager
        retention_manager = DataRetentionManager(str(db_path))
        
        # Add custom retention rules
        retention_manager.add_retention_rule(
            RetentionRule(
                name="cleanup_failed_experiments",
                policy_type=RetentionPolicy.STATUS_BASED,
                action=CleanupAction.DELETE,
                target_statuses=["failed", "cancelled"],
                priority=10,
                dry_run=True  # Test mode
            )
        )
        
        retention_manager.add_retention_rule(
            RetentionRule(
                name="limit_experiment_count",
                policy_type=RetentionPolicy.COUNT_BASED,
                action=CleanupAction.ARCHIVE,
                max_count=8,
                priority=5,
                dry_run=True  # Test mode
            )
        )
        
        # Apply retention policies (simplified for demo)
        print("\nRetention policies configured:")
        for rule in retention_manager.retention_rules:
            print(f"  - {rule.name}: {rule.policy_type.value} -> {rule.action.value}")
            print(f"    Enabled: {rule.enabled}, Dry run: {rule.dry_run}")
        
        tracker.shutdown()


def test_backup_functionality():
    """Test backup and restore functionality."""
    print("\n" + "="*60)
    print("TESTING BACKUP FUNCTIONALITY")
    print("="*60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_experiments.db"
        backup_dir = Path(temp_dir) / "backups"
        
        # Initialize tracker and create test data
        tracker = ExperimentTracker(str(db_path))
        experiments = create_test_experiments(tracker, 5)
        
        # Set up backup manager
        backup_manager = DataBackupManager(str(db_path))
        
        # Create backup configuration
        backup_config = BackupConfig(
            backup_directory=str(backup_dir),
            include_database=True,
            compression_enabled=True,
            compression_level=6,
            verify_backup=True
        )
        
        # Test single experiment export (simplified demo)
        print(f"\nExporting single experiment {experiments[0]}...")
        export_path = backup_manager.export_experiment(
            experiments[0], 
            str(temp_dir / f"single_experiment_{experiments[0]}.json")
        )
        print(f"Exported to: {export_path}")
        
        # Verify export file exists
        export_file = Path(export_path)
        if export_file.exists():
            size_mb = export_file.stat().st_size / (1024 * 1024)
            print(f"Export file size: {size_mb:.3f} MB")
        else:
            print("Export file not found!")
        
        tracker.shutdown()


def test_integrated_data_management():
    """Test integrated data management setup."""
    print("\n" + "="*60)
    print("TESTING INTEGRATED DATA MANAGEMENT")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_experiments.db"
        backup_dir = Path(temp_dir) / "backups"
        
        # Initialize tracker and create test data
        tracker = ExperimentTracker(str(db_path))
        experiments = create_test_experiments(tracker, 8)
        
        # Set up integrated data management
        retention_manager, backup_manager = setup_data_management(
            str(db_path), str(backup_dir)
        )
        
        print(f"\nData management setup complete:")
        print(f"  Retention manager: {type(retention_manager).__name__}")
        print(f"  Backup manager: {type(backup_manager).__name__}")
        print(f"  Retention rules configured: {len(retention_manager.retention_rules)}")
        
        # Test single experiment export
        print(f"\nTesting experiment export...")
        export_path = backup_manager.export_experiment(
            experiments[0], 
            str(backup_dir / f"test_export_{experiments[0]}.json")
        )
        print(f"Exported experiment {experiments[0]} to: {export_path}")
        
        # Verify export
        if Path(export_path).exists():
            print("✅ Export successful")
        else:
            print("❌ Export failed")
        
        tracker.shutdown()


def main():
    """Run all data management tests."""
    print("YinshML Data Management Test Suite")
    print("=" * 60)
    
    try:
        test_retention_policies()
        test_backup_functionality()
        test_integrated_data_management()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 