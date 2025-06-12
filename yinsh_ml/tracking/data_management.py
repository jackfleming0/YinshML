"""
Data management policies and utilities for YinshML experiment tracking.
"""

import json
import gzip
import shutil
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from .utils import get_connection, transaction_context, DatabaseError
except ImportError:
    from utils import get_connection, transaction_context, DatabaseError

logger = logging.getLogger(__name__)

class RetentionPolicy(Enum):
    AGE_BASED = "age_based"
    COUNT_BASED = "count_based"
    STATUS_BASED = "status_based"

class CleanupAction(Enum):
    DELETE = "delete"
    ARCHIVE = "archive"

@dataclass
class RetentionRule:
    name: str
    policy_type: RetentionPolicy
    action: CleanupAction
    enabled: bool = True
    max_age_days: Optional[int] = None
    max_count: Optional[int] = None
    target_statuses: Optional[List[str]] = None
    priority: int = 0
    dry_run: bool = False

@dataclass
class BackupConfig:
    backup_directory: str
    include_database: bool = True
    compression_enabled: bool = True
    verify_backup: bool = True

class DataRetentionManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.retention_rules: List[RetentionRule] = []
    
    def add_retention_rule(self, rule: RetentionRule) -> None:
        self.retention_rules.append(rule)
        logger.info(f"Added retention rule: {rule.name}")

class DataBackupManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path
    
    def export_experiment(self, experiment_id: int, output_path: str = None) -> str:
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
                exp_row = cursor.fetchone()
                
                if not exp_row:
                    raise ValueError(f"Experiment {experiment_id} not found")
                
                experiment = dict(exp_row)
                experiment["config"] = json.loads(experiment["config_json"])
                experiment["environment"] = json.loads(experiment["environment_json"])
                del experiment["config_json"]
                del experiment["environment_json"]
                
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"experiment_{experiment_id}_{timestamp}.json"
                
                with open(output_path, "w") as f:
                    json.dump(experiment, f, indent=2, default=str)
                
                logger.info(f"Exported experiment {experiment_id} to {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Failed to export experiment {experiment_id}: {e}")
            raise

def setup_data_management(db_path: str, backup_dir: str = "./backups") -> Tuple[DataRetentionManager, DataBackupManager]:
    retention_manager = DataRetentionManager(db_path)
    backup_manager = DataBackupManager(db_path)
    logger.info("Data management setup complete")
    return retention_manager, backup_manager
