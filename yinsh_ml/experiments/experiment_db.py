"""
SQLite-backed experiment database for tracking and querying experiments.

Provides persistent storage for:
- Experiment metadata and configuration
- Per-iteration metrics
- Final outcomes and status
"""

import sqlite3
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
import uuid

logger = logging.getLogger(__name__)


@dataclass
class IterationMetrics:
    """Metrics captured at the end of each training iteration."""
    iteration: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Self-play metrics
    selfplay_time: float = 0.0
    selfplay_games: int = 0
    avg_game_length: float = 0.0
    avg_ring_mobility: float = 0.0

    # Training metrics
    training_time: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    value_accuracy: float = 0.0
    value_variance: float = 0.0
    buffer_size: int = 0

    # Tournament metrics
    tournament_elo: float = 1500.0
    tournament_win_rate: float = 0.0
    wilson_lower_bound: float = 0.0

    # Model selection
    promoted: bool = False
    reverted: bool = False
    active_iteration: int = 0

    # Memory metrics
    memory_mb: float = 0.0
    tensor_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IterationMetrics':
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class ExperimentRecord:
    """Complete record of an experiment."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "unnamed"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Git info for reproducibility
    git_commit: str = ""
    git_branch: str = ""

    # Configuration (JSON serialized)
    config_json: str = "{}"

    # Status
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    current_iteration: int = 0
    total_iterations: int = 10

    # Final outcomes
    final_elo: float = 1500.0
    best_elo: float = 1500.0
    final_policy_loss: float = 0.0
    final_value_loss: float = 0.0
    promoted_count: int = 0
    rejected_count: int = 0
    total_runtime_seconds: float = 0.0

    # Iterations stored separately (not in this dataclass)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRecord':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class ExperimentDB:
    """SQLite database for experiment tracking."""

    def __init__(self, db_path: str = "experiments/experiments.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    config_json TEXT,
                    status TEXT DEFAULT 'pending',
                    current_iteration INTEGER DEFAULT 0,
                    total_iterations INTEGER DEFAULT 10,
                    final_elo REAL DEFAULT 1500.0,
                    best_elo REAL DEFAULT 1500.0,
                    final_policy_loss REAL DEFAULT 0.0,
                    final_value_loss REAL DEFAULT 0.0,
                    promoted_count INTEGER DEFAULT 0,
                    rejected_count INTEGER DEFAULT 0,
                    total_runtime_seconds REAL DEFAULT 0.0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS iteration_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                    UNIQUE(experiment_id, iteration)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_iterations_experiment
                ON iteration_metrics(experiment_id)
            """)

            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def create_experiment(self, record: ExperimentRecord) -> str:
        """Create a new experiment record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (
                    experiment_id, name, description, created_at, updated_at,
                    git_commit, git_branch, config_json, status,
                    current_iteration, total_iterations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.experiment_id, record.name, record.description,
                record.created_at, record.updated_at,
                record.git_commit, record.git_branch, record.config_json,
                record.status, record.current_iteration, record.total_iterations
            ))
            conn.commit()
        logger.info(f"Created experiment: {record.experiment_id} ({record.name})")
        return record.experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()
            if row:
                return ExperimentRecord.from_dict(dict(row))
        return None

    def get_experiment_by_name(self, name: str) -> Optional[ExperimentRecord]:
        """Get experiment by name (returns most recent if multiple)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                return ExperimentRecord.from_dict(dict(row))
        return None

    def update_experiment(self, experiment_id: str, **updates):
        """Update experiment fields."""
        updates['updated_at'] = datetime.now().isoformat()

        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [experiment_id]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE experiments SET {set_clause} WHERE experiment_id = ?",
                values
            )
            conn.commit()

    def log_iteration(self, experiment_id: str, metrics: IterationMetrics):
        """Log metrics for an iteration."""
        metrics_json = json.dumps(metrics.to_dict())

        with sqlite3.connect(self.db_path) as conn:
            # Upsert iteration metrics
            conn.execute("""
                INSERT OR REPLACE INTO iteration_metrics
                (experiment_id, iteration, timestamp, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, metrics.iteration, metrics.timestamp, metrics_json))

            # Update experiment progress
            conn.execute("""
                UPDATE experiments
                SET current_iteration = ?,
                    updated_at = ?,
                    final_elo = ?,
                    best_elo = MAX(best_elo, ?),
                    final_policy_loss = ?,
                    final_value_loss = ?,
                    promoted_count = promoted_count + ?,
                    rejected_count = rejected_count + ?
                WHERE experiment_id = ?
            """, (
                metrics.iteration + 1,
                datetime.now().isoformat(),
                metrics.tournament_elo,
                metrics.tournament_elo,
                metrics.policy_loss,
                metrics.value_loss,
                1 if metrics.promoted else 0,
                0 if metrics.promoted else 1,
                experiment_id
            ))
            conn.commit()

    def get_iterations(self, experiment_id: str) -> List[IterationMetrics]:
        """Get all iteration metrics for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT metrics_json FROM iteration_metrics
                WHERE experiment_id = ?
                ORDER BY iteration
            """, (experiment_id,))

            results = []
            for row in cursor:
                data = json.loads(row['metrics_json'])
                results.append(IterationMetrics.from_dict(data))
            return results

    def list_experiments(
        self,
        status: Optional[str] = None,
        order_by: str = 'created_at',
        descending: bool = True,
        limit: int = 100
    ) -> List[ExperimentRecord]:
        """List experiments with optional filtering."""
        query = "SELECT * FROM experiments"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        order = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_by} {order} LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [ExperimentRecord.from_dict(dict(row)) for row in cursor]

    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        experiments = []
        all_iterations = {}

        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
                all_iterations[exp_id] = self.get_iterations(exp_id)

        if not experiments:
            return {}

        # Build comparison structure
        comparison = {
            'experiments': [e.to_dict() for e in experiments],
            'iterations': {
                exp_id: [m.to_dict() for m in metrics]
                for exp_id, metrics in all_iterations.items()
            },
            'summary': {
                'elo_progression': {},
                'loss_progression': {},
                'final_comparison': []
            }
        }

        # Build ELO and loss progression
        for exp_id, metrics in all_iterations.items():
            comparison['summary']['elo_progression'][exp_id] = [
                m.tournament_elo for m in metrics
            ]
            comparison['summary']['loss_progression'][exp_id] = [
                {'policy': m.policy_loss, 'value': m.value_loss}
                for m in metrics
            ]

        # Final comparison table
        for exp in experiments:
            comparison['summary']['final_comparison'].append({
                'id': exp.experiment_id,
                'name': exp.name,
                'final_elo': exp.final_elo,
                'best_elo': exp.best_elo,
                'final_policy_loss': exp.final_policy_loss,
                'final_value_loss': exp.final_value_loss,
                'promoted': exp.promoted_count,
                'rejected': exp.rejected_count,
                'runtime_hours': exp.total_runtime_seconds / 3600
            })

        return comparison

    def get_parameter_impact(self, parameter_path: str) -> List[Dict[str, Any]]:
        """
        Analyze impact of a specific parameter across experiments.

        Returns list of (parameter_value, final_elo, experiment_id) tuples.
        """
        results = []
        experiments = self.list_experiments(status='completed')

        for exp in experiments:
            try:
                config = json.loads(exp.config_json)
                # Navigate to parameter using dot notation
                parts = parameter_path.split('.')
                value = config
                for part in parts:
                    value = value.get(part)
                    if value is None:
                        break

                if value is not None:
                    results.append({
                        'experiment_id': exp.experiment_id,
                        'name': exp.name,
                        'parameter_value': value,
                        'final_elo': exp.final_elo,
                        'best_elo': exp.best_elo
                    })
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return sorted(results, key=lambda x: x['parameter_value'])

    def export_to_csv(self, output_path: str):
        """Export all experiments to CSV."""
        import csv

        experiments = self.list_experiments(limit=10000)

        with open(output_path, 'w', newline='') as f:
            if not experiments:
                return

            writer = csv.DictWriter(f, fieldnames=experiments[0].to_dict().keys())
            writer.writeheader()
            for exp in experiments:
                writer.writerow(exp.to_dict())

        logger.info(f"Exported {len(experiments)} experiments to {output_path}")

    def delete_experiment(self, experiment_id: str):
        """Delete an experiment and its iterations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM iteration_metrics WHERE experiment_id = ?",
                (experiment_id,)
            )
            conn.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            conn.commit()
        logger.info(f"Deleted experiment: {experiment_id}")
