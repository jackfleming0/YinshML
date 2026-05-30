"""SQLite-backed registry for the orchestration layer.

This is the *tabular truth* half of the storage split (the Markdown narrative is
``journal``). It extends the existing ``experiments.db`` with two tables the base
``ExperimentDB`` doesn't have — ``eval_results`` (one row per Tier-0 evaluation)
and ``gate_queue`` (the blocking ratification queue) — and adds the orchestration
status lifecycle on top of the base ``experiments`` table.

It reuses the *same* database file as ``ExperimentDB`` on purpose: querying "every
candidate that beat its anchor" should be one join, not a cross-store merge.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Orchestration status lifecycle, layered over ExperimentRecord's base statuses
# (pending/running/completed/failed/cancelled). A candidate flows:
#   queued -> running -> evaluated -> (awaiting_ratification) -> promoted | rejected
Status = Literal[
    "queued",
    "running",
    "evaluated",
    "awaiting_ratification",
    "promoted",
    "rejected",
    "failed",
]

GateKind = Literal["promotion", "review"]
GateState = Literal["open", "ratified", "rejected"]


@dataclass
class EvalResult:
    """One Tier-0 evaluation of a candidate against its baseline."""

    experiment_id: str
    baseline_id: Optional[str]
    tier: int
    wins: int
    losses: int
    draws: int
    sprt_verdict: str
    sprt_llr: float
    wilson_lower: float
    wilson_upper: float
    panel_green: bool
    panel_json: str = "{}"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[int] = None

    def panel(self) -> Dict[str, Any]:
        return json.loads(self.panel_json)


@dataclass
class GateItem:
    """A decision waiting on the human: a promotion or an ambiguous review."""

    experiment_id: str
    kind: GateKind
    summary: str
    report_path: str = ""
    state: GateState = "open"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    id: Optional[int] = None


class OrchestrationStore:
    """Thin data-access layer over the shared experiments SQLite file."""

    def __init__(self, db_path: str = "experiments/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    baseline_id TEXT,
                    tier INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    losses INTEGER NOT NULL,
                    draws INTEGER NOT NULL,
                    sprt_verdict TEXT NOT NULL,
                    sprt_llr REAL NOT NULL,
                    wilson_lower REAL NOT NULL,
                    wilson_upper REAL NOT NULL,
                    panel_green INTEGER NOT NULL,
                    panel_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gate_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    report_path TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_experiment ON eval_results(experiment_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gate_state ON gate_queue(state)"
            )
            conn.commit()

    # --- status lifecycle -------------------------------------------------

    def set_status(self, experiment_id: str, status: Status) -> None:
        """Update the orchestration status on the shared experiments row.

        Uses the base ``experiments`` table written by ``ExperimentDB`` so a
        single query sees both the run's training outcome and its funnel state.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE experiment_id = ?",
                (status, datetime.now().isoformat(), experiment_id),
            )
            conn.commit()

    def get_status(self, experiment_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            return row["status"] if row else None

    # --- eval results -----------------------------------------------------

    def record_eval(self, result: EvalResult) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO eval_results (
                    experiment_id, baseline_id, tier, wins, losses, draws,
                    sprt_verdict, sprt_llr, wilson_lower, wilson_upper,
                    panel_green, panel_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id, result.baseline_id, result.tier,
                    result.wins, result.losses, result.draws,
                    result.sprt_verdict, result.sprt_llr,
                    result.wilson_lower, result.wilson_upper,
                    1 if result.panel_green else 0, result.panel_json,
                    result.created_at,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_evals(self, experiment_id: str) -> List[EvalResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE experiment_id = ? ORDER BY id",
                (experiment_id,),
            ).fetchall()
            return [self._row_to_eval(r) for r in rows]

    # --- gate queue -------------------------------------------------------

    def enqueue_gate(self, item: GateItem) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO gate_queue (
                    experiment_id, kind, summary, report_path, state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.experiment_id, item.kind, item.summary,
                    item.report_path, item.state, item.created_at,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def open_gate_items(self) -> List[GateItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM gate_queue WHERE state = 'open' ORDER BY created_at",
            ).fetchall()
            return [self._row_to_gate(r) for r in rows]

    def get_gate_item(self, gate_id: int) -> Optional[GateItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gate_queue WHERE id = ?", (gate_id,)
            ).fetchone()
            return self._row_to_gate(row) if row else None

    def open_gate_for_experiment(self, experiment_id: str) -> Optional[GateItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM gate_queue WHERE experiment_id = ? AND state = 'open' "
                "ORDER BY created_at DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            return self._row_to_gate(row) if row else None

    def resolve_gate(self, gate_id: int, state: GateState) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE gate_queue SET state = ?, resolved_at = ? WHERE id = ?",
                (state, datetime.now().isoformat(), gate_id),
            )
            conn.commit()

    # --- row mappers ------------------------------------------------------

    @staticmethod
    def _row_to_eval(row: sqlite3.Row) -> EvalResult:
        return EvalResult(
            id=row["id"],
            experiment_id=row["experiment_id"],
            baseline_id=row["baseline_id"],
            tier=row["tier"],
            wins=row["wins"],
            losses=row["losses"],
            draws=row["draws"],
            sprt_verdict=row["sprt_verdict"],
            sprt_llr=row["sprt_llr"],
            wilson_lower=row["wilson_lower"],
            wilson_upper=row["wilson_upper"],
            panel_green=bool(row["panel_green"]),
            panel_json=row["panel_json"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_gate(row: sqlite3.Row) -> GateItem:
        return GateItem(
            id=row["id"],
            experiment_id=row["experiment_id"],
            kind=row["kind"],
            summary=row["summary"],
            report_path=row["report_path"],
            state=row["state"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )
