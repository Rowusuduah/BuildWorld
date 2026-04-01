"""context_trace.store
~~~~~~~~~~~~~~~~~~~
Optional SQLite persistence for AttributionReports.

Usage::

    store = AttributionStore("ctrace.db")
    run_id = store.save(report, label="rag_pipeline_v2")
    history = store.list_runs()
    data = store.get(run_id)
    store.close()

    # Context manager
    with AttributionStore("ctrace.db") as store:
        store.save(report, label="test_run")
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from context_trace.tracer import AttributionReport


class AttributionStore:
    """Persists AttributionReports to SQLite for trend analysis and comparisons."""

    def __init__(self, db_path: str = "ctrace.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                label        TEXT    DEFAULT '',
                created_at   TEXT    NOT NULL,
                total_api_calls INTEGER,
                elapsed_seconds REAL,
                estimated_cost_usd REAL,
                top_score    REAL,
                report_json  TEXT    NOT NULL
            )
        """)
        conn.commit()

    def save(self, report: AttributionReport, label: str = "") -> int:
        """Persist a report and return its row ID."""
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO runs
                (label, created_at, total_api_calls, elapsed_seconds,
                 estimated_cost_usd, top_score, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                label,
                datetime.now(timezone.utc).isoformat(),
                report.total_api_calls,
                report.elapsed_seconds,
                report.estimated_cost_usd,
                report.top_score,
                json.dumps(report.to_dict()),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def list_runs(self, limit: int = 50) -> List[Dict]:
        """Return run metadata (without full JSON) for the most recent runs."""
        conn = self._connect()
        cursor = conn.execute(
            """
            SELECT id, label, created_at, total_api_calls,
                   elapsed_seconds, estimated_cost_usd, top_score
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get(self, run_id: int) -> Optional[Dict]:
        """Retrieve full report dict by run ID. Returns None if not found."""
        conn = self._connect()
        cursor = conn.execute(
            "SELECT report_json FROM runs WHERE id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, run_id: int) -> bool:
        """Delete a run by ID. Returns True if a row was deleted."""
        conn = self._connect()
        cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return total number of stored runs."""
        conn = self._connect()
        cursor = conn.execute("SELECT COUNT(*) FROM runs")
        return cursor.fetchone()[0]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AttributionStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
