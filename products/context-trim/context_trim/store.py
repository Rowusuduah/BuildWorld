"""
SQLite-backed trim history store. Zero dependencies — uses stdlib sqlite3.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import TrimResult


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trim_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    pipeline_id TEXT    NOT NULL DEFAULT 'default',
    strategy    TEXT    NOT NULL,
    original_count  INTEGER NOT NULL,
    final_count     INTEGER NOT NULL,
    original_tokens INTEGER NOT NULL,
    final_tokens    INTEGER NOT NULL,
    dropped_count   INTEGER NOT NULL,
    trim_ratio  REAL    NOT NULL,
    within_budget INTEGER NOT NULL,
    max_tokens  INTEGER NOT NULL,
    reserved    INTEGER NOT NULL
);
"""

_INSERT = """
INSERT INTO trim_history
    (ts, pipeline_id, strategy, original_count, final_count,
     original_tokens, final_tokens, dropped_count, trim_ratio,
     within_budget, max_tokens, reserved)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_HISTORY = """
SELECT id, ts, pipeline_id, strategy, original_count, final_count,
       original_tokens, final_tokens, dropped_count, trim_ratio,
       within_budget, max_tokens, reserved
FROM trim_history
WHERE pipeline_id = ?
ORDER BY ts DESC
LIMIT ?;
"""

_SELECT_ALL = """
SELECT id, ts, pipeline_id, strategy, original_count, final_count,
       original_tokens, final_tokens, dropped_count, trim_ratio,
       within_budget, max_tokens, reserved
FROM trim_history
ORDER BY ts DESC
LIMIT ?;
"""


class TrimStore:
    """Stores and retrieves trim operation history in a local SQLite database.

    Thread-safe: each call opens/closes a connection (suitable for scripts).
    For high-throughput use, pass a shared connection via *conn*.
    """

    def __init__(self, db_path: str = "context_trim_history.db") -> None:
        self._db_path = str(Path(db_path).expanduser().resolve())
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            conn.commit()

    def record(self, result: "TrimResult", pipeline_id: str = "default") -> int:
        """Persist a TrimResult. Returns the new row ID."""
        with self._connect() as conn:
            cur = conn.execute(
                _INSERT,
                (
                    time.time(),
                    pipeline_id,
                    result.strategy.value,
                    result.original_count,
                    result.final_count,
                    result.original_tokens,
                    result.final_tokens,
                    result.dropped_count,
                    result.trim_ratio,
                    int(result.within_budget),
                    result.budget.max_tokens,
                    result.budget.reserved_tokens,
                ),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def history(
        self, pipeline_id: str = "default", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return the most recent *limit* records for *pipeline_id*."""
        with self._connect() as conn:
            rows = conn.execute(_SELECT_HISTORY, (pipeline_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def all_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent *limit* records across all pipelines."""
        with self._connect() as conn:
            rows = conn.execute(_SELECT_ALL, (limit,)).fetchall()
        return [dict(row) for row in rows]

    def stats(self, pipeline_id: str = "default") -> dict[str, Any]:
        """Return aggregate stats for a pipeline."""
        records = self.history(pipeline_id, limit=1000)
        if not records:
            return {"pipeline_id": pipeline_id, "total_runs": 0}
        total = len(records)
        over_budget = sum(1 for r in records if not r["within_budget"])
        avg_ratio = sum(r["trim_ratio"] for r in records) / total
        avg_dropped = sum(r["dropped_count"] for r in records) / total
        return {
            "pipeline_id": pipeline_id,
            "total_runs": total,
            "over_budget_runs": over_budget,
            "avg_trim_ratio": round(avg_ratio, 4),
            "avg_dropped_count": round(avg_dropped, 2),
        }
