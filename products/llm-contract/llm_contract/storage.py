"""SQLite storage for drift logging."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from llm_contract.models import ContractResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contract_evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    function_name   TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    passed          INTEGER NOT NULL,
    overall_score   REAL NOT NULL,
    rule_results    TEXT NOT NULL,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_function_ts
    ON contract_evaluations (function_name, ts);
"""


@contextmanager
def _connect(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        yield conn
    finally:
        conn.close()


def log_result(result: ContractResult, db_path: str) -> None:
    """Persist a ContractResult to the SQLite drift log.

    Args:
        result: The evaluation result to persist.
        db_path: Path to the SQLite database file.
    """
    rule_data = [
        {
            "rule_name": r.rule_name,
            "passed": r.passed,
            "confidence": r.confidence,
            "reason": r.reason,
            "weight": r.weight,
        }
        for r in result.rule_results
    ]
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO contract_evaluations
                (ts, function_name, contract_version, provider, model,
                 passed, overall_score, rule_results, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                result.function_name,
                result.contract_version,
                result.provider,
                result.model,
                1 if result.passed else 0,
                result.overall_score,
                json.dumps(rule_data),
                result.error,
            ),
        )
        conn.commit()


def get_pass_rate(
    function_name: str,
    db_path: str,
    days: Optional[int] = None,
) -> Optional[float]:
    """Return the pass rate for a contract over the last N days.

    Args:
        function_name: The decorated function name.
        db_path: Path to the SQLite database file.
        days: If provided, only look at evaluations from the last N days.
              If None, use all evaluations.

    Returns:
        Pass rate as a float in [0.0, 1.0], or None if no evaluations found.
    """
    with _connect(db_path) as conn:
        if days is not None:
            rows = conn.execute(
                """
                SELECT passed FROM contract_evaluations
                WHERE function_name = ?
                  AND ts >= datetime('now', ?)
                """,
                (function_name, f"-{days} days"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT passed FROM contract_evaluations WHERE function_name = ?",
                (function_name,),
            ).fetchall()

    if not rows:
        return None
    return sum(r["passed"] for r in rows) / len(rows)


def get_drift_report(function_name: str, db_path: str, days: int = 30) -> dict:
    """Compare pass rate now vs N days ago for drift detection.

    Returns:
        Dict with keys: function_name, current_pass_rate, prior_pass_rate,
        drift_pp (percentage points), has_drift (bool), evaluation_count.
    """
    with _connect(db_path) as conn:
        recent = conn.execute(
            """
            SELECT passed FROM contract_evaluations
            WHERE function_name = ?
              AND ts >= datetime('now', ?)
            """,
            (function_name, f"-{days // 2} days"),
        ).fetchall()

        prior = conn.execute(
            """
            SELECT passed FROM contract_evaluations
            WHERE function_name = ?
              AND ts >= datetime('now', ?)
              AND ts < datetime('now', ?)
            """,
            (function_name, f"-{days} days", f"-{days // 2} days"),
        ).fetchall()

    current_rate = sum(r["passed"] for r in recent) / len(recent) if recent else None
    prior_rate = sum(r["passed"] for r in prior) / len(prior) if prior else None

    drift_pp = None
    has_drift = False
    if current_rate is not None and prior_rate is not None:
        drift_pp = (current_rate - prior_rate) * 100
        has_drift = abs(drift_pp) >= 5.0  # 5pp threshold

    return {
        "function_name": function_name,
        "current_pass_rate": current_rate,
        "prior_pass_rate": prior_rate,
        "drift_pp": drift_pp,
        "has_drift": has_drift,
        "evaluation_count": len(recent) + len(prior),
    }


def list_contracts(db_path: str) -> list[dict]:
    """Return all unique contract function names in the log."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT function_name, contract_version, provider, model,
                       COUNT(*) as total,
                       SUM(passed) as passed_count,
                       MAX(ts) as last_seen
                FROM contract_evaluations
                GROUP BY function_name, contract_version, provider, model
                ORDER BY last_seen DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
