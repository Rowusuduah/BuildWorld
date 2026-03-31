"""
semantic_pass_k.store
---------------------
SQLite-backed persistence for consistency results and reports.
Zero dependencies — uses stdlib sqlite3 only.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    CriticalityLevel,
    ConsistencyResult,
    ConsistencyReport,
    ConsistencyVerdict,
)


_CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS consistency_results (
    run_id          TEXT PRIMARY KEY,
    prompt_hash     TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    agent_label     TEXT NOT NULL,
    k               INTEGER NOT NULL,
    criticality     TEXT NOT NULL,
    threshold       REAL NOT NULL,
    borderline_band REAL NOT NULL,
    consistency_score REAL NOT NULL,
    verdict         TEXT NOT NULL,
    pairwise_scores TEXT NOT NULL,
    outputs         TEXT NOT NULL,
    tested_at       TEXT NOT NULL,
    metadata        TEXT NOT NULL
)
"""

_CREATE_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS consistency_reports (
    report_id       TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    criticality     TEXT NOT NULL,
    threshold       REAL NOT NULL,
    overall_score   REAL NOT NULL,
    pass_rate       REAL NOT NULL,
    verdict         TEXT NOT NULL,
    total_results   INTEGER NOT NULL,
    passed_results  INTEGER NOT NULL,
    failed_results  INTEGER NOT NULL,
    borderline_results INTEGER NOT NULL,
    result_ids      TEXT NOT NULL,
    generated_at    TEXT NOT NULL
)
"""


class ConsistencyStore:
    """
    Persist ConsistencyResult and ConsistencyReport objects to SQLite.

    Example:
        store = ConsistencyStore("consistency.db")
        store.save_result(result)
        history = store.get_results_by_label("gpt-4o")
    """

    def __init__(self, db_path: str = "consistency_history.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_RESULTS_TABLE)
            conn.execute(_CREATE_REPORTS_TABLE)

    # ── Results ───────────────────────────────────────────────────────────────

    def save_result(self, result: ConsistencyResult) -> None:
        """Persist a ConsistencyResult."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO consistency_results
                (run_id, prompt_hash, prompt, agent_label, k, criticality,
                 threshold, borderline_band, consistency_score, verdict,
                 pairwise_scores, outputs, tested_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.run_id,
                    result.prompt_hash,
                    result.prompt,
                    result.agent_label,
                    result.k,
                    result.criticality,
                    result.threshold,
                    result.borderline_band,
                    result.consistency_score,
                    result.verdict,
                    json.dumps(result.pairwise_scores),
                    json.dumps(result.outputs),
                    result.tested_at.isoformat(),
                    json.dumps(result.metadata),
                ),
            )

    def get_result(self, run_id: str) -> Optional[ConsistencyResult]:
        """Retrieve a single ConsistencyResult by run_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM consistency_results WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_result(row) if row else None

    def get_results_by_label(self, agent_label: str) -> List[ConsistencyResult]:
        """Retrieve all results for a given agent label."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM consistency_results WHERE agent_label = ? ORDER BY tested_at DESC",
                (agent_label,),
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def get_results_by_prompt_hash(self, prompt_hash: str) -> List[ConsistencyResult]:
        """Retrieve all results for a given prompt hash."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM consistency_results WHERE prompt_hash = ? ORDER BY tested_at DESC",
                (prompt_hash,),
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def list_results(self, limit: int = 50) -> List[ConsistencyResult]:
        """Return the most recent results."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM consistency_results ORDER BY tested_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def _row_to_result(self, row: sqlite3.Row) -> ConsistencyResult:
        return ConsistencyResult(
            run_id=row["run_id"],
            prompt_hash=row["prompt_hash"],
            prompt=row["prompt"],
            agent_label=row["agent_label"],
            k=row["k"],
            criticality=row["criticality"],
            threshold=row["threshold"],
            borderline_band=row["borderline_band"],
            consistency_score=row["consistency_score"],
            verdict=row["verdict"],
            pairwise_scores=json.loads(row["pairwise_scores"]),
            outputs=json.loads(row["outputs"]),
            tested_at=datetime.fromisoformat(row["tested_at"]),
            metadata=json.loads(row["metadata"]),
        )

    # ── Reports ───────────────────────────────────────────────────────────────

    def save_report(self, report: ConsistencyReport) -> None:
        """Persist a ConsistencyReport and all its child results."""
        for result in report.results:
            self.save_result(result)

        result_ids = [r.run_id for r in report.results]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO consistency_reports
                (report_id, label, criticality, threshold, overall_score,
                 pass_rate, verdict, total_results, passed_results,
                 failed_results, borderline_results, result_ids, generated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    report.report_id,
                    report.label,
                    report.criticality,
                    report.threshold,
                    report.overall_score,
                    report.pass_rate,
                    report.verdict,
                    report.total_results,
                    report.passed_results,
                    report.failed_results,
                    report.borderline_results,
                    json.dumps(result_ids),
                    report.generated_at.isoformat(),
                ),
            )

    def list_reports(self, limit: int = 20) -> List[dict]:
        """Return summary rows for the most recent reports."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT report_id, label, criticality, overall_score, verdict,
                       pass_rate, total_results, generated_at
                FROM consistency_reports
                ORDER BY generated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
