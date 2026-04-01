"""
cot_fidelity.store
------------------
SQLite-backed persistence for FidelityResult history and drift tracking.
Zero external dependencies (sqlite3 is stdlib).
"""
from __future__ import annotations

import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import DriftPoint, DriftReport, FidelityResult, FidelityVerdict


_SCHEMA = """
CREATE TABLE IF NOT EXISTS fidelity_results (
    id              TEXT PRIMARY KEY,
    prompt_hash     TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    cot_chain       TEXT NOT NULL,
    full_output     TEXT NOT NULL,
    suppressed_output TEXT NOT NULL,
    similarity      REAL NOT NULL,
    faithfulness_score REAL NOT NULL,
    verdict         TEXT NOT NULL,
    faithful_threshold REAL NOT NULL,
    unfaithful_threshold REAL NOT NULL,
    runs            INTEGER NOT NULL DEFAULT 1,
    model_version   TEXT NOT NULL DEFAULT '',
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_hash ON fidelity_results(prompt_hash);
CREATE INDEX IF NOT EXISTS idx_verdict ON fidelity_results(verdict);
CREATE INDEX IF NOT EXISTS idx_recorded_at ON fidelity_results(recorded_at);
"""


class FidelityStore:
    """
    Persistent SQLite store for FidelityResult records.
    Enables longitudinal drift monitoring.

    Usage:
        store = FidelityStore()  # ~/.cot_fidelity/history.db
        store.save(result, model_version="claude-3-7-sonnet")
        drift = store.detect_drift(window=50)
        print(drift.trend)  # STABLE / DEGRADING / IMPROVING
    """

    DEFAULT_DB = Path.home() / ".cot_fidelity" / "history.db"

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, result: FidelityResult, model_version: str = "") -> str:
        """Persist a FidelityResult. Returns the generated row id."""
        row_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fidelity_results
                  (id, prompt_hash, prompt, cot_chain, full_output,
                   suppressed_output, similarity, faithfulness_score,
                   verdict, faithful_threshold, unfaithful_threshold,
                   runs, model_version, recorded_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row_id,
                    result.prompt_hash,
                    result.prompt,
                    result.cot_chain,
                    result.full_output,
                    result.suppressed_output,
                    result.similarity,
                    result.faithfulness_score,
                    result.verdict,
                    result.faithful_threshold,
                    result.unfaithful_threshold,
                    result.runs,
                    model_version,
                    result.tested_at.isoformat(),
                ),
            )
        return row_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM fidelity_results").fetchone()
            return row[0]

    def recent(self, n: int = 50) -> List[FidelityResult]:
        """Return the n most recent results, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fidelity_results ORDER BY recorded_at DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def by_prompt_hash(self, prompt_hash: str, limit: int = 100) -> List[FidelityResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fidelity_results WHERE prompt_hash=? ORDER BY recorded_at DESC LIMIT ?",
                (prompt_hash, limit),
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def by_verdict(self, verdict: FidelityVerdict, limit: int = 100) -> List[FidelityResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fidelity_results WHERE verdict=? ORDER BY recorded_at DESC LIMIT ?",
                (verdict, limit),
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    # ── Drift Detection ───────────────────────────────────────────────────────

    def detect_drift(self, window: int = 50, baseline_window: int = 200) -> DriftReport:
        """
        Compare recent faithfulness scores to a longer-term baseline.
        Drift is detected when the recent mean drops more than 1 std below baseline.
        """
        with self._connect() as conn:
            recent_rows = conn.execute(
                "SELECT faithfulness_score, verdict, prompt_hash, model_version, recorded_at, id "
                "FROM fidelity_results ORDER BY recorded_at DESC LIMIT ?",
                (window,),
            ).fetchall()
            baseline_rows = conn.execute(
                "SELECT faithfulness_score FROM fidelity_results ORDER BY recorded_at DESC LIMIT ?",
                (baseline_window,),
            ).fetchall()

        if len(recent_rows) < 3:
            return DriftReport(
                points=[],
                window=window,
                mean_score=0.0,
                std_score=0.0,
                drift_detected=False,
                trend="INSUFFICIENT_DATA",
            )

        recent_scores = [r["faithfulness_score"] for r in recent_rows]
        baseline_scores = [r["faithfulness_score"] for r in baseline_rows]

        recent_mean = sum(recent_scores) / len(recent_scores)
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        baseline_var = sum((s - baseline_mean) ** 2 for s in baseline_scores) / len(baseline_scores)
        baseline_std = math.sqrt(baseline_var)

        drift_detected = recent_mean < (baseline_mean - baseline_std) and baseline_std > 1e-6

        # Trend: compare first half vs second half of recent window
        mid = len(recent_scores) // 2
        first_half_mean = sum(recent_scores[mid:]) / len(recent_scores[mid:])  # older
        second_half_mean = sum(recent_scores[:mid]) / len(recent_scores[:mid])  # newer
        delta = second_half_mean - first_half_mean

        if len(recent_scores) < 10:
            trend = "INSUFFICIENT_DATA"
        elif abs(delta) < 0.02:
            trend = "STABLE"
        elif delta < 0:
            trend = "DEGRADING"
        else:
            trend = "IMPROVING"

        recent_var = sum((s - recent_mean) ** 2 for s in recent_scores) / len(recent_scores)
        recent_std = math.sqrt(recent_var)

        points = [
            DriftPoint(
                run_id=r["id"],
                prompt_hash=r["prompt_hash"],
                faithfulness_score=r["faithfulness_score"],
                verdict=r["verdict"],
                model_version=r["model_version"],
                recorded_at=datetime.fromisoformat(r["recorded_at"]),
            )
            for r in recent_rows
        ]

        return DriftReport(
            points=points,
            window=window,
            mean_score=recent_mean,
            std_score=recent_std,
            drift_detected=drift_detected,
            trend=trend,
        )

    def clear(self) -> int:
        """Delete all rows. Returns count deleted."""
        with self._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM fidelity_results").fetchone()[0]
            conn.execute("DELETE FROM fidelity_results")
        return n

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> FidelityResult:
        return FidelityResult(
            prompt=row["prompt"],
            full_output=row["full_output"],
            suppressed_output=row["suppressed_output"],
            cot_chain=row["cot_chain"],
            similarity=row["similarity"],
            faithfulness_score=row["faithfulness_score"],
            verdict=row["verdict"],
            faithful_threshold=row["faithful_threshold"],
            unfaithful_threshold=row["unfaithful_threshold"],
            runs=row["runs"],
            prompt_hash=row["prompt_hash"],
            tested_at=datetime.fromisoformat(row["recorded_at"]),
        )
