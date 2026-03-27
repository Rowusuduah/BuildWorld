"""SQLite-backed trace ledger: every eval run is recorded with commit SHA."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _get_commit_sha() -> Optional[str]:
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha[:8]
    except Exception:
        return None


class TraceLedger:
    """Append-only SQLite ledger for eval and calibration runs.

    Every run is linked to the current git commit SHA so diffs can be computed
    between commits to detect regressions over time.
    """

    def __init__(self, db_path: str | Path = ".prompt-lock/traces.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    commit_sha      TEXT,
                    prompt_path     TEXT    NOT NULL,
                    prompt_hash     TEXT    NOT NULL,
                    eval_type       TEXT    NOT NULL,
                    model           TEXT,
                    score           REAL    NOT NULL,
                    passed          INTEGER NOT NULL,
                    threshold       REAL    NOT NULL,
                    details         TEXT,
                    input_text      TEXT,
                    output_text     TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_runs (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp            TEXT    NOT NULL,
                    commit_sha           TEXT,
                    model                TEXT    NOT NULL,
                    criteria             TEXT    NOT NULL,
                    agreement_rate       REAL    NOT NULL,
                    spearman_correlation REAL    NOT NULL,
                    bias                 REAL    NOT NULL,
                    n_examples           INTEGER NOT NULL,
                    passed               INTEGER NOT NULL,
                    details              TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_prompt "
                "ON eval_runs(prompt_path, eval_type, timestamp)"
            )

    def log_eval(
        self,
        prompt_path: str,
        prompt_content: str,
        eval_type: str,
        score: float,
        passed: bool,
        threshold: float,
        model: Optional[str] = None,
        details: str = "",
        input_text: str = "",
        output_text: str = "",
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO eval_runs
                   (timestamp, commit_sha, prompt_path, prompt_hash, eval_type, model,
                    score, passed, threshold, details, input_text, output_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _now_iso(),
                    _get_commit_sha(),
                    prompt_path,
                    _sha(prompt_content),
                    eval_type,
                    model,
                    score,
                    int(passed),
                    threshold,
                    details,
                    input_text[:500],
                    output_text[:500],
                ),
            )
            return cursor.lastrowid

    def log_calibration(
        self,
        model: str,
        criteria: str,
        agreement_rate: float,
        spearman_correlation: float,
        bias: float,
        n_examples: int,
        passed: bool,
        details: Optional[list[dict]] = None,
    ) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO calibration_runs
                   (timestamp, commit_sha, model, criteria, agreement_rate,
                    spearman_correlation, bias, n_examples, passed, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _now_iso(),
                    _get_commit_sha(),
                    model,
                    criteria,
                    agreement_rate,
                    spearman_correlation,
                    bias,
                    n_examples,
                    int(passed),
                    json.dumps(details) if details else None,
                ),
            )
            return cursor.lastrowid

    def get_baseline_score(
        self, prompt_path: str, eval_type: str, n_recent: int = 5
    ) -> Optional[float]:
        """Return average score of the N most recent *passing* runs for this prompt+eval_type."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT score FROM eval_runs
                   WHERE prompt_path = ? AND eval_type = ? AND passed = 1
                   ORDER BY timestamp DESC LIMIT ?""",
                (prompt_path, eval_type, n_recent),
            ).fetchall()
        if not rows:
            return None
        return sum(r["score"] for r in rows) / len(rows)

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT timestamp, commit_sha, prompt_path, eval_type,
                          score, passed, threshold, details
                   FROM eval_runs ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "timestamp": r["timestamp"],
                "commit": r["commit_sha"],
                "prompt": r["prompt_path"],
                "eval_type": r["eval_type"],
                "score": r["score"],
                "passed": bool(r["passed"]),
                "threshold": r["threshold"],
                "details": r["details"],
            }
            for r in rows
        ]

    def diff_commits(self, commit_a: str, commit_b: str) -> list[dict]:
        """Compare average eval scores between two commits."""
        with self._conn() as conn:
            rows_a = conn.execute(
                """SELECT prompt_path, eval_type, AVG(score) as avg
                   FROM eval_runs WHERE commit_sha = ? GROUP BY prompt_path, eval_type""",
                (commit_a,),
            ).fetchall()
            rows_b = conn.execute(
                """SELECT prompt_path, eval_type, AVG(score) as avg
                   FROM eval_runs WHERE commit_sha = ? GROUP BY prompt_path, eval_type""",
                (commit_b,),
            ).fetchall()

        map_a = {(r["prompt_path"], r["eval_type"]): r["avg"] for r in rows_a}
        map_b = {(r["prompt_path"], r["eval_type"]): r["avg"] for r in rows_b}
        all_keys = sorted(set(map_a.keys()) | set(map_b.keys()))

        results = []
        for key in all_keys:
            sa = map_a.get(key)
            sb = map_b.get(key)
            delta = (sb - sa) if (sa is not None and sb is not None) else None
            results.append(
                {
                    "prompt": key[0],
                    "eval_type": key[1],
                    f"score_{commit_a}": sa,
                    f"score_{commit_b}": sb,
                    "delta": delta,
                }
            )
        return results
