"""spec-drift: Semantic specification drift detector for LLM outputs.

"Among those who approach me I will be proved holy" — Leviticus 10:3

Structural validation is not enough. Semantic compliance must be declared
and monitored. spec-drift catches semantic drift that Pydantic cannot see.

pip install spec-drift
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar
from functools import wraps

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    raise ImportError("spec-drift requires pydantic v2: pip install 'pydantic>=2.0'")

__version__ = "0.1.0"
__all__ = [
    "spec",
    "SemanticConstraint",
    "ConstraintType",
    "DriftSeverity",
    "DriftMonitor",
    "Observation",
    "ObservationStore",
    "run_ci_gate",
]

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConstraintType(str, Enum):
    AUTHORIZED_VALUES = "authorized_values"
    LENGTH_BOUNDS = "length_bounds"
    DISTRIBUTION = "distribution"
    PATTERN_MATCH = "pattern_match"
    CORRELATION = "correlation"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# SemanticConstraint — declares a semantic rule on a single field
# ---------------------------------------------------------------------------

@dataclass
class SemanticConstraint:
    """Declares a semantic constraint on a single LLM output field.

    Use the class-method factories to build constraints:
        SemanticConstraint.from_authorized_values(["pos", "neg", "neu"])
        SemanticConstraint.from_length_bounds(min_words=30, max_words=300)
        SemanticConstraint.from_distribution(mean=6.5, std=2.0)
        SemanticConstraint.from_pattern(regex=r"^[A-Z]")
    """

    constraint_type: ConstraintType
    params: Dict[str, Any]
    alert_threshold: float = 0.15
    field_name: Optional[str] = None  # populated by @spec decorator

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_authorized_values(
        cls,
        authorized: List[Any],
        tolerance: float = 0.02,
        alert_threshold: float = 0.10,
    ) -> "SemanticConstraint":
        """Field values must be drawn from the authorized set.

        Args:
            authorized: List of permitted values.
            tolerance: Fraction of observations allowed outside the set
                before the constraint flags a violation.
                (Currently used for documentation; check() is strict.)
            alert_threshold: Rolling violation rate that triggers an alert.
        """
        return cls(
            constraint_type=ConstraintType.AUTHORIZED_VALUES,
            params={"authorized": list(authorized), "tolerance": tolerance},
            alert_threshold=alert_threshold,
        )

    @classmethod
    def from_length_bounds(
        cls,
        min_words: int = 0,
        max_words: int = 10_000,
        alert_threshold: float = 0.15,
    ) -> "SemanticConstraint":
        """String field word count must be within [min_words, max_words]."""
        return cls(
            constraint_type=ConstraintType.LENGTH_BOUNDS,
            params={"min_words": min_words, "max_words": max_words},
            alert_threshold=alert_threshold,
        )

    @classmethod
    def from_distribution(
        cls,
        mean: float,
        std: float,
        drift_threshold: float = 1.0,
        alert_threshold: float = 0.20,
    ) -> "SemanticConstraint":
        """Numeric field should be within 3σ of the declared (mean, std).

        Args:
            mean: Expected mean of the numeric distribution.
            std: Expected standard deviation.
            drift_threshold: Not used in per-observation check; reserved
                for future rolling distribution drift detection.
            alert_threshold: Rolling violation rate that triggers an alert.
        """
        return cls(
            constraint_type=ConstraintType.DISTRIBUTION,
            params={"mean": mean, "std": std, "drift_threshold": drift_threshold},
            alert_threshold=alert_threshold,
        )

    @classmethod
    def from_pattern(
        cls,
        regex: str,
        min_match_rate: float = 0.90,
        alert_threshold: float = 0.15,
    ) -> "SemanticConstraint":
        """String field should match the regex pattern.

        Args:
            regex: Regular expression that the field value must match.
            min_match_rate: Minimum fraction of observations that must
                match before the constraint flags (reserved for v0.2
                rolling-window check; check() is strict per-observation).
            alert_threshold: Rolling violation rate that triggers an alert.
        """
        # Validate regex at definition time
        re.compile(regex)
        return cls(
            constraint_type=ConstraintType.PATTERN_MATCH,
            params={"regex": regex, "min_match_rate": min_match_rate},
            alert_threshold=alert_threshold,
        )

    # ------------------------------------------------------------------
    # Per-observation check
    # ------------------------------------------------------------------

    def check(self, value: Any) -> Tuple[bool, str]:
        """Check a single value against this constraint.

        Returns:
            (passed, reason) where passed is True if the constraint is
            satisfied and reason is a human-readable explanation.
        """
        ct = self.constraint_type

        if ct == ConstraintType.AUTHORIZED_VALUES:
            authorized = self.params["authorized"]
            if value in authorized:
                return True, f"value '{value}' is authorized"
            return False, f"value '{value}' not in authorized set {authorized}"

        if ct == ConstraintType.LENGTH_BOUNDS:
            if not isinstance(value, str):
                return False, f"expected str, got {type(value).__name__}"
            wc = len(value.split())
            lo, hi = self.params["min_words"], self.params["max_words"]
            if lo <= wc <= hi:
                return True, f"word count {wc} within [{lo}, {hi}]"
            return False, f"word count {wc} outside [{lo}, {hi}]"

        if ct == ConstraintType.DISTRIBUTION:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False, f"cannot convert '{value}' to float"
            mean, std = self.params["mean"], self.params["std"]
            sigma = std if std > 0 else 1e-9
            z = abs(v - mean) / sigma
            if z <= 3.0:
                return True, f"value {v:.4g} within 3σ of mean {mean}"
            return False, f"value {v:.4g} is {z:.2f}σ from mean {mean}"

        if ct == ConstraintType.PATTERN_MATCH:
            if not isinstance(value, str):
                return False, f"expected str, got {type(value).__name__}"
            if re.search(self.params["regex"], value):
                return True, "pattern matched"
            return False, f"value did not match pattern '{self.params['regex']}'"

        if ct == ConstraintType.CORRELATION:
            # v0.2: cross-field correlation checking
            # For now, always passes — placeholder
            return True, "correlation check skipped (v0.2)"

        return True, "no constraint applied"  # pragma: no cover


# ---------------------------------------------------------------------------
# @spec decorator — attaches constraints to a Pydantic model class
# ---------------------------------------------------------------------------

def spec(**constraints: SemanticConstraint):
    """Attach semantic constraints to a Pydantic model class.

    Usage::

        @spec(
            category=SemanticConstraint.from_authorized_values(["pos", "neg"]),
            reasoning=SemanticConstraint.from_length_bounds(30, 300),
        )
        class SentimentAnalysis(BaseModel):
            category: str
            reasoning: str
            score: float

    The @spec decorator does NOT modify the model's behavior — it only
    attaches ``__spec_constraints__`` as class metadata. Pydantic
    validation continues to work exactly as before.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        for fname, constraint in constraints.items():
            constraint.field_name = fname
        cls.__spec_constraints__ = dict(constraints)
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Observation — one LLM output observation with check results
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single observed LLM output with per-field constraint results."""

    timestamp: float
    spec_name: str
    output_data: Dict[str, Any]
    constraint_results: Dict[str, Tuple[bool, str]]  # field -> (passed, reason)
    model_version: Optional[str] = None
    prompt_hash: Optional[str] = None
    call_id: str = field(
        default_factory=lambda: hashlib.md5(
            str(time.time_ns()).encode()
        ).hexdigest()[:16]
    )

    @property
    def passed(self) -> bool:
        """True if ALL constraint checks passed."""
        return all(r[0] for r in self.constraint_results.values())

    @property
    def violation_count(self) -> int:
        """Number of constraint checks that failed."""
        return sum(1 for r in self.constraint_results.values() if not r[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp,
            "spec_name": self.spec_name,
            "passed": int(self.passed),
            "violation_count": self.violation_count,
            "model_version": self.model_version,
            "prompt_hash": self.prompt_hash,
            "output_data": json.dumps(self.output_data),
            "constraint_results": json.dumps({
                k: {"passed": v[0], "reason": v[1]}
                for k, v in self.constraint_results.items()
            }),
        }


# ---------------------------------------------------------------------------
# ObservationStore — SQLite persistence, zero infrastructure required
# ---------------------------------------------------------------------------

class ObservationStore:
    """Persists observations to SQLite. Zero infrastructure required.

    Args:
        db_path: Path to the SQLite database file, or ``:memory:`` for
            an in-process, in-memory database (useful for testing).
    """

    _CREATE = """
    CREATE TABLE IF NOT EXISTS observations (
        call_id          TEXT PRIMARY KEY,
        timestamp        REAL NOT NULL,
        spec_name        TEXT NOT NULL,
        passed           INTEGER NOT NULL,
        violation_count  INTEGER NOT NULL,
        model_version    TEXT,
        prompt_hash      TEXT,
        output_data      TEXT,
        constraint_results TEXT
    )
    """

    def __init__(self, db_path: str = "./spec_drift.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a connection; reuse persistent one for :memory: databases."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
            return self._conn
        return sqlite3.connect(self.db_path)

    def _release(self, conn: sqlite3.Connection) -> None:
        if self.db_path != ":memory:":
            conn.close()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(self._CREATE)
        conn.commit()
        self._release(conn)

    def save(self, obs: Observation) -> None:
        """Persist an observation to the store."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO observations VALUES "
            "(:call_id,:timestamp,:spec_name,:passed,:violation_count,"
            ":model_version,:prompt_hash,:output_data,:constraint_results)",
            obs.to_dict(),
        )
        conn.commit()
        self._release(conn)

    def query(
        self,
        spec_name: str,
        since_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        """Return all observations for a spec within the time window."""
        cutoff = time.time() - since_hours * 3600
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observations "
            "WHERE spec_name=? AND timestamp>=? "
            "ORDER BY timestamp DESC",
            (spec_name, cutoff),
        ).fetchall()
        self._release(conn)
        return [dict(r) for r in rows]

    def violation_rate(self, spec_name: str, since_hours: float = 24.0) -> float:
        """Fraction of observations that failed at least one constraint."""
        rows = self.query(spec_name, since_hours)
        if not rows:
            return 0.0
        return sum(1 for r in rows if not r["passed"]) / len(rows)

    def list_specs(self) -> List[str]:
        """Return distinct spec names in the store."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT spec_name FROM observations ORDER BY spec_name"
        ).fetchall()
        self._release(conn)
        return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# DriftMonitor — runtime semantic compliance monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Wraps LLM functions to monitor semantic specification compliance.

    Usage::

        monitor = DriftMonitor(spec=SentimentAnalysis)

        @monitor.watch
        def analyze(text: str) -> SentimentAnalysis:
            ...  # call your LLM here

        # Or inline:
        result = monitor.observe(output_obj)

        # Drift report:
        report = monitor.drift_report(since_hours=168)
    """

    def __init__(
        self,
        spec: Type[T],
        db_path: str = "./spec_drift.db",
        model_version: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        alert_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.spec_class = spec
        self.spec_name = spec.__name__
        self.constraints: Dict[str, SemanticConstraint] = getattr(
            spec, "__spec_constraints__", {}
        )
        self.store = ObservationStore(db_path)
        self.model_version = model_version
        self.prompt_hash = prompt_hash
        self.alert_callback = alert_callback

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, output: T) -> T:
        """Check a Pydantic model instance and log the observation.

        Returns the output unchanged — safe to use inline::

            result = monitor.observe(my_llm_function(text))
        """
        if not isinstance(output, self.spec_class):
            raise TypeError(
                f"Expected {self.spec_class.__name__}, got {type(output).__name__}"
            )

        data = output.model_dump()
        results: Dict[str, Tuple[bool, str]] = {}
        for fname, constraint in self.constraints.items():
            value = data.get(fname)
            passed, reason = constraint.check(value)
            results[fname] = (passed, reason)

        obs = Observation(
            timestamp=time.time(),
            spec_name=self.spec_name,
            output_data=data,
            constraint_results=results,
            model_version=self.model_version,
            prompt_hash=self.prompt_hash,
        )
        self.store.save(obs)

        if self.alert_callback and results:
            vrate = self.store.violation_rate(self.spec_name, since_hours=1.0)
            for fname, constraint in self.constraints.items():
                if vrate > constraint.alert_threshold:
                    self.alert_callback(
                        f"spec-drift ALERT [{self.spec_name}.{fname}]: "
                        f"violation rate {vrate:.1%} exceeds threshold "
                        f"{constraint.alert_threshold:.1%}",
                        vrate,
                    )
                    break  # one alert per observe() call

        return output

    def watch(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator: automatically observe the return value of an LLM function.

        Usage::

            @monitor.watch
            def my_llm_function(text: str) -> SentimentAnalysis:
                ...
        """
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = fn(*args, **kwargs)
            return self.observe(result)
        return wrapper

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def drift_report(self, since_hours: float = 168.0) -> Dict[str, Any]:
        """Generate a semantic drift report for the last N hours.

        Returns a dict with keys: spec, period_hours, observations,
        violation_rate, severity, field_violation_rates.
        """
        rows = self.store.query(self.spec_name, since_hours)
        if not rows:
            return {
                "spec": self.spec_name,
                "period_hours": since_hours,
                "observations": 0,
                "status": "no_data",
            }

        total = len(rows)
        violations = sum(1 for r in rows if not r["passed"])
        vrate = violations / total

        field_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "fail": 0}
        )
        for row in rows:
            cr = json.loads(row["constraint_results"])
            for fname, result in cr.items():
                key = "pass" if result["passed"] else "fail"
                field_stats[fname][key] += 1

        field_rates = {
            fname: round(s["fail"] / (s["pass"] + s["fail"]), 4)
            for fname, s in field_stats.items()
            if s["pass"] + s["fail"] > 0
        }

        return {
            "spec": self.spec_name,
            "period_hours": since_hours,
            "observations": total,
            "violation_rate": round(vrate, 4),
            "severity": self._severity(vrate).value,
            "field_violation_rates": field_rates,
        }

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(vrate: float) -> DriftSeverity:
        if vrate == 0.0:
            return DriftSeverity.NONE
        if vrate < 0.05:
            return DriftSeverity.LOW
        if vrate < 0.15:
            return DriftSeverity.MEDIUM
        if vrate < 0.30:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL


# ---------------------------------------------------------------------------
# CI gate
# ---------------------------------------------------------------------------

def run_ci_gate(
    monitor: DriftMonitor,
    test_outputs: List[T],
    threshold: float = 0.20,
) -> Tuple[bool, Dict[str, Any]]:
    """Run a CI gate check on a batch of test outputs.

    Observes each output in test_outputs, then generates a report and
    compares the violation rate against threshold.

    Returns:
        (passed, report) — passed is False if violation_rate > threshold.

    Usage::

        passed, report = run_ci_gate(monitor, outputs, threshold=0.20)
        sys.exit(0 if passed else 1)
    """
    for output in test_outputs:
        monitor.observe(output)

    # Use a very short window to capture only this batch
    report = monitor.drift_report(since_hours=0.1)
    vrate = report.get("violation_rate", 0.0)
    passed = vrate <= threshold
    report["ci_threshold"] = threshold
    report["ci_passed"] = passed
    return passed, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spec-drift",
        description="Semantic specification drift detector for LLM outputs.",
    )
    p.add_argument("--version", action="version", version=f"spec-drift {__version__}")

    sub = p.add_subparsers(dest="command")

    # report command
    rpt = sub.add_parser(
        "report",
        help="Print a drift report for all specs in a database.",
    )
    rpt.add_argument("--db", default="./spec_drift.db", help="SQLite database path")
    rpt.add_argument(
        "--since",
        type=float,
        default=168.0,
        metavar="HOURS",
        help="Report window in hours (default: 168 = 7 days)",
    )
    rpt.add_argument("--json", action="store_true", help="Output as JSON")

    # ci command
    ci = sub.add_parser(
        "ci",
        help="CI gate: exit 1 if any spec's violation rate exceeds threshold.",
    )
    ci.add_argument("--db", default="./spec_drift.db", help="SQLite database path")
    ci.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        metavar="RATE",
        help="Violation rate threshold (default: 0.20)",
    )
    ci.add_argument(
        "--since",
        type=float,
        default=168.0,
        metavar="HOURS",
        help="Look-back window in hours (default: 168)",
    )

    return p


def _cli_main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "report":
        store = ObservationStore(db_path=args.db)
        specs = store.list_specs()
        if not specs:
            print("No observations found in database.")
            return 0

        if args.json:
            reports = []
            for spec_name in specs:
                rows = store.query(spec_name, since_hours=args.since)
                if not rows:
                    continue
                total = len(rows)
                violations = sum(1 for r in rows if not r["passed"])
                vrate = violations / total
                reports.append({
                    "spec": spec_name,
                    "observations": total,
                    "violation_rate": round(vrate, 4),
                    "severity": DriftMonitor._severity(vrate).value,
                })
            print(json.dumps(reports, indent=2))
        else:
            _print_report_table(store, specs, args.since)
        return 0

    if args.command == "ci":
        store = ObservationStore(db_path=args.db)
        specs = store.list_specs()
        if not specs:
            print("spec-drift CI: No observations found. Passing by default.")
            return 0

        failed: List[str] = []
        for spec_name in specs:
            rows = store.query(spec_name, since_hours=args.since)
            if not rows:
                continue
            total = len(rows)
            violations = sum(1 for r in rows if not r["passed"])
            vrate = violations / total
            severity = DriftMonitor._severity(vrate)
            status = "PASS" if vrate <= args.threshold else "FAIL"
            if vrate > args.threshold:
                failed.append(spec_name)
            print(
                f"  {status}  {spec_name:40s}  "
                f"violation_rate={vrate:.1%}  "
                f"severity={severity.value}"
            )

        if failed:
            print(
                f"\nspec-drift CI FAILED: {len(failed)} spec(s) exceeded "
                f"threshold {args.threshold:.1%}: {', '.join(failed)}"
            )
            return 1

        print(f"\nspec-drift CI PASSED: all specs within threshold {args.threshold:.1%}")
        return 0

    parser.print_help()
    return 0


def _print_report_table(
    store: ObservationStore,
    specs: List[str],
    since_hours: float,
) -> None:
    print(f"\nspec-drift report (last {since_hours:.0f}h)\n")
    print(f"  {'SPEC':<40}  {'OBS':>6}  {'VIOL%':>7}  SEVERITY")
    print(f"  {'-'*40}  {'-'*6}  {'-'*7}  {'-'*8}")
    for spec_name in specs:
        rows = store.query(spec_name, since_hours=since_hours)
        if not rows:
            continue
        total = len(rows)
        violations = sum(1 for r in rows if not r["passed"])
        vrate = violations / total
        severity = DriftMonitor._severity(vrate)
        print(
            f"  {spec_name:<40}  {total:>6}  "
            f"{vrate:>6.1%}  {severity.value}"
        )
    print()


def main() -> None:  # pragma: no cover
    sys.exit(_cli_main())


if __name__ == "__main__":  # pragma: no cover
    main()
