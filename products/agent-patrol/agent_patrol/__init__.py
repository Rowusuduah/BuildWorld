"""
agent-patrol: Runtime pathology detection for AI agents.

Diagnoses loops, stalls, oscillation, drift, and silent abandonment.
Framework-agnostic. Zero hard dependencies.
"""

from __future__ import annotations

__version__ = "0.1.0"

import enum
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ─── Enums ───────────────────────────────────────────────────────────────────


class Pathology(str, enum.Enum):
    """The five agent pathologies."""
    FUTILE_CYCLE = "futile_cycle"
    OSCILLATION = "oscillation"
    STALL = "stall"
    DRIFT = "drift"
    ABANDONMENT = "abandonment"


class Severity(str, enum.Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class Sensitivity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class StepObservation:
    """A single observed agent step."""
    action: str
    result: str = ""
    step_number: int = 0
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def fingerprint(self) -> str:
        raw = self.action.strip().lower()
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def tokens(self) -> set[str]:
        """Bag-of-words tokenization for similarity."""
        words = re.findall(r'\w+', (self.action + " " + self.result).lower())
        return set(words)


@dataclass
class PathologyReport:
    """Report from a single observation."""
    step_number: int
    pathology: Optional[Pathology] = None
    severity: Optional[Severity] = None
    confidence: float = 0.0
    evidence: str = ""
    recommended_action: str = ""
    step_detected: Optional[int] = None

    @property
    def is_healthy(self) -> bool:
        return self.pathology is None

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "pathology": self.pathology.value if self.pathology else None,
            "severity": self.severity.value if self.severity else None,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
        }


@dataclass
class PatrolSummary:
    """Summary of a full agent run."""
    total_steps: int = 0
    pathologies_detected: list[PathologyReport] = field(default_factory=list)
    health_score: float = 1.0
    verdict: str = "healthy"

    @property
    def is_healthy(self) -> bool:
        return len(self.pathologies_detected) == 0

    def to_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "health_score": round(self.health_score, 3),
            "verdict": self.verdict,
            "pathologies": [p.to_dict() for p in self.pathologies_detected],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─── Similarity Engine (Pure Python, no dependencies) ────────────────────────


def _jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _cosine_similarity_bow(a: str, b: str) -> float:
    """Cosine similarity using bag-of-words."""
    words_a = re.findall(r'\w+', a.lower())
    words_b = re.findall(r'\w+', b.lower())
    if not words_a or not words_b:
        return 0.0

    counter_a = Counter(words_a)
    counter_b = Counter(words_b)
    all_words = set(counter_a.keys()) | set(counter_b.keys())

    dot = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Detectors ───────────────────────────────────────────────────────────────


def _get_thresholds(sensitivity: Sensitivity) -> dict:
    """Get detection thresholds based on sensitivity."""
    if sensitivity == Sensitivity.HIGH:
        return {
            "cycle_similarity": 0.65,
            "cycle_window": 3,
            "oscillation_similarity": 0.55,
            "stall_similarity": 0.60,
            "stall_window": 3,
            "drift_threshold": 0.35,
            "abandon_threshold": 0.25,
        }
    elif sensitivity == Sensitivity.LOW:
        return {
            "cycle_similarity": 0.90,
            "cycle_window": 5,
            "oscillation_similarity": 0.80,
            "stall_similarity": 0.85,
            "stall_window": 5,
            "drift_threshold": 0.15,
            "abandon_threshold": 0.10,
        }
    else:  # MEDIUM
        return {
            "cycle_similarity": 0.80,
            "cycle_window": 4,
            "oscillation_similarity": 0.70,
            "stall_similarity": 0.75,
            "stall_window": 4,
            "drift_threshold": 0.25,
            "abandon_threshold": 0.15,
        }


def _detect_futile_cycle(
    observations: list[StepObservation],
    thresholds: dict,
) -> Optional[PathologyReport]:
    """Detect when agent repeats semantically similar actions."""
    window = thresholds["cycle_window"]
    sim_threshold = thresholds["cycle_similarity"]

    if len(observations) < window:
        return None

    recent = observations[-window:]
    similarities = []
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            sim = _jaccard_similarity(recent[i].tokens, recent[j].tokens)
            similarities.append(sim)

    if not similarities:
        return None

    avg_sim = sum(similarities) / len(similarities)

    if avg_sim >= sim_threshold:
        confidence = min(1.0, (avg_sim - sim_threshold) / (1.0 - sim_threshold) + 0.5)
        return PathologyReport(
            step_number=observations[-1].step_number,
            pathology=Pathology.FUTILE_CYCLE,
            severity=Severity.CRITICAL if avg_sim > 0.9 else Severity.WARNING,
            confidence=confidence,
            evidence=f"Last {window} actions have avg similarity {avg_sim:.2f} "
                     f"(threshold: {sim_threshold:.2f})",
            recommended_action="Break the loop: inject new context, change strategy, or terminate.",
            step_detected=observations[-1].step_number,
        )
    return None


def _detect_oscillation(
    observations: list[StepObservation],
    thresholds: dict,
) -> Optional[PathologyReport]:
    """Detect when agent alternates between two contradictory actions."""
    if len(observations) < 4:
        return None

    recent = observations[-6:] if len(observations) >= 6 else observations

    # Check for A-B-A-B pattern
    for i in range(len(recent) - 3):
        sim_02 = _jaccard_similarity(recent[i].tokens, recent[i + 2].tokens)
        sim_13 = _jaccard_similarity(recent[i + 1].tokens, recent[i + 3].tokens)
        sim_01 = _jaccard_similarity(recent[i].tokens, recent[i + 1].tokens)

        osc_threshold = thresholds["oscillation_similarity"]

        if sim_02 >= osc_threshold and sim_13 >= osc_threshold and sim_01 < osc_threshold:
            confidence = min(1.0, ((sim_02 + sim_13) / 2 - osc_threshold) /
                             (1.0 - osc_threshold) + 0.5)
            return PathologyReport(
                step_number=observations[-1].step_number,
                pathology=Pathology.OSCILLATION,
                severity=Severity.CRITICAL,
                confidence=confidence,
                evidence=f"A-B-A-B pattern detected: steps {recent[i].step_number}-"
                         f"{recent[i+3].step_number} alternate between two action groups "
                         f"(cross-sim: {sim_02:.2f}, {sim_13:.2f})",
                recommended_action="Agent is double-minded. Force a decision or provide "
                                   "conflict resolution context.",
                step_detected=recent[i].step_number,
            )
    return None


def _detect_stall(
    observations: list[StepObservation],
    task_description: str,
    milestones: list[str],
    thresholds: dict,
) -> Optional[PathologyReport]:
    """Detect when agent is active but not making progress."""
    window = thresholds["stall_window"]
    sim_threshold = thresholds["stall_similarity"]

    if len(observations) < window:
        return None

    recent = observations[-window:]

    # Check if recent actions are similar to each other (not progressing)
    similarities = []
    for i in range(len(recent) - 1):
        sim = _jaccard_similarity(recent[i].tokens, recent[i + 1].tokens)
        similarities.append(sim)

    if not similarities:
        return None

    avg_consecutive_sim = sum(similarities) / len(similarities)

    # Also check if any milestone language is appearing
    milestone_tokens = set()
    for m in milestones:
        milestone_tokens.update(re.findall(r'\w+', m.lower()))

    recent_tokens = set()
    for obs in recent:
        recent_tokens.update(obs.tokens)

    milestone_overlap = len(milestone_tokens & recent_tokens) / max(len(milestone_tokens), 1)

    if avg_consecutive_sim >= sim_threshold and milestone_overlap < 0.3:
        confidence = min(1.0, avg_consecutive_sim * 0.7 + (1 - milestone_overlap) * 0.3)
        return PathologyReport(
            step_number=observations[-1].step_number,
            pathology=Pathology.STALL,
            severity=Severity.WARNING,
            confidence=confidence,
            evidence=f"Last {window} actions show high repetition "
                     f"(avg sim: {avg_consecutive_sim:.2f}) with low milestone progress "
                     f"(overlap: {milestone_overlap:.2f})",
            recommended_action="Agent is spinning wheels. Provide intermediate guidance "
                               "or decompose the current subtask.",
            step_detected=observations[-window].step_number,
        )
    return None


def _detect_drift(
    observations: list[StepObservation],
    task_description: str,
    thresholds: dict,
) -> Optional[PathologyReport]:
    """Detect when agent wanders from original task."""
    if len(observations) < 3:
        return None

    drift_threshold = thresholds["drift_threshold"]
    task_tokens = set(re.findall(r'\w+', task_description.lower()))

    if not task_tokens:
        return None

    # Compare early actions to recent actions in relation to task
    early = observations[:min(3, len(observations))]
    recent = observations[-3:]

    early_task_sim = sum(
        _jaccard_similarity(o.tokens, task_tokens) for o in early
    ) / len(early)

    recent_task_sim = sum(
        _jaccard_similarity(o.tokens, task_tokens) for o in recent
    ) / len(recent)

    drift_amount = early_task_sim - recent_task_sim

    if drift_amount > drift_threshold and recent_task_sim < 0.3:
        confidence = min(1.0, drift_amount / 0.5 * 0.7 + 0.3)
        return PathologyReport(
            step_number=observations[-1].step_number,
            pathology=Pathology.DRIFT,
            severity=Severity.CRITICAL if drift_amount > 0.5 else Severity.WARNING,
            confidence=confidence,
            evidence=f"Task relevance dropped from {early_task_sim:.2f} to "
                     f"{recent_task_sim:.2f} (drift: {drift_amount:.2f})",
            recommended_action="Re-anchor agent to original task. Inject task reminder.",
            step_detected=observations[-3].step_number,
        )
    return None


def _detect_abandonment(
    observations: list[StepObservation],
    task_description: str,
    thresholds: dict,
) -> Optional[PathologyReport]:
    """Detect when agent silently abandons task for something else."""
    if len(observations) < 5:
        return None

    abandon_threshold = thresholds["abandon_threshold"]
    task_tokens = set(re.findall(r'\w+', task_description.lower()))

    if not task_tokens:
        return None

    recent = observations[-3:]

    # Low similarity to original task
    task_sim = sum(
        _jaccard_similarity(o.tokens, task_tokens) for o in recent
    ) / len(recent)

    # BUT high coherence among recent actions (doing something else consistently)
    recent_coherence = 0.0
    if len(recent) >= 2:
        pairs = []
        for i in range(len(recent) - 1):
            pairs.append(_jaccard_similarity(recent[i].tokens, recent[i + 1].tokens))
        recent_coherence = sum(pairs) / len(pairs)

    if task_sim < abandon_threshold and recent_coherence > 0.5:
        confidence = min(1.0, (1 - task_sim) * 0.5 + recent_coherence * 0.5)
        return PathologyReport(
            step_number=observations[-1].step_number,
            pathology=Pathology.ABANDONMENT,
            severity=Severity.CRITICAL,
            confidence=confidence,
            evidence=f"Task relevance: {task_sim:.2f} (very low) but internal coherence: "
                     f"{recent_coherence:.2f} (high) — agent is consistently doing "
                     f"something other than its task",
            recommended_action="Agent has abandoned its mission. Terminate and restart "
                               "with stronger task anchoring.",
            step_detected=observations[-3].step_number,
        )
    return None


# ─── PatrolMonitor ───────────────────────────────────────────────────────────


class PatrolMonitor:
    """
    Main monitoring class. Observes agent steps and detects pathologies.

    Usage:
        monitor = PatrolMonitor(task_description="Research GDP data")
        for step in agent_loop():
            report = monitor.observe(step.action, step.result)
            if report.pathology:
                handle_pathology(report)
    """

    def __init__(
        self,
        task_description: str = "",
        milestones: Optional[list[str]] = None,
        sensitivity: str | Sensitivity = Sensitivity.MEDIUM,
        on_pathology: str = "log",
    ):
        self.task_description = task_description
        self.milestones = milestones or []
        self.sensitivity = (
            Sensitivity(sensitivity) if isinstance(sensitivity, str) else sensitivity
        )
        self.on_pathology = on_pathology
        self._observations: list[StepObservation] = []
        self._reports: list[PathologyReport] = []
        self._step_counter = 0
        self._thresholds = _get_thresholds(self.sensitivity)

    @property
    def observations(self) -> list[StepObservation]:
        return list(self._observations)

    @property
    def step_count(self) -> int:
        return self._step_counter

    def observe(self, action: str, result: str = "", **metadata) -> PathologyReport:
        """
        Observe a single agent step and check for pathologies.

        Args:
            action: Description of what the agent did.
            result: The result/output of the action.
            **metadata: Additional metadata to store.

        Returns:
            PathologyReport (check .is_healthy or .pathology).
        """
        self._step_counter += 1
        obs = StepObservation(
            action=action,
            result=result,
            step_number=self._step_counter,
            metadata=metadata,
        )
        self._observations.append(obs)

        # Run all detectors
        report = self._run_detectors()
        if not report.is_healthy:
            self._reports.append(report)
        return report

    def _run_detectors(self) -> PathologyReport:
        """Run all pathology detectors and return the most severe finding."""
        detections: list[PathologyReport] = []

        cycle = _detect_futile_cycle(self._observations, self._thresholds)
        if cycle:
            detections.append(cycle)

        osc = _detect_oscillation(self._observations, self._thresholds)
        if osc:
            detections.append(osc)

        stall = _detect_stall(
            self._observations, self.task_description,
            self.milestones, self._thresholds,
        )
        if stall:
            detections.append(stall)

        drift = _detect_drift(
            self._observations, self.task_description, self._thresholds,
        )
        if drift:
            detections.append(drift)

        abandon = _detect_abandonment(
            self._observations, self.task_description, self._thresholds,
        )
        if abandon:
            detections.append(abandon)

        if not detections:
            return PathologyReport(step_number=self._step_counter)

        # Return highest confidence detection
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[0]

    def summary(self) -> PatrolSummary:
        """Generate summary of the full monitoring session."""
        if self._step_counter == 0:
            return PatrolSummary()

        pathology_count = len(self._reports)
        health_score = max(0.0, 1.0 - (pathology_count / max(self._step_counter, 1)))

        if pathology_count == 0:
            verdict = "healthy"
        elif any(r.severity == Severity.CRITICAL for r in self._reports):
            verdict = "critical"
        else:
            verdict = "degraded"

        return PatrolSummary(
            total_steps=self._step_counter,
            pathologies_detected=list(self._reports),
            health_score=health_score,
            verdict=verdict,
        )

    def reset(self):
        """Reset monitor state for a new run."""
        self._observations.clear()
        self._reports.clear()
        self._step_counter = 0


# ─── Decorator API ───────────────────────────────────────────────────────────


_default_monitor: Optional[PatrolMonitor] = None


def patrol(
    on_pathology: str = "log",
    sensitivity: str = "medium",
    task_description: str = "",
) -> Callable:
    """
    Decorator to monitor agent step functions for pathologies.

    Args:
        on_pathology: "log" (print warning), "raise" (raise exception), "ignore"
        sensitivity: "low", "medium", "high"
        task_description: What the agent is supposed to be doing.
    """
    def decorator(func: Callable) -> Callable:
        monitor = PatrolMonitor(
            task_description=task_description or func.__name__,
            sensitivity=sensitivity,
            on_pathology=on_pathology,
        )

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            action_str = f"{func.__name__}({', '.join(str(a)[:50] for a in args)})"
            result_str = str(result)[:200] if result is not None else ""

            report = monitor.observe(action_str, result_str)

            if not report.is_healthy:
                msg = (f"[agent-patrol] {report.pathology.value} detected at step "
                       f"{report.step_number}: {report.evidence}")
                if on_pathology == "raise":
                    raise AgentPathologyError(msg, report)
                elif on_pathology == "log":
                    import sys
                    print(msg, file=sys.stderr)

            return result

        wrapper._patrol_monitor = monitor
        wrapper.__wrapped__ = func
        return wrapper

    return decorator


class AgentPathologyError(Exception):
    """Raised when a pathology is detected and on_pathology='raise'."""

    def __init__(self, message: str, report: PathologyReport):
        super().__init__(message)
        self.report = report


# ─── Store ───────────────────────────────────────────────────────────────────


class PatrolStore:
    """SQLite store for patrol run history."""

    def __init__(self, db_path: str | Path = "agent_patrol.db"):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    def _ensure_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS patrol_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_description TEXT,
                total_steps INTEGER NOT NULL,
                health_score REAL NOT NULL,
                verdict TEXT NOT NULL,
                pathology_count INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def save(self, summary: PatrolSummary, task_description: str = "") -> int:
        cur = self._conn.execute(
            """INSERT INTO patrol_runs
               (task_description, total_steps, health_score, verdict,
                pathology_count, summary_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                task_description,
                summary.total_steps,
                summary.health_score,
                summary.verdict,
                len(summary.pathologies_detected),
                summary.to_json(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_history(self, limit: int = 10) -> list[dict]:
        cur = self._conn.execute(
            """SELECT id, task_description, total_steps, health_score,
                      verdict, pathology_count, created_at
               FROM patrol_runs ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "id": row[0], "task": row[1], "steps": row[2],
                "health_score": row[3], "verdict": row[4],
                "pathologies": row[5], "created_at": row[6],
            }
            for row in cur.fetchall()
        ]

    def get_run(self, run_id: int) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT summary_json FROM patrol_runs WHERE id = ?", (run_id,)
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self._conn.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _cli_main():
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("agent-patrol: Runtime pathology detection for AI agents")
        print()
        print("Usage:")
        print("  agent-patrol version        — Show version")
        print("  agent-patrol history         — Show recent patrol runs")
        print("  agent-patrol show <run_id>   — Show details of a run")
        print()
        print("Python API:")
        print("  from agent_patrol import PatrolMonitor, patrol")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "version":
        print(f"agent-patrol {__version__}")
    elif cmd == "history":
        store = PatrolStore()
        for h in store.get_history():
            status = "✓" if h["verdict"] == "healthy" else "✗"
            print(f"  {status} #{h['id']} {h['verdict']} "
                  f"({h['pathologies']} issues in {h['steps']} steps) "
                  f"@ {h['created_at']}")
        store.close()
    elif cmd == "show":
        if len(sys.argv) < 3:
            print("Usage: agent-patrol show <run_id>")
            sys.exit(1)
        store = PatrolStore()
        run = store.get_run(int(sys.argv[2]))
        print(json.dumps(run, indent=2) if run else f"Run #{sys.argv[2]} not found.")
        store.close()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
