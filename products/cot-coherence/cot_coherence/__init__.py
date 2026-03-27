"""
cot-coherence: Chain-of-Thought Coherence Verifier
===================================================
The first pip-installable tool that verifies whether an LLM's chain-of-thought
reasoning is internally coherent — step continuity, conclusion grounding,
internal consistency, reasoning completeness, and confidence calibration.

The gap: LLMs routinely produce reasoning that *looks* correct but is incoherent.
Steps don't follow from each other. Contradictions hide between lines. Conclusions
aren't entailed by the reasoning. In CI/CD pipelines that trust LLM reasoning as
a gate, this is a silent production risk.

Biblical Pattern: Proverbs 14:12 — "There is a way that appears to be right,
but in the end it leads to death." The reasoning looks valid. The conclusion is
wrong. cot-coherence makes the gap visible.

Author: BuildWorld — Cycle 006
License: MIT
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CoherenceStatus(str, Enum):
    COHERENT = "COHERENT"       # All dimensions pass — reasoning is sound
    DEGRADED = "DEGRADED"       # Some issues detected — review recommended
    INCOHERENT = "INCOHERENT"   # Critical issues — do not trust this reasoning
    SKIP = "SKIP"               # Cannot evaluate (empty input, etc.)


class ViolationType(str, Enum):
    STEP_GAP = "STEP_GAP"                           # Step doesn't follow from previous
    CONTRADICTION = "CONTRADICTION"                 # Two steps contradict each other
    UNSUPPORTED_CONCLUSION = "UNSUPPORTED_CONCLUSION"  # Conclusion not entailed by reasoning
    REASONING_LEAP = "REASONING_LEAP"               # Unexplained logical jump
    OVERCONFIDENCE = "OVERCONFIDENCE"               # Claimed certainty not warranted by steps
    CIRCULAR = "CIRCULAR"                           # Step merely restates a prior step
    SCOPE_SHIFT = "SCOPE_SHIFT"                     # Reasoning suddenly shifts domain/scope


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CoTStep:
    """A single step in a chain-of-thought reasoning sequence."""
    index: int
    text: str

    def short(self) -> str:
        return self.text[:120] + ("..." if len(self.text) > 120 else "")


@dataclass
class CoherenceViolation:
    """A detected coherence violation in the reasoning chain."""
    violation_type: ViolationType
    severity: float          # 0.0 (minor) – 1.0 (critical)
    step_indices: list[int]  # Which steps are involved (empty = conclusion)
    description: str         # Human-readable explanation
    evidence: str            # Specific quote or paraphrase from the steps

    def is_critical(self) -> bool:
        return self.severity >= 0.7


@dataclass
class DimensionScore:
    """Score for a single coherence dimension."""
    name: str
    score: float      # 1.0 = perfect, 0.0 = completely failed
    passed: bool
    notes: str


@dataclass
class CoherenceReport:
    """
    Full coherence verification report for a chain-of-thought sequence.

    coherence_score: 0.0 = perfectly incoherent, 1.0 = perfectly coherent
    (inverse of the drift_score convention used in drift-guard — coherence
    is a positive quality score here, not an error score.)
    """
    steps: list[CoTStep]
    conclusion: str
    status: CoherenceStatus
    coherence_score: float          # 0.0 – 1.0 (higher = more coherent)
    incoherence_score: float        # 1.0 - coherence_score (for CI thresholds)
    overall_confidence: float       # LLM's confidence in its own assessment
    violations: list[CoherenceViolation]
    dimensions: list[DimensionScore]
    summary: str                    # 2-3 sentence synthesis
    timestamp: str
    model_used: str
    step_count: int

    def passed(self) -> bool:
        return self.status == CoherenceStatus.COHERENT

    def critical_violations(self) -> list[CoherenceViolation]:
        return [v for v in self.violations if v.is_critical()]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "coherence_score": round(self.coherence_score, 4),
            "incoherence_score": round(self.incoherence_score, 4),
            "overall_confidence": round(self.overall_confidence, 4),
            "step_count": self.step_count,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": round(v.severity, 4),
                    "step_indices": v.step_indices,
                    "description": v.description,
                    "evidence": v.evidence,
                }
                for v in self.violations
            ],
            "dimensions": [
                {
                    "name": d.name,
                    "score": round(d.score, 4),
                    "passed": d.passed,
                    "notes": d.notes,
                }
                for d in self.dimensions
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
        }

    def to_markdown(self) -> str:
        icon = {
            CoherenceStatus.COHERENT: "✅",
            CoherenceStatus.DEGRADED: "⚠️",
            CoherenceStatus.INCOHERENT: "❌",
            CoherenceStatus.SKIP: "⏭️",
        }
        lines = [
            "## cot-coherence Report",
            "",
            f"**Status:** {icon.get(self.status, '?')} {self.status.value}",
            f"**Coherence Score:** {self.coherence_score:.2f} / 1.00",
            f"**Steps:** {self.step_count}",
            f"**Violations:** {len(self.violations)} "
            f"({len(self.critical_violations())} critical)",
            "",
            "### Reasoning Steps",
            "",
        ]
        for s in self.steps:
            lines.append(f"{s.index + 1}. {s.short()}")
        lines += [
            "",
            f"**Conclusion:** {self.conclusion[:200]}",
            "",
            "### Dimensions",
            "",
        ]
        dim_icon = {True: "✅", False: "❌"}
        for d in self.dimensions:
            lines.append(f"- {dim_icon[d.passed]} **{d.name}** ({d.score:.2f}) — {d.notes}")
        if self.violations:
            lines += ["", "### Violations", ""]
            sev_icon = {True: "🔴", False: "🟡"}
            for v in self.violations:
                lines.append(
                    f"- {sev_icon[v.is_critical()]} **{v.violation_type.value}** "
                    f"(severity {v.severity:.2f})"
                )
                lines.append(f"  - {v.description}")
                if v.evidence:
                    lines.append(f"  - *Evidence:* `{v.evidence[:120]}`")
        lines += [
            "",
            "### Summary",
            "",
            self.summary,
            "",
            "---",
            f"*Generated by cot-coherence at {self.timestamp} using {self.model_used}*",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------

def parse_steps(raw: str | list[str]) -> list[CoTStep]:
    """
    Parse reasoning steps from either a list of strings or a single string.
    If a single string, splits on common CoT delimiters:
      - Numbered lines: "1. ...", "Step 1:", "Step 1 —"
      - Blank-line separated paragraphs
      - Explicit markers: "First", "Second", "Then", "Therefore", "Finally"
    """
    if isinstance(raw, list):
        return [
            CoTStep(index=i, text=s.strip())
            for i, s in enumerate(raw)
            if s.strip()
        ]

    # Try numbered list first
    numbered = re.split(r"\n\s*(?:\d+[.)]\s*|Step\s+\d+[:.—\s])", raw.strip())
    if len(numbered) > 1:
        return [
            CoTStep(index=i, text=s.strip())
            for i, s in enumerate(numbered)
            if s.strip() and len(s.strip()) > 5
        ]

    # Try blank-line paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", raw.strip()) if p.strip()]
    if len(paragraphs) > 1:
        return [CoTStep(index=i, text=p) for i, p in enumerate(paragraphs)]

    # Try natural connectors
    connector_split = re.split(
        r"(?:^|\. |\n)(?=(?:First|Second|Third|Fourth|Next|Then|After that|"
        r"Therefore|Thus|Finally|However|But|Additionally|Furthermore|"
        r"Also|Moreover)\b)",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    if len(connector_split) > 1:
        return [
            CoTStep(index=i, text=s.strip())
            for i, s in enumerate(connector_split)
            if s.strip() and len(s.strip()) > 5
        ]

    # Fallback: treat as single step
    return [CoTStep(index=0, text=raw.strip())]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are cot-coherence, a chain-of-thought reasoning coherence verifier.
Your job is to evaluate whether a sequence of reasoning steps logically and
coherently leads to the stated conclusion.

You evaluate FIVE dimensions:

1. step_continuity (0.0–1.0)
   Does each step follow naturally from the step before it?
   Score 1.0 if every transition is smooth and logically warranted.
   Penalize for gaps, non-sequiturs, or abrupt topic shifts between steps.

2. conclusion_grounding (0.0–1.0)
   Does the final conclusion actually follow from the last step(s)?
   Score 1.0 if the conclusion is clearly entailed.
   Penalize if the conclusion introduces new claims not established in the steps.

3. internal_consistency (0.0–1.0)
   Are all steps internally consistent with each other?
   Score 1.0 if no two steps contradict each other.
   Penalize for contradictions, circular reasoning, or restated premises.

4. reasoning_completeness (0.0–1.0)
   Does the reasoning cover all necessary logical territory?
   Score 1.0 if no important step is skipped.
   Penalize for unexplained leaps where a critical intermediate step is missing.

5. confidence_calibration (0.0–1.0)
   Is the expressed certainty in the steps and conclusion warranted?
   Score 1.0 if confidence is proportional to what the steps establish.
   Penalize if the conclusion claims certainty ("clearly", "definitely", "must")
   that the steps do not fully justify.

Also identify specific VIOLATIONS (may be empty list):
- STEP_GAP: Step N does not follow from Step N-1
- CONTRADICTION: Two steps assert contradictory things
- UNSUPPORTED_CONCLUSION: Conclusion is not entailed by the reasoning
- REASONING_LEAP: A critical intermediate step is missing
- OVERCONFIDENCE: Certainty claimed not warranted by the steps
- CIRCULAR: A step merely restates a prior step without adding new logic
- SCOPE_SHIFT: Reasoning suddenly shifts domain/subject without justification

Respond with ONLY valid JSON in this EXACT schema:
{
  "dimensions": {
    "step_continuity": {"score": 0.0, "notes": "..."},
    "conclusion_grounding": {"score": 0.0, "notes": "..."},
    "internal_consistency": {"score": 0.0, "notes": "..."},
    "reasoning_completeness": {"score": 0.0, "notes": "..."},
    "confidence_calibration": {"score": 0.0, "notes": "..."}
  },
  "violations": [
    {
      "type": "STEP_GAP",
      "severity": 0.8,
      "step_indices": [1, 2],
      "description": "Step 2 introduces X without explaining how it follows from Step 1's claim about Y.",
      "evidence": "Step 1: '...'; Step 2: '...'"
    }
  ],
  "coherence_score": 0.75,
  "overall_confidence": 0.90,
  "summary": "2-3 sentence synthesis of the coherence assessment. Be specific about what is strong and what is weak in the reasoning chain."
}

Rules:
- coherence_score is the AVERAGE of the five dimension scores.
- overall_confidence is YOUR confidence in your own assessment (not the reasoning's confidence).
- Be concrete. Quote or paraphrase specific steps when identifying violations.
- Empty violations list is correct when reasoning is genuinely sound.
- A single-step reasoning chain cannot have STEP_GAP but can have UNSUPPORTED_CONCLUSION.
"""

DEFAULT_MODEL = os.environ.get("COT_COHERENCE_MODEL", "claude-haiku-4-5-20251001")

# Coherence thresholds for automatic status assignment
_THRESHOLD_COHERENT = 0.80    # >= 0.80 → COHERENT
_THRESHOLD_DEGRADED = 0.55    # 0.55–0.79 → DEGRADED
                               # < 0.55 → INCOHERENT


def _call_judge(
    steps: list[CoTStep],
    conclusion: str,
    model: str,
    api_key: Optional[str],
) -> dict:
    """Call the Claude judge and return the parsed JSON response."""
    if anthropic is None:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    steps_text = "\n".join(
        f"Step {s.index + 1}: {s.text[:600]}"
        for s in steps
    )

    user_content = f"""REASONING STEPS ({len(steps)} total):

{steps_text}

CONCLUSION:
{conclusion[:800]}

Evaluate the coherence of this chain-of-thought and return the JSON report."""

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    content = message.content[0].text.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return json.loads(content)


def _parse_judge_response(
    raw: dict,
    steps: list[CoTStep],
    conclusion: str,
    model: str,
) -> CoherenceReport:
    """Convert a raw LLM judge response into a CoherenceReport."""
    dim_data = raw.get("dimensions", {})
    dim_names = [
        "step_continuity",
        "conclusion_grounding",
        "internal_consistency",
        "reasoning_completeness",
        "confidence_calibration",
    ]
    dimensions: list[DimensionScore] = []
    for name in dim_names:
        d = dim_data.get(name, {})
        score = float(d.get("score", 0.5))
        dimensions.append(DimensionScore(
            name=name,
            score=score,
            passed=score >= 0.7,
            notes=d.get("notes", ""),
        ))

    coherence_score = float(raw.get("coherence_score", 0.5))
    # Recompute from dimensions if LLM gave inconsistent value
    if dimensions:
        recomputed = sum(d.score for d in dimensions) / len(dimensions)
        # Use LLM value if close, otherwise use recomputed
        if abs(recomputed - coherence_score) > 0.15:
            coherence_score = recomputed

    violations: list[CoherenceViolation] = []
    for v in raw.get("violations", []):
        try:
            vtype = ViolationType(v.get("type", "REASONING_LEAP"))
        except ValueError:
            vtype = ViolationType.REASONING_LEAP
        violations.append(CoherenceViolation(
            violation_type=vtype,
            severity=float(v.get("severity", 0.5)),
            step_indices=v.get("step_indices", []),
            description=v.get("description", ""),
            evidence=v.get("evidence", ""),
        ))

    if coherence_score >= _THRESHOLD_COHERENT:
        status = CoherenceStatus.COHERENT
    elif coherence_score >= _THRESHOLD_DEGRADED:
        status = CoherenceStatus.DEGRADED
    else:
        status = CoherenceStatus.INCOHERENT

    return CoherenceReport(
        steps=steps,
        conclusion=conclusion,
        status=status,
        coherence_score=coherence_score,
        incoherence_score=round(1.0 - coherence_score, 4),
        overall_confidence=float(raw.get("overall_confidence", 0.8)),
        violations=violations,
        dimensions=dimensions,
        summary=raw.get("summary", ""),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_used=model,
        step_count=len(steps),
    )


# ---------------------------------------------------------------------------
# SQLite trace log
# ---------------------------------------------------------------------------

def _get_db_path() -> Path:
    return Path(os.environ.get("COT_COHERENCE_DB", ".cot-coherence.db"))


def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coherence_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            coherence_score REAL NOT NULL,
            incoherence_score REAL NOT NULL,
            step_count INTEGER NOT NULL,
            violation_count INTEGER NOT NULL,
            critical_violation_count INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            report_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_report(report: CoherenceReport, db_path: Optional[Path] = None) -> None:
    """Persist a CoherenceReport to the SQLite trace log."""
    if db_path is None:
        db_path = _get_db_path()
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO coherence_reports
            (timestamp, status, coherence_score, incoherence_score, step_count,
             violation_count, critical_violation_count, model_used, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report.timestamp,
        report.status.value,
        report.coherence_score,
        report.incoherence_score,
        report.step_count,
        len(report.violations),
        len(report.critical_violations()),
        report.model_used,
        json.dumps(report.to_dict()),
    ))
    conn.commit()
    conn.close()


def load_recent_reports(n: int = 10, db_path: Optional[Path] = None) -> list[dict]:
    """Load the n most recent reports from the trace log."""
    if db_path is None:
        db_path = _get_db_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT timestamp, status, coherence_score, step_count, violation_count
        FROM coherence_reports
        ORDER BY id DESC LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [
        {
            "timestamp": r[0],
            "status": r[1],
            "coherence_score": r[2],
            "step_count": r[3],
            "violation_count": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def check(
    steps: str | list[str],
    conclusion: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    save: bool = True,
    db_path: Optional[Path] = None,
) -> CoherenceReport:
    """
    Check the coherence of a chain-of-thought reasoning sequence.

    Args:
        steps:      Reasoning steps as a list of strings, or a single string
                    that will be parsed into steps automatically.
        conclusion: The final conclusion or answer the reasoning reaches.
        model:      Anthropic model for the coherence judge.
        api_key:    Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        save:       Whether to persist the report to the SQLite trace log.
        db_path:    Path to SQLite database (default: .cot-coherence.db).

    Returns:
        CoherenceReport with full coherence assessment.

    Raises:
        ImportError: If anthropic package not installed.
        ValueError:  If steps or conclusion are empty.
    """
    parsed = parse_steps(steps)
    conclusion = conclusion.strip()

    if not parsed:
        raise ValueError("steps cannot be empty")
    if not conclusion:
        raise ValueError("conclusion cannot be empty")

    if len(parsed) == 1 and len(parsed[0].text) < 10:
        return CoherenceReport(
            steps=parsed,
            conclusion=conclusion,
            status=CoherenceStatus.SKIP,
            coherence_score=0.0,
            incoherence_score=1.0,
            overall_confidence=0.0,
            violations=[],
            dimensions=[],
            summary="Input too short to evaluate.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used=model,
            step_count=len(parsed),
        )

    raw = _call_judge(parsed, conclusion, model, api_key)
    report = _parse_judge_response(raw, parsed, conclusion, model)

    if save:
        save_report(report, db_path)

    return report


class CoherenceChecker:
    """
    Reusable checker instance. Useful when running multiple checks with the
    same model / API key configuration.

    Example::

        checker = CoherenceChecker(model="claude-haiku-4-5-20251001")
        report = checker.check(steps=[...], conclusion="...")
        print(report.status)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        save: bool = True,
        db_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.save = save
        self.db_path = db_path

    def check(
        self,
        steps: str | list[str],
        conclusion: str,
    ) -> CoherenceReport:
        return check(
            steps=steps,
            conclusion=conclusion,
            model=self.model,
            api_key=self.api_key,
            save=self.save,
            db_path=self.db_path,
        )

    def batch_check(
        self,
        samples: list[dict],
    ) -> list[CoherenceReport]:
        """
        Check multiple CoT samples.

        Args:
            samples: List of dicts with keys "steps" and "conclusion".

        Returns:
            List of CoherenceReport in the same order.
        """
        results = []
        for s in samples:
            results.append(self.check(
                steps=s["steps"],
                conclusion=s["conclusion"],
            ))
        return results


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def coherence_check(
    threshold: float = 0.55,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    raise_on_fail: bool = False,
):
    """
    Decorator that verifies coherence of a function's CoT output.

    The decorated function must return either:
      - A dict with keys "steps" (list[str]) and "conclusion" (str)
      - A CoTResult namedtuple-like object with .steps and .conclusion

    If the coherence_score falls below ``threshold`` and ``raise_on_fail``
    is True, raises CoherenceError. Otherwise, the report is attached to the
    return value as ``._coherence_report``.

    Example::

        @coherence_check(threshold=0.7, raise_on_fail=True)
        def classify(text: str) -> dict:
            return {
                "steps": ["Step 1: ...", "Step 2: ..."],
                "conclusion": "The text is positive.",
            }
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)

            if isinstance(result, dict):
                steps = result.get("steps", [])
                conclusion = result.get("conclusion", "")
            elif hasattr(result, "steps") and hasattr(result, "conclusion"):
                steps = result.steps
                conclusion = result.conclusion
            else:
                return result  # Can't inspect — pass through

            report = check(
                steps=steps,
                conclusion=conclusion,
                model=model,
                api_key=api_key,
                save=True,
            )

            if raise_on_fail and report.coherence_score < threshold:
                raise CoherenceError(
                    f"CoT coherence score {report.coherence_score:.2f} "
                    f"below threshold {threshold}. "
                    f"Status: {report.status.value}. "
                    f"Violations: {len(report.violations)}.",
                    report=report,
                )

            if isinstance(result, dict):
                result["_coherence_report"] = report
            else:
                try:
                    object.__setattr__(result, "_coherence_report", report)
                except (AttributeError, TypeError):
                    pass

            return result

        return wrapper

    return decorator


class CoherenceError(Exception):
    """Raised by @coherence_check when coherence score is below threshold."""

    def __init__(self, message: str, report: CoherenceReport) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# YAML test suite runner
# ---------------------------------------------------------------------------

def run_yaml_suite(
    suite_path: str | Path,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    threshold: float = 0.55,
) -> dict:
    """
    Run a YAML coherence test suite.

    Expected YAML format::

        suite: "My CoT test suite"
        threshold: 0.7
        cases:
          - id: "case_001"
            steps:
              - "Step 1: ..."
              - "Step 2: ..."
            conclusion: "Therefore ..."
            expect: "COHERENT"  # COHERENT | DEGRADED | INCOHERENT

    Returns a dict with keys: total, passed, failed, results.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml not installed. Run: pip install pyyaml")

    suite_path = Path(suite_path)
    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    suite_threshold = suite.get("threshold", threshold)
    cases = suite.get("cases", [])
    results = []
    passed = 0
    failed = 0

    for case in cases:
        report = check(
            steps=case["steps"],
            conclusion=case["conclusion"],
            model=model,
            api_key=api_key,
            save=False,
        )
        expected = case.get("expect", "COHERENT")
        ok = report.status.value == expected
        if ok:
            passed += 1
        else:
            failed += 1
        results.append({
            "id": case.get("id", "?"),
            "passed": ok,
            "expected": expected,
            "actual": report.status.value,
            "coherence_score": report.coherence_score,
            "violations": len(report.violations),
        })

    return {
        "suite": suite.get("suite", suite_path.name),
        "total": len(cases),
        "passed": passed,
        "failed": failed,
        "threshold": suite_threshold,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cot-coherence",
        description=(
            "Verify that an LLM's chain-of-thought reasoning is internally coherent. "
            "CI gate: exits 1 if incoherence exceeds threshold."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- check command ---
    check_cmd = sub.add_parser("check", help="Run a coherence check")
    check_cmd.add_argument(
        "--steps", nargs="+", metavar="STEP",
        help="Reasoning steps (one per argument). Alternative: --file.",
    )
    check_cmd.add_argument(
        "--raw-steps", metavar="TEXT",
        help="Steps as a single block of text (auto-parsed).",
    )
    check_cmd.add_argument(
        "--conclusion", required=True,
        help="The final conclusion the reasoning reaches.",
    )
    check_cmd.add_argument("--model", default=DEFAULT_MODEL)
    check_cmd.add_argument(
        "--format", choices=["json", "markdown", "text"], default="text",
    )
    check_cmd.add_argument("--no-save", action="store_true")
    check_cmd.add_argument(
        "--threshold", type=float, default=0.45,
        help="Incoherence threshold — fail CI if incoherence_score exceeds this "
             "(default 0.45, i.e., coherence_score < 0.55).",
    )

    # --- suite command ---
    suite_cmd = sub.add_parser("suite", help="Run a YAML coherence test suite")
    suite_cmd.add_argument("suite_file", help="Path to YAML suite file")
    suite_cmd.add_argument("--model", default=DEFAULT_MODEL)
    suite_cmd.add_argument("--format", choices=["json", "text"], default="text")
    suite_cmd.add_argument("--threshold", type=float, default=0.45)

    # --- history command ---
    hist_cmd = sub.add_parser("history", help="Show recent coherence check history")
    hist_cmd.add_argument("--n", type=int, default=10)

    args = parser.parse_args(argv)

    if args.command == "check":
        if args.steps:
            steps_input = args.steps
        elif args.raw_steps:
            steps_input = args.raw_steps
        else:
            parser.error("Provide --steps or --raw-steps")
            return 2

        report = check(
            steps=steps_input,
            conclusion=args.conclusion,
            model=args.model,
            save=not args.no_save,
        )

        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        elif args.format == "markdown":
            print(report.to_markdown())
        else:
            status_icon = {
                CoherenceStatus.COHERENT: "[COHERENT]",
                CoherenceStatus.DEGRADED: "[DEGRADED]",
                CoherenceStatus.INCOHERENT: "[INCOHERENT]",
                CoherenceStatus.SKIP: "[SKIP]",
            }
            print(f"\ncot-coherence result: {status_icon[report.status]}")
            print(f"Coherence score:   {report.coherence_score:.2f} / 1.00")
            print(f"Incoherence score: {report.incoherence_score:.2f} (threshold: {args.threshold})")
            print(f"Confidence:        {report.overall_confidence:.0%}")
            print(f"Steps evaluated:   {report.step_count}")
            print(f"\nDimensions:")
            for d in report.dimensions:
                mark = "[+]" if d.passed else "[X]"
                print(f"  {mark} {d.name:<28} {d.score:.2f}  {d.notes[:60]}")
            if report.violations:
                print(f"\nViolations ({len(report.violations)}):")
                for v in report.violations:
                    crit = " [CRITICAL]" if v.is_critical() else ""
                    print(f"  {v.violation_type.value}{crit} (sev {v.severity:.2f})")
                    print(f"    {v.description[:100]}")
            print(f"\nSummary:\n  {report.summary}")

        if report.incoherence_score > args.threshold:
            print(
                f"\ncot-coherence: GATE FAILED — incoherence {report.incoherence_score:.2f} "
                f"exceeds threshold {args.threshold}"
            )
            return 1
        return 0

    elif args.command == "suite":
        result = run_yaml_suite(
            suite_path=args.suite_file,
            model=args.model,
            threshold=args.threshold,
        )
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"\nSuite: {result['suite']}")
            print(f"Results: {result['passed']}/{result['total']} passed, "
                  f"{result['failed']} failed")
            print()
            for r in result["results"]:
                mark = "[+]" if r["passed"] else "[X]"
                print(
                    f"  {mark} {r['id']:<20} "
                    f"coherence={r['coherence_score']:.2f}  "
                    f"expected={r['expected']}  actual={r['actual']}"
                )
        if result["failed"] > 0:
            print(f"\ncot-coherence suite: {result['failed']} case(s) failed")
            return 1
        return 0

    elif args.command == "history":
        rows = load_recent_reports(args.n)
        if not rows:
            print("No cot-coherence history found.")
            return 0
        print(
            f"{'Timestamp':<28} {'Status':<12} {'Score':>7} "
            f"{'Steps':>5} {'Violations':>10}"
        )
        print("-" * 72)
        for r in rows:
            print(
                f"{r['timestamp']:<28} {r['status']:<12} "
                f"{r['coherence_score']:>7.2f} {r['step_count']:>5} "
                f"{r['violation_count']:>10}"
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
