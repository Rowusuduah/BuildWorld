"""context_trace.gate
~~~~~~~~~~~~~~~~~~
AttributionGate: CI-gateable thresholds for AttributionReports.

Usage::

    gate = AttributionGate(
        max_single_chunk_score=0.90,
        min_chunks_contributing=2,
    )
    gate.check(report)  # raises AttributionGateFailure if violated

    # Non-raising variants
    passed = gate.passed(report)            # bool
    ok, violations = gate.result(report)    # (bool, List[str])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from context_trace.tracer import AttributionReport


class AttributionGateFailure(Exception):
    """Raised when an AttributionReport fails a gate check."""

    def __init__(self, message: str, violations: List[str]) -> None:
        super().__init__(message)
        self.violations = violations


@dataclass
class AttributionGate:
    """
    CI gate for attribution reports.

    Attributes:
        max_single_chunk_score: Fail if any chunk's score exceeds this.
        min_chunks_contributing: Fail if fewer than N chunks score >= contributing_threshold.
        contributing_threshold: Score threshold for counting a chunk as "contributing".
        min_top_contributor_score: Fail if the top contributor's score is below this.
        max_total_api_calls: Fail if the report used more than this many API calls.
    """

    max_single_chunk_score: Optional[float] = None
    min_chunks_contributing: Optional[int] = None
    contributing_threshold: float = 0.30
    min_top_contributor_score: Optional[float] = None
    max_total_api_calls: Optional[int] = None

    def check(self, report: AttributionReport) -> None:
        """
        Validate report against all configured thresholds.

        Raises:
            AttributionGateFailure: if any threshold is violated.
        """
        violations: List[str] = []

        if self.max_single_chunk_score is not None:
            for name, score in report.chunk_scores.items():
                if score.attribution_score > self.max_single_chunk_score:
                    violations.append(
                        f"Chunk '{name}' attribution_score {score.attribution_score:.3f} "
                        f"exceeds max_single_chunk_score={self.max_single_chunk_score:.3f}"
                    )

        if self.min_chunks_contributing is not None:
            contributing = [
                name
                for name, score in report.chunk_scores.items()
                if score.attribution_score >= self.contributing_threshold
            ]
            if len(contributing) < self.min_chunks_contributing:
                violations.append(
                    f"Only {len(contributing)} chunk(s) score >= {self.contributing_threshold}; "
                    f"min_chunks_contributing={self.min_chunks_contributing} required. "
                    f"Contributing: {contributing or 'none'}"
                )

        if self.min_top_contributor_score is not None:
            if report.top_score < self.min_top_contributor_score:
                violations.append(
                    f"Top contributor score {report.top_score:.3f} is below "
                    f"min_top_contributor_score={self.min_top_contributor_score:.3f}"
                )

        if self.max_total_api_calls is not None:
            if report.total_api_calls > self.max_total_api_calls:
                violations.append(
                    f"Report used {report.total_api_calls} API calls; "
                    f"max_total_api_calls={self.max_total_api_calls}"
                )

        if violations:
            summary = f"AttributionGate failed ({len(violations)} violation(s))"
            raise AttributionGateFailure(summary, violations)

    def passed(self, report: AttributionReport) -> bool:
        """Return True if report passes all gate checks."""
        try:
            self.check(report)
            return True
        except AttributionGateFailure:
            return False

    def result(self, report: AttributionReport) -> Tuple[bool, List[str]]:
        """Return (passed: bool, violations: List[str]) tuple."""
        try:
            self.check(report)
            return True, []
        except AttributionGateFailure as e:
            return False, e.violations
