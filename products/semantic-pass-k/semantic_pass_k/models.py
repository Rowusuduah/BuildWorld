"""
semantic_pass_k.models
----------------------
Data models for semantic consistency measurement.

Inspired by Numbers 23:19 (PAT-062):
"Does he speak and then not act? Does he promise and then not fulfill?"
Consistency is not an internal property — it is verified empirically by
comparing outputs across runs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional


# ── Criticality Tiers ─────────────────────────────────────────────────────────

CriticalityLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
ConsistencyVerdict = Literal["CONSISTENT", "INCONSISTENT", "BORDERLINE"]

#: Minimum ConsistencyScore required to pass at each tier.
CRITICALITY_THRESHOLDS: Dict[str, float] = {
    "CRITICAL": 0.99,   # Medical, legal, financial outputs — near-perfect consistency
    "HIGH":     0.90,   # Production agents, customer-facing outputs
    "MEDIUM":   0.75,   # Internal tools, best-effort tasks
    "LOW":      0.60,   # Exploratory / brainstorming agents
}


def get_threshold(criticality: CriticalityLevel) -> float:
    """Return the ConsistencyScore threshold for the given criticality tier."""
    return CRITICALITY_THRESHOLDS[criticality]


def score_to_verdict(
    score: float,
    criticality: CriticalityLevel,
    borderline_band: float = 0.05,
) -> ConsistencyVerdict:
    """
    Map a ConsistencyScore to a verdict for a given criticality tier.

    Args:
        score: ConsistencyScore in [0.0, 1.0].
        criticality: One of CRITICAL / HIGH / MEDIUM / LOW.
        borderline_band: Score within this distance BELOW the threshold is
                         BORDERLINE rather than INCONSISTENT.

    Returns:
        "CONSISTENT" | "BORDERLINE" | "INCONSISTENT"
    """
    threshold = get_threshold(criticality)
    if score >= threshold:
        return "CONSISTENT"
    if score >= threshold - borderline_band:
        return "BORDERLINE"
    return "INCONSISTENT"


# ── Core Result Types ─────────────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    """
    Result of a single semantic-pass-k evaluation run.

    A "run" means: the same prompt was sent to an agent k times,
    all k outputs were embedded, pairwise cosine similarities were computed,
    and a ConsistencyScore (mean pairwise cosine) was derived.
    """
    run_id: str
    prompt: str
    outputs: List[str]
    k: int
    consistency_score: float          # mean pairwise cosine similarity [0, 1]
    pairwise_scores: List[float]      # all n*(n-1)/2 pairwise cosine values
    verdict: ConsistencyVerdict
    criticality: CriticalityLevel
    threshold: float
    borderline_band: float
    agent_label: str
    tested_at: datetime
    prompt_hash: str
    metadata: Dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "CONSISTENT"

    @property
    def n_pairs(self) -> int:
        return len(self.pairwise_scores)

    def summary(self) -> str:
        return (
            f"[{self.verdict}] score={self.consistency_score:.3f} "
            f"threshold={self.threshold:.2f} "
            f"k={self.k} criticality={self.criticality}"
        )


@dataclass
class ConsistencyReport:
    """
    Aggregate report across multiple ConsistencyResult objects.

    Useful for running consistency checks across multiple prompts and
    summarising the overall pass rate for a CI gate decision.
    """
    report_id: str
    label: str
    results: List[ConsistencyResult]
    criticality: CriticalityLevel
    threshold: float
    overall_score: float          # mean ConsistencyScore across all results
    pass_rate: float              # fraction of results that passed
    verdict: ConsistencyVerdict   # CONSISTENT if all pass, else INCONSISTENT
    total_results: int
    passed_results: int
    failed_results: int
    borderline_results: int
    generated_at: datetime

    @classmethod
    def from_results(
        cls,
        results: List[ConsistencyResult],
        label: str = "consistency_report",
        borderline_band: float = 0.05,
    ) -> "ConsistencyReport":
        """Build a report from a list of ConsistencyResult objects."""
        import uuid
        if not results:
            raise ValueError("Cannot build report from empty results list.")

        criticality = results[0].criticality
        threshold = results[0].threshold

        scores = [r.consistency_score for r in results]
        overall_score = sum(scores) / len(scores)

        passed = sum(1 for r in results if r.verdict == "CONSISTENT")
        borderline = sum(1 for r in results if r.verdict == "BORDERLINE")
        failed = sum(1 for r in results if r.verdict == "INCONSISTENT")
        pass_rate = passed / len(results)

        # Aggregate verdict: all must pass for CONSISTENT
        if passed == len(results):
            verdict: ConsistencyVerdict = "CONSISTENT"
        elif failed == 0:
            verdict = "BORDERLINE"
        else:
            verdict = "INCONSISTENT"

        return cls(
            report_id=str(uuid.uuid4()),
            label=label,
            results=results,
            criticality=criticality,
            threshold=threshold,
            overall_score=overall_score,
            pass_rate=pass_rate,
            verdict=verdict,
            total_results=len(results),
            passed_results=passed,
            failed_results=failed,
            borderline_results=borderline,
            generated_at=datetime.now(timezone.utc),
        )

    def summary(self) -> str:
        return (
            f"[{self.verdict}] overall_score={self.overall_score:.3f} "
            f"pass_rate={self.pass_rate:.1%} "
            f"({self.passed_results}/{self.total_results} passed) "
            f"criticality={self.criticality} threshold={self.threshold:.2f}"
        )

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "label": self.label,
            "criticality": self.criticality,
            "threshold": self.threshold,
            "overall_score": self.overall_score,
            "pass_rate": self.pass_rate,
            "verdict": self.verdict,
            "total_results": self.total_results,
            "passed_results": self.passed_results,
            "failed_results": self.failed_results,
            "borderline_results": self.borderline_results,
            "generated_at": self.generated_at.isoformat(),
            "results": [
                {
                    "run_id": r.run_id,
                    "prompt_hash": r.prompt_hash,
                    "agent_label": r.agent_label,
                    "k": r.k,
                    "consistency_score": r.consistency_score,
                    "verdict": r.verdict,
                    "tested_at": r.tested_at.isoformat(),
                }
                for r in self.results
            ],
        }
