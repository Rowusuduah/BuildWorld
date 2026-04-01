"""
cot_fidelity.models
-------------------
Core data structures for CoT faithfulness measurement.

Biblical Foundation (internal): PAT-059 (Genesis 3:1-6) — Eve's stated reasoning
chain (v.3) was demonstrably non-causal relative to her actual decision (v.6).
The counterfactual test: suppress the stated chain and observe whether output changes.
If identical → unfaithful. PAT-059 scored 10.0/10 — first perfect pattern in BibleWorld.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional


FidelityVerdict = Literal["FAITHFUL", "UNFAITHFUL", "INCONCLUSIVE"]


@dataclass
class FidelityResult:
    """
    Output from a single counterfactual suppression test.

    faithfulness_score = 1 - cosine_similarity(full_output, suppressed_output)
    High score  → CoT WAS causal (outputs differ) → FAITHFUL
    Low score   → CoT was NOT causal (outputs identical) → UNFAITHFUL
    """
    prompt: str
    full_output: str               # output with reasoning chain present
    suppressed_output: str         # output with reasoning chain stripped
    cot_chain: str                 # the reasoning chain that was tested
    similarity: float              # raw cosine similarity (0.0 – 1.0)
    faithfulness_score: float      # 1 - similarity
    verdict: FidelityVerdict       # FAITHFUL / UNFAITHFUL / INCONCLUSIVE
    faithful_threshold: float      # threshold for FAITHFUL
    unfaithful_threshold: float    # threshold for UNFAITHFUL
    runs: int = 1                  # number of suppressed runs averaged
    prompt_hash: str = ""          # sha256[:16] of prompt
    tested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "prompt_hash": self.prompt_hash,
            "similarity": round(self.similarity, 6),
            "faithfulness_score": round(self.faithfulness_score, 6),
            "verdict": self.verdict,
            "faithful_threshold": self.faithful_threshold,
            "unfaithful_threshold": self.unfaithful_threshold,
            "runs": self.runs,
            "tested_at": self.tested_at.isoformat(),
            "cot_chain_length": len(self.cot_chain),
            "full_output_length": len(self.full_output),
            "suppressed_output_length": len(self.suppressed_output),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        verdict_icon = {"FAITHFUL": "✅", "UNFAITHFUL": "❌", "INCONCLUSIVE": "⚠️"}[self.verdict]
        lines = [
            f"# FidelityResult",
            f"",
            f"**Verdict:** {verdict_icon} {self.verdict}",
            f"**FaithfulnessScore:** {self.faithfulness_score:.4f}",
            f"**Similarity (raw):** {self.similarity:.4f}",
            f"**Thresholds:** FAITHFUL ≥ {self.faithful_threshold:.2f} | UNFAITHFUL < {self.unfaithful_threshold:.2f}",
            f"**Suppressed runs:** {self.runs}",
            f"**Tested at:** {self.tested_at.isoformat()}",
            f"",
            f"## CoT Chain (first 300 chars)",
            f"```",
            f"{self.cot_chain[:300]}{'...' if len(self.cot_chain) > 300 else ''}",
            f"```",
            f"",
            f"## Full Output (first 300 chars)",
            f"```",
            f"{self.full_output[:300]}{'...' if len(self.full_output) > 300 else ''}",
            f"```",
            f"",
            f"## Suppressed Output (first 300 chars)",
            f"```",
            f"{self.suppressed_output[:300]}{'...' if len(self.suppressed_output) > 300 else ''}",
            f"```",
        ]
        return "\n".join(lines)

    @property
    def is_faithful(self) -> bool:
        return self.verdict == "FAITHFUL"

    @property
    def is_unfaithful(self) -> bool:
        return self.verdict == "UNFAITHFUL"

    @property
    def is_inconclusive(self) -> bool:
        return self.verdict == "INCONCLUSIVE"


@dataclass
class FidelityBatchReport:
    """
    Aggregate report across multiple FidelityResult runs.
    """
    results: List[FidelityResult]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def faithful_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "FAITHFUL")

    @property
    def unfaithful_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "UNFAITHFUL")

    @property
    def inconclusive_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == "INCONCLUSIVE")

    @property
    def faithfulness_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.faithful_count / self.total

    @property
    def unfaithfulness_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.unfaithful_count / self.total

    @property
    def mean_faithfulness_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.faithfulness_score for r in self.results) / self.total

    @property
    def mean_similarity(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.similarity for r in self.results) / self.total

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "faithful_count": self.faithful_count,
            "unfaithful_count": self.unfaithful_count,
            "inconclusive_count": self.inconclusive_count,
            "faithfulness_rate": round(self.faithfulness_rate, 4),
            "unfaithfulness_rate": round(self.unfaithfulness_rate, 4),
            "mean_faithfulness_score": round(self.mean_faithfulness_score, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "generated_at": self.generated_at.isoformat(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# FidelityBatchReport",
            f"",
            f"**Total tests:** {self.total}",
            f"**Faithful:** {self.faithful_count} ({self.faithfulness_rate:.1%})",
            f"**Unfaithful:** {self.unfaithful_count} ({self.unfaithfulness_rate:.1%})",
            f"**Inconclusive:** {self.inconclusive_count}",
            f"**Mean faithfulness score:** {self.mean_faithfulness_score:.4f}",
            f"**Mean similarity:** {self.mean_similarity:.4f}",
            f"**Generated:** {self.generated_at.isoformat()}",
            f"",
            f"## Per-Result Summary",
            f"",
            f"| # | Verdict | Score | Similarity |",
            f"|---|---------|-------|------------|",
        ]
        for i, r in enumerate(self.results, 1):
            icon = {"FAITHFUL": "✅", "UNFAITHFUL": "❌", "INCONCLUSIVE": "⚠️"}[r.verdict]
            lines.append(
                f"| {i} | {icon} {r.verdict} | {r.faithfulness_score:.4f} | {r.similarity:.4f} |"
            )
        return "\n".join(lines)


@dataclass
class DriftPoint:
    """A single data point in a faithfulness drift log."""
    run_id: str
    prompt_hash: str
    faithfulness_score: float
    verdict: FidelityVerdict
    model_version: str
    recorded_at: datetime


@dataclass
class DriftReport:
    """Rolling-window faithfulness drift analysis."""
    points: List[DriftPoint]
    window: int
    mean_score: float
    std_score: float
    drift_detected: bool
    trend: Literal["STABLE", "DEGRADING", "IMPROVING", "INSUFFICIENT_DATA"]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "points": len(self.points),
            "mean_score": round(self.mean_score, 4),
            "std_score": round(self.std_score, 4),
            "drift_detected": self.drift_detected,
            "trend": self.trend,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        icon = "🔴" if self.drift_detected else "🟢"
        return (
            f"# DriftReport\n\n"
            f"**Status:** {icon} {'DRIFT DETECTED' if self.drift_detected else 'STABLE'}\n"
            f"**Trend:** {self.trend}\n"
            f"**Window:** last {self.window} runs\n"
            f"**Mean faithfulness:** {self.mean_score:.4f} ± {self.std_score:.4f}\n"
        )
