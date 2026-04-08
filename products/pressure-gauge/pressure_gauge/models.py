"""
pressure-gauge models
---------------------
Data models for ContextPressureScore and behavioral drift detection.

Pattern source: PAT-078 — Daniel 5:5-6, 27 (The TEKEL Pressure Drift Pattern)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CriticalityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DriftVerdict(str, Enum):
    STABLE = "STABLE"       # No significant drift detected
    MILD = "MILD"           # Minor drift, within acceptable bounds
    MODERATE = "MODERATE"   # Noticeable drift, approaching threshold
    SEVERE = "SEVERE"       # Significant drift; gate fail


# Minimum ContextPressureScore to PASS the gate
# (higher score = more stable = less drift)
PRESSURE_THRESHOLDS: Dict[CriticalityLevel, float] = {
    CriticalityLevel.CRITICAL: 0.95,
    CriticalityLevel.HIGH: 0.85,
    CriticalityLevel.MEDIUM: 0.75,
    CriticalityLevel.LOW: 0.65,
}


def get_threshold(criticality: CriticalityLevel) -> float:
    return PRESSURE_THRESHOLDS[criticality]


def score_to_verdict(score: float, criticality: CriticalityLevel) -> DriftVerdict:
    threshold = get_threshold(criticality)
    if score >= threshold:
        return DriftVerdict.STABLE
    elif score >= threshold - 0.10:
        return DriftVerdict.MILD
    elif score >= threshold - 0.20:
        return DriftVerdict.MODERATE
    else:
        return DriftVerdict.SEVERE


@dataclass
class PressureConfig:
    """Configuration for a pressure sweep run."""

    # Token budget of the target model
    model_context_limit: int = 8192

    # Fill levels to sweep (fraction of context window)
    fill_levels: List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9]
    )

    # Minimum similarity to baseline to pass gate
    stability_threshold: float = 0.85

    # Criticality tier
    criticality: CriticalityLevel = CriticalityLevel.HIGH

    # Padding strategy: "lorem_ipsum", "repeat_text", "inject_history"
    padding_strategy: str = "lorem_ipsum"

    # Custom padding text (used when padding_strategy="repeat_text")
    padding_text: Optional[str] = None

    # Approximate chars per token for padding size estimation
    chars_per_token: float = 4.0

    # Number of times to run agent at each fill level (for averaging)
    runs_per_level: int = 1

    def __post_init__(self) -> None:
        if not self.fill_levels:
            raise ValueError("fill_levels must not be empty")
        if not all(0.0 < fl <= 1.0 for fl in self.fill_levels):
            raise ValueError("All fill_levels must be in (0, 1]")
        if self.model_context_limit <= 0:
            raise ValueError("model_context_limit must be positive")
        if not 0.0 < self.stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be in (0, 1]")
        valid_strategies = ("lorem_ipsum", "repeat_text", "inject_history")
        if self.padding_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown padding_strategy: {self.padding_strategy!r}. "
                f"Valid: {valid_strategies}"
            )
        if self.runs_per_level < 1:
            raise ValueError("runs_per_level must be >= 1")
        # Normalize: sort and deduplicate
        self.fill_levels = sorted(set(self.fill_levels))

    @property
    def baseline_fill_level(self) -> float:
        return self.fill_levels[0]

    def tokens_for_level(self, fill_level: float) -> int:
        return int(fill_level * self.model_context_limit)


@dataclass
class DriftPoint:
    """A single data point in the ContextDriftCurve."""

    fill_level: float
    token_count: int
    similarity_to_baseline: float
    verdict: DriftVerdict
    outputs: List[str] = field(default_factory=list)


@dataclass
class PressureReport:
    """Result of a PressureGauge.sweep() run."""

    config: PressureConfig
    drift_curve: List[DriftPoint]

    # Core metric: mean similarity across all non-baseline fill levels
    context_pressure_score: float

    # Token count where similarity first dropped below stability_threshold
    # None if onset not detected
    pressure_onset_token: Optional[int]

    # Overall verdict
    verdict: DriftVerdict

    # CI gate: True = stable, False = drifting
    gate_passed: bool

    # Human-readable recommendation
    recommendation: str

    def summary(self) -> str:
        lines = [
            f"ContextPressureScore: {self.context_pressure_score:.4f}",
            f"Verdict: {self.verdict.value}",
            f"Gate: {'PASSED' if self.gate_passed else 'FAILED'}",
        ]
        if self.pressure_onset_token is not None:
            lines.append(f"Pressure onset: ~{self.pressure_onset_token:,} tokens")
        else:
            lines.append("Pressure onset: not detected (stable across all fill levels)")
        lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "context_pressure_score": self.context_pressure_score,
            "pressure_onset_token": self.pressure_onset_token,
            "verdict": self.verdict.value,
            "gate_passed": self.gate_passed,
            "recommendation": self.recommendation,
            "criticality": self.config.criticality.value,
            "model_context_limit": self.config.model_context_limit,
            "drift_curve": [
                {
                    "fill_level": dp.fill_level,
                    "token_count": dp.token_count,
                    "similarity_to_baseline": dp.similarity_to_baseline,
                    "verdict": dp.verdict.value,
                }
                for dp in self.drift_curve
            ],
        }
