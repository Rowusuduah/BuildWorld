"""
livelock_probe.models
---------------------
Data models for AI agent livelock detection.

Inspired by PAT-075 (John 5:5-9 — The 38-Year Stuck State Pattern):
The man at the pool was active, trying, and failing — not idle, not erroring.
His state was structurally stuck: every attempt to enter the pool was beaten
by someone else. livelock-probe detects the same pattern in AI agents:
active, not erroring, making zero net progress toward the goal.

LivelockScore is a scalar in [0.0, 1.0]:
  0.0 = all steps progressing toward the goal
  1.0 = all steps stuck (zero net progress per step)

Criticality tiers (max LivelockScore to pass):
  CRITICAL: 0.05  (medical, legal, financial — near-zero tolerance for livelock)
  HIGH:     0.15  (production agents, customer-facing)
  MEDIUM:   0.30  (internal tools, best-effort tasks)
  LOW:      0.50  (exploratory / brainstorming agents)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Literal, Optional


# ── Criticality Tiers ─────────────────────────────────────────────────────────

CriticalityLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
LivelockVerdict = Literal["LIVELOCK_FREE", "LIVELOCK_DETECTED", "BORDERLINE"]

#: Maximum LivelockScore allowed before flagging livelock per tier.
LIVELOCK_THRESHOLDS: Dict[str, float] = {
    "CRITICAL": 0.05,
    "HIGH":     0.15,
    "MEDIUM":   0.30,
    "LOW":      0.50,
}

#: Borderline band: livelock_score within this distance ABOVE the threshold
#: is flagged as BORDERLINE rather than LIVELOCK_DETECTED.
DEFAULT_BORDERLINE_BAND: float = 0.05


def get_threshold(criticality: CriticalityLevel) -> float:
    """Return the maximum allowed LivelockScore for the given criticality tier."""
    return LIVELOCK_THRESHOLDS[criticality]


def score_to_verdict(
    livelock_score: float,
    criticality: CriticalityLevel,
    borderline_band: float = DEFAULT_BORDERLINE_BAND,
) -> LivelockVerdict:
    """
    Map a LivelockScore to a verdict for a given criticality tier.

    Args:
        livelock_score: LivelockScore in [0.0, 1.0].
        criticality: One of CRITICAL / HIGH / MEDIUM / LOW.
        borderline_band: Scores within this distance ABOVE the threshold
                         are BORDERLINE rather than LIVELOCK_DETECTED.

    Returns:
        "LIVELOCK_FREE" | "BORDERLINE" | "LIVELOCK_DETECTED"
    """
    threshold = get_threshold(criticality)
    if livelock_score <= threshold:
        return "LIVELOCK_FREE"
    if livelock_score <= threshold + borderline_band:
        return "BORDERLINE"
    return "LIVELOCK_DETECTED"


def make_recommendation(
    livelock_score: float,
    livelock_detected: bool,
    max_consecutive_stuck: int,
    k: int,
    stuck_window_start: Optional[int],
) -> str:
    """Generate a human-readable recommendation from livelock metrics."""
    if not livelock_detected:
        if livelock_score < 0.05:
            return "Agent is progressing normally. No livelock risk detected."
        return (
            f"Agent is mostly progressing (LivelockScore={livelock_score:.3f}). "
            f"Monitor for extended stuck windows if score rises."
        )
    loc = f" starting at step {stuck_window_start}" if stuck_window_start is not None else ""
    return (
        f"LIVELOCK DETECTED: Agent entered a structurally stuck state{loc}. "
        f"Longest consecutive stuck window: {max_consecutive_stuck} steps "
        f"(threshold k={k}). "
        f"LivelockScore={livelock_score:.3f}. "
        f"Recommend: inspect steps around index {stuck_window_start}; "
        f"consider goal reformulation, retrieval strategy change, or forced termination."
    )


# ── Core Data Classes ─────────────────────────────────────────────────────────

@dataclass
class ProgressConfig:
    """
    Configuration for livelock detection on an agent run.

    Args:
        goal: Natural language description of what the agent is trying to achieve.
              Used as the reference embedding to measure progress.
        k: Minimum number of consecutive stuck steps to trigger livelock detection.
           Default 5. Lower = more sensitive; higher = fewer false positives.
        epsilon: Progress threshold. A step is "stuck" if |progress_delta| < epsilon.
                 Default 0.05. Calibrate against your similarity backend.
        criticality: Task criticality tier. Controls max LivelockScore before flagging.
        budget_steps: Soft limit on agent steps. Not enforced by livelock-probe itself;
                      exposed for use by external orchestration layers.
        use_neural: Use sentence-transformers (all-MiniLM-L6-v2) for embeddings.
                    Falls back to TF-IDF if sentence-transformers is not installed.
        similarity_fn: Injectable pairwise similarity function(a, b) -> [0, 1].
                       Overrides both TF-IDF and neural. Useful for testing.
        borderline_band: Score range above threshold classified as BORDERLINE.
        agent_label: Human-readable label for the agent under test.
    """
    goal: str
    k: int = 5
    epsilon: float = 0.05
    criticality: CriticalityLevel = "HIGH"
    budget_steps: int = 100
    use_neural: bool = False
    similarity_fn: Optional[Callable[[str, str], float]] = None
    borderline_band: float = DEFAULT_BORDERLINE_BAND
    agent_label: str = "default"

    def __post_init__(self) -> None:
        if not self.goal or not self.goal.strip():
            raise ValueError("ProgressConfig.goal must be a non-empty string.")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}.")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}.")
        if self.budget_steps < 1:
            raise ValueError(f"budget_steps must be >= 1, got {self.budget_steps}.")
        if self.borderline_band < 0.0:
            raise ValueError(f"borderline_band must be >= 0, got {self.borderline_band}.")


@dataclass
class StepRecord:
    """
    Record for a single agent step.

    Attributes:
        step_id: Zero-based step index.
        output: The string output produced at this step.
        progress_to_goal: Cosine similarity between step output and the goal.
                          Ranges [0, 1]; higher = closer to goal.
        progress_delta: Change in progress_to_goal from previous step.
                        Positive = advancing; near-zero = stuck; negative = regressing.
                        Step 0 uses progress_to_goal as the baseline delta.
        is_stuck: True if |progress_delta| < epsilon (this step counts as stuck).
    """
    step_id: int
    output: str
    progress_to_goal: float
    progress_delta: float
    is_stuck: bool


@dataclass
class LivelockReport:
    """
    Full livelock detection report for a single agent run.

    Attributes:
        report_id: Unique identifier for this report.
        goal: The goal description used for progress measurement.
        livelock_score: Fraction of steps that are stuck [0.0, 1.0].
                        0.0 = no stuck steps; 1.0 = all steps stuck.
        livelock_detected: True if max_consecutive_stuck >= k.
        stuck_window_start: Step index where the longest stuck window begins.
                            None if no stuck window found.
        stuck_window_end: Step index where the longest stuck window ends.
                          None if no stuck window or window is ongoing.
        total_steps: Total number of steps recorded.
        progress_vector: Per-step progress_to_goal values.
        progress_deltas: Per-step progress delta values.
        mean_progress: Mean progress delta across all steps.
        max_consecutive_stuck: Longest run of consecutive stuck steps.
        gate_passed: True if livelock_score <= criticality threshold.
        verdict: "LIVELOCK_FREE" | "BORDERLINE" | "LIVELOCK_DETECTED"
        steps: Full per-step trace.
        recommendation: Human-readable summary and next-step guidance.
        criticality: Criticality tier used for gate evaluation.
        threshold: Max LivelockScore threshold for this criticality tier.
        k: Minimum consecutive stuck steps configuration.
        epsilon: Progress threshold configuration.
        agent_label: Label for the agent under test.
        tested_at: UTC timestamp when compute() was called.
        metadata: Optional extra metadata dict.
    """
    report_id: str
    goal: str
    livelock_score: float
    livelock_detected: bool
    stuck_window_start: Optional[int]
    stuck_window_end: Optional[int]
    total_steps: int
    progress_vector: List[float]
    progress_deltas: List[float]
    mean_progress: float
    max_consecutive_stuck: int
    gate_passed: bool
    verdict: LivelockVerdict
    steps: List[StepRecord]
    recommendation: str
    criticality: CriticalityLevel
    threshold: float
    k: int
    epsilon: float
    agent_label: str
    tested_at: datetime
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.verdict}] livelock_score={self.livelock_score:.3f} "
            f"threshold={self.threshold:.2f} steps={self.total_steps} "
            f"max_consecutive_stuck={self.max_consecutive_stuck}/{self.k} "
            f"criticality={self.criticality}"
        )

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "goal": self.goal,
            "livelock_score": self.livelock_score,
            "livelock_detected": self.livelock_detected,
            "verdict": self.verdict,
            "gate_passed": self.gate_passed,
            "stuck_window_start": self.stuck_window_start,
            "stuck_window_end": self.stuck_window_end,
            "total_steps": self.total_steps,
            "progress_vector": self.progress_vector,
            "progress_deltas": self.progress_deltas,
            "mean_progress": self.mean_progress,
            "max_consecutive_stuck": self.max_consecutive_stuck,
            "criticality": self.criticality,
            "threshold": self.threshold,
            "k": self.k,
            "epsilon": self.epsilon,
            "agent_label": self.agent_label,
            "recommendation": self.recommendation,
            "tested_at": self.tested_at.isoformat(),
            "metadata": self.metadata,
        }
