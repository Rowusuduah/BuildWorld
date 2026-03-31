"""
semantic-pass-k — Semantic consistency testing for AI agents.

Run your agent k times on the same prompt.
Measure whether the outputs are semantically equivalent.
Get a CI-gateable ConsistencyScore with criticality-tier thresholds.

Quick start:
    from semantic_pass_k import ConsistencyEngine

    engine = ConsistencyEngine()
    result = engine.evaluate(
        prompt="What is Ghana's GDP?",
        outputs=[
            "Ghana's GDP is approximately $50 billion.",
            "Ghana has a GDP of around $50B USD.",
            "The GDP of Ghana is about 50 billion dollars.",
        ],
        criticality="HIGH",
    )
    print(result.verdict)  # CONSISTENT
    print(result.consistency_score)  # 0.92

Author: BuildWorld
Pattern: PAT-062 — Numbers 23:19 — The Perfect Consistency Standard
"""
from .engine import ConsistencyEngine
from .runner import ConsistencyRunner
from .store import ConsistencyStore
from .models import (
    ConsistencyResult,
    ConsistencyReport,
    CriticalityLevel,
    ConsistencyVerdict,
    CRITICALITY_THRESHOLDS,
    get_threshold,
    score_to_verdict,
)
from .decorators import consistency_probe, ConsistencyError
from .pytest_plugin import assert_consistent

__version__ = "0.1.0"
__all__ = [
    "ConsistencyEngine",
    "ConsistencyRunner",
    "ConsistencyStore",
    "ConsistencyResult",
    "ConsistencyReport",
    "CriticalityLevel",
    "ConsistencyVerdict",
    "CRITICALITY_THRESHOLDS",
    "get_threshold",
    "score_to_verdict",
    "consistency_probe",
    "ConsistencyError",
    "assert_consistent",
]
