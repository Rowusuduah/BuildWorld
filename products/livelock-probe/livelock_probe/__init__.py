"""
livelock-probe
--------------
Detect when an AI agent is making zero net progress toward its goal.

The only tool that measures structural livelock in AI agents:
active, not erroring, going nowhere.

Quick start:
    from livelock_probe import LivelockSuite, ProgressConfig

    suite = LivelockSuite(ProgressConfig(
        goal="resolve the database connection error",
        k=5,
        criticality="HIGH",
    ))

    for output in agent_outputs:
        suite.record_step(output)

    report = suite.compute()
    print(report.summary())
    assert report.gate_passed, f"Livelock: {report.recommendation}"

Pattern source: PAT-075 — John 5:5-9 (The 38-Year Stuck State)
"""
from .models import (
    CriticalityLevel,
    LivelockVerdict,
    LivelockReport,
    ProgressConfig,
    StepRecord,
    LIVELOCK_THRESHOLDS,
    get_threshold,
    score_to_verdict,
)
from .engine import LivelockEngine
from .suite import LivelockSuite
from .decorator import livelock_probe_decorator, LivelockError

__version__ = "0.1.0"
__all__ = [
    # Config
    "ProgressConfig",
    "CriticalityLevel",
    "LIVELOCK_THRESHOLDS",
    "get_threshold",
    "score_to_verdict",
    # Data models
    "LivelockReport",
    "LivelockVerdict",
    "StepRecord",
    # Core
    "LivelockEngine",
    "LivelockSuite",
    # Decorator
    "livelock_probe_decorator",
    "LivelockError",
    # Version
    "__version__",
]
