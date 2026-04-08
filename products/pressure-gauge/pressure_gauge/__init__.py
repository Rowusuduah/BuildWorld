"""
pressure-gauge
--------------
Detect LLM behavioral drift caused by context window fill level.

The first pip-installable ContextPressureScore for AI agents.
Measures "context anxiety": agents that rush to completion, summarize
prematurely, or change behavior as their context window fills.

Quick start::

    from pressure_gauge import PressureGauge, PressureConfig

    gauge = PressureGauge(PressureConfig(
        model_context_limit=8192,
        criticality="HIGH",
    ))

    report = gauge.sweep(
        agent_fn=my_agent,          # Callable[[str], str]
        base_context="Analyze the document above.",
    )
    print(report.summary())
    assert report.gate_passed, f"Context pressure: {report.recommendation}"

Pattern source: PAT-078 — Daniel 5:5-6, 27 (The TEKEL Pressure Drift Pattern)
"""
from .models import (
    CriticalityLevel,
    DriftPoint,
    DriftVerdict,
    PressureConfig,
    PressureReport,
    PRESSURE_THRESHOLDS,
    get_threshold,
    score_to_verdict,
)
from .engine import (
    approx_token_count,
    build_padded_context,
    compute_similarities,
    cosine_similarity,
    generate_padding,
    run_sweep,
)
from .gauge import PressureGauge
from .decorator import pressure_probe, PressureError

__version__ = "0.1.0"

__all__ = [
    # Config & models
    "PressureConfig",
    "CriticalityLevel",
    "DriftVerdict",
    "DriftPoint",
    "PressureReport",
    "PRESSURE_THRESHOLDS",
    "get_threshold",
    "score_to_verdict",
    # Engine utilities
    "approx_token_count",
    "build_padded_context",
    "compute_similarities",
    "cosine_similarity",
    "generate_padding",
    "run_sweep",
    # High-level API
    "PressureGauge",
    # Decorator
    "pressure_probe",
    "PressureError",
    # Version
    "__version__",
]
