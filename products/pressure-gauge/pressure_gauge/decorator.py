"""
pressure-gauge: @pressure_probe decorator
------------------------------------------
Wraps an agent function and attaches a .pressure_sweep() method.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional

from .gauge import PressureGauge
from .models import CriticalityLevel, PressureConfig, PressureReport


class PressureError(Exception):
    """Raised when gate_passed is False and raise_on_fail=True."""

    def __init__(self, report: PressureReport) -> None:
        self.report = report
        super().__init__(
            f"Context pressure gate FAILED: {report.recommendation}"
        )


def pressure_probe(
    config: Optional[PressureConfig] = None,
    *,
    model_context_limit: int = 8192,
    fill_levels: Optional[List[float]] = None,
    stability_threshold: float = 0.85,
    criticality: CriticalityLevel = CriticalityLevel.HIGH,
    padding_strategy: str = "lorem_ipsum",
    base_context: str = "",
    raise_on_fail: bool = True,
    use_neural: bool = False,
) -> Callable:
    """
    Decorator that attaches a .pressure_sweep() method to an agent function.

    The decorated function behaves normally when called. Call
    ``my_agent.pressure_sweep(base_context="task")`` to run a sweep.

    Usage::

        @pressure_probe(model_context_limit=8192, criticality="HIGH")
        def my_agent(context: str) -> str:
            return llm.complete(context)

        report = my_agent.pressure_sweep(base_context="Summarize the document.")
        assert report.gate_passed, report.recommendation
    """

    def decorator(agent_fn: Callable[[str], str]) -> Callable:
        gauge = PressureGauge(
            config=config,
            model_context_limit=model_context_limit,
            fill_levels=fill_levels or [0.1, 0.3, 0.5, 0.7, 0.9],
            stability_threshold=stability_threshold,
            criticality=criticality,
            padding_strategy=padding_strategy,
            use_neural=use_neural,
        )

        @functools.wraps(agent_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return agent_fn(*args, **kwargs)

        def pressure_sweep(ctx: str = base_context) -> PressureReport:
            report = gauge.sweep(agent_fn=agent_fn, base_context=ctx)
            if raise_on_fail and not report.gate_passed:
                raise PressureError(report)
            return report

        wrapper.pressure_sweep = pressure_sweep  # type: ignore[attr-defined]
        wrapper._pressure_gauge = gauge  # type: ignore[attr-defined]
        return wrapper

    return decorator
