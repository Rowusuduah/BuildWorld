"""
pressure-gauge pytest plugin
-----------------------------
Auto-loaded via pytest11 entry point.
Provides the ``pressure_gauge_suite`` fixture.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import pytest

from .gauge import PressureGauge
from .models import CriticalityLevel, PressureConfig, PressureReport


class PressureGaugeSuite:
    """
    Pytest fixture wrapper for context pressure testing.

    Usage in tests::

        def test_agent_is_stable(pressure_gauge_suite):
            pressure_gauge_suite.configure(
                model_context_limit=8192,
                criticality="HIGH",
            )
            report = pressure_gauge_suite.sweep(my_agent, base_context="task")
            assert report.gate_passed, report.recommendation
    """

    def __init__(self) -> None:
        self._config: Optional[PressureConfig] = None
        self._use_neural: bool = False

    def configure(
        self,
        model_context_limit: int = 8192,
        fill_levels: Optional[List[float]] = None,
        stability_threshold: float = 0.85,
        criticality: CriticalityLevel = CriticalityLevel.HIGH,
        padding_strategy: str = "lorem_ipsum",
        use_neural: bool = False,
    ) -> "PressureGaugeSuite":
        kwargs: dict = {
            "model_context_limit": model_context_limit,
            "stability_threshold": stability_threshold,
            "criticality": criticality,
            "padding_strategy": padding_strategy,
        }
        if fill_levels is not None:
            kwargs["fill_levels"] = fill_levels
        self._config = PressureConfig(**kwargs)
        self._use_neural = use_neural
        return self

    def sweep(
        self,
        agent_fn: Callable[[str], str],
        base_context: str = "",
    ) -> PressureReport:
        gauge = PressureGauge(config=self._config, use_neural=self._use_neural)
        return gauge.sweep(agent_fn=agent_fn, base_context=base_context)

    def quick(
        self,
        agent_fn: Callable[[str], str],
        base_context: str = "",
    ) -> PressureReport:
        gauge = PressureGauge(config=self._config, use_neural=self._use_neural)
        return gauge.quick(agent_fn=agent_fn, base_context=base_context)


@pytest.fixture
def pressure_gauge_suite() -> PressureGaugeSuite:
    """Pytest fixture: returns a fresh PressureGaugeSuite instance."""
    return PressureGaugeSuite()
