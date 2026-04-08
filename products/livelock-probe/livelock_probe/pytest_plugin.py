"""
livelock_probe.pytest_plugin
-----------------------------
pytest plugin providing the `livelock_suite` fixture.

Auto-loaded via entry point "pytest11" = "livelock_probe = livelock_probe.pytest_plugin".

Usage in conftest.py or tests:

    def test_agent_no_livelock(livelock_suite):
        suite = livelock_suite(
            goal="resolve customer support ticket",
            k=5,
            criticality="HIGH",
        )
        # Simulate agent steps
        steps = [
            "I am reading the ticket.",
            "I found the error in the database config.",
            "I am applying the fix.",
            "Fix applied. Verifying.",
            "Verification passed.",
        ]
        suite.record_steps(steps)
        report = suite.compute()
        assert report.gate_passed, (
            f"Agent livelock detected: LivelockScore={report.livelock_score:.3f}, "
            f"stuck from step {report.stuck_window_start}"
        )
"""
from __future__ import annotations

from typing import Optional

import pytest

from .models import CriticalityLevel, ProgressConfig
from .suite import LivelockSuite


@pytest.fixture
def livelock_suite():
    """
    pytest fixture that returns a factory for creating LivelockSuite instances.

    Usage:
        def test_my_agent(livelock_suite):
            suite = livelock_suite(goal="accomplish X", k=3, criticality="HIGH")
            suite.record_step("step 1 output")
            suite.record_step("step 2 output")
            report = suite.compute()
            assert report.gate_passed
    """
    created: list = []

    def factory(
        goal: str,
        k: int = 5,
        epsilon: float = 0.05,
        criticality: CriticalityLevel = "HIGH",
        budget_steps: int = 100,
        use_neural: bool = False,
        agent_label: str = "pytest_agent",
        borderline_band: float = 0.05,
    ) -> LivelockSuite:
        config = ProgressConfig(
            goal=goal,
            k=k,
            epsilon=epsilon,
            criticality=criticality,
            budget_steps=budget_steps,
            use_neural=use_neural,
            agent_label=agent_label,
            borderline_band=borderline_band,
        )
        suite = LivelockSuite(config)
        created.append(suite)
        return suite

    yield factory

    # Teardown: nothing to clean up — suites are in-memory only
