"""
semantic_pass_k.pytest_plugin
------------------------------
pytest plugin for semantic-pass-k.

Usage:
    # conftest.py
    from semantic_pass_k.pytest_plugin import assert_consistent

    def test_agent_consistency(my_agent):
        outputs = [my_agent("Summarise the 2024 budget.") for _ in range(5)]
        assert_consistent(outputs, criticality="HIGH", agent_label="summariser")

Fixtures provided:
    consistency_engine  -- a default ConsistencyEngine instance
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .engine import ConsistencyEngine
from .models import CriticalityLevel, ConsistencyResult


def assert_consistent(
    outputs: List[str],
    prompt: str = "<test_prompt>",
    criticality: CriticalityLevel = "HIGH",
    agent_label: str = "pytest_agent",
    engine: Optional[ConsistencyEngine] = None,
    borderline_passes: bool = True,
) -> ConsistencyResult:
    """
    Assert that a list of outputs is semantically consistent.

    Args:
        outputs: List of k outputs to check (must have >= 2).
        prompt: The prompt that produced the outputs (for logging).
        criticality: Criticality tier for threshold selection.
        agent_label: Label for logging.
        engine: Optional ConsistencyEngine (default TF-IDF engine used if None).
        borderline_passes: If True, BORDERLINE counts as passing (default True).

    Returns:
        ConsistencyResult (for inspection if needed)

    Raises:
        AssertionError: If verdict is INCONSISTENT (or BORDERLINE when borderline_passes=False).
    """
    _engine = engine or ConsistencyEngine(agent_label=agent_label)
    result = _engine.evaluate(prompt, outputs, criticality)

    fails = {"INCONSISTENT"}
    if not borderline_passes:
        fails.add("BORDERLINE")

    assert result.verdict not in fails, (
        f"Consistency check FAILED for agent '{agent_label}':\n"
        f"  verdict: {result.verdict}\n"
        f"  score:   {result.consistency_score:.3f}\n"
        f"  threshold: {result.threshold:.2f} (criticality={criticality})\n"
        f"  k: {result.k} outputs\n"
        f"  pairwise min: {min(result.pairwise_scores):.3f}\n"
        f"  pairwise max: {max(result.pairwise_scores):.3f}"
    )
    return result


# ── pytest fixtures (optional) ────────────────────────────────────────────────

try:
    import pytest

    @pytest.fixture
    def consistency_engine() -> ConsistencyEngine:
        """Provide a default ConsistencyEngine for pytest tests."""
        return ConsistencyEngine()

    @pytest.fixture
    def assert_agent_consistent():
        """
        Fixture that returns assert_consistent for use in tests.

        Example:
            def test_my_agent(assert_agent_consistent):
                outputs = ["A", "A", "B"]
                assert_agent_consistent(outputs, criticality="LOW")
        """
        return assert_consistent

except ImportError:
    pass
