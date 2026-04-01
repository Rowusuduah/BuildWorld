"""context_trace.pytest_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~
pytest plugin providing the ctrace_tracer fixture.

Enable in your conftest.py::

    pytest_plugins = ["context_trace.pytest_plugin"]

Then define a ctrace_runner_fn fixture and use ctrace_tracer::

    @pytest.fixture
    def ctrace_runner_fn():
        return my_llm_runner  # Callable[[str], str]

    def test_rag_attribution(ctrace_tracer):
        output = my_rag_pipeline("What is X?")
        report = ctrace_tracer.trace(
            prompt=full_prompt,
            original_output=output,
            chunks={"doc1": doc1, "system": system_prompt},
        )
        assert report.top_score > 0.3, "At least one chunk should be causal"
        assert len(report.contributors_above(0.3)) >= 1
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import pytest

from context_trace.tracer import AttributionReport, ContextTracer, CostBudget


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("context-trace")
    group.addoption(
        "--ctrace-k",
        default=3,
        type=int,
        help="Number of masked runs per chunk (default: 3).",
    )
    group.addoption(
        "--ctrace-budget-calls",
        default=200,
        type=int,
        help="Max API calls per test (default: 200).",
    )


class CTraceRunner:
    """
    Fixture object returned by ctrace_tracer.

    Call .trace() to run attribution on a real LLM output.
    """

    def __init__(
        self,
        runner: Callable[[str], str],
        k: int,
        budget: CostBudget,
    ) -> None:
        self._runner = runner
        self._k = k
        self._budget = budget
        self._tracer: Optional[ContextTracer] = None

    def trace(
        self,
        prompt: str,
        original_output: str,
        chunks: Dict[str, str],
    ) -> AttributionReport:
        if self._tracer is None:
            self._tracer = ContextTracer(
                runner=self._runner,
                k=self._k,
                budget=self._budget,
            )
        return self._tracer.trace(
            prompt=prompt,
            original_output=original_output,
            chunks=chunks,
        )


@pytest.fixture
def ctrace_tracer(request: pytest.FixtureRequest) -> CTraceRunner:
    """
    CTraceRunner fixture. Requires ctrace_runner_fn fixture to be provided.

    CLI options:
        --ctrace-k N              Runs per chunk (default: 3)
        --ctrace-budget-calls N   Max API calls per test (default: 200)
    """
    runner_fn = request.getfixturevalue("ctrace_runner_fn")
    k = request.config.getoption("--ctrace-k", default=3)
    budget_calls = request.config.getoption("--ctrace-budget-calls", default=200)
    budget = CostBudget(max_api_calls=budget_calls)
    return CTraceRunner(runner=runner_fn, k=k, budget=budget)
