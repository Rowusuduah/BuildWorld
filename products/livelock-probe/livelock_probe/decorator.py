"""
livelock_probe.decorator
------------------------
@livelock_probe decorator for automatic agent run instrumentation.

Usage:
    from livelock_probe import livelock_probe_decorator

    @livelock_probe_decorator(
        goal="resolve the customer support ticket",
        k=5,
        criticality="HIGH",
    )
    def my_agent(ticket: dict) -> str:
        # Each call returns one step output
        ...

The decorated function is called repeatedly. Each return value is recorded
as a step. After all calls complete, the livelock report is attached to
the result as `result._livelock_report`.

For step-by-step agents (generators), use LivelockSuite directly.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional

from .models import CriticalityLevel, ProgressConfig
from .suite import LivelockSuite


def livelock_probe_decorator(
    goal: str,
    k: int = 5,
    epsilon: float = 0.05,
    criticality: CriticalityLevel = "HIGH",
    budget_steps: int = 100,
    use_neural: bool = False,
    agent_label: Optional[str] = None,
    raise_on_livelock: bool = False,
) -> Callable:
    """
    Decorator that wraps a function returning step outputs and computes livelock.

    The decorated function should represent a SINGLE STEP of an agent run.
    Call it repeatedly (in a loop) to simulate agent execution.
    After each call, the step output is recorded. Access the final report
    via `suite = func._livelock_suite; report = suite.compute()`.

    Args:
        goal: Natural language description of the agent's objective.
        k: Minimum consecutive stuck steps to trigger livelock detection.
        epsilon: Progress threshold (|delta| < epsilon = stuck).
        criticality: Criticality tier for gate threshold.
        budget_steps: Soft limit on total steps.
        use_neural: Use sentence-transformers for embeddings.
        agent_label: Label for the agent. Defaults to the function name.
        raise_on_livelock: If True, raise LivelockError when livelock is detected
                           after budget_steps are exceeded.

    Returns:
        Decorated function with a `_livelock_suite` attribute attached.

    Example:
        @livelock_probe_decorator(goal="fix the database error", k=3)
        def agent_step(context):
            return llm.call(context)

        outputs = []
        for i in range(20):
            output = agent_step(context)
            outputs.append(output)
            if agent_step._livelock_suite.step_count() >= 3:
                if not agent_step._livelock_suite.gate():
                    print("Livelock detected — stopping")
                    break
        report = agent_step._livelock_suite.compute()
    """
    def decorator(fn: Callable) -> Callable:
        label = agent_label or fn.__name__
        config = ProgressConfig(
            goal=goal,
            k=k,
            epsilon=epsilon,
            criticality=criticality,
            budget_steps=budget_steps,
            use_neural=use_neural,
            agent_label=label,
        )
        suite = LivelockSuite(config)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            # Record the return value as a step (convert to str if needed)
            step_output = result if isinstance(result, str) else str(result)
            suite.record_step(step_output)

            if raise_on_livelock and suite.is_over_budget():
                report = suite.compute()
                if report.livelock_detected:
                    raise LivelockError(
                        f"Agent '{label}' entered livelock state. "
                        f"LivelockScore={report.livelock_score:.3f}. "
                        f"Stuck from step {report.stuck_window_start}."
                    )
            return result

        # Attach suite so callers can access it
        wrapper._livelock_suite = suite  # type: ignore[attr-defined]
        return wrapper

    return decorator


class LivelockError(Exception):
    """
    Raised when raise_on_livelock=True and an agent exceeds its budget
    while in a detected livelock state.
    """
    pass
