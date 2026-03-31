"""
semantic_pass_k.decorators
--------------------------
@consistency_probe: decorator for measuring semantic consistency of LLM-calling functions.

Usage:
    @consistency_probe(k=5, criticality="HIGH", agent_label="summariser")
    def summarise(prompt: str) -> str:
        return call_llm(prompt)

    result = summarise("Summarise Ghana's 2024 budget.")
    # Returns the function output as normal, but also captures consistency result.
    # Access the last result via: summarise.last_consistency_result
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from .engine import ConsistencyEngine
from .models import CriticalityLevel, ConsistencyResult


def consistency_probe(
    k: int = 5,
    criticality: CriticalityLevel = "HIGH",
    agent_label: Optional[str] = None,
    engine: Optional[ConsistencyEngine] = None,
    raise_on_fail: bool = False,
) -> Callable:
    """
    Decorator that wraps an LLM-calling function and measures semantic consistency.

    The decorated function is called k times with the same arguments.
    All k outputs are evaluated for consistency.
    The original function still returns a single output (the first run's result).

    Args:
        k: Number of times to call the decorated function.
        criticality: Consistency threshold tier.
        agent_label: Label for the agent (defaults to function name).
        engine: Optional ConsistencyEngine. Default TF-IDF engine is used if None.
        raise_on_fail: If True, raise ConsistencyError when verdict is INCONSISTENT.

    Attributes added to the decorated function:
        last_consistency_result: The ConsistencyResult from the last call.

    Example:
        @consistency_probe(k=5, criticality="CRITICAL", raise_on_fail=True)
        def medical_diagnosis(patient_notes: str) -> str:
            return call_llm(patient_notes)
    """

    def decorator(fn: Callable) -> Callable:
        label = agent_label or fn.__name__
        _engine = engine or ConsistencyEngine(agent_label=label)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract prompt from first positional arg or 'prompt' kwarg
            prompt = args[0] if args else kwargs.get("prompt", str(args))

            # Run k times
            outputs = [fn(*args, **kwargs) for _ in range(k)]

            # Evaluate consistency
            result = _engine.evaluate(str(prompt), outputs, criticality)
            wrapper.last_consistency_result = result  # type: ignore[attr-defined]

            if raise_on_fail and result.verdict == "INCONSISTENT":
                raise ConsistencyError(
                    f"Agent '{label}' failed consistency check: "
                    f"score={result.consistency_score:.3f} "
                    f"threshold={result.threshold:.2f} "
                    f"criticality={criticality}"
                )

            # Return the first output (deterministic from caller's perspective)
            return outputs[0]

        wrapper.last_consistency_result = None  # type: ignore[attr-defined]
        return wrapper

    return decorator


class ConsistencyError(Exception):
    """Raised when an agent fails a consistency probe with raise_on_fail=True."""
    pass
