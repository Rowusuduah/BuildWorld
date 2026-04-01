"""
cot_fidelity.decorators
-----------------------
@faithfulness_probe — drop-in decorator for monitoring CoT faithfulness.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional, Tuple

from .engine import FidelityEngine
from .models import FidelityResult
from .store import FidelityStore


class UnfaithfulCoTError(Exception):
    """Raised by @faithfulness_probe when verdict is UNFAITHFUL and raise_on_unfaithful=True."""

    def __init__(self, result: FidelityResult) -> None:
        self.result = result
        super().__init__(
            f"CoT faithfulness gate failed: verdict={result.verdict}, "
            f"score={result.faithfulness_score:.4f}"
        )


def faithfulness_probe(
    *,
    with_cot_fn: Callable,
    without_cot_fn: Callable,
    cot_extractor: Callable,
    output_extractor: Callable,
    faithful_threshold: float = FidelityEngine.DEFAULT_FAITHFUL_THRESHOLD,
    unfaithful_threshold: float = FidelityEngine.DEFAULT_UNFAITHFUL_THRESHOLD,
    use_neural: bool = False,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    suppressed_runs: int = 3,
    store: Optional[FidelityStore] = None,
    model_version: str = "",
    raise_on_unfaithful: bool = False,
    attach_result: bool = True,
) -> Callable:
    """
    Decorator that wraps a function calling a reasoning model and attaches
    a FidelityResult to the output.

    The decorated function must accept a `prompt` parameter (positional or keyword).
    The result is appended as a `.fidelity` attribute on the return value.

    Args:
        with_cot_fn: Function that runs the model WITH CoT. Called with (prompt,).
        without_cot_fn: Function that runs the model WITHOUT CoT. Called with (prompt,).
        cot_extractor: Extracts the CoT chain from the with_cot_fn response.
        output_extractor: Extracts the final answer from any response.
        faithful_threshold: Score >= this → FAITHFUL.
        unfaithful_threshold: Score < this → UNFAITHFUL.
        use_neural: Use sentence-transformers for similarity.
        similarity_fn: Injectable similarity function (overrides use_neural).
        suppressed_runs: Number of suppressed runs to average.
        store: Optional FidelityStore for persistence.
        model_version: Tag for store entries.
        raise_on_unfaithful: Raise UnfaithfulCoTError if verdict == UNFAITHFUL.
        attach_result: Attach .fidelity to the return value if possible.

    Usage:
        @faithfulness_probe(
            with_cot_fn=call_model_with_thinking,
            without_cot_fn=call_model_without_thinking,
            cot_extractor=lambda r: r.thinking,
            output_extractor=lambda r: r.text,
        )
        def answer(prompt: str) -> str:
            return call_model_with_thinking(prompt).text

        result = answer("What is 15% of 240?")
        print(result)               # "36.0"
        print(result.fidelity)      # FidelityResult(verdict='FAITHFUL', ...)
    """
    engine = FidelityEngine(
        faithful_threshold=faithful_threshold,
        unfaithful_threshold=unfaithful_threshold,
        use_neural=use_neural,
        similarity_fn=similarity_fn,
        suppressed_runs=suppressed_runs,
    )

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Extract prompt — first positional arg or 'prompt' kwarg
            if args:
                prompt = args[0]
            else:
                prompt = kwargs.get("prompt", "")

            fidelity_result = engine.test_with_fns(
                prompt=prompt,
                with_cot_fn=with_cot_fn,
                without_cot_fn=without_cot_fn,
                cot_extractor=cot_extractor,
                output_extractor=output_extractor,
            )

            if store:
                store.save(fidelity_result, model_version=model_version)

            if raise_on_unfaithful and fidelity_result.verdict == "UNFAITHFUL":
                raise UnfaithfulCoTError(fidelity_result)

            # Call the original function
            return_value = fn(*args, **kwargs)

            # Attach .fidelity if possible
            if attach_result:
                try:
                    return_value.fidelity = fidelity_result
                except (AttributeError, TypeError):
                    pass

            return return_value

        # Expose last_result for inspection
        wrapper._fidelity_engine = engine
        return wrapper

    return decorator


def faithfulness_probe_pair(
    *,
    cot_extractor: Callable,
    output_extractor: Callable,
    faithful_threshold: float = FidelityEngine.DEFAULT_FAITHFUL_THRESHOLD,
    unfaithful_threshold: float = FidelityEngine.DEFAULT_UNFAITHFUL_THRESHOLD,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    raise_on_unfaithful: bool = False,
) -> Callable:
    """
    Simpler decorator for functions that return (with_cot_response, without_cot_response).
    The function should return a tuple of two raw responses.
    The decorator extracts CoT and outputs, computes faithfulness, and returns FidelityResult.

    Usage:
        @faithfulness_probe_pair(
            cot_extractor=lambda r: r.thinking,
            output_extractor=lambda r: r.text,
        )
        def run_pair(prompt: str):
            return (call_with_cot(prompt), call_without_cot(prompt))

        result = run_pair("Explain Newton's laws")
        assert isinstance(result, FidelityResult)
    """
    engine = FidelityEngine(
        faithful_threshold=faithful_threshold,
        unfaithful_threshold=unfaithful_threshold,
        similarity_fn=similarity_fn,
    )

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if args:
                prompt = args[0]
            else:
                prompt = kwargs.get("prompt", "")

            pair: Tuple = fn(*args, **kwargs)
            with_cot_response, without_cot_response = pair

            cot_chain = cot_extractor(with_cot_response)
            full_output = output_extractor(with_cot_response)
            suppressed_output = output_extractor(without_cot_response)

            result = engine.test(prompt, cot_chain, full_output, suppressed_output)

            if raise_on_unfaithful and result.verdict == "UNFAITHFUL":
                raise UnfaithfulCoTError(result)

            return result

        return wrapper

    return decorator
