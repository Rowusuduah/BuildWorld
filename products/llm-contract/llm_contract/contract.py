"""The @contract decorator — the main entry point for llm-contract."""

from __future__ import annotations

import logging
import warnings
from functools import wraps
from typing import Any, Callable, Optional, Type

from llm_contract.config import get_config
from llm_contract.judge import compute_overall_score, evaluate_rule
from llm_contract.models import (
    ContractResult,
    ContractViolationError,
    RuleResult,
    SemanticRule,
    ViolationStrategy,
)
from llm_contract.storage import log_result

logger = logging.getLogger("llm_contract")


def contract(
    *,
    schema: Optional[Type] = None,
    semantic_rules: Optional[list[SemanticRule]] = None,
    version: str = "0.1.0",
    threshold: Optional[float] = None,
    on_violation: str = "raise",
    fallback: Optional[Callable] = None,
    validate_semantic: bool = True,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
    log_results: Optional[bool] = None,
) -> Callable:
    """Behavioral contract decorator for LLM function calls.

    Wraps any function that returns LLM output and enforces:
    1. Structural validation via Pydantic (if ``schema`` is provided)
    2. Semantic/behavioral validation via LLM judge (if ``semantic_rules`` provided)
    3. Contract version tracking
    4. Drift logging to SQLite

    Args:
        schema: Pydantic model class. If provided, the function's return value
            must be parseable into this schema (structural validation).
        semantic_rules: List of :class:`SemanticRule` objects defining behavioral
            requirements. Evaluated by an LLM judge.
        version: SemVer string for this contract (e.g. ``"1.0.0"``).
        threshold: Minimum weighted score to pass the contract (0.0–1.0).
            Defaults to the global ``default_threshold`` (0.90).
        on_violation: What to do when the contract is violated. One of:
            ``"raise"`` (default), ``"warn"``, ``"log"``, ``"fallback"``.
        fallback: Callable invoked with the same arguments as the decorated
            function when ``on_violation="fallback"``. Must be provided if
            ``on_violation="fallback"``.
        validate_semantic: If False, skip LLM judge evaluation entirely.
            Structural validation still runs. Default True.
        judge_model: Override the global judge model for this contract.
        judge_provider: Override the global judge provider for this contract.
        log_results: Override the global ``log_all_results`` for this contract.

    Returns:
        The decorated function, enhanced with contract enforcement.

    Raises:
        ContractViolationError: When ``on_violation="raise"`` and the contract fails.
        ValueError: If ``on_violation="fallback"`` but no ``fallback`` is provided.

    Example::

        from pydantic import BaseModel
        from llm_contract import contract, SemanticRule

        class Summary(BaseModel):
            title: str
            body: str

        @contract(
            schema=Summary,
            semantic_rules=[
                SemanticRule("no_hallucination", "Must not invent facts", weight=1.0),
            ],
            version="1.0.0",
        )
        def summarize(text: str) -> Summary:
            ...
    """
    if on_violation == ViolationStrategy.FALLBACK and fallback is None:
        raise ValueError("on_violation='fallback' requires a fallback callable.")

    violation_strategy = ViolationStrategy(on_violation)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = get_config()
            effective_threshold = threshold if threshold is not None else config.default_threshold
            effective_provider = judge_provider or config.default_judge_provider
            effective_model = judge_model or config.default_judge_model
            should_log = log_results if log_results is not None else config.log_all_results

            # Call the actual LLM function
            raw_output = func(*args, **kwargs)

            # --- Layer 1: Structural validation (Pydantic) ---
            structured_output = raw_output
            structural_error = None
            if schema is not None:
                try:
                    if isinstance(raw_output, schema):
                        structured_output = raw_output
                    elif isinstance(raw_output, dict):
                        structured_output = schema(**raw_output)
                    else:
                        structured_output = schema.model_validate(raw_output)
                except Exception as exc:
                    structural_error = str(exc)

            # --- Layer 2: Semantic validation (LLM judge) ---
            rule_results: list[RuleResult] = []
            if validate_semantic and semantic_rules and structural_error is None:
                for rule in semantic_rules:
                    result = evaluate_rule(
                        rule=rule,
                        output=structured_output,
                        provider=effective_provider,
                        model=effective_model,
                    )
                    rule_results.append(result)

            # --- Compute overall pass/fail ---
            if structural_error is not None:
                overall_score = 0.0
                passed = False
            else:
                overall_score = compute_overall_score(rule_results)
                # Critical rules (weight=1.0) failing means contract fails
                critical_failed = any(
                    r.weight >= 1.0 and not r.passed for r in rule_results
                )
                passed = not critical_failed and overall_score >= effective_threshold

            contract_result = ContractResult(
                passed=passed,
                overall_score=overall_score,
                rule_results=rule_results,
                contract_version=version,
                function_name=func.__name__,
                provider=effective_provider,
                model=effective_model,
                error=structural_error,
            )

            # --- Drift logging ---
            if should_log or not passed:
                try:
                    log_result(contract_result, config.db_path)
                except Exception as log_exc:
                    logger.warning("Failed to log contract result: %s", log_exc)

            # --- Violation handling ---
            if not passed:
                if violation_strategy == ViolationStrategy.RAISE:
                    raise ContractViolationError(contract_result, structured_output)
                elif violation_strategy == ViolationStrategy.WARN:
                    warnings.warn(
                        f"Contract violation in {func.__name__!r}: "
                        f"score={overall_score:.2%} | "
                        f"failed={[r.rule_name for r in contract_result.failed_rules]}",
                        stacklevel=2,
                    )
                elif violation_strategy == ViolationStrategy.LOG:
                    logger.warning(
                        "Contract violation in %r: score=%.2f failed=%s",
                        func.__name__,
                        overall_score,
                        [r.rule_name for r in contract_result.failed_rules],
                    )
                elif violation_strategy == ViolationStrategy.FALLBACK:
                    return fallback(*args, **kwargs)

            return structured_output if schema is not None else raw_output

        # Attach metadata for introspection
        wrapper.__contract_version__ = version
        wrapper.__contract_schema__ = schema
        wrapper.__contract_rules__ = semantic_rules or []
        return wrapper

    return decorator
