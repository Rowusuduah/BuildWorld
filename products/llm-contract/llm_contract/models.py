"""Core data models for llm-contract."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ViolationStrategy(str, Enum):
    RAISE = "raise"
    WARN = "warn"
    LOG = "log"
    FALLBACK = "fallback"


@dataclass
class SemanticRule:
    """A behavioral rule that an LLM function output must satisfy.

    Args:
        name: Unique identifier for the rule (used in logs and error messages).
        description: Natural-language description of the required behavior. This is
            sent to the LLM judge verbatim, so be specific.
        weight: Importance of this rule (0.0–1.0). Rules with weight=1.0 are
            treated as critical — if they fail, the contract fails regardless of
            other rules. Defaults to 1.0.
        threshold: Minimum confidence score from judge to count as passing.
            Defaults to 0.7.
        enabled: Whether to evaluate this rule at all. Useful for disabling
            expensive rules in hot paths. Defaults to True.

    Example::

        SemanticRule(
            name="no_fabrication",
            description="Summary must only contain facts present in the source document. "
                        "Do not introduce statistics, names, or claims not in the input.",
            weight=1.0,
        )
    """

    name: str
    description: str
    weight: float = 1.0
    threshold: float = 0.7
    enabled: bool = True

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z][a-z0-9_]*$", self.name):
            raise ValueError(
                f"SemanticRule name must be lowercase snake_case, got: {self.name!r}"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"SemanticRule weight must be in [0.0, 1.0], got: {self.weight}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"SemanticRule threshold must be in [0.0, 1.0], got: {self.threshold}"
            )


@dataclass
class RuleResult:
    """Result of evaluating a single SemanticRule."""

    rule_name: str
    passed: bool
    confidence: float  # 0.0–1.0 from judge
    reason: str        # Judge's explanation
    weight: float


@dataclass
class ContractResult:
    """Aggregated result of evaluating all rules in a contract.

    Attributes:
        passed: True if the overall contract passed.
        overall_score: Weighted average of rule confidences.
        rule_results: Individual results per rule.
        contract_version: Version string from the @contract decorator.
        function_name: Decorated function name.
        provider: LLM provider used for judgment (e.g. "anthropic").
        model: Model used for judgment.
        error: If structural (Pydantic) validation failed, the error message.
    """

    passed: bool
    overall_score: float
    rule_results: list[RuleResult]
    contract_version: str
    function_name: str
    provider: str
    model: str
    error: Optional[str] = None

    @property
    def failed_rules(self) -> list[RuleResult]:
        return [r for r in self.rule_results if not r.passed]

    @property
    def passed_rules(self) -> list[RuleResult]:
        return [r for r in self.rule_results if r.passed]


class ContractViolationError(Exception):
    """Raised when an LLM function output violates its behavioral contract.

    Attributes:
        result: The full ContractResult with details on which rules failed.
        output: The raw output from the LLM function.

    Example::

        try:
            summary = summarize_document(doc)
        except ContractViolationError as e:
            print(f"Contract violated: {e.result.failed_rules}")
            print(f"Overall score: {e.result.overall_score:.2%}")
    """

    def __init__(self, result: ContractResult, output: Any) -> None:
        self.result = result
        self.output = output
        failed = ", ".join(r.rule_name for r in result.failed_rules)
        super().__init__(
            f"Contract violation in {result.function_name!r} v{result.contract_version}: "
            f"failed rules [{failed}] | score={result.overall_score:.2%}"
        )
