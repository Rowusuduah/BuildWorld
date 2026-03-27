"""Tests for llm_contract.models."""

from __future__ import annotations

import pytest

from llm_contract.models import (
    ContractResult,
    ContractViolationError,
    RuleResult,
    SemanticRule,
    ViolationStrategy,
)


class TestSemanticRule:
    def test_valid_creation(self):
        rule = SemanticRule(name="no_hallucination", description="No fabricated facts.")
        assert rule.name == "no_hallucination"
        assert rule.weight == 1.0
        assert rule.threshold == 0.7
        assert rule.enabled is True

    def test_custom_weight_and_threshold(self):
        rule = SemanticRule(
            name="concise",
            description="Must be concise.",
            weight=0.5,
            threshold=0.6,
        )
        assert rule.weight == 0.5
        assert rule.threshold == 0.6

    def test_invalid_name_uppercase(self):
        with pytest.raises(ValueError, match="lowercase snake_case"):
            SemanticRule(name="NoHallucination", description="desc")

    def test_invalid_name_hyphen(self):
        with pytest.raises(ValueError, match="lowercase snake_case"):
            SemanticRule(name="no-hallucination", description="desc")

    def test_invalid_name_starts_with_number(self):
        with pytest.raises(ValueError, match="lowercase snake_case"):
            SemanticRule(name="1rule", description="desc")

    def test_invalid_weight_too_high(self):
        with pytest.raises(ValueError, match="weight"):
            SemanticRule(name="rule", description="desc", weight=1.5)

    def test_invalid_weight_negative(self):
        with pytest.raises(ValueError, match="weight"):
            SemanticRule(name="rule", description="desc", weight=-0.1)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            SemanticRule(name="rule", description="desc", threshold=1.1)

    def test_disabled_rule(self):
        rule = SemanticRule(name="optional_check", description="desc", enabled=False)
        assert rule.enabled is False

    def test_name_with_numbers_valid(self):
        rule = SemanticRule(name="rule_v2_check", description="desc")
        assert rule.name == "rule_v2_check"


class TestContractResult:
    def make_result(self, passed_list: list[bool]) -> ContractResult:
        rule_results = [
            RuleResult(
                rule_name=f"rule_{i}",
                passed=p,
                confidence=0.9 if p else 0.1,
                reason="ok" if p else "fail",
                weight=1.0,
            )
            for i, p in enumerate(passed_list)
        ]
        return ContractResult(
            passed=all(passed_list),
            overall_score=0.9 if all(passed_list) else 0.1,
            rule_results=rule_results,
            contract_version="1.0.0",
            function_name="test_fn",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )

    def test_passed_rules_filter(self):
        result = self.make_result([True, False, True])
        assert len(result.passed_rules) == 2
        assert len(result.failed_rules) == 1

    def test_all_pass(self):
        result = self.make_result([True, True])
        assert result.failed_rules == []

    def test_all_fail(self):
        result = self.make_result([False, False])
        assert result.passed_rules == []


class TestContractViolationError:
    def test_error_message_contains_function_name(self, failing_rule_result):
        result = ContractResult(
            passed=False,
            overall_score=0.1,
            rule_results=[failing_rule_result],
            contract_version="1.0.0",
            function_name="my_llm_function",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )
        error = ContractViolationError(result, output={"text": "bad"})
        assert "my_llm_function" in str(error)
        assert "no_hallucination" in str(error)

    def test_error_has_result_attribute(self, failing_rule_result):
        result = ContractResult(
            passed=False,
            overall_score=0.0,
            rule_results=[failing_rule_result],
            contract_version="2.0.0",
            function_name="fn",
            provider="openai",
            model="gpt-4o-mini",
        )
        error = ContractViolationError(result, output="bad output")
        assert error.result is result
        assert error.output == "bad output"


class TestViolationStrategy:
    def test_all_strategies_valid(self):
        for v in ("raise", "warn", "log", "fallback"):
            assert ViolationStrategy(v) is not None

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            ViolationStrategy("ignore")
