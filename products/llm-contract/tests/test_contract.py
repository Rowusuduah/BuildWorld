"""Tests for the @contract decorator."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_contract import contract
from llm_contract.models import ContractViolationError, SemanticRule


class Summary(BaseModel):
    title: str
    body: str
    word_count: int = 0


def _mock_judge_pass():
    """Patch evaluate_rule to always return a passing result."""
    from llm_contract.models import RuleResult

    def _passing_evaluate(rule, output, provider, model):
        return RuleResult(
            rule_name=rule.name,
            passed=True,
            confidence=0.95,
            reason="Looks good.",
            weight=rule.weight,
        )

    return patch("llm_contract.contract.evaluate_rule", side_effect=_passing_evaluate)


def _mock_judge_fail():
    """Patch evaluate_rule to always return a failing result."""
    from llm_contract.models import RuleResult

    def _failing_evaluate(rule, output, provider, model):
        return RuleResult(
            rule_name=rule.name,
            passed=False,
            confidence=0.1,
            reason="Rule violated.",
            weight=rule.weight,
        )

    return patch("llm_contract.contract.evaluate_rule", side_effect=_failing_evaluate)


class TestContractStructuralValidation:
    def test_returns_pydantic_instance_unchanged(self, tmp_db):
        @contract(schema=Summary, version="1.0.0", validate_semantic=False, log_results=False)
        def fn(text: str) -> Summary:
            return Summary(title="T", body="B")

        result = fn("input")
        assert isinstance(result, Summary)
        assert result.title == "T"

    def test_returns_dict_coerced_to_schema(self, tmp_db):
        @contract(schema=Summary, version="1.0.0", validate_semantic=False, log_results=False)
        def fn(text: str):
            return {"title": "T", "body": "B"}

        result = fn("input")
        assert isinstance(result, Summary)

    def test_structural_violation_raises(self, tmp_db):
        @contract(
            schema=Summary, version="1.0.0", validate_semantic=False,
            on_violation="raise", log_results=False
        )
        def fn(text: str):
            return {"title": 123}  # Missing 'body', wrong type for title

        with pytest.raises(ContractViolationError) as exc_info:
            fn("input")
        assert exc_info.value.result.error is not None

    def test_no_schema_passthrough(self, tmp_db):
        @contract(version="1.0.0", validate_semantic=False, log_results=False)
        def fn():
            return "raw string output"

        assert fn() == "raw string output"


class TestContractSemanticValidation:
    def test_passing_semantic_rules(self, tmp_db, simple_rule):
        @contract(
            schema=Summary,
            semantic_rules=[simple_rule],
            version="1.0.0",
            log_results=False,
        )
        def fn() -> Summary:
            return Summary(title="T", body="B")

        with _mock_judge_pass():
            result = fn()
        assert isinstance(result, Summary)

    def test_failing_semantic_rules_raises(self, tmp_db, simple_rule):
        @contract(
            schema=Summary,
            semantic_rules=[simple_rule],
            version="1.0.0",
            on_violation="raise",
            log_results=False,
        )
        def fn() -> Summary:
            return Summary(title="T", body="B with hallucination")

        with _mock_judge_fail():
            with pytest.raises(ContractViolationError) as exc_info:
                fn()
        assert "no_hallucination" in str(exc_info.value)

    def test_validate_semantic_false_skips_judge(self, tmp_db, simple_rule):
        """With validate_semantic=False, judge should never be called."""
        @contract(
            schema=Summary,
            semantic_rules=[simple_rule],
            version="1.0.0",
            validate_semantic=False,
            log_results=False,
        )
        def fn() -> Summary:
            return Summary(title="T", body="B")

        with patch("llm_contract.contract.evaluate_rule") as mock_eval:
            fn()
            mock_eval.assert_not_called()

    def test_no_semantic_rules_no_judge_call(self, tmp_db):
        @contract(schema=Summary, version="1.0.0", log_results=False)
        def fn() -> Summary:
            return Summary(title="T", body="B")

        with patch("llm_contract.contract.evaluate_rule") as mock_eval:
            fn()
            mock_eval.assert_not_called()


class TestViolationStrategies:
    def test_warn_strategy(self, simple_rule):
        @contract(
            semantic_rules=[simple_rule],
            version="1.0.0",
            on_violation="warn",
            log_results=False,
        )
        def fn():
            return "output"

        with _mock_judge_fail():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = fn()
                assert len(w) == 1
                assert "Contract violation" in str(w[0].message)
        assert result == "output"

    def test_log_strategy_no_raise(self, simple_rule):
        @contract(
            semantic_rules=[simple_rule],
            version="1.0.0",
            on_violation="log",
            log_results=False,
        )
        def fn():
            return "output"

        with _mock_judge_fail():
            result = fn()
        assert result == "output"

    def test_fallback_strategy(self, simple_rule):
        fallback_called = []

        def my_fallback(x):
            fallback_called.append(x)
            return "fallback_output"

        @contract(
            semantic_rules=[simple_rule],
            version="1.0.0",
            on_violation="fallback",
            fallback=my_fallback,
            log_results=False,
        )
        def fn(x):
            return "primary_output"

        with _mock_judge_fail():
            result = fn("arg")
        assert result == "fallback_output"
        assert fallback_called == ["arg"]

    def test_fallback_requires_callable(self):
        with pytest.raises(ValueError, match="fallback callable"):
            @contract(on_violation="fallback")
            def fn():
                return "x"

    def test_raise_strategy_default(self, simple_rule):
        @contract(semantic_rules=[simple_rule], version="1.0.0", log_results=False)
        def fn():
            return "bad output"

        with _mock_judge_fail():
            with pytest.raises(ContractViolationError):
                fn()


class TestContractMetadata:
    def test_version_attached_to_wrapper(self):
        @contract(version="2.3.1", validate_semantic=False, log_results=False)
        def fn():
            return "x"

        assert fn.__contract_version__ == "2.3.1"

    def test_schema_attached_to_wrapper(self):
        @contract(schema=Summary, validate_semantic=False, log_results=False)
        def fn():
            return Summary(title="T", body="B")

        assert fn.__contract_schema__ is Summary

    def test_rules_attached_to_wrapper(self, simple_rule):
        @contract(semantic_rules=[simple_rule], validate_semantic=False, log_results=False)
        def fn():
            return "x"

        assert fn.__contract_rules__ == [simple_rule]

    def test_function_name_preserved(self):
        @contract(version="1.0.0", validate_semantic=False, log_results=False)
        def my_special_function():
            return "x"

        assert my_special_function.__name__ == "my_special_function"


class TestContractConfig:
    def test_custom_threshold(self, simple_rule):
        """Contract with threshold=1.0 should fail if confidence < 1.0."""
        from llm_contract.models import RuleResult

        def _evaluate(rule, output, provider, model):
            return RuleResult(rule.name, True, 0.85, "ok", rule.weight)

        @contract(
            semantic_rules=[simple_rule],
            version="1.0.0",
            threshold=0.95,  # Requires 95% — 0.85 won't pass
            on_violation="raise",
            log_results=False,
        )
        def fn():
            return "output"

        with patch("llm_contract.contract.evaluate_rule", side_effect=_evaluate):
            with pytest.raises(ContractViolationError):
                fn()
