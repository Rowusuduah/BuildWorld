"""Tests for llm_contract.judge (using mocks — no real LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_contract.judge import _serialize_output, compute_overall_score, evaluate_rule
from llm_contract.models import RuleResult, SemanticRule


class TestSerializeOutput:
    def test_pydantic_model(self):
        from pydantic import BaseModel

        class Summary(BaseModel):
            title: str
            body: str

        out = _serialize_output(Summary(title="T", body="B"))
        assert "title" in out
        assert '"T"' in out

    def test_plain_dict(self):
        out = _serialize_output({"key": "value"})
        assert '"key"' in out
        assert '"value"' in out

    def test_string(self):
        out = _serialize_output("hello world")
        assert out == "hello world"

    def test_list(self):
        out = _serialize_output([1, 2, 3])
        assert "1" in out

    def test_object_with_dict(self):
        class Obj:
            def __init__(self):
                self.x = 42

        out = _serialize_output(Obj())
        assert "x" in out


class TestComputeOverallScore:
    def test_all_passing(self):
        results = [
            RuleResult("r1", True, 0.9, "ok", weight=1.0),
            RuleResult("r2", True, 0.8, "ok", weight=1.0),
        ]
        score = compute_overall_score(results)
        assert abs(score - 0.85) < 0.01

    def test_all_failing(self):
        results = [
            RuleResult("r1", False, 0.2, "fail", weight=1.0),
            RuleResult("r2", False, 0.1, "fail", weight=1.0),
        ]
        score = compute_overall_score(results)
        assert score == 0.0

    def test_mixed_weights(self):
        results = [
            RuleResult("critical", True, 1.0, "ok", weight=1.0),
            RuleResult("soft", False, 0.0, "fail", weight=0.3),
        ]
        score = compute_overall_score(results)
        # critical: 1.0 * 1.0 = 1.0, soft: 0.0 * 0.3 = 0.0
        # total_weight = 1.3
        expected = 1.0 / 1.3
        assert abs(score - expected) < 0.01

    def test_empty_rules(self):
        assert compute_overall_score([]) == 1.0

    def test_zero_weight(self):
        results = [RuleResult("r", True, 0.9, "ok", weight=0.0)]
        # Zero total weight returns 1.0
        assert compute_overall_score(results) == 1.0


class TestEvaluateRule:
    def _mock_anthropic_response(self, passed: bool, confidence: float, reason: str):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        import json
        mock_response.content[0].text = json.dumps({
            "passed": passed,
            "confidence": confidence,
            "reason": reason,
        })
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_passing_rule(self, simple_rule):
        mock_client = self._mock_anthropic_response(True, 0.95, "No hallucinations found.")
        with patch("llm_contract.judge.anthropic") as mock_anthropic_module:
            mock_anthropic_module.Anthropic.return_value = mock_client
            result = evaluate_rule(simple_rule, "Good output", "anthropic", "claude-haiku-4-5-20251001")

        assert result.passed is True
        assert result.confidence == 0.95
        assert result.rule_name == "no_hallucination"

    def test_failing_rule(self, simple_rule):
        mock_client = self._mock_anthropic_response(False, 0.85, "Fabricated statistic detected.")
        with patch("llm_contract.judge.anthropic") as mock_anthropic_module:
            mock_anthropic_module.Anthropic.return_value = mock_client
            result = evaluate_rule(simple_rule, "Bad output", "anthropic", "claude-haiku-4-5-20251001")

        assert result.passed is False
        assert "Fabricated" in result.reason

    def test_disabled_rule(self):
        rule = SemanticRule("disabled_rule", "Should be skipped.", enabled=False)
        result = evaluate_rule(rule, "any output", "anthropic", "any-model")
        assert result.passed is True
        assert result.confidence == 1.0
        assert "disabled" in result.reason.lower()

    def test_confidence_below_threshold_fails(self, simple_rule):
        # simple_rule threshold=0.7, judge returns confidence=0.5 with passed=True
        mock_client = self._mock_anthropic_response(True, 0.5, "Barely passed.")
        with patch("llm_contract.judge.anthropic") as mock_anthropic_module:
            mock_anthropic_module.Anthropic.return_value = mock_client
            result = evaluate_rule(simple_rule, "output", "anthropic", "model")

        assert result.passed is False
        assert "threshold" in result.reason

    def test_unsupported_provider(self, simple_rule):
        with pytest.raises(ValueError, match="Unsupported judge provider"):
            evaluate_rule(simple_rule, "output", "mistral", "model")

    def test_openai_provider(self, simple_rule):
        mock_client = MagicMock()
        mock_response = MagicMock()
        import json
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "passed": True, "confidence": 0.9, "reason": "Good."
        })
        mock_client.chat.completions.create.return_value = mock_response

        with patch("llm_contract.judge.openai") as mock_openai:
            mock_openai.OpenAI.return_value = mock_client
            result = evaluate_rule(simple_rule, "output", "openai", "gpt-4o-mini")

        assert result.passed is True
