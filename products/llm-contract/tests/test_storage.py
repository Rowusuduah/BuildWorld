"""Tests for llm_contract.storage (SQLite drift logging)."""

from __future__ import annotations

import pytest

from llm_contract.models import ContractResult, RuleResult
from llm_contract.storage import get_drift_report, get_pass_rate, list_contracts, log_result


def _make_result(function_name: str, passed: bool, score: float) -> ContractResult:
    return ContractResult(
        passed=passed,
        overall_score=score,
        rule_results=[
            RuleResult(
                rule_name="test_rule",
                passed=passed,
                confidence=score,
                reason="test",
                weight=1.0,
            )
        ],
        contract_version="1.0.0",
        function_name=function_name,
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
    )


class TestLogResult:
    def test_log_creates_db(self, tmp_db):
        result = _make_result("my_fn", True, 0.95)
        log_result(result, tmp_db)
        import os
        assert os.path.exists(tmp_db)

    def test_log_and_retrieve(self, tmp_db):
        result = _make_result("summarize", True, 0.92)
        log_result(result, tmp_db)
        contracts = list_contracts(tmp_db)
        assert len(contracts) == 1
        assert contracts[0]["function_name"] == "summarize"
        assert contracts[0]["total"] == 1
        assert contracts[0]["passed_count"] == 1

    def test_log_multiple_results(self, tmp_db):
        for i in range(5):
            log_result(_make_result("fn", i % 2 == 0, 0.9), tmp_db)
        contracts = list_contracts(tmp_db)
        assert contracts[0]["total"] == 5
        assert contracts[0]["passed_count"] == 3

    def test_log_failed_result(self, tmp_db):
        log_result(_make_result("fn", False, 0.3), tmp_db)
        contracts = list_contracts(tmp_db)
        assert contracts[0]["passed_count"] == 0

    def test_log_result_with_error(self, tmp_db):
        result = ContractResult(
            passed=False,
            overall_score=0.0,
            rule_results=[],
            contract_version="1.0.0",
            function_name="broken_fn",
            provider="anthropic",
            model="model",
            error="Pydantic validation error: field missing",
        )
        log_result(result, tmp_db)
        contracts = list_contracts(tmp_db)
        assert contracts[0]["function_name"] == "broken_fn"


class TestGetPassRate:
    def test_no_evaluations(self, tmp_db):
        rate = get_pass_rate("nonexistent_fn", tmp_db)
        assert rate is None

    def test_all_pass(self, tmp_db):
        for _ in range(4):
            log_result(_make_result("fn", True, 0.9), tmp_db)
        rate = get_pass_rate("fn", tmp_db)
        assert rate == 1.0

    def test_mixed_pass_fail(self, tmp_db):
        log_result(_make_result("fn", True, 0.9), tmp_db)
        log_result(_make_result("fn", False, 0.2), tmp_db)
        log_result(_make_result("fn", True, 0.9), tmp_db)
        rate = get_pass_rate("fn", tmp_db)
        assert abs(rate - 2 / 3) < 0.01

    def test_different_functions_isolated(self, tmp_db):
        log_result(_make_result("fn_a", True, 0.9), tmp_db)
        log_result(_make_result("fn_b", False, 0.1), tmp_db)
        assert get_pass_rate("fn_a", tmp_db) == 1.0
        assert get_pass_rate("fn_b", tmp_db) == 0.0


class TestListContracts:
    def test_empty_db(self, tmp_db):
        assert list_contracts(tmp_db) == []

    def test_nonexistent_db(self, tmp_path):
        assert list_contracts(str(tmp_path / "nofile.db")) == []

    def test_multiple_functions(self, tmp_db):
        log_result(_make_result("fn_a", True, 0.9), tmp_db)
        log_result(_make_result("fn_b", True, 0.8), tmp_db)
        contracts = list_contracts(tmp_db)
        names = {c["function_name"] for c in contracts}
        assert "fn_a" in names
        assert "fn_b" in names


class TestGetDriftReport:
    def test_no_data(self, tmp_db):
        report = get_drift_report("fn", tmp_db, days=30)
        assert report["current_pass_rate"] is None
        assert report["has_drift"] is False

    def test_stable_no_drift(self, tmp_db):
        for _ in range(10):
            log_result(_make_result("fn", True, 0.95), tmp_db)
        report = get_drift_report("fn", tmp_db, days=30)
        assert not report["has_drift"]

    def test_report_keys_present(self, tmp_db):
        log_result(_make_result("fn", True, 0.9), tmp_db)
        report = get_drift_report("fn", tmp_db, days=30)
        assert "function_name" in report
        assert "current_pass_rate" in report
        assert "prior_pass_rate" in report
        assert "drift_pp" in report
        assert "has_drift" in report
        assert "evaluation_count" in report
