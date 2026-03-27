"""Shared fixtures for llm-contract tests."""

from __future__ import annotations

import os
import tempfile

import pytest

from llm_contract.models import RuleResult, SemanticRule


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary SQLite database."""
    return str(tmp_path / "test_contracts.db")


@pytest.fixture
def simple_rule():
    return SemanticRule(
        name="no_hallucination",
        description="Output must not contain fabricated facts.",
        weight=1.0,
    )


@pytest.fixture
def soft_rule():
    return SemanticRule(
        name="concise",
        description="Response should be concise (under 200 words).",
        weight=0.5,
        threshold=0.6,
    )


@pytest.fixture
def passing_rule_result():
    return RuleResult(
        rule_name="no_hallucination",
        passed=True,
        confidence=0.95,
        reason="No fabricated facts detected.",
        weight=1.0,
    )


@pytest.fixture
def failing_rule_result():
    return RuleResult(
        rule_name="no_hallucination",
        passed=False,
        confidence=0.10,
        reason="Output contains invented statistics.",
        weight=1.0,
    )


@pytest.fixture(autouse=True)
def reset_llm_contract_config():
    """Reset llm-contract global config between tests."""
    import llm_contract
    from llm_contract.config import LLMContractConfig, _DEFAULT_DB_PATH, _DEFAULT_JUDGE_MODEL, _DEFAULT_JUDGE_PROVIDER, _DEFAULT_THRESHOLD
    import llm_contract.config as cfg
    original = cfg._config
    yield
    cfg._config = original
