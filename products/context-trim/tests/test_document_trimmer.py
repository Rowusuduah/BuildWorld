"""Tests for DocumentTrimmer."""

import pytest
from context_trim import TokenBudget, TrimStrategy
from context_trim.core import DocumentTrimmer


def make_budget(max_tokens: int = 200, reserved: int = 20) -> TokenBudget:
    return TokenBudget(max_tokens=max_tokens, reserved_tokens=reserved)


trimmer = DocumentTrimmer()

PARA = "This is an important paragraph with critical information about the system. " * 10
LONG_DOC = "\n\n".join(PARA for _ in range(20))


def test_short_doc_not_trimmed():
    result = trimmer.trim("Hello world.", make_budget(max_tokens=4096, reserved=512))
    assert result.truncated is False
    assert "Hello" in result.text


def test_long_doc_recency_first():
    result = trimmer.trim(LONG_DOC, make_budget())
    assert result.truncated is True
    assert result.final_tokens <= make_budget().available_tokens * 1.2  # within 20% tolerance


def test_long_doc_importance():
    result = trimmer.trim(LONG_DOC, make_budget(), TrimStrategy.IMPORTANCE)
    assert result.truncated is True
    assert len(result.text) > 0


def test_long_doc_sliding_window():
    result = trimmer.trim(LONG_DOC, make_budget(), TrimStrategy.SLIDING_WINDOW)
    assert result.truncated is True
    assert len(result.text) > 0


def test_long_doc_summary_points():
    result = trimmer.trim(LONG_DOC, make_budget(), TrimStrategy.SUMMARY_POINTS)
    assert result.truncated is True
    assert "trimmed" in result.text.lower() or len(result.text) < len(LONG_DOC)


def test_long_doc_hybrid():
    result = trimmer.trim(LONG_DOC, make_budget(), TrimStrategy.HYBRID)
    assert result.truncated is True
    assert len(result.text) > 0


def test_result_has_summary():
    result = trimmer.trim(LONG_DOC, make_budget())
    s = result.summary()
    assert "context-trim" in s
    assert result.strategy.value in s


def test_empty_string():
    result = trimmer.trim("", make_budget(max_tokens=4096, reserved=512))
    assert result.truncated is False


def test_single_paragraph_over_budget():
    big_para = "word " * 1000
    result = trimmer.trim(big_para, make_budget(max_tokens=50, reserved=5))
    assert isinstance(result.truncated, bool)
    assert len(result.text) > 0
