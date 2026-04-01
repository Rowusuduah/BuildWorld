"""Tests for the high-level ContextTrim API."""

import pytest
from context_trim import ContextTrim, TrimStrategy


def make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


LONG = "x" * 500


# --- basic API ---


def test_context_trim_init():
    ct = ContextTrim(max_tokens=8192)
    assert ct.budget.max_tokens == 8192


def test_estimate_returns_int():
    ct = ContextTrim(max_tokens=8192)
    msgs = [make_msg("user", "hello")]
    assert isinstance(ct.estimate(msgs), int)
    assert ct.estimate(msgs) > 0


def test_estimate_empty():
    ct = ContextTrim(max_tokens=8192)
    assert ct.estimate([]) == 0


def test_fits_small():
    ct = ContextTrim(max_tokens=8192)
    assert ct.fits([make_msg("user", "hi")]) is True


def test_fits_large():
    ct = ContextTrim(max_tokens=50, reserved_tokens=10)
    assert ct.fits([make_msg("user", "x" * 5000)]) is False


def test_tokens_over_zero_when_fits():
    ct = ContextTrim(max_tokens=8192)
    assert ct.tokens_over([make_msg("user", "short")]) == 0


def test_tokens_over_positive():
    ct = ContextTrim(max_tokens=50, reserved_tokens=10)
    assert ct.tokens_over([make_msg("user", "x" * 5000)]) > 0


def test_trim_returns_result():
    ct = ContextTrim(max_tokens=200, reserved_tokens=20)
    msgs = [make_msg("user", LONG) for _ in range(10)]
    result = ct.trim(msgs)
    assert result.within_budget is True
    assert len(result.messages) > 0


def test_trim_default_strategy_is_hybrid():
    ct = ContextTrim(max_tokens=200, reserved_tokens=20)
    msgs = [make_msg("user", LONG) for _ in range(10)]
    result = ct.trim(msgs)
    assert result.strategy == TrimStrategy.HYBRID


def test_trim_explicit_strategy():
    ct = ContextTrim(max_tokens=200, reserved_tokens=20)
    msgs = [make_msg("user", LONG) for _ in range(10)]
    result = ct.trim(msgs, strategy=TrimStrategy.RECENCY_FIRST)
    assert result.strategy == TrimStrategy.RECENCY_FIRST


def test_trim_document():
    ct = ContextTrim(max_tokens=200, reserved_tokens=20)
    long_text = "paragraph content. " * 500
    result = ct.trim_document(long_text)
    assert result.final_tokens <= ct.budget.available_tokens * 1.2


# --- CI gate ---


def test_ci_gate_passes():
    ct = ContextTrim(max_tokens=8192)
    msgs = [make_msg("user", "hello")]
    ct.ci_gate(msgs)  # should not raise


def test_ci_gate_raises():
    ct = ContextTrim(max_tokens=50, reserved_tokens=10)
    msgs = [make_msg("user", "x" * 5000)]
    with pytest.raises(RuntimeError) as exc_info:
        ct.ci_gate(msgs)
    assert "CI gate FAILED" in str(exc_info.value)


def test_ci_gate_custom_message():
    ct = ContextTrim(max_tokens=50, reserved_tokens=10)
    msgs = [make_msg("user", "x" * 5000)]
    with pytest.raises(RuntimeError, match="custom error"):
        ct.ci_gate(msgs, fail_message="custom error")


# --- store integration ---


def test_trim_with_store(tmp_path):
    db = str(tmp_path / "test.db")
    ct = ContextTrim(max_tokens=200, reserved_tokens=20, db_path=db)
    msgs = [make_msg("user", LONG) for _ in range(5)]
    ct.trim(msgs, pipeline_id="test_pipeline")

    from context_trim import TrimStore
    store = TrimStore(db_path=db)
    history = store.history("test_pipeline")
    assert len(history) == 1


def test_trim_with_store_multiple(tmp_path):
    db = str(tmp_path / "test2.db")
    ct = ContextTrim(max_tokens=200, reserved_tokens=20, db_path=db)
    msgs = [make_msg("user", LONG) for _ in range(5)]
    for i in range(3):
        ct.trim(msgs, pipeline_id="pipeline_a")

    from context_trim import TrimStore
    store = TrimStore(db_path=db)
    history = store.history("pipeline_a")
    assert len(history) == 3
