"""Tests for MessageTrimmer — all five strategies."""

import pytest
from context_trim import TokenBudget, TrimStrategy
from context_trim.core import MessageTrimmer


def make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def make_budget(max_tokens: int = 4096, reserved: int = 512) -> TokenBudget:
    return TokenBudget(max_tokens=max_tokens, reserved_tokens=reserved)


trimmer = MessageTrimmer()

LONG_CONTENT = "x" * 500  # ~125 tokens each


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------


def test_empty_messages_returns_empty():
    result = trimmer.trim([], make_budget())
    assert result.messages == []
    assert result.original_count == 0
    assert result.final_count == 0
    assert result.dropped_count == 0


def test_fits_without_trimming():
    msgs = [make_msg("user", "hi"), make_msg("assistant", "hello")]
    result = trimmer.trim(msgs, make_budget())
    assert result.final_count == 2
    assert result.dropped_count == 0
    assert result.within_budget is True


# ---------------------------------------------------------------------------
# TrimResult fields
# ---------------------------------------------------------------------------


def test_trim_result_metadata():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=200, reserved=20)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    assert result.original_count == 20
    assert result.original_tokens > 0
    assert result.final_tokens <= result.original_tokens
    assert result.dropped_count == result.original_count - result.final_count
    assert 0.0 <= result.trim_ratio <= 1.0


def test_trim_result_summary_string():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(10)]
    budget = make_budget(max_tokens=100, reserved=10)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    s = result.summary()
    assert "context-trim" in s
    assert result.strategy.value in s


def test_trim_result_to_dict():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(5)]
    budget = make_budget(max_tokens=100, reserved=10)
    result = trimmer.trim(msgs, budget)
    d = result.to_dict()
    assert "within_budget" in d
    assert "strategy" in d
    assert "original_count" in d
    assert "final_count" in d
    assert "dropped_count" in d


# ---------------------------------------------------------------------------
# RECENCY_FIRST
# ---------------------------------------------------------------------------


def test_recency_first_drops_oldest():
    msgs = [make_msg("user", f"message {i} " + LONG_CONTENT) for i in range(10)]
    budget = make_budget(max_tokens=500, reserved=50)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    assert result.within_budget is True
    # The kept messages should be from the tail of the original list
    if result.final_count < result.original_count:
        # Last message should be retained
        assert result.messages[-1] == msgs[-1]


def test_recency_first_preserves_system():
    system = make_msg("system", "Always be helpful.")
    msgs = [system] + [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=300, reserved=30)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    roles = [m["role"] for m in result.messages]
    assert "system" in roles
    assert result.messages[0]["role"] == "system"


def test_recency_first_within_budget():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(50)]
    budget = make_budget(max_tokens=1000, reserved=100)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    assert result.within_budget is True


# ---------------------------------------------------------------------------
# IMPORTANCE
# ---------------------------------------------------------------------------


def test_importance_preserves_system():
    system = make_msg("system", "System instruction.")
    msgs = [system] + [make_msg("user", LONG_CONTENT) for _ in range(10)]
    budget = make_budget(max_tokens=300, reserved=30)
    result = trimmer.trim(msgs, budget, TrimStrategy.IMPORTANCE)
    roles = [m["role"] for m in result.messages]
    assert "system" in roles


def test_importance_within_budget():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=500, reserved=50)
    result = trimmer.trim(msgs, budget, TrimStrategy.IMPORTANCE)
    assert result.within_budget is True


def test_importance_drops_some_messages():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(30)]
    budget = make_budget(max_tokens=400, reserved=40)
    result = trimmer.trim(msgs, budget, TrimStrategy.IMPORTANCE)
    assert result.dropped_count > 0


# ---------------------------------------------------------------------------
# SLIDING_WINDOW
# ---------------------------------------------------------------------------


def test_sliding_window_keeps_tail():
    msgs = [make_msg("user", f"msg {i} " + LONG_CONTENT) for i in range(10)]
    budget = make_budget(max_tokens=500, reserved=50)
    result = trimmer.trim(msgs, budget, TrimStrategy.SLIDING_WINDOW)
    assert result.within_budget is True
    # Should include the last message
    if result.final_count > 0:
        assert result.messages[-1]["content"].startswith("msg 9")


def test_sliding_window_preserves_system():
    system = make_msg("system", "Be concise.")
    msgs = [system] + [make_msg("user", LONG_CONTENT) for _ in range(15)]
    budget = make_budget(max_tokens=400, reserved=40)
    result = trimmer.trim(msgs, budget, TrimStrategy.SLIDING_WINDOW)
    assert result.messages[0]["role"] == "system"


def test_sliding_window_within_budget():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=600, reserved=60)
    result = trimmer.trim(msgs, budget, TrimStrategy.SLIDING_WINDOW)
    assert result.within_budget is True


# ---------------------------------------------------------------------------
# SUMMARY_POINTS
# ---------------------------------------------------------------------------


def test_summary_points_inserts_summary():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(15)]
    budget = make_budget(max_tokens=400, reserved=40)
    result = trimmer.trim(msgs, budget, TrimStrategy.SUMMARY_POINTS)
    assert result.within_budget is True
    contents = [m["content"] for m in result.messages]
    has_summary = any("Context summary" in c or "summary" in c.lower() for c in contents)
    assert has_summary or result.dropped_count == 0


def test_summary_points_preserves_system():
    system = make_msg("system", "You are helpful.")
    msgs = [system] + [make_msg("user", LONG_CONTENT) for _ in range(10)]
    budget = make_budget(max_tokens=300, reserved=30)
    result = trimmer.trim(msgs, budget, TrimStrategy.SUMMARY_POINTS)
    roles = [m["role"] for m in result.messages]
    assert "system" in roles


def test_summary_points_within_budget():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=600, reserved=60)
    result = trimmer.trim(msgs, budget, TrimStrategy.SUMMARY_POINTS)
    assert result.within_budget is True


# ---------------------------------------------------------------------------
# HYBRID
# ---------------------------------------------------------------------------


def test_hybrid_within_budget():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(25)]
    budget = make_budget(max_tokens=600, reserved=60)
    result = trimmer.trim(msgs, budget, TrimStrategy.HYBRID)
    assert result.within_budget is True


def test_hybrid_preserves_system():
    system = make_msg("system", "Critical system context.")
    msgs = [system] + [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=400, reserved=40)
    result = trimmer.trim(msgs, budget, TrimStrategy.HYBRID)
    roles = [m["role"] for m in result.messages]
    assert "system" in roles


def test_hybrid_drops_low_score():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(20)]
    budget = make_budget(max_tokens=400, reserved=40)
    result = trimmer.trim(msgs, budget, TrimStrategy.HYBRID)
    assert result.dropped_count > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_message_over_budget():
    """A single very long message: trimmer should still return something."""
    msgs = [make_msg("user", "x" * 10000)]
    budget = make_budget(max_tokens=100, reserved=10)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    # It can't trim a single message — result may be over budget but not crash
    assert isinstance(result.within_budget, bool)


def test_only_system_messages():
    msgs = [make_msg("system", "Instruction."), make_msg("system", "More instruction.")]
    budget = make_budget(max_tokens=4096)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    assert result.within_budget is True


def test_unicode_content():
    msgs = [make_msg("user", "こんにちは世界 " * 50) for _ in range(10)]
    budget = make_budget(max_tokens=300, reserved=30)
    result = trimmer.trim(msgs, budget, TrimStrategy.RECENCY_FIRST)
    assert result.within_budget is True


def test_none_content():
    msgs = [{"role": "user", "content": None}]
    budget = make_budget(max_tokens=4096)
    result = trimmer.trim(msgs, budget)
    assert result.final_count >= 0


def test_all_strategies_produce_result():
    msgs = [make_msg("user", LONG_CONTENT) for _ in range(15)]
    budget = make_budget(max_tokens=400, reserved=40)
    for strategy in TrimStrategy:
        result = trimmer.trim(msgs, budget, strategy)
        assert result.within_budget is True
        assert result.final_count >= 0
