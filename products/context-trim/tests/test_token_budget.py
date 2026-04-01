"""Tests for TokenBudget."""

import pytest
from context_trim import TokenBudget


def make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# --- construction ---


def test_token_budget_basic():
    b = TokenBudget(max_tokens=4096)
    assert b.max_tokens == 4096
    assert b.reserved_tokens == 512
    assert b.available_tokens == 3584


def test_token_budget_custom_reserved():
    b = TokenBudget(max_tokens=8192, reserved_tokens=1024)
    assert b.available_tokens == 7168


def test_token_budget_zero_reserved():
    b = TokenBudget(max_tokens=1000, reserved_tokens=0)
    assert b.available_tokens == 1000


def test_token_budget_invalid_max():
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=0)


def test_token_budget_invalid_max_negative():
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=-1)


def test_token_budget_invalid_reserved_negative():
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=1000, reserved_tokens=-1)


def test_token_budget_reserved_equals_max():
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=512, reserved_tokens=512)


def test_token_budget_reserved_exceeds_max():
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=512, reserved_tokens=600)


# --- estimate ---


def test_estimate_empty():
    b = TokenBudget(max_tokens=4096)
    assert b.estimate("") == 0


def test_estimate_short():
    b = TokenBudget(max_tokens=4096)
    # "hello" = 5 chars → ceil(5/4) = 2
    assert b.estimate("hello") >= 1


def test_estimate_long():
    b = TokenBudget(max_tokens=4096)
    text = "a" * 400  # 400 chars / 4 = 100 tokens
    assert b.estimate(text) == 100


def test_estimate_messages_empty():
    b = TokenBudget(max_tokens=4096)
    assert b.estimate_messages([]) == 0


def test_estimate_messages_single():
    b = TokenBudget(max_tokens=4096)
    msgs = [make_msg("user", "hello")]
    est = b.estimate_messages(msgs)
    assert est > 0


def test_estimate_messages_grows_with_content():
    b = TokenBudget(max_tokens=4096)
    short = [make_msg("user", "hi")]
    long = [make_msg("user", "hi " * 100)]
    assert b.estimate_messages(long) > b.estimate_messages(short)


# --- fits ---


def test_fits_empty_messages():
    b = TokenBudget(max_tokens=4096)
    assert b.fits([]) is True


def test_fits_small():
    b = TokenBudget(max_tokens=4096)
    msgs = [make_msg("user", "short")]
    assert b.fits(msgs) is True


def test_fits_over_budget():
    b = TokenBudget(max_tokens=50, reserved_tokens=10)  # 40 available
    msgs = [make_msg("user", "x" * 1000)]  # ~250 tokens
    assert b.fits(msgs) is False


# --- tokens_over ---


def test_tokens_over_zero_when_fits():
    b = TokenBudget(max_tokens=4096)
    assert b.tokens_over([make_msg("user", "hi")]) == 0


def test_tokens_over_positive_when_exceeds():
    b = TokenBudget(max_tokens=50, reserved_tokens=10)
    msgs = [make_msg("user", "x" * 1000)]
    assert b.tokens_over(msgs) > 0
