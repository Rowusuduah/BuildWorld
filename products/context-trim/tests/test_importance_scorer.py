"""Tests for ImportanceScorer."""

from context_trim.core import ImportanceScorer


def make_msg(role: str, content: str = "") -> dict:
    return {"role": role, "content": content}


scorer = ImportanceScorer()


def test_system_always_max():
    msg = make_msg("system", "You are a helpful assistant.")
    score = scorer.score(msg, 0, 1)
    assert score == 1.0


def test_user_scores_above_assistant():
    user = make_msg("user", "What is the capital of France?")
    assistant = make_msg("assistant", "Paris.")
    u_score = scorer.score(user, 1, 3)
    a_score = scorer.score(assistant, 1, 3)
    assert u_score > a_score


def test_recency_newest_higher():
    msg = make_msg("user", "hello")
    old_score = scorer.score(msg, 0, 10)
    new_score = scorer.score(msg, 9, 10)
    assert new_score > old_score


def test_question_bonus():
    with_q = make_msg("user", "What is the plan?")
    without_q = make_msg("user", "The plan is ready.")
    s_q = scorer.score(with_q, 5, 10)
    s_n = scorer.score(without_q, 5, 10)
    assert s_q > s_n


def test_keyword_boost():
    boosted = make_msg("user", "This is a critical error in the system.")
    plain = make_msg("user", "The weather is nice today.")
    s_b = scorer.score(boosted, 5, 10)
    s_p = scorer.score(plain, 5, 10)
    assert s_b > s_p


def test_all_scores_in_range():
    messages = [
        make_msg("system", "You are helpful."),
        make_msg("user", "Hello."),
        make_msg("assistant", "Hi there."),
        make_msg("user", "What is the capital of France?"),
        make_msg("assistant", "Paris."),
    ]
    scores = scorer.score_all(messages)
    assert len(scores) == len(messages)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_score_all_length_matches():
    msgs = [make_msg("user", f"msg {i}") for i in range(10)]
    scores = scorer.score_all(msgs)
    assert len(scores) == 10


def test_single_message_recency():
    msg = make_msg("user", "hello")
    score = scorer.score(msg, 0, 1)
    assert 0.0 <= score <= 1.0


def test_unknown_role_defaults():
    msg = make_msg("custom_role", "some content")
    score = scorer.score(msg, 0, 5)
    assert 0.0 <= score <= 1.0
