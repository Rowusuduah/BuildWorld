import pytest
from prompt_lock.runner import run_exact_match, run_regex, run_eval
from prompt_lock.config import EvalConfig


class TestExactMatch:
    def test_pass(self):
        r = run_exact_match("q", "hello world", "hello world")
        assert r.passed is True
        assert r.score == 1.0

    def test_fail(self):
        r = run_exact_match("q", "hello world", "goodbye world")
        assert r.passed is False
        assert r.score == 0.0

    def test_strips_whitespace(self):
        r = run_exact_match("q", "  hello  ", "hello")
        assert r.passed is True

    def test_eval_type(self):
        r = run_exact_match("q", "x", "x")
        assert r.eval_type == "exact_match"


class TestRegex:
    def test_match(self):
        r = run_regex("q", "The answer is 42.", r"\d+")
        assert r.passed is True
        assert r.score == 1.0

    def test_no_match(self):
        r = run_regex("q", "The answer is forty-two.", r"\d+")
        assert r.passed is False
        assert r.score == 0.0

    def test_multiline(self):
        r = run_regex("q", "line1\nline2\nline3", r"line1.*line3", threshold=1.0)
        # DOTALL flag should make this match
        assert r.passed is True

    def test_eval_type(self):
        r = run_regex("q", "abc", r"abc")
        assert r.eval_type == "regex"


class TestRunEval:
    def test_exact_match_dispatch(self):
        cfg = EvalConfig(type="exact_match", threshold=1.0)
        r = run_eval(cfg, "q", "hello", expected="hello")
        assert r.passed is True

    def test_exact_match_requires_expected(self):
        cfg = EvalConfig(type="exact_match")
        with pytest.raises(ValueError, match="expected_output"):
            run_eval(cfg, "q", "hello", expected=None)

    def test_regex_dispatch(self):
        cfg = EvalConfig(type="regex", pattern=r"\d+")
        r = run_eval(cfg, "q", "The answer is 5.")
        assert r.passed is True

    def test_regex_requires_pattern(self):
        cfg = EvalConfig(type="regex")
        with pytest.raises(ValueError, match="pattern"):
            run_eval(cfg, "q", "hello")

    def test_llm_judge_requires_criteria(self):
        cfg = EvalConfig(type="llm_judge")
        with pytest.raises(ValueError, match="criteria"):
            run_eval(cfg, "q", "hello")

    def test_unknown_type(self):
        cfg = EvalConfig(type="exact_match")
        cfg.type = "unknown_type"  # type: ignore
        with pytest.raises(ValueError, match="Unknown eval type"):
            run_eval(cfg, "q", "hello")
