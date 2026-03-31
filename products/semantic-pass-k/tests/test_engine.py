"""Tests for semantic_pass_k.engine"""
from __future__ import annotations
import pytest

from semantic_pass_k.engine import (
    ConsistencyEngine,
    _tfidf_cosine,
    _tokenize,
)
from semantic_pass_k.models import CRITICALITY_THRESHOLDS


# ── Similarity stubs ─────────────────────────────────────────────────────────

ALWAYS_ONE = lambda a, b: 1.0       # perfect consistency
ALWAYS_ZERO = lambda a, b: 0.0     # zero consistency
ALWAYS_HALF = lambda a, b: 0.5     # medium consistency


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercases(self):
        assert "hello" in _tokenize("Hello World")

    def test_removes_punctuation(self):
        tokens = _tokenize("hello, world!")
        assert "," not in tokens

    def test_handles_empty_string(self):
        assert _tokenize("") == []

    def test_alphanumeric_kept(self):
        tokens = _tokenize("abc123 def")
        assert "abc123" in tokens
        assert "def" in tokens


# ── _tfidf_cosine ─────────────────────────────────────────────────────────────

class TestTfidfCosine:
    def test_identical_texts_near_one(self):
        sim = _tfidf_cosine("the quick brown fox", "the quick brown fox")
        assert sim > 0.99

    def test_empty_text_a_returns_zero(self):
        assert _tfidf_cosine("", "something") == 0.0

    def test_empty_text_b_returns_zero(self):
        assert _tfidf_cosine("something", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert _tfidf_cosine("", "") == 0.0

    def test_completely_different_texts(self):
        sim = _tfidf_cosine("alpha beta gamma", "delta epsilon zeta")
        assert sim == 0.0

    def test_overlapping_moderate_similarity(self):
        a = "water boils at one hundred degrees celsius at sea level"
        b = "water boils at one hundred degrees fahrenheit too"
        sim = _tfidf_cosine(a, b)
        assert 0.2 < sim < 0.95

    def test_result_in_range(self):
        texts = [
            ("foo bar baz", "baz qux foo"),
            ("hello world", "hello world hello"),
            ("abc", "def"),
        ]
        for a, b in texts:
            sim = _tfidf_cosine(a, b)
            assert 0.0 <= sim <= 1.0, f"Out of range: {sim} for ({a!r}, {b!r})"

    def test_symmetric(self):
        a = "the quick brown fox"
        b = "the quick lazy dog"
        assert abs(_tfidf_cosine(a, b) - _tfidf_cosine(b, a)) < 1e-9


# ── ConsistencyEngine ─────────────────────────────────────────────────────────

class TestConsistencyEngineInit:
    def test_default_engine_created(self):
        e = ConsistencyEngine()
        assert e is not None

    def test_custom_borderline_band(self):
        e = ConsistencyEngine(borderline_band=0.10)
        assert e.borderline_band == 0.10

    def test_custom_agent_label(self):
        e = ConsistencyEngine(agent_label="my_agent")
        assert e.agent_label == "my_agent"

    def test_injected_similarity_fn_stored(self):
        fn = lambda a, b: 0.5
        e = ConsistencyEngine(similarity_fn=fn)
        assert e._similarity_fn is fn


class TestConsistencyEngineEvaluate:
    def test_consistent_verdict_with_high_similarity(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b", "c"], "HIGH")
        assert result.verdict == "CONSISTENT"

    def test_inconsistent_verdict_with_zero_similarity(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ZERO)
        result = e.evaluate("p", ["a", "b", "c"], "HIGH")
        assert result.verdict == "INCONSISTENT"

    def test_borderline_verdict(self):
        # HIGH threshold = 0.90, borderline_band=0.05 → borderline if 0.85 <= score < 0.90
        e = ConsistencyEngine(similarity_fn=lambda a, b: 0.87, borderline_band=0.05)
        result = e.evaluate("p", ["a", "b", "c"], "HIGH")
        assert result.verdict == "BORDERLINE"

    def test_consistency_score_is_mean_of_pairwise(self):
        calls = []

        def counting_fn(a, b):
            calls.append(0.8)
            return 0.8

        e = ConsistencyEngine(similarity_fn=counting_fn)
        result = e.evaluate("p", ["a", "b", "c"], "HIGH")
        # 3 outputs → 3 pairs
        assert len(result.pairwise_scores) == 3
        assert abs(result.consistency_score - 0.8) < 1e-9

    def test_k_stored_correctly(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b", "c", "d", "e"], "HIGH")
        assert result.k == 5

    def test_pairwise_count_for_k3(self):
        # k=3 → 3*(3-1)/2 = 3 pairs
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b", "c"], "HIGH")
        assert len(result.pairwise_scores) == 3

    def test_pairwise_count_for_k5(self):
        # k=5 → 5*4/2 = 10 pairs
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b", "c", "d", "e"], "HIGH")
        assert len(result.pairwise_scores) == 10

    def test_pairwise_count_for_k2(self):
        # k=2 → 1 pair
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b"], "HIGH")
        assert len(result.pairwise_scores) == 1

    def test_raises_on_single_output(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        with pytest.raises(ValueError, match="at least 2"):
            e.evaluate("p", ["only one"], "HIGH")

    def test_raises_on_empty_outputs(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        with pytest.raises(ValueError):
            e.evaluate("p", [], "HIGH")

    def test_prompt_stored(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("my test prompt", ["a", "b"], "HIGH")
        assert result.prompt == "my test prompt"

    def test_outputs_stored(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["out1", "out2", "out3"], "HIGH")
        assert result.outputs == ["out1", "out2", "out3"]

    def test_criticality_stored(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b"], "CRITICAL")
        assert result.criticality == "CRITICAL"

    def test_threshold_matches_tier(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            result = e.evaluate("p", ["a", "b"], tier)  # type: ignore[arg-type]
            assert result.threshold == CRITICALITY_THRESHOLDS[tier]

    def test_run_id_is_string(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b"], "HIGH")
        assert isinstance(result.run_id, str)
        assert len(result.run_id) > 0

    def test_prompt_hash_is_16_chars(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = e.evaluate("p", ["a", "b"], "HIGH")
        assert len(result.prompt_hash) == 16

    def test_same_prompt_same_hash(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        r1 = e.evaluate("same prompt", ["a", "b"], "HIGH")
        r2 = e.evaluate("same prompt", ["c", "d"], "HIGH")
        assert r1.prompt_hash == r2.prompt_hash

    def test_different_prompts_different_hash(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        r1 = e.evaluate("prompt a", ["x", "y"], "HIGH")
        r2 = e.evaluate("prompt b", ["x", "y"], "HIGH")
        assert r1.prompt_hash != r2.prompt_hash

    def test_critical_tier_threshold(self):
        e = ConsistencyEngine(similarity_fn=lambda a, b: 0.985)
        result = e.evaluate("p", ["a", "b"], "CRITICAL")
        assert result.verdict == "BORDERLINE"  # 0.985 < 0.99

    def test_low_tier_passes_at_0_65(self):
        e = ConsistencyEngine(similarity_fn=lambda a, b: 0.65)
        result = e.evaluate("p", ["a", "b"], "LOW")
        assert result.verdict == "CONSISTENT"  # 0.65 >= 0.60


class TestConsistencyEngineEvaluateBatch:
    def test_returns_list_of_results(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        results = e.evaluate_batch(
            ["prompt1", "prompt2"],
            [["a", "b"], ["c", "d"]],
        )
        assert len(results) == 2

    def test_all_results_use_given_criticality(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        results = e.evaluate_batch(
            ["p1", "p2"],
            [["a", "b"], ["c", "d"]],
            criticality="MEDIUM",
        )
        for r in results:
            assert r.criticality == "MEDIUM"

    def test_mismatched_lengths_raise(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        with pytest.raises(ValueError):
            e.evaluate_batch(["p1", "p2"], [["a", "b"]])

    def test_empty_raises(self):
        e = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        with pytest.raises((ValueError, IndexError)):
            e.evaluate_batch([], [])
