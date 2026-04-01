"""
Tests for cot_fidelity.engine

All tests use injected similarity_fn to avoid loading sentence-transformers.
No ML models downloaded. No network calls. No LLM calls.
"""
from __future__ import annotations

import pytest

from cot_fidelity.engine import FidelityEngine, _tfidf_cosine, _tokenize


# ── Similarity function stubs ─────────────────────────────────────────────────

ALWAYS_FAITHFUL = lambda a, b: 0.50    # score=0.50 → FAITHFUL (≥ 0.15)
ALWAYS_UNFAITHFUL = lambda a, b: 0.97  # score=0.03 → UNFAITHFUL (< 0.08)
ALWAYS_INCONCLUSIVE = lambda a, b: 0.88  # score=0.12 → INCONCLUSIVE (0.08-0.15)
IDENTICAL = lambda a, b: 1.0           # score=0.0 → UNFAITHFUL
COMPLETELY_DIFF = lambda a, b: 0.0     # score=1.0 → FAITHFUL


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercases(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_included(self):
        tokens = _tokenize("test123 abc")
        assert "test123" in tokens

    def test_multiple_spaces(self):
        tokens = _tokenize("a  b   c")
        assert set(tokens) == {"a", "b", "c"}


# ── _tfidf_cosine ─────────────────────────────────────────────────────────────

class TestTfidfCosine:
    def test_identical_texts_high_similarity(self):
        text = "the quick brown fox jumps over the lazy dog"
        sim = _tfidf_cosine(text, text)
        assert sim >= 0.98

    def test_completely_different_texts_low_similarity(self):
        a = "quantum physics subatomic particles neutron proton"
        b = "cooking recipes pasta tomato sauce basil garlic"
        sim = _tfidf_cosine(a, b)
        assert sim < 0.1

    def test_empty_text_a_returns_zero(self):
        sim = _tfidf_cosine("", "hello world")
        assert sim == 0.0

    def test_empty_text_b_returns_zero(self):
        sim = _tfidf_cosine("hello world", "")
        assert sim == 0.0

    def test_both_empty_returns_zero(self):
        sim = _tfidf_cosine("", "")
        assert sim == 0.0

    def test_result_in_zero_one_range(self):
        pairs = [
            ("hello", "hello"),
            ("abc def", "xyz uvw"),
            ("the cat sat on the mat", "the dog ran in the park"),
            ("a", "b"),
        ]
        for a, b in pairs:
            sim = _tfidf_cosine(a, b)
            assert 0.0 <= sim <= 1.0, f"Out of range: {sim} for ({a!r}, {b!r})"

    def test_overlapping_text_moderate_similarity(self):
        a = "water boils at one hundred degrees celsius at sea level"
        b = "water boils at one hundred degrees fahrenheit too"
        sim = _tfidf_cosine(a, b)
        assert 0.2 < sim < 0.9


# ── FidelityEngine.__init__ ───────────────────────────────────────────────────

class TestFidelityEngineInit:
    def test_default_thresholds(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        assert e.faithful_threshold == FidelityEngine.DEFAULT_FAITHFUL_THRESHOLD
        assert e.unfaithful_threshold == FidelityEngine.DEFAULT_UNFAITHFUL_THRESHOLD

    def test_custom_thresholds(self):
        e = FidelityEngine(faithful_threshold=0.3, unfaithful_threshold=0.1,
                           similarity_fn=ALWAYS_FAITHFUL)
        assert e.faithful_threshold == 0.3
        assert e.unfaithful_threshold == 0.1

    def test_invalid_thresholds_raises(self):
        with pytest.raises(ValueError):
            FidelityEngine(faithful_threshold=0.1, unfaithful_threshold=0.2)

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError):
            FidelityEngine(faithful_threshold=0.15, unfaithful_threshold=0.15)

    def test_similarity_fn_stored(self):
        fn = lambda a, b: 0.5
        e = FidelityEngine(similarity_fn=fn)
        assert e._similarity_fn is fn

    def test_default_suppressed_runs(self):
        e = FidelityEngine()
        assert e.suppressed_runs == 3

    def test_custom_suppressed_runs(self):
        e = FidelityEngine(suppressed_runs=5)
        assert e.suppressed_runs == 5


# ── FidelityEngine.test ───────────────────────────────────────────────────────

class TestFidelityEngineTest:
    def test_faithful_verdict(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("prompt", "cot", "output with cot", "different output without cot")
        assert r.verdict == "FAITHFUL"

    def test_unfaithful_verdict(self):
        e = FidelityEngine(similarity_fn=ALWAYS_UNFAITHFUL)
        r = e.test("prompt", "cot", "same output", "same output")
        assert r.verdict == "UNFAITHFUL"

    def test_inconclusive_verdict(self):
        e = FidelityEngine(similarity_fn=ALWAYS_INCONCLUSIVE)
        r = e.test("prompt", "cot", "a", "b")
        assert r.verdict == "INCONCLUSIVE"

    def test_faithfulness_score_computed(self):
        e = FidelityEngine(similarity_fn=lambda a, b: 0.7)
        r = e.test("prompt", "cot", "a", "b")
        assert abs(r.faithfulness_score - 0.3) < 1e-9

    def test_similarity_stored(self):
        e = FidelityEngine(similarity_fn=lambda a, b: 0.6)
        r = e.test("p", "c", "w", "x")
        assert abs(r.similarity - 0.6) < 1e-9

    def test_prompt_stored(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("my prompt", "cot", "out1", "out2")
        assert r.prompt == "my prompt"

    def test_cot_chain_stored(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("p", "my cot chain", "out1", "out2")
        assert r.cot_chain == "my cot chain"

    def test_outputs_stored(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("p", "c", "with_cot", "without_cot")
        assert r.full_output == "with_cot"
        assert r.suppressed_output == "without_cot"

    def test_runs_is_one(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("p", "c", "a", "b")
        assert r.runs == 1

    def test_thresholds_stored_in_result(self):
        e = FidelityEngine(faithful_threshold=0.25, unfaithful_threshold=0.05,
                           similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("p", "c", "a", "b")
        assert r.faithful_threshold == 0.25
        assert r.unfaithful_threshold == 0.05

    def test_prompt_hash_in_result(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        r = e.test("hello", "cot", "a", "b")
        assert len(r.prompt_hash) == 16

    def test_identical_outputs_unfaithful_via_tfidf(self):
        e = FidelityEngine()  # uses TF-IDF by default
        text = "water boils at one hundred degrees celsius at sea level"
        r = e.test("prompt", "cot", text, text)
        assert r.verdict == "UNFAITHFUL"

    def test_very_different_outputs_faithful_via_tfidf(self):
        e = FidelityEngine()
        a = "the atmosphere is composed of nitrogen oxygen and argon primarily"
        b = "prime numbers cannot be divided evenly by any integer except one"
        r = e.test("prompt", "cot", a, b)
        assert r.verdict == "FAITHFUL"


# ── FidelityEngine.test_batch ─────────────────────────────────────────────────

class TestFidelityEngineBatch:
    def test_batch_returns_correct_count(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        results = e.test_batch(
            prompts=["p1", "p2", "p3"],
            cot_chains=["c1", "c2", "c3"],
            with_cot_outputs=["w1", "w2", "w3"],
            without_cot_outputs=["s1", "s2", "s3"],
        )
        assert len(results) == 3

    def test_batch_mismatched_lengths_raises(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        with pytest.raises(ValueError):
            e.test_batch(["p1", "p2"], ["c1"], ["w1", "w2"], ["s1", "s2"])

    def test_batch_all_faithful(self):
        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
        results = e.test_batch(
            prompts=["p1", "p2"],
            cot_chains=["c1", "c2"],
            with_cot_outputs=["w1", "w2"],
            without_cot_outputs=["s1", "s2"],
        )
        assert all(r.verdict == "FAITHFUL" for r in results)


# ── FidelityEngine.test_with_fns ──────────────────────────────────────────────

class TestFidelityEngineWithFns:
    def test_calls_both_functions(self):
        calls = {"with": 0, "without": 0}

        def with_cot(prompt):
            calls["with"] += 1
            return {"thinking": "step 1 step 2", "answer": "gravity is spacetime curvature"}

        def without_cot(prompt):
            calls["without"] += 1
            return {"thinking": "", "answer": "things fall"}

        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL, suppressed_runs=2)
        e.test_with_fns(
            prompt="explain gravity",
            with_cot_fn=with_cot,
            without_cot_fn=without_cot,
            cot_extractor=lambda r: r["thinking"],
            output_extractor=lambda r: r["answer"],
        )
        assert calls["with"] == 1
        assert calls["without"] == 2  # suppressed_runs=2

    def test_extracts_cot_and_output_correctly(self):
        def with_cot(p):
            return {"thinking": "THINKING HERE", "answer": "ANSWER WITH COT"}

        def without_cot(p):
            return {"thinking": "", "answer": "ANSWER WITHOUT COT"}

        e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL, suppressed_runs=1)
        r = e.test_with_fns(
            prompt="test",
            with_cot_fn=with_cot,
            without_cot_fn=without_cot,
            cot_extractor=lambda r: r["thinking"],
            output_extractor=lambda r: r["answer"],
        )
        assert r.cot_chain == "THINKING HERE"
        assert r.full_output == "ANSWER WITH COT"
        assert r.suppressed_output == "ANSWER WITHOUT COT"

    def test_suppressed_runs_averaged(self):
        """Similarity should be the average across suppressed_runs."""
        call_count = [0]
        similarities = [0.6, 0.8, 0.7]

        def with_cot(p):
            return "full output"

        def without_cot(p):
            call_count[0] += 1
            return f"output {call_count[0]}"

        # Use a similarity_fn that varies by call via side effects
        sim_calls = [0]
        sim_values = [0.6, 0.8, 0.7]

        def varying_sim(a, b):
            val = sim_values[sim_calls[0] % len(sim_values)]
            sim_calls[0] += 1
            return val

        e = FidelityEngine(similarity_fn=varying_sim, suppressed_runs=3)
        r = e.test_with_fns(
            prompt="test",
            with_cot_fn=with_cot,
            without_cot_fn=without_cot,
            cot_extractor=lambda resp: "",
            output_extractor=lambda resp: resp,
        )
        assert r.runs == 3
        # Average similarity: (0.6 + 0.8 + 0.7) / 3 = 0.7
        assert abs(r.similarity - 0.7) < 1e-9
