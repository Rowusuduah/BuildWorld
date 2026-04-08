"""Tests for pressure_gauge.engine"""
import pytest

from pressure_gauge.engine import (
    _compute_tfidf,
    _tokenize,
    approx_token_count,
    build_padded_context,
    compute_similarities,
    cosine_similarity,
    generate_padding,
    run_sweep,
)
from pressure_gauge.models import CriticalityLevel, PressureConfig, PressureReport


# ---------------------------------------------------------------------------
# approx_token_count
# ---------------------------------------------------------------------------

class TestApproxTokenCount:
    def test_empty_string_returns_one(self):
        assert approx_token_count("") == 1

    def test_four_char_word(self):
        # "abcd" = 4 chars / 4.0 = 1 token
        assert approx_token_count("abcd") == 1

    def test_eight_chars(self):
        assert approx_token_count("abcdefgh") == 2

    def test_custom_chars_per_token(self):
        assert approx_token_count("abcdefgh", chars_per_token=8.0) == 1

    def test_long_text(self):
        text = "word " * 1000  # 5000 chars
        count = approx_token_count(text, chars_per_token=4.0)
        assert count == 1250

    def test_returns_int(self):
        assert isinstance(approx_token_count("hello world"), int)

    def test_minimum_is_one(self):
        assert approx_token_count("") >= 1
        assert approx_token_count("a") >= 1


# ---------------------------------------------------------------------------
# generate_padding
# ---------------------------------------------------------------------------

class TestGeneratePadding:
    def test_lorem_ipsum_strategy(self):
        pad = generate_padding(100, "lorem_ipsum")
        assert len(pad) > 0
        assert isinstance(pad, str)

    def test_repeat_text_strategy_uses_custom(self):
        pad = generate_padding(200, "repeat_text", custom_text="CUSTOM_TEXT_XYZ")
        assert "CUSTOM_TEXT_XYZ" in pad

    def test_inject_history_strategy(self):
        pad = generate_padding(100, "inject_history")
        assert len(pad) > 0

    def test_zero_tokens_returns_empty(self):
        pad = generate_padding(0, "lorem_ipsum")
        assert pad == ""

    def test_negative_tokens_returns_empty(self):
        pad = generate_padding(-10, "lorem_ipsum")
        assert pad == ""

    def test_approximate_length_reached(self):
        target_tokens = 500
        chars_per_token = 4.0
        pad = generate_padding(target_tokens, "lorem_ipsum", chars_per_token)
        expected_chars = int(target_tokens * chars_per_token)
        # Should be approximately target length
        assert abs(len(pad) - expected_chars) <= 10

    def test_empty_custom_text_falls_back(self):
        pad = generate_padding(100, "repeat_text", custom_text="")
        assert len(pad) > 0

    def test_unknown_strategy_falls_back_to_lorem(self):
        pad = generate_padding(50, "unknown_xyz")
        assert len(pad) > 0

    def test_large_padding(self):
        pad = generate_padding(10000, "lorem_ipsum")
        assert len(pad) >= 1000  # At minimum ~chars


# ---------------------------------------------------------------------------
# build_padded_context
# ---------------------------------------------------------------------------

class TestBuildPaddedContext:
    def test_pads_short_context(self):
        base = "What is 2+2?"
        result = build_padded_context(base, target_tokens=1000, strategy="lorem_ipsum")
        assert base in result
        assert len(result) > len(base)

    def test_does_not_shrink_already_long_context(self):
        base = "word " * 5000  # Very long
        result = build_padded_context(base, target_tokens=10, strategy="lorem_ipsum")
        assert result == base

    def test_base_context_preserved_at_end(self):
        base = "UNIQUE_SENTINEL_XYZ"
        result = build_padded_context(base, target_tokens=500, strategy="lorem_ipsum")
        assert "UNIQUE_SENTINEL_XYZ" in result

    def test_padding_prepended(self):
        base = "TASK_HERE"
        result = build_padded_context(base, target_tokens=500, strategy="lorem_ipsum")
        # Base context should appear at the end
        assert result.endswith(base)

    def test_returns_string(self):
        result = build_padded_context("hello", target_tokens=100, strategy="lorem_ipsum")
        assert isinstance(result, str)

    def test_padding_strategy_respected(self):
        result_lorem = build_padded_context("task", 500, "lorem_ipsum")
        result_hist = build_padded_context("task", 500, "inject_history")
        # Both should be longer than base
        assert len(result_lorem) > len("task")
        assert len(result_hist) > len("task")


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_words(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tokens = _tokenize("UPPER CASE text")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_included(self):
        tokens = _tokenize("answer is 42")
        assert "42" in tokens

    def test_apostrophes_kept(self):
        tokens = _tokenize("it's fine")
        assert any("'" in t for t in tokens) or "its" in tokens or "it" in tokens

    def test_returns_list(self):
        assert isinstance(_tokenize("hello"), list)


# ---------------------------------------------------------------------------
# _compute_tfidf
# ---------------------------------------------------------------------------

class TestComputeTFIDF:
    def test_empty_docs_returns_empty(self):
        assert _compute_tfidf([]) == []

    def test_single_doc_has_terms(self):
        result = _compute_tfidf(["hello world"])
        assert len(result) == 1
        assert "hello" in result[0] or "world" in result[0]

    def test_multiple_docs_same_length(self):
        result = _compute_tfidf(["hello world", "world peace", "hello peace"])
        assert len(result) == 3

    def test_empty_doc_returns_empty_dict(self):
        result = _compute_tfidf(["hello world", ""])
        assert result[1] == {}

    def test_positive_values(self):
        result = _compute_tfidf(["hello world"])
        for v in result[0].values():
            assert v > 0

    def test_common_term_has_lower_idf(self):
        # "world" appears in all 3 docs; "unique_term" only in first
        docs = ["hello world unique_term", "world foo", "world bar"]
        result = _compute_tfidf(docs)
        # unique_term should have higher TF-IDF than world in doc 0
        if "unique_term" in result[0] and "world" in result[0]:
            assert result[0]["unique_term"] >= result[0]["world"]

    def test_returns_list_of_dicts(self):
        result = _compute_tfidf(["a b c"])
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        vec = {"a": 1.0, "b": 2.0, "c": 0.5}
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_empty_both_returns_0(self):
        assert cosine_similarity({}, {}) == 0.0

    def test_one_empty_returns_0(self):
        assert cosine_similarity({"a": 1.0}, {}) == 0.0
        assert cosine_similarity({}, {"b": 1.0}) == 0.0

    def test_orthogonal_returns_0(self):
        sim = cosine_similarity({"a": 1.0}, {"b": 1.0})
        assert sim == 0.0

    def test_partial_overlap_between_0_and_1(self):
        sim = cosine_similarity(
            {"a": 1.0, "b": 1.0},
            {"a": 1.0, "c": 1.0},
        )
        assert 0.0 < sim < 1.0

    def test_returns_float(self):
        sim = cosine_similarity({"x": 0.5}, {"x": 0.5})
        assert isinstance(sim, float)

    def test_bounded_above_by_1(self):
        sim = cosine_similarity({"a": 100.0}, {"a": 100.0, "b": 0.001})
        assert sim <= 1.0

    def test_bounded_below_by_0(self):
        sim = cosine_similarity({"a": 1.0}, {"b": 1.0})
        assert sim >= 0.0

    def test_scaling_invariant(self):
        vec_a = {"a": 1.0, "b": 2.0}
        vec_b = {"a": 2.0, "b": 4.0}  # 2x scale of vec_a
        sim = cosine_similarity(vec_a, vec_b)
        assert abs(sim - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_similarities
# ---------------------------------------------------------------------------

class TestComputeSimilarities:
    def test_empty_list_returns_empty(self):
        assert compute_similarities([]) == []

    def test_single_output_returns_list_of_one(self):
        sims = compute_similarities(["hello world"])
        assert len(sims) == 1

    def test_identical_outputs_all_near_1(self):
        text = "the quick brown fox jumps over the lazy dog"
        sims = compute_similarities([text, text, text])
        for s in sims:
            assert s > 0.99

    def test_first_element_is_baseline(self):
        sims = compute_similarities(["baseline text here", "different text entirely"])
        # Baseline similarity to itself
        assert abs(sims[0] - 1.0) < 1e-6

    def test_completely_different_lower_similarity(self):
        sims = compute_similarities([
            "apple orange banana fruit salad",
            "dog cat bird mammal animal habitat",
        ])
        assert sims[0] > sims[1] or sims[1] < 1.0

    def test_returns_list_of_floats(self):
        sims = compute_similarities(["a b c", "d e f"])
        assert all(isinstance(s, float) for s in sims)

    def test_length_matches_input(self):
        outputs = ["a", "b", "c", "d", "e"]
        sims = compute_similarities(outputs)
        assert len(sims) == 5

    def test_all_values_in_0_to_1(self):
        outputs = ["hello world", "world peace", "hello peace today"]
        sims = compute_similarities(outputs)
        for s in sims:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def _stable_agent(self, ctx: str) -> str:
        return "The answer is 42. This is a complete and thorough response."

    def test_returns_pressure_report(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = run_sweep(config, self._stable_agent)
        assert isinstance(report, PressureReport)

    def test_stable_agent_passes_low_criticality(self):
        config = PressureConfig(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5, 0.9],
            criticality=CriticalityLevel.LOW,
        )
        report = run_sweep(config, self._stable_agent)
        assert report.gate_passed

    def test_drift_curve_length_matches_fill_levels(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.3, 0.5])
        report = run_sweep(config, self._stable_agent)
        assert len(report.drift_curve) == 3

    def test_drift_curve_fill_levels_correct(self):
        levels = [0.1, 0.4, 0.8]
        config = PressureConfig(model_context_limit=2048, fill_levels=levels)
        report = run_sweep(config, self._stable_agent)
        actual = [dp.fill_level for dp in report.drift_curve]
        assert actual == sorted(levels)

    def test_score_between_0_and_1(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = run_sweep(config, self._stable_agent)
        assert 0.0 <= report.context_pressure_score <= 1.0

    def test_drifting_agent_fails_critical_threshold(self):
        call_count = [0]

        def drifting_agent(ctx: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                return "extremely thorough comprehensive detailed analysis"
            return "x"

        config = PressureConfig(
            model_context_limit=2048,
            fill_levels=[0.1, 0.9],
            stability_threshold=0.999,
            criticality=CriticalityLevel.CRITICAL,
        )
        report = run_sweep(config, drifting_agent)
        assert not report.gate_passed

    def test_base_context_injected_at_every_level(self):
        received = []

        def recording_agent(ctx: str) -> str:
            received.append(ctx)
            return "response"

        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        run_sweep(config, recording_agent, base_context="UNIQUE_TASK_MARKER")
        assert all("UNIQUE_TASK_MARKER" in ctx for ctx in received)

    def test_single_fill_level_score_computed(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.5])
        report = run_sweep(config, self._stable_agent)
        assert len(report.drift_curve) == 1
        assert report.context_pressure_score >= 0.0

    def test_stable_agent_onset_not_detected(self):
        config = PressureConfig(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5, 0.9],
            criticality=CriticalityLevel.LOW,
        )
        report = run_sweep(config, self._stable_agent)
        if report.gate_passed:
            assert report.pressure_onset_token is None

    def test_padding_strategies_all_work(self):
        for strategy in ("lorem_ipsum", "repeat_text", "inject_history"):
            config = PressureConfig(
                model_context_limit=1024,
                fill_levels=[0.1, 0.5],
                padding_strategy=strategy,
            )
            report = run_sweep(config, self._stable_agent)
            assert isinstance(report, PressureReport)

    def test_recommendation_not_empty(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = run_sweep(config, self._stable_agent)
        assert len(report.recommendation) > 0

    def test_verdict_set(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = run_sweep(config, self._stable_agent)
        assert report.verdict is not None
