"""
Tests for cot_fidelity.decorators
"""
from __future__ import annotations

import pytest

from cot_fidelity.decorators import (
    UnfaithfulCoTError,
    faithfulness_probe,
    faithfulness_probe_pair,
)
from cot_fidelity.models import FidelityResult


# ── Stubs ─────────────────────────────────────────────────────────────────────

ALWAYS_FAITHFUL = lambda a, b: 0.50
ALWAYS_UNFAITHFUL = lambda a, b: 0.97


def make_with_cot_fn():
    def fn(prompt):
        return {"thinking": "step by step reasoning here", "answer": "detailed answer"}
    return fn


def make_without_cot_fn():
    def fn(prompt):
        return {"thinking": "", "answer": "simple answer"}
    return fn


COT_EXTRACTOR = lambda r: r["thinking"]
OUTPUT_EXTRACTOR = lambda r: r["answer"]


# ── faithfulness_probe ────────────────────────────────────────────────────────

class TestFaithfulnessProbe:
    def test_decorator_preserves_function_name(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_FAITHFUL,
            suppressed_runs=1,
        )
        def my_func(prompt: str) -> str:
            return "answer"

        assert my_func.__name__ == "my_func"

    def test_decorator_returns_original_value(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_FAITHFUL,
            suppressed_runs=1,
        )
        def my_func(prompt: str):
            return {"result": "answer", "value": 42}

        result = my_func("explain gravity")
        assert result["result"] == "answer"
        assert result["value"] == 42

    def test_attaches_fidelity_to_result(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_FAITHFUL,
            suppressed_runs=1,
        )
        def my_func(prompt: str):
            return type("R", (), {})()  # simple object that can have .fidelity set

        result = my_func("explain gravity")
        assert hasattr(result, "fidelity")
        assert isinstance(result.fidelity, FidelityResult)

    def test_raises_when_unfaithful_and_flag_set(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_UNFAITHFUL,
            suppressed_runs=1,
            raise_on_unfaithful=True,
        )
        def my_func(prompt: str) -> str:
            return "answer"

        with pytest.raises(UnfaithfulCoTError) as exc_info:
            my_func("explain gravity")
        assert exc_info.value.result.verdict == "UNFAITHFUL"

    def test_no_raise_when_unfaithful_flag_false(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_UNFAITHFUL,
            suppressed_runs=1,
            raise_on_unfaithful=False,
        )
        def my_func(prompt: str) -> str:
            return "answer"

        result = my_func("explain gravity")
        assert result == "answer"

    def test_exposes_engine_on_wrapper(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_FAITHFUL,
            suppressed_runs=1,
        )
        def my_func(prompt: str) -> str:
            return "x"

        assert hasattr(my_func, "_fidelity_engine")

    def test_works_with_keyword_prompt(self):
        @faithfulness_probe(
            with_cot_fn=make_with_cot_fn(),
            without_cot_fn=make_without_cot_fn(),
            cot_extractor=COT_EXTRACTOR,
            output_extractor=OUTPUT_EXTRACTOR,
            similarity_fn=ALWAYS_FAITHFUL,
            suppressed_runs=1,
        )
        def my_func(prompt: str) -> str:
            return "answer"

        result = my_func(prompt="gravity question")
        assert result == "answer"


# ── UnfaithfulCoTError ────────────────────────────────────────────────────────

class TestUnfaithfulCoTError:
    def test_stores_result(self):
        r = FidelityResult(
            prompt="p",
            full_output="a",
            suppressed_output="a",
            cot_chain="c",
            similarity=0.99,
            faithfulness_score=0.01,
            verdict="UNFAITHFUL",
            faithful_threshold=0.15,
            unfaithful_threshold=0.08,
        )
        err = UnfaithfulCoTError(r)
        assert err.result is r

    def test_message_contains_verdict(self):
        r = FidelityResult(
            prompt="p",
            full_output="a",
            suppressed_output="a",
            cot_chain="c",
            similarity=0.99,
            faithfulness_score=0.01,
            verdict="UNFAITHFUL",
            faithful_threshold=0.15,
            unfaithful_threshold=0.08,
        )
        err = UnfaithfulCoTError(r)
        assert "UNFAITHFUL" in str(err)


# ── faithfulness_probe_pair ───────────────────────────────────────────────────

class TestFaithfulnessProbePair:
    def test_returns_fidelity_result(self):
        @faithfulness_probe_pair(
            cot_extractor=lambda r: r["thinking"],
            output_extractor=lambda r: r["answer"],
            similarity_fn=ALWAYS_FAITHFUL,
        )
        def run_pair(prompt: str):
            return (
                {"thinking": "step", "answer": "detailed"},
                {"thinking": "", "answer": "simple"},
            )

        result = run_pair("test prompt")
        assert isinstance(result, FidelityResult)

    def test_faithful_verdict(self):
        @faithfulness_probe_pair(
            cot_extractor=lambda r: r[0],
            output_extractor=lambda r: r[1],
            similarity_fn=ALWAYS_FAITHFUL,
        )
        def run_pair(prompt: str):
            return (("thinking", "long answer here"), ("", "short"))

        result = run_pair("prompt")
        assert result.verdict == "FAITHFUL"

    def test_raises_on_unfaithful_when_flag_set(self):
        @faithfulness_probe_pair(
            cot_extractor=lambda r: r[0],
            output_extractor=lambda r: r[1],
            similarity_fn=ALWAYS_UNFAITHFUL,
            raise_on_unfaithful=True,
        )
        def run_pair(prompt: str):
            return (("thinking", "same output"), ("", "same output"))

        with pytest.raises(UnfaithfulCoTError):
            run_pair("prompt")
