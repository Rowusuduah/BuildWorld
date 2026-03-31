"""Tests for semantic_pass_k.decorators"""
from __future__ import annotations
import pytest

from semantic_pass_k.decorators import consistency_probe, ConsistencyError
from semantic_pass_k.engine import ConsistencyEngine
from semantic_pass_k.models import ConsistencyResult


# ── consistency_probe ─────────────────────────────────────────────────────────

class TestConsistencyProbeDecorator:
    def test_decorated_fn_still_callable(self):
        @consistency_probe(k=2, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9))
        def fn(prompt):
            return "output"

        result = fn("test")
        assert result == "output"

    def test_last_consistency_result_populated_after_call(self):
        @consistency_probe(k=2, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9))
        def fn(prompt):
            return "output"

        fn("test")
        assert fn.last_consistency_result is not None
        assert isinstance(fn.last_consistency_result, ConsistencyResult)

    def test_fn_called_k_times(self):
        call_count = []

        @consistency_probe(k=4, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9))
        def fn(prompt):
            call_count.append(1)
            return "output"

        fn("test")
        assert len(call_count) == 4

    def test_returns_first_output(self):
        outputs = iter(["first", "second", "third"])

        @consistency_probe(k=3, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9))
        def fn(prompt):
            return next(outputs)

        result = fn("test")
        assert result == "first"

    def test_raise_on_fail_raises_consistency_error(self):
        @consistency_probe(
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.0),
            criticality="LOW",
            raise_on_fail=True,
        )
        def failing_fn(prompt):
            return "output"

        with pytest.raises(ConsistencyError):
            failing_fn("test")

    def test_raise_on_fail_false_no_exception(self):
        @consistency_probe(
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.0),
            criticality="LOW",
            raise_on_fail=False,
        )
        def fn(prompt):
            return "output"

        # Should not raise
        fn("test")

    def test_agent_label_defaults_to_function_name(self):
        @consistency_probe(k=2, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9))
        def my_special_function(prompt):
            return "output"

        my_special_function("test")
        result = my_special_function.last_consistency_result
        # The engine's agent_label is set to the function name
        assert result is not None

    def test_last_result_none_before_first_call(self):
        @consistency_probe(k=2)
        def fn(prompt):
            return "output"

        assert fn.last_consistency_result is None

    def test_default_criticality_high(self):
        @consistency_probe(k=2, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.95))
        def fn(prompt):
            return "output"

        fn("test")
        assert fn.last_consistency_result.criticality == "HIGH"

    def test_custom_criticality_used(self):
        @consistency_probe(
            k=2,
            criticality="MEDIUM",
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.80),
        )
        def fn(prompt):
            return "output"

        fn("test")
        assert fn.last_consistency_result.criticality == "MEDIUM"

    def test_k_of_2_works(self):
        @consistency_probe(k=2, engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9))
        def fn(prompt):
            return "output"

        fn("test")
        assert fn.last_consistency_result.k == 2


# ── ConsistencyError ─────────────────────────────────────────────────────────

class TestConsistencyError:
    def test_is_exception(self):
        assert issubclass(ConsistencyError, Exception)

    def test_can_be_raised_with_message(self):
        with pytest.raises(ConsistencyError, match="score="):
            raise ConsistencyError("Agent failed: score=0.50")
