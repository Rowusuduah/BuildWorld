"""Tests for semantic_pass_k.pytest_plugin"""
from __future__ import annotations
import pytest

from semantic_pass_k.pytest_plugin import assert_consistent
from semantic_pass_k.engine import ConsistencyEngine
from semantic_pass_k.models import ConsistencyResult


ALWAYS_ONE = lambda a, b: 1.0
ALWAYS_ZERO = lambda a, b: 0.0
ALWAYS_HALF = lambda a, b: 0.5


class TestAssertConsistent:
    def test_passes_when_consistent(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        # Should not raise
        result = assert_consistent(["a", "b", "c"], criticality="HIGH", engine=engine)
        assert result.verdict == "CONSISTENT"

    def test_raises_on_inconsistent(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ZERO)
        with pytest.raises(AssertionError, match="Consistency check FAILED"):
            assert_consistent(["a", "b", "c"], criticality="HIGH", engine=engine)

    def test_borderline_passes_by_default(self):
        # HIGH threshold = 0.90; similarity = 0.87 → BORDERLINE
        engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.87, borderline_band=0.05)
        # borderline_passes=True (default) → should not raise
        assert_consistent(["a", "b"], criticality="HIGH", engine=engine)

    def test_borderline_fails_when_borderline_passes_false(self):
        engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.87, borderline_band=0.05)
        with pytest.raises(AssertionError):
            assert_consistent(
                ["a", "b"],
                criticality="HIGH",
                engine=engine,
                borderline_passes=False,
            )

    def test_returns_consistency_result(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        result = assert_consistent(["a", "b"], engine=engine)
        assert isinstance(result, ConsistencyResult)

    def test_assertion_message_contains_score(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ZERO)
        try:
            assert_consistent(["a", "b"], criticality="LOW", engine=engine)
        except AssertionError as e:
            assert "score" in str(e).lower()

    def test_assertion_message_contains_criticality(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ZERO)
        try:
            assert_consistent(["a", "b"], criticality="CRITICAL", engine=engine)
        except AssertionError as e:
            assert "CRITICAL" in str(e)

    def test_assertion_message_contains_k(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ZERO)
        try:
            assert_consistent(["a", "b", "c"], criticality="LOW", engine=engine)
        except AssertionError as e:
            assert "k" in str(e) or "3" in str(e)

    def test_low_criticality_is_easier_to_pass(self):
        engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.65)
        # LOW threshold = 0.60 → should pass
        result = assert_consistent(["a", "b"], criticality="LOW", engine=engine)
        assert result.verdict == "CONSISTENT"

    def test_critical_tier_fails_at_0_97(self):
        engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.97, borderline_band=0.05)
        # CRITICAL threshold = 0.99; 0.97 is borderline; borderline_passes=True → passes
        result = assert_consistent(["a", "b"], criticality="CRITICAL", engine=engine)
        assert result.verdict == "BORDERLINE"

    def test_at_least_2_outputs_required(self):
        engine = ConsistencyEngine(similarity_fn=ALWAYS_ONE)
        with pytest.raises((ValueError, AssertionError)):
            assert_consistent(["only_one"], engine=engine)
