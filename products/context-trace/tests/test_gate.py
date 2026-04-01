"""Tests for context_trace.gate — AttributionGate, AttributionGateFailure."""

from __future__ import annotations

import pytest

from context_trace.gate import AttributionGate, AttributionGateFailure
from context_trace.tracer import AttributionReport, ChunkScore
from tests.conftest import make_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report_with_scores(**scores) -> AttributionReport:
    """Build a report with explicit {chunk_name: score} mapping."""
    chunk_scores = {
        name: ChunkScore(name, score, 1.0 - score, 0.0, 3)
        for name, score in scores.items()
    }
    return make_report(chunk_scores=chunk_scores)


# ---------------------------------------------------------------------------
# max_single_chunk_score
# ---------------------------------------------------------------------------

class TestMaxSingleChunkScore:
    def test_passes_all_below(self):
        gate = AttributionGate(max_single_chunk_score=0.90)
        report = report_with_scores(a=0.80, b=0.85)
        gate.check(report)  # no exception

    def test_passes_exactly_at_limit(self):
        gate = AttributionGate(max_single_chunk_score=0.90)
        report = report_with_scores(a=0.90)
        gate.check(report)  # at limit, not exceeded

    def test_fails_one_above(self):
        gate = AttributionGate(max_single_chunk_score=0.90)
        report = report_with_scores(a=0.95)
        with pytest.raises(AttributionGateFailure) as exc_info:
            gate.check(report)
        assert len(exc_info.value.violations) == 1
        assert "a" in exc_info.value.violations[0]

    def test_fails_multiple_above(self):
        gate = AttributionGate(max_single_chunk_score=0.50)
        report = report_with_scores(a=0.80, b=0.70, c=0.30)
        with pytest.raises(AttributionGateFailure) as exc_info:
            gate.check(report)
        assert len(exc_info.value.violations) == 2


# ---------------------------------------------------------------------------
# min_chunks_contributing
# ---------------------------------------------------------------------------

class TestMinChunksContributing:
    def test_passes_enough_contributors(self):
        gate = AttributionGate(min_chunks_contributing=2, contributing_threshold=0.3)
        report = report_with_scores(a=0.8, b=0.5, c=0.1)
        gate.check(report)  # a and b are contributing

    def test_fails_not_enough_contributors(self):
        gate = AttributionGate(min_chunks_contributing=3, contributing_threshold=0.3)
        report = report_with_scores(a=0.8, b=0.1, c=0.05)
        with pytest.raises(AttributionGateFailure):
            gate.check(report)

    def test_passes_exactly_minimum(self):
        gate = AttributionGate(min_chunks_contributing=2, contributing_threshold=0.3)
        report = report_with_scores(a=0.8, b=0.31, c=0.1)
        gate.check(report)  # exactly 2 contributing

    def test_custom_contributing_threshold(self):
        gate = AttributionGate(min_chunks_contributing=1, contributing_threshold=0.9)
        report = report_with_scores(a=0.95, b=0.50)
        gate.check(report)  # only a >= 0.9

    def test_fails_with_custom_threshold_none_above(self):
        gate = AttributionGate(min_chunks_contributing=1, contributing_threshold=0.99)
        report = report_with_scores(a=0.80, b=0.85)
        with pytest.raises(AttributionGateFailure):
            gate.check(report)


# ---------------------------------------------------------------------------
# min_top_contributor_score
# ---------------------------------------------------------------------------

class TestMinTopContributorScore:
    def test_passes_top_above_minimum(self):
        gate = AttributionGate(min_top_contributor_score=0.5)
        report = report_with_scores(a=0.8)
        gate.check(report)

    def test_fails_top_below_minimum(self):
        gate = AttributionGate(min_top_contributor_score=0.9)
        report = report_with_scores(a=0.8, b=0.5)
        with pytest.raises(AttributionGateFailure) as exc_info:
            gate.check(report)
        assert "min_top_contributor_score" in exc_info.value.violations[0]

    def test_passes_top_exactly_at_minimum(self):
        gate = AttributionGate(min_top_contributor_score=0.8)
        report = report_with_scores(a=0.8)
        gate.check(report)


# ---------------------------------------------------------------------------
# max_total_api_calls
# ---------------------------------------------------------------------------

class TestMaxTotalApiCalls:
    def test_passes_under_limit(self):
        gate = AttributionGate(max_total_api_calls=20)
        report = make_report(total_api_calls=9)
        gate.check(report)

    def test_fails_over_limit(self):
        gate = AttributionGate(max_total_api_calls=5)
        report = make_report(total_api_calls=9)
        with pytest.raises(AttributionGateFailure):
            gate.check(report)

    def test_passes_exactly_at_limit(self):
        gate = AttributionGate(max_total_api_calls=9)
        report = make_report(total_api_calls=9)
        gate.check(report)


# ---------------------------------------------------------------------------
# Combined gates
# ---------------------------------------------------------------------------

class TestCombinedGates:
    def test_multiple_violations_all_reported(self):
        gate = AttributionGate(
            max_single_chunk_score=0.5,   # doc1 (0.85) will fail
            min_chunks_contributing=5,     # only 3 chunks, will fail
            min_top_contributor_score=0.99, # top is 0.85, will fail
        )
        report = make_report()  # doc1=0.85, doc2=0.20, system=0.10
        with pytest.raises(AttributionGateFailure) as exc_info:
            gate.check(report)
        assert len(exc_info.value.violations) == 3

    def test_none_gates_skip_all_checks(self):
        gate = AttributionGate()  # all None — no checks
        report = report_with_scores(a=0.99, b=0.01)
        gate.check(report)  # should not raise


# ---------------------------------------------------------------------------
# Non-raising variants
# ---------------------------------------------------------------------------

class TestGateNonRaising:
    def test_passed_returns_true_on_pass(self):
        gate = AttributionGate(max_single_chunk_score=0.90)
        report = report_with_scores(a=0.70)
        assert gate.passed(report) is True

    def test_passed_returns_false_on_fail(self):
        gate = AttributionGate(max_single_chunk_score=0.50)
        report = report_with_scores(a=0.80)
        assert gate.passed(report) is False

    def test_result_returns_true_empty_list(self):
        gate = AttributionGate(max_single_chunk_score=0.90)
        report = report_with_scores(a=0.70)
        ok, violations = gate.result(report)
        assert ok is True
        assert violations == []

    def test_result_returns_false_with_violations(self):
        gate = AttributionGate(max_single_chunk_score=0.50)
        report = report_with_scores(a=0.80)
        ok, violations = gate.result(report)
        assert ok is False
        assert len(violations) == 1
