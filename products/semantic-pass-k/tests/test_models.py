"""Tests for semantic_pass_k.models"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone

from semantic_pass_k.models import (
    CRITICALITY_THRESHOLDS,
    ConsistencyReport,
    ConsistencyResult,
    CriticalityLevel,
    get_threshold,
    score_to_verdict,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_result(
    score: float = 0.95,
    criticality: CriticalityLevel = "HIGH",
    verdict: str = "CONSISTENT",
) -> ConsistencyResult:
    return ConsistencyResult(
        run_id="test-run-id",
        prompt="test prompt",
        outputs=["output a", "output b", "output c"],
        k=3,
        consistency_score=score,
        pairwise_scores=[0.95, 0.92, 0.98],
        verdict=verdict,
        criticality=criticality,
        threshold=CRITICALITY_THRESHOLDS[criticality],
        borderline_band=0.05,
        agent_label="test_agent",
        tested_at=datetime.now(timezone.utc),
        prompt_hash="abc123",
    )


# ── CRITICALITY_THRESHOLDS ────────────────────────────────────────────────────

class TestCriticalityThresholds:
    def test_all_four_tiers_present(self):
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            assert tier in CRITICALITY_THRESHOLDS

    def test_critical_highest(self):
        assert CRITICALITY_THRESHOLDS["CRITICAL"] > CRITICALITY_THRESHOLDS["HIGH"]

    def test_high_above_medium(self):
        assert CRITICALITY_THRESHOLDS["HIGH"] > CRITICALITY_THRESHOLDS["MEDIUM"]

    def test_medium_above_low(self):
        assert CRITICALITY_THRESHOLDS["MEDIUM"] > CRITICALITY_THRESHOLDS["LOW"]

    def test_critical_value(self):
        assert CRITICALITY_THRESHOLDS["CRITICAL"] == 0.99

    def test_high_value(self):
        assert CRITICALITY_THRESHOLDS["HIGH"] == 0.90

    def test_medium_value(self):
        assert CRITICALITY_THRESHOLDS["MEDIUM"] == 0.75

    def test_low_value(self):
        assert CRITICALITY_THRESHOLDS["LOW"] == 0.60

    def test_all_thresholds_between_zero_and_one(self):
        for v in CRITICALITY_THRESHOLDS.values():
            assert 0.0 < v <= 1.0


# ── get_threshold ─────────────────────────────────────────────────────────────

class TestGetThreshold:
    def test_critical(self):
        assert get_threshold("CRITICAL") == 0.99

    def test_high(self):
        assert get_threshold("HIGH") == 0.90

    def test_medium(self):
        assert get_threshold("MEDIUM") == 0.75

    def test_low(self):
        assert get_threshold("LOW") == 0.60


# ── score_to_verdict ─────────────────────────────────────────────────────────

class TestScoreToVerdict:
    def test_above_threshold_is_consistent(self):
        assert score_to_verdict(0.95, "HIGH") == "CONSISTENT"

    def test_at_threshold_is_consistent(self):
        assert score_to_verdict(0.90, "HIGH") == "CONSISTENT"

    def test_just_below_threshold_is_borderline(self):
        # 0.90 - 0.05 = 0.85 → still in borderline band
        assert score_to_verdict(0.87, "HIGH") == "BORDERLINE"

    def test_at_borderline_lower_edge_is_borderline(self):
        # 0.90 - 0.05 = 0.85 → exactly borderline lower edge
        assert score_to_verdict(0.85, "HIGH") == "BORDERLINE"

    def test_below_borderline_is_inconsistent(self):
        assert score_to_verdict(0.50, "HIGH") == "INCONSISTENT"

    def test_zero_score_is_inconsistent(self):
        assert score_to_verdict(0.0, "HIGH") == "INCONSISTENT"

    def test_critical_tier_above_threshold(self):
        assert score_to_verdict(0.995, "CRITICAL") == "CONSISTENT"

    def test_critical_tier_below_threshold(self):
        assert score_to_verdict(0.97, "CRITICAL") == "BORDERLINE"

    def test_critical_tier_far_below(self):
        assert score_to_verdict(0.80, "CRITICAL") == "INCONSISTENT"

    def test_low_tier_consistent(self):
        assert score_to_verdict(0.70, "LOW") == "CONSISTENT"

    def test_low_tier_borderline(self):
        assert score_to_verdict(0.57, "LOW") == "BORDERLINE"

    def test_custom_borderline_band(self):
        # With band=0.10, range 0.80-0.90 is borderline for HIGH
        assert score_to_verdict(0.83, "HIGH", borderline_band=0.10) == "BORDERLINE"
        assert score_to_verdict(0.79, "HIGH", borderline_band=0.10) == "INCONSISTENT"


# ── ConsistencyResult ────────────────────────────────────────────────────────

class TestConsistencyResult:
    def test_passed_property_consistent(self):
        r = make_result(score=0.95, verdict="CONSISTENT")
        assert r.passed is True

    def test_passed_property_borderline(self):
        r = make_result(score=0.87, verdict="BORDERLINE")
        assert r.passed is False

    def test_passed_property_inconsistent(self):
        r = make_result(score=0.50, verdict="INCONSISTENT")
        assert r.passed is False

    def test_n_pairs_correct(self):
        r = make_result()
        assert r.n_pairs == 3  # 3 pairwise scores in fixture

    def test_summary_contains_verdict(self):
        r = make_result(verdict="CONSISTENT")
        assert "CONSISTENT" in r.summary()

    def test_summary_contains_score(self):
        r = make_result(score=0.95)
        assert "0.950" in r.summary()

    def test_summary_contains_threshold(self):
        r = make_result(criticality="HIGH")
        assert "0.90" in r.summary()

    def test_summary_contains_k(self):
        r = make_result()
        assert "k=3" in r.summary()

    def test_metadata_defaults_to_empty_dict(self):
        r = make_result()
        assert r.metadata == {}


# ── ConsistencyReport ────────────────────────────────────────────────────────

class TestConsistencyReport:
    def _make_results(self, verdicts):
        score_map = {
            "CONSISTENT": 0.95,
            "BORDERLINE": 0.87,
            "INCONSISTENT": 0.50,
        }
        return [
            make_result(
                score=score_map[v],
                verdict=v,
                criticality="HIGH",
            )
            for v in verdicts
        ]

    def test_from_results_all_consistent(self):
        results = self._make_results(["CONSISTENT", "CONSISTENT", "CONSISTENT"])
        report = ConsistencyReport.from_results(results)
        assert report.verdict == "CONSISTENT"
        assert report.pass_rate == 1.0
        assert report.passed_results == 3
        assert report.failed_results == 0

    def test_from_results_some_inconsistent(self):
        results = self._make_results(["CONSISTENT", "INCONSISTENT", "CONSISTENT"])
        report = ConsistencyReport.from_results(results)
        assert report.verdict == "INCONSISTENT"
        assert report.failed_results == 1

    def test_from_results_all_borderline(self):
        results = self._make_results(["BORDERLINE", "BORDERLINE"])
        report = ConsistencyReport.from_results(results)
        assert report.verdict == "BORDERLINE"
        assert report.borderline_results == 2

    def test_overall_score_is_mean(self):
        results = self._make_results(["CONSISTENT", "INCONSISTENT"])
        report = ConsistencyReport.from_results(results)
        assert abs(report.overall_score - (0.95 + 0.50) / 2) < 1e-9

    def test_total_results_count(self):
        results = self._make_results(["CONSISTENT", "CONSISTENT", "INCONSISTENT"])
        report = ConsistencyReport.from_results(results)
        assert report.total_results == 3

    def test_label_stored(self):
        results = self._make_results(["CONSISTENT"])
        report = ConsistencyReport.from_results(results, label="my_agent_suite")
        assert report.label == "my_agent_suite"

    def test_empty_results_raises(self):
        with pytest.raises(ValueError):
            ConsistencyReport.from_results([])

    def test_summary_string(self):
        results = self._make_results(["CONSISTENT", "CONSISTENT"])
        report = ConsistencyReport.from_results(results)
        s = report.summary()
        assert "CONSISTENT" in s
        assert "pass_rate" in s

    def test_to_dict_keys(self):
        results = self._make_results(["CONSISTENT"])
        report = ConsistencyReport.from_results(results)
        d = report.to_dict()
        for key in ("report_id", "label", "verdict", "overall_score", "pass_rate", "results"):
            assert key in d

    def test_to_dict_results_list(self):
        results = self._make_results(["CONSISTENT", "BORDERLINE"])
        report = ConsistencyReport.from_results(results)
        d = report.to_dict()
        assert len(d["results"]) == 2
