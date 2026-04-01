"""
Tests for cot_fidelity.store
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from cot_fidelity.models import FidelityResult
from cot_fidelity.store import FidelityStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_result(
    verdict="FAITHFUL",
    faithfulness_score=0.30,
    similarity=0.70,
    prompt="why does gravity work",
) -> FidelityResult:
    return FidelityResult(
        prompt=prompt,
        full_output="Gravity is spacetime curvature.",
        suppressed_output="Things fall.",
        cot_chain="spacetime bends mass...",
        similarity=similarity,
        faithfulness_score=faithfulness_score,
        verdict=verdict,
        faithful_threshold=0.15,
        unfaithful_threshold=0.08,
    )


@pytest.fixture
def store(tmp_path):
    return FidelityStore(db_path=tmp_path / "test.db")


# ── FidelityStore basic operations ────────────────────────────────────────────

class TestFidelityStoreBasic:
    def test_empty_on_init(self, store):
        assert store.count() == 0

    def test_save_returns_id(self, store):
        row_id = store.save(make_result())
        assert isinstance(row_id, str)
        assert len(row_id) == 36  # UUID format

    def test_count_increments_on_save(self, store):
        store.save(make_result())
        store.save(make_result())
        assert store.count() == 2

    def test_recent_returns_results(self, store):
        store.save(make_result(verdict="FAITHFUL"))
        results = store.recent(10)
        assert len(results) == 1
        assert results[0].verdict == "FAITHFUL"

    def test_recent_respects_limit(self, store):
        for _ in range(5):
            store.save(make_result())
        results = store.recent(3)
        assert len(results) == 3

    def test_recent_newest_first(self, store):
        store.save(make_result(faithfulness_score=0.1))
        store.save(make_result(faithfulness_score=0.9))
        results = store.recent(2)
        # Newest (0.9) should be first
        assert results[0].faithfulness_score == pytest.approx(0.9, abs=0.01)

    def test_by_verdict_faithful(self, store):
        store.save(make_result(verdict="FAITHFUL"))
        store.save(make_result(verdict="UNFAITHFUL"))
        faithful = store.by_verdict("FAITHFUL")
        assert len(faithful) == 1
        assert faithful[0].verdict == "FAITHFUL"

    def test_by_verdict_unfaithful(self, store):
        store.save(make_result(verdict="UNFAITHFUL"))
        store.save(make_result(verdict="FAITHFUL"))
        unfaithful = store.by_verdict("UNFAITHFUL")
        assert len(unfaithful) == 1

    def test_by_prompt_hash(self, store):
        r = make_result(prompt="unique prompt abc123")
        store.save(r)
        store.save(make_result(prompt="other prompt xyz"))
        results = store.by_prompt_hash(r.prompt_hash)
        assert len(results) == 1

    def test_clear_removes_all(self, store):
        store.save(make_result())
        store.save(make_result())
        n = store.clear()
        assert n == 2
        assert store.count() == 0

    def test_clear_returns_count(self, store):
        store.save(make_result())
        n = store.clear()
        assert n == 1

    def test_result_roundtrip(self, store):
        r = make_result(
            verdict="UNFAITHFUL",
            faithfulness_score=0.05,
            similarity=0.95,
            prompt="test roundtrip prompt",
        )
        store.save(r, model_version="v1.0")
        retrieved = store.recent(1)[0]
        assert retrieved.verdict == "UNFAITHFUL"
        assert abs(retrieved.faithfulness_score - 0.05) < 1e-6
        assert abs(retrieved.similarity - 0.95) < 1e-6
        assert retrieved.prompt == "test roundtrip prompt"

    def test_creates_parent_dirs(self, tmp_path):
        nested_path = tmp_path / "deep" / "nested" / "test.db"
        store = FidelityStore(db_path=nested_path)
        store.save(make_result())
        assert store.count() == 1


# ── Drift Detection ───────────────────────────────────────────────────────────

class TestDriftDetection:
    def test_insufficient_data_with_few_results(self, store):
        store.save(make_result())
        store.save(make_result())
        report = store.detect_drift(window=50)
        assert report.trend == "INSUFFICIENT_DATA"

    def test_stable_when_consistent_scores(self, store):
        for _ in range(20):
            store.save(make_result(faithfulness_score=0.4))
        report = store.detect_drift(window=20)
        assert report.trend in ("STABLE", "INSUFFICIENT_DATA")

    def test_mean_score_computed(self, store):
        scores = [0.2, 0.4, 0.6, 0.8]
        for s in scores:
            store.save(make_result(faithfulness_score=s))
        report = store.detect_drift(window=4)
        if report.trend != "INSUFFICIENT_DATA":
            assert abs(report.mean_score - 0.5) < 0.05

    def test_drift_report_has_window(self, store):
        for _ in range(5):
            store.save(make_result())
        report = store.detect_drift(window=30)
        assert report.window == 30

    def test_drift_report_has_points(self, store):
        for _ in range(5):
            store.save(make_result())
        report = store.detect_drift(window=10)
        assert len(report.points) <= 10
