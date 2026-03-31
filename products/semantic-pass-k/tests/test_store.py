"""Tests for semantic_pass_k.store"""
from __future__ import annotations
import os
import tempfile
import pytest
from datetime import datetime, timezone

from semantic_pass_k.store import ConsistencyStore
from semantic_pass_k.models import ConsistencyResult, ConsistencyReport


def make_result(
    run_id: str = "run-001",
    agent_label: str = "test_agent",
    score: float = 0.95,
    verdict: str = "CONSISTENT",
) -> ConsistencyResult:
    return ConsistencyResult(
        run_id=run_id,
        prompt="What is 2+2?",
        prompt_hash="deadbeef",
        outputs=["4", "four", "4.0"],
        k=3,
        consistency_score=score,
        pairwise_scores=[0.95, 0.92, 0.98],
        verdict=verdict,
        criticality="HIGH",
        threshold=0.90,
        borderline_band=0.05,
        agent_label=agent_label,
        tested_at=datetime.now(timezone.utc),
    )


# ── ConsistencyStore ─────────────────────────────────────────────────────────

class TestConsistencyStore:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_consistency.db")
        return ConsistencyStore(db_path)

    def test_store_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        ConsistencyStore(db_path)
        assert os.path.exists(db_path)

    def test_save_and_retrieve_result(self, store):
        result = make_result(run_id="abc-123")
        store.save_result(result)
        retrieved = store.get_result("abc-123")
        assert retrieved is not None
        assert retrieved.run_id == "abc-123"

    def test_retrieved_score_matches(self, store):
        result = make_result(run_id="r1", score=0.87)
        store.save_result(result)
        retrieved = store.get_result("r1")
        assert abs(retrieved.consistency_score - 0.87) < 1e-9

    def test_retrieved_verdict_matches(self, store):
        result = make_result(run_id="r2", verdict="INCONSISTENT")
        store.save_result(result)
        retrieved = store.get_result("r2")
        assert retrieved.verdict == "INCONSISTENT"

    def test_retrieved_outputs_match(self, store):
        result = make_result(run_id="r3")
        store.save_result(result)
        retrieved = store.get_result("r3")
        assert retrieved.outputs == result.outputs

    def test_retrieved_pairwise_scores_match(self, store):
        result = make_result(run_id="r4")
        store.save_result(result)
        retrieved = store.get_result("r4")
        assert retrieved.pairwise_scores == result.pairwise_scores

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_result("nonexistent") is None

    def test_get_results_by_label(self, store):
        for i in range(3):
            store.save_result(make_result(run_id=f"r{i}", agent_label="agent_x"))
        store.save_result(make_result(run_id="other", agent_label="agent_y"))
        results = store.get_results_by_label("agent_x")
        assert len(results) == 3
        for r in results:
            assert r.agent_label == "agent_x"

    def test_get_results_by_prompt_hash(self, store):
        r1 = make_result(run_id="rh1")
        r1.prompt_hash = "hash_abc"
        r2 = make_result(run_id="rh2")
        r2.prompt_hash = "hash_abc"
        r3 = make_result(run_id="rh3")
        r3.prompt_hash = "hash_xyz"
        store.save_result(r1)
        store.save_result(r2)
        store.save_result(r3)
        results = store.get_results_by_prompt_hash("hash_abc")
        assert len(results) == 2

    def test_list_results_respects_limit(self, store):
        for i in range(10):
            store.save_result(make_result(run_id=f"list-{i}"))
        results = store.list_results(limit=5)
        assert len(results) == 5

    def test_save_replaces_on_duplicate_id(self, store):
        r = make_result(run_id="dup", score=0.90, verdict="CONSISTENT")
        store.save_result(r)
        r2 = make_result(run_id="dup", score=0.50, verdict="INCONSISTENT")
        store.save_result(r2)
        retrieved = store.get_result("dup")
        assert retrieved.verdict == "INCONSISTENT"

    # ── Reports ───────────────────────────────────────────────────────────────

    def test_save_and_list_reports(self, store):
        results = [make_result(run_id=f"rep-{i}") for i in range(3)]
        report = ConsistencyReport.from_results(results, label="test_suite")
        store.save_report(report)
        reports = store.list_reports(limit=10)
        assert len(reports) >= 1
        report_ids = [r["report_id"] for r in reports]
        assert report.report_id in report_ids

    def test_save_report_also_saves_results(self, store):
        results = [make_result(run_id=f"saved-{i}") for i in range(2)]
        report = ConsistencyReport.from_results(results)
        store.save_report(report)
        for r in results:
            retrieved = store.get_result(r.run_id)
            assert retrieved is not None

    def test_list_reports_empty_initially(self, tmp_path):
        store = ConsistencyStore(str(tmp_path / "empty.db"))
        assert store.list_reports() == []
