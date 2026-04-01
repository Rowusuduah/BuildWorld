"""Tests for context_trace.store — AttributionStore SQLite persistence."""

from __future__ import annotations

import json

import pytest

from context_trace.store import AttributionStore
from tests.conftest import make_report


@pytest.fixture
def tmp_store(tmp_path):
    db = str(tmp_path / "test.db")
    store = AttributionStore(db)
    yield store
    store.close()


class TestAttributionStore:
    def test_save_returns_positive_id(self, tmp_store, sample_report):
        run_id = tmp_store.save(sample_report, label="test")
        assert run_id > 0

    def test_save_increments_id(self, tmp_store, sample_report):
        id1 = tmp_store.save(sample_report)
        id2 = tmp_store.save(sample_report)
        assert id2 > id1

    def test_count_after_saves(self, tmp_store, sample_report):
        assert tmp_store.count() == 0
        tmp_store.save(sample_report)
        assert tmp_store.count() == 1
        tmp_store.save(sample_report)
        assert tmp_store.count() == 2

    def test_list_runs_empty(self, tmp_store):
        assert tmp_store.list_runs() == []

    def test_list_runs_returns_metadata(self, tmp_store, sample_report):
        tmp_store.save(sample_report, label="run1")
        runs = tmp_store.list_runs()
        assert len(runs) == 1
        run = runs[0]
        assert "id" in run
        assert "label" in run
        assert "created_at" in run
        assert "total_api_calls" in run
        assert "top_score" in run

    def test_list_runs_label_preserved(self, tmp_store, sample_report):
        tmp_store.save(sample_report, label="my_label")
        runs = tmp_store.list_runs()
        assert runs[0]["label"] == "my_label"

    def test_list_runs_sorted_desc(self, tmp_store, sample_report):
        for i in range(3):
            tmp_store.save(sample_report, label=str(i))
        runs = tmp_store.list_runs()
        ids = [r["id"] for r in runs]
        assert ids == sorted(ids, reverse=True)

    def test_get_existing_run(self, tmp_store, sample_report):
        run_id = tmp_store.save(sample_report)
        data = tmp_store.get(run_id)
        assert data is not None
        assert "chunk_scores" in data

    def test_get_missing_run_returns_none(self, tmp_store):
        assert tmp_store.get(9999) is None

    def test_get_preserves_chunk_scores(self, tmp_store, sample_report):
        run_id = tmp_store.save(sample_report)
        data = tmp_store.get(run_id)
        assert "doc1" in data["chunk_scores"]
        assert data["chunk_scores"]["doc1"]["attribution_score"] == pytest.approx(0.85)

    def test_delete_existing_run(self, tmp_store, sample_report):
        run_id = tmp_store.save(sample_report)
        assert tmp_store.delete(run_id) is True
        assert tmp_store.get(run_id) is None
        assert tmp_store.count() == 0

    def test_delete_nonexistent_returns_false(self, tmp_store):
        assert tmp_store.delete(9999) is False

    def test_context_manager_closes(self, tmp_path, sample_report):
        db = str(tmp_path / "cm_test.db")
        with AttributionStore(db) as store:
            store.save(sample_report)
        # After context exit, connection should be None
        assert store._conn is None

    def test_list_runs_limit(self, tmp_store, sample_report):
        for _ in range(10):
            tmp_store.save(sample_report)
        runs = tmp_store.list_runs(limit=3)
        assert len(runs) == 3

    def test_top_score_stored_correctly(self, tmp_store, sample_report):
        tmp_store.save(sample_report)
        runs = tmp_store.list_runs()
        assert runs[0]["top_score"] == pytest.approx(0.85)
