"""Tests for TrimStore (SQLite persistence)."""

import time
import pytest
from context_trim import TrimStore, TokenBudget, TrimStrategy
from context_trim.core import TrimResult


def make_result(strategy=TrimStrategy.RECENCY_FIRST, dropped=2, within_budget=True) -> TrimResult:
    budget = TokenBudget(max_tokens=4096, reserved_tokens=512)
    return TrimResult(
        messages=[{"role": "user", "content": "hello"}],
        original_count=5,
        final_count=5 - dropped,
        original_tokens=800,
        final_tokens=800 - dropped * 100,
        strategy=strategy,
        dropped_count=dropped,
        trim_ratio=dropped * 100 / 800,
        budget=budget,
        within_budget=within_budget,
    )


def test_store_creates_db(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    import os
    assert os.path.exists(db)


def test_record_returns_id(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    row_id = store.record(make_result())
    assert isinstance(row_id, int)
    assert row_id >= 1


def test_history_empty(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    assert store.history("nonexistent") == []


def test_history_after_record(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    store.record(make_result(), pipeline_id="p1")
    history = store.history("p1")
    assert len(history) == 1
    row = history[0]
    assert row["pipeline_id"] == "p1"
    assert row["strategy"] == TrimStrategy.RECENCY_FIRST.value


def test_history_multiple(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    for _ in range(5):
        store.record(make_result(), pipeline_id="pipeline_x")
    history = store.history("pipeline_x")
    assert len(history) == 5


def test_history_limit(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    for _ in range(10):
        store.record(make_result(), pipeline_id="p2")
    history = store.history("p2", limit=3)
    assert len(history) == 3


def test_all_history(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    store.record(make_result(), pipeline_id="a")
    store.record(make_result(), pipeline_id="b")
    store.record(make_result(), pipeline_id="c")
    all_h = store.all_history()
    assert len(all_h) == 3


def test_stats_empty(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    stats = store.stats("nonexistent")
    assert stats["total_runs"] == 0


def test_stats_with_records(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    store.record(make_result(dropped=2, within_budget=True), pipeline_id="s1")
    store.record(make_result(dropped=3, within_budget=False), pipeline_id="s1")
    stats = store.stats("s1")
    assert stats["total_runs"] == 2
    assert stats["over_budget_runs"] == 1


def test_history_isolation(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    store.record(make_result(), pipeline_id="alpha")
    store.record(make_result(), pipeline_id="beta")
    assert len(store.history("alpha")) == 1
    assert len(store.history("beta")) == 1


def test_record_fields(tmp_path):
    db = str(tmp_path / "test.db")
    store = TrimStore(db_path=db)
    result = make_result(strategy=TrimStrategy.HYBRID, dropped=1)
    store.record(result, pipeline_id="field_test")
    row = store.history("field_test")[0]
    assert row["strategy"] == "hybrid"
    assert row["dropped_count"] == 1
    assert row["original_count"] == 5
    assert row["max_tokens"] == 4096
