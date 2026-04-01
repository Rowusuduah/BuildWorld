"""
Tests for cot_fidelity.runner
"""
from __future__ import annotations

import json
import sys
import pytest

from cot_fidelity.models import FidelityBatchReport, FidelityResult
from cot_fidelity.runner import FidelityRunner


# ── Stubs ─────────────────────────────────────────────────────────────────────

ALWAYS_FAITHFUL = lambda a, b: 0.50
ALWAYS_UNFAITHFUL = lambda a, b: 0.97


def make_runner(similarity_fn=None, **kwargs):
    fn = similarity_fn or ALWAYS_FAITHFUL
    return FidelityRunner(similarity_fn=fn, **kwargs)


def _make_results(verdicts):
    from cot_fidelity.engine import FidelityEngine
    e = FidelityEngine(similarity_fn=ALWAYS_FAITHFUL)
    return [
        FidelityResult(
            prompt=f"prompt {i}",
            full_output="out",
            suppressed_output="other",
            cot_chain="chain",
            similarity=0.5 if v == "FAITHFUL" else 0.97,
            faithfulness_score=0.5 if v == "FAITHFUL" else 0.03,
            verdict=v,
            faithful_threshold=0.15,
            unfaithful_threshold=0.08,
        )
        for i, v in enumerate(verdicts)
    ]


# ── FidelityRunner.test ───────────────────────────────────────────────────────

class TestFidelityRunnerTest:
    def test_returns_fidelity_result(self):
        runner = make_runner()
        r = runner.test("prompt", "cot", "with_out", "without_out")
        assert isinstance(r, FidelityResult)

    def test_verdict_faithful(self):
        runner = make_runner(similarity_fn=ALWAYS_FAITHFUL)
        r = runner.test("prompt", "cot", "a", "b")
        assert r.verdict == "FAITHFUL"

    def test_verdict_unfaithful(self):
        runner = make_runner(similarity_fn=ALWAYS_UNFAITHFUL)
        r = runner.test("prompt", "cot", "a", "b")
        assert r.verdict == "UNFAITHFUL"

    def test_stores_result_when_store_provided(self, tmp_path):
        from cot_fidelity.store import FidelityStore
        store = FidelityStore(db_path=tmp_path / "test.db")
        runner = make_runner(store=store)
        runner.test("prompt", "cot", "a", "b")
        assert store.count() == 1

    def test_no_store_does_not_raise(self):
        runner = make_runner(store=None)
        r = runner.test("prompt", "cot", "a", "b")
        assert r is not None

    def test_model_version_passed_to_store(self, tmp_path):
        from cot_fidelity.store import FidelityStore
        store = FidelityStore(db_path=tmp_path / "test.db")
        runner = make_runner(store=store, model_version="claude-3-7")
        runner.test("prompt", "cot", "a", "b")
        results = store.recent(1)
        assert len(results) == 1


# ── FidelityRunner.test_batch ─────────────────────────────────────────────────

class TestFidelityRunnerBatch:
    def test_returns_batch_report(self):
        runner = make_runner()
        report = runner.test_batch(
            prompts=["p1", "p2"],
            cot_chains=["c1", "c2"],
            with_cot_outputs=["w1", "w2"],
            without_cot_outputs=["s1", "s2"],
        )
        assert isinstance(report, FidelityBatchReport)
        assert report.total == 2

    def test_batch_stores_all_results(self, tmp_path):
        from cot_fidelity.store import FidelityStore
        store = FidelityStore(db_path=tmp_path / "test.db")
        runner = make_runner(store=store)
        runner.test_batch(
            prompts=["p1", "p2", "p3"],
            cot_chains=["c1", "c2", "c3"],
            with_cot_outputs=["w1", "w2", "w3"],
            without_cot_outputs=["s1", "s2", "s3"],
        )
        assert store.count() == 3

    def test_batch_faithful_rate(self):
        runner = make_runner(similarity_fn=ALWAYS_FAITHFUL)
        report = runner.test_batch(
            prompts=["p1", "p2"],
            cot_chains=["c1", "c2"],
            with_cot_outputs=["w1", "w2"],
            without_cot_outputs=["s1", "s2"],
        )
        assert report.faithfulness_rate == 1.0


# ── FidelityRunner.ci_gate ────────────────────────────────────────────────────

class TestCIGate:
    def test_passes_when_rate_above_threshold(self):
        runner = make_runner()
        results = _make_results(["FAITHFUL", "FAITHFUL", "FAITHFUL"])
        assert runner.ci_gate(results, min_faithfulness_rate=0.5) is True

    def test_fails_when_rate_below_threshold(self):
        runner = make_runner()
        results = _make_results(["UNFAITHFUL", "UNFAITHFUL", "FAITHFUL"])
        with pytest.raises(SystemExit) as exc_info:
            runner.ci_gate(results, min_faithfulness_rate=0.8)
        assert exc_info.value.code == 1

    def test_passes_empty_results(self):
        runner = make_runner()
        assert runner.ci_gate([]) is True

    def test_fail_on_unfaithful_flag(self):
        runner = make_runner()
        results = _make_results(["FAITHFUL", "UNFAITHFUL"])
        with pytest.raises(SystemExit):
            runner.ci_gate(results, fail_on_unfaithful=True)

    def test_all_unfaithful_below_threshold(self):
        runner = make_runner()
        results = _make_results(["UNFAITHFUL", "UNFAITHFUL"])
        with pytest.raises(SystemExit):
            runner.ci_gate(results, min_faithfulness_rate=0.5)

    def test_json_output_format(self, capsys):
        runner = make_runner()
        results = _make_results(["FAITHFUL"])
        runner.ci_gate(results, output_format="json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "PASS"

    def test_text_output_format(self, capsys):
        runner = make_runner()
        results = _make_results(["FAITHFUL"])
        runner.ci_gate(results, output_format="text")
        captured = capsys.readouterr()
        assert "PASS" in captured.out


# ── FidelityRunner.report ─────────────────────────────────────────────────────

class TestFidelityRunnerReport:
    def test_markdown_report(self):
        runner = make_runner()
        results = _make_results(["FAITHFUL", "UNFAITHFUL"])
        md = runner.report(results, output_format="markdown")
        assert "FidelityBatchReport" in md

    def test_json_report(self):
        runner = make_runner()
        results = _make_results(["FAITHFUL"])
        j = runner.report(results, output_format="json")
        data = json.loads(j)
        assert data["total"] == 1
