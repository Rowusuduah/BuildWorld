"""Tests for semantic_pass_k.runner"""
from __future__ import annotations
import pytest

from semantic_pass_k.engine import ConsistencyEngine
from semantic_pass_k.runner import ConsistencyRunner
from semantic_pass_k.models import ConsistencyReport, ConsistencyResult


# ── Agent stubs ───────────────────────────────────────────────────────────────

def _always_same(prompt: str) -> str:
    return "This is always the same output."


def _always_different(prompt: str) -> str:
    import random
    import string
    return "".join(random.choices(string.ascii_letters, k=30))


_call_count = 0

def _counting_agent(prompt: str) -> str:
    global _call_count
    _call_count += 1
    return f"output_{_call_count}"


# ── ConsistencyRunner init ────────────────────────────────────────────────────

class TestConsistencyRunnerInit:
    def test_default_k_is_5(self):
        runner = ConsistencyRunner(_always_same)
        assert runner.k == 5

    def test_custom_k_stored(self):
        runner = ConsistencyRunner(_always_same, k=3)
        assert runner.k == 3

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            ConsistencyRunner(_always_same, k=1)

    def test_k_of_1_raises(self):
        with pytest.raises(ValueError):
            ConsistencyRunner(_always_same, k=1)

    def test_default_criticality_high(self):
        runner = ConsistencyRunner(_always_same)
        assert runner.default_criticality == "HIGH"

    def test_custom_criticality_stored(self):
        runner = ConsistencyRunner(_always_same, criticality="CRITICAL")
        assert runner.default_criticality == "CRITICAL"

    def test_custom_engine_stored(self):
        engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.9)
        runner = ConsistencyRunner(_always_same, engine=engine)
        assert runner.engine is engine

    def test_default_engine_created_if_none(self):
        runner = ConsistencyRunner(_always_same)
        assert runner.engine is not None


# ── ConsistencyRunner.run ─────────────────────────────────────────────────────

class TestConsistencyRunnerRun:
    def test_returns_consistency_result(self):
        runner = ConsistencyRunner(
            _always_same,
            k=3,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.95),
        )
        result = runner.run("test prompt")
        assert isinstance(result, ConsistencyResult)

    def test_agent_called_k_times(self):
        call_count = []

        def counting(prompt):
            call_count.append(1)
            return "output"

        runner = ConsistencyRunner(counting, k=4)
        runner.run("prompt")
        assert len(call_count) == 4

    def test_k_override_per_run(self):
        call_count = []

        def counting(prompt):
            call_count.append(1)
            return "output"

        runner = ConsistencyRunner(counting, k=3)
        runner.run("prompt", k=5)
        assert len(call_count) == 5

    def test_k_override_less_than_2_raises(self):
        runner = ConsistencyRunner(_always_same, k=3)
        with pytest.raises(ValueError):
            runner.run("prompt", k=1)

    def test_criticality_override_per_run(self):
        runner = ConsistencyRunner(
            _always_same,
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.95),
        )
        result = runner.run("p", criticality="MEDIUM")
        assert result.criticality == "MEDIUM"

    def test_result_has_correct_k(self):
        runner = ConsistencyRunner(
            _always_same,
            k=3,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.95),
        )
        result = runner.run("p")
        assert result.k == 3

    def test_consistent_agent_passes(self):
        runner = ConsistencyRunner(
            _always_same,
            k=3,
            criticality="LOW",
        )
        result = runner.run("What is 2+2?")
        assert result.verdict == "CONSISTENT"

    def test_prompt_stored_in_result(self):
        runner = ConsistencyRunner(
            _always_same,
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9),
        )
        result = runner.run("my special prompt")
        assert result.prompt == "my special prompt"


# ── ConsistencyRunner.run_batch ───────────────────────────────────────────────

class TestConsistencyRunnerRunBatch:
    def test_returns_consistency_report(self):
        runner = ConsistencyRunner(
            _always_same,
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.95),
        )
        report = runner.run_batch(["p1", "p2"])
        assert isinstance(report, ConsistencyReport)

    def test_report_has_correct_total_results(self):
        runner = ConsistencyRunner(
            _always_same,
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.95),
        )
        report = runner.run_batch(["p1", "p2", "p3"])
        assert report.total_results == 3

    def test_empty_prompts_raises(self):
        runner = ConsistencyRunner(_always_same, k=2)
        with pytest.raises(ValueError):
            runner.run_batch([])

    def test_label_stored_in_report(self):
        runner = ConsistencyRunner(
            _always_same,
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9),
        )
        report = runner.run_batch(["p1"], label="my_suite")
        assert report.label == "my_suite"

    def test_criticality_override_applies_to_all(self):
        runner = ConsistencyRunner(
            _always_same,
            k=2,
            engine=ConsistencyEngine(similarity_fn=lambda a, b: 0.9),
        )
        report = runner.run_batch(["p1", "p2"], criticality="MEDIUM")
        for r in report.results:
            assert r.criticality == "MEDIUM"

    def test_k_override_applies_to_all(self):
        call_counts = []

        def counting(prompt):
            call_counts.append(1)
            return "output"

        runner = ConsistencyRunner(counting, k=3)
        runner.run_batch(["p1", "p2"], k=2)
        # 2 prompts × 2 calls each = 4
        assert len(call_counts) == 4
