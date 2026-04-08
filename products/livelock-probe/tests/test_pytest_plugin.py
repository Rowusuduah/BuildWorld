"""
Tests for livelock_probe.pytest_plugin (via the livelock_suite fixture).
"""
import pytest

from livelock_probe.suite import LivelockSuite


class TestLivelockSuiteFixture:
    def test_fixture_returns_factory(self, livelock_suite):
        suite = livelock_suite(goal="resolve issue", k=3)
        assert isinstance(suite, LivelockSuite)

    def test_fixture_factory_creates_independent_suites(self, livelock_suite):
        suite_a = livelock_suite(goal="task A", k=3)
        suite_b = livelock_suite(goal="task B", k=5)
        assert suite_a is not suite_b

    def test_suite_from_fixture_records_steps(self, livelock_suite):
        suite = livelock_suite(goal="resolve error", k=3)
        suite.record_steps(["found error", "fixing error", "error fixed"])
        assert suite.step_count() == 3

    def test_suite_from_fixture_computes_report(self, livelock_suite):
        sims = [0.2, 0.5, 0.8]
        seq = iter(sims)
        suite = livelock_suite(goal="resolve error", k=2)
        suite._engine._similarity_fn = lambda a, b: next(seq)
        suite.record_steps(["a", "b", "c"])
        report = suite.compute()
        assert report.total_steps == 3
        assert report.goal == "resolve error"

    def test_no_livelock_assertion_passes(self, livelock_suite):
        suite = livelock_suite(goal="resolve DB connection error", k=3)
        # Inject increasing progress
        sims = [0.2, 0.4, 0.6, 0.8, 1.0]
        seq = iter(sims)
        suite._engine._similarity_fn = lambda a, b: next(seq)
        suite.record_steps(["init", "found issue", "applying fix", "testing", "resolved"])
        report = suite.compute()
        assert report.gate_passed, (
            f"Unexpected livelock: LivelockScore={report.livelock_score:.3f}"
        )

    def test_livelock_detected_in_test(self, livelock_suite):
        suite = livelock_suite(goal="fix error", k=3)
        # Inject flat progress → livelock
        suite._engine._similarity_fn = lambda a, b: 0.5
        suite.record_steps(["retry", "retry", "retry", "retry"])
        report = suite.compute()
        assert report.livelock_detected is True
