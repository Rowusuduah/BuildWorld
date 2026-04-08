"""
Tests for livelock_probe.suite
"""
import pytest

from livelock_probe.models import ProgressConfig
from livelock_probe.suite import LivelockSuite


def make_suite(goal="resolve issue", k=3, epsilon=0.05, sim_val=0.5):
    """Factory for test LivelockSuites with a fixed similarity function."""
    config = ProgressConfig(
        goal=goal, k=k, epsilon=epsilon,
        similarity_fn=lambda a, b: sim_val,
    )
    return LivelockSuite(config)


class TestLivelockSuiteRecording:
    def test_record_single_step(self):
        suite = make_suite()
        suite.record_step("output 1")
        assert suite.step_count() == 1

    def test_record_multiple_steps(self):
        suite = make_suite()
        suite.record_steps(["a", "b", "c"])
        assert suite.step_count() == 3

    def test_explicit_step_id_matches_index(self):
        suite = make_suite()
        suite.record_step("output", step_id=0)
        assert suite.step_count() == 1

    def test_explicit_step_id_mismatch_raises(self):
        suite = make_suite()
        suite.record_step("output 0", step_id=0)
        with pytest.raises(ValueError, match="step_id"):
            suite.record_step("output bad", step_id=5)

    def test_step_id_none_is_automatic(self):
        suite = make_suite()
        suite.record_step("a")
        suite.record_step("b")
        suite.record_step("c")
        assert suite.step_count() == 3

    def test_reset_clears_steps(self):
        suite = make_suite()
        suite.record_steps(["a", "b", "c"])
        suite.reset()
        assert suite.step_count() == 0

    def test_reset_clears_cached_report(self):
        suite = make_suite()
        suite.record_steps(["a", "b"])
        suite.compute()
        assert suite.last_report() is not None
        suite.reset()
        assert suite.last_report() is None


class TestLivelockSuiteCompute:
    def test_compute_returns_report(self):
        suite = make_suite()
        suite.record_steps(["a", "b", "c"])
        report = suite.compute()
        assert report.total_steps == 3

    def test_compute_no_steps_raises(self):
        suite = make_suite()
        with pytest.raises(ValueError, match="No steps"):
            suite.compute()

    def test_compute_caches_report(self):
        suite = make_suite()
        suite.record_steps(["a", "b", "c"])
        report1 = suite.compute()
        report2 = suite.compute()
        assert report1.report_id == report2.report_id

    def test_new_step_invalidates_cache(self):
        suite = make_suite()
        suite.record_steps(["a", "b", "c"])
        report1 = suite.compute()
        suite.record_step("d")
        suite.compute()  # recomputes
        # After new step, a fresh compute is triggered
        assert suite.last_report() is not None
        assert suite.last_report().total_steps == 4

    def test_gate_passes_when_progressing(self):
        # Increasing similarity → low livelock score → gate passes
        sims = [0.1, 0.4, 0.7]
        seq = iter(sims)
        config = ProgressConfig(
            goal="resolve issue", k=2,
            similarity_fn=lambda a, b: next(seq),
        )
        suite = LivelockSuite(config)
        suite.record_steps(["x", "y", "z"])
        assert suite.gate() is True

    def test_gate_fails_on_livelock(self):
        # Fixed similarity → deltas all ~0 → livelock
        sims = [0.3, 0.3, 0.3, 0.3, 0.3]
        seq = iter(sims)
        config = ProgressConfig(
            goal="resolve issue", k=3,
            similarity_fn=lambda a, b: next(seq),
        )
        suite = LivelockSuite(config)
        suite.record_steps(["retry"] * 5)
        assert suite.gate() is False


class TestLivelockSuiteBudget:
    def test_is_over_budget_false_initially(self):
        config = ProgressConfig(goal="task", budget_steps=5)
        suite = LivelockSuite(config)
        assert suite.is_over_budget() is False

    def test_is_over_budget_true_when_exceeded(self):
        config = ProgressConfig(goal="task", budget_steps=3)
        suite = LivelockSuite(config)
        suite.record_steps(["a", "b", "c"])
        assert suite.is_over_budget() is True

    def test_budget_not_enforced_by_suite(self):
        """Suite does not stop recording at budget; it just reports over-budget."""
        config = ProgressConfig(goal="task", budget_steps=2)
        suite = LivelockSuite(config)
        suite.record_steps(["a", "b", "c", "d", "e"])
        assert suite.step_count() == 5


class TestLivelockSuiteMonitor:
    def test_context_manager_records_steps(self):
        suite = make_suite()
        with suite.monitor() as m:
            m.record("step 1")
            m.record("step 2")
            m.record("step 3")
        assert suite.step_count() == 3

    def test_context_manager_compute_after(self):
        suite = make_suite()
        with suite.monitor() as m:
            for output in ["a", "b", "c"]:
                m.record(output)
        report = suite.compute()
        assert report.total_steps == 3

    def test_context_manager_exception_preserves_steps(self):
        suite = make_suite()
        try:
            with suite.monitor() as m:
                m.record("step 1")
                raise RuntimeError("agent crashed")
        except RuntimeError:
            pass
        # Steps recorded before exception should be preserved
        assert suite.step_count() == 1
