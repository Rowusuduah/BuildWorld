"""
Tests for livelock_probe.decorator
"""
import pytest

from livelock_probe.decorator import LivelockError, livelock_probe_decorator
from livelock_probe.models import ProgressConfig


class TestLivelockProbeDecorator:
    def test_basic_decoration(self):
        @livelock_probe_decorator(goal="resolve issue", k=3)
        def agent_step(n):
            return f"step output {n}"

        result = agent_step(0)
        assert result == "step output 0"

    def test_steps_recorded(self):
        @livelock_probe_decorator(goal="resolve issue", k=3)
        def agent_step(n):
            return f"output {n}"

        for i in range(5):
            agent_step(i)

        assert agent_step._livelock_suite.step_count() == 5

    def test_suite_attached_to_wrapper(self):
        @livelock_probe_decorator(goal="resolve issue", k=3)
        def my_agent(x):
            return x

        assert hasattr(my_agent, "_livelock_suite")

    def test_non_string_return_recorded(self):
        """Non-string return values are stringified before recording."""
        @livelock_probe_decorator(goal="resolve issue", k=3)
        def agent_step(n):
            return {"status": "ok", "n": n}

        agent_step(1)
        assert agent_step._livelock_suite.step_count() == 1

    def test_report_accessible_after_run(self):
        sims = [0.1, 0.3, 0.5, 0.7, 0.9]
        seq = iter(sims)

        @livelock_probe_decorator(
            goal="resolve issue", k=3,
        )
        def agent_step(n):
            return f"step {n}"

        # Inject custom similarity into the suite's engine after decoration
        agent_step._livelock_suite._engine._similarity_fn = lambda a, b: next(seq)

        for i in range(5):
            agent_step(i)

        report = agent_step._livelock_suite.compute()
        assert report.total_steps == 5

    def test_raise_on_livelock_raises_error(self):
        """When raise_on_livelock=True and budget exceeded in livelock, raises LivelockError."""
        # Fixed sim → all deltas near 0 after step 0 → livelock
        @livelock_probe_decorator(
            goal="resolve issue",
            k=2,
            budget_steps=3,
            raise_on_livelock=True,
        )
        def stuck_agent(n):
            return "same output every time"

        # Inject sticky similarity
        stuck_agent._livelock_suite._engine._similarity_fn = lambda a, b: 0.5

        # First two calls are fine (budget_steps=3 means after 3rd call it checks)
        stuck_agent(0)
        stuck_agent(1)

        # Third call triggers budget check → livelock detected → raises
        with pytest.raises(LivelockError, match="livelock state"):
            stuck_agent(2)

    def test_functools_wraps_preserves_name(self):
        @livelock_probe_decorator(goal="resolve issue", k=3)
        def my_special_agent(x):
            return x

        assert my_special_agent.__name__ == "my_special_agent"

    def test_custom_agent_label(self):
        @livelock_probe_decorator(goal="resolve issue", k=3, agent_label="custom-agent")
        def agent(x):
            return x

        agent("test")
        report = agent._livelock_suite.compute()
        assert report.agent_label == "custom-agent"


class TestLivelockError:
    def test_is_exception(self):
        assert issubclass(LivelockError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LivelockError):
            raise LivelockError("agent is stuck")
