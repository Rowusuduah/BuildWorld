"""Tests for pressure_gauge.decorator"""
import pytest

from pressure_gauge.decorator import PressureError, pressure_probe
from pressure_gauge.gauge import PressureGauge
from pressure_gauge.models import CriticalityLevel, PressureReport


# ---------------------------------------------------------------------------
# @pressure_probe decorator
# ---------------------------------------------------------------------------

class TestPressureProbeDecorator:
    def test_decorated_function_still_works(self):
        @pressure_probe(model_context_limit=2048)
        def my_agent(ctx: str) -> str:
            return "normal result"

        assert my_agent("hello") == "normal result"

    def test_adds_pressure_sweep_attribute(self):
        @pressure_probe(model_context_limit=2048)
        def my_agent(ctx: str) -> str:
            return "result"

        assert hasattr(my_agent, "pressure_sweep")
        assert callable(my_agent.pressure_sweep)

    def test_adds_pressure_gauge_attribute(self):
        @pressure_probe(model_context_limit=2048)
        def my_agent(ctx: str) -> str:
            return "result"

        assert hasattr(my_agent, "_pressure_gauge")
        assert isinstance(my_agent._pressure_gauge, PressureGauge)

    def test_pressure_sweep_returns_report(self):
        @pressure_probe(model_context_limit=2048, fill_levels=[0.1, 0.5])
        def my_agent(ctx: str) -> str:
            return "Complete analysis of the task at hand."

        report = my_agent.pressure_sweep()
        assert isinstance(report, PressureReport)

    def test_stable_agent_passes_low_criticality(self):
        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5],
            criticality=CriticalityLevel.LOW,
            raise_on_fail=False,
        )
        def my_agent(ctx: str) -> str:
            return "Complete stable answer to the problem."

        report = my_agent.pressure_sweep()
        assert report.gate_passed

    def test_raise_on_fail_true_raises_pressure_error(self):
        call_n = [0]

        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.1, 0.9],
            stability_threshold=0.999,
            criticality=CriticalityLevel.CRITICAL,
            raise_on_fail=True,
        )
        def drifting_agent(ctx: str) -> str:
            call_n[0] += 1
            if call_n[0] == 1:
                return "very thorough complete analysis with many extensive details"
            return "x"

        with pytest.raises(PressureError):
            drifting_agent.pressure_sweep()

    def test_raise_on_fail_false_no_exception(self):
        call_n = [0]

        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.1, 0.9],
            stability_threshold=0.999,
            raise_on_fail=False,
        )
        def drifting_agent(ctx: str) -> str:
            call_n[0] += 1
            if call_n[0] == 1:
                return "very thorough complete analysis"
            return "x"

        report = drifting_agent.pressure_sweep()
        assert isinstance(report, PressureReport)

    def test_preserves_function_name(self):
        @pressure_probe(model_context_limit=2048)
        def named_function(ctx: str) -> str:
            return "result"

        assert named_function.__name__ == "named_function"

    def test_preserves_function_docstring(self):
        @pressure_probe(model_context_limit=2048)
        def documented_agent(ctx: str) -> str:
            """My documented agent."""
            return "result"

        assert documented_agent.__doc__ == "My documented agent."

    def test_base_context_default_used(self):
        received = []

        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5],
            base_context="DEFAULT_CONTEXT",
        )
        def capturing_agent(ctx: str) -> str:
            received.append(ctx)
            return "done"

        capturing_agent.pressure_sweep()
        assert all("DEFAULT_CONTEXT" in c for c in received)

    def test_base_context_override_works(self):
        received = []

        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5],
            base_context="DEFAULT",
        )
        def capturing_agent(ctx: str) -> str:
            received.append(ctx)
            return "done"

        received.clear()
        capturing_agent.pressure_sweep(ctx="OVERRIDE_CONTEXT")
        assert all("OVERRIDE_CONTEXT" in c for c in received)

    def test_custom_fill_levels(self):
        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.2, 0.8],
        )
        def my_agent(ctx: str) -> str:
            return "result"

        report = my_agent.pressure_sweep()
        assert len(report.drift_curve) == 2


# ---------------------------------------------------------------------------
# PressureError
# ---------------------------------------------------------------------------

class TestPressureError:
    def _make_failed_report(self) -> PressureReport:
        from pressure_gauge.models import DriftVerdict, PressureConfig
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        return PressureReport(
            config=config,
            drift_curve=[],
            context_pressure_score=0.4,
            pressure_onset_token=500,
            verdict=DriftVerdict.SEVERE,
            gate_passed=False,
            recommendation="Context pressure detected.",
        )

    def test_has_report_attribute(self):
        report = self._make_failed_report()
        error = PressureError(report)
        assert error.report is report

    def test_message_contains_failed(self):
        report = self._make_failed_report()
        error = PressureError(report)
        assert "FAILED" in str(error)

    def test_is_exception(self):
        report = self._make_failed_report()
        error = PressureError(report)
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        report = self._make_failed_report()
        with pytest.raises(PressureError) as exc_info:
            raise PressureError(report)
        assert exc_info.value.report is report
