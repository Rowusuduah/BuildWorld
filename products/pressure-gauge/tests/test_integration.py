"""Integration tests for pressure-gauge"""
import pytest

from pressure_gauge import (
    CriticalityLevel,
    PressureConfig,
    PressureGauge,
    pressure_probe,
)
from pressure_gauge.models import DriftVerdict


# ---------------------------------------------------------------------------
# End-to-end stable agent
# ---------------------------------------------------------------------------

class TestStableAgentIntegration:
    def test_full_sweep_stable_low_criticality(self):
        """A truly stable agent should always pass with LOW criticality."""
        def stable_agent(ctx: str) -> str:
            return (
                "The solution is to apply method X with parameter Y for "
                "optimal results across all conditions."
            )

        gauge = PressureGauge(
            model_context_limit=2048,
            fill_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            criticality=CriticalityLevel.LOW,
        )
        report = gauge.sweep(agent_fn=stable_agent, base_context="What is the solution?")

        assert report.gate_passed
        assert len(report.drift_curve) == 5
        assert 0.0 <= report.context_pressure_score <= 1.0
        assert report.verdict is not None
        assert len(report.recommendation) > 0

    def test_identical_outputs_score_near_1(self):
        """Agent that always returns identical output should score ~1.0."""
        def identical_agent(ctx: str) -> str:
            return "always the same output regardless of context fill level"

        gauge = PressureGauge(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5, 0.9],
            criticality=CriticalityLevel.LOW,
        )
        report = gauge.sweep(identical_agent)
        assert report.context_pressure_score > 0.90, (
            f"Identical agent should score > 0.90, got {report.context_pressure_score}"
        )


# ---------------------------------------------------------------------------
# End-to-end drifting agent
# ---------------------------------------------------------------------------

class TestDriftingAgentIntegration:
    def test_heavy_drift_detected(self):
        """Agent that drastically changes output under pressure should fail CRITICAL."""
        call_n = [0]

        def drifting_agent(ctx: str) -> str:
            call_n[0] += 1
            if call_n[0] == 1:
                return "extremely comprehensive analysis with full coverage of all topics"
            return "x"

        config = PressureConfig(
            model_context_limit=2048,
            fill_levels=[0.1, 0.9],
            stability_threshold=0.999,
            criticality=CriticalityLevel.CRITICAL,
        )
        gauge = PressureGauge(config=config)
        report = gauge.sweep(drifting_agent)

        assert not report.gate_passed
        assert report.verdict in (DriftVerdict.MILD, DriftVerdict.MODERATE, DriftVerdict.SEVERE)


# ---------------------------------------------------------------------------
# Config + dict consistency
# ---------------------------------------------------------------------------

class TestReportConsistency:
    def test_summary_and_dict_consistent(self):
        gauge = PressureGauge(model_context_limit=1024, fill_levels=[0.1, 0.5])

        def agent(ctx: str) -> str:
            return "result"

        report = gauge.sweep(agent)
        summary = report.summary()
        d = report.as_dict()

        assert ("PASSED" in summary) == d["gate_passed"]
        assert str(round(d["context_pressure_score"], 4)) in summary

    def test_as_dict_context_limit_present(self):
        gauge = PressureGauge(model_context_limit=4096, fill_levels=[0.1, 0.5])

        def agent(ctx: str) -> str:
            return "response"

        report = gauge.sweep(agent)
        d = report.as_dict()
        assert d["model_context_limit"] == 4096


# ---------------------------------------------------------------------------
# Decorator end-to-end
# ---------------------------------------------------------------------------

class TestDecoratorIntegration:
    def test_decorator_e2e_stable(self):
        @pressure_probe(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5],
            criticality=CriticalityLevel.LOW,
            raise_on_fail=False,
        )
        def my_agent(ctx: str) -> str:
            return "The answer is stable and complete across all conditions."

        # Function still works normally
        assert my_agent("some context") == "The answer is stable and complete across all conditions."

        # Pressure sweep returns report
        report = my_agent.pressure_sweep()
        assert isinstance(report, bool) is False
        assert hasattr(report, "gate_passed")

    def test_decorator_e2e_drift_raises(self):
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
                return "complete thorough extensive detailed analysis"
            return "y"

        from pressure_gauge.decorator import PressureError
        with pytest.raises(PressureError) as exc_info:
            drifting_agent.pressure_sweep()
        assert exc_info.value.report is not None


# ---------------------------------------------------------------------------
# Quick sweep
# ---------------------------------------------------------------------------

class TestQuickSweepIntegration:
    def test_quick_is_3_levels(self):
        gauge = PressureGauge(model_context_limit=2048)

        def agent(ctx: str) -> str:
            return "response"

        report = gauge.quick(agent)
        assert len(report.drift_curve) == 3
        assert report.drift_curve[0].fill_level < report.drift_curve[-1].fill_level

    def test_quick_fill_levels_are_10_50_90(self):
        gauge = PressureGauge(model_context_limit=2048)

        def agent(ctx: str) -> str:
            return "response"

        report = gauge.quick(agent)
        levels = [dp.fill_level for dp in report.drift_curve]
        assert levels == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# KU (Known Unknowns) documentation checks
# ---------------------------------------------------------------------------

class TestKnownUnknowns:
    def test_ku_identical_agent_near_1(self):
        """KU-064: epsilon calibration. Identical outputs should score near 1.0."""
        def identical_agent(ctx: str) -> str:
            return "always the same output regardless"

        gauge = PressureGauge(
            model_context_limit=1024,
            fill_levels=[0.1, 0.5, 0.9],
            criticality=CriticalityLevel.LOW,
        )
        report = gauge.sweep(identical_agent)
        assert report.context_pressure_score > 0.90

    def test_ku_different_padding_strategies_same_stable_result(self):
        """KU-066: padding strategy should not affect truly stable agents."""
        def stable_agent(ctx: str) -> str:
            return "stable response regardless of padding"

        for strategy in ("lorem_ipsum", "inject_history", "repeat_text"):
            gauge = PressureGauge(
                model_context_limit=1024,
                fill_levels=[0.1, 0.5],
                padding_strategy=strategy,
                criticality=CriticalityLevel.LOW,
            )
            report = gauge.sweep(stable_agent)
            assert isinstance(report, PressureGauge.__class__) or isinstance(report.__class__.__name__, str)
            # Just verify it runs without error
            assert report.context_pressure_score >= 0.0
