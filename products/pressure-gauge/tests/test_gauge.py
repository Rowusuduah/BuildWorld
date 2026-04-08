"""Tests for pressure_gauge.gauge.PressureGauge"""
import pytest

from pressure_gauge.gauge import PressureGauge
from pressure_gauge.models import CriticalityLevel, PressureConfig, PressureReport


def stable_agent(ctx: str) -> str:
    return "The result is 42. Complete and thorough analysis provided."


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestPressureGaugeInit:
    def test_default_init(self):
        gauge = PressureGauge()
        assert gauge.config is not None
        assert gauge.config.model_context_limit == 8192

    def test_config_init(self):
        config = PressureConfig(model_context_limit=4096)
        gauge = PressureGauge(config=config)
        assert gauge.config.model_context_limit == 4096

    def test_kwargs_context_limit(self):
        gauge = PressureGauge(model_context_limit=2048)
        assert gauge.config.model_context_limit == 2048

    def test_kwargs_criticality(self):
        gauge = PressureGauge(criticality=CriticalityLevel.LOW)
        assert gauge.config.criticality == CriticalityLevel.LOW

    def test_kwargs_fill_levels(self):
        gauge = PressureGauge(fill_levels=[0.2, 0.6, 1.0])
        assert 1.0 in gauge.config.fill_levels

    def test_kwargs_stability_threshold(self):
        gauge = PressureGauge(stability_threshold=0.70)
        assert gauge.config.stability_threshold == 0.70

    def test_config_takes_precedence_over_kwargs(self):
        config = PressureConfig(model_context_limit=1234)
        gauge = PressureGauge(config=config, model_context_limit=9999)
        assert gauge.config.model_context_limit == 1234

    def test_use_neural_default_false(self):
        gauge = PressureGauge()
        assert gauge._use_neural is False

    def test_use_neural_can_be_set(self):
        gauge = PressureGauge(use_neural=True)
        assert gauge._use_neural is True


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

class TestPressureGaugeSweep:
    def test_returns_pressure_report(self):
        gauge = PressureGauge(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = gauge.sweep(agent_fn=stable_agent)
        assert isinstance(report, PressureReport)

    def test_stable_agent_low_criticality_passes(self):
        gauge = PressureGauge(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5],
            criticality=CriticalityLevel.LOW,
        )
        report = gauge.sweep(stable_agent)
        assert report.gate_passed

    def test_base_context_injected(self):
        received = []

        def capturing(ctx: str) -> str:
            received.append(ctx)
            return "done"

        gauge = PressureGauge(model_context_limit=2048, fill_levels=[0.1, 0.5])
        gauge.sweep(agent_fn=capturing, base_context="MARKER_TEXT")
        assert all("MARKER_TEXT" in c for c in received)

    def test_summary_contains_score(self):
        gauge = PressureGauge(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = gauge.sweep(stable_agent)
        assert "ContextPressureScore" in report.summary()

    def test_as_dict_has_required_keys(self):
        gauge = PressureGauge(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = gauge.sweep(stable_agent)
        d = report.as_dict()
        for key in ("context_pressure_score", "drift_curve", "gate_passed", "verdict"):
            assert key in d

    def test_drift_curve_length(self):
        gauge = PressureGauge(model_context_limit=2048, fill_levels=[0.1, 0.3, 0.5, 0.7])
        report = gauge.sweep(stable_agent)
        assert len(report.drift_curve) == 4

    def test_agent_called_once_per_level(self):
        call_count = [0]

        def counting_agent(ctx: str) -> str:
            call_count[0] += 1
            return "response"

        gauge = PressureGauge(model_context_limit=2048, fill_levels=[0.1, 0.5, 0.9])
        gauge.sweep(counting_agent)
        assert call_count[0] == 3

    def test_multiple_runs_per_level(self):
        call_count = [0]

        def counting_agent(ctx: str) -> str:
            call_count[0] += 1
            return "response"

        config = PressureConfig(
            model_context_limit=2048,
            fill_levels=[0.1, 0.5],
            runs_per_level=3,
        )
        gauge = PressureGauge(config=config)
        gauge.sweep(counting_agent)
        assert call_count[0] == 6  # 2 levels × 3 runs


# ---------------------------------------------------------------------------
# quick
# ---------------------------------------------------------------------------

class TestPressureGaugeQuick:
    def test_returns_report(self):
        gauge = PressureGauge(model_context_limit=2048)
        report = gauge.quick(agent_fn=stable_agent)
        assert isinstance(report, PressureReport)

    def test_default_3_fill_levels(self):
        gauge = PressureGauge(model_context_limit=2048)
        report = gauge.quick(agent_fn=stable_agent)
        assert len(report.drift_curve) == 3

    def test_custom_fill_levels(self):
        gauge = PressureGauge(model_context_limit=2048)
        report = gauge.quick(agent_fn=stable_agent, fill_levels=[0.2, 0.8])
        assert len(report.drift_curve) == 2

    def test_base_context_injected(self):
        received = []

        def capturing(ctx: str) -> str:
            received.append(ctx)
            return "result"

        gauge = PressureGauge(model_context_limit=2048)
        gauge.quick(capturing, base_context="QUICK_MARKER")
        assert all("QUICK_MARKER" in c for c in received)


# ---------------------------------------------------------------------------
# estimate_onset
# ---------------------------------------------------------------------------

class TestEstimateOnset:
    def test_returns_none_for_stable_low_threshold(self):
        gauge = PressureGauge(
            model_context_limit=2048,
            stability_threshold=0.10,
            criticality=CriticalityLevel.LOW,
        )
        onset = gauge.estimate_onset(stable_agent, granularity=5)
        assert onset is None

    def test_returns_int_or_none(self):
        gauge = PressureGauge(model_context_limit=2048)
        onset = gauge.estimate_onset(stable_agent, granularity=3)
        assert onset is None or isinstance(onset, int)

    def test_onset_positive_when_detected(self):
        call_n = [0]

        def drifting(ctx: str) -> str:
            call_n[0] += 1
            if call_n[0] == 1:
                return "thorough comprehensive detailed analysis"
            return "x"

        gauge = PressureGauge(
            model_context_limit=2048,
            stability_threshold=0.999,
            criticality=CriticalityLevel.CRITICAL,
        )
        onset = gauge.estimate_onset(drifting, granularity=3)
        if onset is not None:
            assert onset > 0
