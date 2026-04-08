"""Tests for pressure_gauge.models"""
import pytest

from pressure_gauge.models import (
    CriticalityLevel,
    DriftPoint,
    DriftVerdict,
    PressureConfig,
    PressureReport,
    PRESSURE_THRESHOLDS,
    get_threshold,
    score_to_verdict,
)


# ---------------------------------------------------------------------------
# CriticalityLevel
# ---------------------------------------------------------------------------

class TestCriticalityLevel:
    def test_critical_value(self):
        assert CriticalityLevel.CRITICAL == "CRITICAL"

    def test_high_value(self):
        assert CriticalityLevel.HIGH == "HIGH"

    def test_medium_value(self):
        assert CriticalityLevel.MEDIUM == "MEDIUM"

    def test_low_value(self):
        assert CriticalityLevel.LOW == "LOW"

    def test_from_string(self):
        assert CriticalityLevel("HIGH") == CriticalityLevel.HIGH

    def test_all_four_levels(self):
        levels = list(CriticalityLevel)
        assert len(levels) == 4

    def test_all_levels_in_thresholds(self):
        for level in CriticalityLevel:
            assert level in PRESSURE_THRESHOLDS


# ---------------------------------------------------------------------------
# PRESSURE_THRESHOLDS
# ---------------------------------------------------------------------------

class TestPressureThresholds:
    def test_critical_highest(self):
        assert (
            PRESSURE_THRESHOLDS[CriticalityLevel.CRITICAL]
            > PRESSURE_THRESHOLDS[CriticalityLevel.HIGH]
        )

    def test_high_above_medium(self):
        assert (
            PRESSURE_THRESHOLDS[CriticalityLevel.HIGH]
            > PRESSURE_THRESHOLDS[CriticalityLevel.MEDIUM]
        )

    def test_medium_above_low(self):
        assert (
            PRESSURE_THRESHOLDS[CriticalityLevel.MEDIUM]
            > PRESSURE_THRESHOLDS[CriticalityLevel.LOW]
        )

    def test_all_between_0_and_1(self):
        for v in PRESSURE_THRESHOLDS.values():
            assert 0.0 < v <= 1.0

    def test_critical_at_least_0_90(self):
        assert PRESSURE_THRESHOLDS[CriticalityLevel.CRITICAL] >= 0.90

    def test_low_at_least_0_50(self):
        assert PRESSURE_THRESHOLDS[CriticalityLevel.LOW] >= 0.50


# ---------------------------------------------------------------------------
# get_threshold
# ---------------------------------------------------------------------------

class TestGetThreshold:
    def test_returns_correct_value(self):
        for level in CriticalityLevel:
            assert get_threshold(level) == PRESSURE_THRESHOLDS[level]

    def test_critical_highest(self):
        assert get_threshold(CriticalityLevel.CRITICAL) > get_threshold(
            CriticalityLevel.HIGH
        )

    def test_returns_float(self):
        assert isinstance(get_threshold(CriticalityLevel.HIGH), float)


# ---------------------------------------------------------------------------
# score_to_verdict
# ---------------------------------------------------------------------------

class TestScoreToVerdict:
    def test_perfect_score_stable(self):
        verdict = score_to_verdict(1.0, CriticalityLevel.HIGH)
        assert verdict == DriftVerdict.STABLE

    def test_exactly_at_threshold_stable(self):
        t = get_threshold(CriticalityLevel.HIGH)
        verdict = score_to_verdict(t, CriticalityLevel.HIGH)
        assert verdict == DriftVerdict.STABLE

    def test_just_below_threshold_mild(self):
        t = get_threshold(CriticalityLevel.HIGH)
        verdict = score_to_verdict(t - 0.05, CriticalityLevel.HIGH)
        assert verdict == DriftVerdict.MILD

    def test_moderate_drift(self):
        t = get_threshold(CriticalityLevel.HIGH)
        verdict = score_to_verdict(t - 0.15, CriticalityLevel.HIGH)
        assert verdict == DriftVerdict.MODERATE

    def test_far_below_severe(self):
        t = get_threshold(CriticalityLevel.HIGH)
        verdict = score_to_verdict(t - 0.35, CriticalityLevel.HIGH)
        assert verdict == DriftVerdict.SEVERE

    def test_zero_score_severe(self):
        for level in CriticalityLevel:
            verdict = score_to_verdict(0.0, level)
            assert verdict == DriftVerdict.SEVERE

    def test_critical_passes_at_1(self):
        verdict = score_to_verdict(1.0, CriticalityLevel.CRITICAL)
        assert verdict == DriftVerdict.STABLE

    def test_low_threshold_lower(self):
        # Score that fails HIGH should pass LOW if above LOW threshold
        t_low = get_threshold(CriticalityLevel.LOW)
        verdict = score_to_verdict(t_low, CriticalityLevel.LOW)
        assert verdict == DriftVerdict.STABLE


# ---------------------------------------------------------------------------
# PressureConfig
# ---------------------------------------------------------------------------

class TestPressureConfig:
    def test_defaults(self):
        config = PressureConfig()
        assert config.model_context_limit == 8192
        assert len(config.fill_levels) == 5
        assert config.criticality == CriticalityLevel.HIGH
        assert config.padding_strategy == "lorem_ipsum"
        assert config.chars_per_token == 4.0
        assert config.runs_per_level == 1

    def test_custom_fill_levels(self):
        config = PressureConfig(fill_levels=[0.1, 0.5, 0.9])
        assert config.fill_levels == [0.1, 0.5, 0.9]

    def test_fill_levels_sorted(self):
        config = PressureConfig(fill_levels=[0.9, 0.1, 0.5])
        assert config.fill_levels == [0.1, 0.5, 0.9]

    def test_fill_levels_deduplicated(self):
        config = PressureConfig(fill_levels=[0.1, 0.1, 0.5, 0.5])
        assert len(config.fill_levels) == 2

    def test_baseline_fill_level(self):
        config = PressureConfig(fill_levels=[0.2, 0.5, 0.8])
        assert config.baseline_fill_level == 0.2

    def test_tokens_for_level(self):
        config = PressureConfig(model_context_limit=10000)
        assert config.tokens_for_level(0.5) == 5000
        assert config.tokens_for_level(0.1) == 1000

    def test_tokens_for_level_truncates(self):
        config = PressureConfig(model_context_limit=8192)
        result = config.tokens_for_level(0.3)
        assert isinstance(result, int)

    def test_invalid_empty_fill_levels(self):
        with pytest.raises(ValueError, match="fill_levels"):
            PressureConfig(fill_levels=[])

    def test_invalid_fill_level_above_1(self):
        with pytest.raises(ValueError):
            PressureConfig(fill_levels=[0.5, 1.5])

    def test_invalid_fill_level_zero(self):
        with pytest.raises(ValueError):
            PressureConfig(fill_levels=[0.0, 0.5])

    def test_invalid_context_limit_zero(self):
        with pytest.raises(ValueError):
            PressureConfig(model_context_limit=0)

    def test_invalid_context_limit_negative(self):
        with pytest.raises(ValueError):
            PressureConfig(model_context_limit=-100)

    def test_invalid_stability_threshold_zero(self):
        with pytest.raises(ValueError):
            PressureConfig(stability_threshold=0.0)

    def test_invalid_stability_threshold_above_1(self):
        with pytest.raises(ValueError):
            PressureConfig(stability_threshold=1.1)

    def test_invalid_padding_strategy(self):
        with pytest.raises(ValueError, match="padding_strategy"):
            PressureConfig(padding_strategy="neural_magic")

    def test_invalid_runs_per_level(self):
        with pytest.raises(ValueError):
            PressureConfig(runs_per_level=0)

    def test_valid_strategies(self):
        for s in ("lorem_ipsum", "repeat_text", "inject_history"):
            config = PressureConfig(padding_strategy=s)
            assert config.padding_strategy == s

    def test_criticality_critical(self):
        config = PressureConfig(criticality=CriticalityLevel.CRITICAL)
        assert config.criticality == CriticalityLevel.CRITICAL

    def test_fill_level_exactly_1(self):
        config = PressureConfig(fill_levels=[0.5, 1.0])
        assert 1.0 in config.fill_levels

    def test_custom_chars_per_token(self):
        config = PressureConfig(chars_per_token=3.0)
        assert config.chars_per_token == 3.0


# ---------------------------------------------------------------------------
# DriftPoint
# ---------------------------------------------------------------------------

class TestDriftPoint:
    def test_basic_creation(self):
        dp = DriftPoint(
            fill_level=0.5,
            token_count=4096,
            similarity_to_baseline=0.9,
            verdict=DriftVerdict.STABLE,
        )
        assert dp.fill_level == 0.5
        assert dp.token_count == 4096
        assert dp.similarity_to_baseline == 0.9
        assert dp.verdict == DriftVerdict.STABLE
        assert dp.outputs == []

    def test_with_outputs(self):
        dp = DriftPoint(
            fill_level=0.7,
            token_count=5734,
            similarity_to_baseline=0.7,
            verdict=DriftVerdict.MODERATE,
            outputs=["output A", "output B"],
        )
        assert len(dp.outputs) == 2


# ---------------------------------------------------------------------------
# PressureReport
# ---------------------------------------------------------------------------

class TestPressureReport:
    def _make_report(self, gate_passed: bool = True) -> PressureReport:
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5, 0.9])
        drift_curve = [
            DriftPoint(0.1, 205, 1.0, DriftVerdict.STABLE),
            DriftPoint(0.5, 1024, 0.9, DriftVerdict.STABLE),
            DriftPoint(0.9, 1843, 0.88, DriftVerdict.STABLE),
        ]
        return PressureReport(
            config=config,
            drift_curve=drift_curve,
            context_pressure_score=0.89,
            pressure_onset_token=None,
            verdict=DriftVerdict.STABLE,
            gate_passed=gate_passed,
            recommendation="Stable.",
        )

    def test_summary_contains_score(self):
        report = self._make_report()
        summary = report.summary()
        assert "ContextPressureScore" in summary

    def test_summary_contains_gate(self):
        report = self._make_report()
        summary = report.summary()
        assert "Gate" in summary

    def test_summary_passed(self):
        report = self._make_report(gate_passed=True)
        assert "PASSED" in report.summary()

    def test_summary_failed(self):
        report = self._make_report(gate_passed=False)
        assert "FAILED" in report.summary()

    def test_as_dict_keys(self):
        report = self._make_report()
        d = report.as_dict()
        assert "context_pressure_score" in d
        assert "drift_curve" in d
        assert "gate_passed" in d
        assert "verdict" in d
        assert "recommendation" in d

    def test_as_dict_drift_curve_list(self):
        report = self._make_report()
        d = report.as_dict()
        assert isinstance(d["drift_curve"], list)
        assert len(d["drift_curve"]) == 3

    def test_as_dict_drift_curve_fields(self):
        report = self._make_report()
        dp_dict = report.as_dict()["drift_curve"][0]
        assert "fill_level" in dp_dict
        assert "token_count" in dp_dict
        assert "similarity_to_baseline" in dp_dict
        assert "verdict" in dp_dict

    def test_summary_no_onset(self):
        report = self._make_report()
        assert "not detected" in report.summary()

    def test_summary_with_onset(self):
        config = PressureConfig(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = PressureReport(
            config=config,
            drift_curve=[],
            context_pressure_score=0.7,
            pressure_onset_token=500,
            verdict=DriftVerdict.MILD,
            gate_passed=False,
            recommendation="Onset at 500.",
        )
        assert "500" in report.summary()
