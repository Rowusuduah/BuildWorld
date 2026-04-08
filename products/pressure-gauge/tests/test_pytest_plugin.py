"""Tests for pressure_gauge.pytest_plugin"""
import pytest

from pressure_gauge.pytest_plugin import PressureGaugeSuite
from pressure_gauge.models import CriticalityLevel, PressureReport


def stable_agent(ctx: str) -> str:
    return "Complete and thorough analysis with all details provided."


# ---------------------------------------------------------------------------
# PressureGaugeSuite class
# ---------------------------------------------------------------------------

class TestPressureGaugeSuiteClass:
    def test_configure_returns_self(self):
        suite = PressureGaugeSuite()
        result = suite.configure(model_context_limit=2048)
        assert result is suite

    def test_default_config_is_none(self):
        suite = PressureGaugeSuite()
        assert suite._config is None

    def test_configure_sets_config(self):
        suite = PressureGaugeSuite()
        suite.configure(model_context_limit=4096)
        assert suite._config is not None
        assert suite._config.model_context_limit == 4096

    def test_configure_criticality(self):
        suite = PressureGaugeSuite()
        suite.configure(criticality=CriticalityLevel.LOW)
        assert suite._config.criticality == CriticalityLevel.LOW

    def test_configure_fill_levels(self):
        suite = PressureGaugeSuite()
        suite.configure(fill_levels=[0.2, 0.8])
        assert suite._config.fill_levels == [0.2, 0.8]

    def test_configure_use_neural(self):
        suite = PressureGaugeSuite()
        suite.configure(use_neural=True)
        assert suite._use_neural is True

    def test_sweep_returns_report(self):
        suite = PressureGaugeSuite()
        suite.configure(model_context_limit=2048, fill_levels=[0.1, 0.5])
        report = suite.sweep(stable_agent)
        assert isinstance(report, PressureReport)

    def test_quick_returns_report(self):
        suite = PressureGaugeSuite()
        suite.configure(model_context_limit=2048)
        report = suite.quick(stable_agent)
        assert isinstance(report, PressureReport)

    def test_quick_has_3_drift_points(self):
        suite = PressureGaugeSuite()
        suite.configure(model_context_limit=2048)
        report = suite.quick(stable_agent)
        assert len(report.drift_curve) == 3

    def test_sweep_with_no_config_uses_defaults(self):
        suite = PressureGaugeSuite()
        report = suite.sweep(stable_agent)
        assert isinstance(report, PressureReport)


# ---------------------------------------------------------------------------
# pytest fixture
# ---------------------------------------------------------------------------

def test_pressure_gauge_suite_fixture_is_suite(pressure_gauge_suite):
    """The fixture should return a PressureGaugeSuite instance."""
    assert isinstance(pressure_gauge_suite, PressureGaugeSuite)


def test_fixture_sweep(pressure_gauge_suite):
    """Fixture sweep should return a PressureReport."""
    pressure_gauge_suite.configure(
        model_context_limit=1024,
        fill_levels=[0.1, 0.5],
    )
    report = pressure_gauge_suite.sweep(stable_agent)
    assert isinstance(report, PressureReport)
    assert 0.0 <= report.context_pressure_score <= 1.0


def test_fixture_quick(pressure_gauge_suite):
    """Fixture quick should complete and return a report."""
    pressure_gauge_suite.configure(model_context_limit=1024)
    report = pressure_gauge_suite.quick(stable_agent)
    assert report.verdict is not None
