"""Tests for spec-drift v0.1.0.

Coverage targets: SemanticConstraint, @spec, Observation, ObservationStore,
DriftMonitor, run_ci_gate, CLI.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import pytest
from pydantic import BaseModel

from spec_drift import (
    ConstraintType,
    DriftMonitor,
    DriftSeverity,
    Observation,
    ObservationStore,
    SemanticConstraint,
    run_ci_gate,
    spec,
    __version__,
    _cli_main,
)


# ---------------------------------------------------------------------------
# Fixtures — shared Pydantic models
# ---------------------------------------------------------------------------

@spec(
    category=SemanticConstraint.from_authorized_values(
        ["positive", "negative", "neutral"],
        alert_threshold=0.10,
    ),
    reasoning=SemanticConstraint.from_length_bounds(
        min_words=5,
        max_words=100,
        alert_threshold=0.15,
    ),
    score=SemanticConstraint.from_distribution(
        mean=5.0,
        std=2.0,
        alert_threshold=0.20,
    ),
)
class SentimentOutput(BaseModel):
    category: str
    reasoning: str
    score: float


@spec(
    label=SemanticConstraint.from_authorized_values(["spam", "ham"]),
    subject=SemanticConstraint.from_pattern(regex=r"^[A-Z]"),
)
class EmailClassification(BaseModel):
    label: str
    subject: str


def _monitor(db_path: str = ":memory:") -> DriftMonitor:
    return DriftMonitor(spec=SentimentOutput, db_path=db_path)


def _good_output() -> SentimentOutput:
    return SentimentOutput(
        category="positive",
        reasoning="The text expresses clear positive sentiment.",
        score=6.0,
    )


def _bad_output() -> SentimentOutput:
    """Output that violates all three constraints."""
    return SentimentOutput(
        category="ambivalent",      # unauthorized
        reasoning="ok",             # too short (1 word < 5)
        score=99.0,                 # >> 3σ from mean 5.0
    )


# ---------------------------------------------------------------------------
# 1. __version__
# ---------------------------------------------------------------------------

def test_version_string():
    assert __version__ == "0.1.0"


# ---------------------------------------------------------------------------
# 2. ConstraintType enum
# ---------------------------------------------------------------------------

def test_constraint_type_values():
    assert ConstraintType.AUTHORIZED_VALUES == "authorized_values"
    assert ConstraintType.LENGTH_BOUNDS == "length_bounds"
    assert ConstraintType.DISTRIBUTION == "distribution"
    assert ConstraintType.PATTERN_MATCH == "pattern_match"
    assert ConstraintType.CORRELATION == "correlation"


# ---------------------------------------------------------------------------
# 3. DriftSeverity enum
# ---------------------------------------------------------------------------

def test_drift_severity_values():
    assert DriftSeverity.NONE == "none"
    assert DriftSeverity.LOW == "low"
    assert DriftSeverity.MEDIUM == "medium"
    assert DriftSeverity.HIGH == "high"
    assert DriftSeverity.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# 4. SemanticConstraint.from_authorized_values
# ---------------------------------------------------------------------------

def test_authorized_values_pass():
    c = SemanticConstraint.from_authorized_values(["a", "b", "c"])
    passed, reason = c.check("a")
    assert passed
    assert "authorized" in reason


def test_authorized_values_fail():
    c = SemanticConstraint.from_authorized_values(["a", "b"])
    passed, reason = c.check("x")
    assert not passed
    assert "'x'" in reason


def test_authorized_values_stores_params():
    c = SemanticConstraint.from_authorized_values(["yes", "no"], tolerance=0.05)
    assert c.params["authorized"] == ["yes", "no"]
    assert c.params["tolerance"] == 0.05
    assert c.constraint_type == ConstraintType.AUTHORIZED_VALUES


def test_authorized_values_numeric():
    c = SemanticConstraint.from_authorized_values([1, 2, 3])
    passed, _ = c.check(2)
    assert passed
    passed2, _ = c.check(4)
    assert not passed2


# ---------------------------------------------------------------------------
# 5. SemanticConstraint.from_length_bounds
# ---------------------------------------------------------------------------

def test_length_bounds_pass():
    c = SemanticConstraint.from_length_bounds(min_words=3, max_words=10)
    passed, reason = c.check("hello world foo")
    assert passed
    assert "word count" in reason


def test_length_bounds_too_short():
    c = SemanticConstraint.from_length_bounds(min_words=5, max_words=10)
    passed, reason = c.check("too short")
    assert not passed
    assert "outside" in reason


def test_length_bounds_too_long():
    c = SemanticConstraint.from_length_bounds(min_words=1, max_words=2)
    passed, reason = c.check("one two three four")
    assert not passed


def test_length_bounds_non_string():
    c = SemanticConstraint.from_length_bounds(min_words=1, max_words=10)
    passed, reason = c.check(42)
    assert not passed
    assert "str" in reason


def test_length_bounds_exact_min():
    c = SemanticConstraint.from_length_bounds(min_words=3, max_words=10)
    passed, _ = c.check("one two three")
    assert passed


def test_length_bounds_exact_max():
    c = SemanticConstraint.from_length_bounds(min_words=1, max_words=3)
    passed, _ = c.check("one two three")
    assert passed


# ---------------------------------------------------------------------------
# 6. SemanticConstraint.from_distribution
# ---------------------------------------------------------------------------

def test_distribution_pass():
    c = SemanticConstraint.from_distribution(mean=5.0, std=1.0)
    passed, reason = c.check(5.5)
    assert passed
    assert "within" in reason


def test_distribution_fail_outlier():
    c = SemanticConstraint.from_distribution(mean=5.0, std=1.0)
    passed, reason = c.check(100.0)
    assert not passed
    assert "σ from mean" in reason


def test_distribution_non_numeric():
    c = SemanticConstraint.from_distribution(mean=5.0, std=1.0)
    passed, reason = c.check("not a number")
    assert not passed
    assert "convert" in reason.lower()


def test_distribution_zero_std():
    """std=0 should not cause a divide-by-zero."""
    c = SemanticConstraint.from_distribution(mean=5.0, std=0.0)
    passed, _ = c.check(5.0)
    assert passed
    passed2, _ = c.check(100.0)
    assert not passed2


# ---------------------------------------------------------------------------
# 7. SemanticConstraint.from_pattern
# ---------------------------------------------------------------------------

def test_pattern_pass():
    c = SemanticConstraint.from_pattern(regex=r"^\d{3}-\d{4}$")
    passed, reason = c.check("555-1234")
    assert passed
    assert "matched" in reason


def test_pattern_fail():
    c = SemanticConstraint.from_pattern(regex=r"^\d{3}-\d{4}$")
    passed, reason = c.check("not-a-phone")
    assert not passed
    assert "did not match" in reason


def test_pattern_non_string():
    c = SemanticConstraint.from_pattern(regex=r"^\d+$")
    passed, reason = c.check(42)
    assert not passed
    assert "str" in reason


def test_pattern_invalid_regex_raises():
    with pytest.raises(re.error if False else Exception):
        # from_pattern validates regex at creation time
        SemanticConstraint.from_pattern(regex="[invalid")


# ---------------------------------------------------------------------------
# 8. @spec decorator
# ---------------------------------------------------------------------------

def test_spec_attaches_constraints():
    assert hasattr(SentimentOutput, "__spec_constraints__")
    constraints = SentimentOutput.__spec_constraints__
    assert "category" in constraints
    assert "reasoning" in constraints
    assert "score" in constraints


def test_spec_sets_field_names():
    constraints = SentimentOutput.__spec_constraints__
    assert constraints["category"].field_name == "category"
    assert constraints["reasoning"].field_name == "reasoning"


def test_spec_does_not_break_pydantic():
    output = SentimentOutput(
        category="positive",
        reasoning="This is a valid reasoning text.",
        score=5.0,
    )
    assert output.category == "positive"
    assert output.score == 5.0


def test_spec_multiple_constraint_types():
    constraints = EmailClassification.__spec_constraints__
    assert constraints["label"].constraint_type == ConstraintType.AUTHORIZED_VALUES
    assert constraints["subject"].constraint_type == ConstraintType.PATTERN_MATCH


# ---------------------------------------------------------------------------
# 9. Observation
# ---------------------------------------------------------------------------

def test_observation_passed_all_pass():
    obs = Observation(
        timestamp=time.time(),
        spec_name="Test",
        output_data={"x": 1},
        constraint_results={"f1": (True, "ok"), "f2": (True, "ok")},
    )
    assert obs.passed is True
    assert obs.violation_count == 0


def test_observation_passed_any_fail():
    obs = Observation(
        timestamp=time.time(),
        spec_name="Test",
        output_data={"x": 1},
        constraint_results={"f1": (True, "ok"), "f2": (False, "bad")},
    )
    assert obs.passed is False
    assert obs.violation_count == 1


def test_observation_violation_count_multiple():
    obs = Observation(
        timestamp=time.time(),
        spec_name="Test",
        output_data={},
        constraint_results={
            "a": (False, "fail"),
            "b": (False, "fail"),
            "c": (True, "ok"),
        },
    )
    assert obs.violation_count == 2


def test_observation_to_dict_serializable():
    obs = Observation(
        timestamp=time.time(),
        spec_name="Test",
        output_data={"val": "hello"},
        constraint_results={"f": (True, "ok")},
        model_version="haiku",
        prompt_hash="abc123",
    )
    d = obs.to_dict()
    assert d["spec_name"] == "Test"
    assert d["model_version"] == "haiku"
    assert d["prompt_hash"] == "abc123"
    assert isinstance(d["output_data"], str)  # JSON string
    assert isinstance(d["constraint_results"], str)  # JSON string
    # Verify JSON is valid
    json.loads(d["output_data"])
    json.loads(d["constraint_results"])


def test_observation_call_id_unique():
    obs1 = Observation(
        timestamp=time.time(), spec_name="T", output_data={}, constraint_results={}
    )
    obs2 = Observation(
        timestamp=time.time(), spec_name="T", output_data={}, constraint_results={}
    )
    assert obs1.call_id != obs2.call_id


# ---------------------------------------------------------------------------
# 10. ObservationStore
# ---------------------------------------------------------------------------

def test_store_saves_and_queries():
    store = ObservationStore(db_path=":memory:")
    obs = Observation(
        timestamp=time.time(),
        spec_name="MySpec",
        output_data={"x": 1},
        constraint_results={"f": (True, "ok")},
    )
    store.save(obs)
    rows = store.query("MySpec", since_hours=1.0)
    assert len(rows) == 1
    assert rows[0]["spec_name"] == "MySpec"


def test_store_filters_by_spec_name():
    store = ObservationStore(db_path=":memory:")
    for sn in ["SpecA", "SpecB", "SpecA"]:
        store.save(Observation(
            timestamp=time.time(), spec_name=sn,
            output_data={}, constraint_results={},
        ))
    assert len(store.query("SpecA", 1.0)) == 2
    assert len(store.query("SpecB", 1.0)) == 1
    assert len(store.query("SpecC", 1.0)) == 0


def test_store_violation_rate_zero():
    store = ObservationStore(db_path=":memory:")
    assert store.violation_rate("NoSpec", since_hours=1.0) == 0.0


def test_store_violation_rate_calculation():
    store = ObservationStore(db_path=":memory:")
    # 3 pass, 1 fail → 25%
    for i in range(3):
        store.save(Observation(
            timestamp=time.time(), spec_name="S",
            output_data={}, constraint_results={"f": (True, "ok")},
        ))
    store.save(Observation(
        timestamp=time.time(), spec_name="S",
        output_data={}, constraint_results={"f": (False, "bad")},
    ))
    rate = store.violation_rate("S", since_hours=1.0)
    assert abs(rate - 0.25) < 1e-6


def test_store_list_specs():
    store = ObservationStore(db_path=":memory:")
    for sn in ["Alpha", "Beta", "Alpha"]:
        store.save(Observation(
            timestamp=time.time(), spec_name=sn,
            output_data={}, constraint_results={},
        ))
    specs = store.list_specs()
    assert set(specs) == {"Alpha", "Beta"}


# ---------------------------------------------------------------------------
# 11. DriftMonitor
# ---------------------------------------------------------------------------

def test_monitor_observe_returns_output():
    monitor = _monitor()
    output = _good_output()
    result = monitor.observe(output)
    assert result is output


def test_monitor_observe_wrong_type_raises():
    monitor = _monitor()

    class OtherModel(BaseModel):
        x: int

    with pytest.raises(TypeError, match="Expected SentimentOutput"):
        monitor.observe(OtherModel(x=1))  # type: ignore


def test_monitor_observe_logs_observation():
    monitor = _monitor()
    monitor.observe(_good_output())
    rows = monitor.store.query("SentimentOutput", since_hours=1.0)
    assert len(rows) == 1


def test_monitor_observe_violation_logged():
    monitor = _monitor()
    monitor.observe(_bad_output())
    rows = monitor.store.query("SentimentOutput", since_hours=1.0)
    assert rows[0]["passed"] == 0  # SQLite stores int


def test_monitor_watch_decorator():
    monitor = _monitor()

    @monitor.watch
    def fake_llm(text: str) -> SentimentOutput:
        return SentimentOutput(
            category="neutral",
            reasoning="Neutral tone throughout the document.",
            score=5.0,
        )

    result = fake_llm("some text")
    assert result.category == "neutral"
    rows = monitor.store.query("SentimentOutput", since_hours=1.0)
    assert len(rows) == 1


def test_monitor_drift_report_no_data():
    monitor = _monitor()
    report = monitor.drift_report(since_hours=1.0)
    assert report["status"] == "no_data"
    assert report["observations"] == 0


def test_monitor_drift_report_all_pass():
    monitor = _monitor()
    for _ in range(5):
        monitor.observe(_good_output())
    report = monitor.drift_report(since_hours=1.0)
    assert report["observations"] == 5
    assert report["violation_rate"] == 0.0
    assert report["severity"] == "none"


def test_monitor_drift_report_violations():
    monitor = _monitor()
    for _ in range(4):
        monitor.observe(_good_output())
    monitor.observe(_bad_output())
    report = monitor.drift_report(since_hours=1.0)
    assert report["observations"] == 5
    assert abs(report["violation_rate"] - 0.20) < 1e-4
    assert "field_violation_rates" in report


def test_monitor_drift_report_field_rates():
    monitor = _monitor()
    # 1 bad (violates category, reasoning, score), 1 good
    monitor.observe(_bad_output())
    monitor.observe(_good_output())
    report = monitor.drift_report(since_hours=1.0)
    frates = report["field_violation_rates"]
    assert "category" in frates
    assert frates["category"] == 0.5


def test_monitor_model_version_tracked():
    monitor = DriftMonitor(
        spec=SentimentOutput,
        db_path=":memory:",
        model_version="claude-haiku-4-5-20251001",
    )
    monitor.observe(_good_output())
    rows = monitor.store.query("SentimentOutput", since_hours=1.0)
    assert rows[0]["model_version"] == "claude-haiku-4-5-20251001"


def test_monitor_prompt_hash_tracked():
    monitor = DriftMonitor(
        spec=SentimentOutput,
        db_path=":memory:",
        prompt_hash="deadbeef",
    )
    monitor.observe(_good_output())
    rows = monitor.store.query("SentimentOutput", since_hours=1.0)
    assert rows[0]["prompt_hash"] == "deadbeef"


# ---------------------------------------------------------------------------
# 12. DriftMonitor._severity
# ---------------------------------------------------------------------------

def test_severity_none():
    assert DriftMonitor._severity(0.0) == DriftSeverity.NONE


def test_severity_low():
    assert DriftMonitor._severity(0.01) == DriftSeverity.LOW
    assert DriftMonitor._severity(0.049) == DriftSeverity.LOW


def test_severity_medium():
    assert DriftMonitor._severity(0.05) == DriftSeverity.MEDIUM
    assert DriftMonitor._severity(0.14) == DriftSeverity.MEDIUM


def test_severity_high():
    assert DriftMonitor._severity(0.15) == DriftSeverity.HIGH
    assert DriftMonitor._severity(0.29) == DriftSeverity.HIGH


def test_severity_critical():
    assert DriftMonitor._severity(0.30) == DriftSeverity.CRITICAL
    assert DriftMonitor._severity(1.0) == DriftSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 13. Alert callback
# ---------------------------------------------------------------------------

def test_alert_callback_fires():
    alerts = []

    def on_alert(msg: str, rate: float):
        alerts.append((msg, rate))

    monitor = DriftMonitor(
        spec=SentimentOutput,
        db_path=":memory:",
        alert_callback=on_alert,
    )
    # Observe 20 bad outputs to push rate > any alert_threshold (0.10 for category)
    for _ in range(20):
        monitor.observe(_bad_output())

    assert len(alerts) > 0
    assert "spec-drift ALERT" in alerts[0][0]


def test_alert_callback_not_fires_on_clean():
    alerts = []

    def on_alert(msg: str, rate: float):
        alerts.append(msg)

    monitor = DriftMonitor(
        spec=SentimentOutput,
        db_path=":memory:",
        alert_callback=on_alert,
    )
    for _ in range(5):
        monitor.observe(_good_output())
    assert len(alerts) == 0


# ---------------------------------------------------------------------------
# 14. run_ci_gate
# ---------------------------------------------------------------------------

def test_ci_gate_passes_clean_batch():
    monitor = _monitor()
    outputs = [_good_output() for _ in range(10)]
    passed, report = run_ci_gate(monitor, outputs, threshold=0.20)
    assert passed is True
    assert report["ci_passed"] is True
    assert report["ci_threshold"] == 0.20


def test_ci_gate_fails_dirty_batch():
    monitor = _monitor()
    outputs = [_bad_output() for _ in range(10)]
    passed, report = run_ci_gate(monitor, outputs, threshold=0.20)
    assert passed is False
    assert report["ci_passed"] is False


def test_ci_gate_report_fields():
    monitor = _monitor()
    passed, report = run_ci_gate(monitor, [_good_output()], threshold=0.50)
    assert "ci_threshold" in report
    assert "ci_passed" in report
    assert "violation_rate" in report


def test_ci_gate_empty_batch():
    monitor = _monitor()
    passed, report = run_ci_gate(monitor, [], threshold=0.20)
    # No observations → no_data status → passed by default
    assert passed is True


# ---------------------------------------------------------------------------
# 15. CLI
# ---------------------------------------------------------------------------

def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_report_no_data(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    exit_code = _cli_main(["report", "--db", db])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No observations" in out


def test_cli_ci_no_data(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    exit_code = _cli_main(["ci", "--db", db])
    assert exit_code == 0


def test_cli_report_with_data(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    monitor = DriftMonitor(spec=SentimentOutput, db_path=db)
    for _ in range(3):
        monitor.observe(_good_output())
    exit_code = _cli_main(["report", "--db", db])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "SentimentOutput" in out


def test_cli_ci_pass(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    monitor = DriftMonitor(spec=SentimentOutput, db_path=db)
    for _ in range(5):
        monitor.observe(_good_output())
    exit_code = _cli_main(["ci", "--db", db, "--threshold", "0.20"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASSED" in out


def test_cli_ci_fail(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    monitor = DriftMonitor(spec=SentimentOutput, db_path=db)
    for _ in range(5):
        monitor.observe(_bad_output())
    exit_code = _cli_main(["ci", "--db", db, "--threshold", "0.20"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_cli_report_json(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    monitor = DriftMonitor(spec=SentimentOutput, db_path=db)
    monitor.observe(_good_output())
    exit_code = _cli_main(["report", "--db", db, "--json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["spec"] == "SentimentOutput"


# ---------------------------------------------------------------------------
# 16. Edge cases
# ---------------------------------------------------------------------------

def test_spec_with_no_constraints():
    """A class with no @spec constraints can still be wrapped."""
    class Bare(BaseModel):
        x: int

    monitor = DriftMonitor(spec=Bare, db_path=":memory:")
    result = monitor.observe(Bare(x=42))
    assert result.x == 42
    report = monitor.drift_report(since_hours=1.0)
    assert report["violation_rate"] == 0.0


def test_authorized_values_empty_list():
    """Empty authorized list means every value fails."""
    c = SemanticConstraint.from_authorized_values([])
    passed, reason = c.check("anything")
    assert not passed


def test_length_bounds_empty_string():
    c = SemanticConstraint.from_length_bounds(min_words=1, max_words=10)
    passed, _ = c.check("")
    assert not passed  # 0 words < 1


# Re-import for pattern test
import re
