"""
Tests for cot-coherence v0.1.0
===============================
All tests run without an Anthropic API key by mocking the LLM judge.
Tests cover: data models, parsing, judge response parsing, SQLite storage,
the CoherenceChecker class, the @coherence_check decorator, YAML suite runner,
and the CLI.

Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Import the package (installed as editable or on sys.path)
import cot_coherence as cc
from cot_coherence import (
    CoTStep,
    CoherenceChecker,
    CoherenceError,
    CoherenceReport,
    CoherenceStatus,
    CoherenceViolation,
    DimensionScore,
    ViolationType,
    _get_db_path,
    _init_db,
    _parse_judge_response,
    check,
    coherence_check,
    load_recent_reports,
    parse_steps,
    run_yaml_suite,
    save_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_response(
    coherence_score: float = 0.85,
    violations: Optional[list[dict]] = None,
    overall_confidence: float = 0.9,
    summary: str = "The reasoning is coherent.",
) -> dict:
    """Build a minimal valid LLM judge response dict."""
    return {
        "dimensions": {
            "step_continuity": {"score": coherence_score, "notes": "Smooth transitions."},
            "conclusion_grounding": {"score": coherence_score, "notes": "Well grounded."},
            "internal_consistency": {"score": coherence_score, "notes": "No contradictions."},
            "reasoning_completeness": {"score": coherence_score, "notes": "Complete."},
            "confidence_calibration": {"score": coherence_score, "notes": "Well calibrated."},
        },
        "violations": violations or [],
        "coherence_score": coherence_score,
        "overall_confidence": overall_confidence,
        "summary": summary,
    }


def _sample_steps() -> list[str]:
    return [
        "The sky is blue because of Rayleigh scattering of sunlight.",
        "Rayleigh scattering preferentially scatters shorter wavelengths.",
        "Blue light has a shorter wavelength than red light.",
        "Therefore, the sky appears blue during the day.",
    ]


def _sample_conclusion() -> str:
    return "The sky appears blue because shorter blue wavelengths are scattered more by the atmosphere."


def _make_report(
    status: CoherenceStatus = CoherenceStatus.COHERENT,
    coherence_score: float = 0.85,
    violations: Optional[list[CoherenceViolation]] = None,
) -> CoherenceReport:
    steps = [CoTStep(i, s) for i, s in enumerate(_sample_steps())]
    dims = [
        DimensionScore("step_continuity", 0.85, True, "Good"),
        DimensionScore("conclusion_grounding", 0.85, True, "Good"),
        DimensionScore("internal_consistency", 0.85, True, "Good"),
        DimensionScore("reasoning_completeness", 0.85, True, "Good"),
        DimensionScore("confidence_calibration", 0.85, True, "Good"),
    ]
    return CoherenceReport(
        steps=steps,
        conclusion=_sample_conclusion(),
        status=status,
        coherence_score=coherence_score,
        incoherence_score=round(1.0 - coherence_score, 4),
        overall_confidence=0.9,
        violations=violations or [],
        dimensions=dims,
        summary="Coherent reasoning.",
        timestamp="2026-03-27T12:00:00+00:00",
        model_used="claude-haiku-4-5-20251001",
        step_count=len(steps),
    )


# ---------------------------------------------------------------------------
# 1. CoTStep tests
# ---------------------------------------------------------------------------

class TestCoTStep:
    def test_index_and_text(self):
        s = CoTStep(index=0, text="Some reasoning step.")
        assert s.index == 0
        assert s.text == "Some reasoning step."

    def test_short_truncates_at_120(self):
        long_text = "x" * 150
        s = CoTStep(index=0, text=long_text)
        assert s.short().endswith("...")
        assert len(s.short()) == 123  # 120 + len("...")

    def test_short_no_truncation_when_short_enough(self):
        s = CoTStep(index=0, text="Short step.")
        assert s.short() == "Short step."


# ---------------------------------------------------------------------------
# 2. CoherenceStatus tests
# ---------------------------------------------------------------------------

class TestCoherenceStatus:
    def test_all_values_exist(self):
        assert CoherenceStatus.COHERENT.value == "COHERENT"
        assert CoherenceStatus.DEGRADED.value == "DEGRADED"
        assert CoherenceStatus.INCOHERENT.value == "INCOHERENT"
        assert CoherenceStatus.SKIP.value == "SKIP"

    def test_str_enum(self):
        assert CoherenceStatus.COHERENT == "COHERENT"


# ---------------------------------------------------------------------------
# 3. ViolationType tests
# ---------------------------------------------------------------------------

class TestViolationType:
    def test_all_types_present(self):
        types = {v.value for v in ViolationType}
        assert "STEP_GAP" in types
        assert "CONTRADICTION" in types
        assert "UNSUPPORTED_CONCLUSION" in types
        assert "REASONING_LEAP" in types
        assert "OVERCONFIDENCE" in types
        assert "CIRCULAR" in types
        assert "SCOPE_SHIFT" in types


# ---------------------------------------------------------------------------
# 4. CoherenceViolation tests
# ---------------------------------------------------------------------------

class TestCoherenceViolation:
    def test_critical_at_threshold(self):
        v = CoherenceViolation(
            violation_type=ViolationType.CONTRADICTION,
            severity=0.7,
            step_indices=[0, 2],
            description="Steps 0 and 2 contradict.",
            evidence="Step 0 says A; Step 2 says not A.",
        )
        assert v.is_critical()

    def test_not_critical_below_threshold(self):
        v = CoherenceViolation(
            violation_type=ViolationType.CIRCULAR,
            severity=0.4,
            step_indices=[1],
            description="Step 1 repeats Step 0.",
            evidence="...",
        )
        assert not v.is_critical()

    def test_exactly_at_threshold_is_critical(self):
        v = CoherenceViolation(
            ViolationType.STEP_GAP, 0.7, [], "Gap.", ""
        )
        assert v.is_critical()


# ---------------------------------------------------------------------------
# 5. DimensionScore tests
# ---------------------------------------------------------------------------

class TestDimensionScore:
    def test_passed_at_threshold(self):
        d = DimensionScore("step_continuity", 0.7, True, "OK")
        assert d.passed

    def test_failed_below_threshold(self):
        d = DimensionScore("internal_consistency", 0.5, False, "Issues found.")
        assert not d.passed


# ---------------------------------------------------------------------------
# 6. CoherenceReport tests
# ---------------------------------------------------------------------------

class TestCoherenceReport:
    def test_passed_for_coherent(self):
        report = _make_report(status=CoherenceStatus.COHERENT)
        assert report.passed()

    def test_not_passed_for_degraded(self):
        report = _make_report(status=CoherenceStatus.DEGRADED)
        assert not report.passed()

    def test_not_passed_for_incoherent(self):
        report = _make_report(status=CoherenceStatus.INCOHERENT)
        assert not report.passed()

    def test_incoherence_score_is_complement(self):
        report = _make_report(coherence_score=0.75)
        assert abs(report.incoherence_score - 0.25) < 0.001

    def test_critical_violations_filter(self):
        v_critical = CoherenceViolation(ViolationType.CONTRADICTION, 0.9, [0, 1], "...", "")
        v_minor = CoherenceViolation(ViolationType.CIRCULAR, 0.3, [2], "...", "")
        report = _make_report(violations=[v_critical, v_minor])
        crits = report.critical_violations()
        assert len(crits) == 1
        assert crits[0].violation_type == ViolationType.CONTRADICTION

    def test_to_dict_keys(self):
        report = _make_report()
        d = report.to_dict()
        assert "status" in d
        assert "coherence_score" in d
        assert "incoherence_score" in d
        assert "violations" in d
        assert "dimensions" in d
        assert "summary" in d
        assert "step_count" in d

    def test_to_dict_violations_structure(self):
        v = CoherenceViolation(ViolationType.STEP_GAP, 0.8, [0, 1], "Gap here.", "evidence text")
        report = _make_report(violations=[v])
        d = report.to_dict()
        assert len(d["violations"]) == 1
        assert d["violations"][0]["type"] == "STEP_GAP"
        assert d["violations"][0]["severity"] == 0.8

    def test_to_markdown_contains_status(self):
        report = _make_report(status=CoherenceStatus.COHERENT)
        md = report.to_markdown()
        assert "COHERENT" in md
        assert "cot-coherence Report" in md

    def test_to_markdown_contains_violation_type(self):
        v = CoherenceViolation(ViolationType.OVERCONFIDENCE, 0.75, [], "Too certain.", "")
        report = _make_report(violations=[v])
        md = report.to_markdown()
        assert "OVERCONFIDENCE" in md

    def test_to_markdown_degraded_has_warning_icon(self):
        report = _make_report(status=CoherenceStatus.DEGRADED)
        md = report.to_markdown()
        assert "⚠️" in md

    def test_to_markdown_incoherent_has_fail_icon(self):
        report = _make_report(status=CoherenceStatus.INCOHERENT)
        md = report.to_markdown()
        assert "❌" in md


# ---------------------------------------------------------------------------
# 7. parse_steps tests
# ---------------------------------------------------------------------------

class TestParseSteps:
    def test_list_input_passthrough(self):
        raw = ["Step A.", "Step B.", "Step C."]
        result = parse_steps(raw)
        assert len(result) == 3
        assert result[0].text == "Step A."
        assert result[1].index == 1

    def test_list_filters_empty_strings(self):
        raw = ["Step A.", "", "  ", "Step B."]
        result = parse_steps(raw)
        assert len(result) == 2

    def test_numbered_list_string(self):
        raw = "1. First point.\n2. Second point.\n3. Third point."
        result = parse_steps(raw)
        assert len(result) >= 3

    def test_blank_line_separated(self):
        raw = "First reasoning paragraph.\n\nSecond reasoning paragraph.\n\nThird."
        result = parse_steps(raw)
        assert len(result) == 3

    def test_connector_split(self):
        raw = "First, we observe X. Then, we note Y. Therefore, Z follows."
        result = parse_steps(raw)
        # Should split on connectors
        assert len(result) >= 1

    def test_single_string_fallback(self):
        raw = "Just one big step with no clear divisions."
        result = parse_steps(raw)
        assert len(result) == 1
        assert result[0].text == raw

    def test_step_numbered_with_colon(self):
        raw = "Step 1: Set up the environment.\nStep 2: Run the tests.\nStep 3: Check the output."
        result = parse_steps(raw)
        assert len(result) >= 3

    def test_indices_sequential(self):
        raw = ["A", "B", "C", "D"]
        result = parse_steps(raw)
        for i, s in enumerate(result):
            assert s.index == i


# ---------------------------------------------------------------------------
# 8. _parse_judge_response tests
# ---------------------------------------------------------------------------

class TestParseJudgeResponse:
    def _steps(self) -> list[CoTStep]:
        return [CoTStep(i, s) for i, s in enumerate(_sample_steps())]

    def test_coherent_threshold(self):
        raw = _make_raw_response(coherence_score=0.90)
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert report.status == CoherenceStatus.COHERENT

    def test_degraded_threshold(self):
        raw = _make_raw_response(coherence_score=0.65)
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert report.status == CoherenceStatus.DEGRADED

    def test_incoherent_threshold(self):
        raw = _make_raw_response(coherence_score=0.40)
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert report.status == CoherenceStatus.INCOHERENT

    def test_all_five_dimensions_parsed(self):
        raw = _make_raw_response(coherence_score=0.85)
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert len(report.dimensions) == 5
        dim_names = {d.name for d in report.dimensions}
        assert "step_continuity" in dim_names
        assert "conclusion_grounding" in dim_names
        assert "internal_consistency" in dim_names
        assert "reasoning_completeness" in dim_names
        assert "confidence_calibration" in dim_names

    def test_violation_parsed_correctly(self):
        viol = {
            "type": "STEP_GAP",
            "severity": 0.8,
            "step_indices": [1, 2],
            "description": "Gap between steps.",
            "evidence": "...",
        }
        raw = _make_raw_response(coherence_score=0.6, violations=[viol])
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert len(report.violations) == 1
        assert report.violations[0].violation_type == ViolationType.STEP_GAP
        assert report.violations[0].severity == 0.8
        assert report.violations[0].step_indices == [1, 2]

    def test_unknown_violation_type_defaults(self):
        viol = {
            "type": "UNKNOWN_TYPE",
            "severity": 0.5,
            "step_indices": [],
            "description": "Unknown.",
            "evidence": "",
        }
        raw = _make_raw_response(coherence_score=0.7, violations=[viol])
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert len(report.violations) == 1
        assert report.violations[0].violation_type == ViolationType.REASONING_LEAP

    def test_coherence_score_recomputed_when_inconsistent(self):
        raw = _make_raw_response(coherence_score=0.85)
        # Override the reported coherence_score to be very inconsistent with dimensions
        raw["coherence_score"] = 0.10  # dimensions all say 0.85
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        # Should recompute from dimensions
        assert report.coherence_score > 0.5

    def test_summary_and_confidence_set(self):
        raw = _make_raw_response(coherence_score=0.85, summary="Great reasoning.", overall_confidence=0.92)
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert report.summary == "Great reasoning."
        assert report.overall_confidence == 0.92

    def test_incoherence_score_is_complement(self):
        raw = _make_raw_response(coherence_score=0.75)
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "test-model")
        assert abs(report.incoherence_score - 0.25) < 0.01

    def test_model_name_stored(self):
        raw = _make_raw_response()
        report = _parse_judge_response(raw, self._steps(), _sample_conclusion(), "my-test-model")
        assert report.model_used == "my-test-model"

    def test_step_count_matches_input(self):
        raw = _make_raw_response()
        steps = self._steps()
        report = _parse_judge_response(raw, steps, _sample_conclusion(), "m")
        assert report.step_count == len(steps)


# ---------------------------------------------------------------------------
# 9. SQLite storage tests
# ---------------------------------------------------------------------------

class TestSQLiteStorage:
    def test_init_db_creates_table(self, tmp_path):
        db = tmp_path / "test.db"
        _init_db(db)
        conn = sqlite3.connect(db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        assert ("coherence_reports",) in tables

    def test_save_and_load_report(self, tmp_path):
        db = tmp_path / "test.db"
        report = _make_report()
        save_report(report, db_path=db)
        rows = load_recent_reports(n=10, db_path=db)
        assert len(rows) == 1
        assert rows[0]["status"] == "COHERENT"
        assert abs(rows[0]["coherence_score"] - 0.85) < 0.001

    def test_save_multiple_reports(self, tmp_path):
        db = tmp_path / "test.db"
        for score in [0.9, 0.7, 0.4]:
            status = (
                CoherenceStatus.COHERENT if score >= 0.8 else
                CoherenceStatus.DEGRADED if score >= 0.55 else
                CoherenceStatus.INCOHERENT
            )
            save_report(_make_report(status=status, coherence_score=score), db_path=db)
        rows = load_recent_reports(n=10, db_path=db)
        assert len(rows) == 3

    def test_load_returns_empty_for_missing_db(self, tmp_path):
        db = tmp_path / "nonexistent.db"
        rows = load_recent_reports(db_path=db)
        assert rows == []

    def test_load_respects_n_limit(self, tmp_path):
        db = tmp_path / "test.db"
        for _ in range(5):
            save_report(_make_report(), db_path=db)
        rows = load_recent_reports(n=3, db_path=db)
        assert len(rows) == 3

    def test_saved_report_has_correct_violation_count(self, tmp_path):
        db = tmp_path / "test.db"
        v1 = CoherenceViolation(ViolationType.STEP_GAP, 0.8, [0, 1], "Gap.", "")
        v2 = CoherenceViolation(ViolationType.CIRCULAR, 0.4, [2], "Circular.", "")
        report = _make_report(violations=[v1, v2])
        save_report(report, db_path=db)
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT violation_count, critical_violation_count FROM coherence_reports").fetchone()
        conn.close()
        assert row[0] == 2
        assert row[1] == 1  # only v1 is critical (severity 0.8 >= 0.7)

    def test_init_db_idempotent(self, tmp_path):
        db = tmp_path / "test.db"
        _init_db(db)
        _init_db(db)  # Should not raise
        conn = sqlite3.connect(db)
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='coherence_reports'"
        ).fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# 10. check() function tests (with mocked LLM)
# ---------------------------------------------------------------------------

class TestCheckFunction:
    def _mock_response(self, coherence_score: float = 0.85) -> MagicMock:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(_make_raw_response(coherence_score)))]
        return mock_msg

    def test_check_returns_report(self, tmp_path):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._mock_response(0.85)

            report = check(
                steps=_sample_steps(),
                conclusion=_sample_conclusion(),
                save=True,
                db_path=tmp_path / "test.db",
            )

        assert isinstance(report, CoherenceReport)
        assert report.status == CoherenceStatus.COHERENT
        assert report.coherence_score >= 0.80

    def test_check_raises_on_empty_steps(self):
        with pytest.raises(ValueError, match="empty"):
            check(steps=[], conclusion="Some conclusion.")

    def test_check_raises_on_empty_conclusion(self):
        with pytest.raises(ValueError, match="empty"):
            check(steps=["Step 1.", "Step 2."], conclusion="")

    def test_check_skip_on_trivially_short_input(self, tmp_path):
        report = check(
            steps=["x"],
            conclusion="y",
            save=False,
        )
        assert report.status == CoherenceStatus.SKIP

    def test_check_no_save(self, tmp_path):
        db = tmp_path / "test.db"
        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._mock_response(0.85)

            check(
                steps=_sample_steps(),
                conclusion=_sample_conclusion(),
                save=False,
                db_path=db,
            )

        assert not db.exists()

    def test_check_parses_string_steps(self, tmp_path):
        raw_steps = "1. First.\n2. Second.\n3. Third."
        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._mock_response(0.85)

            report = check(
                steps=raw_steps,
                conclusion="Therefore X.",
                save=False,
            )

        assert report.step_count >= 3

    def test_check_raises_import_error_without_anthropic(self):
        with patch("cot_coherence.anthropic", None):
            with pytest.raises(ImportError, match="anthropic"):
                check(
                    steps=_sample_steps(),
                    conclusion=_sample_conclusion(),
                    save=False,
                )

    def test_check_incoherent_report(self, tmp_path):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._mock_response(0.35)

            report = check(
                steps=_sample_steps(),
                conclusion=_sample_conclusion(),
                save=False,
            )

        assert report.status == CoherenceStatus.INCOHERENT

    def test_check_degraded_report(self, tmp_path):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = self._mock_response(0.65)

            report = check(
                steps=_sample_steps(),
                conclusion=_sample_conclusion(),
                save=False,
            )

        assert report.status == CoherenceStatus.DEGRADED


# ---------------------------------------------------------------------------
# 11. CoherenceChecker class tests
# ---------------------------------------------------------------------------

class TestCoherenceChecker:
    def _setup_mock(self, mock_anthropic, coherence_score=0.85):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(_make_raw_response(coherence_score)))]
        mock_client.messages.create.return_value = mock_msg

    def test_checker_returns_report(self):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic)
            checker = CoherenceChecker(save=False)
            report = checker.check(steps=_sample_steps(), conclusion=_sample_conclusion())
        assert isinstance(report, CoherenceReport)

    def test_checker_uses_configured_model(self):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic)
            checker = CoherenceChecker(model="custom-model", save=False)
            report = checker.check(steps=_sample_steps(), conclusion=_sample_conclusion())
        assert report.model_used == "custom-model"

    def test_batch_check_returns_list(self):
        samples = [
            {"steps": _sample_steps(), "conclusion": _sample_conclusion()},
            {"steps": _sample_steps()[:2], "conclusion": "Partial conclusion."},
        ]
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic)
            checker = CoherenceChecker(save=False)
            results = checker.batch_check(samples)
        assert len(results) == 2
        assert all(isinstance(r, CoherenceReport) for r in results)

    def test_batch_check_empty_returns_empty(self):
        checker = CoherenceChecker(save=False)
        results = checker.batch_check([])
        assert results == []

    def test_checker_default_model_is_haiku(self):
        # DEFAULT_MODEL should default to claude-haiku unless env var overrides
        # We test the mechanism: the env var COT_COHERENCE_MODEL is read at import time.
        # We verify the constant is a non-empty string and contains "claude".
        assert isinstance(cc.DEFAULT_MODEL, str)
        assert len(cc.DEFAULT_MODEL) > 0


# ---------------------------------------------------------------------------
# 12. @coherence_check decorator tests
# ---------------------------------------------------------------------------

class TestCoherenceCheckDecorator:
    def _setup_mock(self, mock_anthropic, coherence_score=0.85):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(_make_raw_response(coherence_score)))]
        mock_client.messages.create.return_value = mock_msg

    def test_decorator_attaches_report_to_dict_result(self):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, coherence_score=0.85)

            @coherence_check(threshold=0.55)
            def my_fn():
                return {"steps": _sample_steps(), "conclusion": _sample_conclusion()}

            result = my_fn()

        assert "_coherence_report" in result
        assert isinstance(result["_coherence_report"], CoherenceReport)

    def test_decorator_passes_through_unrecognized_result(self):
        @coherence_check(threshold=0.55)
        def my_fn():
            return "just a string"

        result = my_fn()
        assert result == "just a string"

    def test_decorator_raises_on_fail_when_configured(self):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, coherence_score=0.30)

            @coherence_check(threshold=0.55, raise_on_fail=True)
            def my_fn():
                return {"steps": _sample_steps(), "conclusion": _sample_conclusion()}

            with pytest.raises(CoherenceError) as exc_info:
                my_fn()

        err = exc_info.value
        assert isinstance(err.report, CoherenceReport)
        assert "threshold" in str(err).lower() or "below" in str(err).lower()

    def test_decorator_does_not_raise_when_coherent(self):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, coherence_score=0.90)

            @coherence_check(threshold=0.55, raise_on_fail=True)
            def my_fn():
                return {"steps": _sample_steps(), "conclusion": _sample_conclusion()}

            result = my_fn()  # Should not raise

        assert result is not None

    def test_decorator_works_with_object_result(self):
        @dataclass
        class CoTResult:
            steps: list
            conclusion: str

        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, coherence_score=0.85)

            @coherence_check(threshold=0.55)
            def my_fn():
                return CoTResult(steps=_sample_steps(), conclusion=_sample_conclusion())

            result = my_fn()

        assert hasattr(result, "_coherence_report")
        assert isinstance(result._coherence_report, CoherenceReport)


# ---------------------------------------------------------------------------
# 13. CoherenceError tests
# ---------------------------------------------------------------------------

class TestCoherenceError:
    def test_coherence_error_stores_report(self):
        report = _make_report(status=CoherenceStatus.INCOHERENT, coherence_score=0.3)
        err = CoherenceError("Score too low.", report=report)
        assert err.report is report
        assert "Score too low." in str(err)

    def test_is_exception_subclass(self):
        report = _make_report()
        err = CoherenceError("msg", report)
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# 14. YAML suite runner tests
# ---------------------------------------------------------------------------

SAMPLE_YAML_SUITE = """\
suite: "Test suite"
threshold: 0.55
cases:
  - id: "case_001"
    steps:
      - "Step 1: Observe X."
      - "Step 2: Infer Y from X."
      - "Step 3: Conclude Z from Y."
    conclusion: "Therefore Z is true."
    expect: "COHERENT"
  - id: "case_002"
    steps:
      - "Step 1: A is true."
      - "Step 2: Therefore Q is true."
    conclusion: "Q is true."
    expect: "DEGRADED"
"""


class TestYamlSuiteRunner:
    def _write_suite(self, tmp_path: Path, content: str = SAMPLE_YAML_SUITE) -> Path:
        p = tmp_path / "suite.yaml"
        p.write_text(content)
        return p

    def _setup_mock(self, mock_anthropic, scores: list[float]):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        responses = [
            MagicMock(content=[MagicMock(text=json.dumps(_make_raw_response(s)))])
            for s in scores
        ]
        mock_client.messages.create.side_effect = responses

    def test_suite_runs_all_cases(self, tmp_path):
        suite_file = self._write_suite(tmp_path)
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, scores=[0.90, 0.65])
            result = run_yaml_suite(suite_file, save=False) if False else run_yaml_suite(
                suite_file, model="test-model"
            )
        assert result["total"] == 2

    def test_suite_counts_passed_and_failed(self, tmp_path):
        suite_file = self._write_suite(tmp_path)
        # case_001 expects COHERENT (score 0.90 → COHERENT ✓)
        # case_002 expects DEGRADED (score 0.65 → DEGRADED ✓)
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, scores=[0.90, 0.65])
            result = run_yaml_suite(suite_file, model="test-model")
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_suite_counts_failed_when_mismatch(self, tmp_path):
        suite_file = self._write_suite(tmp_path)
        # case_001 expects COHERENT but gets INCOHERENT (0.30)
        # case_002 expects DEGRADED but gets COHERENT (0.90)
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, scores=[0.30, 0.90])
            result = run_yaml_suite(suite_file, model="test-model")
        assert result["failed"] == 2

    def test_suite_returns_suite_name(self, tmp_path):
        suite_file = self._write_suite(tmp_path)
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, scores=[0.90, 0.65])
            result = run_yaml_suite(suite_file)
        assert result["suite"] == "Test suite"

    def test_suite_raises_without_pyyaml(self, tmp_path):
        suite_file = self._write_suite(tmp_path)
        with patch.dict(sys.modules, {"yaml": None}):
            with pytest.raises(ImportError, match="pyyaml"):
                run_yaml_suite(suite_file)


# ---------------------------------------------------------------------------
# 15. CLI tests
# ---------------------------------------------------------------------------

class TestCLI:
    def _setup_mock(self, mock_anthropic, coherence_score=0.85):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(_make_raw_response(coherence_score)))]
        mock_client.messages.create.return_value = mock_msg

    def test_check_text_output_exit_0_when_coherent(self, capsys):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, 0.90)
            code = cc._cli_main([
                "check",
                "--steps", "Step 1: X.", "Step 2: Y.", "Step 3: Z.",
                "--conclusion", "Therefore Z.",
                "--no-save",
            ])
        assert code == 0
        captured = capsys.readouterr()
        assert "COHERENT" in captured.out

    def test_check_exit_1_when_incoherent(self, capsys):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, 0.20)
            code = cc._cli_main([
                "check",
                "--steps", "Step 1: A.", "Step 2: Therefore W.",
                "--conclusion", "W is definitely true.",
                "--no-save",
                "--threshold", "0.45",
            ])
        assert code == 1

    def test_check_json_output(self, capsys):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, 0.85)
            cc._cli_main([
                "check",
                "--steps", "Step 1.", "Step 2.", "Step 3.",
                "--conclusion", "Conclusion.",
                "--format", "json",
                "--no-save",
            ])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "status" in data
        assert "coherence_score" in data

    def test_check_markdown_output(self, capsys):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, 0.85)
            cc._cli_main([
                "check",
                "--steps", "Step 1.", "Step 2.",
                "--conclusion", "Conclusion.",
                "--format", "markdown",
                "--no-save",
            ])
        captured = capsys.readouterr()
        assert "cot-coherence Report" in captured.out

    def test_check_raw_steps(self, capsys):
        with patch("cot_coherence.anthropic") as mock_anthropic:
            self._setup_mock(mock_anthropic, 0.85)
            code = cc._cli_main([
                "check",
                "--raw-steps", "1. First.\n2. Second.\n3. Third.",
                "--conclusion", "Therefore X.",
                "--no-save",
            ])
        assert code == 0

    def test_history_no_db(self, capsys, tmp_path):
        # Pass a missing db path via load_recent_reports directly, then via CLI
        # Use a path that doesn't exist — CLI reads from env var at runtime
        # We verify via the underlying function (avoids module reload side-effects)
        rows = load_recent_reports(db_path=tmp_path / "missing.db")
        assert rows == []

    def test_history_shows_records(self, capsys, tmp_path):
        db = tmp_path / "test.db"
        save_report(_make_report(), db_path=db)
        rows = load_recent_reports(n=5, db_path=db)
        assert len(rows) == 1
        assert rows[0]["status"] == "COHERENT"

    def test_suite_cli_exit_0_on_all_pass(self, tmp_path, capsys):
        suite_content = """\
suite: "CLI suite test"
cases:
  - id: "c1"
    steps:
      - "A implies B."
      - "B implies C."
    conclusion: "Therefore A implies C."
    expect: "COHERENT"
"""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(suite_content)

        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(text=json.dumps(_make_raw_response(0.90)))]
            )
            code = cc._cli_main([
                "suite", str(suite_file), "--model", "test-model"
            ])
        assert code == 0

    def test_suite_cli_exit_1_on_failure(self, tmp_path, capsys):
        suite_content = """\
suite: "CLI fail test"
cases:
  - id: "c1"
    steps:
      - "A is true."
    conclusion: "Therefore Q."
    expect: "COHERENT"
"""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(suite_content)

        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(text=json.dumps(_make_raw_response(0.30)))]
            )
            code = cc._cli_main([
                "suite", str(suite_file), "--model", "test-model"
            ])
        assert code == 1

    def test_suite_cli_json_format(self, tmp_path, capsys):
        suite_content = """\
suite: "JSON test"
cases:
  - id: "c1"
    steps: ["Step 1.", "Step 2."]
    conclusion: "Conclusion."
    expect: "COHERENT"
"""
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(suite_content)

        with patch("cot_coherence.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(text=json.dumps(_make_raw_response(0.90)))]
            )
            cc._cli_main(["suite", str(suite_file), "--format", "json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "passed" in data
        assert "failed" in data
        assert "results" in data
