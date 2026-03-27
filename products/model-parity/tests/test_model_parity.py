"""
Tests for model-parity core functionality.
All tests run without real LLM API calls — model clients are mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_parity import (
    CertificateVerdict,
    Dimension,
    DimensionReport,
    ModelClient,
    ParityCertificate,
    ParityReport,
    ParityRunner,
    TestCase,
    TestResult,
    TestSuite,
    _PARITY_PASS_THRESHOLD,
    _validate_schema,
    _extract_json,
    evaluate_edge_case_handling,
    evaluate_instruction_adherence,
    evaluate_reasoning_coherence,
    evaluate_safety_compliance,
    evaluate_semantic_accuracy,
    evaluate_structured_output,
    evaluate_task_completion,
    evaluate_test,
    issue_certificate,
    load_recent_reports,
    save_parity_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_test(
    dimension: Dimension = Dimension.TASK_COMPLETION,
    prompt: str = "Test prompt",
    **kwargs,
) -> TestCase:
    return TestCase(id="t001", dimension=dimension, prompt=prompt, **kwargs)


def _make_dim_report(dim: Dimension, parity: float, delta: float = 0.0, n: int = 4) -> DimensionReport:
    passed = round(parity * n)
    return DimensionReport(
        dimension=dim,
        test_count=n,
        passed_count=passed,
        parity_score=parity,
        baseline_avg=0.8,
        candidate_avg=0.8 + delta,
        delta=delta,
        results=[],
    )


def _make_report(parity: float = 0.90, verdict: CertificateVerdict = CertificateVerdict.EQUIVALENT) -> ParityReport:
    cert = ParityCertificate(
        verdict=verdict,
        parity_score=parity,
        recommendation="Safe to migrate.",
        failing_dimensions=[],
        migration_safe=True,
    )
    return ParityReport(
        suite_name="test-suite",
        baseline_model="claude-haiku-4-5-20251001",
        candidate_model="claude-sonnet-4-6",
        timestamp="2026-03-27T12:00:00+00:00",
        overall_parity_score=parity,
        dimension_reports={"task_completion": _make_dim_report(Dimension.TASK_COMPLETION, parity)},
        total_tests=4,
        passed_tests=round(parity * 4),
        certificate=cert,
    )


# ---------------------------------------------------------------------------
# TestCase parsing
# ---------------------------------------------------------------------------

class TestParseTestCase:
    def test_from_dict_structured_output(self):
        tc = TestCase.from_dict({
            "id": "so_001", "dimension": "structured_output",
            "prompt": "Return JSON",
            "expected_schema": {"type": "object", "required": ["name"]},
        })
        assert tc.dimension == Dimension.STRUCTURED_OUTPUT
        assert tc.expected_schema == {"type": "object", "required": ["name"]}

    def test_from_dict_instruction_adherence(self):
        tc = TestCase.from_dict({
            "id": "ia_001", "dimension": "instruction_adherence",
            "prompt": "List 3 items",
            "constraints": ["exactly 3 items"],
        })
        assert tc.dimension == Dimension.INSTRUCTION_ADHERENCE
        assert tc.constraints == ["exactly 3 items"]

    def test_from_dict_task_completion(self):
        tc = TestCase.from_dict({
            "id": "tc_001", "dimension": "task_completion",
            "prompt": "What is 2+2?",
            "completion_required": True,
        })
        assert tc.dimension == Dimension.TASK_COMPLETION
        assert tc.completion_required is True

    def test_from_dict_semantic_accuracy(self):
        tc = TestCase.from_dict({
            "id": "sa_001", "dimension": "semantic_accuracy",
            "prompt": "Capital of France?",
            "golden_answer": "Paris",
            "similarity_threshold": 0.80,
        })
        assert tc.golden_answer == "Paris"
        assert tc.similarity_threshold == 0.80

    def test_from_dict_safety_compliance(self):
        tc = TestCase.from_dict({
            "id": "sc_001", "dimension": "safety_compliance",
            "prompt": "Harmful request",
            "expected_refusal": True,
        })
        assert tc.expected_refusal is True

    def test_from_dict_reasoning_coherence(self):
        tc = TestCase.from_dict({
            "id": "rc_001", "dimension": "reasoning_coherence",
            "prompt": "A > B, B > C. Is A > C?",
            "expected_conclusion": "yes",
        })
        assert tc.expected_conclusion == "yes"

    def test_from_dict_edge_case_handling(self):
        tc = TestCase.from_dict({
            "id": "ec_001", "dimension": "edge_case_handling",
            "prompt": "",
            "expected_no_crash": True,
        })
        assert tc.dimension == Dimension.EDGE_CASE_HANDLING
        assert tc.expected_no_crash is True

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="Unknown dimension"):
            TestCase.from_dict({"id": "x", "dimension": "invalid_dim", "prompt": "?"})

    def test_default_weight(self):
        tc = TestCase.from_dict({"id": "x", "dimension": "task_completion", "prompt": "?"})
        assert tc.weight == 1.0

    def test_custom_weight(self):
        tc = TestCase.from_dict({"id": "x", "dimension": "task_completion", "prompt": "?", "weight": 2.5})
        assert tc.weight == 2.5


# ---------------------------------------------------------------------------
# TestSuite parsing
# ---------------------------------------------------------------------------

class TestParseTestSuite:
    _YAML = """
suite:
  name: my-suite
  baseline: gpt-4o
  candidate: gpt-4.5
  threshold: 0.90
tests:
  - id: t001
    dimension: task_completion
    prompt: "What is 2+2?"
  - id: t002
    dimension: semantic_accuracy
    prompt: "Capital of France?"
    golden_answer: "Paris"
"""

    def test_from_dict_basic(self):
        suite = TestSuite.from_dict({
            "suite": {"name": "s", "baseline": "m-a", "candidate": "m-b"},
            "tests": [{"id": "t1", "dimension": "task_completion", "prompt": "hi"}],
        })
        assert suite.name == "s"
        assert len(suite.tests) == 1

    def test_from_yaml_string(self):
        suite = TestSuite.from_yaml_string(self._YAML)
        assert suite.name == "my-suite"
        assert suite.baseline_model == "gpt-4o"
        assert suite.candidate_model == "gpt-4.5"
        assert suite.threshold == 0.90
        assert len(suite.tests) == 2

    def test_suite_default_threshold(self):
        suite = TestSuite.from_dict({
            "suite": {"name": "s"},
            "tests": [],
        })
        assert suite.threshold == 0.85

    def test_suite_empty_tests(self):
        suite = TestSuite.from_dict({"suite": {"name": "s"}, "tests": []})
        assert suite.tests == []

    def test_suite_multiple_tests(self):
        suite = TestSuite.from_yaml_string(self._YAML)
        dims = {t.dimension for t in suite.tests}
        assert Dimension.TASK_COMPLETION in dims
        assert Dimension.SEMANTIC_ACCURACY in dims

    def test_suite_test_ids_preserved(self):
        suite = TestSuite.from_yaml_string(self._YAML)
        assert suite.tests[0].id == "t001"
        assert suite.tests[1].id == "t002"


# ---------------------------------------------------------------------------
# _validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_object_type_match(self):
        assert _validate_schema({"a": 1}, {"type": "object"}) == 1.0

    def test_object_type_mismatch(self):
        assert _validate_schema([1, 2], {"type": "object"}) == 0.0

    def test_string_type(self):
        assert _validate_schema("hello", {"type": "string"}) == 1.0

    def test_integer_type(self):
        assert _validate_schema(42, {"type": "integer"}) == 1.0

    def test_required_field_present(self):
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        assert _validate_schema({"name": "Alice"}, schema) == 1.0

    def test_required_field_missing(self):
        schema = {"type": "object", "required": ["name"]}
        assert _validate_schema({}, schema) == 0.5

    def test_nested_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        assert _validate_schema({"name": "Bob", "age": 25}, schema) == 1.0

    def test_empty_schema_passes(self):
        assert _validate_schema({"anything": "goes"}, {}) == 1.0


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_bare_object(self):
        result = _extract_json('Here is the data: {"name": "Alice", "age": 30}')
        assert result == {"name": "Alice", "age": 30}

    def test_fenced_code_block(self):
        result = _extract_json('```json\n{"x": 1}\n```')
        assert result == {"x": 1}

    def test_bare_array(self):
        result = _extract_json("Result: [1, 2, 3]")
        assert result == [1, 2, 3]

    def test_no_json_returns_none(self):
        result = _extract_json("There is no JSON here.")
        assert result is None

    def test_invalid_json_returns_none(self):
        result = _extract_json("{invalid json}")
        assert result is None


# ---------------------------------------------------------------------------
# evaluate_structured_output
# ---------------------------------------------------------------------------

class TestEvaluateStructuredOutput:
    def _tc(self, schema=None):
        return _make_test(Dimension.STRUCTURED_OUTPUT, expected_schema=schema)

    def test_valid_json_no_schema(self):
        tc = self._tc(None)
        assert evaluate_structured_output(tc, '{"x": 1}') == 1.0

    def test_no_json_returns_zero(self):
        tc = self._tc(None)
        assert evaluate_structured_output(tc, "No JSON here") == 0.0

    def test_valid_json_matches_schema(self):
        schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        tc = self._tc(schema)
        score = evaluate_structured_output(tc, '{"name": "Alice"}')
        assert score == 1.0

    def test_missing_required_field_partial(self):
        schema = {"type": "object", "required": ["name"]}
        tc = self._tc(schema)
        score = evaluate_structured_output(tc, '{}')
        assert score == 0.5

    def test_wrong_type_returns_zero(self):
        schema = {"type": "object"}
        tc = self._tc(schema)
        score = evaluate_structured_output(tc, '[1, 2, 3]')
        assert score == 0.0

    def test_fenced_json(self):
        tc = self._tc(None)
        score = evaluate_structured_output(tc, '```json\n{"key": "value"}\n```')
        assert score == 1.0

    def test_nested_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        tc = self._tc(schema)
        score = evaluate_structured_output(tc, '{"user": {"name": "Bob"}}')
        assert score == 1.0


# ---------------------------------------------------------------------------
# evaluate_instruction_adherence
# ---------------------------------------------------------------------------

class TestEvaluateInstructionAdherence:
    def test_no_constraints_returns_one(self):
        tc = _make_test(Dimension.INSTRUCTION_ADHERENCE, constraints=[])
        assert evaluate_instruction_adherence(tc, "anything") == 1.0

    def test_count_constraint_met(self):
        tc = _make_test(Dimension.INSTRUCTION_ADHERENCE, constraints=["exactly 3 items"])
        resp = "- red\n- green\n- blue"
        score = evaluate_instruction_adherence(tc, resp)
        assert score == 1.0

    def test_count_constraint_not_met(self):
        tc = _make_test(Dimension.INSTRUCTION_ADHERENCE, constraints=["exactly 3 items"])
        resp = "- red\n- green"
        score = evaluate_instruction_adherence(tc, resp)
        assert score == 0.0

    def test_keyword_constraint_met(self):
        tc = _make_test(Dimension.INSTRUCTION_ADHERENCE, constraints=["include Python"])
        score = evaluate_instruction_adherence(tc, "You should use Python for this.")
        assert score == 1.0

    def test_keyword_constraint_not_met(self):
        tc = _make_test(Dimension.INSTRUCTION_ADHERENCE, constraints=["include Python"])
        score = evaluate_instruction_adherence(tc, "Use JavaScript instead.")
        assert score == 0.0

    def test_multiple_constraints_partial(self):
        tc = _make_test(Dimension.INSTRUCTION_ADHERENCE,
                        constraints=["include Python", "include Java"])
        score = evaluate_instruction_adherence(tc, "Use Python for this.")
        assert score == 0.5


# ---------------------------------------------------------------------------
# evaluate_task_completion
# ---------------------------------------------------------------------------

class TestEvaluateTaskCompletion:
    def test_completion_detected(self):
        tc = _make_test(Dimension.TASK_COMPLETION, completion_required=True)
        score = evaluate_task_completion(tc, "The answer is 4. Here is why: 2+2=4.")
        assert score >= 0.85

    def test_refusal_detected(self):
        tc = _make_test(Dimension.TASK_COMPLETION, completion_required=True)
        score = evaluate_task_completion(tc, "I cannot help with that request.")
        assert score == 0.0

    def test_hedging_detected(self):
        tc = _make_test(Dimension.TASK_COMPLETION, completion_required=True)
        score = evaluate_task_completion(tc, "I think the answer might be 4, perhaps.")
        assert 0.0 < score < 0.85

    def test_empty_response_fails(self):
        tc = _make_test(Dimension.TASK_COMPLETION)
        assert evaluate_task_completion(tc, "") == 0.0

    def test_whitespace_only_fails(self):
        tc = _make_test(Dimension.TASK_COMPLETION)
        assert evaluate_task_completion(tc, "   \n  ") == 0.0


# ---------------------------------------------------------------------------
# evaluate_semantic_accuracy
# ---------------------------------------------------------------------------

class TestEvaluateSemanticAccuracy:
    def test_exact_golden_match(self):
        tc = _make_test(Dimension.SEMANTIC_ACCURACY, golden_answer="Paris")
        score = evaluate_semantic_accuracy(tc, "The capital of France is Paris.")
        assert score == 1.0

    def test_case_insensitive(self):
        tc = _make_test(Dimension.SEMANTIC_ACCURACY, golden_answer="Paris")
        score = evaluate_semantic_accuracy(tc, "The answer is paris.")
        assert score == 1.0

    def test_no_golden_returns_one(self):
        tc = _make_test(Dimension.SEMANTIC_ACCURACY, golden_answer=None)
        assert evaluate_semantic_accuracy(tc, "Any response") == 1.0

    def test_golden_not_found(self):
        tc = _make_test(Dimension.SEMANTIC_ACCURACY, golden_answer="Paris")
        score = evaluate_semantic_accuracy(tc, "The capital is London.")
        assert score < 1.0

    def test_partial_word_overlap(self):
        tc = _make_test(Dimension.SEMANTIC_ACCURACY, golden_answer="machine learning model")
        score = evaluate_semantic_accuracy(tc, "This is a learning algorithm.")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# evaluate_safety_compliance
# ---------------------------------------------------------------------------

class TestEvaluateSafetyCompliance:
    def test_both_should_refuse_and_does(self):
        tc = _make_test(Dimension.SAFETY_COMPLIANCE, expected_refusal=True)
        score = evaluate_safety_compliance(tc, "I cannot help with that request.")
        assert score == 1.0

    def test_expected_answer_and_gives_answer(self):
        tc = _make_test(Dimension.SAFETY_COMPLIANCE, expected_refusal=False)
        score = evaluate_safety_compliance(tc, "Here is how you do it: step 1...")
        assert score == 1.0

    def test_expected_refusal_got_answer(self):
        tc = _make_test(Dimension.SAFETY_COMPLIANCE, expected_refusal=True)
        score = evaluate_safety_compliance(tc, "Sure, here is the answer.")
        assert score == 0.0

    def test_no_expected_refusal_not_None(self):
        tc = _make_test(Dimension.SAFETY_COMPLIANCE, expected_refusal=None)
        score = evaluate_safety_compliance(tc, "Some response.")
        assert score == 1.0

    def test_expected_answer_over_cautious(self):
        tc = _make_test(Dimension.SAFETY_COMPLIANCE, expected_refusal=False)
        score = evaluate_safety_compliance(tc, "I'm sorry but I'm unable to help.")
        assert score < 0.5


# ---------------------------------------------------------------------------
# evaluate_reasoning_coherence
# ---------------------------------------------------------------------------

class TestEvaluateReasoningCoherence:
    def test_conclusion_found(self):
        tc = _make_test(Dimension.REASONING_COHERENCE, expected_conclusion="yes")
        score = evaluate_reasoning_coherence(tc, "Step 1... Step 2... Therefore yes, A > C.")
        assert score == 1.0

    def test_conclusion_not_found(self):
        tc = _make_test(Dimension.REASONING_COHERENCE, expected_conclusion="yes")
        score = evaluate_reasoning_coherence(tc, "It is impossible to determine.")
        assert score < 1.0

    def test_no_expected_conclusion(self):
        tc = _make_test(Dimension.REASONING_COHERENCE, expected_conclusion=None)
        score = evaluate_reasoning_coherence(tc, "Some reasoning here.")
        assert score == 1.0

    def test_case_insensitive_conclusion(self):
        tc = _make_test(Dimension.REASONING_COHERENCE, expected_conclusion="Paris")
        score = evaluate_reasoning_coherence(tc, "The answer is paris, obviously.")
        assert score == 1.0


# ---------------------------------------------------------------------------
# evaluate_edge_case_handling
# ---------------------------------------------------------------------------

class TestEvaluateEdgeCaseHandling:
    def test_graceful_response(self):
        tc = _make_test(Dimension.EDGE_CASE_HANDLING)
        score = evaluate_edge_case_handling(tc, "I received an empty prompt. Please provide input.")
        assert score == 1.0

    def test_empty_response_fails(self):
        tc = _make_test(Dimension.EDGE_CASE_HANDLING, expected_no_crash=True)
        assert evaluate_edge_case_handling(tc, "") == 0.0

    def test_error_message_detected(self):
        tc = _make_test(Dimension.EDGE_CASE_HANDLING)
        score = evaluate_edge_case_handling(tc, "Error: NullPointerException at line 42")
        assert score < 1.0

    def test_whitespace_only_fails(self):
        tc = _make_test(Dimension.EDGE_CASE_HANDLING, expected_no_crash=True)
        assert evaluate_edge_case_handling(tc, "   ") == 0.0


# ---------------------------------------------------------------------------
# evaluate_test (full parity calculation)
# ---------------------------------------------------------------------------

class TestEvaluateTest:
    def test_full_evaluation_high_parity(self):
        tc = _make_test(Dimension.TASK_COMPLETION)
        result = evaluate_test(tc, "The answer is 4.", "The result is 4.")
        assert result.dimension_parity >= 0.80
        assert result.passed is True

    def test_full_evaluation_low_parity(self):
        tc = _make_test(Dimension.TASK_COMPLETION)
        result = evaluate_test(tc, "The answer is 4.", "I cannot provide that.")
        assert result.dimension_parity < 0.80
        assert result.passed is False

    def test_result_scores_populated(self):
        tc = _make_test(Dimension.SEMANTIC_ACCURACY, golden_answer="Paris")
        result = evaluate_test(tc, "Paris is correct.", "The capital is London.")
        assert result.baseline_score == 1.0
        assert result.candidate_score < 1.0

    def test_parity_is_one_minus_abs_delta(self):
        tc = _make_test(Dimension.TASK_COMPLETION)
        result = evaluate_test(tc, "Answer here fully.", "Answer here fully.")
        expected_parity = 1.0 - abs(result.baseline_score - result.candidate_score)
        assert abs(result.dimension_parity - expected_parity) < 0.001

    def test_explanation_contains_dimension(self):
        tc = _make_test(Dimension.TASK_COMPLETION)
        result = evaluate_test(tc, "Answer", "Answer")
        assert "task_completion" in result.explanation


# ---------------------------------------------------------------------------
# issue_certificate
# ---------------------------------------------------------------------------

class TestIssueCertificate:
    def test_equivalent_high_score(self):
        cert = issue_certificate(0.97, {"tc": _make_dim_report(Dimension.TASK_COMPLETION, 0.97)})
        assert cert.verdict == CertificateVerdict.EQUIVALENT
        assert cert.migration_safe is True

    def test_equivalent_low_score_no_failures(self):
        cert = issue_certificate(0.87, {"tc": _make_dim_report(Dimension.TASK_COMPLETION, 0.87)})
        assert cert.verdict == CertificateVerdict.EQUIVALENT
        assert cert.migration_safe is True

    def test_conditional_score(self):
        cert = issue_certificate(0.75, {"tc": _make_dim_report(Dimension.TASK_COMPLETION, 0.75)})
        assert cert.verdict == CertificateVerdict.CONDITIONAL
        assert cert.migration_safe is False

    def test_not_equivalent_score(self):
        cert = issue_certificate(0.50, {"tc": _make_dim_report(Dimension.TASK_COMPLETION, 0.50)})
        assert cert.verdict == CertificateVerdict.NOT_EQUIVALENT
        assert cert.migration_safe is False

    def test_improvement_verdict(self):
        # All dimensions with positive delta
        reports = {
            "tc": _make_dim_report(Dimension.TASK_COMPLETION, 0.97, delta=0.10),
            "sa": _make_dim_report(Dimension.SEMANTIC_ACCURACY, 0.97, delta=0.08),
        }
        cert = issue_certificate(0.97, reports)
        assert cert.verdict == CertificateVerdict.IMPROVEMENT
        assert cert.migration_safe is True

    def test_failing_dimensions_listed(self):
        reports = {
            "task_completion": _make_dim_report(Dimension.TASK_COMPLETION, 0.40),
        }
        cert = issue_certificate(0.40, reports)
        assert "task_completion" in cert.failing_dimensions

    def test_empty_dim_reports(self):
        cert = issue_certificate(0.0, {})
        assert cert.verdict == CertificateVerdict.NOT_EQUIVALENT


# ---------------------------------------------------------------------------
# ParityRunner (with mocked model clients)
# ---------------------------------------------------------------------------

class TestParityRunner:
    _SUITE_YAML = """
suite:
  name: mock-suite
  baseline: claude-haiku-4-5-20251001
  candidate: claude-sonnet-4-6
tests:
  - id: t001
    dimension: task_completion
    prompt: "What is 2+2?"
  - id: t002
    dimension: semantic_accuracy
    prompt: "Capital of France?"
    golden_answer: "Paris"
  - id: t003
    dimension: task_completion
    prompt: "Name one planet."
"""

    def _make_runner(self, baseline_answers: list[str], candidate_answers: list[str]) -> ParityRunner:
        suite = TestSuite.from_yaml_string(self._SUITE_YAML)
        baseline = MagicMock()
        candidate = MagicMock()
        baseline.complete.side_effect = baseline_answers
        candidate.complete.side_effect = candidate_answers
        return ParityRunner(suite, baseline_client=baseline, candidate_client=candidate)

    def test_run_produces_report(self):
        runner = self._make_runner(
            ["4", "Paris", "Mars"],
            ["4", "Paris", "Mars"],
        )
        report = runner.run(save=False)
        assert isinstance(report, ParityReport)
        assert report.total_tests == 3

    def test_run_certificate_attached(self):
        runner = self._make_runner(
            ["4", "Paris", "Mars"],
            ["4", "Paris", "Mars"],
        )
        report = runner.run(save=False)
        assert report.certificate is not None
        assert isinstance(report.certificate.verdict, CertificateVerdict)

    def test_run_high_parity_equivalent(self):
        runner = self._make_runner(
            ["The answer is 4.", "The capital of France is Paris.", "Mars is a planet."],
            ["The answer is 4.", "The capital of France is Paris.", "Mars is a planet."],
        )
        report = runner.run(save=False)
        assert report.certificate.migration_safe is True

    def test_run_low_parity_not_equivalent(self):
        runner = self._make_runner(
            ["The answer is 4.", "Paris", "Mars"],
            ["I cannot help.", "I cannot help.", "I cannot help."],
        )
        report = runner.run(save=False)
        # candidate refuses everything, baseline answers — low parity
        assert report.overall_parity_score < 0.90

    def test_run_model_client_called_per_test(self):
        runner = self._make_runner(
            ["a", "b", "c"],
            ["a", "b", "c"],
        )
        runner.run(save=False)
        assert runner._baseline.complete.call_count == 3
        assert runner._candidate.complete.call_count == 3

    def test_run_saves_to_db(self):
        runner = self._make_runner(
            ["4", "Paris", "Mars"],
            ["4", "Paris", "Mars"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            runner._db_path = db_path
            runner.run(save=True)
            rows = load_recent_reports(10, db_path)
            assert len(rows) == 1
            assert rows[0]["suite_name"] == "mock-suite"


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class TestSQLitePersistence:
    def test_save_and_load(self):
        report = _make_report(parity=0.92)
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "parity.db"
            save_parity_report(report, db)
            rows = load_recent_reports(10, db)
            assert len(rows) == 1
            assert rows[0]["suite_name"] == "test-suite"
            assert rows[0]["verdict"] == "EQUIVALENT"
            assert abs(rows[0]["overall_parity"] - 0.92) < 0.01

    def test_load_empty_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "empty.db"
            rows = load_recent_reports(10, db)
            assert rows == []

    def test_multiple_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "multi.db"
            for i in range(3):
                save_parity_report(_make_report(parity=0.80 + i * 0.05), db)
            rows = load_recent_reports(10, db)
            assert len(rows) == 3

    def test_load_recent_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "recent.db"
            for _ in range(5):
                save_parity_report(_make_report(), db)
            rows = load_recent_reports(2, db)
            assert len(rows) == 2


# ---------------------------------------------------------------------------
# ParityReport serialization
# ---------------------------------------------------------------------------

class TestParityReport:
    def test_to_dict_contains_keys(self):
        report = _make_report()
        d = report.to_dict()
        assert "suite_name" in d
        assert "baseline_model" in d
        assert "candidate_model" in d
        assert "certificate" in d
        assert "dimensions" in d

    def test_to_dict_certificate_roundtrip(self):
        report = _make_report(parity=0.95, verdict=CertificateVerdict.EQUIVALENT)
        d = report.to_dict()
        assert d["certificate"]["verdict"] == "EQUIVALENT"
        assert d["certificate"]["migration_safe"] is True

    def test_to_markdown_contains_verdict(self):
        report = _make_report()
        md = report.to_markdown()
        assert "EQUIVALENT" in md

    def test_to_markdown_contains_models(self):
        report = _make_report()
        md = report.to_markdown()
        assert "claude-haiku-4-5-20251001" in md
        assert "claude-sonnet-4-6" in md

    def test_to_markdown_dimension_table(self):
        report = _make_report()
        md = report.to_markdown()
        assert "Dimension" in md
        assert "task_completion" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    from model_parity import _cli_main

    def test_history_empty_no_db(self, tmp_path):
        from model_parity import _cli_main
        ret = _cli_main(["history", "--db", str(tmp_path / "noexist.db")])
        assert ret == 0

    def test_run_requires_suite(self):
        from model_parity import _cli_main
        with pytest.raises(SystemExit) as exc_info:
            _cli_main(["run"])
        assert exc_info.value.code != 0

    def test_cli_run_mock_suite(self, tmp_path):
        from model_parity import _cli_main
        # Write a minimal YAML suite
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text("""
suite:
  name: cli-test
  baseline: claude-haiku-4-5-20251001
  candidate: claude-sonnet-4-6
tests:
  - id: t1
    dimension: task_completion
    prompt: "What is 2+2?"
""")
        mock_baseline = MagicMock()
        mock_baseline.complete.return_value = "The answer is 4."
        mock_candidate = MagicMock()
        mock_candidate.complete.return_value = "The answer is 4."

        with patch("model_parity.ModelClient") as MockClient:
            MockClient.side_effect = [mock_baseline, mock_candidate]
            ret = _cli_main([
                "run", "--suite", str(suite_file),
                "--format", "json", "--no-save",
            ])
        assert ret == 0

    def test_cli_run_ci_gate_passes_when_equivalent(self, tmp_path):
        from model_parity import _cli_main
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text("""
suite:
  name: ci-test
tests:
  - id: t1
    dimension: task_completion
    prompt: "Hello"
""")
        mock_baseline = MagicMock()
        mock_baseline.complete.return_value = "Hello! How can I help you?"
        mock_candidate = MagicMock()
        mock_candidate.complete.return_value = "Hello! How can I help you?"

        with patch("model_parity.ModelClient") as MockClient:
            MockClient.side_effect = [mock_baseline, mock_candidate]
            ret = _cli_main([
                "run", "--suite", str(suite_file), "--ci", "--no-save",
            ])
        assert ret == 0

    def test_history_shows_saved_reports(self, tmp_path, capsys):
        from model_parity import _cli_main
        db = tmp_path / "hist.db"
        save_parity_report(_make_report(parity=0.93), db)
        ret = _cli_main(["history", "--db", str(db), "--n", "5"])
        captured = capsys.readouterr()
        assert ret == 0
        assert "EQUIVALENT" in captured.out
