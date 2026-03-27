"""
Tests for drift-guard core functionality.
Tests that do NOT require the LLM API are fast; LLM tests use mocks.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from drift_guard import (
    parse_intent,
    parse_diff,
    diff_stats,
    DriftReport,
    VerifyStatus,
    IntentClause,
    ClauseVerification,
    save_report,
    load_recent_reports,
)


# ---------------------------------------------------------------------------
# Intent Parser Tests
# ---------------------------------------------------------------------------

class TestParseIntent:
    def test_adds_clause(self):
        clauses = parse_intent("Adds None check for email field")
        assert len(clauses) >= 1
        assert any(c.clause_type == "adds" for c in clauses)

    def test_fixes_clause(self):
        clauses = parse_intent("Fixes null pointer exception in register_user")
        assert any(c.clause_type == "fixes" for c in clauses)

    def test_removes_clause(self):
        clauses = parse_intent("Removes deprecated stripe.charge() calls")
        assert any(c.clause_type == "removes" for c in clauses)

    def test_does_not_clause(self):
        clauses = parse_intent("Does not change the public API of charge_customer")
        assert any(c.clause_type == "does_not" for c in clauses)

    def test_multi_clause_description(self):
        desc = """
        - Adds validation for email field
        - Removes the deprecated customer.sources usage
        - Does not change the public API
        - Fixes null pointer in registration flow
        """
        clauses = parse_intent(desc)
        types = {c.clause_type for c in clauses}
        assert "adds" in types
        assert "removes" in types
        assert "does_not" in types
        assert "fixes" in types

    def test_empty_description(self):
        clauses = parse_intent("")
        assert clauses == []

    def test_very_short_description(self):
        clauses = parse_intent("fix")
        assert len(clauses) == 0

    def test_caps_at_12_clauses(self):
        desc = "\n".join(f"Adds feature {i}" for i in range(20))
        clauses = parse_intent(desc)
        assert len(clauses) <= 12

    def test_modifies_default_clause(self):
        clauses = parse_intent("Updates the configuration parsing logic in settings module")
        assert any(c.clause_type == "modifies" for c in clauses)

    def test_ensures_clause(self):
        clauses = parse_intent("Ensures the email field is validated before insert")
        assert any(c.clause_type == "ensures" for c in clauses)


# ---------------------------------------------------------------------------
# Diff Parser Tests
# ---------------------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -45,6 +45,10 @@ def register_user(email, password):
     \"\"\"Register a new user.\"\"\"
+    if email is None:
+        raise ValueError("email cannot be None")
+    if not email.strip():
+        raise ValueError("email cannot be empty")
     email = email.lower()
     return db.create_user(email, hash_password(password))

diff --git a/tests/test_auth.py b/tests/test_auth.py
index 111aaaa..222bbbb 100644
--- a/tests/test_auth.py
+++ b/tests/test_auth.py
@@ -88,3 +88,9 @@ class TestRegister:
+    def test_register_null_email(self):
+        with pytest.raises(ValueError):
+            register_user(None, "password123")
+
+    def test_register_empty_email(self):
+        with pytest.raises(ValueError):
+            register_user("", "password123")
"""

SAMPLE_DIFF_WITH_REMOVALS = """\
diff --git a/src/payment.py b/src/payment.py
index abc1234..def5678 100644
--- a/src/payment.py
+++ b/src/payment.py
@@ -10,8 +10,6 @@ def charge_customer(amount, customer_id):
-    # deprecated: use PaymentIntent instead
-    result = stripe.charge(amount=amount, customer=customer_id)
+    intent = stripe.PaymentIntent.create(amount=amount, customer=customer_id)
+    return intent.id
"""


class TestParseDiff:
    def test_parses_hunks(self):
        hunks = parse_diff(SAMPLE_DIFF)
        assert len(hunks) >= 2

    def test_extracts_file_paths(self):
        hunks = parse_diff(SAMPLE_DIFF)
        paths = {h.file_path for h in hunks}
        assert "src/auth.py" in paths
        assert "tests/test_auth.py" in paths

    def test_extracts_added_lines(self):
        hunks = parse_diff(SAMPLE_DIFF)
        all_added = [l for h in hunks for l in h.lines_added]
        assert any("email is None" in l for l in all_added)

    def test_extracts_removed_lines(self):
        hunks = parse_diff(SAMPLE_DIFF)
        all_removed = [l for h in hunks for l in h.lines_removed]
        assert len(all_removed) == 0

    def test_extracts_removals_from_diff_with_removals(self):
        hunks = parse_diff(SAMPLE_DIFF_WITH_REMOVALS)
        all_removed = [l for h in hunks for l in h.lines_removed]
        assert len(all_removed) > 0
        assert any("stripe.charge" in l for l in all_removed)

    def test_diff_stats(self):
        files, added, removed = diff_stats(SAMPLE_DIFF)
        assert "src/auth.py" in files
        assert "tests/test_auth.py" in files
        assert added > 0
        assert removed == 0

    def test_diff_stats_with_removals(self):
        files, added, removed = diff_stats(SAMPLE_DIFF_WITH_REMOVALS)
        assert "src/payment.py" in files
        assert added > 0
        assert removed > 0

    def test_empty_diff(self):
        hunks = parse_diff("")
        assert hunks == []

    def test_diff_hunk_is_pure_addition(self):
        hunks = parse_diff(SAMPLE_DIFF)
        auth_hunks = [h for h in hunks if h.file_path == "src/auth.py"]
        assert any(h.is_pure_addition for h in auth_hunks)

    def test_diff_hunk_is_pure_removal(self):
        hunks = parse_diff(SAMPLE_DIFF_WITH_REMOVALS)
        # Hunk has both additions and removals so it's neither pure
        payment_hunks = [h for h in hunks if h.file_path == "src/payment.py"]
        assert len(payment_hunks) > 0


# ---------------------------------------------------------------------------
# DriftReport Tests
# ---------------------------------------------------------------------------

def make_sample_report(status: VerifyStatus = VerifyStatus.PASS, drift: float = 0.05) -> DriftReport:
    clause = IntentClause(text="Adds None check for email", clause_type="adds", subject="None check")
    verification = ClauseVerification(
        clause=clause,
        status=VerifyStatus.PASS,
        evidence="if email is None:",
        confidence=0.95,
        explanation="Diff shows None check added in register_user()",
    )
    return DriftReport(
        pr_title="Fix null pointer in user registration",
        pr_description="Adds None check",
        intent_summary="PR fulfills stated intent.",
        clauses=[clause],
        verifications=[verification],
        overall_status=status,
        overall_confidence=0.92,
        files_changed=["src/auth.py", "tests/test_auth.py"],
        lines_added=10,
        lines_removed=0,
        drift_score=drift,
        timestamp="2026-03-27T12:00:00+00:00",
        model_used="claude-haiku-4-5-20251001",
        commit_sha="abc1234",
    )


class TestDriftReport:
    def test_passed_when_pass(self):
        r = make_sample_report(VerifyStatus.PASS)
        assert r.passed() is True

    def test_not_passed_when_fail(self):
        r = make_sample_report(VerifyStatus.FAIL, drift=0.8)
        assert r.passed() is False

    def test_not_passed_when_warn(self):
        r = make_sample_report(VerifyStatus.WARN, drift=0.4)
        assert r.passed() is False

    def test_not_passed_when_skip(self):
        r = make_sample_report(VerifyStatus.SKIP, drift=0.0)
        assert r.passed() is False

    def test_to_dict_keys(self):
        r = make_sample_report()
        d = r.to_dict()
        assert "pr_title" in d
        assert "overall_status" in d
        assert "drift_score" in d
        assert "clauses" in d
        assert "intent_summary" in d
        assert "timestamp" in d
        assert "model_used" in d
        assert "commit_sha" in d

    def test_to_dict_serializable(self):
        r = make_sample_report()
        d = r.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_dict_clause_structure(self):
        r = make_sample_report()
        d = r.to_dict()
        assert len(d["clauses"]) == 1
        clause = d["clauses"][0]
        assert "text" in clause
        assert "status" in clause
        assert "confidence" in clause
        assert "explanation" in clause
        assert "evidence" in clause

    def test_to_markdown_contains_status(self):
        r = make_sample_report()
        md = r.to_markdown()
        assert "PASS" in md
        assert "drift-guard" in md.lower()

    def test_to_markdown_contains_clauses(self):
        r = make_sample_report()
        md = r.to_markdown()
        assert "Adds None check" in md

    def test_to_markdown_contains_drift_score(self):
        r = make_sample_report()
        md = r.to_markdown()
        assert "0.05" in md

    def test_to_markdown_fail_report(self):
        r = make_sample_report(VerifyStatus.FAIL, drift=0.85)
        md = r.to_markdown()
        assert "FAIL" in md


# ---------------------------------------------------------------------------
# SQLite Trace Log Tests
# ---------------------------------------------------------------------------

class TestSQLiteLog:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            report = make_sample_report()
            save_report(report, db_path=db_path)
            rows = load_recent_reports(n=5, db_path=db_path)
            assert len(rows) == 1
            assert rows[0]["status"] == "PASS"
            assert rows[0]["drift_score"] == pytest.approx(0.05)

    def test_empty_db_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nonexistent.db"
            rows = load_recent_reports(db_path=db_path)
            assert rows == []

    def test_save_multiple_and_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            for i in range(5):
                save_report(make_sample_report(), db_path=db_path)
            rows = load_recent_reports(n=3, db_path=db_path)
            assert len(rows) == 3

    def test_save_multiple_and_order_desc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            for drift in [0.1, 0.5, 0.9]:
                save_report(make_sample_report(drift=drift), db_path=db_path)
            rows = load_recent_reports(n=10, db_path=db_path)
            assert len(rows) == 3
            # Most recent first
            assert rows[0]["drift_score"] == pytest.approx(0.9)

    def test_load_row_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_report(make_sample_report(), db_path=db_path)
            rows = load_recent_reports(db_path=db_path)
            row = rows[0]
            assert "timestamp" in row
            assert "commit" in row
            assert "pr_title" in row
            assert "status" in row
            assert "drift_score" in row


# ---------------------------------------------------------------------------
# LLM Verifier Tests (mocked)
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = {
    "clauses": [
        {
            "clause_index": 0,
            "status": "PASS",
            "confidence": 0.95,
            "evidence": "if email is None:",
            "explanation": "Diff shows None check added in register_user()"
        }
    ],
    "overall_status": "PASS",
    "overall_confidence": 0.95,
    "drift_score": 0.04,
    "intent_summary": "PR claims to add None check; diff confirms this exactly."
}


class TestVerifyWithLLM:
    @patch("drift_guard.anthropic")
    def test_verify_returns_report(self, mock_anthropic_module):
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(MOCK_LLM_RESPONSE))]
        mock_client.messages.create.return_value = mock_message

        from drift_guard import verify
        report = verify(
            pr_title="Fix null pointer in user registration",
            pr_description="Adds None check before calling .lower() on email field",
            diff=SAMPLE_DIFF,
            save=False,
        )

        assert report.overall_status == VerifyStatus.PASS
        assert report.drift_score == pytest.approx(0.04)
        assert report.passed() is True
        assert len(report.verifications) == 1

    @patch("drift_guard.anthropic")
    def test_verify_fail_returns_non_zero_drift(self, mock_anthropic_module):
        fail_response = {
            "clauses": [
                {
                    "clause_index": 0,
                    "status": "FAIL",
                    "confidence": 0.9,
                    "evidence": "",
                    "explanation": "No changes to payment processor found in diff"
                }
            ],
            "overall_status": "FAIL",
            "overall_confidence": 0.9,
            "drift_score": 0.85,
            "intent_summary": "PR claims to update payment processor; diff only modifies auth."
        }
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(fail_response))]
        mock_client.messages.create.return_value = mock_message

        from drift_guard import verify
        report = verify(
            pr_title="Refactor payment processor",
            pr_description="Updates stripe.charge() to PaymentIntent.create()",
            diff=SAMPLE_DIFF,
            save=False,
        )

        assert report.overall_status == VerifyStatus.FAIL
        assert report.drift_score > 0.3
        assert not report.passed()

    @patch("drift_guard.anthropic")
    def test_verify_warn_report(self, mock_anthropic_module):
        warn_response = {
            "clauses": [
                {
                    "clause_index": 0,
                    "status": "WARN",
                    "confidence": 0.7,
                    "evidence": "partial change found",
                    "explanation": "Only part of the stated change was found in diff"
                }
            ],
            "overall_status": "WARN",
            "overall_confidence": 0.7,
            "drift_score": 0.4,
            "intent_summary": "PR partially fulfills intent."
        }
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(warn_response))]
        mock_client.messages.create.return_value = mock_message

        from drift_guard import verify
        report = verify(
            pr_title="Refactor payment processor",
            pr_description="Updates stripe.charge() to PaymentIntent.create()",
            diff=SAMPLE_DIFF,
            save=False,
        )

        assert report.overall_status == VerifyStatus.WARN
        assert not report.passed()

    @patch("drift_guard.anthropic")
    def test_verify_no_clauses_returns_skip(self, mock_anthropic_module):
        from drift_guard import verify
        report = verify(
            pr_title="x",
            pr_description="y",
            diff=SAMPLE_DIFF,
            save=False,
        )

        assert report.overall_status == VerifyStatus.SKIP
        assert report.verifications == []

    @patch("drift_guard.anthropic")
    def test_verify_markdown_codeblock_stripped(self, mock_anthropic_module):
        """Verify that JSON wrapped in markdown code blocks is handled."""
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        # Simulate LLM wrapping response in ```json ... ```
        wrapped = f"```json\n{json.dumps(MOCK_LLM_RESPONSE)}\n```"
        mock_message.content = [MagicMock(text=wrapped)]
        mock_client.messages.create.return_value = mock_message

        from drift_guard import verify
        report = verify(
            pr_title="Fix null pointer in user registration",
            pr_description="Adds None check before calling .lower() on email field",
            diff=SAMPLE_DIFF,
            save=False,
        )

        assert report.overall_status == VerifyStatus.PASS
        assert report.drift_score == pytest.approx(0.04)
