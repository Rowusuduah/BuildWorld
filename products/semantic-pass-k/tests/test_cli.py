"""Tests for semantic_pass_k.cli"""
from __future__ import annotations
import json
import pytest

try:
    from click.testing import CliRunner
    from semantic_pass_k.cli import cli
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

pytestmark = pytest.mark.skipif(not HAS_CLICK, reason="click not installed")


@pytest.fixture
def runner():
    return CliRunner()


class TestRunCommand:
    def test_run_with_two_identical_outputs_passes(self, runner):
        result = runner.invoke(
            cli,
            ["run", "output A", "output A", "--criticality", "LOW"],
        )
        assert result.exit_code == 0

    def test_run_with_completely_different_outputs_exits_1(self, runner):
        result = runner.invoke(
            cli,
            ["run", "alpha beta gamma delta", "zeta eta theta iota", "--criticality", "CRITICAL"],
        )
        # Should be INCONSISTENT → exit code 1
        assert result.exit_code == 1

    def test_run_with_single_output_fails(self, runner):
        result = runner.invoke(cli, ["run", "only one"])
        assert result.exit_code != 0

    def test_json_output_flag(self, runner):
        result = runner.invoke(
            cli,
            ["run", "output A", "output B", "--criticality", "LOW", "--json-output"],
        )
        # Either passes or fails, but output should be valid JSON
        try:
            data = json.loads(result.output)
            assert "consistency_score" in data
            assert "verdict" in data
        except json.JSONDecodeError:
            pytest.fail("JSON output was not valid JSON")

    def test_run_stores_to_db(self, runner, tmp_path):
        db_path = str(tmp_path / "test.db")
        result = runner.invoke(
            cli,
            ["run", "output A", "output A",
             "--criticality", "LOW",
             "--store-db", db_path],
        )
        import os
        assert os.path.exists(db_path)

    def test_run_shows_score(self, runner):
        result = runner.invoke(
            cli,
            ["run", "same text here", "same text here", "--criticality", "LOW"],
        )
        assert "score" in result.output.lower() or "CONSISTENT" in result.output


class TestReportCommand:
    def test_report_empty_db(self, runner, tmp_path):
        db_path = str(tmp_path / "empty.db")
        result = runner.invoke(cli, ["report", "--store-db", db_path])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_report_shows_saved_results(self, runner, tmp_path):
        db_path = str(tmp_path / "with_results.db")
        # Save a result first
        runner.invoke(
            cli,
            ["run", "output A", "output A",
             "--criticality", "LOW",
             "--store-db", db_path],
        )
        result = runner.invoke(cli, ["report", "--store-db", db_path])
        assert result.exit_code == 0


class TestCiCommand:
    def test_ci_exits_0_no_results(self, runner, tmp_path):
        db_path = str(tmp_path / "empty.db")
        result = runner.invoke(cli, ["ci", "--store-db", db_path, "--label", "agent_x"])
        # No results → exit 0 with warning
        assert result.exit_code == 0

    def test_ci_exits_0_when_consistent(self, runner, tmp_path):
        db_path = str(tmp_path / "ci_test.db")
        runner.invoke(
            cli,
            ["run", "same output", "same output",
             "--criticality", "LOW",
             "--label", "my_agent",
             "--store-db", db_path],
        )
        result = runner.invoke(
            cli,
            ["ci", "--store-db", db_path, "--label", "my_agent"],
        )
        assert result.exit_code == 0

    def test_ci_exits_1_when_inconsistent(self, runner, tmp_path):
        db_path = str(tmp_path / "ci_fail.db")
        runner.invoke(
            cli,
            ["run",
             "alpha beta gamma delta epsilon",
             "zeta eta theta iota kappa",
             "--criticality", "CRITICAL",
             "--label", "bad_agent",
             "--store-db", db_path],
        )
        result = runner.invoke(
            cli,
            ["ci", "--store-db", db_path, "--label", "bad_agent"],
        )
        assert result.exit_code == 1


class TestBudgetCommand:
    def test_budget_missing_label_a_exits_1(self, runner, tmp_path):
        db_path = str(tmp_path / "budget.db")
        result = runner.invoke(cli, ["budget", "--store-db", db_path, "agent_a", "agent_b"])
        assert result.exit_code == 1

    def test_budget_compares_two_agents(self, runner, tmp_path):
        db_path = str(tmp_path / "budget2.db")
        # Store results for two agents
        runner.invoke(
            cli,
            ["run", "same", "same", "--criticality", "LOW",
             "--label", "agent_a", "--store-db", db_path],
        )
        runner.invoke(
            cli,
            ["run", "same", "same", "--criticality", "LOW",
             "--label", "agent_b", "--store-db", db_path],
        )
        result = runner.invoke(cli, ["budget", "--store-db", db_path, "agent_a", "agent_b"])
        assert result.exit_code == 0
        assert "agent_a" in result.output
        assert "agent_b" in result.output
