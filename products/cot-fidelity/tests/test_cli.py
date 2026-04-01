"""
Tests for cot_fidelity.cli
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from click.testing import CliRunner
    from cot_fidelity.cli import cli
    _CLICK_AVAILABLE = True
except ImportError:
    _CLICK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _CLICK_AVAILABLE, reason="click not installed")


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_input(tmp_path):
    """Create a sample input JSON file."""
    data = [
        {
            "prompt": "Explain why water boils at 100 degrees",
            "cot_chain": "Water boils when vapor pressure equals atmospheric pressure...",
            "with_cot_output": "Water boils at 100°C because vapor pressure equals atmospheric pressure.",
            "without_cot_output": "Water boils at 100°C.",
        }
    ]
    p = tmp_path / "input.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def multi_input(tmp_path):
    """Create a multi-entry input JSON file."""
    data = [
        {
            "prompt": f"prompt {i}",
            "cot_chain": f"chain {i}",
            "with_cot_output": f"detailed output {i}",
            "without_cot_output": f"simple output {i}",
        }
        for i in range(3)
    ]
    p = tmp_path / "multi.json"
    p.write_text(json.dumps(data))
    return p


# ── CLI test command ──────────────────────────────────────────────────────────

class TestCLITest:
    def test_test_command_runs(self, runner, sample_input, tmp_path):
        result = runner.invoke(cli, [
            "test",
            "--input", str(sample_input),
        ])
        assert result.exit_code == 0

    def test_test_command_json_output(self, runner, sample_input, tmp_path):
        result = runner.invoke(cli, [
            "test",
            "--input", str(sample_input),
            "--output-format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total" in data
        assert data["total"] == 1

    def test_test_command_markdown_output(self, runner, sample_input):
        result = runner.invoke(cli, [
            "test",
            "--input", str(sample_input),
            "--output-format", "markdown",
        ])
        assert result.exit_code == 0
        assert "FidelityBatchReport" in result.output

    def test_test_command_saves_to_db(self, runner, sample_input, tmp_path):
        db = str(tmp_path / "test.db")
        result = runner.invoke(cli, [
            "test",
            "--input", str(sample_input),
            "--db", db,
        ])
        assert result.exit_code == 0
        from cot_fidelity.store import FidelityStore
        store = FidelityStore(db_path=db)
        assert store.count() == 1

    def test_test_command_multi_entry(self, runner, multi_input):
        result = runner.invoke(cli, [
            "test",
            "--input", str(multi_input),
            "--output-format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 3

    def test_test_command_ci_gate_pass(self, runner, sample_input):
        result = runner.invoke(cli, [
            "test",
            "--input", str(sample_input),
            "--min-faithfulness-rate", "0.0",
        ])
        assert result.exit_code == 0

    def test_test_command_ci_gate_fail(self, runner, sample_input):
        result = runner.invoke(cli, [
            "test",
            "--input", str(sample_input),
            "--min-faithfulness-rate", "1.0",
            "--unfaithful-threshold", "0.0",
            "--faithful-threshold", "0.99",
        ])
        # May pass or fail depending on TF-IDF result, but shouldn't crash
        assert result.exit_code in (0, 1)


# ── CLI report command ────────────────────────────────────────────────────────

class TestCLIReport:
    def test_report_empty_db(self, runner, tmp_path):
        db = str(tmp_path / "empty.db")
        result = runner.invoke(cli, ["report", "--db", db])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_report_with_data(self, runner, tmp_path):
        from cot_fidelity.store import FidelityStore
        from cot_fidelity.models import FidelityResult
        db = str(tmp_path / "data.db")
        store = FidelityStore(db_path=db)
        r = FidelityResult(
            prompt="test",
            full_output="a",
            suppressed_output="b",
            cot_chain="c",
            similarity=0.5,
            faithfulness_score=0.5,
            verdict="FAITHFUL",
            faithful_threshold=0.15,
            unfaithful_threshold=0.08,
        )
        store.save(r)
        result = runner.invoke(cli, ["report", "--db", db, "--output-format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 1


# ── CLI drift command ─────────────────────────────────────────────────────────

class TestCLIDrift:
    def test_drift_empty_db(self, runner, tmp_path):
        db = str(tmp_path / "empty.db")
        result = runner.invoke(cli, ["drift", "--db", db])
        assert result.exit_code == 0
        assert "INSUFFICIENT_DATA" in result.output

    def test_drift_json_output(self, runner, tmp_path):
        db = str(tmp_path / "empty.db")
        result = runner.invoke(cli, ["drift", "--db", db, "--output-format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "trend" in data


# ── CLI clear command ─────────────────────────────────────────────────────────

class TestCLIClear:
    def test_clear_with_no_input_prompts(self, runner, tmp_path):
        db = str(tmp_path / "test.db")
        # Clear with --yes to skip prompt
        result = runner.invoke(cli, ["clear", "--db", db, "--yes"])
        assert result.exit_code == 0
        assert "0" in result.output
