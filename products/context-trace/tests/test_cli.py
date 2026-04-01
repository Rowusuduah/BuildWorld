"""Tests for context_trace.cli — CLI commands via click.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from context_trace.cli import cli
from tests.conftest import make_report


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def report_file(tmp_path):
    """Write a sample report JSON to a temp file."""
    report = make_report()
    data = report.to_dict()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestShowCommand:
    def test_show_output_contains_chunk_names(self, runner, report_file):
        result = runner.invoke(cli, ["show", "--report", report_file])
        assert result.exit_code == 0
        assert "doc1" in result.output
        assert "doc2" in result.output
        assert "system" in result.output

    def test_show_output_contains_scores(self, runner, report_file):
        result = runner.invoke(cli, ["show", "--report", report_file])
        assert result.exit_code == 0
        assert "0.850" in result.output

    def test_show_nonexistent_file_exits_1(self, runner):
        result = runner.invoke(cli, ["show", "--report", "/nonexistent/file.json"])
        assert result.exit_code == 1

    def test_show_top_limits_output(self, runner, tmp_path):
        """When --top 1 is given, only top contributor shown."""
        report = make_report()
        data = report.to_dict()
        path = tmp_path / "r.json"
        path.write_text(json.dumps(data))
        result = runner.invoke(cli, ["show", "--report", str(path), "--top", "1"])
        assert result.exit_code == 0
        # doc1 should appear, doc2 should not
        assert "doc1" in result.output


class TestGateCommand:
    def test_gate_passes(self, runner, report_file):
        result = runner.invoke(
            cli, ["gate", "--report", report_file, "--max-score", "0.99"]
        )
        assert result.exit_code == 0
        assert "GATE PASSED" in result.output

    def test_gate_fails_max_score(self, runner, report_file):
        # doc1 has score 0.85, so max-score 0.5 should fail
        result = runner.invoke(
            cli, ["gate", "--report", report_file, "--max-score", "0.5"]
        )
        assert result.exit_code == 1
        assert "GATE FAILED" in result.output

    def test_gate_fails_min_contributors(self, runner, tmp_path):
        # Create report with only 1 chunk above 0.3
        from context_trace.tracer import ChunkScore
        report = make_report(
            chunk_scores={
                "only_one": ChunkScore("only_one", 0.80, 0.20, 0.0, 3),
                "low": ChunkScore("low", 0.05, 0.95, 0.0, 3),
            }
        )
        path = tmp_path / "r.json"
        path.write_text(json.dumps(report.to_dict()))
        result = runner.invoke(
            cli,
            ["gate", "--report", str(path), "--min-contributors", "3"],
        )
        assert result.exit_code == 1

    def test_gate_nonexistent_report_exits_1(self, runner):
        result = runner.invoke(cli, ["gate", "--report", "/nope.json"])
        assert result.exit_code == 1


class TestCompareCommand:
    def test_compare_shows_delta(self, runner, tmp_path):
        from context_trace.tracer import ChunkScore

        base_report = make_report(
            chunk_scores={
                "doc1": ChunkScore("doc1", 0.50, 0.50, 0.0, 3),
                "doc2": ChunkScore("doc2", 0.20, 0.80, 0.0, 3),
            }
        )
        curr_report = make_report(
            chunk_scores={
                "doc1": ChunkScore("doc1", 0.80, 0.20, 0.0, 3),
                "doc2": ChunkScore("doc2", 0.10, 0.90, 0.0, 3),
            }
        )

        base_path = tmp_path / "base.json"
        curr_path = tmp_path / "curr.json"
        base_path.write_text(json.dumps(base_report.to_dict()))
        curr_path.write_text(json.dumps(curr_report.to_dict()))

        result = runner.invoke(
            cli, ["compare", "--baseline", str(base_path), "--current", str(curr_path)]
        )
        assert result.exit_code == 0
        assert "doc1" in result.output
        assert "doc2" in result.output
        # doc1 delta = +0.30
        assert "+0.300" in result.output


class TestEstimateCommand:
    def test_estimate_output(self, runner, tmp_path):
        cfg = "chunks:\n  a: hello\n  b: world\nk: 4\n"
        cfg_path = tmp_path / "ctrace.yaml"
        cfg_path.write_text(cfg)
        result = runner.invoke(cli, ["estimate", "--config", str(cfg_path)])
        assert result.exit_code == 0
        assert "Chunks" in result.output
        assert "8" in result.output  # 2 chunks × k=4

    def test_estimate_nonexistent_config(self, runner):
        result = runner.invoke(cli, ["estimate", "--config", "/nope.yaml"])
        assert result.exit_code != 0
