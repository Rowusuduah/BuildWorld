"""Tests for the context-trim CLI."""

import json
import sys
import pytest
from pathlib import Path
from context_trim.cli import main


def write_messages(path: Path, messages: list) -> None:
    path.write_text(json.dumps(messages), encoding="utf-8")


LONG = "x" * 500
SHORT = "hello"


# --- estimate command ---


def test_cli_estimate_fits(tmp_path, capsys):
    p = tmp_path / "msgs.json"
    write_messages(p, [{"role": "user", "content": SHORT}])
    main(["estimate", str(p), "--max-tokens", "4096"])
    out = capsys.readouterr().out
    assert "OK" in out


def test_cli_estimate_over_budget(tmp_path, capsys):
    p = tmp_path / "msgs.json"
    write_messages(p, [{"role": "user", "content": LONG * 10}])
    main(["estimate", str(p), "--max-tokens", "50", "--reserved", "10"])
    out = capsys.readouterr().out
    assert "OVER_BUDGET" in out


def test_cli_estimate_shows_token_count(tmp_path, capsys):
    p = tmp_path / "msgs.json"
    write_messages(p, [{"role": "user", "content": "hello world"}])
    main(["estimate", str(p)])
    out = capsys.readouterr().out
    assert "tokens" in out.lower()


# --- trim command ---


def test_cli_trim_outputs_json(tmp_path, capsys):
    p = tmp_path / "msgs.json"
    msgs = [{"role": "user", "content": LONG} for _ in range(10)]
    write_messages(p, msgs)
    main(["trim", str(p), "--max-tokens", "300", "--reserved", "30"])
    out = capsys.readouterr().out
    # Output should contain trimmed JSON plus summary
    assert "context-trim" in out


def test_cli_trim_writes_output_file(tmp_path):
    p = tmp_path / "msgs.json"
    out_p = tmp_path / "out.json"
    msgs = [{"role": "user", "content": LONG} for _ in range(10)]
    write_messages(p, msgs)
    main(["trim", str(p), "--max-tokens", "300", "--reserved", "30", "--output", str(out_p)])
    assert out_p.exists()
    data = json.loads(out_p.read_text())
    assert isinstance(data, list)


def test_cli_trim_strategy_recency(tmp_path, capsys):
    p = tmp_path / "msgs.json"
    msgs = [{"role": "user", "content": LONG} for _ in range(8)]
    write_messages(p, msgs)
    main(["trim", str(p), "--max-tokens", "300", "--reserved", "30", "--strategy", "recency_first"])
    out = capsys.readouterr().out
    assert "recency_first" in out


def test_cli_trim_invalid_strategy(tmp_path):
    p = tmp_path / "msgs.json"
    write_messages(p, [{"role": "user", "content": "hi"}])
    with pytest.raises(SystemExit):
        main(["trim", str(p), "--strategy", "nonexistent_strategy"])


# --- ci command ---


def test_cli_ci_passes(tmp_path):
    p = tmp_path / "msgs.json"
    write_messages(p, [{"role": "user", "content": SHORT}])
    with pytest.raises(SystemExit) as exc:
        main(["ci", str(p), "--max-tokens", "4096"])
    assert exc.value.code == 0


def test_cli_ci_fails(tmp_path):
    p = tmp_path / "msgs.json"
    msgs = [{"role": "user", "content": LONG * 10}]
    write_messages(p, msgs)
    with pytest.raises(SystemExit) as exc:
        main(["ci", str(p), "--max-tokens", "50", "--reserved", "10"])
    assert exc.value.code == 1


# --- history command ---


def test_cli_history_empty(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    main(["history", "--db", db])
    out = capsys.readouterr().out
    assert "No history" in out


def test_cli_history_after_trim(tmp_path, capsys):
    db = str(tmp_path / "hist.db")
    from context_trim import ContextTrim, TrimStrategy
    ct = ContextTrim(max_tokens=200, reserved_tokens=20, db_path=db)
    msgs = [{"role": "user", "content": LONG} for _ in range(5)]
    ct.trim(msgs, pipeline_id="pipeline1")
    main(["history", "--db", db, "--pipeline", "pipeline1"])
    out = capsys.readouterr().out
    assert "pipeline1" in out


# --- error handling ---


def test_cli_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        main(["estimate", str(tmp_path / "nonexistent.json")])


def test_cli_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["estimate", str(p)])


def test_cli_not_a_list(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text('{"role": "user", "content": "hi"}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["estimate", str(p)])
