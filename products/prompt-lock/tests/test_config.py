import pytest
import yaml
from pathlib import Path
from prompt_lock.config import PromptLockConfig, EvalConfig, GateConfig


def test_minimal_config():
    cfg = PromptLockConfig.model_validate({"version": "1"})
    assert cfg.model == "gpt-4o-mini"
    assert cfg.prompts == []
    assert cfg.gate.mode == "regression"


def test_full_config():
    data = {
        "version": "1",
        "model": "claude-haiku-4-5-20251001",
        "prompts": [
            {
                "path": "prompts/*.txt",
                "name": "My Prompts",
                "test_cases_file": "tests/cases.jsonl",
                "evals": [
                    {"type": "llm_judge", "criteria": "Is it helpful?", "threshold": 0.7},
                    {"type": "exact_match", "threshold": 1.0},
                ],
            }
        ],
        "gate": {"mode": "hard", "hard_threshold": 0.6},
        "tracer": {"db_path": "/tmp/traces.db", "enabled": True},
    }
    cfg = PromptLockConfig.model_validate(data)
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert len(cfg.prompts) == 1
    assert len(cfg.prompts[0].evals) == 2
    assert cfg.gate.mode == "hard"
    assert cfg.gate.hard_threshold == 0.6


def test_from_file(tmp_path):
    data = {
        "version": "1",
        "prompts": [
            {
                "path": "prompts/*.txt",
                "evals": [{"type": "regex", "pattern": r"\d+"}],
            }
        ],
    }
    cfg_file = tmp_path / ".prompt-lock.yml"
    cfg_file.write_text(yaml.dump(data))
    cfg = PromptLockConfig.from_file(cfg_file)
    assert cfg.prompts[0].evals[0].type == "regex"


def test_from_file_not_found():
    with pytest.raises(FileNotFoundError, match="prompt-lock init"):
        PromptLockConfig.from_file("/nonexistent/path/.prompt-lock.yml")


def test_judge_config():
    data = {
        "version": "1",
        "judge": {
            "enabled": True,
            "human_labels_file": "tests/labels.jsonl",
            "model": "gpt-4o-mini",
            "criteria": "Is it good?",
            "min_agreement": 0.85,
            "min_spearman": 0.75,
        },
    }
    cfg = PromptLockConfig.model_validate(data)
    assert cfg.judge is not None
    assert cfg.judge.min_agreement == 0.85
    assert cfg.judge.min_spearman == 0.75
