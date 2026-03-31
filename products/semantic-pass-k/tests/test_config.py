"""Tests for semantic_pass_k.config"""
from __future__ import annotations
import os
import tempfile
import pytest

from semantic_pass_k.config import SemPassConfig, load_config, _parse_simple_yaml, _dict_to_config


class TestSemPassConfigDefaults:
    def test_default_label(self):
        cfg = SemPassConfig()
        assert cfg.label == "sempass_run"

    def test_default_k(self):
        cfg = SemPassConfig()
        assert cfg.k == 5

    def test_default_criticality(self):
        cfg = SemPassConfig()
        assert cfg.criticality == "HIGH"

    def test_default_borderline_band(self):
        cfg = SemPassConfig()
        assert cfg.borderline_band == 0.05

    def test_default_agent_label(self):
        cfg = SemPassConfig()
        assert cfg.agent_label == "default"

    def test_default_prompts_empty(self):
        cfg = SemPassConfig()
        assert cfg.prompts == []


class TestParseSimpleYaml:
    def test_parses_string_value(self):
        result = _parse_simple_yaml("label: my_run")
        assert result["label"] == "my_run"

    def test_parses_int_value(self):
        result = _parse_simple_yaml("k: 7")
        assert result["k"] == 7

    def test_parses_float_value(self):
        result = _parse_simple_yaml("borderline_band: 0.10")
        assert abs(result["borderline_band"] - 0.10) < 1e-9

    def test_parses_list(self):
        yaml_text = "prompts:\n  - prompt one\n  - prompt two\n"
        result = _parse_simple_yaml(yaml_text)
        assert result["prompts"] == ["prompt one", "prompt two"]

    def test_ignores_comment_lines(self):
        yaml_text = "# this is a comment\nk: 3\n"
        result = _parse_simple_yaml(yaml_text)
        assert result["k"] == 3

    def test_ignores_blank_lines(self):
        result = _parse_simple_yaml("\nk: 3\n\nlabel: test\n")
        assert result["k"] == 3
        assert result["label"] == "test"


class TestDictToConfig:
    def test_overrides_label(self):
        cfg = _dict_to_config({"label": "my_suite"})
        assert cfg.label == "my_suite"

    def test_overrides_k(self):
        cfg = _dict_to_config({"k": 10})
        assert cfg.k == 10

    def test_criticality_uppercased(self):
        cfg = _dict_to_config({"criticality": "medium"})
        assert cfg.criticality == "MEDIUM"

    def test_prompts_loaded(self):
        cfg = _dict_to_config({"prompts": ["a", "b", "c"]})
        assert cfg.prompts == ["a", "b", "c"]

    def test_empty_dict_uses_defaults(self):
        cfg = _dict_to_config({})
        assert cfg.k == 5
        assert cfg.criticality == "HIGH"


class TestLoadConfig:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/sempass.yaml")

    def test_loads_simple_yaml(self, tmp_path):
        config_path = str(tmp_path / "sempass.yaml")
        with open(config_path, "w") as f:
            f.write("label: test_run\nk: 3\ncriticality: MEDIUM\n")
        cfg = load_config(config_path)
        assert cfg.label == "test_run"
        assert cfg.k == 3
        assert cfg.criticality == "MEDIUM"

    def test_loads_with_prompts(self, tmp_path):
        config_path = str(tmp_path / "sempass.yaml")
        with open(config_path, "w") as f:
            f.write("prompts:\n  - hello world\n  - foo bar\n")
        cfg = load_config(config_path)
        assert "hello world" in cfg.prompts
        assert "foo bar" in cfg.prompts
