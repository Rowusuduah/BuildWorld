"""
semantic_pass_k.config
----------------------
YAML configuration loader for semantic-pass-k.

Config file format (sempass.yaml):
    label: my_agent_suite
    k: 5
    criticality: HIGH
    borderline_band: 0.05
    agent_label: gpt-4o
    store: consistency_history.db
    prompts:
      - "Summarise the report."
      - "What are the key findings?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SemPassConfig:
    """Loaded configuration for a semantic-pass-k evaluation run."""
    label: str = "sempass_run"
    k: int = 5
    criticality: str = "HIGH"
    borderline_band: float = 0.05
    agent_label: str = "default"
    store: str = "consistency_history.db"
    prompts: List[str] = field(default_factory=list)


def load_config(path: str) -> SemPassConfig:
    """
    Load a SemPassConfig from a YAML file.

    Falls back to stdlib-only if PyYAML is not installed
    (parses simple key: value lines only).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        data = _parse_simple_yaml(config_path.read_text())

    return _dict_to_config(data)


def _dict_to_config(data: dict) -> SemPassConfig:
    cfg = SemPassConfig()
    cfg.label = str(data.get("label", cfg.label))
    cfg.k = int(data.get("k", cfg.k))
    cfg.criticality = str(data.get("criticality", cfg.criticality)).upper()
    cfg.borderline_band = float(data.get("borderline_band", cfg.borderline_band))
    cfg.agent_label = str(data.get("agent_label", cfg.agent_label))
    cfg.store = str(data.get("store", cfg.store))
    raw_prompts = data.get("prompts", [])
    cfg.prompts = [str(p) for p in raw_prompts] if isinstance(raw_prompts, list) else []
    return cfg


def _parse_simple_yaml(text: str) -> dict:
    """
    Minimal YAML parser for simple key: value lines.
    Does NOT handle nested structures — for prompts list only.
    """
    result: dict = {}
    current_list_key: Optional[str] = None
    current_list: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list_key:
            current_list.append(stripped[2:].strip())
            continue
        if ":" in stripped:
            if current_list_key:
                result[current_list_key] = current_list
                current_list_key = None
                current_list = []
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()
            if not value:
                current_list_key = key
                current_list = []
            else:
                # Attempt type coercion
                if value.isdigit():
                    result[key] = int(value)
                else:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value

    if current_list_key:
        result[current_list_key] = current_list

    return result
