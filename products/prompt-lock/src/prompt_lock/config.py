"""Configuration models for prompt-lock."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class JudgeCalibrationConfig(BaseModel):
    """Config for validating that an LLM judge agrees with human labels."""

    enabled: bool = True
    human_labels_file: str  # path to JSONL: {"input", "output", "human_score"}
    model: str = "gpt-4o-mini"
    criteria: str = "Rate the quality of this response on a scale of 0.0 to 1.0."
    min_agreement: float = 0.80  # fraction of examples judge must agree with humans
    min_spearman: float = 0.70   # minimum Spearman correlation with human scores
    agreement_tolerance: float = 0.15  # ±0.15 counts as agreement


class EvalConfig(BaseModel):
    """A single evaluation to run against prompt outputs."""

    type: Literal["llm_judge", "exact_match", "semantic_similarity", "regex", "custom"]
    criteria: Optional[str] = None      # for llm_judge
    threshold: float = 0.70
    model: Optional[str] = None         # for llm_judge (overrides top-level model)
    pattern: Optional[str] = None       # for regex
    custom_fn: Optional[str] = None     # for custom: "module.function_name"


class PromptConfig(BaseModel):
    """A set of prompt files to watch and evaluate."""

    path: str  # glob pattern relative to repo root
    name: Optional[str] = None
    evals: list[EvalConfig] = Field(default_factory=list)
    test_cases_file: Optional[str] = None  # JSONL: {"input", "output", "expected_output"}


class GateConfig(BaseModel):
    """CI gate behavior."""

    mode: Literal["hard", "regression", "soft"] = "regression"
    regression_threshold: float = 0.05  # fail if score drops more than this from baseline
    hard_threshold: float = 0.70        # fail if score below this (hard mode only)


class TracerConfig(BaseModel):
    """SQLite trace ledger settings."""

    db_path: str = ".prompt-lock/traces.db"
    enabled: bool = True


class PromptLockConfig(BaseModel):
    """Root configuration for prompt-lock."""

    version: str = "1"
    model: str = "gpt-4o-mini"
    judge: Optional[JudgeCalibrationConfig] = None
    prompts: list[PromptConfig] = Field(default_factory=list)
    gate: GateConfig = Field(default_factory=GateConfig)
    tracer: TracerConfig = Field(default_factory=TracerConfig)

    @classmethod
    def from_file(cls, path: Path | str = ".prompt-lock.yml") -> "PromptLockConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                "Run `prompt-lock init` to create one."
            )
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})
