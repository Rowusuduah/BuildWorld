"""Global configuration for llm-contract."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

_DEFAULT_DB_PATH = "./llm_contract_logs.db"
_DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_JUDGE_PROVIDER = "anthropic"
_DEFAULT_THRESHOLD = 0.90


@dataclass
class LLMContractConfig:
    """Global runtime configuration.

    Use :func:`configure` to set these values rather than constructing directly.
    """

    default_judge_model: str = _DEFAULT_JUDGE_MODEL
    default_judge_provider: str = _DEFAULT_JUDGE_PROVIDER
    db_path: str = _DEFAULT_DB_PATH
    log_all_results: bool = True
    default_threshold: float = _DEFAULT_THRESHOLD


_config: LLMContractConfig = LLMContractConfig()


def configure(
    *,
    default_judge_model: Optional[str] = None,
    default_judge_provider: Optional[str] = None,
    db_path: Optional[str] = None,
    log_all_results: Optional[bool] = None,
    default_threshold: Optional[float] = None,
) -> None:
    """Configure global llm-contract settings.

    All parameters are optional; only provided values are updated.

    Args:
        default_judge_model: Model used to evaluate SemanticRules.
            Defaults to ``claude-haiku-4-5-20251001``.
        default_judge_provider: Provider for the judge model.
            One of ``"anthropic"`` or ``"openai"``. Defaults to ``"anthropic"``.
        db_path: Path to the SQLite database for drift logging.
            Defaults to ``"./llm_contract_logs.db"``.
        log_all_results: If True, logs both passing and failing evaluations.
            If False, only logs failures. Defaults to True.
        default_threshold: Global minimum weighted pass score.
            Can be overridden per ``@contract``. Defaults to 0.90.

    Example::

        import llm_contract
        llm_contract.configure(
            default_judge_model="claude-haiku-4-5-20251001",
            db_path="/var/log/contracts.db",
        )
    """
    global _config
    if default_judge_model is not None:
        _config.default_judge_model = default_judge_model
    if default_judge_provider is not None:
        _config.default_judge_provider = default_judge_provider
    if db_path is not None:
        _config.db_path = db_path
    if log_all_results is not None:
        _config.log_all_results = log_all_results
    if default_threshold is not None:
        _config.default_threshold = default_threshold


def get_config() -> LLMContractConfig:
    """Return the current global configuration, with env-var overrides applied."""
    # Environment variable overrides (non-destructive)
    effective = LLMContractConfig(
        default_judge_model=os.environ.get(
            "LLM_CONTRACT_JUDGE_MODEL", _config.default_judge_model
        ),
        default_judge_provider=os.environ.get(
            "LLM_CONTRACT_JUDGE_PROVIDER", _config.default_judge_provider
        ),
        db_path=os.environ.get("LLM_CONTRACT_DB_PATH", _config.db_path),
        log_all_results=_config.log_all_results,
        default_threshold=_config.default_threshold,
    )
    return effective
