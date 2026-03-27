"""llm-contract — Define, version, and enforce behavioral contracts on LLM function calls."""

from llm_contract.models import (
    SemanticRule,
    ContractViolationError,
    ContractResult,
    RuleResult,
    ViolationStrategy,
)
from llm_contract.contract import contract
from llm_contract.config import configure, get_config

__version__ = "0.1.0"

__all__ = [
    "contract",
    "SemanticRule",
    "ContractViolationError",
    "ContractResult",
    "RuleResult",
    "ViolationStrategy",
    "configure",
    "get_config",
]
