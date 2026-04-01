"""
context-trim: Production-grade conversational context window trimmer.

Trim conversation histories and documents to fit within LLM token budgets.
Zero dependencies. No local model required. Works with any LLM provider.
"""

from .core import (
    TokenBudget,
    TrimStrategy,
    TrimResult,
    MessageTrimmer,
    DocumentTrimmer,
    ImportanceScorer,
    ContextTrim,
)
from .store import TrimStore

__version__ = "0.1.0"
__all__ = [
    "TokenBudget",
    "TrimStrategy",
    "TrimResult",
    "MessageTrimmer",
    "DocumentTrimmer",
    "ImportanceScorer",
    "ContextTrim",
    "TrimStore",
]
