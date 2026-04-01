"""
context-trace
~~~~~~~~~~~~~
Per-context-chunk causal AttributionScore for LLM outputs.

Answers: *which part of my context caused this output?*

Quick start::

    from context_trace import ContextTracer, AttributionGate, CostBudget

    def my_llm(prompt: str) -> str:
        ...  # your LLM call

    tracer = ContextTracer(runner=my_llm, k=3)
    report = tracer.trace(
        prompt=full_prompt,
        original_output=llm_response,
        chunks={
            "system_prompt": system_text,
            "retrieved_doc": doc_text,
            "user_message": user_text,
        },
    )
    print(report.attribution_heatmap)
    # system_prompt  [███░░░░░░░] 0.31
    # retrieved_doc  [██████████] 0.87
    # user_message   [██░░░░░░░░] 0.18

    gate = AttributionGate(max_single_chunk_score=0.90, min_chunks_contributing=2)
    gate.check(report)  # raises AttributionGateFailure if violated
"""

from context_trace.tracer import (
    AttributionReport,
    BudgetExceededError,
    ChunkScore,
    ContextTracer,
    CostBudget,
)
from context_trace.gate import AttributionGate, AttributionGateFailure
from context_trace.embedder import (
    IdentityEmbedder,
    MockEmbedder,
    SentenceTransformerEmbedder,
)
from context_trace.store import AttributionStore
from context_trace.runners import anthropic_runner, openai_runner

__version__ = "0.1.0"

__all__ = [
    "ContextTracer",
    "AttributionReport",
    "ChunkScore",
    "CostBudget",
    "BudgetExceededError",
    "AttributionGate",
    "AttributionGateFailure",
    "MockEmbedder",
    "IdentityEmbedder",
    "SentenceTransformerEmbedder",
    "AttributionStore",
    "anthropic_runner",
    "openai_runner",
]
