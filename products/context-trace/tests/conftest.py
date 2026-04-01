"""Shared fixtures for context-trace tests."""

from __future__ import annotations

import numpy as np
import pytest

from context_trace.embedder import IdentityEmbedder, MockEmbedder
from context_trace.tracer import AttributionReport, ChunkScore, ContextTracer, CostBudget


# ---------------------------------------------------------------------------
# Embedder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=16)


@pytest.fixture
def identity_embedder() -> IdentityEmbedder:
    """Returns the same vector for every input → attribution_score = 0 always."""
    return IdentityEmbedder(dim=4)


# ---------------------------------------------------------------------------
# Runner fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_runner():
    """Always returns the exact same output regardless of prompt."""
    def _run(prompt: str) -> str:
        return "The capital of France is Paris."
    return _run


@pytest.fixture
def echo_runner():
    """Returns the prompt back as the output."""
    def _run(prompt: str) -> str:
        return prompt
    return _run


@pytest.fixture
def diverging_runner():
    """
    Returns 'DIVERGED OUTPUT' when prompt contains '[REMOVED]',
    otherwise returns the original answer.
    Simulates a chunk that causally mattered.
    """
    def _run(prompt: str) -> str:
        if "[REMOVED]" in prompt:
            return "DIVERGED OUTPUT: I cannot answer without that information."
        return "The answer is 42."
    return _run


@pytest.fixture
def passthrough_runner():
    """Returns a fixed response identical to what we pass as original_output."""
    fixed = "The answer is 42."
    def _run(prompt: str) -> str:
        return fixed
    return _run


# ---------------------------------------------------------------------------
# Pre-built report fixture
# ---------------------------------------------------------------------------

def make_report(**overrides) -> AttributionReport:
    defaults = dict(
        chunk_scores={
            "doc1": ChunkScore("doc1", 0.85, 0.15, 0.05, 3),
            "doc2": ChunkScore("doc2", 0.20, 0.80, 0.03, 3),
            "system": ChunkScore("system", 0.10, 0.90, 0.02, 3),
        },
        original_output="The answer is 42.",
        prompt="doc1_text doc2_text system_text",
        k=3,
        total_api_calls=9,
        estimated_cost_usd=0.009,
        elapsed_seconds=1.2,
        skipped_chunks=[],
    )
    defaults.update(overrides)
    return AttributionReport(**defaults)


@pytest.fixture
def sample_report() -> AttributionReport:
    return make_report()
