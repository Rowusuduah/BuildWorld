"""Tests for context_trace.tracer — ContextTracer, AttributionReport, CostBudget."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from context_trace.embedder import IdentityEmbedder, MockEmbedder
from context_trace.tracer import (
    AttributionReport,
    BudgetExceededError,
    ChunkScore,
    ContextTracer,
    CostBudget,
)


# ---------------------------------------------------------------------------
# CostBudget
# ---------------------------------------------------------------------------

class TestCostBudget:
    def test_default_values(self):
        b = CostBudget()
        assert b.max_api_calls == 500
        assert b.max_cost_usd == 10.0
        assert b.cost_per_call_usd == 0.001

    def test_would_exceed_below(self):
        b = CostBudget(max_api_calls=10)
        assert not b.would_exceed(9)

    def test_would_exceed_at_limit(self):
        b = CostBudget(max_api_calls=10)
        # 10 calls with max 10 — strictly greater → not exceeded
        assert not b.would_exceed(10)

    def test_would_exceed_above(self):
        b = CostBudget(max_api_calls=10)
        assert b.would_exceed(11)

    def test_estimated_cost(self):
        b = CostBudget(cost_per_call_usd=0.002)
        assert b.estimated_cost(5) == pytest.approx(0.01)

    def test_estimated_cost_zero(self):
        b = CostBudget()
        assert b.estimated_cost(0) == 0.0


# ---------------------------------------------------------------------------
# ChunkScore clamping
# ---------------------------------------------------------------------------

class TestChunkScore:
    def test_attribution_clamped_above(self):
        cs = ChunkScore("x", 1.5, -0.5, 0.0, 3)
        assert cs.attribution_score == 1.0

    def test_attribution_clamped_below(self):
        cs = ChunkScore("x", -0.2, 1.2, 0.0, 3)
        assert cs.attribution_score == 0.0

    def test_attribution_in_range(self):
        cs = ChunkScore("x", 0.75, 0.25, 0.05, 3)
        assert cs.attribution_score == pytest.approx(0.75)

    def test_mean_similarity_clamped(self):
        cs = ChunkScore("x", 0.5, 1.5, 0.0, 3)
        assert cs.mean_similarity == 1.0

    def test_masked_outputs_default_empty(self):
        cs = ChunkScore("x", 0.5, 0.5, 0.0, 3)
        assert cs.masked_outputs == []


# ---------------------------------------------------------------------------
# ContextTracer — basic tracing
# ---------------------------------------------------------------------------

class TestContextTracerBasic:
    def test_empty_chunks_returns_empty_report(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=3)
        report = tracer.trace(prompt="hello", original_output="world", chunks={})
        assert report.chunk_scores == {}
        assert report.total_api_calls == 0
        assert report.skipped_chunks == []

    def test_api_call_count(self, diverging_runner, mock_embedder):
        prompt = "AAA BBB CCC"
        chunks = {"a": "AAA", "b": "BBB"}
        tracer = ContextTracer(runner=diverging_runner, embedder=mock_embedder, k=2)
        report = tracer.trace(prompt=prompt, original_output="The answer is 42.", chunks=chunks)
        # 2 chunks × k=2 = 4 calls
        assert report.total_api_calls == 4

    def test_k_runs_stored_per_chunk(self, constant_runner, mock_embedder):
        prompt = "TOKEN1 TOKEN2"
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=3)
        report = tracer.trace(
            prompt=prompt,
            original_output="hello",
            chunks={"a": "TOKEN1"},
        )
        assert report.chunk_scores["a"].runs == 3
        assert len(report.chunk_scores["a"].masked_outputs) == 3

    def test_chunk_not_in_prompt_skipped(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = tracer.trace(
                prompt="hello world",
                original_output="hi",
                chunks={"missing": "NOT_IN_PROMPT"},
            )
        assert "missing" in report.skipped_chunks
        assert "missing" not in report.chunk_scores
        assert len(w) == 1
        assert "not found verbatim" in str(w[0].message).lower()

    def test_multiple_chunks_mixed_found_and_missing(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            report = tracer.trace(
                prompt="FOUND_TOKEN extra text",
                original_output="result",
                chunks={"present": "FOUND_TOKEN", "absent": "MISSING_XYZ"},
            )
        assert "present" in report.chunk_scores
        assert "absent" in report.skipped_chunks

    def test_budget_exceeded_raises(self, constant_runner, mock_embedder):
        budget = CostBudget(max_api_calls=2)
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=3, budget=budget)
        # 2 chunks × k=3 = 6 calls > budget of 2
        with pytest.raises(BudgetExceededError, match="6 API calls"):
            tracer.trace(
                prompt="A B",
                original_output="x",
                chunks={"a": "A", "b": "B"},
            )

    def test_budget_not_exceeded_at_limit(self, constant_runner, mock_embedder):
        budget = CostBudget(max_api_calls=6)
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=3, budget=budget)
        # 2 chunks × k=3 = 6 calls == budget of 6 → allowed
        report = tracer.trace(
            prompt="A B",
            original_output="x",
            chunks={"a": "A", "b": "B"},
        )
        assert report.total_api_calls == 6

    def test_estimated_cost_calculated(self, constant_runner, mock_embedder):
        budget = CostBudget(cost_per_call_usd=0.01)
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=2, budget=budget)
        report = tracer.trace(
            prompt="TOKEN rest",
            original_output="x",
            chunks={"t": "TOKEN"},
        )
        # 1 chunk × k=2 = 2 calls × $0.01 = $0.02
        assert report.estimated_cost_usd == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# ContextTracer — attribution score semantics
# ---------------------------------------------------------------------------

class TestContextTracerAttribution:
    def test_identity_embedder_gives_zero_attribution(self, constant_runner):
        """
        With IdentityEmbedder (same vector for all text),
        cosine_similarity = 1.0, attribution = 1 - 1 = 0.0 for every chunk.
        """
        embedder = IdentityEmbedder()
        tracer = ContextTracer(runner=constant_runner, embedder=embedder, k=2)
        report = tracer.trace(
            prompt="CHUNK1 other text",
            original_output="answer",
            chunks={"c": "CHUNK1"},
        )
        assert report.chunk_scores["c"].attribution_score == pytest.approx(0.0, abs=1e-6)
        assert report.chunk_scores["c"].mean_similarity == pytest.approx(1.0, abs=1e-6)

    def test_same_output_runner_attribution_near_zero(self, mock_embedder):
        """
        When masked runner returns original_output verbatim,
        embed(masked) == embed(original) → attribution ≈ 0.
        """
        original = "The capital of France is Paris."

        def same_runner(prompt: str) -> str:
            return original  # always same as original_output

        tracer = ContextTracer(runner=same_runner, embedder=mock_embedder, k=3)
        report = tracer.trace(
            prompt="CHUNK_TOKEN some context",
            original_output=original,
            chunks={"chunk": "CHUNK_TOKEN"},
        )
        # Same string → same hash-seed vector → cosine_sim = 1.0 exactly
        assert report.chunk_scores["chunk"].attribution_score == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_same_vector(self):
        vec = np.array([1.0, 0.0, 0.0])
        sim = ContextTracer._cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = ContextTracer._cosine_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = ContextTracer._cosine_similarity(a, b)
        assert sim == 0.0

    def test_mask_token_appears_in_masked_prompt(self, constant_runner, mock_embedder):
        """Verify the masked prompt contains the mask token."""
        received_prompts = []

        def recording_runner(prompt: str) -> str:
            received_prompts.append(prompt)
            return "response"

        tracer = ContextTracer(
            runner=recording_runner, embedder=mock_embedder, k=1, mask_token="[REMOVED]"
        )
        tracer.trace(
            prompt="HELLO world",
            original_output="x",
            chunks={"chunk": "HELLO"},
        )
        assert any("[REMOVED]" in p for p in received_prompts)
        assert any("HELLO" not in p for p in received_prompts)

    def test_mask_prompt_not_found_returns_none(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder)
        result = tracer._mask_prompt("hello world", "NOTFOUND", "x")
        assert result is None

    def test_mask_prompt_replaces_first_occurrence(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder)
        result = tracer._mask_prompt("AAA BBB AAA", "AAA", "chunk")
        assert result is not None
        # First occurrence replaced, second preserved
        assert result.count("AAA") == 1
        assert "[REMOVED]:chunk" in result

    def test_empty_chunk_content_skipped(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            report = tracer.trace(
                prompt="hello world",
                original_output="hi",
                chunks={"empty_chunk": ""},
            )
        assert "empty_chunk" in report.skipped_chunks


# ---------------------------------------------------------------------------
# AttributionReport
# ---------------------------------------------------------------------------

class TestAttributionReport:
    def test_top_contributors_sorted(self, sample_report):
        top = sample_report.top_contributors()
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_top_contributors_limit(self, sample_report):
        top = sample_report.top_contributors(n=2)
        assert len(top) == 2

    def test_top_contributors_default_n5(self, sample_report):
        top = sample_report.top_contributors()
        # Only 3 chunks in sample_report, so returns all 3
        assert len(top) == 3

    def test_top_contributors_names_correct(self, sample_report):
        top = sample_report.top_contributors()
        # doc1 has highest score (0.85), should be first
        assert top[0][0] == "doc1"

    def test_contributors_above_threshold(self, sample_report):
        above = sample_report.contributors_above(0.5)
        assert len(above) == 1
        assert above[0][0] == "doc1"

    def test_contributors_above_zero(self, sample_report):
        above = sample_report.contributors_above(0.0)
        assert len(above) == 3

    def test_contributors_above_one(self, sample_report):
        above = sample_report.contributors_above(1.0)
        assert len(above) == 0

    def test_top_score_correct(self, sample_report):
        assert sample_report.top_score == pytest.approx(0.85)

    def test_top_score_empty_report(self):
        report = AttributionReport(
            chunk_scores={},
            original_output="x",
            prompt="x",
            k=3,
            total_api_calls=0,
            estimated_cost_usd=0.0,
            elapsed_seconds=0.0,
        )
        assert report.top_score == 0.0

    def test_attribution_heatmap_format(self, sample_report):
        heatmap = sample_report.attribution_heatmap
        assert "doc1" in heatmap
        assert "doc2" in heatmap
        assert "system" in heatmap
        # Highest scorer should appear first
        lines = heatmap.split("\n")
        assert "doc1" in lines[0]

    def test_attribution_heatmap_empty(self):
        report = AttributionReport(
            chunk_scores={},
            original_output="x",
            prompt="x",
            k=3,
            total_api_calls=0,
            estimated_cost_usd=0.0,
            elapsed_seconds=0.0,
        )
        assert report.attribution_heatmap == "(no chunks scored)"

    def test_to_dict_keys(self, sample_report):
        d = sample_report.to_dict()
        assert "chunk_scores" in d
        assert "top_contributors" in d
        assert "total_api_calls" in d
        assert "estimated_cost_usd" in d
        assert "elapsed_seconds" in d
        assert "skipped_chunks" in d

    def test_to_dict_chunk_score_fields(self, sample_report):
        d = sample_report.to_dict()
        doc1 = d["chunk_scores"]["doc1"]
        assert "attribution_score" in doc1
        assert "mean_similarity" in doc1
        assert "std_similarity" in doc1
        assert "runs" in doc1

    def test_to_dict_top_contributors_sorted(self, sample_report):
        d = sample_report.to_dict()
        scores = [s for _, s in d["top_contributors"]]
        assert scores == sorted(scores, reverse=True)

    def test_elapsed_seconds_positive(self, constant_runner, mock_embedder):
        tracer = ContextTracer(runner=constant_runner, embedder=mock_embedder, k=1)
        report = tracer.trace(
            prompt="TOKEN extra",
            original_output="result",
            chunks={"t": "TOKEN"},
        )
        assert report.elapsed_seconds >= 0.0
