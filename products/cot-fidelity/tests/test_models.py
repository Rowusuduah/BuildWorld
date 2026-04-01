"""
Tests for cot_fidelity.models
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cot_fidelity.models import (
    DriftPoint,
    DriftReport,
    FidelityBatchReport,
    FidelityResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_result(
    verdict="FAITHFUL",
    faithfulness_score=0.30,
    similarity=0.70,
    prompt="Why does gravity work?",
    cot_chain="Because mass bends spacetime...",
    with_cot_output="Gravity is spacetime curvature.",
    without_cot_output="Things fall.",
) -> FidelityResult:
    return FidelityResult(
        prompt=prompt,
        full_output=with_cot_output,
        suppressed_output=without_cot_output,
        cot_chain=cot_chain,
        similarity=similarity,
        faithfulness_score=faithfulness_score,
        verdict=verdict,
        faithful_threshold=0.15,
        unfaithful_threshold=0.08,
    )


# ── FidelityResult ────────────────────────────────────────────────────────────

class TestFidelityResult:
    def test_creates_with_required_fields(self):
        r = make_result()
        assert r.verdict == "FAITHFUL"
        assert r.faithfulness_score == 0.30
        assert r.similarity == 0.70

    def test_prompt_hash_generated_auto(self):
        r = make_result()
        assert len(r.prompt_hash) == 16
        assert all(c in "0123456789abcdef" for c in r.prompt_hash)

    def test_same_prompt_same_hash(self):
        r1 = make_result(prompt="hello world")
        r2 = make_result(prompt="hello world")
        assert r1.prompt_hash == r2.prompt_hash

    def test_different_prompt_different_hash(self):
        r1 = make_result(prompt="hello world")
        r2 = make_result(prompt="goodbye world")
        assert r1.prompt_hash != r2.prompt_hash

    def test_explicit_prompt_hash(self):
        r = FidelityResult(
            prompt="x",
            full_output="a",
            suppressed_output="b",
            cot_chain="c",
            similarity=0.5,
            faithfulness_score=0.5,
            verdict="FAITHFUL",
            faithful_threshold=0.15,
            unfaithful_threshold=0.08,
            prompt_hash="customhash1234567",
        )
        assert r.prompt_hash == "customhash1234567"

    def test_is_faithful_property(self):
        assert make_result(verdict="FAITHFUL").is_faithful
        assert not make_result(verdict="UNFAITHFUL").is_faithful
        assert not make_result(verdict="INCONCLUSIVE").is_faithful

    def test_is_unfaithful_property(self):
        assert make_result(verdict="UNFAITHFUL").is_unfaithful
        assert not make_result(verdict="FAITHFUL").is_unfaithful

    def test_is_inconclusive_property(self):
        assert make_result(verdict="INCONCLUSIVE").is_inconclusive
        assert not make_result(verdict="FAITHFUL").is_inconclusive

    def test_to_dict_contains_required_keys(self):
        d = make_result().to_dict()
        for key in ["verdict", "faithfulness_score", "similarity", "prompt_hash", "runs"]:
            assert key in d

    def test_to_json_is_valid_json(self):
        j = make_result().to_json()
        data = json.loads(j)
        assert data["verdict"] == "FAITHFUL"

    def test_to_markdown_contains_verdict(self):
        md = make_result(verdict="FAITHFUL").to_markdown()
        assert "FAITHFUL" in md

    def test_to_markdown_unfaithful_contains_x_icon(self):
        md = make_result(verdict="UNFAITHFUL").to_markdown()
        assert "UNFAITHFUL" in md

    def test_to_markdown_truncates_long_cot(self):
        long_cot = "x" * 500
        md = make_result(cot_chain=long_cot).to_markdown()
        assert "..." in md

    def test_tested_at_defaults_to_now(self):
        before = datetime.now(timezone.utc)
        r = make_result()
        after = datetime.now(timezone.utc)
        assert before <= r.tested_at <= after

    def test_rounds_faithfulness_score_in_dict(self):
        r = make_result(faithfulness_score=0.1234567)
        d = r.to_dict()
        assert d["faithfulness_score"] == 0.123457


# ── FidelityBatchReport ────────────────────────────────────────────────────────

class TestFidelityBatchReport:
    def _batch(self, verdicts=("FAITHFUL", "UNFAITHFUL", "INCONCLUSIVE")):
        results = [make_result(verdict=v) for v in verdicts]
        return FidelityBatchReport(results=results)

    def test_total(self):
        assert self._batch().total == 3

    def test_faithful_count(self):
        assert self._batch().faithful_count == 1

    def test_unfaithful_count(self):
        assert self._batch().unfaithful_count == 1

    def test_inconclusive_count(self):
        assert self._batch().inconclusive_count == 1

    def test_faithfulness_rate(self):
        b = self._batch(["FAITHFUL", "FAITHFUL", "UNFAITHFUL", "UNFAITHFUL"])
        assert b.faithfulness_rate == 0.5

    def test_unfaithfulness_rate(self):
        b = self._batch(["UNFAITHFUL", "UNFAITHFUL"])
        assert b.unfaithfulness_rate == 1.0

    def test_empty_batch_rates(self):
        b = FidelityBatchReport(results=[])
        assert b.faithfulness_rate == 0.0
        assert b.unfaithfulness_rate == 0.0
        assert b.mean_faithfulness_score == 0.0
        assert b.mean_similarity == 0.0

    def test_mean_faithfulness_score(self):
        r1 = make_result(faithfulness_score=0.3)
        r2 = make_result(faithfulness_score=0.1)
        b = FidelityBatchReport(results=[r1, r2])
        assert abs(b.mean_faithfulness_score - 0.2) < 1e-9

    def test_to_dict_structure(self):
        d = self._batch().to_dict()
        assert d["total"] == 3
        assert "results" in d
        assert len(d["results"]) == 3

    def test_to_json(self):
        j = self._batch().to_json()
        data = json.loads(j)
        assert data["total"] == 3

    def test_to_markdown_contains_header(self):
        md = self._batch().to_markdown()
        assert "FidelityBatchReport" in md
        assert "FAITHFUL" in md
