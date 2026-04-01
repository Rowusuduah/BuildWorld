"""context_trace.tracer
~~~~~~~~~~~~~~~~~~~~
Core attribution engine: per-context-chunk causal AttributionScore for LLM outputs.

Algorithm (per chunk):
    1. Replace chunk content in prompt with "[REMOVED:<chunk_name>]"
    2. Run masked prompt k times through the runner
    3. Embed original_output and each masked output via sentence-transformers
    4. attribution_score = 1.0 - mean(cosine_similarity(original, masked_i))

Interpretation:
    - Score near 1.0: removing this chunk changes output dramatically → high causal contribution
    - Score near 0.0: removing this chunk barely changes output → low causal contribution
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_MASK_TOKEN = "[REMOVED]"


@dataclass
class CostBudget:
    """Hard cap on API calls and estimated cost."""

    max_api_calls: int = 500
    max_cost_usd: float = 10.0
    cost_per_call_usd: float = 0.001  # ~$0.001/call for claude-haiku

    def would_exceed(self, n_calls: int) -> bool:
        return n_calls > self.max_api_calls

    def estimated_cost(self, n_calls: int) -> float:
        return n_calls * self.cost_per_call_usd


@dataclass
class ChunkScore:
    """Attribution result for a single context chunk."""

    chunk_name: str
    attribution_score: float  # 1.0 - mean_similarity; higher = more causal
    mean_similarity: float
    std_similarity: float
    runs: int
    masked_outputs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Clamp: cosine similarity can float slightly outside [-1, 1] due to fp
        self.attribution_score = max(0.0, min(1.0, self.attribution_score))
        self.mean_similarity = max(-1.0, min(1.0, self.mean_similarity))


@dataclass
class AttributionReport:
    """Full attribution result for a single LLM call."""

    chunk_scores: Dict[str, ChunkScore]
    original_output: str
    prompt: str
    k: int
    total_api_calls: int
    estimated_cost_usd: float
    elapsed_seconds: float
    skipped_chunks: List[str] = field(default_factory=list)

    def top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-n (chunk_name, attribution_score) sorted descending."""
        sorted_chunks = sorted(
            self.chunk_scores.items(),
            key=lambda x: x[1].attribution_score,
            reverse=True,
        )
        return [(name, score.attribution_score) for name, score in sorted_chunks[:n]]

    def contributors_above(self, threshold: float) -> List[Tuple[str, float]]:
        """Return all chunks with attribution_score >= threshold."""
        return [
            (name, score.attribution_score)
            for name, score in self.chunk_scores.items()
            if score.attribution_score >= threshold
        ]

    @property
    def top_score(self) -> float:
        """Highest attribution_score across all chunks."""
        if not self.chunk_scores:
            return 0.0
        return max(s.attribution_score for s in self.chunk_scores.values())

    @property
    def attribution_heatmap(self) -> str:
        """ASCII bar chart of attribution scores, sorted descending."""
        if not self.chunk_scores:
            return "(no chunks scored)"
        max_score = max(s.attribution_score for s in self.chunk_scores.values())
        if max_score == 0.0:
            max_score = 1.0
        sorted_chunks = sorted(
            self.chunk_scores.items(),
            key=lambda x: x[1].attribution_score,
            reverse=True,
        )
        max_name_len = max(len(name) for name in self.chunk_scores)
        lines = []
        for name, score in sorted_chunks:
            bar_len = round((score.attribution_score / max_score) * 10)
            bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
            lines.append(f"{name:<{max_name_len}} [{bar}] {score.attribution_score:.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "chunk_scores": {
                name: {
                    "attribution_score": cs.attribution_score,
                    "mean_similarity": cs.mean_similarity,
                    "std_similarity": cs.std_similarity,
                    "runs": cs.runs,
                }
                for name, cs in self.chunk_scores.items()
            },
            "top_contributors": self.top_contributors(),
            "k": self.k,
            "total_api_calls": self.total_api_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "skipped_chunks": self.skipped_chunks,
        }


class BudgetExceededError(Exception):
    """Raised when the attribution run would exceed the configured budget."""


class ContextTracer:
    """
    Per-context-chunk causal attribution for LLM outputs.

    Usage::

        tracer = ContextTracer(runner=my_llm_call, k=3)
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
        print(report.top_contributors(n=3))
    """

    def __init__(
        self,
        runner: Callable[[str], str],
        embedder: Optional[object] = None,
        k: int = 3,
        budget: Optional[CostBudget] = None,
        mask_token: str = _MASK_TOKEN,
    ) -> None:
        self.runner = runner
        self.k = k
        self.budget = budget or CostBudget()
        self.mask_token = mask_token
        self._embedder = embedder  # injected; lazy-loads default on first trace()

    def _get_embedder(self):
        if self._embedder is None:
            from context_trace.embedder import SentenceTransformerEmbedder
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    def _mask_prompt(self, prompt: str, chunk_content: str, chunk_name: str) -> Optional[str]:
        """Replace chunk_content in prompt with mask token. Returns None if not found."""
        if not chunk_content or chunk_content not in prompt:
            return None
        return prompt.replace(chunk_content, f"{self.mask_token}:{chunk_name}", 1)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def trace(
        self,
        prompt: str,
        original_output: str,
        chunks: Dict[str, str],
    ) -> AttributionReport:
        """
        Compute per-chunk causal AttributionScore.

        Args:
            prompt: The full prompt sent to the LLM (must contain chunk text verbatim).
            original_output: The LLM response to the original prompt.
            chunks: Named context segments. Values must appear verbatim in prompt.

        Returns:
            AttributionReport with per-chunk scores, heatmap, and cost estimate.

        Raises:
            BudgetExceededError: If required API calls exceed budget.max_api_calls.
        """
        if not chunks:
            return AttributionReport(
                chunk_scores={},
                original_output=original_output,
                prompt=prompt,
                k=self.k,
                total_api_calls=0,
                estimated_cost_usd=0.0,
                elapsed_seconds=0.0,
            )

        required_calls = len(chunks) * self.k
        if self.budget.would_exceed(required_calls):
            raise BudgetExceededError(
                f"Attribution requires {required_calls} API calls "
                f"(budget max: {self.budget.max_api_calls}). "
                "Reduce k, reduce chunk count, or raise budget.max_api_calls."
            )

        embedder = self._get_embedder()
        start = time.monotonic()

        vec_original = embedder.embed(original_output)
        chunk_scores: Dict[str, ChunkScore] = {}
        skipped: List[str] = []
        total_calls = 0

        for chunk_name, chunk_content in chunks.items():
            masked_prompt = self._mask_prompt(prompt, chunk_content, chunk_name)
            if masked_prompt is None:
                warnings.warn(
                    f"Chunk '{chunk_name}' text not found verbatim in prompt. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                skipped.append(chunk_name)
                continue

            masked_outputs: List[str] = []
            for _ in range(self.k):
                masked_outputs.append(self.runner(masked_prompt))
                total_calls += 1

            similarities: List[float] = []
            for output in masked_outputs:
                vec_masked = embedder.embed(output)
                sim = self._cosine_similarity(vec_original, vec_masked)
                similarities.append(sim)

            mean_sim = float(np.mean(similarities))
            std_sim = float(np.std(similarities))
            attribution = 1.0 - mean_sim

            chunk_scores[chunk_name] = ChunkScore(
                chunk_name=chunk_name,
                attribution_score=attribution,
                mean_similarity=mean_sim,
                std_similarity=std_sim,
                runs=self.k,
                masked_outputs=masked_outputs,
            )

        elapsed = time.monotonic() - start
        return AttributionReport(
            chunk_scores=chunk_scores,
            original_output=original_output,
            prompt=prompt,
            k=self.k,
            total_api_calls=total_calls,
            estimated_cost_usd=self.budget.estimated_cost(total_calls),
            elapsed_seconds=elapsed,
            skipped_chunks=skipped,
        )
