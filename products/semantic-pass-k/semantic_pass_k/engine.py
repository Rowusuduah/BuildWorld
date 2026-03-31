"""
semantic_pass_k.engine
----------------------
Core semantic consistency measurement engine.

Algorithm (Numbers 23:19 / PAT-062):
  1. Collect k outputs from the same agent on the same prompt
  2. Embed each output (TF-IDF cosine by default; injectable for sentence-transformers)
  3. Compute all n*(n-1)/2 pairwise cosine similarities
  4. ConsistencyScore = mean of pairwise similarities
  5. Compare ConsistencyScore against criticality-tier threshold → verdict

Default similarity: smoothed TF-IDF cosine (zero external dependencies).
Injectable similarity_fn: use sentence-transformers or any other embedder.
"""
from __future__ import annotations

import math
import re
import uuid
import hashlib
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, List, Optional

from .models import (
    CriticalityLevel,
    ConsistencyResult,
    ConsistencyVerdict,
    get_threshold,
    score_to_verdict,
)


# ── TF-IDF Similarity (zero dependencies) ────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """
    Smoothed TF-IDF cosine similarity between two texts.
    Zero external dependencies — uses stdlib only.
    Returns value in [0.0, 1.0].

    Uses add-1 smoothed IDF to prevent near-zero weights for terms
    shared across both documents (which collapses similarity in
    small 2-document corpora).
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    vocab = set(tf_a) | set(tf_b)

    # Smoothed IDF: log(1 + N/df) where N=2
    idf: dict[str, float] = {}
    for term in vocab:
        df = sum(1 for tf in (tf_a, tf_b) if term in tf)
        idf[term] = math.log(1.0 + 2.0 / df)

    def tfidf_vec(tf: Counter) -> dict[str, float]:
        total = sum(tf.values()) or 1
        return {t: (tf[t] / total) * idf[t] for t in tf}

    vec_a = tfidf_vec(tf_a)
    vec_b = tfidf_vec(tf_b)

    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vocab)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values())) or 1e-9
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values())) or 1e-9

    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _neural_cosine(text_a: str, text_b: str) -> float:
    """
    Neural cosine similarity via sentence-transformers.
    Falls back to TF-IDF if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_a, emb_b = model.encode([text_a, text_b])
        cos = float(
            np.dot(emb_a, emb_b)
            / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9)
        )
        return max(0.0, min(1.0, cos))
    except ImportError:
        return _tfidf_cosine(text_a, text_b)


# ── ConsistencyEngine ─────────────────────────────────────────────────────────

class ConsistencyEngine:
    """
    Measures semantic consistency across k outputs of the same agent/prompt.

    Usage (with injected similarity function for testing):
        engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.95)
        result = engine.evaluate(
            prompt="What is Ghana's GDP?",
            outputs=["GDP is $50B", "GDP is about $50 billion", "GDP is $50B USD"],
            criticality="HIGH",
        )
        print(result.verdict)  # CONSISTENT / INCONSISTENT / BORDERLINE

    Usage (default TF-IDF, zero dependencies):
        engine = ConsistencyEngine()
        ...

    Usage (neural embeddings):
        engine = ConsistencyEngine(use_neural=True)
        ...
    """

    def __init__(
        self,
        *,
        use_neural: bool = False,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        borderline_band: float = 0.05,
        agent_label: str = "default",
    ) -> None:
        """
        Args:
            use_neural: Use sentence-transformers (all-MiniLM-L6-v2) by default.
                        Falls back to TF-IDF if not installed.
            similarity_fn: Injectable pairwise similarity function(a, b) -> [0, 1].
                           Overrides both TF-IDF and neural. Useful for tests.
            borderline_band: Score range below threshold that is "BORDERLINE"
                             rather than "INCONSISTENT".
            agent_label: Human-readable label for the agent under test.
        """
        self._similarity_fn = similarity_fn
        self.use_neural = use_neural
        self.borderline_band = borderline_band
        self.agent_label = agent_label

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        if self._similarity_fn is not None:
            return self._similarity_fn(text_a, text_b)
        if self.use_neural:
            return _neural_cosine(text_a, text_b)
        return _tfidf_cosine(text_a, text_b)

    def _pairwise_scores(self, outputs: List[str]) -> List[float]:
        """Compute all n*(n-1)/2 pairwise cosine similarities (upper triangle)."""
        n = len(outputs)
        return [
            self._compute_similarity(outputs[i], outputs[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]

    def evaluate(
        self,
        prompt: str,
        outputs: List[str],
        criticality: CriticalityLevel = "HIGH",
    ) -> ConsistencyResult:
        """
        Evaluate semantic consistency across a set of outputs.

        Args:
            prompt: The prompt that produced all outputs.
            outputs: List of k outputs from the same agent on the same prompt.
                     Must have at least 2 elements.
            criticality: Task criticality tier for threshold selection.

        Returns:
            ConsistencyResult with score, verdict, and pairwise details.

        Raises:
            ValueError: If fewer than 2 outputs are provided.
        """
        if len(outputs) < 2:
            raise ValueError(
                f"semantic-pass-k requires at least 2 outputs for consistency "
                f"measurement. Got {len(outputs)}."
            )

        pairwise = self._pairwise_scores(outputs)
        consistency_score = sum(pairwise) / len(pairwise)
        threshold = get_threshold(criticality)
        verdict = score_to_verdict(consistency_score, criticality, self.borderline_band)

        return ConsistencyResult(
            run_id=str(uuid.uuid4()),
            prompt=prompt,
            outputs=outputs,
            k=len(outputs),
            consistency_score=consistency_score,
            pairwise_scores=pairwise,
            verdict=verdict,
            criticality=criticality,
            threshold=threshold,
            borderline_band=self.borderline_band,
            agent_label=self.agent_label,
            tested_at=datetime.now(timezone.utc),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        )

    def evaluate_batch(
        self,
        prompts: List[str],
        outputs_per_prompt: List[List[str]],
        criticality: CriticalityLevel = "HIGH",
    ) -> List[ConsistencyResult]:
        """
        Evaluate consistency for multiple prompts at once.

        Args:
            prompts: List of prompts.
            outputs_per_prompt: List of output lists, one per prompt.
                                Each inner list must have at least 2 elements.
            criticality: Applied uniformly to all prompts.

        Returns:
            List of ConsistencyResult, one per prompt.
        """
        if not prompts:
            raise ValueError("prompts list cannot be empty.")
        if len(prompts) != len(outputs_per_prompt):
            raise ValueError(
                "prompts and outputs_per_prompt must have the same length."
            )
        return [
            self.evaluate(p, outs, criticality)
            for p, outs in zip(prompts, outputs_per_prompt)
        ]
