"""context_trace.embedder
~~~~~~~~~~~~~~~~~~~~~~
Embedding abstraction for attribution scoring.

Default: SentenceTransformerEmbedder (lazy-loaded, all-MiniLM-L6-v2).
Testing: MockEmbedder (hash-seeded deterministic unit vectors, no model download).
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Interface that all embedders must satisfy."""

    def embed(self, text: str) -> np.ndarray:
        ...

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """
    Sentence-transformers embedder (lazy-loaded on first call).

    Default model: all-MiniLM-L6-v2
        - 22 MB download
        - 384-dimensional embeddings
        - CPU-friendly, ~5ms/embed on modern hardware
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> np.ndarray:
        self._load()
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        self._load()
        return self._model.encode(texts, normalize_embeddings=True)


class MockEmbedder:
    """
    Deterministic test embedder — no model download required.

    Returns a unit vector seeded from hash(text). Same text → same vector.
    Different texts → independent random vectors (expected cosine similarity ≈ 0).

    Use in tests by injecting into ContextTracer::

        tracer = ContextTracer(runner=my_runner, embedder=MockEmbedder())
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self.dim)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return np.ones(self.dim, dtype=float) / (self.dim ** 0.5)
        return vec / norm

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


class IdentityEmbedder:
    """
    Test embedder that returns the same fixed vector for every input.

    Useful for verifying that attribution_score = 0 when masked output
    is semantically identical to original (cosine similarity = 1.0).
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self._vec = np.ones(dim, dtype=float) / (dim ** 0.5)

    def embed(self, text: str) -> np.ndarray:
        return self._vec.copy()

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])
