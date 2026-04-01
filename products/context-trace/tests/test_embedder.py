"""Tests for context_trace.embedder."""

from __future__ import annotations

import numpy as np
import pytest

from context_trace.embedder import IdentityEmbedder, MockEmbedder


class TestMockEmbedder:
    def test_returns_unit_vector(self):
        emb = MockEmbedder(dim=8)
        vec = emb.embed("hello world")
        norm = float(np.linalg.norm(vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_deterministic_same_text(self):
        emb = MockEmbedder(dim=8)
        v1 = emb.embed("test sentence")
        v2 = emb.embed("test sentence")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_different_texts_different_vectors(self):
        emb = MockEmbedder(dim=8)
        v1 = emb.embed("hello")
        v2 = emb.embed("world")
        # Very unlikely to be equal for different hash seeds
        assert not np.allclose(v1, v2)

    def test_correct_dimension(self):
        for dim in [4, 8, 16, 64]:
            emb = MockEmbedder(dim=dim)
            vec = emb.embed("x")
            assert vec.shape == (dim,)

    def test_embed_batch_returns_matrix(self):
        emb = MockEmbedder(dim=8)
        texts = ["a", "b", "c"]
        mat = emb.embed_batch(texts)
        assert mat.shape == (3, 8)

    def test_embed_batch_consistent_with_embed(self):
        emb = MockEmbedder(dim=8)
        texts = ["hello", "world"]
        batch = emb.embed_batch(texts)
        single_0 = emb.embed("hello")
        single_1 = emb.embed("world")
        np.testing.assert_array_almost_equal(batch[0], single_0)
        np.testing.assert_array_almost_equal(batch[1], single_1)

    def test_same_text_cosine_similarity_is_one(self):
        emb = MockEmbedder(dim=16)
        text = "the quick brown fox"
        v1 = emb.embed(text)
        v2 = emb.embed(text)
        sim = float(np.dot(v1, v2))
        assert sim == pytest.approx(1.0, abs=1e-6)


class TestIdentityEmbedder:
    def test_returns_unit_vector(self):
        emb = IdentityEmbedder(dim=4)
        vec = emb.embed("anything")
        norm = float(np.linalg.norm(vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_same_vector_for_all_inputs(self):
        emb = IdentityEmbedder(dim=4)
        v1 = emb.embed("hello")
        v2 = emb.embed("completely different text")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_cosine_similarity_always_one(self):
        emb = IdentityEmbedder(dim=4)
        v1 = emb.embed("a")
        v2 = emb.embed("z")
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_correct_dimension(self):
        emb = IdentityEmbedder(dim=8)
        assert emb.embed("x").shape == (8,)

    def test_embed_batch_all_identical(self):
        emb = IdentityEmbedder(dim=4)
        batch = emb.embed_batch(["a", "b", "c"])
        assert batch.shape == (3, 4)
        np.testing.assert_array_almost_equal(batch[0], batch[1])
        np.testing.assert_array_almost_equal(batch[1], batch[2])
