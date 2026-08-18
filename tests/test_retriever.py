"""
Tests for HybridRetriever: BM25 scoring, dense search, RRF fusion math,
and hybrid_search end-to-end behavior.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from backend.retriever import (
    HybridRetriever,
    _deterministic_embedding,
    _simple_tokenize,
)
from backend.models import RetrievedDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    """Single shared retriever instance to avoid re-building the index per test."""
    return HybridRetriever()


MINI_CORPUS = [
    {
        "id": "mini_001",
        "title": "FAISS Vector Search",
        "source": "mini/faiss.md",
        "text": "FAISS is a library for efficient similarity search of dense vectors using approximate nearest neighbor algorithms.",
    },
    {
        "id": "mini_002",
        "title": "BM25 Sparse Retrieval",
        "source": "mini/bm25.md",
        "text": "BM25 is a probabilistic ranking function used in information retrieval systems for lexical keyword matching.",
    },
    {
        "id": "mini_003",
        "title": "Kubernetes Pod Scheduling",
        "source": "mini/k8s.md",
        "text": "Kubernetes schedules pods onto nodes based on resource requests, taints, tolerations, and affinity rules.",
    },
    {
        "id": "mini_004",
        "title": "Redis Caching Patterns",
        "source": "mini/redis.md",
        "text": "Redis provides in-memory data storage with TTL expiry, pub/sub messaging, and sorted set data structures.",
    },
    {
        "id": "mini_005",
        "title": "Transformer Attention",
        "source": "mini/attention.md",
        "text": "Multi-head attention allows transformers to attend to different positions in the sequence simultaneously.",
    },
]


@pytest.fixture(scope="module")
def mini_retriever() -> HybridRetriever:
    """Retriever built on a small 5-document corpus for deterministic tests."""
    return HybridRetriever(corpus=MINI_CORPUS)


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

class TestSimpleTokenize:
    def test_lowercases(self):
        tokens = _simple_tokenize("FAISS BM25 Redis")
        assert all(t == t.lower() for t in tokens)

    def test_strips_punctuation(self):
        tokens = _simple_tokenize("hello, world! this is a test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self):
        assert _simple_tokenize("") == []

    def test_alphanumeric_retained(self):
        tokens = _simple_tokenize("top-k=10 recall@100")
        assert "top" in tokens
        assert "k" in tokens or "10" in tokens


# ---------------------------------------------------------------------------
# Deterministic embedding
# ---------------------------------------------------------------------------

class TestDeterministicEmbedding:
    def test_consistent_output(self):
        e1 = _deterministic_embedding("test query about FAISS")
        e2 = _deterministic_embedding("test query about FAISS")
        np.testing.assert_array_equal(e1, e2)

    def test_different_texts_differ(self):
        e1 = _deterministic_embedding("FAISS vector search")
        e2 = _deterministic_embedding("Kubernetes pod scheduling")
        # Should not be identical
        assert not np.allclose(e1, e2)

    def test_output_is_unit_vector(self):
        e = _deterministic_embedding("some text for testing")
        norm = np.linalg.norm(e)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_output_shape(self):
        e = _deterministic_embedding("test", dim=64)
        assert e.shape == (64,)

    def test_dtype_float32(self):
        e = _deterministic_embedding("dtype check")
        assert e.dtype == np.float32

    def test_similar_texts_closer_than_dissimilar(self):
        """Texts sharing tokens should produce more similar embeddings."""
        e_faiss1 = _deterministic_embedding("FAISS similarity search dense vectors")
        e_faiss2 = _deterministic_embedding("FAISS approximate nearest neighbor search")
        e_k8s = _deterministic_embedding("Kubernetes pod scheduling affinity taints")

        sim_faiss = float(np.dot(e_faiss1, e_faiss2))
        sim_cross = float(np.dot(e_faiss1, e_k8s))
        assert sim_faiss > sim_cross, (
            f"Expected FAISS-FAISS similarity ({sim_faiss:.4f}) > "
            f"FAISS-K8s similarity ({sim_cross:.4f})"
        )


# ---------------------------------------------------------------------------
# RRF fusion math
# ---------------------------------------------------------------------------

class TestRRFFusion:
    """Verify the mathematical properties of Reciprocal Rank Fusion."""

    def _rrf(self, ranked_lists, k=60):
        return HybridRetriever._reciprocal_rank_fusion(ranked_lists, k=k)

    def test_single_ranker_monotone_decreasing(self):
        """Documents ranked earlier get higher RRF scores with a single ranker."""
        ranked = [(i, float(10 - i)) for i in range(10)]
        fused = self._rrf([ranked])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_formula(self):
        """Verify the exact formula: score(d) = sum(1 / (k + rank_i(d)))."""
        k = 60
        # rank 1 in list A, rank 3 in list B
        list_a = [(0, 10.0), (1, 8.0), (2, 6.0)]
        list_b = [(1, 9.0), (2, 7.0), (0, 5.0)]
        fused = dict(self._rrf([list_a, list_b], k=k))

        expected_doc0 = 1.0 / (k + 1) + 1.0 / (k + 3)  # rank 1 in A, rank 3 in B
        expected_doc1 = 1.0 / (k + 2) + 1.0 / (k + 1)  # rank 2 in A, rank 1 in B
        expected_doc2 = 1.0 / (k + 3) + 1.0 / (k + 2)  # rank 3 in A, rank 2 in B

        assert abs(fused[0] - expected_doc0) < 1e-10
        assert abs(fused[1] - expected_doc1) < 1e-10
        assert abs(fused[2] - expected_doc2) < 1e-10

    def test_document_only_in_one_ranker(self):
        """A document appearing only in one ranker still gets an RRF score."""
        list_a = [(0, 1.0)]
        list_b = [(1, 1.0)]
        fused = dict(self._rrf([list_a, list_b]))
        assert 0 in fused
        assert 1 in fused

    def test_rrf_ignores_raw_scores(self):
        """
        RRF depends only on rank order, not raw scores.
        Two rankers with same ordering but different scores should yield identical RRF.
        """
        order = [0, 1, 2, 3, 4]
        list_high = [(i, 1000.0 - i * 100) for i in order]
        list_low  = [(i, 1.0 - i * 0.1) for i in order]
        fused_high = dict(self._rrf([list_high]))
        fused_low  = dict(self._rrf([list_low]))
        for doc_id in order:
            assert abs(fused_high[doc_id] - fused_low[doc_id]) < 1e-12

    def test_rrf_k_parameter_effect(self):
        """Lower k gives higher relative boost to top-ranked documents."""
        ranked = [(i, float(10 - i)) for i in range(10)]
        fused_k10  = dict(HybridRetriever._reciprocal_rank_fusion([ranked], k=10))
        fused_k100 = dict(HybridRetriever._reciprocal_rank_fusion([ranked], k=100))
        # Top doc advantage: (1/(10+1))/(1/(100+1)) = 101/11 ≈ 9.18
        ratio_k10  = fused_k10[0] / fused_k10[9]
        ratio_k100 = fused_k100[0] / fused_k100[9]
        assert ratio_k10 > ratio_k100

    def test_rrf_output_sorted_descending(self):
        """Output of RRF is sorted descending by score."""
        list_a = [(i, float(10 - i)) for i in range(8)]
        list_b = [(7 - i, float(10 - i)) for i in range(8)]
        fused = self._rrf([list_a, list_b])
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_input(self):
        """Empty ranked lists produce empty output."""
        result = self._rrf([[], []])
        assert result == []

    def test_rrf_k60_rank1_value(self):
        """Verify exact value: rank 1 with k=60 gives 1/61."""
        ranked = [(42, 999.0)]
        fused = dict(self._rrf([ranked], k=60))
        assert abs(fused[42] - 1.0 / 61.0) < 1e-12


# ---------------------------------------------------------------------------
# Hybrid search integration
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_returns_correct_number_of_docs(self, retriever):
        results = retriever.hybrid_search("vector database search", top_k=5)
        assert len(results) == 5

    def test_top_k_respected(self, retriever):
        for k in [1, 3, 7, 10]:
            results = retriever.hybrid_search("distributed systems consensus", top_k=k)
            assert len(results) == k

    def test_returns_retrieved_doc_instances(self, retriever):
        results = retriever.hybrid_search("BM25 retrieval", top_k=3)
        for doc in results:
            assert isinstance(doc, RetrievedDoc)

    def test_rrf_scores_descending(self, retriever):
        results = retriever.hybrid_search("FAISS approximate nearest neighbor", top_k=10)
        scores = [d.score for d in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by RRF score desc"

    def test_scores_positive(self, retriever):
        results = retriever.hybrid_search("semantic cache cosine similarity", top_k=5)
        for doc in results:
            assert doc.score > 0, "RRF score must be positive"

    def test_dense_and_sparse_scores_populated(self, retriever):
        results = retriever.hybrid_search("cross encoder reranking", top_k=5)
        for doc in results:
            assert 0.0 <= doc.dense_score <= 1.0, f"dense_score out of range: {doc.dense_score}"
            assert 0.0 <= doc.sparse_score <= 1.0, f"sparse_score out of range: {doc.sparse_score}"

    def test_rerank_score_initially_none(self, retriever):
        """hybrid_search does not populate rerank_score — that is the reranker's job."""
        results = retriever.hybrid_search("vector quantization product", top_k=5)
        for doc in results:
            assert doc.rerank_score is None

    def test_relevant_doc_in_top_results_faiss(self, mini_retriever):
        """Query about FAISS should surface the FAISS document in top-3."""
        results = mini_retriever.hybrid_search("FAISS dense vector similarity search", top_k=3)
        ids = [d.id for d in results]
        assert "mini_001" in ids, f"Expected mini_001 (FAISS doc) in top-3, got {ids}"

    def test_relevant_doc_in_top_results_redis(self, mini_retriever):
        """Query about Redis should surface the Redis document in top-3."""
        results = mini_retriever.hybrid_search("Redis TTL expiry caching", top_k=3)
        ids = [d.id for d in results]
        assert "mini_004" in ids, f"Expected mini_004 (Redis doc) in top-3, got {ids}"

    def test_no_duplicate_ids(self, retriever):
        results = retriever.hybrid_search("hybrid search RRF fusion", top_k=10)
        ids = [d.id for d in results]
        assert len(ids) == len(set(ids)), "Duplicate document IDs in results"

    def test_doc_fields_populated(self, retriever):
        results = retriever.hybrid_search("Raft consensus distributed systems", top_k=3)
        for doc in results:
            assert doc.id
            assert doc.title
            assert doc.text
            assert doc.source

    def test_embed_query_unit_vector(self, retriever):
        emb = retriever.embed_query("test query embedding")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_single_word_query(self, retriever):
        results = retriever.hybrid_search("FAISS", top_k=3)
        assert len(results) == 3

    def test_very_long_query(self, retriever):
        long_query = " ".join(["vector database similarity search retrieval"] * 20)
        results = retriever.hybrid_search(long_query, top_k=5)
        assert len(results) == 5

    def test_query_unrelated_to_corpus(self, mini_retriever):
        """Even an unrelated query returns results (all documents have some score)."""
        results = mini_retriever.hybrid_search("ancient roman architecture colosseum", top_k=3)
        assert len(results) == 3
