"""
Hybrid retriever combining dense vector search (FAISS/numpy) and BM25 sparse
lexical retrieval, fused with Reciprocal Rank Fusion (RRF).

FAISS is used when available; otherwise a numpy-based brute-force cosine
similarity search serves as a transparent fallback. BM25 is always provided
by the rank-bm25 library.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from rank_bm25 import BM25Okapi  # type: ignore

from backend.corpus import DOCUMENTS
from backend.models import RetrievedDoc

logger = logging.getLogger(__name__)

# Embedding dimension used for all dense representations
_EMBEDDING_DIM = 128

# RRF constant — k=60 is the standard default
_RRF_K = 60


def _simple_tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _deterministic_embedding(text: str, dim: int = _EMBEDDING_DIM) -> np.ndarray:
    """
    Produce a deterministic pseudo-embedding for *text* using a seeded random
    projection keyed on the text content. This simulates a real embedding model
    and is consistent — identical text always produces the same vector.

    In production this would be replaced by a call to a sentence-transformers model
    or an embedding API endpoint.
    """
    # Create a seed from the text content deterministically
    seed = abs(hash(text)) % (2 ** 31)
    rng = np.random.default_rng(seed)
    # Base noise vector
    base = rng.standard_normal(dim).astype(np.float32)

    # Add term-frequency signal: for each unique token, add a small directional
    # perturbation so similar texts produce more similar vectors
    tokens = _simple_tokenize(text[:500])  # use first 500 chars for speed
    for token in set(tokens):
        token_seed = abs(hash(token)) % (2 ** 31)
        token_rng = np.random.default_rng(token_seed)
        direction = token_rng.standard_normal(dim).astype(np.float32)
        base += 0.3 * direction

    # L2-normalize
    norm = np.linalg.norm(base)
    if norm > 0:
        base /= norm
    return base


class _NumpyFlatIndex:
    """
    Brute-force cosine similarity index backed by numpy.
    Drop-in replacement for faiss.IndexFlatIP when FAISS is not installed.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: List[np.ndarray] = []

    def add(self, matrix: np.ndarray) -> None:
        for row in matrix:
            self._vectors.append(row.copy())

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        matrix = np.stack(self._vectors)  # (N, dim)
        # Vectors are L2-normalized, so dot product == cosine similarity
        scores = matrix @ query[0]  # (N,)
        k_actual = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k_actual]
        top_scores = scores[top_indices]
        return top_scores.reshape(1, -1), top_indices.reshape(1, -1).astype(np.int64)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse search and FAISS/numpy dense search,
    fused with Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    corpus : list of dict
        Document corpus as returned by corpus.py.
    rrf_k : int
        The RRF smoothing constant (default 60).
    dense_candidates : int
        Number of candidates to fetch from dense retrieval before fusion.
    sparse_candidates : int
        Number of candidates to fetch from BM25 before fusion.
    """

    def __init__(
        self,
        corpus: List[Dict] = None,
        rrf_k: int = _RRF_K,
        dense_candidates: int = 20,
        sparse_candidates: int = 20,
    ) -> None:
        self.corpus = corpus if corpus is not None else DOCUMENTS
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates

        self._doc_texts: List[str] = [d["text"] for d in self.corpus]
        self._doc_titles: List[str] = [d["title"] for d in self.corpus]

        logger.info("Building BM25 index over %d documents…", len(self.corpus))
        self._bm25 = self._build_bm25()

        logger.info("Building dense index (FAISS=%s)…", _FAISS_AVAILABLE)
        self._dense_index, self._embeddings = self._build_dense_index()

        logger.info("HybridRetriever ready.")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_bm25(self) -> BM25Okapi:
        tokenized = [_simple_tokenize(text) for text in self._doc_texts]
        return BM25Okapi(tokenized)

    def _build_dense_index(self):
        """Build FAISS or numpy flat index from document embeddings."""
        embeddings = np.stack([
            _deterministic_embedding(title + " " + text)
            for title, text in zip(self._doc_titles, self._doc_texts)
        ])  # shape: (N, dim)

        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            # Vectors must be float32 and L2-normalized for inner product == cosine sim
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            logger.info("FAISS index built with %d vectors.", index.ntotal)
        else:
            index = _NumpyFlatIndex(embeddings.shape[1])
            # Vectors are already L2-normalized by _deterministic_embedding
            index.add(embeddings)
            logger.info("Numpy fallback index built with %d vectors.", len(self.corpus))

        return index, embeddings

    # ------------------------------------------------------------------
    # Individual retrievers
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Return (doc_index, score) pairs from dense vector search, sorted by score desc.
        """
        q_emb = _deterministic_embedding(query).reshape(1, -1)

        if _FAISS_AVAILABLE:
            faiss.normalize_L2(q_emb)
            scores, indices = self._dense_index.search(q_emb, k)
        else:
            scores, indices = self._dense_index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for padding
                continue
            results.append((int(idx), float(score)))
        return results  # already sorted desc by score

    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Return (doc_index, score) pairs from BM25, sorted by score desc.
        """
        tokens = _simple_tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[int, float]]],
        k: int = _RRF_K,
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.

        Parameters
        ----------
        ranked_lists : list of list of (doc_index, raw_score)
            Each inner list is sorted descending by score. raw_scores are not used
            by RRF — only the rank order matters.
        k : int
            RRF smoothing constant.

        Returns
        -------
        List of (doc_index, rrf_score) sorted descending by rrf_score.
        """
        rrf_scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_idx, _) in enumerate(ranked, start=1):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """
        Perform hybrid BM25 + dense search fused with RRF and return top_k results.

        Parameters
        ----------
        query : str
            User query string.
        top_k : int
            Number of documents to return.

        Returns
        -------
        List of RetrievedDoc sorted by descending RRF score.
        """
        n_candidates = max(top_k * 3, self.dense_candidates, self.sparse_candidates)
        n_candidates = min(n_candidates, len(self.corpus))

        # Run both retrievers
        dense_results = self._dense_search(query, n_candidates)
        sparse_results = self._sparse_search(query, n_candidates)

        # Build per-doc score lookups for annotation
        dense_scores: Dict[int, float] = {idx: score for idx, score in dense_results}
        sparse_scores: Dict[int, float] = {idx: score for idx, score in sparse_results}

        # Normalize BM25 scores to [0, 1] for consistent reporting
        max_bm25 = max((s for s in sparse_scores.values()), default=1.0) or 1.0
        sparse_scores_norm = {idx: s / max_bm25 for idx, s in sparse_scores.items()}

        # Normalize dense scores: they are cosine similarities in [-1, 1]; shift to [0, 1]
        dense_scores_norm = {idx: (s + 1.0) / 2.0 for idx, s in dense_scores.items()}

        # RRF fusion
        fused = self._reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)

        # Build RetrievedDoc objects
        results: List[RetrievedDoc] = []
        for doc_idx, rrf_score in fused[:top_k]:
            doc = self.corpus[doc_idx]
            results.append(
                RetrievedDoc(
                    id=doc["id"],
                    title=doc["title"],
                    text=doc["text"],
                    source=doc["source"],
                    score=rrf_score,
                    dense_score=dense_scores_norm.get(doc_idx, 0.0),
                    sparse_score=sparse_scores_norm.get(doc_idx, 0.0),
                    rerank_score=None,
                )
            )

        return results

    def embed_query(self, query: str) -> np.ndarray:
        """
        Return the dense embedding for a query string.
        Used by the semantic cache for similarity lookup.
        """
        return _deterministic_embedding(query)
