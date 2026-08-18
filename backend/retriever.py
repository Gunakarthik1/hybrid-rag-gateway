"""
Hybrid retriever combining TF-IDF/numpy dense vector search and BM25 sparse
lexical retrieval, fused with Reciprocal Rank Fusion (RRF).

Uses sklearn TfidfVectorizer + numpy dot product similarity for dense retrieval
instead of FAISS or sentence-transformers, keeping memory well within 512MB.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore

from backend.corpus import DOCUMENTS
from backend.models import RetrievedDoc

logger = logging.getLogger(__name__)

# RRF constant — k=60 is the standard default
_RRF_K = 60


def _simple_tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse search and TF-IDF dense search,
    fused with Reciprocal Rank Fusion (RRF).

    Uses sklearn TfidfVectorizer + numpy cosine similarity for the dense
    (FAISS-equivalent) retrieval path to stay within the 512MB RAM limit.

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
        self.corpus = corpus if corpus is not None else list(DOCUMENTS)
        self.rrf_k = rrf_k
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates

        self._rebuild_indexes()
        logger.info("HybridRetriever ready.")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        """Build (or rebuild) BM25 and TF-IDF indexes from the current corpus."""
        self._doc_texts: List[str] = [d["text"] for d in self.corpus]
        self._doc_titles: List[str] = [d["title"] for d in self.corpus]

        combined = [
            d["title"] + " " + d["text"] for d in self.corpus
        ]

        logger.info("Building BM25 index over %d documents…", len(self.corpus))
        self._bm25 = self._build_bm25()

        logger.info("Building TF-IDF (FAISS-equivalent) index over %d documents…", len(self.corpus))
        self._tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(combined)
        # Normalize rows so dot product == cosine similarity
        norms = np.sqrt(self._tfidf_matrix.multiply(self._tfidf_matrix).sum(axis=1))
        norms = np.asarray(norms).flatten()
        norms[norms == 0] = 1.0
        # Keep as sparse; normalization applied during search
        self._tfidf_norms = norms

        logger.info("Indexes built: %d docs, tfidf shape=%s", len(self.corpus), self._tfidf_matrix.shape)

    def _build_bm25(self) -> BM25Okapi:
        tokenized = [_simple_tokenize(t + " " + tx) for t, tx in zip(self._doc_titles, self._doc_texts)]
        return BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Individual retrievers
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Return (doc_index, score) pairs from TF-IDF cosine similarity search.
        This is the FAISS-equivalent dense retrieval path.
        """
        q_vec = self._tfidf_vectorizer.transform([query])
        # Cosine similarity: (q_vec @ tfidf_matrix.T) / (||q|| * ||doc||)
        q_norm = np.sqrt(q_vec.multiply(q_vec).sum())
        if q_norm == 0:
            return []
        dot_products = (self._tfidf_matrix @ q_vec.T).toarray().flatten()
        cos_sims = dot_products / (self._tfidf_norms * float(q_norm))

        k_actual = min(k, len(cos_sims))
        top_indices = np.argsort(cos_sims)[::-1][:k_actual]
        return [(int(idx), float(cos_sims[idx])) for idx in top_indices]

    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Return (doc_index, score) pairs from BM25, sorted by score desc.
        """
        tokens = _simple_tokenize(query)
        scores = self._bm25.get_scores(tokens)
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
        """
        rrf_scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_idx, _) in enumerate(ranked, start=1):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_separate(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run all three retrievers and return results separately.

        Returns
        -------
        (bm25_results, faiss_results, hybrid_results)
        Each is a list of dicts: {doc_id, title, text, score, rank}
        """
        n_candidates = max(top_k * 3, self.dense_candidates, self.sparse_candidates)
        n_candidates = min(n_candidates, len(self.corpus))

        dense_results = self._dense_search(query, n_candidates)
        sparse_results = self._sparse_search(query, n_candidates)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max((s for _, s in sparse_results), default=1.0) or 1.0
        # Dense (TF-IDF cosine) already in [0, 1]

        def _make_result(doc_idx: int, score: float, rank: int) -> Dict:
            doc = self.corpus[doc_idx]
            return {
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "score": round(score, 6),
                "rank": rank,
            }

        bm25_results = [
            _make_result(idx, score / max_bm25, rank)
            for rank, (idx, score) in enumerate(sparse_results[:top_k], start=1)
        ]

        faiss_results = [
            _make_result(idx, score, rank)
            for rank, (idx, score) in enumerate(dense_results[:top_k], start=1)
        ]

        # RRF fusion
        fused = self._reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        hybrid_results = [
            _make_result(idx, rrf_score, rank)
            for rank, (idx, rrf_score) in enumerate(fused[:top_k], start=1)
        ]

        return bm25_results, faiss_results, hybrid_results

    def hybrid_search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        """
        Perform hybrid BM25 + dense search fused with RRF and return top_k results.
        Backward-compatible API used by the /api/query streaming endpoint.
        """
        n_candidates = max(top_k * 3, self.dense_candidates, self.sparse_candidates)
        n_candidates = min(n_candidates, len(self.corpus))

        dense_results = self._dense_search(query, n_candidates)
        sparse_results = self._sparse_search(query, n_candidates)

        dense_scores: Dict[int, float] = {idx: score for idx, score in dense_results}
        sparse_scores: Dict[int, float] = {idx: score for idx, score in sparse_results}

        max_bm25 = max((s for s in sparse_scores.values()), default=1.0) or 1.0
        sparse_scores_norm = {idx: s / max_bm25 for idx, s in sparse_scores.items()}

        fused = self._reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)

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
                    dense_score=dense_scores.get(doc_idx, 0.0),
                    sparse_score=sparse_scores_norm.get(doc_idx, 0.0),
                    rerank_score=None,
                )
            )

        return results

    def add_document(self, doc: Dict) -> None:
        """Add a single document to the corpus and rebuild indexes."""
        self.corpus.append(doc)
        self._rebuild_indexes()
        logger.info("Added document '%s'; corpus size now %d.", doc.get("title"), len(self.corpus))

    def embed_query(self, query: str) -> np.ndarray:
        """
        Return a TF-IDF vector for the query for semantic cache similarity lookup.
        Returns a dense numpy array (1-D, L2-normalized).
        """
        q_vec = self._tfidf_vectorizer.transform([query])
        dense = np.asarray(q_vec.todense()).flatten()
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense = dense / norm
        return dense

    def index_size_mb(self) -> float:
        """Approximate size of the TF-IDF matrix in MB."""
        import sys
        mat = self._tfidf_matrix
        # sparse CSR: data + indices + indptr arrays
        nbytes = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
        return round(nbytes / (1024 * 1024), 3)
