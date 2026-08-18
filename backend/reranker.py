"""
Cross-encoder reranker for the Hybrid RAG Gateway.

Simulates cross-encoder scoring using token overlap, length penalty,
and keyword boost without requiring a GPU or external model download.
The scoring function captures the same signal as a lightweight cross-encoder:
lexical overlap between query terms and document terms, boosted by position
of query terms in the document and penalized for very long documents that
dilute the relevant signal.
"""

from __future__ import annotations

import math
import re
from typing import List

from backend.models import RetrievedDoc


class CrossEncoderReranker:
    """
    Simulated cross-encoder reranker.

    Scores query-document pairs using:
    - Token overlap (Jaccard-like) between query and document
    - Positional boost: query terms appearing early in the document score higher
    - Length penalty: very long documents are gently penalized
    - Keyword density: ratio of query tokens found in document, weighted by IDF proxy
    - Title match bonus: query terms appearing in the document title score extra

    Parameters
    ----------
    alpha : float
        Weight for token overlap component (0–1).
    beta : float
        Weight for keyword density component.
    title_boost : float
        Additive bonus when query terms appear in the document title.
    length_penalty_factor : float
        Controls how aggressively long documents are penalized.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        title_boost: float = 0.15,
        length_penalty_factor: float = 0.0002,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.title_boost = title_boost
        self.length_penalty_factor = length_penalty_factor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase and split text into alphabetic tokens, removing stopwords."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "against", "between", "through", "during", "before", "after", "above",
            "below", "from", "up", "down", "out", "off", "over", "under", "again",
            "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
            "not", "only", "own", "same", "than", "too", "very", "just", "that",
            "this", "these", "those", "it", "its", "their", "they", "them", "we",
            "our", "which", "who", "whom", "what", "when", "where", "why", "how",
        }
        tokens = re.findall(r"[a-z]+", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def _token_overlap(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Jaccard-like token overlap between query and document token sets."""
        q_set = set(query_tokens)
        d_set = set(doc_tokens)
        if not q_set:
            return 0.0
        intersection = q_set & d_set
        union = q_set | d_set
        return len(intersection) / len(union)

    def _keyword_density(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Fraction of unique query tokens found in the document,
        weighted by a simple IDF proxy (inverse log of frequency bucket).
        """
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_set = set(doc_tokens)
        # Rare query terms (longer words) get higher weight as IDF proxy
        total_weight = 0.0
        matched_weight = 0.0
        for token in set(query_tokens):
            # Use word length as IDF proxy — longer words are rarer and more specific
            weight = math.log1p(len(token))
            total_weight += weight
            if token in doc_set:
                matched_weight += weight
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def _positional_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Score based on how early in the document query terms appear.
        Terms found in the first 30% of the document receive full credit;
        later appearances receive decayed credit.
        """
        if not query_tokens or not doc_tokens:
            return 0.0
        n = len(doc_tokens)
        cutoff_30 = max(1, int(n * 0.30))
        q_set = set(query_tokens)
        score = 0.0
        for i, token in enumerate(doc_tokens):
            if token in q_set:
                # Exponential decay: full credit in first 30%, decays to ~37% at end
                decay = math.exp(-3.0 * i / n)
                score += decay
        # Normalize by number of query tokens
        return min(1.0, score / max(1, len(q_set)))

    def _length_penalty(self, doc_tokens: List[str]) -> float:
        """Gentle penalty for very long documents (above ~300 tokens)."""
        n = len(doc_tokens)
        return 1.0 / (1.0 + self.length_penalty_factor * max(0, n - 300))

    def _title_match_score(self, query_tokens: List[str], title: str) -> float:
        """Bonus for query terms appearing in the document title."""
        if not query_tokens:
            return 0.0
        title_tokens = set(self._tokenize(title))
        q_set = set(query_tokens)
        overlap = q_set & title_tokens
        return len(overlap) / len(q_set) if q_set else 0.0

    def score(self, query: str, doc: RetrievedDoc) -> float:
        """
        Compute cross-encoder relevance score for a (query, document) pair.

        Returns a score in approximately [0, 1].
        """
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(doc.text)

        overlap = self._token_overlap(q_tokens, d_tokens)
        density = self._keyword_density(q_tokens, d_tokens)
        positional = self._positional_score(q_tokens, d_tokens)
        length_pen = self._length_penalty(d_tokens)
        title_bonus = self._title_match_score(q_tokens, doc.title)

        raw_score = (
            self.alpha * overlap
            + self.beta * density
            + (1.0 - self.alpha - self.beta) * positional
        ) * length_pen + self.title_boost * title_bonus

        return min(1.0, max(0.0, raw_score))

    def rerank(self, query: str, docs: List[RetrievedDoc], top_n: int = 5) -> List[RetrievedDoc]:
        """
        Rerank a list of retrieved documents and return the top_n most relevant.

        Each document's *rerank_score* field is populated with the cross-encoder score.
        The list is sorted descending by rerank_score.

        Parameters
        ----------
        query : str
            The user's original query string.
        docs : List[RetrievedDoc]
            Candidate documents from the hybrid retriever.
        top_n : int
            Number of documents to return after reranking.

        Returns
        -------
        List[RetrievedDoc] sorted by rerank_score descending, length <= top_n.
        """
        scored: List[RetrievedDoc] = []
        for doc in docs:
            rerank_score = self.score(query, doc)
            scored.append(doc.model_copy(update={"rerank_score": rerank_score}))

        scored.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
        return scored[:top_n]
