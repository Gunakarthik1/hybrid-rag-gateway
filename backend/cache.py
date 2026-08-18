"""
Semantic cache for the Hybrid RAG Gateway.

Uses in-memory storage with TTL expiry. Similarity matching via cosine similarity
against stored query embeddings — entries with similarity >= threshold are treated
as cache hits, avoiding redundant retrieval and generation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SemanticCache:
    """
    In-memory semantic cache keyed by query embeddings.

    On each lookup the cache computes cosine similarity between the incoming
    query embedding and every stored embedding. If the maximum similarity
    exceeds *threshold* and the entry has not expired, the cached response
    is returned without executing retrieval or generation.

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold for a cache hit (default 0.96).
    ttl_seconds : int
        Time-to-live for cache entries in seconds (default 3600 = 1 hour).
    max_size : int
        Maximum number of entries before oldest entries are evicted (default 1000).
    """

    def __init__(
        self,
        threshold: float = 0.96,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
    ) -> None:
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        # Each entry: {"embedding": np.ndarray, "response": Any, "timestamp": float}
        self._store: Dict[str, Dict[str, Any]] = {}
        # Ordered list of keys for eviction (insertion order)
        self._insertion_order: List[str] = []

        # Stats counters
        self._hits: int = 0
        self._misses: int = 0
        self._total_requests: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _evict_expired(self) -> None:
        """Remove all entries whose TTL has elapsed."""
        now = time.time()
        expired_keys = [
            k for k, v in self._store.items()
            if now - v["timestamp"] > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._store[k]
            if k in self._insertion_order:
                self._insertion_order.remove(k)

    def _evict_oldest(self) -> None:
        """Remove the oldest entry when the cache is at capacity."""
        if self._insertion_order:
            oldest = self._insertion_order.pop(0)
            self._store.pop(oldest, None)

    def _make_key(self, embedding: np.ndarray) -> str:
        """Create a string key from the first few embedding dimensions (for internal indexing)."""
        # We use a counter-based unique key; similarity lookup is done by scan anyway.
        return str(id(embedding)) + "_" + str(time.monotonic_ns())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query_embedding: np.ndarray) -> Tuple[Optional[Any], float]:
        """
        Look up a cached response for the given query embedding.

        Returns
        -------
        (response, similarity) if a hit is found, else (None, 0.0).
        """
        self._total_requests += 1
        self._evict_expired()

        if not self._store:
            self._misses += 1
            return None, 0.0

        now = time.time()
        best_similarity = 0.0
        best_response = None

        for entry in self._store.values():
            # Skip if expired (belt-and-suspenders after _evict_expired)
            if now - entry["timestamp"] > self.ttl_seconds:
                continue
            sim = self._cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_response = entry["response"]

        if best_similarity >= self.threshold and best_response is not None:
            self._hits += 1
            return best_response, best_similarity

        self._misses += 1
        return None, best_similarity

    def set(self, query_embedding: np.ndarray, response: Any) -> None:
        """
        Store a response in the cache keyed by query embedding.

        If at capacity, the oldest entry is evicted first.
        """
        self._evict_expired()

        if len(self._store) >= self.max_size:
            self._evict_oldest()

        key = self._make_key(query_embedding)
        self._store[key] = {
            "embedding": query_embedding.copy(),
            "response": response,
            "timestamp": time.time(),
        }
        self._insertion_order.append(key)

    def clear(self) -> int:
        """Clear all entries. Returns number of entries removed."""
        count = len(self._store)
        self._store.clear()
        self._insertion_order.clear()
        return count

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        self._evict_expired()
        total = self._total_requests
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hit_rate": round(hit_rate, 4),
            "total_requests": total,
            "hits": self._hits,
            "misses": self._misses,
            "cache_size": len(self._store),
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters without clearing the cache."""
        self._hits = 0
        self._misses = 0
        self._total_requests = 0
