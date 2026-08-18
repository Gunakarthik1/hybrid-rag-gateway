"""
Tests for SemanticCache: cosine similarity threshold, TTL expiry,
cache statistics, eviction, and edge cases.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from backend.cache import SemanticCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of v."""
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def _random_unit(dim: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return _unit(v)


def _near_unit(base: np.ndarray, noise_scale: float = 0.01, seed: int = 99) -> np.ndarray:
    """Return a vector very close (high cosine similarity) to *base*."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(base.shape).astype(np.float32) * noise_scale
    return _unit(base + noise)


def _orthogonal_unit(base: np.ndarray, seed: int = 7) -> np.ndarray:
    """Return a unit vector that has low similarity with *base*."""
    rng = np.random.default_rng(seed)
    dim = base.shape[0]
    candidate = rng.standard_normal(dim).astype(np.float32)
    # Gram-Schmidt: remove component along base
    candidate = candidate - np.dot(candidate, base) * base
    return _unit(candidate)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache() -> SemanticCache:
    """Fresh cache with default settings for each test."""
    return SemanticCache(threshold=0.96, ttl_seconds=60, max_size=100)


@pytest.fixture
def tight_cache() -> SemanticCache:
    """Cache with threshold 0.99 — only near-identical vectors hit."""
    return SemanticCache(threshold=0.99, ttl_seconds=60, max_size=100)


@pytest.fixture
def loose_cache() -> SemanticCache:
    """Cache with threshold 0.50 — almost any non-orthogonal vector hits."""
    return SemanticCache(threshold=0.50, ttl_seconds=60, max_size=100)


# ---------------------------------------------------------------------------
# Cosine similarity threshold tests
# ---------------------------------------------------------------------------

class TestCacheSimilarityThreshold:
    def test_exact_match_is_hit(self, cache):
        emb = _random_unit(seed=1)
        cache.set(emb, {"answer": "hello"})
        result, sim = cache.get(emb)
        assert result is not None
        assert result["answer"] == "hello"
        assert sim >= 0.96

    def test_very_similar_vector_is_hit(self, cache):
        emb = _random_unit(seed=2)
        similar = _near_unit(emb, noise_scale=0.005, seed=10)
        cache.set(emb, {"answer": "world"})
        result, sim = cache.get(similar)
        assert result is not None, f"Expected cache hit, got miss (similarity={sim:.4f})"

    def test_orthogonal_vector_is_miss(self, cache):
        emb = _random_unit(seed=3)
        orthogonal = _orthogonal_unit(emb, seed=15)
        cache.set(emb, {"answer": "foo"})
        result, sim = cache.get(orthogonal)
        assert result is None, f"Expected cache miss, got hit (similarity={sim:.4f})"

    def test_negated_vector_is_miss(self, cache):
        """Negated (anti-parallel) vector has cosine similarity -1.0 — must miss."""
        emb = _random_unit(seed=4)
        anti = _unit(-emb)
        cache.set(emb, {"answer": "bar"})
        result, _ = cache.get(anti)
        assert result is None

    def test_threshold_boundary_just_below(self, cache):
        """
        Construct a vector with similarity just below 0.96 — should miss.
        We achieve this by interpolating: sim = cos(angle) and choosing angle slightly > arccos(0.96).
        """
        emb = _random_unit(seed=5)
        ortho = _orthogonal_unit(emb, seed=20)
        target_sim = 0.955  # just below 0.96
        alpha = target_sim
        beta = np.sqrt(1 - alpha ** 2)
        near_miss = _unit(alpha * emb + beta * ortho)
        actual_sim = float(np.dot(near_miss, emb))
        cache.set(emb, {"answer": "boundary"})
        result, sim = cache.get(near_miss)
        assert result is None, (
            f"Expected miss at sim={actual_sim:.4f} (threshold=0.96), got hit"
        )

    def test_threshold_boundary_just_above(self, cache):
        """
        Construct a vector with similarity just above 0.96 — should hit.
        """
        emb = _random_unit(seed=6)
        ortho = _orthogonal_unit(emb, seed=21)
        target_sim = 0.965  # just above 0.96
        alpha = target_sim
        beta = np.sqrt(1 - alpha ** 2)
        near_hit = _unit(alpha * emb + beta * ortho)
        actual_sim = float(np.dot(near_hit, emb))
        cache.set(emb, {"answer": "above boundary"})
        result, sim = cache.get(near_hit)
        assert result is not None, (
            f"Expected hit at sim={actual_sim:.4f} (threshold=0.96), got miss"
        )

    def test_tight_threshold_rejects_moderately_similar(self, tight_cache):
        """With threshold=0.99, moderate similarity (0.97) should miss."""
        emb = _random_unit(seed=7)
        ortho = _orthogonal_unit(emb, seed=22)
        target_sim = 0.97
        alpha = target_sim
        beta = np.sqrt(1 - alpha ** 2)
        similar = _unit(alpha * emb + beta * ortho)
        tight_cache.set(emb, {"answer": "tight"})
        result, _ = tight_cache.get(similar)
        assert result is None

    def test_loose_threshold_accepts_moderate_similarity(self, loose_cache):
        """With threshold=0.50, moderate similarity should hit."""
        emb = _random_unit(seed=8)
        ortho = _orthogonal_unit(emb, seed=23)
        target_sim = 0.7
        alpha = target_sim
        beta = np.sqrt(1 - alpha ** 2)
        similar = _unit(alpha * emb + beta * ortho)
        loose_cache.set(emb, {"answer": "loose"})
        result, sim = loose_cache.get(similar)
        assert result is not None, f"Expected hit with loose threshold, sim={sim:.4f}"

    def test_best_match_returned_when_multiple_entries(self, cache):
        """When multiple entries exist, the one with highest similarity is returned."""
        emb_base = _random_unit(seed=9)
        ortho1 = _orthogonal_unit(emb_base, seed=30)
        ortho2 = _orthogonal_unit(emb_base, seed=31)

        # Two stored entries at different similarities from query
        close = _unit(0.98 * emb_base + 0.1 * ortho1)
        far   = _unit(0.97 * emb_base + 0.1 * ortho2)
        cache.set(close, {"answer": "close"})
        cache.set(far, {"answer": "far"})

        query = emb_base  # most similar to 'close'
        result, sim = cache.get(query)
        assert result is not None
        # Result should be the 'close' entry (highest similarity to query)
        assert result["answer"] == "close"


# ---------------------------------------------------------------------------
# TTL expiry tests
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def test_entry_valid_within_ttl(self):
        cache = SemanticCache(threshold=0.96, ttl_seconds=10, max_size=100)
        emb = _random_unit(seed=50)
        cache.set(emb, {"answer": "valid"})
        result, _ = cache.get(emb)
        assert result is not None

    def test_entry_expired_after_ttl(self):
        """Mock time to simulate TTL expiry without sleeping."""
        cache = SemanticCache(threshold=0.96, ttl_seconds=5, max_size=100)
        emb = _random_unit(seed=51)

        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.set(emb, {"answer": "expires"})

        # Simulate 10 seconds passing (> ttl_seconds=5)
        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1010.0
            result, _ = cache.get(emb)
        assert result is None, "Entry should have expired after TTL elapsed"

    def test_unexpired_entry_survives_eviction_sweep(self):
        """Unexpired entries should survive _evict_expired()."""
        cache = SemanticCache(threshold=0.96, ttl_seconds=3600, max_size=100)
        emb = _random_unit(seed=52)
        cache.set(emb, {"answer": "survives"})
        cache._evict_expired()  # manually trigger sweep
        result, _ = cache.get(emb)
        assert result is not None

    def test_multiple_entries_partial_expiry(self):
        """Some entries expire while others remain valid."""
        cache = SemanticCache(threshold=0.96, ttl_seconds=5, max_size=100)
        emb_old = _random_unit(seed=60)
        emb_new = _random_unit(seed=61)

        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.set(emb_old, {"answer": "old"})

        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1003.0
            cache.set(emb_new, {"answer": "new"})

        # 7 seconds later: old (set at t=1000, ttl=5) expired, new (set at t=1003) valid
        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1007.0
            result_old, _ = cache.get(emb_old)
            result_new, _ = cache.get(emb_new)

        assert result_old is None, "Old entry should have expired"
        assert result_new is not None, "New entry should still be valid"

    def test_cache_size_shrinks_after_expiry_sweep(self):
        cache = SemanticCache(threshold=0.96, ttl_seconds=5, max_size=100)
        embeddings = [_random_unit(seed=70 + i) for i in range(5)]

        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            for i, emb in enumerate(embeddings):
                cache.set(emb, {"answer": f"doc_{i}"})
            # Check size while mock time is still active (entries not yet expired)
            assert len(cache._store) == 5

        # Advance time past TTL and evict within the same mock context
        with patch("backend.cache.time") as mock_time:
            mock_time.time.return_value = 1010.0
            cache._evict_expired()
            assert len(cache._store) == 0


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_initial_stats_zeroed(self):
        cache = SemanticCache()
        stats = cache.stats()
        assert stats["total_requests"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["cache_size"] == 0

    def test_miss_increments_miss_counter(self, cache):
        emb = _random_unit(seed=80)
        cache.get(emb)  # nothing stored — must be a miss
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["total_requests"] == 1
        assert stats["hits"] == 0

    def test_hit_increments_hit_counter(self, cache):
        emb = _random_unit(seed=81)
        cache.set(emb, {"answer": "hit test"})
        cache.get(emb)
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["total_requests"] == 1

    def test_hit_rate_calculation(self, cache):
        emb = _random_unit(seed=82)
        cache.set(emb, {"answer": "rate test"})
        # 3 hits, 1 miss
        cache.get(emb)
        cache.get(emb)
        cache.get(emb)
        cache.get(_random_unit(seed=99))  # miss
        stats = cache.stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["total_requests"] == 4
        assert abs(stats["hit_rate"] - 0.75) < 1e-4

    def test_cache_size_tracks_inserts(self, cache):
        for i in range(5):
            cache.set(_random_unit(seed=100 + i), {"answer": f"doc {i}"})
        assert cache.stats()["cache_size"] == 5

    def test_reset_stats_clears_counters_not_data(self, cache):
        emb = _random_unit(seed=90)
        cache.set(emb, {"answer": "reset"})
        cache.get(emb)
        cache.reset_stats()
        stats = cache.stats()
        assert stats["total_requests"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        # Data should still be there
        result, _ = cache.get(emb)
        assert result is not None


# ---------------------------------------------------------------------------
# Eviction and capacity tests
# ---------------------------------------------------------------------------

class TestCacheEviction:
    def test_max_size_respected(self):
        cache = SemanticCache(threshold=0.96, ttl_seconds=3600, max_size=5)
        for i in range(10):
            cache.set(_random_unit(seed=200 + i), {"answer": f"doc {i}"})
        assert cache.stats()["cache_size"] <= 5

    def test_clear_removes_all_entries(self, cache):
        for i in range(5):
            cache.set(_random_unit(seed=300 + i), {"answer": f"doc {i}"})
        removed = cache.clear()
        assert removed == 5
        assert cache.stats()["cache_size"] == 0

    def test_clear_returns_correct_count(self, cache):
        n = 7
        for i in range(n):
            cache.set(_random_unit(seed=400 + i), {"answer": f"doc {i}"})
        removed = cache.clear()
        assert removed == n

    def test_empty_cache_miss(self, cache):
        emb = _random_unit(seed=500)
        result, sim = cache.get(emb)
        assert result is None
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCacheEdgeCases:
    def test_zero_vector_no_crash(self, cache):
        """Zero vector should not cause division by zero."""
        zero = np.zeros(128, dtype=np.float32)
        cache.set(_random_unit(seed=600), {"answer": "valid"})
        result, sim = cache.get(zero)
        # Zero vector has undefined cosine similarity — should miss gracefully
        assert sim == 0.0

    def test_overwrite_existing_similar_entry(self, cache):
        """Setting a very similar vector creates a new entry; both can be retrieved."""
        emb1 = _random_unit(seed=700)
        emb2 = _near_unit(emb1, noise_scale=0.001, seed=701)
        cache.set(emb1, {"answer": "first"})
        cache.set(emb2, {"answer": "second"})
        # Both are stored; lookup returns the best match
        result, sim = cache.get(emb1)
        assert result is not None
        assert sim >= 0.96

    def test_response_value_preserved(self, cache):
        """Arbitrary response objects are stored and retrieved correctly."""
        emb = _random_unit(seed=800)
        complex_response = {
            "answer": "test answer",
            "docs": [{"id": "doc_001", "score": 0.95}],
            "latency_ms": 123.45,
        }
        cache.set(emb, complex_response)
        result, _ = cache.get(emb)
        assert result == complex_response

    def test_cosine_similarity_static_method(self):
        """Verify the static cosine similarity implementation."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = SemanticCache._cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6

        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim_ortho = SemanticCache._cosine_similarity(a, c)
        assert abs(sim_ortho - 0.0) < 1e-6

        d = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim_anti = SemanticCache._cosine_similarity(a, d)
        assert abs(sim_anti - (-1.0)) < 1e-6
