"""
Pydantic schemas for the Hybrid RAG Gateway API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to return after reranking")


class RetrievedDoc(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Document text content")
    source: str = Field(..., description="Document source path or URL")
    score: float = Field(..., description="Final fused RRF score")
    dense_score: float = Field(..., description="Score from dense vector retrieval")
    sparse_score: float = Field(..., description="Score from BM25 sparse retrieval")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder reranking score")


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: List[RetrievedDoc]
    cache_hit: bool
    latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    stream_latency_ms: float


class CacheStats(BaseModel):
    hit_rate: float = Field(..., description="Cache hit rate as a fraction between 0 and 1")
    total_requests: int = Field(..., description="Total number of cache lookups")
    cache_size: int = Field(..., description="Number of entries currently in the cache")
    hits: int = Field(..., description="Number of cache hits")
    misses: int = Field(..., description="Number of cache misses")


class SystemStats(BaseModel):
    total_queries: int = Field(..., description="Total queries processed since startup")
    cache: CacheStats
    avg_latency_ms: float = Field(..., description="Average end-to-end latency in milliseconds")
    avg_retrieval_latency_ms: float
    avg_rerank_latency_ms: float
    uptime_seconds: float
