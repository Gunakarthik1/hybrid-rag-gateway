"""
FastAPI application for the Enterprise Hybrid-RAG & Semantic Cache Gateway.

Endpoints
---------
POST /api/query   — hybrid retrieval + reranking, streamed via SSE
GET  /api/stats   — system statistics (cache hit rate, avg latency, total queries)
GET  /api/health  — health check
DELETE /api/cache — flush the semantic cache
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pathlib

from backend.cache import SemanticCache
from backend.models import CacheStats, QueryRequest, SystemStats
from backend.reranker import CrossEncoderReranker
from backend.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hybrid RAG Gateway",
    description="Enterprise hybrid search + semantic cache gateway for RAG systems.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = pathlib.Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

@app.get("/", include_in_schema=False)
async def index():
    index_file = _FRONTEND / "index.html"
    if not index_file.exists():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(f"Frontend not found at {index_file}", status_code=404)
    return FileResponse(str(index_file))

# ---------------------------------------------------------------------------
# Singletons — initialized at startup
# ---------------------------------------------------------------------------
retriever: HybridRetriever = None  # type: ignore
reranker: CrossEncoderReranker = None  # type: ignore
cache: SemanticCache = None  # type: ignore

# In-memory stats tracking
_startup_time: float = time.time()
_total_queries: int = 0
_latency_history: list[float] = []
_retrieval_latency_history: list[float] = []
_rerank_latency_history: list[float] = []


@app.on_event("startup")
async def startup_event() -> None:
    global retriever, reranker, cache
    logger.info("Initializing HybridRetriever…")
    retriever = HybridRetriever()
    logger.info("Initializing CrossEncoderReranker…")
    reranker = CrossEncoderReranker()
    logger.info("Initializing SemanticCache (threshold=0.96, TTL=3600s)…")
    cache = SemanticCache(threshold=0.96, ttl_seconds=3600, max_size=1000)
    logger.info("All components initialized. Gateway ready.")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE message frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_query(query: str, top_k: int) -> AsyncGenerator[str, None]:
    """
    Core SSE generator for the /api/query endpoint.

    Emits events in order:
      cache_hit | retrieval_start → retrieval_done → rerank_done → token* → done
    """
    global _total_queries, _latency_history, _retrieval_latency_history, _rerank_latency_history

    wall_start = time.perf_counter()
    _total_queries += 1

    # ------------------------------------------------------------------
    # 1. Semantic cache lookup
    # ------------------------------------------------------------------
    q_embedding = retriever.embed_query(query)
    cached_response, similarity = cache.get(q_embedding)

    if cached_response is not None:
        cache_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        yield _sse_event("cache_hit", {
            "similarity": round(similarity, 4),
            "latency_ms": cache_ms,
        })
        # Stream cached tokens
        answer: str = cached_response["answer"]
        docs = cached_response["docs"]
        for chunk in _chunk_text(answer, chunk_size=4):
            yield _sse_event("token", {"text": chunk})
            await asyncio.sleep(0.005)  # simulate stream pacing

        total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        _latency_history.append(total_ms)
        yield _sse_event("done", {
            "cache_hit": True,
            "answer": answer,
            "docs": docs,
            "latency_ms": total_ms,
            "retrieval_latency_ms": 0,
            "rerank_latency_ms": 0,
            "stream_latency_ms": cache_ms,
        })
        return

    # ------------------------------------------------------------------
    # 2. Retrieval
    # ------------------------------------------------------------------
    yield _sse_event("retrieval_start", {"query": query})
    retrieval_start = time.perf_counter()

    # Run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    raw_docs = await loop.run_in_executor(
        None, lambda: retriever.hybrid_search(query, top_k=top_k * 3)
    )

    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
    _retrieval_latency_history.append(retrieval_ms)

    yield _sse_event("retrieval_done", {
        "n_candidates": len(raw_docs),
        "latency_ms": retrieval_ms,
        "top_scores": [round(d.score, 4) for d in raw_docs[:5]],
    })

    # ------------------------------------------------------------------
    # 3. Reranking
    # ------------------------------------------------------------------
    rerank_start = time.perf_counter()
    reranked_docs = await loop.run_in_executor(
        None, lambda: reranker.rerank(query, raw_docs, top_n=top_k)
    )
    rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 2)
    _rerank_latency_history.append(rerank_ms)

    yield _sse_event("rerank_done", {
        "n_results": len(reranked_docs),
        "latency_ms": rerank_ms,
        "top_titles": [d.title for d in reranked_docs[:3]],
    })

    # ------------------------------------------------------------------
    # 4. Answer synthesis (deterministic, no LLM dependency)
    # ------------------------------------------------------------------
    answer = _synthesize_answer(query, reranked_docs)

    stream_start = time.perf_counter()
    for chunk in _chunk_text(answer, chunk_size=4):
        yield _sse_event("token", {"text": chunk})
        await asyncio.sleep(0.008)

    stream_ms = round((time.perf_counter() - stream_start) * 1000, 2)
    total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
    _latency_history.append(total_ms)

    # Serialize docs for caching and response
    docs_payload = [d.model_dump() for d in reranked_docs]

    # Store in cache
    cache.set(q_embedding, {"answer": answer, "docs": docs_payload})

    yield _sse_event("done", {
        "cache_hit": False,
        "answer": answer,
        "docs": docs_payload,
        "latency_ms": total_ms,
        "retrieval_latency_ms": retrieval_ms,
        "rerank_latency_ms": rerank_ms,
        "stream_latency_ms": stream_ms,
    })


def _chunk_text(text: str, chunk_size: int = 4) -> list[str]:
    """Split text into word-level chunks of size *chunk_size* for streaming."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        chunks.append(chunk)
    return chunks


def _synthesize_answer(query: str, docs: list) -> str:
    """
    Synthesize a concise answer from the retrieved and reranked documents.

    Constructs a structured natural-language response by extracting the most
    relevant sentences from the top documents. This approach avoids an LLM
    dependency while producing coherent, factual responses grounded in the corpus.
    """
    if not docs:
        return f"No relevant documents found for query: '{query}'"

    top_doc = docs[0]
    second_doc = docs[1] if len(docs) > 1 else None

    # Extract first two sentences from top document
    sentences = [s.strip() for s in top_doc.text.replace("—", " — ").split(". ") if len(s.strip()) > 20]
    primary_excerpt = ". ".join(sentences[:2]) + "." if sentences else top_doc.text[:300]

    answer_parts = [
        f"Based on the retrieved documentation, here is what I found regarding your query about **{_extract_topic(query)}**:\n\n",
        f"{primary_excerpt}\n\n",
    ]

    if second_doc:
        sec_sentences = [s.strip() for s in second_doc.text.replace("—", " — ").split(". ") if len(s.strip()) > 20]
        secondary_excerpt = sec_sentences[0] + "." if sec_sentences else ""
        if secondary_excerpt:
            answer_parts.append(
                f"Additionally, from *{second_doc.title}*: {secondary_excerpt}\n\n"
            )

    answer_parts.append(
        f"*{len(docs)} source document(s) retrieved — see citations below for full details.*"
    )

    return "".join(answer_parts)


def _extract_topic(query: str) -> str:
    """Extract a short topic phrase from the query for the answer header."""
    q = query.strip().rstrip("?").rstrip(".")
    # Capitalize first letter
    return q[0].upper() + q[1:] if q else "this topic"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/query")
async def query_endpoint(request: QueryRequest) -> StreamingResponse:
    """
    Stream a hybrid RAG response via Server-Sent Events.

    Events emitted (in order):
    - cache_hit (if applicable): {similarity, latency_ms}
    - retrieval_start: {query}
    - retrieval_done: {n_candidates, latency_ms, top_scores}
    - rerank_done: {n_results, latency_ms, top_titles}
    - token: {text}  (many)
    - done: {cache_hit, answer, docs, latency_ms, retrieval_latency_ms, rerank_latency_ms, stream_latency_ms}
    - error: {message}  (only on exception)
    """
    async def safe_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _stream_query(request.query, request.top_k):
                yield chunk
        except Exception as exc:
            logger.exception("Error processing query: %s", exc)
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        safe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.get("/api/stats")
async def stats_endpoint() -> SystemStats:
    """Return system-level statistics including cache performance and latency metrics."""
    cache_stats_dict = cache.stats()
    uptime = time.time() - _startup_time

    avg_latency = (
        sum(_latency_history) / len(_latency_history) if _latency_history else 0.0
    )
    avg_retrieval = (
        sum(_retrieval_latency_history) / len(_retrieval_latency_history)
        if _retrieval_latency_history else 0.0
    )
    avg_rerank = (
        sum(_rerank_latency_history) / len(_rerank_latency_history)
        if _rerank_latency_history else 0.0
    )

    return SystemStats(
        total_queries=_total_queries,
        cache=CacheStats(**cache_stats_dict),
        avg_latency_ms=round(avg_latency, 2),
        avg_retrieval_latency_ms=round(avg_retrieval, 2),
        avg_rerank_latency_ms=round(avg_rerank, 2),
        uptime_seconds=round(uptime, 1),
    )


@app.get("/api/health")
async def health_endpoint() -> dict:
    """Health check endpoint. Returns 200 when the service is ready."""
    return {
        "status": "healthy",
        "components": {
            "retriever": retriever is not None,
            "reranker": reranker is not None,
            "cache": cache is not None,
        },
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@app.delete("/api/cache")
async def clear_cache_endpoint() -> dict:
    """Flush the semantic cache. Returns number of entries removed."""
    removed = cache.clear()
    logger.info("Cache cleared: %d entries removed.", removed)
    return {"status": "cleared", "entries_removed": removed}
