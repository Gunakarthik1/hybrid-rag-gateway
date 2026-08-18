"""
FastAPI application for the Hybrid RAG Gateway.

Endpoints
---------
POST /api/query          — hybrid retrieval + reranking, streamed via SSE
POST /api/search         — retrieve from BM25, FAISS (TF-IDF), and hybrid (RRF) separately
POST /api/upload         — upload a document (txt/pdf) and add to corpus
GET  /api/cache/stats    — semantic cache statistics
GET  /api/corpus/stats   — corpus size and index statistics
GET  /api/stats          — full system statistics (legacy, kept for frontend compat)
GET  /api/health         — health check
DELETE /api/cache        — flush the semantic cache
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pathlib

from backend.cache import SemanticCache
from backend.models import CacheStats, QueryRequest, RetrievedDoc, SystemStats
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
    description="Hybrid search + semantic cache gateway for RAG systems.",
    version="2.0.0",
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

_startup_time: float = time.time()
_total_queries: int = 0
_total_search_queries: int = 0
_latency_history: list[float] = []
_retrieval_latency_history: list[float] = []
_rerank_latency_history: list[float] = []


@app.on_event("startup")
async def startup_event() -> None:
    global retriever, reranker, cache
    logger.info("Initializing HybridRetriever (TF-IDF + BM25 + RRF)…")
    retriever = HybridRetriever()
    logger.info("Initializing CrossEncoderReranker…")
    reranker = CrossEncoderReranker()
    logger.info("Initializing SemanticCache (threshold=0.90, TTL=3600s)…")
    cache = SemanticCache(threshold=0.90, ttl_seconds=3600, max_size=1000)
    logger.info("All components initialized. Gateway ready.")


# ---------------------------------------------------------------------------
# SSE helpers  (legacy /api/query streaming endpoint)
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_query(query: str, top_k: int) -> AsyncGenerator[str, None]:
    global _total_queries, _latency_history, _retrieval_latency_history, _rerank_latency_history

    wall_start = time.perf_counter()
    _total_queries += 1

    # 1. Semantic cache lookup
    q_embedding = retriever.embed_query(query)
    cached_response, similarity = cache.get(q_embedding)

    if cached_response is not None:
        cache_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        yield _sse_event("cache_hit", {"similarity": round(similarity, 4), "latency_ms": cache_ms})
        answer: str = cached_response["answer"]
        docs = cached_response["docs"]
        for chunk in _chunk_text(answer, chunk_size=4):
            yield _sse_event("token", {"text": chunk})
            await asyncio.sleep(0.005)
        total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        _latency_history.append(total_ms)
        yield _sse_event("done", {
            "cache_hit": True, "answer": answer, "docs": docs,
            "latency_ms": total_ms, "retrieval_latency_ms": 0,
            "rerank_latency_ms": 0, "stream_latency_ms": cache_ms,
        })
        return

    # 2. Retrieval
    yield _sse_event("retrieval_start", {"query": query})
    retrieval_start = time.perf_counter()
    loop = asyncio.get_event_loop()
    raw_docs = await loop.run_in_executor(
        None, lambda: retriever.hybrid_search(query, top_k=top_k * 3)
    )
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
    _retrieval_latency_history.append(retrieval_ms)
    yield _sse_event("retrieval_done", {
        "n_candidates": len(raw_docs), "latency_ms": retrieval_ms,
        "top_scores": [round(d.score, 4) for d in raw_docs[:5]],
    })

    # 3. Reranking
    rerank_start = time.perf_counter()
    reranked_docs = await loop.run_in_executor(
        None, lambda: reranker.rerank(query, raw_docs, top_n=top_k)
    )
    rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 2)
    _rerank_latency_history.append(rerank_ms)
    yield _sse_event("rerank_done", {
        "n_results": len(reranked_docs), "latency_ms": rerank_ms,
        "top_titles": [d.title for d in reranked_docs[:3]],
    })

    # 4. Answer synthesis
    answer = _synthesize_answer(query, reranked_docs)
    stream_start = time.perf_counter()
    for chunk in _chunk_text(answer, chunk_size=4):
        yield _sse_event("token", {"text": chunk})
        await asyncio.sleep(0.008)
    stream_ms = round((time.perf_counter() - stream_start) * 1000, 2)
    total_ms = round((time.perf_counter() - wall_start) * 1000, 2)
    _latency_history.append(total_ms)

    docs_payload = [d.model_dump() for d in reranked_docs]
    cache.set(q_embedding, {"answer": answer, "docs": docs_payload})

    yield _sse_event("done", {
        "cache_hit": False, "answer": answer, "docs": docs_payload,
        "latency_ms": total_ms, "retrieval_latency_ms": retrieval_ms,
        "rerank_latency_ms": rerank_ms, "stream_latency_ms": stream_ms,
    })


def _chunk_text(text: str, chunk_size: int = 4) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        chunks.append(chunk)
    return chunks


def _synthesize_answer(query: str, docs: list) -> str:
    if not docs:
        return f"No relevant documents found for query: '{query}'"
    top_doc = docs[0]
    second_doc = docs[1] if len(docs) > 1 else None
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
            answer_parts.append(f"Additionally, from *{second_doc.title}*: {secondary_excerpt}\n\n")
    answer_parts.append(f"*{len(docs)} source document(s) retrieved — see citations below for full details.*")
    return "".join(answer_parts)


def _extract_topic(query: str) -> str:
    q = query.strip().rstrip("?").rstrip(".")
    return q[0].upper() + q[1:] if q else "this topic"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/query")
async def query_endpoint(request: QueryRequest) -> StreamingResponse:
    """Stream a hybrid RAG response via Server-Sent Events."""
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/search")
async def search_endpoint(request: QueryRequest):
    """
    Run BM25, TF-IDF (FAISS-equivalent), and hybrid RRF retrieval separately.

    Returns results from all three retrievers plus cache metadata.

    Input:  { "query": "...", "top_k": 5 }
    Output: {
        "bm25_results":    [{doc_id, title, text, score, rank}, ...],
        "faiss_results":   [{doc_id, title, text, score, rank}, ...],
        "hybrid_results":  [{doc_id, title, text, score, rank}, ...],
        "query_time_ms":   12.3,
        "cache_hit":       false,
        "cache_size":      4,
    }
    """
    global _total_search_queries

    wall_start = time.perf_counter()
    _total_search_queries += 1

    # Semantic cache lookup
    q_embedding = retriever.embed_query(request.query)
    cached_response, _similarity = cache.get(q_embedding)

    if cached_response is not None and "bm25_results" in cached_response:
        query_time_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        return {
            **cached_response,
            "query_time_ms": query_time_ms,
            "cache_hit": True,
            "cache_size": cache.stats()["cache_size"],
        }

    loop = asyncio.get_event_loop()
    bm25_results, faiss_results, hybrid_results = await loop.run_in_executor(
        None,
        lambda: retriever.search_separate(request.query, top_k=request.top_k),
    )

    query_time_ms = round((time.perf_counter() - wall_start) * 1000, 2)

    payload = {
        "bm25_results": bm25_results,
        "faiss_results": faiss_results,
        "hybrid_results": hybrid_results,
    }
    cache.set(q_embedding, payload)

    return {
        **payload,
        "query_time_ms": query_time_ms,
        "cache_hit": False,
        "cache_size": cache.stats()["cache_size"],
    }


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf document and add it to the live corpus.

    Re-indexes BM25 and TF-IDF after ingestion.

    Returns: { doc_id, title, num_chunks }
    """
    filename = file.filename or "uploaded_document"
    ext = pathlib.Path(filename).suffix.lower()

    raw_bytes = await file.read()

    if ext == ".pdf":
        # Try to extract text from PDF using pdfminer if available, else treat as text
        try:
            from pdfminer.high_level import extract_text_to_fp  # type: ignore
            from pdfminer.layout import LAParams
            out = io.StringIO()
            extract_text_to_fp(io.BytesIO(raw_bytes), out, laparams=LAParams())
            text = out.getvalue().strip()
        except ImportError:
            # Fall back: decode as utf-8 ignoring errors
            text = raw_bytes.decode("utf-8", errors="ignore").strip()
    else:
        text = raw_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from the uploaded file.")

    # Derive title from filename (strip extension, replace dashes/underscores)
    title = pathlib.Path(filename).stem.replace("-", " ").replace("_", " ").title()
    doc_id = f"upload_{uuid.uuid4().hex[:8]}"

    doc = {
        "id": doc_id,
        "title": title,
        "source": f"uploads/{filename}",
        "text": text,
    }

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: retriever.add_document(doc))

    # Simple chunking count estimate (every ~300 words = 1 chunk)
    word_count = len(text.split())
    num_chunks = max(1, word_count // 300)

    logger.info("Uploaded doc '%s' (%d words, ~%d chunks)", title, word_count, num_chunks)

    return {"doc_id": doc_id, "title": title, "num_chunks": num_chunks}


@app.get("/api/cache/stats")
async def cache_stats_endpoint():
    """
    Return semantic cache statistics.

    Response: { cache_size, hit_rate, total_queries }
    """
    stats = cache.stats()
    return {
        "cache_size": stats["cache_size"],
        "hit_rate": stats["hit_rate"],
        "total_queries": stats["total_requests"],
        "hits": stats["hits"],
        "misses": stats["misses"],
    }


@app.get("/api/corpus/stats")
async def corpus_stats_endpoint():
    """
    Return corpus and index statistics.

    Response: { total_docs, total_chunks, index_size_mb }
    """
    total_docs = len(retriever.corpus)
    # Estimate chunks: each doc is ~300 words => ~1 chunk per 300 words
    total_words = sum(len(d["text"].split()) for d in retriever.corpus)
    total_chunks = max(total_docs, total_words // 300)
    index_size_mb = retriever.index_size_mb()

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "index_size_mb": index_size_mb,
    }


@app.get("/api/stats")
async def stats_endpoint() -> SystemStats:
    """Return full system statistics (legacy endpoint, kept for frontend compat)."""
    cache_stats_dict = cache.stats()
    uptime = time.time() - _startup_time
    avg_latency = sum(_latency_history) / len(_latency_history) if _latency_history else 0.0
    avg_retrieval = (
        sum(_retrieval_latency_history) / len(_retrieval_latency_history)
        if _retrieval_latency_history else 0.0
    )
    avg_rerank = (
        sum(_rerank_latency_history) / len(_rerank_latency_history)
        if _rerank_latency_history else 0.0
    )
    return SystemStats(
        total_queries=_total_queries + _total_search_queries,
        cache=CacheStats(**cache_stats_dict),
        avg_latency_ms=round(avg_latency, 2),
        avg_retrieval_latency_ms=round(avg_retrieval, 2),
        avg_rerank_latency_ms=round(avg_rerank, 2),
        uptime_seconds=round(uptime, 1),
    )


@app.get("/api/health")
async def health_endpoint() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "retriever": retriever is not None,
            "reranker": reranker is not None,
            "cache": cache is not None,
        },
        "corpus_size": len(retriever.corpus) if retriever else 0,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@app.delete("/api/cache")
async def clear_cache_endpoint() -> dict:
    """Flush the semantic cache."""
    removed = cache.clear()
    logger.info("Cache cleared: %d entries removed.", removed)
    return {"status": "cleared", "entries_removed": removed}
